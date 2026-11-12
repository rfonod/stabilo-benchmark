#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
gcp.py - Ground control points, for auditing the reference estimator.

Ground-plane features are annotated in the reference frame and in sampled frames. A homography
maps each current-frame point into reference coordinates, and its distance to the annotated
reference point is a residual against an independent measurement. Points must lie on the ground
plane, and correspond by label, never by order. `propose_points` proposes a reference point's
position in a current frame for the annotator to accept or correct, independently of any feature
matcher.

Files are LabelMe JSON with `point` shapes, or text with `label x y` per line:

    gcp/
      000000.json     reference frame
      001500.json     an audited current frame
      ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from stabilo_benchmark.regions import rasterize_quads, warp_points
from stabilo_benchmark.scene import platform_motion

GCP_DIR = 'gcp'
# Second, independent annotation session of the same frames (`annotate-gcp --repeat`).
GCP_REPEAT_DIR = 'gcp_repeat'
# The first session's frames placed again without proposals, on its reference points (`annotate-gcp --manual`).
GCP_MANUAL_DIR = 'gcp_manual'
# Fewest points for a fitted homography to count as well determined.
MIN_WELL_DETERMINED_POINTS = 8


class GcpError(ValueError):
    """Raised for an unreadable or inconsistent ground-control-point file."""


@dataclass(frozen=True)
class GroundControlPoints:
    """Labeled image points for one frame, in absolute pixel coordinates."""

    frame_index: int
    labels: Tuple[str, ...]
    points: np.ndarray  # (N, 2)
    source: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.labels)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {label: point for label, point in zip(self.labels, self.points, strict=True)}


def load_gcp_file(path: Path) -> GroundControlPoints:
    """
    Read one frame's ground control points from LabelMe JSON or `label x y` text.

    The frame index comes from the file stem (`000000.json` for frame 0).
    """
    path = Path(path)
    if not path.exists():
        raise GcpError(f"Ground control point file not found: {path}")

    try:
        frame_index = int(path.stem)
    except ValueError as e:
        raise GcpError(
            f"Cannot read a frame index from '{path.name}'. Name GCP files after the frame "
            f"they annotate, zero-padded to six digits (e.g. 001500.json)."
        ) from e

    labels: List[str] = []
    coordinates: List[Sequence[float]] = []

    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        for shape in payload.get('shapes', []):
            if shape.get('shape_type') != 'point':
                continue
            points = shape.get('points') or []
            if not points:
                continue
            labels.append(str(shape.get('label', '')).strip())
            coordinates.append(points[0][:2])
    else:
        for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise GcpError(f"{path}:{line_number}: expected 'label x y', got {line!r}")
            labels.append(fields[0])
            coordinates.append((float(fields[1]), float(fields[2])))

    if not labels:
        raise GcpError(f"No point annotations found in {path}")
    if '' in labels:
        raise GcpError(
            f"{path} contains an unlabeled point. Correspondence is established by label, "
            f"not by order, so every point needs a name that is reused across frames."
        )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise GcpError(f"{path} reuses label(s) {duplicates} within one frame; labels must be unique per frame")

    return GroundControlPoints(
        frame_index=frame_index,
        labels=tuple(labels),
        points=np.asarray(coordinates, dtype=np.float64).reshape(-1, 2),
        source=path,
    )


def load_gcp_set(directory: Path) -> Dict[int, GroundControlPoints]:
    """Load every GCP file in a directory, keyed by frame index."""
    directory = Path(directory)
    if not directory.is_dir():
        raise GcpError(f"Ground control point directory not found: {directory}")

    found: Dict[int, GroundControlPoints] = {}
    for path in sorted([*directory.glob('*.json'), *directory.glob('*.txt')]):
        entry = load_gcp_file(path)
        if entry.frame_index in found:
            raise GcpError(
                f"Two GCP files annotate frame {entry.frame_index}: {found[entry.frame_index].source} and {path}"
            )
        found[entry.frame_index] = entry
    if not found:
        raise GcpError(
            f"No ground control point files in {directory}. Annotate ground-plane features in the "
            f"reference frame and at least one sampled frame (see `stabilo-benchmark annotate-gcp`)."
        )
    return found


def residuals(
    reference: GroundControlPoints,
    current: GroundControlPoints,
    homography: np.ndarray,
) -> Dict[str, Any]:
    """
    Residuals of one frame's ground control points under a candidate homography.

    Points are matched by label; only labels present in both frames contribute. Returns per-point
    distances in reference-frame pixels, their summary statistics and the number matched.
    """
    shared = [label for label in reference.labels if label in set(current.labels)]
    if not shared:
        return {
            'n_points': 0,
            'residual_mean_px': float('nan'),
            'residual_median_px': float('nan'),
            'residual_max_px': float('nan'),
            'residual_rms_px': float('nan'),
            'labels': [],
            'residuals_px': [],
        }

    reference_points = reference.as_dict()
    current_points = current.as_dict()
    source = np.array([current_points[label] for label in shared], dtype=np.float64)
    target = np.array([reference_points[label] for label in shared], dtype=np.float64)

    projected = warp_points(source, np.asarray(homography, dtype=np.float64))
    distances = np.linalg.norm(projected - target, axis=1)

    return {
        'n_points': len(shared),
        'residual_mean_px': float(np.mean(distances)),
        'residual_median_px': float(np.median(distances)),
        'residual_max_px': float(np.max(distances)),
        'residual_rms_px': float(np.sqrt(np.mean(distances**2))),
        'labels': shared,
        'residuals_px': [float(d) for d in distances],
    }


def fit_homography(reference: GroundControlPoints, current: GroundControlPoints) -> Dict[str, Any]:
    """
    The homography current -> reference fitted to the points the two frames share.

    Least squares over all shared points (OpenCV method 0). Returns the matrix (None with fewer than
    four shared points), the point count, the fit's RMS residual, and whether the count reaches
    `MIN_WELL_DETERMINED_POINTS`.
    """
    shared = [label for label in current.labels if label in reference.as_dict()]
    if len(shared) < 4:
        return {'matrix': None, 'n_points': len(shared), 'fit_rms_px': float('nan'), 'well_determined': False}
    reference_points = reference.as_dict()
    current_points = current.as_dict()
    source = np.array([current_points[label] for label in shared], dtype=np.float64)
    target = np.array([reference_points[label] for label in shared], dtype=np.float64)
    matrix, _ = cv2.findHomography(source.reshape(-1, 1, 2), target.reshape(-1, 1, 2), 0)
    if matrix is None:
        return {'matrix': None, 'n_points': len(shared), 'fit_rms_px': float('nan'), 'well_determined': False}
    matrix = np.asarray(matrix, dtype=np.float64)
    fitted = warp_points(source, matrix)
    rms = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return {
        'matrix': matrix,
        'n_points': len(shared),
        'fit_rms_px': rms,
        'well_determined': len(shared) >= MIN_WELL_DETERMINED_POINTS,
    }


def annotation_repeatability(
    first: Mapping[int, GroundControlPoints], second: Mapping[int, GroundControlPoints]
) -> Dict[str, Any]:
    """
    Per-point distance between two annotation sessions of the same frames, pooled.

    Only frames present in both sessions and labels shared within a frame contribute. The summary is
    empty when the sessions share nothing.
    """
    distances: List[float] = []
    frames: List[int] = []
    for index, points in sorted(first.items()):
        repeat = second.get(index)
        if repeat is None:
            continue
        a, b = points.as_dict(), repeat.as_dict()
        shared = [label for label in points.labels if label in b]
        if not shared:
            continue
        frames.append(index)
        distances.extend(float(np.linalg.norm(a[label] - b[label])) for label in shared)
    if not distances:
        return {'n_frames': 0, 'n_points': 0}
    pooled = np.asarray(distances, dtype=np.float64)
    return {
        'n_frames': len(frames),
        'frames': frames,
        'n_points': int(pooled.size),
        'residual_mean_px': float(np.mean(pooled)),
        'residual_median_px': float(np.median(pooled)),
        'residual_p95_px': float(np.percentile(pooled, 95)),
        'residual_max_px': float(np.max(pooled)),
    }


def summarize_residuals(per_frame: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Pool per-frame residuals over individual points.

    Reports the median and the 95th percentile, and `share_within_half_px`, the fraction of points
    within half a pixel of where the audited homography puts them.
    """
    pooled = np.array(
        [value for frame in per_frame for value in frame.get('residuals_px', [])],
        dtype=np.float64,
    )
    if pooled.size == 0:
        return {'n_frames': len(per_frame), 'n_points': 0}

    return {
        'n_frames': len(per_frame),
        'n_points': int(pooled.size),
        'residual_mean_px': float(np.mean(pooled)),
        'residual_median_px': float(np.median(pooled)),
        'residual_p95_px': float(np.percentile(pooled, 95)),
        'residual_max_px': float(np.max(pooled)),
        'residual_rms_px': float(np.sqrt(np.mean(pooled**2))),
        'share_within_half_px': float(np.mean(pooled < 0.5)),
    }


# Proposals for annotation: the reference point's neighborhood (a square of side 2 * radius + 1) is
# fitted to the current frame by an affine ECC, from the platform motion (`scene.platform_motion`),
# searching up to PROPOSAL_SEARCH_PX around it. A proposal stands when the fit correlates at least
# PROPOSAL_MIN_CORRELATION, the fit back from the proposal returns within PROPOSAL_MAX_ROUND_TRIP_PX of
# the reference point, and objects cover less than PROPOSAL_MAX_OCCUPIED of the neighborhood.
PROPOSAL_RADIUS_PX = 40
PROPOSAL_SEARCH_PX = 24
PROPOSAL_MIN_CORRELATION = 0.8
PROPOSAL_MAX_ROUND_TRIP_PX = 0.5
PROPOSAL_MAX_OCCUPIED = 0.4
# Margin around each object excluded from the fit, in pixels.
PROPOSAL_OBJECT_MARGIN_PX = 8
PROPOSAL_ITERATIONS = 200
PROPOSAL_EPSILON = 1e-5


@dataclass(frozen=True)
class Proposal:
    """Where a reference point is proposed in a current frame; `point` is None when a check failed (`reason`)."""

    point: Optional[np.ndarray]
    correlation: float = float('nan')
    round_trip_px: float = float('nan')
    reason: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'point': None if self.point is None else [float(self.point[0]), float(self.point[1])],
            'correlation': None if not np.isfinite(self.correlation) else float(self.correlation),
            'round_trip_px': None if not np.isfinite(self.round_trip_px) else float(self.round_trip_px),
            'reason': self.reason,
        }


def proposal_settings() -> Dict[str, Any]:
    """The proposer's constants, recorded with every annotated frame that uses it."""
    return {
        'method': 'local affine ECC from the platform motion',
        'radius_px': PROPOSAL_RADIUS_PX,
        'search_px': PROPOSAL_SEARCH_PX,
        'min_correlation': PROPOSAL_MIN_CORRELATION,
        'max_round_trip_px': PROPOSAL_MAX_ROUND_TRIP_PX,
        'max_occupied': PROPOSAL_MAX_OCCUPIED,
        'object_margin_px': PROPOSAL_OBJECT_MARGIN_PX,
    }


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _fit_neighborhood(
    template: np.ndarray,
    template_free: np.ndarray,
    image: np.ndarray,
    image_free: np.ndarray,
    point: np.ndarray,
    initial: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    The affine map from `template` to `image` fitted on `point`'s neighborhood, starting from `initial`.

    Returns the 3x3 map in full-frame coordinates (None on failure), the ECC correlation and the reason
    for a failure. Object pixels (False in the `*_free` masks) are filled with the background mean of
    their window, and the image's are left out of the fit.
    """
    radius, search = PROPOSAL_RADIUS_PX, PROPOSAL_SEARCH_PX
    side = 2 * radius + 1
    x0, y0 = int(round(point[0])) - radius, int(round(point[1])) - radius
    if x0 < 0 or y0 < 0 or x0 + side > template.shape[1] or y0 + side > template.shape[0]:
        return None, float('nan'), 'neighborhood leaves the frame'
    patch = template[y0 : y0 + side, x0 : x0 + side].astype(np.float32)
    free = template_free[y0 : y0 + side, x0 : x0 + side]
    if 1.0 - free.mean() > PROPOSAL_MAX_OCCUPIED:
        return None, float('nan'), 'objects cover the neighborhood'
    patch[~free] = patch[free].mean()

    center = warp_points(np.asarray(point, dtype=np.float64).reshape(1, 2), initial)[0]
    if not np.all(np.isfinite(center)):
        return None, float('nan'), 'no platform motion'
    window = side + 2 * search
    cx0, cy0 = int(round(center[0])) - radius - search, int(round(center[1])) - radius - search
    if cx0 < 0 or cy0 < 0 or cx0 + window > image.shape[1] or cy0 + window > image.shape[0]:
        return None, float('nan'), 'neighborhood leaves the current frame'
    crop = image[cy0 : cy0 + window, cx0 : cx0 + window].astype(np.float32)
    crop_free = image_free[cy0 : cy0 + window, cx0 : cx0 + window]
    if crop_free.any():
        crop[~crop_free] = crop[crop_free].mean()

    warp = (_translation(-cx0, -cy0) @ initial @ _translation(x0, y0))[:2].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, PROPOSAL_ITERATIONS, PROPOSAL_EPSILON)
    try:
        correlation, warp = cv2.findTransformECC(
            patch, crop, warp, cv2.MOTION_AFFINE, criteria, crop_free.astype(np.uint8) * 255, 1
        )
    except cv2.error:
        return None, float('nan'), 'the fit did not converge'
    local = np.eye(3)
    local[:2] = warp
    return _translation(cx0, cy0) @ local @ _translation(-x0, -y0), float(correlation), ''


def _gray_and_free(image: np.ndarray, labels: Optional[Any]) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    free = np.ones(gray.shape, dtype=bool)
    if labels is not None and len(labels):
        free = ~rasterize_quads(labels.corners(), gray.shape, dilation_px=PROPOSAL_OBJECT_MARGIN_PX).astype(bool)
    return gray, free


def propose_points(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_points: Mapping[str, Sequence[float]],
    reference_labels: Optional[Any] = None,
    current_labels: Optional[Any] = None,
    initial: Optional[np.ndarray] = None,
) -> Dict[str, Proposal]:
    """
    Propose where each reference point lies in the current frame, independently of any feature matcher.

    Each point's neighborhood is fitted to the current frame by an affine ECC (`_fit_neighborhood`)
    starting from `initial`, the reference-to-current platform motion (`scene.platform_motion` when not
    given), and fitted back from the proposal; the checks are in the `PROPOSAL_*` constants. Objects
    (`labels`, any type with `corners()`) are filled and left out of the fit.
    """
    if initial is None:
        initial = platform_motion(reference_image, current_image, reference_labels, current_labels)
    initial = np.asarray(initial, dtype=np.float64)
    reference_gray, reference_free = _gray_and_free(reference_image, reference_labels)
    current_gray, current_free = _gray_and_free(current_image, current_labels)

    proposals: Dict[str, Proposal] = {}
    for label, given in reference_points.items():
        point = np.asarray(given, dtype=np.float64)
        forward, correlation, reason = _fit_neighborhood(
            reference_gray, reference_free, current_gray, current_free, point, initial
        )
        if forward is None:
            proposals[label] = Proposal(None, reason=reason)
            continue
        proposed = warp_points(point.reshape(1, 2), forward)[0]
        backward, _, reason = _fit_neighborhood(
            current_gray, current_free, reference_gray, reference_free, proposed, np.linalg.inv(forward)
        )
        if backward is None:
            proposals[label] = Proposal(None, correlation, reason=f"fit back: {reason}")
            continue
        round_trip = float(np.linalg.norm(warp_points(proposed.reshape(1, 2), backward)[0] - point))
        if correlation < PROPOSAL_MIN_CORRELATION:
            reason = f"correlation {correlation:.2f} below {PROPOSAL_MIN_CORRELATION}"
        elif not round_trip <= PROPOSAL_MAX_ROUND_TRIP_PX:
            reason = f"round trip {round_trip:.2f} px above {PROPOSAL_MAX_ROUND_TRIP_PX} px"
        proposals[label] = Proposal(None if reason else proposed, correlation, round_trip, reason)
    return proposals


def proposal_review(
    directory: Path, reference_index: int, per_frame: Optional[Sequence[Mapping[str, Any]]] = None
) -> Dict[str, Any]:
    """
    How the points of the annotated frames relate to the proposals recorded with them.

    A point placed where it was proposed was accepted; one placed elsewhere was corrected (by the
    distances reported); one without a standing proposal was placed by hand, in the panel that opens
    near the audited homography's prediction under a hidden offset. Given that homography's
    `per_frame` residuals (`residuals`), `n_hand_within_half_px` counts the hand-placed points within
    half a pixel of the prediction. The reference frame is left out. The summary is empty when no
    frame records proposals.
    """
    accepted, manual = 0, 0
    corrections: List[float] = []
    frames: List[int] = []
    by_hand: Dict[int, List[str]] = {}
    for path in sorted(Path(directory).glob('*.json')):
        if not path.stem.isdigit() or int(path.stem) == int(reference_index):
            continue
        recorded = (json.loads(path.read_text(encoding='utf-8')).get('stabilo_benchmark') or {}).get('proposals')
        if recorded is None:
            continue
        frames.append(int(path.stem))
        for label, point in load_gcp_file(path).as_dict().items():
            proposed = (recorded.get(label) or {}).get('point')
            if proposed is None:
                manual += 1
                by_hand.setdefault(int(path.stem), []).append(label)
                continue
            distance = float(np.linalg.norm(np.asarray(point) - np.asarray(proposed, dtype=np.float64)))
            if distance == 0.0:
                accepted += 1
            else:
                corrections.append(distance)
    if not frames:
        return {'n_frames': 0, 'n_points': 0}
    summary: Dict[str, Any] = {
        'n_frames': len(frames),
        'frames': frames,
        'n_points': accepted + len(corrections) + manual,
        'n_accepted': accepted,
        'n_corrected': len(corrections),
        'n_placed_by_hand': manual,
    }
    if corrections:
        summary['corrections_px'] = corrections
        summary['correction_median_px'] = float(np.median(corrections))
        summary['correction_max_px'] = float(np.max(corrections))
    if per_frame is not None and manual:
        hand_residuals = [
            distance
            for frame in per_frame
            for label, distance in zip(frame.get('labels', []), frame.get('residuals_px', []), strict=True)
            if label in by_hand.get(int(frame['frame_index']), ())
        ]
        summary['n_hand_scored'] = len(hand_residuals)
        summary['n_hand_within_half_px'] = int(sum(distance < 0.5 for distance in hand_residuals))
    return summary


def save_gcp_file(
    path: Path,
    points: Mapping[str, Sequence[float]],
    image_path: Optional[Path] = None,
    image_size: Optional[Tuple[int, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Write labeled points as LabelMe-compatible JSON.

    `image_path` is recorded relative to the file, as LabelMe resolves it. `metadata` is written under
    a `stabilo_benchmark` key, which LabelMe preserves.
    """
    path = Path(path)
    height, width = image_size if image_size else (None, None)
    payload: Dict[str, Any] = {
        'version': '5.2.1',
        'flags': {},
        'shapes': [
            {
                'label': str(label),
                'points': [[float(point[0]), float(point[1])]],
                'group_id': None,
                'shape_type': 'point',
                'flags': {},
            }
            for label, point in points.items()
        ],
        'imagePath': Path(os.path.relpath(Path(image_path).absolute(), path.absolute().parent)).as_posix()
        if image_path
        else None,
        'imageData': None,
        'imageHeight': height,
        'imageWidth': width,
    }
    if metadata:
        payload['stabilo_benchmark'] = dict(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return path
