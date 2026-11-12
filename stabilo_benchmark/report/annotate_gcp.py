#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
annotate_gcp.py - Click-to-annotate helper for ground control points.

Writes LabelMe-compatible JSON, one file per frame, for `audit`.

- Every point is placed in a zoomed panel (`PANEL_ZOOM`) with an exact pixel mapping (`Panel`);
  arrow keys nudge the point by one source pixel.
- In the first session, every frame after the reference opens each point's panel on a proposal
  (`gcp.propose_points`: a local ECC fit of the point's neighborhood, independent of the reference
  estimator) with the point already placed there: Enter accepts it, a click or the arrow keys
  correct it. The proposals are recorded in each frame's file, so `audit` can report how many were
  accepted, corrected, or placed by hand.
- A point without a standing proposal, and every point of the repeat session or of `--no-propose`,
  opens near the reference estimator's prediction (`oracle/ORACLE.json`), displaced by a hidden,
  deterministic jitter of up to `JITTER_PX` per axis that differs between the first and the repeat
  session, with nothing placed. The prediction is not shown.
- A reference panel with a crosshair on the point is shown beside the working panel.
- Points correspond by label: the reference frame is annotated first, and every other frame is
  asked for one named point at a time. A point may be skipped.
- The repeat session re-clicks the reference frame too, guided by a coarse view (`HINT_ZOOM`) of
  the first session's point.
- The manual session (`--manual`, written to `gcp_manual/`) places the first session's frames again
  on the first session's reference points, every point without a proposal: unaided clicks to set
  against the proposals.
- `--labels` with `--frames` re-places only the named points in annotated frames and keeps the rest;
  in the reference frame a new feature may be chosen, shown beside every other point.

Points must lie on the ground plane and should be spread across the frame; the reference frame's
points are checked against its objects (OBB grown by `CLEARANCE_PX`) and the exclusion polygons.

Requires an OpenCV build with GUI support (`cv2.imshow`). On a headless machine, annotate in
LabelMe and put the JSON files into the case's `gcp/` directory.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from stabilo_benchmark.gcp import (
    GCP_DIR,
    GCP_MANUAL_DIR,
    GCP_REPEAT_DIR,
    Proposal,
    load_gcp_file,
    proposal_settings,
    propose_points,
    save_gcp_file,
)
from stabilo_benchmark.masking import Labels
from stabilo_benchmark.oracle import OracleError, frame_path, load_frame_labels, load_oracle, oracle_matrices, pair_key
from stabilo_benchmark.regions import EXCLUSION_FILE, load_exclusion, rasterize_quads, warp_points
from stabilo_benchmark.runners.prepare import load_case
from stabilo_benchmark.sampling import FramePair, plan_bins

# The zoomed panel: PANEL_SIZE display pixels showing PANEL_SIZE / zoom source pixels.
PANEL_SIZE = 360
PANEL_ZOOMS = (2.0, 4.0, 8.0)
DEFAULT_ZOOM = 4.0
# The hint panel of the repeat session: coarse enough to identify a feature, not a pixel.
HINT_ZOOM = 1.0
# Maximum displacement of the panel center from the prediction, per axis, in source pixels.
JITTER_PX = 6.0
NUDGE_PX = 1.0
# Clearance of the reference frame's points from its objects' OBBs: the evaluation region's object
# dilation in the shipped dynamic configs (`metrics.region.mask_dilation_px`).
CLEARANCE_PX = 10
BANNER_HEIGHT = 34
CAPTION_HEIGHT = 22
PANEL_GAP = 16
POINT_COLOR = (60, 200, 60)
TARGET_COLOR = (40, 120, 240)
PLACED_COLOR = (0, 215, 255)
OLD_COLOR = (40, 40, 230)

# Arrow keys as `cv2.waitKeyEx` reports them on macOS, GTK and Qt/Windows builds, plus ijkl.
NUDGE_KEYS = {
    'left': {63234, 65361, 2424832, ord('j')},
    'right': {63235, 65363, 2555904, ord('l')},
    'up': {63232, 65362, 2490368, ord('i')},
    'down': {63233, 65364, 2621440, ord('k')},
}


class AnnotateError(RuntimeError):
    """Raised when annotation cannot proceed: no GUI, missing frame, no reference points."""


@dataclass(frozen=True)
class Placement:
    """What one placement round ended with: a point, a skip, a step back, or the end of the frame."""

    action: str  # 'placed' | 'skip' | 'back' | 'finish'
    point: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Panel:
    """
    A zoomed window into a source image with an exact pixel mapping in both directions.

    Display pixel (u, v) shows the source point (x0 + u / zoom, y0 + v / zoom), and a source
    point (x, y) is drawn at ((x - x0) * zoom, (y - y0) * zoom). The render uses an affine warp
    with exactly that mapping, so what is clicked is what is measured.
    """

    origin: Tuple[float, float]
    zoom: float
    size: int = PANEL_SIZE

    @classmethod
    def centered(cls, center: Sequence[float], zoom: float, size: int = PANEL_SIZE) -> 'Panel':
        half = size / (2.0 * zoom)
        return cls(origin=(float(center[0]) - half, float(center[1]) - half), zoom=float(zoom), size=int(size))

    @property
    def center(self) -> np.ndarray:
        half = self.size / (2.0 * self.zoom)
        return np.array([self.origin[0] + half, self.origin[1] + half], dtype=np.float64)

    def to_source(self, u: float, v: float) -> np.ndarray:
        return np.array([self.origin[0] + u / self.zoom, self.origin[1] + v / self.zoom], dtype=np.float64)

    def to_display(self, point: Sequence[float]) -> Tuple[float, float]:
        return (float(point[0]) - self.origin[0]) * self.zoom, (float(point[1]) - self.origin[1]) * self.zoom

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        u, v = self.to_display(point)
        return margin <= u < self.size - margin and margin <= v < self.size - margin

    def render(
        self,
        image: np.ndarray,
        marker: Optional[Sequence[float]] = None,
        color: Tuple[int, int, int] = PLACED_COLOR,
        crosshair: bool = False,
    ) -> np.ndarray:
        matrix = np.array(
            [[self.zoom, 0.0, -self.zoom * self.origin[0]], [0.0, self.zoom, -self.zoom * self.origin[1]]],
            dtype=np.float64,
        )
        interpolation = cv2.INTER_NEAREST if self.zoom >= 4.0 else cv2.INTER_LINEAR
        view = cv2.warpAffine(
            image, matrix, (self.size, self.size), flags=interpolation, borderMode=cv2.BORDER_CONSTANT
        )
        if view.ndim == 2:
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        if crosshair:
            mid = self.size // 2
            cv2.drawMarker(view, (mid, mid), TARGET_COLOR, cv2.MARKER_CROSS, 40, 1, cv2.LINE_AA)
        if marker is not None:
            u, v = self.to_display(marker)
            if 0 <= u < self.size and 0 <= v < self.size:
                cv2.drawMarker(view, (int(round(u)), int(round(v))), color, cv2.MARKER_CROSS, 28, 1, cv2.LINE_AA)
                cv2.circle(view, (int(round(u)), int(round(v))), 9, color, 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (self.size - 1, self.size - 1), (200, 200, 200), 1)
        return view


def predict_positions(
    reference_points: Mapping[str, Sequence[float]], matrix: Optional[np.ndarray]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Where the reference estimator puts each reference point in the current frame.

    The cached matrix maps the current frame into the reference frame, so the prediction is its
    inverse applied to the reference point. None when there is no usable matrix; a prediction
    that is not finite is dropped.
    """
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        return None
    labels = list(reference_points)
    source = np.array([reference_points[label] for label in labels], dtype=np.float64).reshape(-1, 2)
    projected = warp_points(source, np.linalg.inv(matrix))
    return {label: point for label, point in zip(labels, projected, strict=True) if np.all(np.isfinite(point))}


def jitter(frame_index: int, label: str, session: str, amplitude: float = JITTER_PX) -> np.ndarray:
    """
    Hidden displacement of the panel center from the prediction, per axis in [-amplitude, amplitude].

    Deterministic in (session, frame, label), and different between the first and the repeat session.
    """
    seed = zlib.crc32(f"{session}:{int(frame_index)}:{label}".encode('utf-8'))
    rng = np.random.default_rng(seed)
    return rng.uniform(-float(amplitude), float(amplitude), size=2)


def _check_gui() -> None:
    """Fail early and clearly when the OpenCV build has no window support."""
    try:
        cv2.namedWindow('__gcp_probe__', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('__gcp_probe__')
    except cv2.error as e:
        raise AnnotateError(
            "This OpenCV build has no GUI support, so click annotation is unavailable. "
            "Annotate the frames in LabelMe instead (point shapes, one label per feature) and "
            f"save the JSON into the case's {GCP_DIR}/ directory named after the frame index."
        ) from e


def _display_scale(shape: Tuple[int, int], maximum: int = 1400) -> float:
    """Scale factor that fits a frame on screen; clicks are divided back out by it."""
    height, width = shape[:2]
    return min(1.0, maximum / float(max(height, width)))


def _draw_banner(canvas: np.ndarray, text: str) -> np.ndarray:
    """Prepend an instruction banner, so the prompt is visible inside the window."""
    banner = np.full((BANNER_HEIGHT, canvas.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(banner, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


def _caption(width: int, text: str) -> np.ndarray:
    strip = np.full((CAPTION_HEIGHT, width, 3), 40, dtype=np.uint8)
    cv2.putText(strip, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def _panels_canvas(
    reference_view: Optional[np.ndarray], reference_caption: str, current_view: np.ndarray, current_caption: str
) -> Tuple[np.ndarray, int]:
    """The two panels side by side under their captions; returns the canvas and the current panel's x offset."""
    current = np.vstack([_caption(current_view.shape[1], current_caption), current_view])
    if reference_view is None:
        return current, 0
    reference = np.vstack([_caption(reference_view.shape[1], reference_caption), reference_view])
    gap = np.full((current.shape[0], PANEL_GAP, 3), 24, dtype=np.uint8)
    return np.hstack([reference, gap, current]), reference.shape[1] + PANEL_GAP


def _draw_placed(
    canvas: np.ndarray, placed: Mapping[str, Sequence[float]], scale: float, color: Tuple[int, int, int] = POINT_COLOR
) -> None:
    for name, point in placed.items():
        position = (int(point[0] * scale), int(point[1] * scale))
        cv2.drawMarker(canvas, position, color, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(canvas, name, (position[0] + 7, position[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def place_point(
    image: np.ndarray,
    prompt: str,
    start: Optional[Sequence[float]] = None,
    reference_view: Optional[np.ndarray] = None,
    reference_caption: str = 'reference',
    placed_points: Optional[Mapping[str, Sequence[float]]] = None,
    allow_skip: bool = True,
    zoom: float = DEFAULT_ZOOM,
    window: str = 'Ground control points',
    old_points: Optional[Mapping[str, Sequence[float]]] = None,
    initial: Optional[Sequence[float]] = None,
) -> Placement:
    """
    Place one point in `image`, in a zoomed panel with an exact pixel mapping.

    With `start` the panel opens there; without it the full frame is shown first and a click
    chooses where the panel opens. `initial` places the point before any click, so Enter accepts it.
    In the panel: click to place, arrow keys or ijkl to nudge by one pixel, Enter to confirm, +/- to
    change the zoom, r to recenter on the placed point, f to return to the full frame. s skips (if
    allowed), u steps back one label, q finishes the frame, Esc aborts the session.
    """
    _check_gui()
    scale = _display_scale(image.shape)
    full = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image
    zooms = list(PANEL_ZOOMS)
    zoom_index = zooms.index(zoom) if zoom in zooms else zooms.index(DEFAULT_ZOOM)
    center: Optional[np.ndarray] = None if start is None else np.asarray(start, dtype=np.float64).copy()
    point: Optional[np.ndarray] = None if initial is None else np.asarray(initial, dtype=np.float64).copy()
    mode = 'panel' if center is not None else 'full'
    click: Dict[str, Optional[Tuple[int, int]]] = {'value': None}
    layout: Dict[str, Any] = {}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click['value'] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    skip_hint = 's = skip | ' if allow_skip else ''
    try:
        while True:
            if mode == 'panel':
                panel = Panel.centered(center, zooms[zoom_index])
                current_view = panel.render(image, marker=point)
                canvas, x_offset = _panels_canvas(
                    reference_view, reference_caption, current_view, f"this frame, zoom {zooms[zoom_index]:g}x"
                )
                layout = {'mode': 'panel', 'panel': panel, 'x': x_offset, 'y': CAPTION_HEIGHT}
                hint = (
                    'click = place | arrows/ijkl = nudge 1 px | Enter = confirm | +/- = zoom | '
                    'r = recenter | f = full frame'
                )
            else:
                canvas = full.copy()
                if placed_points:
                    _draw_placed(canvas, placed_points, scale)
                if old_points:
                    _draw_placed(canvas, old_points, scale, OLD_COLOR)
                if reference_view is not None:
                    thumb = cv2.resize(reference_view, (180, 180), interpolation=cv2.INTER_AREA)
                    canvas[10:190, canvas.shape[1] - 190 : canvas.shape[1] - 10] = thumb
                layout = {'mode': 'full', 'scale': scale}
                hint = 'click = open the zoomed panel there'
            framed = _draw_banner(canvas, f"{prompt} | {hint} | {skip_hint}u = back | q = finish | Esc = abort")
            cv2.imshow(window, framed)
            key = cv2.waitKeyEx(20)

            if click['value'] is not None:
                x, y = click['value']
                click['value'] = None
                y -= BANNER_HEIGHT
                if layout['mode'] == 'full':
                    if y >= 0:
                        center = np.array([x / layout['scale'], y / layout['scale']], dtype=np.float64)
                        mode = 'panel'
                else:
                    u, v = x - layout['x'], y - layout['y']
                    if 0 <= u < PANEL_SIZE and 0 <= v < PANEL_SIZE:
                        point = layout['panel'].to_source(u, v)
                continue

            if key == -1:
                continue
            char = chr(key) if 0 <= key < 128 else ''
            if key == 27:
                raise AnnotateError('Annotation aborted')
            if char in ('\r', '\n', ' ') and point is not None:
                return Placement('placed', point)
            if char == 's' and allow_skip:
                return Placement('skip')
            if char == 'u':
                return Placement('back')
            if char == 'q':
                return Placement('finish')
            if char == 'f':
                if mode == 'panel':
                    mode = 'full'
                else:
                    center = point.copy() if point is not None else np.array(image.shape[1::-1], dtype=float) / 2
                    mode = 'panel'
            elif char == 'r' and point is not None:
                center = point.copy()
                mode = 'panel'
            elif char in ('+', '='):
                zoom_index = min(zoom_index + 1, len(zooms) - 1)
            elif char == '-':
                zoom_index = max(zoom_index - 1, 0)
            elif point is not None and mode == 'panel':
                delta = np.zeros(2)
                if key in NUDGE_KEYS['left']:
                    delta[0] = -NUDGE_PX
                elif key in NUDGE_KEYS['right']:
                    delta[0] = NUDGE_PX
                elif key in NUDGE_KEYS['up']:
                    delta[1] = -NUDGE_PX
                elif key in NUDGE_KEYS['down']:
                    delta[1] = NUDGE_PX
                if np.any(delta):
                    point = point + delta
                    if not Panel.centered(center, zooms[zoom_index]).contains(point, margin=8):
                        center = point.copy()
    finally:
        cv2.destroyWindow(window)


def annotate_reference(image: np.ndarray, prefix: str = 'p', zoom: float = DEFAULT_ZOOM) -> Dict[str, List[float]]:
    """
    Annotate the reference frame; labels are assigned in click order.

    Each point is chosen in the full frame and refined in the zoomed panel. u removes the last
    point, q finishes, Esc aborts.
    """
    collected: Dict[str, List[float]] = {}
    while True:
        result = place_point(
            image,
            prompt=f"reference frame: {len(collected)} point(s) placed (GROUND PLANE only, spread them out)",
            placed_points=collected,
            allow_skip=False,
            zoom=zoom,
        )
        if result.action == 'placed':
            collected[f"{prefix}{len(collected) + 1:02d}"] = [float(result.point[0]), float(result.point[1])]
        elif result.action == 'back' and collected:
            collected.pop(next(reversed(collected)))
        elif result.action == 'finish':
            break
    if not collected:
        raise AnnotateError('No points were annotated in the reference frame')
    return collected


def annotate_reference_repeat(
    image: np.ndarray,
    first_session: Mapping[str, Sequence[float]],
    reference_index: int,
    zoom: float = DEFAULT_ZOOM,
    session: str = 'repeat',
    hint_name: str = 'first session',
) -> Dict[str, List[float]]:
    """
    Re-click the reference frame's points, label by label; a skipped label is left out.

    The earlier point is shown at `HINT_ZOOM` with a crosshair, and the working panel opens on it
    displaced by the session's jitter.
    """
    placed: Dict[str, List[float]] = {}
    labels = list(first_session)
    index = 0
    while index < len(labels):
        label = labels[index]
        first = np.asarray(first_session[label], dtype=np.float64)
        hint = Panel.centered(first, HINT_ZOOM).render(image, crosshair=True)
        result = place_point(
            image,
            prompt=f"{session} [{index + 1}/{len(labels)}] re-click '{label}' (hint = {hint_name}, coarse)",
            start=first + jitter(reference_index, label, session),
            reference_view=hint,
            reference_caption=f"{hint_name}: {label}, zoom {HINT_ZOOM:g}x",
            placed_points=placed,
            allow_skip=True,
            zoom=zoom,
        )
        if result.action == 'placed':
            placed[label] = [float(result.point[0]), float(result.point[1])]
            index += 1
        elif result.action == 'skip':
            placed.pop(label, None)
            index += 1
        elif result.action == 'back' and index > 0:
            index -= 1
            placed.pop(labels[index], None)
        elif result.action == 'finish':
            break
    if not placed:
        raise AnnotateError('No points were re-clicked in the reference frame')
    return placed


def annotate_against_reference(
    image: np.ndarray,
    reference_image: np.ndarray,
    reference_points: Mapping[str, Sequence[float]],
    predictions: Optional[Mapping[str, Sequence[float]]] = None,
    frame_index: int = 0,
    session: str = 'first',
    zoom: float = DEFAULT_ZOOM,
    skipped: Optional[List[str]] = None,
    proposals: Optional[Mapping[str, Proposal]] = None,
) -> Dict[str, List[float]]:
    """
    Annotate one frame by answering for each reference label in turn.

    Labels skipped with s are appended to `skipped` when given; labels never reached are in neither.

    The reference neighborhood is shown zoomed with a crosshair on the point. With a standing
    proposal, the working panel opens on it with the point placed there; otherwise it opens at the
    reference estimator's prediction displaced by the session's jitter, or in the full frame when
    there is no prediction or it falls outside the frame.
    """
    height, width = image.shape[:2]
    labels = list(reference_points)
    placed: Dict[str, List[float]] = {}
    index = 0
    while index < len(labels):
        label = labels[index]
        reference_view = Panel.centered(reference_points[label], zoom).render(reference_image, crosshair=True)
        start: Optional[np.ndarray] = None
        initial: Optional[np.ndarray] = None
        note = 'no prediction: click in the full frame first'
        predicted = None if predictions is None else predictions.get(label)
        proposal = None if proposals is None else proposals.get(label)
        if proposal is not None and proposal.point is not None:
            start = initial = np.asarray(proposal.point, dtype=np.float64)
            note = 'proposed: Enter = accept, click or arrows = correct'
        elif predicted is not None and 0 <= predicted[0] < width and 0 <= predicted[1] < height:
            start = np.asarray(predicted, dtype=np.float64) + jitter(frame_index, label, session)
            note = 'panel near the predicted position'
            if proposal is not None:
                note = f"no proposal ({proposal.reason}); {note}"
        elif predicted is not None:
            note = 'predicted outside the frame: s = skip if not visible'
        result = place_point(
            image,
            prompt=f"[{index + 1}/{len(labels)}] place '{label}' ({note})",
            start=start,
            reference_view=reference_view,
            reference_caption=f"reference: {label}, zoom {zoom:g}x",
            placed_points=placed,
            allow_skip=True,
            zoom=zoom,
            initial=initial,
        )
        if result.action == 'placed':
            placed[label] = [float(result.point[0]), float(result.point[1])]
            index += 1
        elif result.action == 'skip':
            placed.pop(label, None)
            if skipped is not None:
                skipped.append(label)
            index += 1
        elif result.action == 'back' and index > 0:
            index -= 1
            placed.pop(labels[index], None)
            if skipped is not None and labels[index] in skipped:
                skipped.remove(labels[index])
        elif result.action == 'finish':
            break
    return placed


def _oracle_homographies(case_dir: Path) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """The cached reference-estimator homographies, or None before `oracle` has run."""
    try:
        return oracle_matrices(load_oracle(case_dir))
    except (OracleError, FileNotFoundError, KeyError, ValueError):
        return None


def _frame_labels(case_dir: Path, index: int, image: np.ndarray) -> Optional[Labels]:
    """A frame's objects, or None when the case is not annotated."""
    labels = load_frame_labels(case_dir, index, image.shape[1], image.shape[0])
    return labels if len(labels) else None


def frame_proposals(
    case_dir: Path,
    reference_index: int,
    reference_image: np.ndarray,
    index: int,
    image: np.ndarray,
    points: Mapping[str, Sequence[float]],
) -> Dict[str, Proposal]:
    """`gcp.propose_points` for one frame of a case, with both frames' objects left out of the fit."""
    return propose_points(
        reference_image,
        image,
        points,
        _frame_labels(case_dir, reference_index, reference_image),
        _frame_labels(case_dir, index, image),
    )


def _proposal_metadata(
    metadata: Mapping[str, Any], proposals: Mapping[str, Proposal], previous: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """A frame's metadata with its proposals recorded under `proposals`, merged over `previous` ones."""
    recorded = dict(previous or {})
    recorded.update({label: proposal.to_dict() for label, proposal in proposals.items()})
    return {**metadata, 'proposal': proposal_settings(), 'proposals': recorded}


def _review_note(points: Mapping[str, Sequence[float]], proposals: Optional[Mapping[str, Proposal]]) -> str:
    """', N accepted, N corrected, N by hand' for a frame annotated with proposals, else ''."""
    if proposals is None:
        return ''
    proposed = {label: p.point for label, p in proposals.items() if p.point is not None}
    accepted = sum(label in proposed and np.array_equal(points[label], proposed[label]) for label in points)
    by_hand = sum(label not in proposed for label in points)
    return f": {accepted} accepted, {len(points) - accepted - by_hand} corrected, {by_hand} by hand"


def placement_conflicts(case_dir: Path, reference_index: int, points: Mapping[str, Sequence[float]]) -> Dict[str, str]:
    """
    Reference points that break the placement rule, with the reason: inside an object's OBB grown by
    `CLEARANCE_PX`, or inside an exclusion polygon (`EXCLUSION.json`), in the reference frame.
    """
    image = cv2.imread(str(frame_path(case_dir, reference_index)))
    if image is None:
        return {}
    shape = image.shape[:2]
    labels = load_frame_labels(case_dir, reference_index, shape[1], shape[0])
    near = rasterize_quads(labels.corners(), shape, CLEARANCE_PX) if len(labels) else np.zeros(shape, bool)
    exclusion_path = Path(case_dir) / EXCLUSION_FILE
    excluded = load_exclusion(exclusion_path, shape) if exclusion_path.is_file() else np.zeros(shape, bool)
    conflicts: Dict[str, str] = {}
    for label, point in points.items():
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            conflicts[label] = 'outside the frame'
        elif near[y, x]:
            conflicts[label] = f"within {CLEARANCE_PX} px of an object"
        elif excluded[y, x]:
            conflicts[label] = 'inside an exclusion polygon'
    return conflicts


def _warn_conflicts(case_dir: Path, reference_index: int, points: Mapping[str, Sequence[float]]) -> None:
    conflicts = placement_conflicts(case_dir, reference_index, points)
    for label, reason in conflicts.items():
        print(f"\033[93mReference point {label}: {reason}\033[0m")
    if conflicts:
        print(
            f"\033[93mRe-place them on the ground plane, clear of objects and structure: "
            f"annotate-gcp {case_dir} --labels {' '.join(conflicts)} --frames {reference_index}\033[0m"
        )


# Audited frames per offset bin when `--frames` is not given.
AUDIT_FRAMES_PER_BIN = 2


def audit_frames(
    frames: Sequence[int],
    reference_index: int,
    bins: Sequence[Tuple[int, Optional[int]]],
    per_bin: int = AUDIT_FRAMES_PER_BIN,
) -> List[int]:
    """
    Sampled frames to audit: `per_bin` from each offset bin, evenly spaced inside it, away from its ends
    when the bin holds enough frames.

    A bin with fewer sampled frames contributes all it has.
    """
    chosen: List[int] = []
    for low, high in bins:
        inside = sorted(
            f
            for f in frames
            if f != reference_index
            and low <= abs(f - reference_index)
            and (high is None or abs(f - reference_index) <= high)
        )
        if len(inside) <= per_bin:
            chosen.extend(inside)
            continue
        positions = np.unique(np.round(np.linspace(0, len(inside) - 1, per_bin + 2)[1:-1]).astype(int))
        if positions.size < per_bin:
            # Too few frames for distinct interior picks: spread over the whole bin instead.
            positions = np.round(np.linspace(0, len(inside) - 1, per_bin)).astype(int)
        chosen.extend(inside[p] for p in positions)
    return chosen


def annotate_reference_replace(
    image: np.ndarray,
    reference_points: Mapping[str, Sequence[float]],
    labels: Sequence[str],
    zoom: float = DEFAULT_ZOOM,
) -> Dict[str, List[float]]:
    """
    Place the named reference points anew, on new features or the same ones, label by label.

    The full frame opens with every other point and the label's old position (red); a click there
    opens the panel. A skipped label keeps its old point.
    """
    current = {label: list(map(float, point)) for label, point in reference_points.items()}
    placed: Dict[str, List[float]] = {}
    index = 0
    while index < len(labels):
        label = labels[index]
        result = place_point(
            image,
            prompt=f"replace [{index + 1}/{len(labels)}] '{label}': a new ground feature, away from vehicles "
            f"(s = keep the old one)",
            placed_points={name: point for name, point in current.items() if name != label},
            old_points={f"{label} (old)": reference_points[label]},
            allow_skip=True,
            zoom=zoom,
        )
        if result.action == 'placed':
            placed[label] = current[label] = [float(result.point[0]), float(result.point[1])]
            index += 1
        elif result.action == 'skip':
            index += 1
        elif result.action == 'back' and index > 0:
            index -= 1
            placed.pop(labels[index], None)
            current[labels[index]] = list(map(float, reference_points[labels[index]]))
        elif result.action == 'finish':
            break
    return placed


def _first_session_points(case_dir: Path, reference_index: int) -> Dict[str, List[float]]:
    """The first session's reference points: the repeat session's guide, the manual session's points."""
    path = case_dir / GCP_DIR / f"{reference_index:06d}.json"
    if not path.exists():
        raise AnnotateError(f"No first-session reference points at {path}; annotate the first session first")
    return {label: point.tolist() for label, point in load_gcp_file(path).as_dict().items()}


def redo_points(
    case_dir: Path,
    gcp_dir: Path,
    frames: Sequence[int],
    labels: Sequence[str],
    reference_index: int,
    reference_image: np.ndarray,
    homographies: Optional[Dict[str, Optional[np.ndarray]]],
    zoom: float = DEFAULT_ZOOM,
    session: str = 'redo',
    hint_points: Optional[Mapping[str, Sequence[float]]] = None,
    propose: bool = False,
) -> List[Path]:
    """
    Re-place the named points in already annotated frames, keeping every other point.

    The reference frame, when listed, is clicked first: in the full frame beside every other point
    (`annotate_reference_replace`), or, for a repeat session, guided by `hint_points`, the first
    session's points; a skipped label keeps the previous point. In other frames the panel opens as in a
    first pass, and a skipped label removes the point from the frame (it is not visible there). A
    label never reached keeps its point, unless its reference point moved, which drops it. The labels
    are recorded under `redone_labels` in the file's metadata. With `propose`, other frames open on
    proposals (`frame_proposals`), recorded beside the frame's earlier ones.
    """
    if not frames:
        raise AnnotateError('--labels re-places points in the frames given by --frames')
    reference_file = gcp_dir / f"{reference_index:06d}.json"
    if not reference_file.exists():
        raise AnnotateError(f"No reference points at {reference_file}; annotate the case first")
    reference_points = {label: point.tolist() for label, point in load_gcp_file(reference_file).as_dict().items()}
    unknown = [label for label in labels if label not in reference_points]
    if unknown:
        raise AnnotateError(f"No reference point labeled {', '.join(unknown)} in {reference_file}")
    missing = [index for index in frames if not (gcp_dir / f"{index:06d}.json").exists()]
    if missing:
        raise AnnotateError(f"Frame(s) {missing} have no points in {gcp_dir}; annotate them first")

    written: List[Path] = []
    moved: set = set()
    # The reference first, so the other frames are shown its corrected points.
    for index in sorted(set(int(i) for i in frames), key=lambda i: (i != reference_index, i)):
        path = gcp_dir / f"{index:06d}.json"
        stored = load_gcp_file(path).as_dict()
        metadata = dict(json.loads(path.read_text(encoding='utf-8')).get('stabilo_benchmark') or {})
        image = reference_image if index == reference_index else cv2.imread(str(frame_path(case_dir, index)))
        if image is None:
            raise AnnotateError(f"Could not read frame {index} from {case_dir}")
        targets = {label: reference_points[label] for label in labels}
        skipped: List[str] = []
        if index == reference_index:
            if hint_points:
                placed = annotate_reference_repeat(
                    image,
                    {label: hint_points[label] for label in labels},
                    reference_index,
                    zoom=zoom,
                    session=session,
                    hint_name='first session',
                )
            else:
                placed = annotate_reference_replace(image, reference_points, labels, zoom=zoom)
            moved = {label for label in placed if not np.allclose(placed[label], reference_points[label])}
            reference_points.update(placed)
        else:
            matrix = None if homographies is None else homographies.get(pair_key(FramePair(reference_index, index)))
            proposals = (
                frame_proposals(case_dir, reference_index, reference_image, index, image, targets) if propose else None
            )
            placed = annotate_against_reference(
                image,
                reference_image,
                targets,
                predict_positions(targets, matrix),
                index,
                session,
                zoom=zoom,
                skipped=skipped,
                proposals=proposals,
            )
            if proposals is not None:
                metadata = _proposal_metadata(metadata, proposals, metadata.get('proposals'))
            # A point left unanswered whose reference point moved marks the old feature: drop it.
            skipped.extend(label for label in moved if label not in placed and label not in skipped)
        current = {label: np.asarray(point).tolist() for label, point in stored.items()}
        current.update(placed)
        for label in skipped:
            current.pop(label, None)
        ordered = {label: current[label] for label in reference_points if label in current}
        ordered.update({label: point for label, point in current.items() if label not in ordered})
        metadata['redone_labels'] = sorted(set(metadata.get('redone_labels') or []) | set(placed) | set(skipped))
        save_gcp_file(
            path, ordered, image_path=frame_path(case_dir, index), image_size=image.shape[:2], metadata=metadata
        )
        changes = [f"{label} placed" for label in placed] + [f"{label} removed" for label in skipped]
        print(f"Frame {index}: {', '.join(changes) or 'unchanged'} -> {path}")
        written.append(path)
    return written


def run_annotate(case_dir: Path, args: argparse.Namespace) -> List[Path]:
    """
    Annotate the reference frame and the requested current frames of a prepared case.

    Existing files are kept unless --overwrite is given, so a session can be resumed.
    """
    case_dir = Path(case_dir).expanduser().resolve()
    case = load_case(case_dir)
    zoom = float(getattr(args, 'zoom', DEFAULT_ZOOM) or DEFAULT_ZOOM)
    # A repeat session writes to its own directory, reference frame included, for `audit`'s repeatability;
    # a manual session to its own, on the first session's reference points.
    repeat = bool(getattr(args, 'repeat', False))
    manual = bool(getattr(args, 'manual', False))
    if repeat and manual:
        raise AnnotateError("--repeat and --manual are separate sessions; give one")
    session = 'repeat' if repeat else 'manual' if manual else 'first'
    gcp_dir = case_dir / {'repeat': GCP_REPEAT_DIR, 'manual': GCP_MANUAL_DIR, 'first': GCP_DIR}[session]
    gcp_dir.mkdir(parents=True, exist_ok=True)
    reference_index = int(case['reference_index'])
    if (repeat or manual) and not (getattr(args, 'frames', None)):
        first = sorted(int(p.stem) for p in (case_dir / GCP_DIR).glob('*.json') if p.stem.isdigit())
        args.frames = [index for index in first if index != reference_index]

    homographies = _oracle_homographies(case_dir)
    if homographies is None:
        print(
            "\033[93mNo oracle cache: panels open in the full frame. Run `stabilo-benchmark oracle` first "
            "to start every panel at the reference estimator's prediction.\033[0m"
        )
    metadata = {
        'session': session,
        'prediction': 'oracle' if homographies is not None else 'none',
        'panel_zoom': zoom,
        'jitter_px': JITTER_PX,
    }

    reference_image = cv2.imread(str(frame_path(case_dir, reference_index)))
    if reference_image is None:
        raise AnnotateError(f"Could not read the reference frame {reference_index} from {case_dir}")

    # Proposals in the first session only: the repeat and manual sessions measure unaided clicks.
    propose = session == 'first' and bool(getattr(args, 'propose', True))
    labels = list(getattr(args, 'labels', None) or [])
    if labels:
        written = redo_points(
            case_dir,
            gcp_dir,
            [int(index) for index in (args.frames or [])],
            labels,
            reference_index,
            reference_image,
            homographies,
            zoom=zoom,
            session=f"redo-{session}",
            hint_points=_first_session_points(case_dir, reference_index) if repeat else None,
            propose=propose,
        )
        reference_file = gcp_dir / f"{reference_index:06d}.json"
        _warn_conflicts(case_dir, reference_index, load_gcp_file(reference_file).as_dict())
        return written

    reference_file = gcp_dir / f"{reference_index:06d}.json"
    if reference_file.exists() and not args.overwrite:
        reference_points = {label: point.tolist() for label, point in load_gcp_file(reference_file).as_dict().items()}
        print(f"Using {len(reference_points)} existing reference point(s) from {reference_file}")
    else:
        if repeat:
            first_file = case_dir / GCP_DIR / f"{reference_index:06d}.json"
            if not first_file.exists():
                raise AnnotateError(f"No first-session reference points at {first_file}; run without --repeat first")
            first_points = {label: point.tolist() for label, point in load_gcp_file(first_file).as_dict().items()}
            print("Re-clicking the reference frame's points for the repeat session.")
            reference_points = annotate_reference_repeat(reference_image, first_points, reference_index, zoom=zoom)
        elif manual:
            reference_points = _first_session_points(case_dir, reference_index)
            print(f"Placing the first session's frames again on its {len(reference_points)} reference point(s).")
        else:
            print("Annotating the reference frame. Click ground-plane features only, spread across the frame.")
            reference_points = annotate_reference(reference_image, zoom=zoom)
        save_gcp_file(
            reference_file,
            reference_points,
            image_path=frame_path(case_dir, reference_index),
            image_size=reference_image.shape[:2],
            metadata=metadata,
        )
        print(f"Saved {len(reference_points)} reference point(s) -> {reference_file}")
    _warn_conflicts(case_dir, reference_index, reference_points)

    written = [reference_file]
    requested = [int(index) for index in (args.frames or [])]
    if not requested:
        requested = audit_frames(case['frames'], reference_index, plan_bins(case.get('sampling'), 'audit_bins'))
        print(
            f"No --frames given; auditing {len(requested)} sampled frame(s), "
            f"{AUDIT_FRAMES_PER_BIN} per offset bin: {requested}"
        )

    for index in requested:
        output = gcp_dir / f"{index:06d}.json"
        if output.exists() and not args.overwrite:
            print(f"Skipping frame {index}: {output} exists (use --overwrite to redo)")
            continue
        image = cv2.imread(str(frame_path(case_dir, index)))
        if image is None:
            print(f"\033[93mSkipping frame {index}: not present in the case\033[0m")
            continue
        matrix = None
        if homographies is not None:
            matrix = homographies.get(pair_key(FramePair(reference_index, index)))
        predictions = predict_positions(reference_points, matrix)
        proposals = (
            frame_proposals(case_dir, reference_index, reference_image, index, image, reference_points)
            if propose
            else None
        )
        points = annotate_against_reference(
            image, reference_image, reference_points, predictions, index, session, zoom=zoom, proposals=proposals
        )
        if not points:
            print(f"\033[93mFrame {index}: no points placed, nothing written\033[0m")
            continue
        save_gcp_file(
            output,
            points,
            image_path=frame_path(case_dir, index),
            image_size=image.shape[:2],
            metadata=metadata if proposals is None else _proposal_metadata(metadata, proposals),
        )
        print(f"Saved {len(points)} point(s){_review_note(points, proposals)} -> {output}")
        written.append(output)

    print(f"\nNext: stabilo-benchmark audit {case_dir}")
    return written
