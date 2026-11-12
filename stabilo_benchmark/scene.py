#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
scene.py - Scene descriptors computed from annotations, used to select Part II's cases.

- coverage: fraction of the frame occupied by objects.
- concentration: how unimodal the object orientations are (`regions.motion_axis`).
- texture: corner density of the static background, at a fixed working resolution.
- object_motion / anchored_motion: where the reference frame's objects are in a later frame
  (static, displaced, moving together, gone), found by appearance with the platform's motion removed.

Everything is read from the frames and their boxes; nothing is tracked through the recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from stabilo_benchmark.masking import Labels
from stabilo_benchmark.regions import motion_axis, rasterize_quads

# Texture is computed with the long side resized to this, so frame size does not change it.
TEXTURE_LONG_SIDE = 1920
TEXTURE_QUALITY = 0.01
TEXTURE_MIN_DISTANCE = 8
# Upper bound on detected corners; `minDistance` saturates density well below it.
TEXTURE_MAX_CORNERS = 200_000

# Margin in pixels around each object excluded from the background.
TEXTURE_OBJECT_MARGIN_PX = 8


@dataclass(frozen=True)
class SceneStats:
    """What one frame looks like, in the terms the dominant-structure argument uses."""

    n_objects: int
    coverage: float
    concentration: float
    axis_deg: float
    texture: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def occupancy_mask(labels: Labels, width: int, height: int, dilation_px: int = 0) -> np.ndarray:
    """Boolean mask of the pixels the annotated objects occupy."""
    if len(labels) == 0:
        return np.zeros((int(height), int(width)), dtype=bool)
    return rasterize_quads(labels.corners(), (int(height), int(width)), dilation_px=dilation_px).astype(bool)


def coverage(labels: Labels, width: int, height: int) -> float:
    """Fraction of the frame occupied by the union of the object boxes."""
    if width <= 0 or height <= 0:
        return float('nan')
    return float(np.count_nonzero(occupancy_mask(labels, width, height)) / (int(width) * int(height)))


def background_texture(
    image: np.ndarray,
    labels: Optional[Labels] = None,
    long_side: int = TEXTURE_LONG_SIDE,
    min_distance: int = TEXTURE_MIN_DISTANCE,
) -> float:
    """
    Shi-Tomasi corners per megapixel of background, at a fixed working resolution.

    Objects are excluded with `TEXTURE_OBJECT_MARGIN_PX`. Returns NaN when no background remains.
    """
    if image is None or image.size == 0:
        return float('nan')
    height, width = image.shape[:2]
    scale = min(1.0, float(long_side) / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    search = np.full(gray.shape, 255, dtype=np.uint8)
    if labels is not None and len(labels):
        corners = labels.corners() * scale
        margin = max(1, round(TEXTURE_OBJECT_MARGIN_PX * scale))
        search[rasterize_quads(corners, gray.shape, dilation_px=margin).astype(bool)] = 0

    background_px = int(np.count_nonzero(search))
    if background_px == 0:
        return float('nan')
    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=TEXTURE_MAX_CORNERS,
        qualityLevel=TEXTURE_QUALITY,
        minDistance=min_distance,
        mask=search,
    )
    count = 0 if found is None else len(found)
    return float(count / (background_px / 1e6))


# --------------------------------------------------------------------------------------
# What a benchmark pair contains: the reference frame's objects, found again in a later frame
# --------------------------------------------------------------------------------------

# Long side at which the platform motion is estimated.
PLATFORM_LONG_SIDE = 960

# Samples along an object's long and short axis when its appearance is compared.
OBJECT_PATCH = (32, 14)

# A reference object and a later object are the same object when each is the other's best match,
# their appearance correlates at least this well, and their box dimensions agree within this ratio.
APPEARANCE_THRESHOLD = 0.5
SIZE_TOLERANCE = 0.3

# A found object is displaced when it moved more than DISPLACED_WIDTHS median object widths, clear of the
# apparent motion of a stationary object (box placement and platform-motion error). Two displaced objects
# move together when their displacements differ by at most COHERENT_WIDTHS median object widths or
# COHERENT_SHARE of the displacement they are compared with, whichever is larger: members of one stream
# drift apart in proportion to how far it moved.
DISPLACED_WIDTHS = 0.1
COHERENT_WIDTHS = 0.1
COHERENT_SHARE = 0.25

FATES = ('static', 'displaced', 'coherent', 'gone')

# Fewest displaced objects, on average over the offsets, for coherence to be defined: with one or two
# displaced objects, any group of them reads as coherent.
MIN_DISPLACED_OBJECTS = 4


def platform_motion(
    reference: np.ndarray,
    current: np.ndarray,
    reference_labels: Optional[Labels] = None,
    current_labels: Optional[Labels] = None,
    long_side: int = PLATFORM_LONG_SIDE,
) -> np.ndarray:
    """
    3x3 affine map from reference pixels to current-frame pixels: the platform's motion.

    Fitted by ECC on the background, with each frame's objects filled with the background mean (zero
    fill would leave object-shaped edges that move with the objects), from a phase-correlation
    translation. The mask is the reference frame's background; OpenCV applies it to the current
    frame, so the fit leaves out the current frame's pixels where the reference frame's objects are.
    Falls back to the translation if ECC does not converge.
    """
    if reference is None or current is None or reference.shape[:2] != current.shape[:2]:
        raise ValueError('platform_motion needs two frames of the same size')
    scale = min(1.0, float(long_side) / max(reference.shape[:2]))
    a, valid = _background(reference, reference_labels, scale)
    b, _ = _background(current, current_labels, scale)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp = cv2.findTransformECC(a, b, warp.copy(), cv2.MOTION_AFFINE, criteria, valid, 3)
    except cv2.error:
        warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    small = np.eye(3)
    small[:2] = warp
    to_small = np.diag([scale, scale, 1.0])
    return np.linalg.inv(to_small) @ small @ to_small


def _background(image: np.ndarray, labels: Optional[Labels], scale: float) -> tuple:
    """Grayscale frame at `scale` with objects filled by the background mean, and the background mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    occupied = np.zeros(gray.shape, dtype=bool)
    if labels is not None and len(labels):
        occupied = rasterize_quads(labels.corners(), gray.shape, dilation_px=TEXTURE_OBJECT_MARGIN_PX).astype(bool)
        if (~occupied).any():
            gray[occupied] = float(gray[~occupied].mean())
    size = (max(8, round(gray.shape[1] * scale)), max(8, round(gray.shape[0] * scale)))
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    valid = cv2.resize((~occupied).astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    return small, valid


def object_patches(image: np.ndarray, corners: np.ndarray, size: tuple = OBJECT_PATCH) -> np.ndarray:
    """
    (N, size[0] * size[1]) appearance of each box: resampled upright along its long axis, then
    normalized to zero mean and unit norm, so a dot product is a normalized cross-correlation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    length, width = size
    target = np.float32([[0, 0], [length - 1, 0], [0, width - 1]])
    out = np.zeros((len(corners), length * width), dtype=np.float32)
    for i, quad in enumerate(np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)):
        if not np.all(np.isfinite(quad)):
            continue
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        first = 0 if sides[0] + sides[2] >= sides[1] + sides[3] else 1
        source = np.float32([quad[first], quad[(first + 1) % 4], quad[(first + 3) % 4]])
        patch = cv2.warpAffine(
            gray, cv2.getAffineTransform(source, target), (length, width), borderMode=cv2.BORDER_REPLICATE
        ).astype(np.float32)
        patch -= patch.mean()
        norm = float(np.linalg.norm(patch))
        if norm > 1e-6:
            out[i] = (patch / norm).ravel()
    return out


def box_dimensions(corners: np.ndarray) -> tuple:
    """(lengths, widths) of (N, 4, 2) boxes: mean of opposite sides, long side first."""
    quads = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    sides = np.linalg.norm(np.roll(quads, -1, axis=1) - quads, axis=2)
    a, b = (sides[:, 0] + sides[:, 2]) / 2, (sides[:, 1] + sides[:, 3]) / 2
    return np.maximum(a, b), np.minimum(a, b)


def _areas(corners: np.ndarray) -> np.ndarray:
    quads = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    x, y = quads[..., 0], quads[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


@dataclass(frozen=True)
class ObjectMotion:
    """
    The reference frame's objects found again in a later frame (`object_motion`), one row each.

    `current_centers` are box centers in current-frame pixels (of the matched object; NaN when not
    found) and `displacement` the move in reference coordinates with the platform motion removed.
    `group` marks the largest displaced group that moved together.
    """

    reference_centers: np.ndarray
    current_centers: np.ndarray
    displacement: np.ndarray
    areas: np.ndarray
    found: np.ndarray
    moved: np.ndarray
    group: np.ndarray
    frame_area: float

    @property
    def shift(self) -> tuple:
        """Median displacement of the group, or NaNs without one."""
        if not self.group.any():
            return (float('nan'), float('nan'))
        return tuple(float(v) for v in np.median(self.displacement[self.group], axis=0))


def object_motion(
    reference: np.ndarray,
    current: np.ndarray,
    reference_labels: Labels,
    current_labels: Labels,
    motion: Optional[np.ndarray] = None,
) -> ObjectMotion:
    """
    Find each reference object in a later frame and measure how it moved.

    Each reference object is compared by appearance (`object_patches`, either 180-degree
    orientation) with every object of the later frame whose box dimensions agree within
    SIZE_TOLERANCE; it is found when its best match correlates at APPEARANCE_THRESHOLD or better and
    has it as its own best match in return. Its displacement is read in reference coordinates with
    the platform motion removed: `motion` maps reference pixels to current-frame pixels (an affine
    map or a homography; `platform_motion` when None). A found object is moved when it is farther
    than DISPLACED_WIDTHS median object widths from where it was; the group is the largest set of
    moved objects, by area, whose displacements agree with one member's within COHERENT_WIDTHS
    median object widths or COHERENT_SHARE of that member's displacement, whichever is larger.
    """
    height, width = reference.shape[:2]
    ref_corners = reference_labels.corners() if len(reference_labels) else np.zeros((0, 4, 2))
    count = len(ref_corners)
    nothing = np.zeros(count, dtype=bool)
    unmatched = np.full((count, 2), np.nan)
    empty = ObjectMotion(
        reference_centers=ref_corners.mean(axis=1) if count else np.zeros((0, 2)),
        current_centers=unmatched,
        displacement=unmatched,
        areas=_areas(ref_corners) if count else np.zeros(0),
        found=nothing,
        moved=nothing,
        group=nothing,
        frame_area=float(width * height),
    )
    if count == 0 or len(current_labels) == 0:
        return empty
    if motion is None:
        motion = platform_motion(reference, current, reference_labels, current_labels)

    cur_corners = current_labels.corners()
    ref_patches = object_patches(reference, ref_corners)
    cur_patches = object_patches(current, cur_corners)
    length, wide = OBJECT_PATCH
    flipped = cur_patches.reshape(-1, wide, length)[:, ::-1, ::-1].reshape(len(cur_patches), -1)
    similarity = np.maximum(ref_patches @ cur_patches.T, ref_patches @ flipped.T)
    ref_lengths, ref_widths = box_dimensions(ref_corners)
    cur_lengths, cur_widths = box_dimensions(cur_corners)
    alike = (np.abs(np.log(ref_lengths[:, None] / cur_lengths[None])) <= np.log1p(SIZE_TOLERANCE)) & (
        np.abs(np.log(ref_widths[:, None] / cur_widths[None])) <= np.log1p(SIZE_TOLERANCE)
    )
    similarity = np.where(alike, similarity, -1.0)
    best = similarity.argmax(axis=1)
    mutual = similarity.argmax(axis=0)[best] == np.arange(len(best))
    found = mutual & (similarity[np.arange(len(best)), best] >= APPEARANCE_THRESHOLD)

    current_centers = cur_corners.mean(axis=1)[best]
    placed = np.c_[current_centers, np.ones(count)] @ np.linalg.inv(motion).T
    displacement = placed[:, :2] / placed[:, 2:] - empty.reference_centers
    median_width = float(np.median(ref_widths))
    moved = found & (np.linalg.norm(displacement, axis=1) > DISPLACED_WIDTHS * median_width)

    group = np.zeros(count, dtype=bool)
    if moved.any():
        members = np.flatnonzero(moved)
        vectors = displacement[members]
        tolerance = np.maximum(COHERENT_WIDTHS * median_width, COHERENT_SHARE * np.linalg.norm(vectors, axis=1))
        together = np.linalg.norm(vectors[:, None] - vectors[None], axis=2) <= tolerance[:, None]
        leader = int(np.argmax(together.astype(np.float64) @ empty.areas[members]))
        group[members[together[leader]]] = True
    return ObjectMotion(
        reference_centers=empty.reference_centers,
        current_centers=np.where(found[:, None], current_centers, np.nan),
        displacement=np.where(found[:, None], displacement, np.nan),
        areas=empty.areas,
        found=found,
        moved=moved,
        group=group,
        frame_area=empty.frame_area,
    )


def anchored_motion(
    reference: np.ndarray,
    current: np.ndarray,
    reference_labels: Labels,
    current_labels: Labels,
    motion: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Where the reference frame's objects are in a later frame (`object_motion`), as fractions of the
    frame area:

        static     found within DISPLACED_WIDTHS median object widths of where it was;
        displaced  found farther away;
        coherent   the largest displaced group whose displacements agree with one member's
                   (`object_motion`): how much of the frame one wrong homography could fit;
        gone       not found: left the frame, occluded, or no longer recognizable;

    with `n_displaced`, the number of displaced objects, and `shift`, the group's median displacement.

    Thresholds scale with the objects, so the descriptors do not depend on ground resolution.
    """
    found = object_motion(reference, current, reference_labels, current_labels, motion=motion)
    share = found.areas / found.frame_area
    return {
        'static': float(share[found.found & ~found.moved].sum()),
        'displaced': float(share[found.moved].sum()),
        'coherent': float(share[found.group].sum()),
        'gone': float(share[~found.found].sum()),
        'n_displaced': int(found.moved.sum()),
        'shift': found.shift,
    }


def summarize_fates(per_offset: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """
    Mean of each fate over the sampled offsets (one per sampling bin, so bins weigh equally), and
    `coherence`, the coherent share of the displaced foreground: NaN unless at least
    MIN_DISPLACED_OBJECTS objects were displaced on average.
    """
    if not per_offset:
        return {**dict.fromkeys(FATES, float('nan')), 'n_displaced': float('nan'), 'coherence': float('nan')}
    means = {fate: float(np.mean([p[fate] for p in per_offset])) for fate in FATES}
    means['n_displaced'] = float(np.mean([p.get('n_displaced', 0) for p in per_offset]))
    defined = means['n_displaced'] >= MIN_DISPLACED_OBJECTS and means['displaced'] > 0
    means['coherence'] = means['coherent'] / means['displaced'] if defined else float('nan')
    return means


def describe(image: Optional[np.ndarray], labels: Labels, width: int, height: int) -> SceneStats:
    """Every descriptor for one frame. `image` may be None when texture is not wanted."""
    axis, concentration = motion_axis(labels.corners()) if len(labels) else (None, 0.0)
    return SceneStats(
        n_objects=len(labels),
        coverage=coverage(labels, width, height),
        concentration=float(concentration),
        axis_deg=float('nan') if axis is None else float(np.degrees(np.arctan2(axis[1], axis[0]))),
        texture=float('nan') if image is None else background_texture(image, labels),
    )
