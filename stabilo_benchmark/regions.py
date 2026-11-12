#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
regions.py - Metric regions: the pixels a similarity metric is computed over.

    region = valid_warp AND NOT objects_reference AND NOT objects_current AND ground_roi

  valid_warp        pixels of the warped current frame that received source data.
  objects_*         annotated objects in either frame, dilated by a constant margin.
  ground_roi        optional: a ground-plane polygon, less optional polygons over above-ground
                    structure (`evaluation_mask`), both in reference coordinates.

The current frame's objects are placed into reference coordinates by `mask_transform`, a
transform shared by all candidates, so every candidate is scored over the same object mask; only
`valid_warp` is candidate-specific. `RegionStats` records the surviving area.

Feature-suppression masks, which use the pipeline's own margin, are in `masking.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


class RegionError(ValueError):
    """Raised when a region cannot be constructed, e.g. from an unreadable ROI file."""


@dataclass(frozen=True)
class RegionStats:
    """How much of the frame a metric was actually computed over."""

    area_px: int
    area_fraction: float
    valid_fraction: float
    object_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'region_area_px': self.area_px,
            'region_area_fraction': round(self.area_fraction, 6),
            'region_valid_fraction': round(self.valid_fraction, 6),
            'region_object_fraction': round(self.object_fraction, 6),
        }


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


def rasterize_quads(corners: np.ndarray, shape: Tuple[int, int], dilation_px: int = 0) -> np.ndarray:
    """
    Rasterize (N, 4, 2) quadrilaterals into a boolean mask (True inside a quad).

    `dilation_px` grows the mask by a constant distance in pixels, whatever the object size.
    """
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)

    corners = np.asarray(corners, dtype=np.float64)
    if corners.size:
        quads = corners.reshape(-1, 4, 2)
        # Non-finite quads are skipped.
        finite = np.all(np.isfinite(quads), axis=(1, 2))
        # One `fillPoly` call per quad: with several polygons in one call it applies the even-odd rule,
        # and overlapping quads cancel.
        for quad in quads[finite]:
            cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)

    if dilation_px > 0 and mask.any():
        size = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)

    return mask.astype(bool)


def warp_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 homography (or 2x3 affine) to points of any leading shape ending in 2.

    Float64 numpy rather than `cv2.perspectiveTransform`, which works in float32 and returns (0, 0)
    when the projective scale is zero or NaN. Here w -> 0 gives inf and NaN stays NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.copy()
    original_shape = points.shape
    flat = points.reshape(-1, 2)

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    elif matrix.shape != (3, 3):
        raise RegionError(f"Transformation matrix must be 3x3 or 2x3, got shape {matrix.shape}")

    homogeneous = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float64)])
    projected = homogeneous @ matrix.T
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = projected[:, :2] / projected[:, 2:3]

    return transformed.reshape(original_shape)


def warp_validity(
    matrix: np.ndarray,
    source_shape: Tuple[int, int],
    target_shape: Optional[Tuple[int, int]] = None,
    erode_px: int = 2,
) -> np.ndarray:
    """
    Pixels of the target frame that receive data when the source is warped by `matrix`.

    Warps an all-ones image with nearest-neighbor sampling, then erodes by `erode_px` to drop the
    interpolation fringe.
    """
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    dst_h, dst_w = (int(target_shape[0]), int(target_shape[1])) if target_shape is not None else (src_h, src_w)

    ones = np.full((src_h, src_w), 255, dtype=np.uint8)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        warped = cv2.warpAffine(ones, matrix, (dst_w, dst_h), flags=cv2.INTER_NEAREST)
    elif matrix.shape == (3, 3):
        warped = cv2.warpPerspective(ones, matrix, (dst_w, dst_h), flags=cv2.INTER_NEAREST)
    else:
        raise RegionError(f"Transformation matrix must be 3x3 or 2x3, got shape {matrix.shape}")

    valid = warped > 0
    if erode_px > 0 and valid.any():
        valid = _erode(valid, erode_px)
    return valid


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Erode a boolean mask, treating everything outside the frame as background.

    `cv2.erode`'s default border counts outside pixels as foreground, so it would not erode at the
    frame edge.
    """
    size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return eroded.astype(bool)


def _load_polygons(path: Path, what: str, allow_empty: bool = False) -> list:
    """Polygons of a LabelMe JSON (`polygon` shapes) or of an `x y` text file (one polygon)."""
    path = Path(path)
    if not path.exists():
        raise RegionError(f"{what} file not found: {path}")
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        polygons = [
            np.round(np.asarray(shape_entry['points'], dtype=np.float64)).astype(np.int32)
            for shape_entry in payload.get('shapes', [])
            if shape_entry.get('shape_type') == 'polygon'
        ]
        if not polygons and not allow_empty:
            raise RegionError(f"No polygon shapes found in {what} file {path}")
        return polygons
    points = np.loadtxt(str(path), ndmin=2)
    if points.shape[1] != 2:
        raise RegionError(f"{what} text file {path} must have two columns (x y), found {points.shape[1]}")
    return [np.round(points).astype(np.int32)]


def _rasterize_polygons(path: Path, shape: Tuple[int, int], what: str, allow_empty: bool = False) -> np.ndarray:
    """
    Union of a file's polygons as a boolean mask; raises when it covers no pixel of the frame, unless
    `allow_empty` and the file holds no polygon at all.
    """
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _load_polygons(path, what, allow_empty)
    # One `fillPoly` call per polygon: several in one call follow the even-odd rule, so overlaps cancel.
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    if polygons and not mask.any():
        raise RegionError(f"{what} from {path} covers no pixels of a {width}x{height} frame")
    return mask.astype(bool)


def load_roi(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """
    Load a ground-plane region of interest as a boolean mask (True = evaluate here).

    Accepts LabelMe JSON with one or more `polygon` shapes, or plain text with two columns of `x y`
    vertices describing one polygon. Polygons are unioned. A missing file, or one that covers no
    pixels, raises.
    """
    return _rasterize_polygons(path, shape, 'Ground-plane ROI')


# A case's exclusion polygons, drawn on its reference frame (`scripts/draw_exclusion.sh`).
EXCLUSION_FILE = 'EXCLUSION.json'


def load_exclusion(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """
    Load polygons over above-ground structure, drawn on the reference frame, as a boolean mask
    (True = excluded from evaluation). Same formats and checks as `load_roi`, except that a LabelMe
    file with no polygon is valid: the frame has nothing to exclude.
    """
    return _rasterize_polygons(path, shape, 'Exclusion', allow_empty=True)


def evaluation_mask(
    shape: Tuple[int, int], roi: Optional[np.ndarray] = None, exclusion: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """The `roi` argument of `metric_region`: the ROI (whole frame when absent) less the exclusion."""
    if roi is None and exclusion is None:
        return None
    mask = np.ones((int(shape[0]), int(shape[1])), dtype=bool) if roi is None else roi.copy()
    if exclusion is not None:
        mask &= ~exclusion
    if not mask.any():
        raise RegionError("The region of interest less the exclusion covers no pixels")
    return mask


# --------------------------------------------------------------------------------------
# Region assembly
# --------------------------------------------------------------------------------------


def object_region(
    reference_corners: Optional[np.ndarray],
    current_corners: Optional[np.ndarray],
    mask_transform: Optional[np.ndarray],
    shape: Tuple[int, int],
    dilation_px: int = 8,
) -> np.ndarray:
    """
    Boolean mask of pixels occupied by an annotated object in either frame (True = object).

    `current_corners` are mapped into reference coordinates by `mask_transform`, the transform shared
    by all candidates; None means they are already in reference coordinates.
    """
    masks = []
    if reference_corners is not None and np.size(reference_corners):
        masks.append(rasterize_quads(reference_corners, shape, dilation_px))
    if current_corners is not None and np.size(current_corners):
        corners = current_corners if mask_transform is None else warp_points(current_corners, mask_transform)
        masks.append(rasterize_quads(corners, shape, dilation_px))

    if not masks:
        return np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    return np.logical_or.reduce(masks)


def metric_region(
    shape: Tuple[int, int],
    homography: np.ndarray,
    reference_corners: Optional[np.ndarray] = None,
    current_corners: Optional[np.ndarray] = None,
    mask_transform: Optional[np.ndarray] = None,
    dilation_px: int = 8,
    border_px: int = 0,
    roi: Optional[np.ndarray] = None,
    source_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, RegionStats]:
    """
    Assemble the region a similarity metric may be computed over.

    Args:
        shape: (height, width) of the reference frame, the frame everything is warped onto.
        homography: The candidate's current -> reference transform, used for the validity term.
        reference_corners: (N, 4, 2) object corners in the reference frame.
        current_corners: (M, 4, 2) object corners in the current frame.
        mask_transform: Common current -> reference transform used to place
            `current_corners`. Defaults to `homography`.
        dilation_px: Constant-pixel margin grown around objects.
        border_px: Additional erosion of the whole region.
        roi: Optional evaluation mask from `evaluation_mask`.
        source_shape: Shape of the current frame if it differs from the reference.

    Returns:
        (region, stats) where `region` is boolean with True meaning "include this pixel".
    """
    height, width = int(shape[0]), int(shape[1])
    total = float(height * width)

    valid = warp_validity(homography, source_shape or (height, width), (height, width))
    objects = object_region(
        reference_corners,
        current_corners,
        homography if mask_transform is None else mask_transform,
        (height, width),
        dilation_px,
    )

    region = valid & ~objects
    if roi is not None:
        if roi.shape != region.shape:
            raise RegionError(f"ROI shape {roi.shape} does not match frame shape {region.shape}")
        region &= roi
    if border_px > 0 and region.any():
        region = _erode(region, border_px)

    area = int(region.sum())
    stats = RegionStats(
        area_px=area,
        area_fraction=area / total,
        valid_fraction=float(valid.sum()) / total,
        object_fraction=float(objects.sum()) / total,
    )
    return region, stats


def region_is_usable(stats: RegionStats, minimum_fraction: float = 0.05) -> bool:
    """Whether a region covers at least `minimum_fraction` of the frame."""
    return stats.area_fraction >= minimum_fraction


def motion_axis(corners: np.ndarray, weight_by_area: bool = True) -> Tuple[Optional[np.ndarray], float]:
    """
    Dominant motion axis of a frame, from the long axes of its oriented boxes.

    Orientations are axial, so the area-weighted mean is taken over doubled angles and halved back.

    Returns:
        (unit vector along the dominant axis, concentration in [0, 1]). Concentration is the
        resultant length of the doubled-angle mean: 1 when every object is aligned, near 0 when
        orientations are spread or bimodal. The axis is None, and concentration 0, when there are no
        usable boxes.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    if corners.shape[0] == 0:
        return None, 0.0

    # Long axis of each quadrilateral: the longer of the two adjacent edge directions.
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 3] - corners[:, 0]
    length_a = np.linalg.norm(edge_a, axis=1)
    length_b = np.linalg.norm(edge_b, axis=1)
    long_axis = np.where((length_a >= length_b)[:, None], edge_a, edge_b)

    lengths = np.linalg.norm(long_axis, axis=1)
    usable = lengths > 1e-9
    if not np.any(usable):
        return None, 0.0
    long_axis, lengths = long_axis[usable], lengths[usable]

    angles = np.arctan2(long_axis[:, 1], long_axis[:, 0])
    if weight_by_area:
        weights = np.abs(_shoelace(corners[usable]))
        if weights.sum() <= 0:
            weights = np.ones_like(angles)
    else:
        weights = np.ones_like(angles)

    resultant = np.sum(weights * np.exp(2j * angles)) / weights.sum()
    concentration = float(np.abs(resultant))
    mean_angle = float(np.angle(resultant) / 2.0)
    return np.array([np.cos(mean_angle), np.sin(mean_angle)]), concentration


def dominant_motion(
    reference_corners: np.ndarray,
    current_corners: np.ndarray,
    matches: Sequence[Tuple[int, int]],
) -> Optional[np.ndarray]:
    """
    Mean displacement of corresponded objects between two frames, in reference pixels.

    `matches` are (reference index, current index) pairs, e.g. from track ids.
    """
    reference_corners = np.asarray(reference_corners, dtype=np.float64).reshape(-1, 4, 2)
    current_corners = np.asarray(current_corners, dtype=np.float64).reshape(-1, 4, 2)

    pairs = [(int(i), int(j)) for i, j in matches]
    if not pairs:
        return None

    reference_centers = reference_corners.mean(axis=1)
    current_centers = current_corners.mean(axis=1)
    displacements = np.array([current_centers[j] - reference_centers[i] for i, j in pairs])
    return displacements.mean(axis=0)


def _shoelace(corners: np.ndarray) -> np.ndarray:
    """Signed polygon area of each (4, 2) quadrilateral, via the shoelace formula."""
    x, y = corners[..., 0], corners[..., 1]
    return 0.5 * (np.sum(x * np.roll(y, -1, axis=-1), axis=-1) - np.sum(y * np.roll(x, -1, axis=-1), axis=-1))
