#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
geometric.py - Error of an estimated homography against a known or reference one.

  CTE        Corner transfer error: the four frame corners are sent through the true transform
             and back through the estimate; the error is their displacement, in pixels.

  HEA(eps)   Homography estimation accuracy: the fraction of trials whose mean CTE is at most
             eps pixels. mAA is its mean over an interval of thresholds.

  IoU        Intersection over union of each object's box with the same box after the round
             trip. Each corner is carried separately and the quadrilaterals are compared as they
             are, with no rectangle or axis-aligned hull refitted.

  OCE        Object center error: distance between each box's center and the center of the same
             box after the round trip, in pixels.

IoU and OCE are computed on OBBs by default, HBBs on request (`metrics.object_boxes`), and
summarized per frame by `OBJECT_STATISTICS`. The Frobenius norm of the matrix difference is also
recorded; it depends on the image size and is not used for ranking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon

from stabilo_benchmark.regions import warp_points


def frame_corners(width: int, height: int) -> np.ndarray:
    """The four frame corners, ordered TL, TR, BR, BL."""
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def corner_transfer(
    true_matrix: np.ndarray,
    estimated_matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Per-corner displacement vectors after a round trip through both transforms.

    `true_matrix` maps reference -> perturbed (or reference -> current); `estimated_matrix` is the
    estimate of the inverse direction. Returns a (4, 2) array of displacement vectors.
    """
    corners = frame_corners(width, height)
    forward = warp_points(corners, np.asarray(true_matrix, dtype=np.float64))
    round_trip = warp_points(forward, np.asarray(estimated_matrix, dtype=np.float64))
    return round_trip - corners


def corner_transfer_error(
    true_matrix: np.ndarray,
    estimated_matrix: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, float]:
    """Scalar summaries of the corner transfer error, in pixels."""
    displacement = corner_transfer(true_matrix, estimated_matrix, width, height)
    distances = np.linalg.norm(displacement, axis=1)
    if not np.all(np.isfinite(distances)):
        return {'cte_mean': float('nan'), 'cte_max': float('nan'), 'cte_rms': float('nan')}
    return {
        'cte_mean': float(distances.mean()),
        'cte_max': float(distances.max()),
        'cte_rms': float(np.sqrt(np.mean(distances**2))),
    }


def corner_disagreement(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, float]:
    """
    Corner transfer error between two homographies of the same direction.

    `matrix_a` is inverted before the round trip. The result is in reference-frame pixels.
    """
    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    if matrix_a.shape == (2, 3):
        matrix_a = np.vstack([matrix_a, [0.0, 0.0, 1.0]])
    try:
        forward = np.linalg.inv(matrix_a)
    except np.linalg.LinAlgError:
        return {'cte_mean': float('nan'), 'cte_max': float('nan'), 'cte_rms': float('nan')}
    return corner_transfer_error(forward, matrix_b, width, height)


def corner_disagreement_vectors(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Per-corner disagreement vectors for two same-direction homographies, or NaNs."""
    matrix_a = np.asarray(matrix_a, dtype=np.float64)
    if matrix_a.shape == (2, 3):
        matrix_a = np.vstack([matrix_a, [0.0, 0.0, 1.0]])
    try:
        forward = np.linalg.inv(matrix_a)
    except np.linalg.LinAlgError:
        return np.full((4, 2), np.nan)
    return corner_transfer(forward, matrix_b, width, height)


def hea(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
    """
    Homography estimation accuracy: the fraction of errors at or below each pixel threshold.

    `errors` are per-trial mean corner transfer errors. NaN entries count as failures.
    """
    values = np.asarray(list(errors), dtype=np.float64)
    if values.size == 0:
        return {f"hea_{threshold_label(t)}": float('nan') for t in thresholds}
    finite = np.isfinite(values)
    return {
        f"hea_{threshold_label(t)}": float(np.mean(finite & (np.nan_to_num(values, nan=np.inf) <= float(t))))
        for t in thresholds
    }


def maa(errors: Sequence[float], upper: float, lower: float = 0.0) -> float:
    """
    Mean average accuracy: the HEA curve averaged over thresholds in [`lower`, `upper`].

    Computed in closed form, since HEA is a step function of the threshold:

        int_lower^upper HEA(e) de = (1/n) * sum_i (upper - clip(e_i, lower, upper)),

    normalized by `upper - lower`. NaN errors clip to `upper` and contribute nothing.
    """
    if not (upper > lower):
        raise ValueError(f"maa needs upper > lower, got lower={lower}, upper={upper}")
    values = np.asarray(list(errors), dtype=np.float64)
    if values.size == 0:
        return float('nan')
    clipped = np.clip(np.nan_to_num(values, nan=np.inf), lower, upper)
    return float(np.mean(upper - clipped) / (upper - lower))


def threshold_label(threshold: float) -> str:
    """A stable, filesystem- and LaTeX-safe column suffix: '1px', '2.5px'."""
    value = float(threshold)
    text = f"{value:g}".replace('.', 'p')
    return f"{text}px"


def homography_frobenius_error(true_matrix: np.ndarray, estimated_matrix: np.ndarray) -> float:
    """
    Frobenius norm of `inv(true) - estimated`, both normalized by their (3, 3) entry.

    Dominated by the translation entries, so it depends on the image size; not used for ranking.
    """
    try:
        target = np.linalg.inv(np.asarray(true_matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return float('nan')
    estimate = np.asarray(estimated_matrix, dtype=np.float64)
    if estimate.shape == (2, 3):
        estimate = np.vstack([estimate, [0.0, 0.0, 1.0]])
    if estimate.shape != (3, 3):
        return float('nan')

    target = _normalize_homography(target)
    estimate = _normalize_homography(estimate)
    if target is None or estimate is None:
        return float('nan')
    return float(np.linalg.norm(target - estimate, 'fro'))


def _normalize_homography(matrix: np.ndarray) -> Optional[np.ndarray]:
    """Scale a homography so its (3, 3) entry is 1, resolving the scale ambiguity."""
    scale = matrix[2, 2]
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return None
    return matrix / scale


# --------------------------------------------------------------------------------------
# Object-level agreement
# --------------------------------------------------------------------------------------


# Summary statistics of a per-object metric over the objects of one frame, in record order.
OBJECT_STATISTICS = ('mean', 'std', 'min', 'q1', 'median', 'q3', 'max')


def quad_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """
    Element-wise IoU of two equal-length sets of (N, 4, 2) quadrilaterals.

    Invalid quadrilaterals are repaired with a zero-width buffer; non-finite or zero-area ones
    score 0.
    """
    a = np.asarray(corners_a, dtype=np.float64).reshape(-1, 4, 2)
    b = np.asarray(corners_b, dtype=np.float64).reshape(-1, 4, 2)
    if a.shape != b.shape:
        raise ValueError(f"Corner arrays must have the same shape, got {a.shape} and {b.shape}")
    if a.size == 0:
        return np.zeros((0,), dtype=np.float64)

    scores = np.zeros(a.shape[0], dtype=np.float64)
    for i, (quad_a, quad_b) in enumerate(zip(a, b, strict=True)):
        if not (np.all(np.isfinite(quad_a)) and np.all(np.isfinite(quad_b))):
            continue
        polygon_a, polygon_b = _to_polygon(quad_a), _to_polygon(quad_b)
        if polygon_a is None or polygon_b is None:
            continue
        union = polygon_a.union(polygon_b).area
        scores[i] = polygon_a.intersection(polygon_b).area / union if union > 0 else 0.0
    return scores


def _to_polygon(quad: np.ndarray) -> Optional[Polygon]:
    """A valid shapely polygon from four corners, or None if it has no area."""
    polygon = Polygon(quad)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def summarize_objects(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    `OBJECT_STATISTICS` of one per-object metric, as `<prefix>_<statistic>` keys.

    The standard deviation is the population one (ddof=0). An empty or non-finite input gives NaN
    for every statistic.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        return nan_object_statistics(prefix)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    stats = {
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': values.max(),
    }
    return {f"{prefix}_{name}": float(stats[name]) for name in OBJECT_STATISTICS}


def nan_object_statistics(prefix: str) -> Dict[str, float]:
    """Every `OBJECT_STATISTICS` key of `prefix` set to NaN."""
    return {f"{prefix}_{name}": float('nan') for name in OBJECT_STATISTICS}


def nan_object_scores() -> Dict[str, Any]:
    """IoU and OCE values for a trial where no transform was produced."""
    return {**nan_object_statistics('iou'), **nan_object_statistics('oce')}


def object_iou(corners_reference: np.ndarray, corners_estimate: np.ndarray) -> Dict[str, Any]:
    """
    IoU statistics over the objects of one frame, plus the object count `n_objects`.

    Both inputs are (N, 4, 2) corner arrays in the reference frame.
    """
    scores = quad_iou(corners_reference, corners_estimate)
    return {**summarize_objects(scores, 'iou'), 'n_objects': int(scores.size)}


def box_centers(corners: np.ndarray) -> np.ndarray:
    """(N, 2) centers of (N, 4, 2) boxes: the mean of the corners, exact for a rectangle."""
    return np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2).mean(axis=1)


def object_center_error(corners_reference: np.ndarray, corners_estimate: np.ndarray) -> Dict[str, float]:
    """
    Object center error (OCE) statistics over the objects of one frame, in pixels.

    Distance between the centers (`box_centers`) of each box as placed by the reference transform
    and by the estimate, both (N, 4, 2) in the reference frame. A non-finite corner makes every
    statistic NaN.
    """
    reference = np.asarray(corners_reference, dtype=np.float64).reshape(-1, 4, 2)
    estimate = np.asarray(corners_estimate, dtype=np.float64).reshape(-1, 4, 2)
    if reference.shape != estimate.shape:
        raise ValueError(f"Corner arrays must have the same shape, got {reference.shape} and {estimate.shape}")
    distances = np.linalg.norm(box_centers(estimate) - box_centers(reference), axis=1)
    return summarize_objects(distances, 'oce')


def geometric_scores(
    true_matrix: np.ndarray,
    estimated_matrix: np.ndarray,
    width: int,
    height: int,
    corners_reference: Optional[np.ndarray] = None,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
) -> Dict[str, Any]:
    """
    Every geometric metric for one estimate, as a flat record.

    Records `within_<eps>` per HEA threshold; `hea` averages it over trials. `corners_reference`
    are the scene's object boxes in the reference frame, used for IoU and OCE.
    """
    scores: Dict[str, Any] = corner_transfer_error(true_matrix, estimated_matrix, width, height)
    scores['h_frobenius'] = homography_frobenius_error(true_matrix, estimated_matrix)
    for threshold in hea_thresholds:
        scores[f"within_{threshold_label(threshold)}"] = bool(scores['cte_mean'] <= float(threshold))

    if corners_reference is not None and np.size(corners_reference):
        round_trip = warp_points(warp_points(corners_reference, true_matrix), estimated_matrix)
        scores.update(object_iou(corners_reference, round_trip))
        scores.update(object_center_error(corners_reference, round_trip))

    return scores
