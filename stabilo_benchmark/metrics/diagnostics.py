#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
diagnostics.py - Internal statistics of an estimate, and the motion-alignment decomposition.

Keypoint, match and inlier statistics describe how a fit was reached; they are recorded but not
used for ranking. `motion_decomposition` splits the registration error into components parallel
and perpendicular to the dominant object motion; `capture_fraction` measures how far an estimate
follows a group of objects that moved together, and `inlier_object_shares` where its inliers lie.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def matching_stats(stabilizer: Any) -> Dict[str, Any]:
    """
    Keypoint, match and inlier statistics of the most recent `stabilize()` call.

    `get_cur_inliers_count()` is None on every estimation-failure path.
    """
    reference_keypoints, current_keypoints = stabilizer.get_cur_num_keypoints()
    matches = stabilizer.get_cur_num_matches()
    inliers = stabilizer.get_cur_inliers_count()

    ratio = float(inliers) / float(matches) if (inliers is not None and matches) else float('nan')
    return {
        'n_keypoints_ref': reference_keypoints,
        'n_keypoints_cur': current_keypoints,
        'n_matches': matches,
        'n_inliers': inliers,
        'inlier_ratio': ratio,
    }


def estimation_failed(stabilizer: Any, matrix: Optional[np.ndarray]) -> bool:
    """
    Whether the estimator failed to produce a transform for this pair.

    In benchmark mode stabilo returns the identity instead of raising, and every such fallback
    leaves the inlier count None. A failure is a missing or non-finite matrix or a None inlier
    count. An exact identity with inliers is an estimate: between near-identical frames, keypoints
    on a pixel grid can match at identical coordinates.
    """
    if matrix is None:
        return True
    if stabilizer.get_cur_inliers_count() is None:
        return True
    return not bool(np.all(np.isfinite(np.asarray(matrix, dtype=np.float64))))


def motion_decomposition(
    displacement: np.ndarray,
    motion: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Split a registration error into components parallel and perpendicular to the object motion.

    Args:
        displacement: (N, 2) error vectors, typically the four corner transfer errors.
        motion: (2,) motion direction, e.g. from `regions.dominant_motion` or `regions.motion_axis`.
            None when there is none.

    Returns:
        The mean absolute parallel and perpendicular components in pixels, plus
        `motion_alignment` = parallel / (parallel + perpendicular), in [0, 1]: ~0.5 for isotropic
        error, -> 1 for error along the motion, -> 0 for error across it.
    """
    empty = {
        'error_parallel_px': float('nan'),
        'error_perpendicular_px': float('nan'),
        'motion_alignment': float('nan'),
    }
    if motion is None:
        return empty

    motion = np.asarray(motion, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(motion))
    if not np.isfinite(norm) or norm < 1e-9:
        return empty  # objects present but effectively stationary: nothing to align with

    direction = motion / norm
    normal = np.array([-direction[1], direction[0]])

    displacement = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    if displacement.size == 0 or not np.all(np.isfinite(displacement)):
        return empty

    parallel = float(np.mean(np.abs(displacement @ direction)))
    perpendicular = float(np.mean(np.abs(displacement @ normal)))
    total = parallel + perpendicular
    alignment = parallel / total if total > 1e-9 else float('nan')

    return {
        'error_parallel_px': parallel,
        'error_perpendicular_px': perpendicular,
        'motion_alignment': alignment,
        'motion_magnitude_px': norm,
    }


# A captured estimate follows at least this share of the moving group's displacement where the group
# lies; an estimate that sees no feature on the objects stays well below it.
CAPTURE_THRESHOLD = 0.1

# An estimate has broken down when its corner error against the reference estimator reaches this.
BREAKDOWN_PX = 100.0


def capture_fraction(
    estimate: Optional[np.ndarray],
    reference: Optional[np.ndarray],
    current_points: np.ndarray,
    displacement: np.ndarray,
) -> float:
    """
    How far an estimate follows a group of objects that moved together, rather than the ground.

    Args:
        estimate, reference: 3x3 maps from current-frame to reference-frame pixels: the candidate's
            and the reference estimator's.
        current_points: (N, 2) the group's object centers in the current frame.
        displacement: (N, 2) their moves in reference coordinates under `reference`.

    Returns:
        The median over the group of -<H(x) - H_ref(x), d> / |d|^2: 0 for an estimate that agrees
        with the reference where the objects are, 1 for one that registers them back onto where they
        were at the reference, which is what a homography fitted to the group does. NaN without an
        estimate or a group.
    """
    points = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    moves = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    if estimate is None or reference is None or len(points) == 0:
        return float('nan')
    error = _project(estimate, points) - _project(reference, points)
    squared = np.sum(moves * moves, axis=1)
    usable = np.isfinite(squared) & (squared > 1e-12) & np.all(np.isfinite(error), axis=1)
    if not usable.any():
        return float('nan')
    return float(np.median(-np.sum(error[usable] * moves[usable], axis=1) / squared[usable]))


def group_error(
    estimate: Optional[np.ndarray],
    reference: Optional[np.ndarray],
    current_points: np.ndarray,
    displacement: np.ndarray,
) -> Tuple[float, float]:
    """
    How far an estimate errs where a group lies, and in which direction.

    Args:
        estimate, reference, current_points, displacement: as for `capture_fraction`.

    Returns:
        The median over the group of the error's length in pixels, and the angle in degrees between
        the error and the group's displacement, from the median cosine: 0 for an estimate that moves
        the objects back along their displacement, 180 for one that moves them further on. NaN
        without an estimate or a group.
    """
    points = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    moves = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    if estimate is None or reference is None or len(points) == 0:
        return float('nan'), float('nan')
    error = _project(reference, points) - _project(estimate, points)
    lengths = np.linalg.norm(error, axis=1)
    scale = lengths * np.linalg.norm(moves, axis=1)
    usable = np.isfinite(scale) & (scale > 1e-12)
    if not usable.any():
        return float('nan'), float('nan')
    cosine = np.sum(error[usable] * moves[usable], axis=1) / scale[usable]
    return float(np.median(lengths[usable])), float(np.degrees(np.arccos(np.clip(np.median(cosine), -1.0, 1.0))))


def inlier_object_shares(
    stabilizer: Any, object_index: np.ndarray, moved: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Where the most recent estimate's inliers lie in the reference frame.

    `object_index` holds, per reference pixel, the index of the object covering it or -1, and
    `moved` flags the objects that moved in this pair. Returns the share of inliers on any object
    (`inliers_on_objects`) and on objects that moved (`inliers_on_moving`); NaN without inliers.
    """
    shares = {'inliers_on_objects': float('nan'), 'inliers_on_moving': float('nan')}
    inliers = getattr(stabilizer, 'cur_inliers', None)
    points = getattr(stabilizer, 'ref_pts', None)
    if inliers is None or points is None or len(points) == 0:
        return shares
    flags = np.asarray(inliers, dtype=bool).ravel()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(flags) != len(points) or not flags.any():
        return shares
    height, width = object_index.shape
    columns = np.clip(np.floor(points[flags, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.floor(points[flags, 1]).astype(int), 0, height - 1)
    covering = object_index[rows, columns]
    shares['inliers_on_objects'] = float(np.mean(covering >= 0))
    if moved is not None:
        moved = np.asarray(moved, dtype=bool)
        on_moving = covering >= 0
        on_moving[on_moving] = moved[covering[on_moving]]
        shares['inliers_on_moving'] = float(np.mean(on_moving))
    return shares


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, dtype=np.float64).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return projected[:, :2] / projected[:, 2:]


def timing_record(stages: Dict[str, float]) -> Dict[str, float]:
    """Stage timings as `time_<stage>_s` keys, plus `time_total_s`."""
    record = {f"time_{name}_s": float(value) for name, value in stages.items()}
    record['time_total_s'] = float(sum(stages.values()))
    return record
