#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
dynamic.py - Part II: the dynamic benchmark, on real hovering-drone footage.

For each sampled frame, the homography onto the reference frame is estimated directly, never by
chaining adjacent transforms, and scored two ways:

  Reference-based     disagreement with the cached reference estimator, in reference-frame
                      pixels, plus IoU and OCE of the current frame's boxes placed by each.
  Reference-free      masked SSIM, ZNCC, mutual information and gradient similarity between the
                      reference frame and the warped current frame, over the static background.

Each measurement also records the inlier diagnostics (never ranked), the error decomposed
along the dominant motion axis, and how far the estimate follows the objects that moved together
(`capture_fraction`), with the share of its inliers on objects and on objects that moved.

Current-frame objects are placed into reference coordinates by the reference estimator, so every
candidate is scored over the same object mask. `repeats` (default 1) re-measures every pair,
for the estimators and detectors that are not bit-exact across calls (`USAC_PARALLEL`, torch
detectors on a GPU).

Also written:

  identity / reference   two baseline runs scored on every pair without a pipeline: the
                         unstabilized frame and the reference estimator's own transform.
  failed pairs           carry the photometric scores of the identity warp, with `failed` true.
  persistence            overlap of the two frames' object footprints in reference coordinates.
  moving foreground      the reference frame's objects found again in the current frame under the
                         reference estimator (`scene.object_motion`): the share of the frame that
                         moved, and of it the largest group that moved together.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from stabilo_benchmark.config import (
    apply_overrides,
    build_oracle,
    build_pipelines,
    config_hash,
    load_config,
    resolve_path,
    run_key,
    validate_config,
)
from stabilo_benchmark.masking import LabelError, Labels, require_annotations
from stabilo_benchmark.metrics.diagnostics import (
    capture_fraction,
    estimation_failed,
    inlier_object_shares,
    matching_stats,
    motion_decomposition,
)
from stabilo_benchmark.metrics.geometric import (
    corner_disagreement,
    corner_disagreement_vectors,
    nan_object_scores,
    object_center_error,
    object_iou,
    threshold_label,
)
from stabilo_benchmark.metrics.photometric import PHOTOMETRIC_METRICS, photometric_scores
from stabilo_benchmark.oracle import (
    boxes_for,
    load_frame,
    load_frame_labels,
    load_oracle,
    oracle_matrices,
    pair_key,
)
from stabilo_benchmark.pipelines import (
    DEFAULT_OBJECT_BOXES,
    PipelineSpec,
    check_resolution,
    opencv_binding_hint,
    order_by_cost,
    processed_pixels,
    unavailable_detectors,
)
from stabilo_benchmark.regions import (
    RegionError,
    evaluation_mask,
    load_exclusion,
    load_roi,
    metric_region,
    motion_axis,
    rasterize_quads,
    region_is_usable,
    warp_points,
)
from stabilo_benchmark.resources import describe_resource_error, is_resource_error, peak_memory, reset_peak_memory
from stabilo_benchmark.results import (
    RUN_COMPLETED,
    RUN_INFEASIBLE,
    ResultStore,
    build_manifest,
    describe_path,
    seed_everything,
)
from stabilo_benchmark.runners.prepare import load_case
from stabilo_benchmark.sampling import FramePair, sample_pairs, summarize
from stabilo_benchmark.scene import MIN_DISPLACED_OBJECTS, ObjectMotion, object_motion
from stabilo_benchmark.scene import coverage as scene_coverage
from stabilo_benchmark.seeding import seed_estimator

# Threads scoring a run's pairs after its registrations; NumPy and OpenCV release the GIL on
# whole-frame operations.
SCORING_WORKERS = max(1, min(8, (os.cpu_count() or 1) // 2))

# Below this axis concentration the motion decomposition is not reported.
MIN_AXIS_CONCENTRATION = 0.3

# Per-pair moving-foreground fields (`motion_fields`).
MOTION_FIELDS = ('moving_share', 'coherent_share', 'n_moving', 'n_coherent', 'coherent_shift_px')

# Baseline runs scored without a pipeline.
BASELINE_IDENTITY = 'identity'
BASELINE_REFERENCE = 'reference'
BASELINES = (BASELINE_IDENTITY, BASELINE_REFERENCE)


class DynamicBenchmarkError(RuntimeError):
    """Raised when the dynamic benchmark cannot run: no case, no reference estimates."""


def plan_pairs(case: Mapping[str, Any], config: Mapping[str, Any]) -> List[FramePair]:
    """
    The frame pairs to evaluate, restricted to frames the prepared case contains.

    The reference frame is the case's (CASE.json), never re-resolved from the config.
    """
    available = [int(index) for index in case['frames']]
    pairs = sample_pairs(
        int(case['video']['frame_count']),
        config.get('sampling', {}) or {},
        reference=int(case['reference_index']),
        seed=int(config['seed']),
        available=available,
    )
    if not pairs:
        raise DynamicBenchmarkError(
            f"The sampling plan selected no frame pairs from the {len(available)} frame(s) in the case. "
            f"Re-prepare the case with this config, or widen the sampling."
        )
    return pairs


def run_pipeline(
    spec: PipelineSpec,
    case_dir: Path,
    pairs: Sequence[FramePair],
    reference_homographies: Mapping[str, Optional[np.ndarray]],
    config: Mapping[str, Any],
    roi: Optional[np.ndarray] = None,
    verbosity: int = 0,
    object_motions: Optional[Mapping[str, Optional[ObjectMotion]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one pipeline over every frame pair, or report that this machine cannot.

    `object_motions` (`pair_object_motion`) is computed here when not given. A resource error aborts
    the run and discards its partial records (see `resources.py`).
    """
    try:
        return _evaluate_pipeline(spec, case_dir, pairs, reference_homographies, config, roi, verbosity, object_motions)
    except Exception as error:  # noqa: BLE001 - re-raised unless the machine is what ran out
        if not is_resource_error(error):
            raise
        if verbosity > 0:
            tqdm.write(
                f"\033[93mInfeasible: '{spec.name}' exhausted memory "
                f"({type(error).__name__}: {error}). Reported as N/A.\033[0m"
            )
        return {
            'params': spec.to_dict(),
            'records': [],
            'n_pairs': len(pairs),
            'repeats': max(1, int(config.get('repeats', 1))),
            'status': RUN_INFEASIBLE,
            'resource_error': describe_resource_error(error),
        }


def _evaluate_pipeline(
    spec: PipelineSpec,
    case_dir: Path,
    pairs: Sequence[FramePair],
    reference_homographies: Mapping[str, Optional[np.ndarray]],
    config: Mapping[str, Any],
    roi: Optional[np.ndarray] = None,
    verbosity: int = 0,
    object_motions: Optional[Mapping[str, Optional[ObjectMotion]]] = None,
) -> Dict[str, Any]:
    """Evaluate one pipeline over every frame pair of a prepared case."""
    metrics_config = config.get('metrics', {}) or {}
    region_config = metrics_config.get('region', {}) or {}
    thresholds = metrics_config.get('hea_thresholds', [1.0, 3.0, 5.0])
    photometric_names = metrics_config.get('photometric')
    object_boxes = metrics_config.get('object_boxes', DEFAULT_OBJECT_BOXES)
    dilation = int(region_config.get('mask_dilation_px', 8))
    border = int(region_config.get('border_px', 0))
    minimum_area = float(region_config.get('min_area_fraction', 0.05))
    # Nine floats per pair; `audit --results` needs them to score candidates against the GCPs.
    store_homographies = bool(config.get('store_homographies', True))

    reference_index = pairs[0].reference_index
    reference_frame = load_frame(case_dir, reference_index)
    height, width = reference_frame.shape[:2]

    work = processed_pixels(spec, width, height)
    warning = check_resolution(spec, width, height)
    if warning and verbosity > 0:
        tqdm.write(f"\033[93m{warning}\033[0m")

    reference_labels = load_frame_labels(case_dir, reference_index, width, height)
    reference_corners = reference_labels.corners()
    object_index = object_index_map(reference_labels, (height, width))
    if object_motions is None:
        object_motions = pair_object_motion(case_dir, pairs, reference_homographies)

    stabilizer = spec.build(
        min_good_match_count_warning=20 if verbosity > 2 else -1,
        min_inliers_match_count_warning=10 if verbosity > 2 else -1,
    )
    box_format = spec.box_format or 'xywh'
    reference_boxes = boxes_for(reference_labels, spec.masking)
    reference_start = time.perf_counter()
    stabilizer.set_ref_frame(reference_frame, reference_boxes, box_format=box_format)
    reference_time = time.perf_counter() - reference_start
    reference_suppressed = suppressed_fraction(stabilizer, reference_boxes, box_format)

    seed = int(config.get('seed', 0))
    repeats = max(1, int(config.get('repeats', 1)))
    case_name = case_dir.name
    cell = _case_cell(case_dir)

    records: List[Dict[str, Any]] = []
    # Registration runs alone, so its timings are not shared with scoring; every pair is scored
    # afterwards: (record, current frame, transform scored, reference transform, labels), and the
    # keys each record takes after its scores.
    scoring: List[tuple] = []
    followups: List[Dict[str, Any]] = []
    schedule = [(repeat, pair) for repeat in range(repeats) for pair in pairs]
    for repeat, pair in tqdm(schedule, desc=spec.name, unit='pair', leave=False, disable=verbosity > 2):
        key = pair_key(pair)
        frame = load_frame(case_dir, pair.current_index)
        labels = load_frame_labels(case_dir, pair.current_index, width, height)

        # Seeded per measurement, as in Part I.
        seed_estimator(seed, case_name, key, repeat)

        current_boxes = boxes_for(labels, spec.masking)
        start = time.perf_counter()
        stabilizer.stabilize(frame, current_boxes, box_format=box_format)
        elapsed = time.perf_counter() - start

        estimate = stabilizer.get_cur_trans_matrix()
        failed = estimation_failed(stabilizer, estimate)
        record: Dict[str, Any] = {
            **pair.to_dict(),
            'pair': key,
            'repeat': repeat,
            'failed': failed,
            'object_boxes': object_boxes,
            'time_stabilize_s': elapsed,
            'time_reference_s': reference_time,
            'processed_pixels': work,
            'n_objects_ref': len(reference_labels),
            'n_objects_cur': len(labels),
            # Object area and coverage of the current frame, recorded per measurement.
            'coverage_ref': scene_coverage(reference_labels, width, height),
            'coverage_cur': scene_coverage(labels, width, height),
            # Share of each frame the feature mask withholds from detection (margin included).
            'suppressed_ref': reference_suppressed,
            'suppressed_cur': suppressed_fraction(stabilizer, current_boxes, box_format),
            'cell': cell,
            **matching_stats(stabilizer),
        }

        reference_matrix = reference_homographies.get(key)
        record['oracle_available'] = reference_matrix is not None
        record['persistence'] = persistence(reference_labels, labels, reference_matrix, (height, width))
        motion = object_motions.get(key)
        record.update(motion_fields(motion))
        record.update(inlier_object_shares(stabilizer, object_index, None if motion is None else motion.moved))

        if failed:
            record.update(_nan_record(thresholds, photometric_names))
            # A failure is scored as the identity warp.
            scoring.append((record, pair.current_index, np.eye(3), reference_matrix, labels))
            followups.append({'photometric_from_identity': True})
            continue

        estimate = np.asarray(estimate, dtype=np.float64)
        if estimate.shape == (2, 3):
            estimate = np.vstack([estimate, [0.0, 0.0, 1.0]])
        if store_homographies:
            record['homography'] = estimate.tolist()

        record.update(
            _reference_based(estimate, reference_matrix, labels.object_corners(object_boxes), width, height, thresholds)
        )
        record['capture_fraction'] = group_capture(estimate, reference_matrix, motion)
        scoring.append((record, pair.current_index, estimate, reference_matrix, labels))
        followups.append(_motion_diagnostics(estimate, reference_matrix, reference_labels, width, height))

    def score(job: tuple) -> Dict[str, Any]:
        _, index, transform, reference_matrix, labels = job
        return _reference_free(
            reference_frame,
            load_frame(case_dir, index),
            transform,
            reference_matrix,
            reference_corners,
            labels,
            (height, width),
            dilation,
            border,
            minimum_area,
            roi,
            photometric_names,
        )

    # Each pair is scored on its own, so the scores depend neither on the order nor on the threads.
    with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as pool:
        for job, scores, followup in zip(scoring, pool.map(score, scoring), followups, strict=True):
            record = job[0]
            record.update(scores)
            record.update(followup)
            records.append(record)

    return {
        'params': spec.to_dict(),
        'records': records,
        'n_pairs': len(pairs),
        'repeats': repeats,
        'reference_index': reference_index,
        'frame_size': {'width': width, 'height': height},
        'status': RUN_COMPLETED,
    }


def _reference_based(
    estimate: np.ndarray,
    reference_matrix: Optional[np.ndarray],
    object_corners: np.ndarray,
    width: int,
    height: int,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """
    Disagreement with the reference estimator: corner error, box IoU and OCE.

    `object_corners` are the current frame's boxes in the geometry `metrics.object_boxes` names.
    """
    if reference_matrix is None:
        return {
            'cte_mean': float('nan'),
            'cte_max': float('nan'),
            'cte_rms': float('nan'),
            **{f"within_{threshold_label(t)}": None for t in thresholds},
        }

    scores: Dict[str, Any] = corner_disagreement(reference_matrix, estimate, width, height)
    for threshold in thresholds:
        scores[f"within_{threshold_label(threshold)}"] = bool(scores['cte_mean'] <= float(threshold))

    # Carry the current frame's boxes by both transforms.
    if object_corners.size:
        by_reference = warp_points(object_corners, reference_matrix)
        by_estimate = warp_points(object_corners, estimate)
        scores.update(object_iou(by_reference, by_estimate))
        scores.update(object_center_error(by_reference, by_estimate))
    return scores


def _reference_free(
    reference_frame: np.ndarray,
    frame: np.ndarray,
    estimate: np.ndarray,
    reference_matrix: Optional[np.ndarray],
    reference_corners: np.ndarray,
    labels: Labels,
    shape: tuple,
    dilation: int,
    border: int,
    minimum_area: float,
    roi: Optional[np.ndarray],
    photometric_names: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Masked background-alignment metrics for one warped frame."""
    height, width = shape

    region, stats = metric_region(
        shape=shape,
        homography=estimate,
        reference_corners=reference_corners if reference_corners.size else None,
        current_corners=labels.corners() if len(labels) else None,
        # The reference transform, shared by every candidate; the candidate's own only without one.
        mask_transform=reference_matrix if reference_matrix is not None else estimate,
        dilation_px=dilation,
        border_px=border,
        roi=roi,
    )

    record: Dict[str, Any] = dict(stats.to_dict())
    if not region_is_usable(stats, minimum_area):
        # An unusable region is scored as the identity warp.
        record['region_usable'] = False
        if _photometric_names(photometric_names) and not np.allclose(estimate, np.eye(3)):
            fallback = _reference_free(
                reference_frame,
                frame,
                np.eye(3),
                reference_matrix,
                reference_corners,
                labels,
                shape,
                dilation,
                border,
                minimum_area,
                roi,
                photometric_names,
            )
            record.update({name: fallback.get(name, float('nan')) for name in _photometric_names(photometric_names)})
            record['photometric_from_identity'] = True
        else:
            record.update({name: float('nan') for name in _photometric_names(photometric_names)})
        return record

    record['region_usable'] = True
    names = _photometric_names(photometric_names)
    if names:
        warped = cv2.warpPerspective(frame, estimate, (width, height))
        record.update(photometric_scores(reference_frame, warped, region, metrics=names))
    return record


def suppressed_fraction(stabilizer: Any, boxes: Optional[np.ndarray], box_format: str) -> float:
    """
    Share of the frame the feature mask excludes, rebuilt with the stabilizer's own rasterization
    (`create_binary_mask`, after `set_ref_frame` has set the frame size); 0 without boxes.
    """
    if boxes is None or len(boxes) == 0:
        return 0.0
    return float(np.mean(stabilizer.create_binary_mask(boxes, box_format) == 0))


def _photometric_names(names: Optional[Sequence[str]]) -> List[str]:
    """The configured photometric metrics: every one when unset, none for an empty list."""
    return list(PHOTOMETRIC_METRICS) if names is None else list(names)


def persistence(
    reference_labels: Labels,
    labels: Labels,
    reference_matrix: Optional[np.ndarray],
    shape: tuple,
) -> float:
    """
    Overlap of the two frames' object footprints in reference coordinates.

    The current frame's boxes are placed by the reference estimator. NaN when the reference failed on
    this pair or neither frame has an object.
    """
    if reference_matrix is None or (len(reference_labels) == 0 and len(labels) == 0):
        return float('nan')
    first = rasterize_quads(reference_labels.corners(), shape) if len(reference_labels) else np.zeros(shape, dtype=bool)
    if len(labels):
        placed = warp_points(labels.corners(), reference_matrix)
        second = rasterize_quads(placed, shape)
    else:
        second = np.zeros(shape, dtype=bool)
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return float('nan')
    return float(np.count_nonzero(first & second) / union)


def pair_object_motion(
    case_dir: Path,
    pairs: Sequence[FramePair],
    reference_homographies: Mapping[str, Optional[np.ndarray]],
) -> Dict[str, Optional[ObjectMotion]]:
    """
    Per pair, the reference frame's objects found again in the current frame, with the reference
    estimator's transform as the platform motion (`scene.object_motion`); None where the reference
    estimator failed. One computation serves every run of a case.
    """
    reference_index = pairs[0].reference_index
    reference_frame = load_frame(case_dir, reference_index)
    height, width = reference_frame.shape[:2]
    reference_labels = load_frame_labels(case_dir, reference_index, width, height)
    motions: Dict[str, Optional[ObjectMotion]] = {}
    for pair in pairs:
        key = pair_key(pair)
        matrix = reference_homographies.get(key)
        if key in motions or matrix is None:
            motions.setdefault(key, None)
            continue
        labels = load_frame_labels(case_dir, pair.current_index, width, height)
        frame = load_frame(case_dir, pair.current_index)
        motions[key] = object_motion(
            reference_frame, frame, reference_labels, labels, motion=np.linalg.inv(np.asarray(matrix, np.float64))
        )
    return motions


def motion_fields(motion: Optional[ObjectMotion]) -> Dict[str, float]:
    """
    The pair's moving foreground under the reference estimator: the share of the frame its moved
    objects cover (`moving_share`) and its largest group that moved together (`coherent_share`), their
    counts, and the group's displacement; NaN where the reference estimator failed.
    """
    if motion is None:
        return dict.fromkeys(MOTION_FIELDS, float('nan'))
    share = motion.areas / motion.frame_area
    return {
        'moving_share': float(share[motion.moved].sum()),
        'coherent_share': float(share[motion.group].sum()),
        'n_moving': int(motion.moved.sum()),
        'n_coherent': int(motion.group.sum()),
        'coherent_shift_px': float(np.hypot(*motion.shift)),
    }


def group_capture(
    estimate: Optional[np.ndarray], reference_matrix: Optional[np.ndarray], motion: Optional[ObjectMotion]
) -> float:
    """`diagnostics.capture_fraction` over the pair's group, defined from MIN_DISPLACED_OBJECTS objects."""
    if motion is None or int(motion.group.sum()) < MIN_DISPLACED_OBJECTS:
        return float('nan')
    group = motion.group
    return capture_fraction(estimate, reference_matrix, motion.current_centers[group], motion.displacement[group])


def object_index_map(labels: Labels, shape: tuple) -> np.ndarray:
    """Per pixel, the index of the object covering it (the later one where boxes overlap), or -1."""
    index = np.full(shape, -1, dtype=np.int32)
    for number, quad in enumerate(labels.corners() if len(labels) else []):
        cv2.fillPoly(index, [np.round(np.asarray(quad, dtype=np.float64)).astype(np.int32)], int(number))
    return index


def _case_cell(case_dir: Path) -> Optional[str]:
    """The survey cell recorded in CASE.json by `prepare --cell`, or None."""
    try:
        value = load_case(case_dir).get('cell')
    except Exception:  # noqa: BLE001 - a case without a manifest is caught by the runner itself
        return None
    return str(value) if value else None


def baseline_params(kind: str) -> Dict[str, Any]:
    """The `params` block of a baseline run, shaped like a pipeline's so the tables read it."""
    if kind not in BASELINES:
        raise DynamicBenchmarkError(f"Unknown baseline '{kind}'; choose from {BASELINES}")
    return {'pipeline': kind, 'masking': 'none', 'detector_name': kind, 'matcher_name': kind, 'baseline': True}


NO_EVALUATION_MASK: Dict[str, Any] = {'ground_plane_roi': None, 'exclusion': None}


def load_evaluation_mask(
    config: Mapping[str, Any], case_dir: Path, shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    The evaluation mask of a case and its record for the manifest.

    `metrics.region.ground_plane_roi` is a path resolved against the config; `metrics.region.exclusion`
    names a file inside the case directory (polygons over above-ground structure on the reference
    frame). A configured file that is missing raises.
    """
    region = (config.get('metrics', {}) or {}).get('region', {}) or {}
    record: Dict[str, Any] = dict(NO_EVALUATION_MASK)
    roi = exclusion = None
    if region.get('ground_plane_roi'):
        path = resolve_path(config, region['ground_plane_roi'])
        roi = load_roi(path, shape)
        record['ground_plane_roi'] = {'file': describe_path(path), 'sha256': _sha256(path)}
    if region.get('exclusion'):
        path = Path(case_dir) / region['exclusion']
        exclusion = load_exclusion(path, shape)
        record['exclusion'] = {
            'file': str(region['exclusion']),
            'sha256': _sha256(path),
            'excluded_fraction': round(float(exclusion.mean()), 6),
        }
    return evaluation_mask(shape, roi, exclusion), record


def check_evaluation_mask(stored: Mapping[str, Any], record: Mapping[str, Any], results_dir: Path) -> None:
    """Raise when a resumed campaign's stored evaluation mask differs from the current one."""
    previous = stored.get('evaluation_mask', NO_EVALUATION_MASK)
    if previous != record:
        raise RegionError(
            f"Cannot resume {results_dir}: the evaluation mask differs from the stored one ({previous} vs "
            f"{dict(record)}). Restore the original files, or start over with --overwrite."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_baseline(
    kind: str,
    case_dir: Path,
    pairs: Sequence[FramePair],
    reference_homographies: Mapping[str, Optional[np.ndarray]],
    config: Mapping[str, Any],
    roi: Optional[np.ndarray] = None,
    object_motions: Optional[Mapping[str, Optional[ObjectMotion]]] = None,
) -> Dict[str, Any]:
    """
    Score the identity or the reference estimator's own transform on every pair.

    Pairs the reference failed on are recorded as failed for the reference baseline.
    """
    metrics_config = config.get('metrics', {}) or {}
    region_config = metrics_config.get('region', {}) or {}
    thresholds = metrics_config.get('hea_thresholds', [1.0, 3.0, 5.0])
    photometric_names = metrics_config.get('photometric')
    object_boxes = metrics_config.get('object_boxes', DEFAULT_OBJECT_BOXES)
    dilation = int(region_config.get('mask_dilation_px', 8))
    border = int(region_config.get('border_px', 0))
    minimum_area = float(region_config.get('min_area_fraction', 0.05))

    reference_index = pairs[0].reference_index
    reference_frame = load_frame(case_dir, reference_index)
    height, width = reference_frame.shape[:2]
    reference_labels = load_frame_labels(case_dir, reference_index, width, height)
    reference_corners = reference_labels.corners()
    cell = _case_cell(case_dir)
    if object_motions is None:
        object_motions = pair_object_motion(case_dir, pairs, reference_homographies)

    records: List[Dict[str, Any]] = []
    for pair in tqdm(pairs, desc=kind, unit='pair', leave=False):
        key = pair_key(pair)
        frame = load_frame(case_dir, pair.current_index)
        labels = load_frame_labels(case_dir, pair.current_index, width, height)
        reference_matrix = reference_homographies.get(key)
        estimate = np.eye(3) if kind == BASELINE_IDENTITY else reference_matrix
        record: Dict[str, Any] = {
            **pair.to_dict(),
            'pair': key,
            'repeat': 0,
            'failed': estimate is None,
            'object_boxes': object_boxes,
            'time_stabilize_s': 0.0,
            'time_reference_s': 0.0,
            'n_objects_ref': len(reference_labels),
            'n_objects_cur': len(labels),
            'coverage_ref': scene_coverage(reference_labels, width, height),
            'coverage_cur': scene_coverage(labels, width, height),
            'cell': cell,
            'oracle_available': reference_matrix is not None,
            'persistence': persistence(reference_labels, labels, reference_matrix, (height, width)),
            **motion_fields(object_motions.get(key)),
        }
        if estimate is None:
            record.update(_nan_record(thresholds, photometric_names))
            records.append(record)
            continue
        estimate = np.asarray(estimate, dtype=np.float64)
        record.update(
            _reference_based(estimate, reference_matrix, labels.object_corners(object_boxes), width, height, thresholds)
        )
        record['capture_fraction'] = group_capture(estimate, reference_matrix, object_motions.get(key))
        record.update(
            _reference_free(
                reference_frame,
                frame,
                estimate,
                reference_matrix,
                reference_corners,
                labels,
                (height, width),
                dilation,
                border,
                minimum_area,
                roi,
                photometric_names,
            )
        )
        record.update(_motion_diagnostics(estimate, reference_matrix, reference_labels, width, height))
        records.append(record)

    return {
        'params': baseline_params(kind),
        'records': records,
        'n_pairs': len(pairs),
        'repeats': 1,
        'reference_index': reference_index,
        'frame_size': {'width': width, 'height': height},
        'status': RUN_COMPLETED,
    }


def _motion_diagnostics(
    estimate: np.ndarray,
    reference_matrix: Optional[np.ndarray],
    reference_labels: Labels,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """
    Decompose the registration error relative to the dominant motion axis.

    The axis needs oriented boxes; with HBB-only labels the decomposition is NaN.
    """
    if reference_matrix is None or not reference_labels.has_obb or len(reference_labels) == 0:
        return {'motion_axis_concentration': float('nan')}

    axis, concentration = motion_axis(reference_labels.corners())
    record: Dict[str, Any] = {'motion_axis_concentration': concentration}
    if axis is None or concentration < MIN_AXIS_CONCENTRATION:
        # Axis too diffuse to report a direction.
        record.update(motion_decomposition(np.zeros((0, 2)), None))
        return record

    record.update(motion_decomposition(corner_disagreement_vectors(reference_matrix, estimate, width, height), axis))
    record['motion_axis_deg'] = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return record


def _nan_record(thresholds: Sequence[float], photometric_names: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Metric values for a pair where the pipeline produced no transform."""
    record: Dict[str, Any] = {
        'cte_mean': float('nan'),
        'cte_max': float('nan'),
        'cte_rms': float('nan'),
        **nan_object_scores(),
        'motion_axis_concentration': float('nan'),
        'capture_fraction': float('nan'),
        'region_usable': False,
    }
    for threshold in thresholds:
        record[f"within_{threshold_label(threshold)}"] = False
    for name in _photometric_names(photometric_names):
        record[name] = float('nan')
    return record


# --------------------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------------------


def run_dynamic(case_dir: Path, config_path: Path, args: argparse.Namespace) -> Path:
    """Execute the dynamic benchmark for one prepared case and one config."""
    case_dir = Path(case_dir).expanduser().absolute()
    case = load_case(case_dir)

    config = load_config(config_path)
    if config.get('kind') != 'dynamic':
        raise DynamicBenchmarkError(f"{config_path} declares kind '{config.get('kind')}', expected 'dynamic'")
    apply_overrides(config, getattr(args, 'set', None))
    if getattr(args, 'name', None):
        config['name'] = str(args.name)
    validate_config(config)

    # Results go to <case>/results/<name> unless --results-dir names another root.
    results_override = getattr(args, 'results_dir', None)
    results_root = Path(results_override) if results_override else case_dir / 'results'
    results_dir = results_root / str(config['name'])
    store = ResultStore(
        results_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        regrid=getattr(args, 'regrid', False),
        run_key=run_key,
    )
    if store.is_complete() and not (args.overwrite or store.resume):
        print(f"\033[33mSkipping '{config['name']}': results already exist. Use --overwrite or --resume.\033[0m")
        return results_dir

    pairs = plan_pairs(case, config)
    reference_payload = load_oracle(case_dir)
    reference_homographies = oracle_matrices(reference_payload)

    missing = [pair_key(pair) for pair in pairs if pair_key(pair) not in reference_homographies]
    if missing:
        raise DynamicBenchmarkError(
            f"{len(missing)} sampled pair(s) have no reference estimate (first: {missing[0]}). "
            f"Recompute with: stabilo-benchmark oracle {case_dir} --overwrite"
        )

    specs = order_by_cost(build_pipelines(config))
    # Preflight: stop before the manifest is written if a detector cannot be constructed.
    missing = unavailable_detectors(specs)
    if missing:
        raise DynamicBenchmarkError(
            f"{len(missing)} pipeline(s) name a detector that is unavailable here: "
            f"{', '.join(missing)}.\n{opencv_binding_hint()}"
        )

    try:
        annotated = require_annotations(case_dir, [spec.masking for spec in specs])
    except LabelError as error:
        raise DynamicBenchmarkError(str(error)) from error

    # Warn when the config's oracle block differs from the cached reference estimator.
    cached_oracle = reference_payload.get('pipeline', {}) or {}
    wanted_oracle = build_oracle(config).to_dict()
    if _oracle_params(cached_oracle) != _oracle_params(wanted_oracle):
        print(
            f"\033[93mNote: the cached reference estimator differs from the one this config names.\n"
            f"  cached: {_oracle_summary(cached_oracle)}\n"
            f"  config: {_oracle_summary(wanted_oracle)}\n"
            f"  The cache is used as-is. To change it: "
            f"stabilo-benchmark oracle {case_dir} -c {config_path} --overwrite\033[0m"
        )

    roi, mask_record = load_evaluation_mask(config, case_dir, load_frame(case_dir, pairs[0].reference_index).shape[:2])
    object_motions = pair_object_motion(case_dir, pairs, reference_homographies)

    grid: Dict[int, Dict[str, Any]] = {index: spec.to_dict() for index, spec in enumerate(specs, start=1)}
    for offset, kind in enumerate(BASELINES, start=len(specs) + 1):
        grid[offset] = baseline_params(kind)
    manifest = build_manifest(config, config_hash(config), grid)
    manifest.metadata['seeds'] = seed_everything(int(config['seed']))
    manifest.metadata['case'] = {
        'dir': describe_path(case_dir),
        'video': case['video']['path'],
        'annotated': annotated,
    }
    manifest.metadata['sampling'] = summarize(pairs, case['video'].get('fps'))
    manifest.metadata['reference_estimator'] = {
        'pipeline': reference_payload.get('pipeline'),
        'n_failed': reference_payload.get('n_failed'),
        'caveat': reference_payload.get('caveat'),
    }
    manifest.metadata['evaluation_mask'] = mask_record
    check_evaluation_mask(store.initialize(manifest).metadata, mask_record, results_dir)

    repeats = max(1, int(config.get('repeats', 1)))
    repeat_note = f" x {repeats} repeat(s)" if repeats > 1 else ''
    print(
        f"\033[92mDynamic benchmark '{config['name']}': {len(specs)} pipeline(s) x {len(pairs)} frame pair(s)"
        f"{repeat_note} = {len(specs) * len(pairs) * repeats} registrations, plus the identity and reference "
        f"baselines\033[0m"
    )
    print(f"\033[92mResults -> {results_dir}\033[0m")
    if not annotated:
        print(
            f"\033[93m{case_dir} is not annotated: masking 'none' only, no IoU, OCE or motion axis, and the "
            f"evaluation region does not exclude objects.\033[0m"
        )

    completed = 0
    empty: List[str] = []
    infeasible: List[str] = []
    for run_number, spec in enumerate(tqdm(specs, desc='Pipeline', unit='run'), start=1):
        if store.resume and store.has_run(run_number):
            # A stored run still counts toward the marker's infeasible and empty lists.
            result = store.read_run(run_number)
        else:
            # Reset per run so each run reports its own peak (see `resources.py`).
            reset_peak_memory()
            try:
                result = run_pipeline(
                    spec,
                    case_dir,
                    pairs,
                    reference_homographies,
                    config,
                    roi=roi,
                    verbosity=args.verbosity,
                    object_motions=object_motions,
                )
            except Exception as error:  # noqa: BLE001 - recorded, then re-raised for the caller
                store.record_failure(run_number, spec.to_dict(), len(specs), error)
                store.finalize(completed, len(specs))
                raise
            result['run_number'] = run_number
            result['peak_memory'] = peak_memory()
            store.write_run(run_number, result)
            if args.verbosity > 0:
                _print_run_summary(spec, result)
        completed += 1
        if result.get('status') == RUN_INFEASIBLE:
            infeasible.append(spec.name)
        elif not result.get('records'):
            empty.append(spec.name)

    for run_number, kind in enumerate(BASELINES, start=len(specs) + 1):
        if store.resume and store.has_run(run_number):
            result = store.read_run(run_number)
        else:
            result = evaluate_baseline(
                kind, case_dir, pairs, reference_homographies, config, roi=roi, object_motions=object_motions
            )
            result['run_number'] = run_number
            store.write_run(run_number, result)
        completed += 1
        if not result.get('records'):
            empty.append(kind)

    store.finalize(completed, len(grid), empty, infeasible)
    return results_dir


def cached_oracle_matches(case_dir: Path, config: Mapping[str, Any]) -> bool:
    """Whether the case's cached reference estimates were computed by the oracle `config` names."""
    cached = load_oracle(Path(case_dir)).get('pipeline', {}) or {}
    return _oracle_params(cached) == _oracle_params(build_oracle(config).to_dict())


def _oracle_params(description: Mapping[str, Any]) -> Dict[str, Any]:
    """The comparable part of an oracle description: its name label is cosmetic."""
    return {k: v for k, v in description.items() if k != 'pipeline'}


def _oracle_summary(description: Mapping[str, Any]) -> str:
    """Short human-readable description of an oracle pipeline, for the drift warning."""
    keys = ('detector_name', 'matcher_name', 'ransac_method', 'downsample_ratio', 'max_features', 'masking')
    return ', '.join(f"{key}={description.get(key)}" for key in keys if key in description)


def _print_run_summary(spec: PipelineSpec, result: Mapping[str, Any]) -> None:
    """One-line summary of a finished run."""
    records = result.get('records', [])
    if result.get('status') == RUN_INFEASIBLE:
        print(f"  {spec.name:<32} infeasible on this machine (out of memory)")
        return
    if not records:
        print(f"  {spec.name:<32} no valid pairs")
        return

    def median(name: str) -> float:
        values = np.array([r.get(name, np.nan) for r in records], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float('nan')

    failures = float(np.mean([bool(r.get('failed')) for r in records]))
    ssim = f" SSIM {median('ssim'):6.3f} |" if any('ssim' in r for r in records) else ''
    print(
        f"  {spec.name:<32} median CTE {median('cte_mean'):7.2f} px |{ssim} "
        f"align {median('motion_alignment'):5.2f} | inlier ratio {median('inlier_ratio'):5.2f} | "
        f"failures {failures:5.1%}"
    )
