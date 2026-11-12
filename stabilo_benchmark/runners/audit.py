#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
audit.py - Bound the reference estimator's error with ground control points.

Ground-plane points annotated in each audited frame are mapped into the reference frame by the
cached homography; the distance to their annotated counterparts is the residual.

- The reference estimator's residual, pooled and per temporal-offset bin. Its 95th percentile
  is Part II's equivalence margin.
- With `--results`, every candidate is scored against the same points: the residual of its own
  homography, and its corner transfer error, IoU and OCE against the homography fitted to the
  points of each audited frame (`gcp.fit_homography`), which depends on no pipeline. The scores
  are written into that results directory (`GCP_SCORES.json`), so each campaign of a case keeps
  its own; the case's AUDIT.json holds the reference estimator alone.
- The click repeatability, from a second session in `gcp_repeat/` (`annotate-gcp --repeat`).
- How the annotated points relate to the proposals recorded with them (`gcp.proposal_review`):
  accepted, corrected, or placed by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stabilo_benchmark.gcp import (
    GCP_DIR,
    GCP_REPEAT_DIR,
    GroundControlPoints,
    annotation_repeatability,
    fit_homography,
    load_gcp_set,
    proposal_review,
    residuals,
    summarize_residuals,
)
from stabilo_benchmark.metrics.geometric import corner_disagreement, object_center_error, object_iou
from stabilo_benchmark.oracle import load_frame_labels, load_oracle, oracle_matrices
from stabilo_benchmark.regions import warp_points
from stabilo_benchmark.results import GCP_SCORES_NAME, ResultStore, system_metadata
from stabilo_benchmark.runners.dynamic import BASELINE_IDENTITY, BASELINE_REFERENCE
from stabilo_benchmark.runners.prepare import load_case
from stabilo_benchmark.sampling import DEFAULT_OFFSET_BINS, bin_label, offset_bin_label, plan_bins

AUDIT_FILE = 'AUDIT.json'


class AuditError(RuntimeError):
    """Raised when the audit cannot proceed: no points, no reference frame annotation."""


def _pair_key(reference_index: int, current_index: int) -> str:
    return f"{reference_index:06d}->{current_index:06d}"


def audit_homographies(
    points: Mapping[int, GroundControlPoints],
    reference_index: int,
    homographies: Mapping[str, Optional[np.ndarray]],
    label: str,
    bins: Sequence[Tuple[int, Optional[int]]] = DEFAULT_OFFSET_BINS,
) -> Dict[str, Any]:
    """
    Per-frame and pooled GCP residuals for one set of homographies, and the residual per offset bin.

    Annotated frames without a homography are reported as skipped.
    """
    if reference_index not in points:
        raise AuditError(
            f"The reference frame {reference_index} has no ground control points. Annotate it first: "
            f"every residual is measured against the reference frame's own points."
        )

    reference_points = points[reference_index]
    per_frame: List[Dict[str, Any]] = []
    skipped: List[int] = []

    for frame_index, current_points in sorted(points.items()):
        if frame_index == reference_index:
            continue
        matrix = homographies.get(_pair_key(reference_index, frame_index))
        if matrix is None:
            skipped.append(frame_index)
            continue
        record = residuals(reference_points, current_points, matrix)
        record['frame_index'] = frame_index
        record['offset'] = frame_index - reference_index
        per_frame.append(record)

    if not per_frame:
        raise AuditError(
            f"No audited frame had both ground control points and a '{label}' homography. "
            f"Annotated frames: {sorted(points)}; skipped for want of a homography: {skipped}."
        )

    return {
        'label': label,
        'reference_index': reference_index,
        'n_reference_points': len(reference_points),
        'per_frame': per_frame,
        'skipped_frames': skipped,
        'summary': summarize_residuals(per_frame),
        'summary_by_offset': summarize_by_offset(per_frame, bins),
    }


def summarize_by_offset(
    per_frame: Sequence[Mapping[str, Any]], bins: Sequence[Tuple[int, Optional[int]]] = DEFAULT_OFFSET_BINS
) -> Dict[str, Any]:
    """The pooled residual summary per temporal-offset bin, in bin order; empty bins are omitted."""
    out: Dict[str, Any] = {}
    for low, high in bins:
        label = bin_label(low, high)
        frames = [f for f in per_frame if offset_bin_label(int(f.get('offset', 0)), bins) == label]
        if frames:
            out[label] = summarize_residuals(frames)
    return out


def fitted_transforms(points: Mapping[int, GroundControlPoints], reference_index: int) -> Dict[int, Dict[str, Any]]:
    """The GCP-fitted homography of every audited frame, keyed by frame index."""
    reference = points[reference_index]
    return {
        index: fit_homography(reference, current)
        for index, current in sorted(points.items())
        if index != reference_index
    }


def score_against_gcp_transform(
    homographies: Mapping[str, Optional[np.ndarray]],
    fitted: Mapping[int, Mapping[str, Any]],
    reference_index: int,
    case_dir: Optional[Path],
    frame_size: Optional[Mapping[str, int]],
    object_boxes: str = 'obb',
) -> Dict[str, Any]:
    """
    One set of homographies against the GCP-fitted transform of every audited frame.

    IoU and OCE need the frame's object boxes, read from the case when given; otherwise NaN.
    """
    width = int(frame_size['width']) if frame_size else None
    height = int(frame_size['height']) if frame_size else None
    per_frame: List[Dict[str, Any]] = []
    for index, fit in sorted(fitted.items()):
        matrix = homographies.get(_pair_key(reference_index, index))
        if matrix is None or fit.get('matrix') is None or width is None:
            continue
        record: Dict[str, Any] = {
            'frame_index': index,
            'offset': index - reference_index,
            'n_points': int(fit['n_points']),
            'fit_rms_px': float(fit['fit_rms_px']),
            'well_determined': bool(fit['well_determined']),
            **corner_disagreement(fit['matrix'], matrix, width, height),
        }
        if case_dir is not None:
            try:
                labels = load_frame_labels(case_dir, index, width, height)
                corners = labels.object_corners(object_boxes)
            except Exception:  # noqa: BLE001 - a frame without labels still has a corner error
                corners = np.zeros((0, 4, 2))
            if corners.size:
                by_truth = warp_points(corners, fit['matrix'])
                by_candidate = warp_points(corners, matrix)
                record.update(object_iou(by_truth, by_candidate))
                record.update(object_center_error(by_truth, by_candidate))
        per_frame.append(record)

    def column(name: str) -> np.ndarray:
        values = np.array([f.get(name, np.nan) for f in per_frame], dtype=float)
        return values[np.isfinite(values)]

    cte = column('cte_mean')
    summary = {
        'n_frames': len(per_frame),
        'n_points': int(sum(f['n_points'] for f in per_frame)),
        'n_well_determined': int(sum(1 for f in per_frame if f['well_determined'])),
        'cte_mean_px': float(np.mean(cte)) if cte.size else float('nan'),
        'cte_median_px': float(np.median(cte)) if cte.size else float('nan'),
        'cte_p90_px': float(np.percentile(cte, 90)) if cte.size else float('nan'),
        'iou_mean': float(np.mean(column('iou_mean'))) if column('iou_mean').size else float('nan'),
        'oce_mean_px': float(np.mean(column('oce_mean'))) if column('oce_mean').size else float('nan'),
        'oce_median_px': float(np.median(column('oce_median'))) if column('oce_median').size else float('nan'),
    }
    return {'per_frame': per_frame, 'summary': summary}


def audit_results(
    results_dir: Path,
    points: Mapping[int, GroundControlPoints],
    reference_index: int,
    case_dir: Optional[Path] = None,
    frame_size: Optional[Mapping[str, int]] = None,
    bins: Sequence[Tuple[int, Optional[int]]] = DEFAULT_OFFSET_BINS,
    unscored: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Audit each candidate pipeline in a results directory against the ground control points.

    A run with no estimate on any audited frame cannot be scored; its label is appended to
    `unscored` when given.

    Uses the per-pair homographies stored in each record (`store_homographies`); runs without them
    are skipped. A failed pair stores none and is scored as the identity warp, as everywhere else,
    so a failure counts against the candidate instead of dropping out. `failed_frames` lists the
    audited frames scored that way and `missing_frames` those the run has no record for.
    """
    store = ResultStore(Path(results_dir))
    audited: List[Dict[str, Any]] = []
    fitted = fitted_transforms(points, reference_index)
    audited_keys = {_pair_key(reference_index, index): index for index in points if index != reference_index}

    for run in store.iter_runs():
        matrices: Dict[str, Optional[np.ndarray]] = {}
        failed: List[int] = []
        object_boxes = 'obb'
        for record in run.get('records', []):
            key = record.get('pair')
            if not key:
                continue
            object_boxes = str(record.get('object_boxes', object_boxes))
            matrix = record.get('homography')
            if matrix is not None:
                matrices[key] = np.asarray(matrix, dtype=np.float64)
            elif record.get('failed'):
                matrices[key] = np.eye(3)
                if key in audited_keys:
                    failed.append(audited_keys[key])
        params = run.get('params', {})
        label = str(params.get('pipeline'))
        if label in (BASELINE_IDENTITY, BASELINE_REFERENCE):
            continue
        if not matrices:
            if unscored is not None:
                unscored.append(label)
            continue
        try:
            entry = audit_homographies(points, reference_index, matrices, label=label, bins=bins)
        except AuditError:
            if unscored is not None:
                unscored.append(label)
            continue
        entry['masking'] = params.get('masking')
        entry['failed_frames'] = sorted(set(failed))
        entry['missing_frames'] = sorted(index for key, index in audited_keys.items() if key not in matrices)
        entry['gcp_transform'] = score_against_gcp_transform(
            matrices, fitted, reference_index, case_dir, frame_size, object_boxes
        )
        audited.append(entry)

    return audited


def run_audit(case_dir: Path, args: argparse.Namespace) -> Path:
    """Audit a case's reference estimator, and optionally its candidate results."""
    case_dir = Path(case_dir).expanduser().absolute()
    case = load_case(case_dir)

    gcp_dir = Path(args.gcp) if getattr(args, 'gcp', None) else case_dir / GCP_DIR
    points = load_gcp_set(gcp_dir)

    payload = load_oracle(case_dir)
    reference_index = int(payload['reference_index'])
    homographies = oracle_matrices(payload)
    frame_size = payload.get('frame_size')

    bins = plan_bins(case.get('sampling'), 'audit_bins')
    reference_audit = audit_homographies(points, reference_index, homographies, label='reference_estimator', bins=bins)
    fitted = fitted_transforms(points, reference_index)
    reference_audit['gcp_transform'] = score_against_gcp_transform(
        homographies, fitted, reference_index, case_dir, frame_size
    )
    report: Dict[str, Any] = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'case': case_dir.name,
        'video': case['video']['path'],
        'gcp_dir': Path(os.path.relpath(gcp_dir.absolute(), case_dir)).as_posix(),
        'annotated_frames': sorted(points),
        'reference_estimator': reference_audit,
        'fitted_transforms': {
            str(index): {k: v for k, v in fit.items() if k != 'matrix'} for index, fit in fitted.items()
        },
        'metadata': system_metadata(),
    }

    review = proposal_review(gcp_dir, reference_index, reference_audit['per_frame'])
    if review.get('n_points'):
        report['proposal_review'] = review

    repeat_dir = case_dir / GCP_REPEAT_DIR
    if repeat_dir.is_dir():
        report['annotation_repeatability'] = annotation_repeatability(points, load_gcp_set(repeat_dir))

    output = case_dir / 'oracle' / AUDIT_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    _print_report(report)
    print(f"\033[92mAudit -> {output}\033[0m")

    if getattr(args, 'results', None):
        results_dir = Path(args.results)
        unscored: List[str] = []
        scores: Dict[str, Any] = {
            'created': report['created'],
            'case': report['case'],
            'gcp_dir': report['gcp_dir'],
            'annotated_frames': report['annotated_frames'],
            'candidates': audit_results(
                results_dir, points, reference_index, case_dir, frame_size, bins=bins, unscored=unscored
            ),
            'unscored_candidates': unscored,
            'metadata': report['metadata'],
        }
        if unscored:
            print(
                f"\033[93m{len(unscored)} run(s) have no estimate on any audited frame and were not scored against "
                f"the GCP-fitted transform: {', '.join(unscored)}. The audited frames must be among the "
                f"campaign's pairs (the full sampling plan includes them).\033[0m"
            )
        for candidate in scores['candidates']:
            if candidate['missing_frames']:
                print(
                    f"\033[93m{candidate['label']}: no record for audited frame(s) {candidate['missing_frames']}; "
                    f"its scores against the GCP-fitted transform cover fewer frames than the audit.\033[0m"
                )
            if candidate['failed_frames']:
                print(
                    f"  {candidate['label']}: failed on audited frame(s) {candidate['failed_frames']} "
                    f"(scored as identity)"
                )
        scores_path = results_dir / GCP_SCORES_NAME
        scores_path.write_text(json.dumps(scores, indent=2, default=str), encoding='utf-8')
        _print_candidates(scores['candidates'])
        print(f"\033[92mCandidates -> {scores_path}\033[0m")
    return output


def _print_report(report: Mapping[str, Any]) -> None:
    """Print the residual table and the resolution floor it implies."""
    estimator = report['reference_estimator']
    summary = estimator['summary']

    print("\n\033[1mReference estimator audit against ground control points\033[0m")
    print(f"  reference frame       {estimator['reference_index']} ({estimator['n_reference_points']} points)")
    print(f"  audited frames        {len(estimator['per_frame'])}")
    if estimator['skipped_frames']:
        print(f"  skipped (no estimate) {estimator['skipped_frames']}")

    print(f"\n  {'frame':>8}  {'offset':>8}  {'points':>7}  {'median px':>9}  {'mean px':>9}  {'max px':>9}")
    print(f"  {'-' * 58}")
    for frame in estimator['per_frame']:
        print(
            f"  {frame['frame_index']:>8}  {frame['offset']:>8}  {frame['n_points']:>7}  "
            f"{frame['residual_median_px']:>9.2f}  {frame['residual_mean_px']:>9.2f}  "
            f"{frame['residual_max_px']:>9.2f}"
        )

    if summary.get('n_points'):
        print(
            f"\n  pooled over {summary['n_points']} point(s): "
            f"median {summary['residual_median_px']:.2f} px, "
            f"p95 {summary['residual_p95_px']:.2f} px, "
            f"max {summary['residual_max_px']:.2f} px"
        )
        for label, bin_summary in (estimator.get('summary_by_offset') or {}).items():
            if bin_summary.get('n_points'):
                print(
                    f"    offset {label:>10}: p95 {bin_summary['residual_p95_px']:.2f} px "
                    f"over {bin_summary['n_points']} point(s)"
                )
        review = report.get('proposal_review') or {}
        if review.get('n_points'):
            corrected = (
                f" (median {review['correction_median_px']:.2f} px, max {review['correction_max_px']:.2f} px)"
                if review['n_corrected']
                else ''
            )
            print(
                f"  proposals: {review['n_accepted']} of {review['n_points']} point(s) accepted, "
                f"{review['n_corrected']} corrected{corrected}, {review['n_placed_by_hand']} placed by hand"
            )
            if review.get('n_hand_scored'):
                print(
                    f"  hand-placed points within 0.5 px of the estimator's prediction: "
                    f"{review['n_hand_within_half_px']} of {review['n_hand_scored']} (the panel opens near that "
                    f"prediction; a large share beside a large click repeatability means the clicks followed it)"
                )
        elif 'share_within_half_px' in summary:
            print(
                f"  points within 0.5 px of the estimator's prediction: {100 * summary['share_within_half_px']:.0f}% "
                f"(the annotation panel opens near that prediction; a large share beside a large click "
                f"repeatability means the clicks followed the panel)"
            )
        repeat = report.get('annotation_repeatability') or {}
        if repeat.get('n_points'):
            print(
                f"  click repeatability: p95 {repeat['residual_p95_px']:.2f} px over {repeat['n_points']} "
                f"point(s) in {repeat['n_frames']} frame(s)"
            )
        else:
            print(
                f"  no {GCP_REPEAT_DIR}/ session: the residual above conflates the estimator's error with the "
                f"annotator's (annotate-gcp --repeat measures the latter)"
            )
        thin = [str(i) for i, fit in (report.get('fitted_transforms') or {}).items() if not fit.get('well_determined')]
        if thin:
            print(f"  frames whose GCP-fitted transform rests on fewer than 8 points: {', '.join(thin)}")
        print(
            f"\n\033[1m  Equivalence margin: {summary['residual_p95_px']:.2f} px.\033[0m A reference-relative "
            f"difference between candidates\n  below this figure lies within the reference estimator's own "
            f"uncertainty and is a tie under the\n  analysis plan. State this margin wherever Part II numbers are "
            f"reported, and read the masking\n  comparison from the scores against the GCP-fitted transform below."
        )


def _print_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Print each candidate's residual and its scores against the GCP-fitted transform."""
    for candidate in candidates:
        candidate_summary = candidate['summary']
        against = (candidate.get('gcp_transform') or {}).get('summary') or {}
        if candidate_summary.get('n_points'):
            print(
                f"  {candidate['label']:<32} true-GT median {candidate_summary['residual_median_px']:7.2f} px "
                f"(p95 {candidate_summary['residual_p95_px']:7.2f} px, n={candidate_summary['n_points']}) | "
                f"vs fitted transform: CTE {against.get('cte_median_px', float('nan')):6.2f} px, "
                f"IoU {against.get('iou_mean', float('nan')):5.3f}, "
                f"OCE {against.get('oce_mean_px', float('nan')):6.2f} px"
            )


def synthetic_check(
    homography: np.ndarray,
    points: Sequence[Sequence[float]],
    offset_px: float = 0.0,
) -> Dict[str, Any]:
    """
    Residuals for a known homography applied to synthetic points.

    With `offset_px=0` the residual is zero; displacing the reference points by `offset_px` gives
    exactly that residual.
    """
    source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    labels = tuple(f"p{index}" for index in range(len(source)))

    target = warp_points(source, np.asarray(homography, dtype=np.float64))
    if offset_px:
        target = target + np.array([[float(offset_px), 0.0]])

    reference = GroundControlPoints(frame_index=0, labels=labels, points=target)
    current = GroundControlPoints(frame_index=1, labels=labels, points=source)
    return residuals(reference, current, homography)
