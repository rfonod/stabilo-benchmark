#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
tables.py - Aggregate result directories into LaTeX tables.

- Error columns report the median and the 90th percentile.
- Failures have their own column and count as misses in every HEA and mAA.
- Inlier statistics are tabulated but never bolded.
- Rows are ranked on mAA over [0, MAA_UPPER_PX]. In Part II the audited residual is the
  equivalence margin (`stats.py`); `maa_floor`, integrated from it, is a sensitivity column.
- Bold marks the tie set of each column (`tie_sets_for`): the best value and every value not
  distinguishable from it across clusters (sites in Part I, cases in Part II). A single-case
  table bolds nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stabilo_benchmark.pipelines import DEFAULT_OBJECT_BOXES
from stabilo_benchmark.report.names import display_label, masking_name, pipeline_name
from stabilo_benchmark.report.survey_cases import CELL_ORDER
from stabilo_benchmark.results import MANIFEST_NAME, RUN_INFEASIBLE, ResultStore, infeasible_runs, to_dataframe
from stabilo_benchmark.sampling import bin_label, offset_bin_label, plan_bins
from stabilo_benchmark.stats import TieSet, margin_for, site_of, tie_set

# Default upper limit of the mAA integration, in pixels.
MAA_UPPER_PX = 5.0
# The reference-free photometric family, in table order.
PHOTOMETRIC_COLUMNS = ('ssim', 'zncc', 'mi', 'gradient')

# Column -> (LaTeX header, format, higher-is-better). Order defines table column order.
# `None` for the direction marks a column that is never bolded.
STATIC_COLUMNS: Dict[str, tuple] = {
    'maa': ('mAA', '{:.4f}', True),
    'hea_1px': (r'HEA$_{1}$', '{:.3f}', True),
    'hea_3px': (r'HEA$_{3}$', '{:.3f}', True),
    'hea_5px': (r'HEA$_{5}$', '{:.3f}', True),
    'cte_median': (r'CTE$_{50}$ [px]', '{:.3f}', False),
    'cte_p90': (r'CTE$_{90}$ [px]', '{:.3f}', False),
    'iou_mean': ('IoU', '{:.4f}', True),
    'oce_mean': ('OCE [px]', '{:.3f}', False),
    'failure_rate': ('Fail', '{:.1%}', False),
    'time_median': ('Time [s]', '{:.3f}', False),
}

ESTIMATOR_COLUMNS: Dict[str, tuple] = {
    'ransac_epipolar_threshold': (r'$\theta$ [px]', '{:g}', None),
    'maa': ('mAA', '{:.4f}', True),
    'hea_1px': (r'HEA$_{1}$', '{:.3f}', True),
    'hea_3px': (r'HEA$_{3}$', '{:.3f}', True),
    'cte_median': (r'CTE$_{50}$ [px]', '{:.3f}', False),
    'cte_p90': (r'CTE$_{90}$ [px]', '{:.3f}', False),
    # Never bolded.
    'inlier_ratio': ('Inlier ratio', '{:.3f}', None),
    'failure_rate': ('Fail', '{:.1%}', False),
    'time_median': ('Time [s]', '{:.3f}', False),
}

DYNAMIC_COLUMNS: Dict[str, tuple] = {
    'maa': ('mAA', '{:.4f}', True),
    # Integrated from the audited residual; never bolded.
    'maa_floor': (r'mAA$_{\varepsilon_{\mathrm{floor}}}$', '{:.4f}', None),
    'cte_median': (r'CTE$_{50}$ [px]', '{:.2f}', False),
    'cte_p90': (r'CTE$_{90}$ [px]', '{:.2f}', False),
    'ssim': ('SSIM', '{:.4f}', True),
    'zncc': ('ZNCC', '{:.4f}', True),
    'mi': ('MI [bits]', '{:.3f}', True),
    'gradient': ('GOS', '{:.4f}', True),
    # Median share of the frame the photometric metrics were computed over; never bolded.
    'region_fraction': ('Region', '{:.1%}', None),
    'iou_mean': ('IoU', '{:.4f}', True),
    'oce_mean': ('OCE [px]', '{:.2f}', False),
    # The direction of the error, not its size: 0.5 is isotropic and below it is not better; never bolded.
    'motion_alignment': ('Align', '{:.3f}', None),
    'inlier_ratio': ('Inlier ratio', '{:.3f}', None),
    'failure_rate': ('Fail', '{:.1%}', False),
    'time_median': ('Time [s]', '{:.3f}', False),
}


# The breakdown tables (by offset, cell, persistence) carry the outcomes the masking contrast is read on.
BREAKDOWN_COLUMNS = (
    'maa',
    'cte_median',
    'zncc',
    'gradient',
    'iou_mean',
    'oce_mean',
    'motion_alignment',
    'failure_rate',
)

# Columns scored on object boxes; `metrics.object_boxes` decides which boxes those are.
OBJECT_BOX_COLUMNS = ('iou_mean', 'oce_mean')

# Candidates scored against the homography fitted to each audited frame's ground control points
# (`gcp.fit_homography`), independent of the reference estimator.
GCP_COLUMNS: Dict[str, tuple] = {
    'cte_median': (r'CTE$^{\mathrm{GCP}}_{50}$ [px]', '{:.2f}', False),
    'cte_p90': (r'CTE$^{\mathrm{GCP}}_{90}$ [px]', '{:.2f}', False),
    'cte_max': (r'CTE$^{\mathrm{GCP}}_{\max}$ [px]', '{:.2f}', None),
    'iou_mean': (r'IoU$^{\mathrm{GCP}}$', '{:.4f}', True),
    'oce_mean': (r'OCE$^{\mathrm{GCP}}$ [px]', '{:.2f}', False),
    'n_frames': ('Frames', '{:.0f}', None),
    'n_points': ('Points', '{:.0f}', None),
}

# The reference estimator's audit, one row per case. Per-bin columns are added by `gcp_audit_columns`.
GCP_AUDIT_COLUMNS: Dict[str, tuple] = {
    'n_frames': ('Frames', '{:.0f}', None),
    'n_points': ('Points', '{:.0f}', None),
    'residual_median_px': (r'Res.$_{50}$ [px]', '{:.2f}', None),
    'residual_p95_px': (r'Res.$_{95}$ [px]', '{:.2f}', None),
    'repeat_p95_px': (r'Rep.$_{95}$ [px]', '{:.2f}', None),
    'share_within_half_px': (r'$<$\,0.5\,px', '{:.0%}', None),
}
GCP_AUDIT_POOLED = 'all cases'


def aggregate_gcp_audit(
    audits: Mapping[str, Mapping[str, Any]], cells: Optional[Mapping[str, str]] = None
) -> pd.DataFrame:
    """
    The reference estimator's audit, one row per case and a last row pooling every point.

    `audits` maps a case name to its AUDIT.json. Residual columns are percentiles over the case's
    points; `p95_<bin>` is the 95th percentile within each audit offset bin; `repeat_p95_px` is the
    click repeatability where a repeat session exists. `share_within_half_px` is the share of the
    points placed in the annotation panel that lie within half a pixel of the prediction: the
    hand-placed points where proposals were recorded (`gcp.proposal_review`), every point otherwise.
    The pooled row recomputes every percentile and share over the points of all cases, and its
    repeatability is the largest of the cases'.
    """
    from stabilo_benchmark.gcp import summarize_residuals  # noqa: PLC0415 - avoids import cycle

    shared = ('n_frames', 'n_points', 'residual_median_px', 'residual_p95_px')
    rows: List[Dict[str, Any]] = []
    frames_by_bin: Dict[str, List[Mapping[str, Any]]] = {}
    all_frames: List[Mapping[str, Any]] = []
    repeats: List[float] = []
    panel_within, panel_scored = 0, 0
    for case, audit in audits.items():
        estimator = audit['reference_estimator']
        summary = estimator.get('summary') or {}
        row: Dict[str, Any] = {
            'case': case,
            'cell': (cells or {}).get(case, ''),
            **{key: summary.get(key, float('nan')) for key in shared},
        }
        within, scored = _panel_points_within_half_px(audit)
        row['share_within_half_px'] = within / scored if scored else float('nan')
        panel_within, panel_scored = panel_within + within, panel_scored + scored
        for label, by_bin in (estimator.get('summary_by_offset') or {}).items():
            row[f"p95_{label}"] = by_bin.get('residual_p95_px', float('nan'))
            frames_by_bin.setdefault(label, [])
        bins = audit_bin_ranges(estimator.get('summary_by_offset') or {})
        for frame in estimator.get('per_frame') or []:
            all_frames.append(frame)
            label = offset_bin_label(int(frame.get('offset', 0)), bins)
            if label is not None:
                frames_by_bin.setdefault(label, []).append(frame)
        repeat = (audit.get('annotation_repeatability') or {}).get('residual_p95_px')
        row['repeat_p95_px'] = float(repeat) if repeat is not None else float('nan')
        if repeat is not None:
            repeats.append(float(repeat))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    pooled = summarize_residuals(all_frames)
    pooled_row: Dict[str, Any] = {
        'case': GCP_AUDIT_POOLED,
        'cell': '',
        **{key: pooled.get(key, float('nan')) for key in shared},
        'share_within_half_px': panel_within / panel_scored if panel_scored else float('nan'),
        'repeat_p95_px': max(repeats) if repeats else float('nan'),
    }
    for label, frames in frames_by_bin.items():
        pooled_row[f"p95_{label}"] = summarize_residuals(frames).get('residual_p95_px', float('nan'))
    rows.append(pooled_row)
    return pd.DataFrame(rows)


def _panel_points_within_half_px(audit: Mapping[str, Any]) -> Tuple[int, int]:
    """(within half a pixel of the prediction, scored) over the points placed in the annotation panel."""
    review = audit.get('proposal_review') or {}
    if review.get('n_points'):
        return int(review.get('n_hand_within_half_px', 0)), int(review.get('n_hand_scored', 0))
    values = [
        value
        for frame in audit['reference_estimator'].get('per_frame') or []
        for value in frame.get('residuals_px', [])
    ]
    return int(sum(value < 0.5 for value in values)), len(values)


# How the annotated points relate to the recorded proposals (`gcp.proposal_review`), one row per case.
GCP_PROPOSAL_COLUMNS: Dict[str, tuple] = {
    'n_points': ('Points', '{:.0f}', None),
    'n_accepted': ('Accepted', '{:.0f}', None),
    'n_corrected': ('Corrected', '{:.0f}', None),
    'correction_median_px': (r'Corr.$_{50}$ [px]', '{:.2f}', None),
    'correction_max_px': (r'Corr.$_{\max}$ [px]', '{:.2f}', None),
    'n_placed_by_hand': ('By hand', '{:.0f}', None),
}


def aggregate_gcp_proposals(
    audits: Mapping[str, Mapping[str, Any]], cells: Optional[Mapping[str, str]] = None
) -> pd.DataFrame:
    """
    The proposal review of every case that records one, and a last row pooling them.

    `audits` maps a case name to its AUDIT.json. The pooled row sums the counts and recomputes the
    correction median and maximum over every corrected point.
    """
    counts = ('n_points', 'n_accepted', 'n_corrected', 'n_placed_by_hand')
    rows: List[Dict[str, Any]] = []
    corrections: List[float] = []
    for case, audit in audits.items():
        review = audit.get('proposal_review') or {}
        if not review.get('n_points'):
            continue
        rows.append(
            {
                'case': case,
                'cell': (cells or {}).get(case, ''),
                **{key: review.get(key, 0) for key in counts},
                'correction_median_px': review.get('correction_median_px', float('nan')),
                'correction_max_px': review.get('correction_max_px', float('nan')),
            }
        )
        corrections.extend(review.get('corrections_px') or [])
    if not rows:
        return pd.DataFrame()
    rows.append(
        {
            'case': GCP_AUDIT_POOLED,
            'cell': '',
            **{key: sum(row[key] for row in rows) for key in counts},
            'correction_median_px': float(np.median(corrections)) if corrections else float('nan'),
            'correction_max_px': float(np.max(corrections)) if corrections else float('nan'),
        }
    )
    return pd.DataFrame(rows)


def audit_bin_ranges(by_offset: Mapping[str, Any]) -> List[Tuple[int, Optional[int]]]:
    """The (low, high) bins behind `summary_by_offset` labels (`1-120`, `>3600`)."""
    bins: List[Tuple[int, Optional[int]]] = []
    for label in by_offset:
        if label.startswith('>'):
            bins.append((int(label[1:]) + 1, None))
        else:
            low, high = label.split('-')
            bins.append((int(low), int(high)))
    return bins


def gcp_audit_columns(frame: pd.DataFrame) -> Dict[str, tuple]:
    """`GCP_AUDIT_COLUMNS` with one 95th-percentile column per audit offset bin, in bin order."""
    columns = dict(GCP_AUDIT_COLUMNS)
    bins = sorted(
        (c for c in frame.columns if c.startswith('p95_')), key=lambda c: int(c[4:].lstrip('>').split('-')[0])
    )
    per_bin = {c: (rf"Res.$_{{95}}$ {c[4:].replace('-', '--')}", '{:.2f}', None) for c in bins}
    keys = list(columns)
    at = keys.index('residual_p95_px') + 1
    return {k: columns[k] for k in keys[:at]} | per_bin | {k: columns[k] for k in keys[at:]}


# Baseline rows: the unstabilized frame (identity) and the reference estimator's transform.
BASELINE_IDENTITY = 'identity'
BASELINE_PIPELINES = (BASELINE_IDENTITY, 'reference')

# Per-frame object statistics averaged over a pipeline's measurements. Mirrors
# `metrics.geometric.OBJECT_STATISTICS`; importing it would create an import cycle.
OBJECT_SUMMARY_COLUMNS = tuple(
    f"{metric}_{statistic}"
    for metric in ('iou', 'oce')
    for statistic in ('mean', 'std', 'min', 'q1', 'median', 'q3', 'max')
)


# Output column -> run parameter, for the N/A row an infeasible run contributes to a table.
INFEASIBLE_COLUMNS = {
    'pipeline': 'pipeline',
    'masking': 'masking',
    'detector': 'detector_name',
    'matcher': 'matcher_name',
}
ESTIMATOR_INFEASIBLE_COLUMNS = {
    'detector': 'detector_name',
    'masking': 'masking',
    'ransac_method': 'ransac_method',
    'ransac_epipolar_threshold': 'ransac_epipolar_threshold',
}


def object_boxes_of(frame: pd.DataFrame) -> str:
    """
    The box geometry a result set's IoU and OCE were scored on.

    Raises when a record does not state it or when the set mixes geometries.
    """
    values = sorted({str(value) for value in frame['object_boxes'].dropna()}) if 'object_boxes' in frame else []
    if not values:
        raise ValueError("These results do not record 'object_boxes'; re-run them with the current version.")
    if len(values) > 1:
        raise ValueError(f"These results mix object box geometries {values}; report each separately.")
    return values[0]


def columns_for_object_boxes(columns: Mapping[str, tuple], object_boxes: str) -> Dict[str, tuple]:
    """Tag the box-based headers with HBB when they were scored on HBBs."""
    columns = dict(columns)
    if object_boxes == DEFAULT_OBJECT_BOXES:
        return columns
    tag = object_boxes.upper()
    for name in OBJECT_BOX_COLUMNS:
        if name in columns:
            header, *rest = columns[name]
            head, _, unit = header.partition(' [')
            columns[name] = (f"{head}$_{{\\mathrm{{{tag}}}}}$" + (f" [{unit}" if unit else ''), *rest)
    return columns


def load_results(results_dir: Path) -> pd.DataFrame:
    """Flatten a result directory into one row per measurement."""
    store = ResultStore(Path(results_dir))
    frame = to_dataframe(store)
    if frame.empty:
        aborted = [str(run['params'].get('pipeline', run['run_number'])) for run in infeasible_runs(store)]
        if aborted:
            raise ValueError(
                f"No measurements found in {results_dir}: all {len(aborted)} run(s) were "
                f"resource-infeasible on the machine that ran them ({', '.join(aborted)})."
            )
        raise ValueError(f"No measurements found in {results_dir}")
    return frame


def infeasible_rows(results_dir: Path, columns: Mapping[str, str] = INFEASIBLE_COLUMNS) -> List[Dict[str, Any]]:
    """
    One N/A row per run this machine could not hold, to append to an aggregate.

    Such a run has no measurements, so `load_results` does not see it. `columns` maps an output
    column to the run parameter it is read from.
    """
    rows: List[Dict[str, Any]] = []
    for run in infeasible_runs(ResultStore(Path(results_dir))):
        params = run.get('params', {})
        row: Dict[str, Any] = {column: params.get(key) for column, key in columns.items()}
        row['status'] = RUN_INFEASIBLE
        row['n'] = 0
        row['infeasible_reason'] = run.get('resource_error', {}).get('type')
        rows.append(row)
    return rows


def _percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float('nan')


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float('nan')


def _group_summary(
    group: pd.DataFrame,
    hea_thresholds: Sequence[float],
    photometric: Sequence[str],
    maa_upper: float = MAA_UPPER_PX,
    maa_lower: float = 0.0,
) -> Dict[str, Any]:
    """Summary statistics for one pipeline's measurements."""
    from stabilo_benchmark.metrics.geometric import hea, maa, threshold_label  # noqa: PLC0415 - avoids import cycle

    summary: Dict[str, Any] = {
        'n': int(len(group)),
        'failure_rate': float(group['failed'].astype(bool).mean()) if 'failed' in group else float('nan'),
        'cte_median': _percentile(group['cte_mean'], 50) if 'cte_mean' in group else float('nan'),
        'cte_p90': _percentile(group['cte_mean'], 90) if 'cte_mean' in group else float('nan'),
    }

    if 'cte_mean' in group:
        errors = pd.to_numeric(group['cte_mean'], errors='coerce').tolist()
        # Non-finite entries count as failures.
        summary.update(hea(errors, hea_thresholds))
        for threshold in hea_thresholds:
            summary.setdefault(f"hea_{threshold_label(threshold)}", float('nan'))
        # The ranking quantity, recorded with its bounds.
        summary['maa'] = maa(errors, upper=maa_upper, lower=0.0)
        summary['maa_lower_px'] = 0.0
        summary['maa_upper_px'] = float(maa_upper)
        # The same integral from the audited residual.
        if maa_lower > 0.0 and maa_upper > maa_lower:
            summary['maa_floor'] = maa(errors, upper=maa_upper, lower=maa_lower)
            summary['maa_floor_px'] = float(maa_lower)

    for column in (*OBJECT_SUMMARY_COLUMNS, 'inlier_ratio', 'motion_alignment', *photometric):
        if column in group:
            summary[column] = _mean(group[column])

    for column in ('n_inliers', 'n_matches', 'n_keypoints_ref'):
        if column in group:
            summary[column] = _mean(group[column])

    if 'region_area_fraction' in group:
        summary['region_fraction'] = _percentile(group['region_area_fraction'], 50)

    if 'time_stabilize_s' in group:
        summary['time_median'] = _percentile(group['time_stabilize_s'], 50)
    # Pixels processed by the detector: the hardware-independent measure of work.
    if 'processed_pixels' in group:
        summary['processed_mp'] = _percentile(group['processed_pixels'], 50) / 1e6

    return summary


def aggregate_static(
    results_dir: Path,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """One row per pipeline, summarizing a static campaign; mAA integrates from zero."""
    frame = load_results(results_dir)
    rows: List[Dict[str, Any]] = []
    for (pipeline, masking), group in frame.groupby(['pipeline', 'masking'], sort=False):
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'detector': group['detector_name'].iloc[0] if 'detector_name' in group else None,
                'matcher': group['matcher_name'].iloc[0] if 'matcher_name' in group else None,
                **_group_summary(group, hea_thresholds, photometric=(), maa_upper=maa_upper),
            }
        )
    return pd.DataFrame([*rows, *infeasible_rows(results_dir)])


# Parameters a sweep varies within one preset, and how a row label names them.
SWEPT_PARAMETERS: Dict[str, str] = {
    'downsample_ratio': 'scale',
    'max_features': 'features',
    'filter_ratio': 'ratio',
    'ransac_epipolar_threshold': 'thr.',
    'ransac_method': 'method',
    'mask_margin_ratio': 'margin',
}


def swept_run_labels(frame: pd.DataFrame) -> Dict[str, str]:
    """
    Readable names for the runs of a sweep: `sift+none#6` becomes `sift+none (scale 0.5, features
    8000)`, naming only the parameters that vary between runs of the same preset and masking.
    Empty when no run label carries a repeat suffix.
    """
    if 'pipeline' not in frame.columns:
        return {}
    labels = frame['pipeline'].astype(str)
    if not labels.str.contains('#', regex=False).any():
        return {}
    columns = [c for c in SWEPT_PARAMETERS if c in frame.columns]
    runs = frame.assign(pipeline=labels).groupby('pipeline', sort=False)[columns].first()
    bare = pd.Series(runs.index.str.replace(r'#\d+$', '', regex=True), index=runs.index)
    varying = [c for c in columns if runs.groupby(bare)[c].nunique(dropna=True).max() > 1]
    if not varying:
        return {}

    def value(v: Any) -> str:
        return f"{float(v):g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)

    # A run without any of the swept values (a baseline) keeps its name.
    return {
        label: f"{bare[label]} ({', '.join(f'{SWEPT_PARAMETERS[c]} {value(runs.at[label, c])}' for c in varying)})"
        for label in runs.index
        if runs.loc[label, varying].notna().any()
    }


def is_budget_campaign(frame: pd.DataFrame) -> bool:
    """Whether a results frame came from a compute-budget sweep (`downsample_ratio` and `max_features` vary)."""
    return all(
        column in frame.columns and frame[column].nunique(dropna=True) > 1
        for column in ('downsample_ratio', 'max_features')
    )


def aggregate_budget(results_dir: Path, maa_upper: float = MAA_UPPER_PX) -> pd.DataFrame:
    """
    One row per (preset, masking, downsample_ratio, max_features) of a budget campaign: mAA and the
    median registration time. Infeasible runs, which measured nothing, are left out.
    """
    frame = load_results(results_dir)
    if not is_budget_campaign(frame):
        raise ValueError("Results do not sweep both 'downsample_ratio' and 'max_features'")
    summary = aggregate_static(results_dir, maa_upper=maa_upper)
    params = frame.groupby('pipeline', sort=False)[['downsample_ratio', 'max_features']].first()
    summary = summary.join(params, on='pipeline', how='inner')
    summary = summary[np.isfinite(pd.to_numeric(summary['maa'], errors='coerce'))]
    preset = summary['pipeline'].astype(str).str.replace(r'\+[a-z]+(#\d+)?$', '', regex=True)
    columns = ['preset', 'masking', 'downsample_ratio', 'max_features', 'maa', 'time_median', 'processed_mp']
    summary = summary.assign(preset=preset)
    return summary[[c for c in columns if c in summary.columns]].reset_index(drop=True)


def sweep_value(config: Mapping[str, Any], key: str) -> float:
    """The value a sweep set at dotted `key`; a [v, v] range reads as v."""
    value: Any = config
    for part in key.split('.'):
        value = value[part]
    if isinstance(value, (list, tuple)):
        if len(set(value)) != 1:
            raise ValueError(f"'{key}' is a range {value!r}, not one sweep point")
        value = value[0]
    return float(value)


def sweep_key(configs: Sequence[Mapping[str, Any]]) -> str:
    """
    The one dotted config key whose value differs between the campaigns of a sweep.

    `name` is ignored. Raises unless exactly one key differs.
    """

    def flatten(mapping: Mapping[str, Any], prefix: str = '') -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in mapping.items():
            path = f"{prefix}{key}"
            if isinstance(value, Mapping):
                out.update(flatten(value, f"{path}."))
            else:
                out[path] = json.dumps(value, sort_keys=True, default=str)
        return out

    flat = [flatten({k: v for k, v in config.items() if k != 'name'}) for config in configs]
    keys = set().union(*flat)
    varying = sorted(k for k in keys if len({f.get(k) for f in flat}) > 1)
    if len(varying) != 1:
        raise ValueError(
            f"A sweep report needs campaigns that differ in exactly one config key; these differ in {varying}"
        )
    return varying[0]


def aggregate_sweep(results_dirs: Sequence[Path], maa_upper: float = MAA_UPPER_PX) -> Tuple[str, pd.DataFrame]:
    """
    One row per (pipeline, masking, sweep value) over static campaigns that differ in one config
    key, which is returned with the rows. Infeasible runs are left out.
    """
    configs = []
    for results_dir in results_dirs:
        manifest = json.loads((Path(results_dir) / MANIFEST_NAME).read_text(encoding='utf-8'))
        configs.append(manifest.get('config') or {})
    key = sweep_key(configs)
    rows = []
    for results_dir, config in zip(results_dirs, configs, strict=True):
        summary = aggregate_static(Path(results_dir), maa_upper=maa_upper)
        rows.append(summary.assign(value=sweep_value(config, key)))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[np.isfinite(pd.to_numeric(frame['maa'], errors='coerce'))]
    frame = frame.assign(pipeline=frame['pipeline'].astype(str).str.replace(r'#\d+$', '', regex=True))
    return key, frame.sort_values(['pipeline', 'value']).reset_index(drop=True)


def is_estimator_campaign(frame: pd.DataFrame) -> bool:
    """Whether a results frame came from a sweep over robust estimators (`ransac_method` varies)."""
    return 'ransac_method' in frame.columns and frame['ransac_method'].nunique(dropna=True) > 1


def aggregate_estimators(
    results_dir: Path,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """
    One row per (pipeline, masking, estimator, threshold) for an estimator campaign.

    The estimator is named by `pipelines.ransac_name`.
    """
    from stabilo_benchmark.pipelines import ransac_name  # noqa: PLC0415 - avoids import cycle

    frame = load_results(results_dir)
    if 'ransac_method' not in frame.columns:
        raise ValueError(f"{results_dir} did not record a 'ransac_method' column; this is not an estimator campaign")

    candidates = ('detector_name', 'masking', 'ransac_method', 'ransac_epipolar_threshold')
    keys = [name for name in candidates if name in frame]
    rows: List[Dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,), strict=True))
        rows.append(
            {
                'detector': row.get('detector_name'),
                'masking': row.get('masking'),
                'estimator': ransac_name(int(row['ransac_method'])),
                'ransac_method': int(row['ransac_method']),
                'ransac_epipolar_threshold': row.get('ransac_epipolar_threshold'),
                **_group_summary(group, hea_thresholds, photometric=(), maa_upper=maa_upper),
            }
        )
    aborted = infeasible_rows(results_dir, ESTIMATOR_INFEASIBLE_COLUMNS)
    for row in aborted:
        if row.get('ransac_method') is not None:
            row['estimator'] = ransac_name(int(row['ransac_method']))
    return pd.DataFrame([*rows, *aborted])


def estimator_overview(summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, tuple]]:
    """
    The estimator campaign on one page: a row per detector (and masking, when it varies) and
    estimator, with mAA at each inlier threshold, the inlier ratio at the smallest and largest
    threshold, and over all thresholds the trials, failures, and trials beyond the mAA interval that
    did not fail. Returns the table and its columns for `write_latex_table`; nothing is bolded.

    `summary` is `aggregate_estimators` output; infeasible rows are left out.
    """
    rows = summary[np.isfinite(pd.to_numeric(summary['maa'], errors='coerce'))].copy()
    thresholds = sorted(float(t) for t in rows['ransac_epipolar_threshold'].dropna().unique())
    keys = ['detector', *(['masking'] if rows['masking'].nunique() > 1 else []), 'estimator']
    n = pd.to_numeric(rows['n'], errors='coerce')
    rows = rows.assign(
        failures=n * pd.to_numeric(rows['failure_rate'], errors='coerce'),
        misses=n * (1.0 - pd.to_numeric(rows[f"hea_{_threshold_label(MAA_UPPER_PX)}"], errors='coerce')),
    )
    table: List[Dict[str, Any]] = []
    for values, group in rows.groupby(keys, sort=False):
        entry: Dict[str, Any] = dict(zip(keys, values, strict=True))
        by_threshold = group.set_index(group['ransac_epipolar_threshold'].astype(float))
        for threshold in thresholds:
            entry[f"maa_at_{threshold:g}"] = (
                float(by_threshold.at[threshold, 'maa']) if threshold in by_threshold.index else float('nan')
            )
        for name, threshold in (('inlier_low', thresholds[0]), ('inlier_high', thresholds[-1])):
            present = threshold in by_threshold.index and 'inlier_ratio' in by_threshold
            entry[name] = float(by_threshold.at[threshold, 'inlier_ratio']) if present else float('nan')
        entry['n'] = float(pd.to_numeric(group['n'], errors='coerce').sum())
        entry['failures'] = float(np.round(group['failures'].sum()))
        entry['beyond'] = float(np.round(group['misses'].sum() - group['failures'].sum()))
        table.append(entry)
    columns: Dict[str, tuple] = {f"maa_at_{t:g}": (f"{t:g}", '{:.4f}', None) for t in thresholds}
    columns |= {
        'inlier_low': (rf"Inl.\,{thresholds[0]:g}", '{:.2f}', None),
        'inlier_high': (rf"Inl.\,{thresholds[-1]:g}", '{:.2f}', None),
        'n': ('Trials', '{:.0f}', None),
        'failures': ('Fail', '{:.0f}', None),
        'beyond': (rf"$>${MAA_UPPER_PX:g}\,px", '{:.0f}', None),
    }
    return pd.DataFrame(table), columns


def _threshold_label(threshold: float) -> str:
    from stabilo_benchmark.metrics.geometric import threshold_label  # noqa: PLC0415 - avoids import cycle

    return threshold_label(threshold)


def aggregate_dynamic(
    results_dir: Path,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    photometric: Sequence[str] = PHOTOMETRIC_COLUMNS,
    epsilon_floor: float = 0.0,
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """
    One row per pipeline, summarizing a dynamic campaign.

    `epsilon_floor` is the reference estimator's audited 95th-percentile residual. It is the
    equivalence margin for the reference-relative pixel columns (`stats.margin_for`) and the lower
    limit of `maa_floor`; the headline mAA integrates from zero. It defaults to zero for an
    unaudited case, and `build_report` warns when that happens.

    The identity and reference baselines appear as rows named after `BASELINE_PIPELINES`, with
    masking `none`.
    """
    frame = load_results(results_dir)
    rows: List[Dict[str, Any]] = []
    for (pipeline, masking), group in frame.groupby(['pipeline', 'masking'], sort=False):
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'detector': group['detector_name'].iloc[0] if 'detector_name' in group else None,
                **_group_summary(
                    group,
                    hea_thresholds,
                    photometric=photometric,
                    maa_upper=maa_upper,
                    maa_lower=epsilon_floor,
                ),
            }
        )
    return pd.DataFrame([*rows, *infeasible_rows(results_dir)])


def aggregate_by_offset(
    results_dir: Optional[Path] = None,
    bins: Optional[Sequence[Tuple[int, Optional[int]]]] = None,
    epsilon_floor: float = 0.0,
    frame: Optional[pd.DataFrame] = None,
    floors: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    Dynamic results grouped by pipeline and temporal-offset bin.

    `bins` default to the campaign's `sampling.bins` from its manifest, else `DEFAULT_OFFSET_BINS`.
    `frame` (from `load_case_results`) replaces reading `results_dir` and pools its cases with equal
    weight, each case's `maa_floor` from `floors`.
    """
    if frame is None:
        frame = load_results(results_dir)
    if 'abs_offset' not in frame:
        raise ValueError("Results do not contain 'abs_offset'; this is a dynamic-benchmark table only")

    if bins is None:
        bins = plan_bins(_manifest_sampling(results_dir) if results_dir is not None else None)
    order = {bin_label(low, high): i for i, (low, high) in enumerate(bins)}
    frame = frame.assign(offset_bin=[offset_bin_label(offset, bins) for offset in frame['abs_offset']])
    frame = frame[frame['offset_bin'].notna()]
    frame = frame.assign(_bin_order=frame['offset_bin'].map(order)).sort_values('_bin_order', kind='stable')

    rows: List[Dict[str, Any]] = []
    for (pipeline, masking, offset_bin), group in frame.groupby(['pipeline', 'masking', 'offset_bin'], sort=False):
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'offset_bin': str(offset_bin),
                **_breakdown_summary(group, epsilon_floor, floors),
            }
        )
    return pd.DataFrame(rows)


# The row label of a breakdown table's mean over the candidate pipelines.
AVERAGE_PIPELINE = 'All pipelines'
MASKING_ORDER = ('none', 'hbb', 'obb')


def average_pipeline_rows(table: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    A breakdown table reduced to its average pipeline, for a table that fits a page.

    Per stratum `key` and masking mode, every numeric column is averaged over the candidate
    pipelines (a sweep's runs are grouped by their swept values, so `sift+obb (margin 0.15)` joins
    `orb+obb (margin 0.15)`); the baseline rows follow each stratum unchanged. `attrs['n_pipelines']`
    counts the pipelines averaged.
    """
    labels = table['pipeline'].astype(str)
    baselines = table[labels.isin(BASELINE_PIPELINES)]
    candidates = table[~labels.isin(BASELINE_PIPELINES)]
    sweep = candidates['pipeline'].astype(str).str.extract(r'( \(.*\))$')[0].fillna('')
    candidates = candidates.assign(_sweep=sweep)
    numeric = [
        column
        for column in candidates.columns
        if column not in ('pipeline', 'masking', key, '_sweep') and pd.api.types.is_numeric_dtype(candidates[column])
    ]
    rows: List[Dict[str, Any]] = []
    for stratum in dict.fromkeys(table[key]):
        block = candidates[candidates[key] == stratum]
        modes = sorted(
            block['masking'].unique(), key=lambda m: (MASKING_ORDER.index(m) if m in MASKING_ORDER else 9, m)
        )
        for masking in modes:
            for suffix, group in block[block['masking'] == masking].groupby('_sweep', sort=False):
                rows.append(
                    {
                        'pipeline': f"{AVERAGE_PIPELINE}{suffix}",
                        'masking': masking,
                        key: stratum,
                        **{column: _mean(group[column]) for column in numeric},
                    }
                )
        rows.extend(baselines[baselines[key] == stratum].to_dict('records'))
    result = pd.DataFrame(rows)
    result.attrs['n_pipelines'] = int(candidates['pipeline'].astype(str).str.split('+', n=1).str[0].nunique())
    return result


def campaign_offset_bins(results_dir: Path, key: str = 'bins') -> Tuple[Tuple[int, Optional[int]], ...]:
    """The offset bins a campaign declares under `sampling.<key>` in its manifest (`plan_bins`), else the defaults."""
    return plan_bins(_manifest_sampling(results_dir), key)


def _manifest_sampling(results_dir: Path) -> Optional[Mapping[str, Any]]:
    """The `sampling` block of a campaign's config, read from its manifest; None when absent."""
    path = Path(results_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return (manifest.get('config') or {}).get('sampling')


def aggregate_by_scenario(
    results_dir: Optional[Path] = None,
    epsilon_floor: float = 0.0,
    frame: Optional[pd.DataFrame] = None,
    floors: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    Dynamic results grouped by pipeline and survey cell, recorded per measurement from `CASE.json`.

    `frame` (from `load_case_results`) replaces reading `results_dir`; a cell then pools its cases
    with equal weight. Raises when no measurement carries a cell (a case prepared without `--cell`).
    """
    if frame is None:
        frame = load_results(results_dir)
    if 'cell' not in frame or not frame['cell'].notna().any():
        raise ValueError("no measurement carries a survey cell; prepare the case with `--cell`")
    frame = frame[frame['cell'].notna()]

    rows: List[Dict[str, Any]] = []
    for (pipeline, masking, cell), group in frame.groupby(['pipeline', 'masking', 'cell'], sort=False, observed=True):
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'scenario': str(cell),
                'coverage_median': float(pd.to_numeric(group['coverage_ref'], errors='coerce').median()),
                **_breakdown_summary(group, epsilon_floor, floors),
            }
        )
    table = pd.DataFrame(rows)
    # Cells in the survey's order, not in the order the cases were passed.
    rank = {cell: index for index, cell in enumerate(CELL_ORDER)}
    table = table.sort_values('scenario', key=lambda cells: cells.map(rank).fillna(len(rank)), kind='stable')
    table = table.reset_index(drop=True)
    table.attrs['splits'] = {'source': 'survey cell'}
    return table


def aggregate_by_persistence(
    results_dir: Optional[Path] = None,
    persistence_split: Optional[float] = None,
    epsilon_floor: float = 0.0,
    frame: Optional[pd.DataFrame] = None,
    floors: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    Dynamic results split into moving and stationary foreground, per pipeline and masking mode.

    `persistence` is the overlap of the two frames' object footprints in reference coordinates,
    recorded per measurement. The split defaults to the median over these results. `frame` (from
    `load_case_results`) replaces reading `results_dir` and pools its cases with equal weight.
    """
    if frame is None:
        frame = load_results(results_dir)
    if 'persistence' not in frame:
        raise ValueError("Results do not contain 'persistence'; re-run the dynamic campaign with the current version")
    persistence = pd.to_numeric(frame['persistence'], errors='coerce')
    split = float(persistence.median()) if persistence_split is None else float(persistence_split)
    frame = frame.assign(
        stratum=[
            'stationary' if pd.notna(p) and p >= split else ('moving' if pd.notna(p) else 'unclassified')
            for p in persistence
        ]
    )
    rows: List[Dict[str, Any]] = []
    for (pipeline, masking, stratum), group in frame.groupby(['pipeline', 'masking', 'stratum'], sort=False):
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'stratum': str(stratum),
                'coverage_median': float(pd.to_numeric(group['coverage_ref'], errors='coerce').median()),
                'persistence_median': float(pd.to_numeric(group['persistence'], errors='coerce').median()),
                **_breakdown_summary(group, epsilon_floor, floors),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs['splits'] = {'persistence': split}
    return table


# --------------------------------------------------------------------------------------
# Several cases at once
# --------------------------------------------------------------------------------------


def load_case_results(results_dirs: Sequence[Path]) -> pd.DataFrame:
    """Every measurement of several result directories, tagged with the case each came from (`case_name`)."""
    frames = [load_results(Path(entry)).assign(case=case_name(entry)) for entry in results_dirs]
    return pd.concat(frames, ignore_index=True)


def case_name(results_dir: Path) -> str:
    """
    The case a dynamic results directory belongs to.

    The manifest's `metadata.case.dir` names it wherever the results were written (`--results-dir`);
    without one, a directory at `<case>/results/<name>` is its case's, and any other is its own path.
    """
    results_dir = Path(results_dir)
    try:
        with open(results_dir / MANIFEST_NAME, 'r', encoding='utf-8') as f:
            recorded = ((json.load(f).get('metadata') or {}).get('case') or {}).get('dir')
    except (OSError, json.JSONDecodeError, AttributeError):
        recorded = None
    if recorded:
        return Path(recorded).name
    return results_dir.parent.parent.name if results_dir.parent.name == 'results' else str(results_dir)


def case_order(results_dirs: Sequence[Path]) -> List[Path]:
    """
    Result directories in the order their cases are pooled, whatever order they were given in.

    The cluster bootstrap draws cases by position, so the pooled tables depend on this order, and a
    shell glob's order depends on the locale. Cases are compared field by field on the underscore-
    separated parts of their names, so `D1_PM1` precedes `D10_PM5`.
    """
    return sorted((Path(d) for d in results_dirs), key=lambda d: (case_name(d).split('_'), str(d)))


def aggregate_cases(
    results_dirs: Sequence[Path],
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    photometric: Sequence[str] = PHOTOMETRIC_COLUMNS,
    epsilon_floors: Optional[Mapping[str, float]] = None,
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """
    One row per pipeline and masking mode, pooled over cases with equal weight per case.

    Each summary column is computed per case and averaged across cases. `epsilon_floors` maps a case
    name to its audited residual, which sets that case's `maa_floor`. `n_cases` counts the cases a
    row was measured on.
    """
    frame = load_case_results(results_dirs)
    floors = dict(epsilon_floors or {})
    per_case: List[Dict[str, Any]] = []
    for (case, pipeline, masking), group in frame.groupby(['case', 'pipeline', 'masking'], sort=False):
        per_case.append(
            {
                'case': case,
                'pipeline': pipeline,
                'masking': masking,
                'detector': group['detector_name'].iloc[0] if 'detector_name' in group else None,
                **_group_summary(
                    group, hea_thresholds, photometric=photometric, maa_upper=maa_upper, maa_lower=floors.get(case, 0.0)
                ),
            }
        )
    table = pd.DataFrame(per_case)
    numeric = [c for c in table.columns if c not in ('case', 'pipeline', 'masking', 'detector')]
    rows: List[Dict[str, Any]] = []
    for (pipeline, masking), group in table.groupby(['pipeline', 'masking'], sort=False):
        row: Dict[str, Any] = {
            'pipeline': pipeline,
            'masking': masking,
            'detector': group['detector'].iloc[0],
            'n_cases': int(group['case'].nunique()),
        }
        for column in numeric:
            row[column] = _mean(group[column])
        row['n'] = int(pd.to_numeric(group['n'], errors='coerce').sum())
        rows.append(row)
    # A configuration infeasible in every case it was planned for is one N/A row; one measured in
    # some cases keeps its measured row, whose `n_cases` shows the shortfall.
    measured = {(row['pipeline'], row['masking']) for row in rows}
    aborted: Dict[Any, Dict[str, Any]] = {}
    for directory in results_dirs:
        for row in infeasible_rows(directory):
            if (row['pipeline'], row['masking']) not in measured:
                aborted.setdefault((row['pipeline'], row['masking']), {**row, 'n_cases': 0})
    pooled = pd.DataFrame([*rows, *aborted.values()])
    pooled.attrs['per_case'] = table
    return pooled


# --------------------------------------------------------------------------------------
# Tie sets: what a table may bold
# --------------------------------------------------------------------------------------


def cluster_table(
    frame: pd.DataFrame,
    column: str,
    group_columns: Sequence[str],
    cluster_column: str,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    photometric: Sequence[str] = PHOTOMETRIC_COLUMNS,
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """
    One summary value per (cluster, group) for one table column: rows clusters, columns groups.

    The value is the `_group_summary` statistic the table shows, on the cluster's measurements.
    """
    records: Dict[Any, Dict[Any, float]] = {}
    for keys, group in frame.groupby([cluster_column, *group_columns], sort=False):
        cluster, *label = keys if isinstance(keys, tuple) else (keys,)
        summary = _group_summary(group, hea_thresholds, photometric=photometric, maa_upper=maa_upper)
        records.setdefault(cluster, {})[tuple(label) if len(label) > 1 else label[0]] = summary.get(
            column, float('nan')
        )
    return pd.DataFrame.from_dict(records, orient='index')


def tie_sets_for(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    columns: Mapping[str, tuple],
    group_columns: Sequence[str],
    cluster_column: str,
    epsilon_floor: Optional[float] = None,
    annotation_floor: Optional[float] = None,
    exclude: Sequence[str] = BASELINE_PIPELINES,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    photometric: Sequence[str] = PHOTOMETRIC_COLUMNS,
) -> Dict[str, TieSet]:
    """
    The tie set of every highlighted column of a summary table.

    `summary` is the aggregated table and `frame` the per-measurement results it was built from. The
    best group per column is the one the summary ranks first; every other group is compared with it
    across clusters (`stats.tie_set`). Baseline rows are excluded.
    """
    if cluster_column not in frame.columns or frame[cluster_column].nunique(dropna=True) < 2:
        return {}
    candidates = frame[~frame['pipeline'].isin(list(exclude))] if 'pipeline' in frame else frame
    rows = summary
    if 'pipeline' in rows.columns:
        rows = rows[~rows['pipeline'].isin(list(exclude))]
    if 'status' in rows.columns:
        rows = rows[rows['status'].astype(str) != RUN_INFEASIBLE]

    def label_of(row: pd.Series) -> Any:
        values = tuple(row[name] for name in group_columns)
        return values if len(values) > 1 else values[0]

    result: Dict[str, TieSet] = {}
    for column, (_, _, direction) in columns.items():
        if direction is None or column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors='coerce')
        if not values.notna().any():
            continue
        best_row = rows.loc[values.idxmax() if direction else values.idxmin()]
        table = cluster_table(candidates, column, group_columns, cluster_column, hea_thresholds, photometric)
        best = label_of(best_row)
        if best not in table.columns:
            continue
        result[column] = tie_set(
            table, best, bool(direction), margin=margin_for(column, epsilon_floor, annotation_floor)
        )
    return result


def comparisons_frame(
    tie_sets: Mapping[str, TieSet],
    columns: Mapping[str, tuple],
) -> pd.DataFrame:
    """
    The analysis plan's numbers, one row per (outcome, group) comparison with the best group.

    Effect size, cluster-bootstrap interval, exact sign-flip p, and `p_holm`, the p adjusted by
    `stats.holm_adjust` within one outcome.
    """
    from stabilo_benchmark.stats import holm_adjust  # noqa: PLC0415 - keeps the import graph flat

    rows: List[Dict[str, Any]] = []
    for column, ties in tie_sets.items():
        header = columns.get(column, (column, '{:.3f}', None))[0]
        block: List[Dict[str, Any]] = []
        for group, comparison in ties.comparisons.items():
            block.append(
                {
                    'column': column,
                    'outcome': header,
                    'group': _group_label(group),
                    'estimate': comparison.estimate,
                    'ci_low': comparison.ci_low,
                    'ci_high': comparison.ci_high,
                    'p_value': comparison.p_value,
                    'margin': comparison.margin,
                    'distinguishable': comparison.distinguishable,
                    'reference': False,
                }
            )
        if not block:
            continue
        for row, adjusted in zip(block, holm_adjust([r['p_value'] for r in block]), strict=True):
            row['p_holm'] = adjusted
        block.sort(key=lambda r: (not r['distinguishable'], r['p_value'], r['group']))
        rows.append(
            {'column': column, 'outcome': header, 'group': f"best: {_group_label(ties.best)}", 'reference': True}
        )
        rows.extend(block)
    return pd.DataFrame(rows)


def _group_label(group: Any) -> str:
    """A pipeline/masking key as one cell: `sift / obb`, the pipeline without its masking suffix."""
    if isinstance(group, tuple) and len(group) == 2:
        pipeline, masking = (str(part) for part in group)
        return f"{_without_masking(pipeline, masking)} / {masking}"
    if isinstance(group, tuple):
        return ' / '.join(str(part) for part in group)
    return str(group)


def _without_masking(pipeline: str, masking: str) -> str:
    """`sift+obb` -> `sift`; `sift+obb (margin 0.15)` -> `sift (margin 0.15)`."""
    suffix = f"+{masking}"
    if pipeline.endswith(suffix):
        return pipeline[: -len(suffix)]
    return pipeline.replace(f"{suffix} (", ' (', 1)


def _display_group(text: str) -> str:
    """A `_group_label` cell in display names: `best: sift / obb` -> `best: SIFT / OBB`."""
    prefix, _, label = text.rpartition('best: ')
    parts = label.split(' / ')
    if len(parts) == 2:
        label = f"{display_label(parts[0])} / {masking_name(parts[1])}"
    return f"{prefix}{'best: ' if _ else ''}{label}"


def write_comparisons_table(
    frame: pd.DataFrame,
    path: Path,
    caption: str = '',
    label: str = '',
) -> Path:
    """
    Render `comparisons_frame` as a booktabs table; nothing is bolded. A table of one outcome has no
    Outcome column (its caption names the outcome).
    """
    single = frame['outcome'].nunique() == 1
    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        f"  \\caption{{{caption}}}" if caption else '',
        f"  \\label{{{label}}}" if label else '',
        r'  \small',
        r'  \begin{tabular}{lrrrc}' if single else r'  \begin{tabular}{llrrrc}',
        r'    \toprule',
        ('' if single else r'    Outcome & ') + r'Comparison & $\Delta$ & 95\% CI & $p_{\mathrm{Holm}}$ & Tied \\',
        r'    \midrule',
    ]
    previous = None
    for _, row in frame.iterrows():
        outcome = str(row['outcome'])
        shown = '' if outcome == previous else outcome
        if shown and previous is not None:
            lines.append(r'    \midrule')
        previous = outcome
        lead = '    ' if single else f"    {shown} & "
        if bool(row.get('reference', False)):
            lines.append(f"{lead}{_escape(_display_group(str(row['group'])))} & -- & -- & -- & \\\\")
            continue
        interval = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        tied = r"\cmark" if not bool(row['distinguishable']) else ''
        lines.append(
            f"{lead}{_escape(_display_group(str(row['group'])))} & {row['estimate']:.3f} & "
            f"{interval} & {row['p_holm']:.4f} & {tied} \\\\"
        )
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(line for line in lines if line != ''), encoding='utf-8')
    return path


def static_cluster_column(frame: pd.DataFrame) -> pd.DataFrame:
    """A Part I frame with its `site` cluster column derived from the scene names."""
    if 'scene' not in frame.columns:
        return frame
    return frame.assign(site=frame['scene'].map(site_of))


# --------------------------------------------------------------------------------------
# Per-pipeline hyperparameter tuning (configs/songdo/static/tuning.yaml)
# --------------------------------------------------------------------------------------

# The two parameters `configs/songdo/static/tuning.yaml` sweeps.
TUNED_PARAMETERS = ('filter_ratio', 'ransac_epipolar_threshold')

# The untuned starting values, pinned by the campaign that evaluates them; present in a repository checkout.
PRETUNING_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'songdo' / 'static' / 'detectors_default.yaml'

TUNING_COLUMNS: Dict[str, tuple] = {
    'filter_ratio': ('$r$', '{:.2f}', None),
    'ransac_epipolar_threshold': (r'$\theta$ [px]', '{:g}', None),
    'maa': ('mAA', '{:.4f}', True),
    'maa_default': (r'mAA$_{\mathrm{default}}$', '{:.4f}', None),
    'maa_gain': (r'$\Delta$mAA', '{:+.4f}', None),
    'cte_median': (r'CTE$_{50}$ [px]', '{:.3f}', False),
    'failure_rate': ('Fail', '{:.1%}', False),
    'n_configurations': ('Grid', '{:.0f}', None),
    'n_within_margin': ('Tied', '{:.0f}', None),
}


# Suffix of the flag `aggregate_tuning` sets when a tuned value lies at an end of its swept range.
EDGE_SUFFIX = '_at_edge'
EDGE_NOTE = (
    '$^{\\dagger}$ marks a selected value at an end of its swept range, where the optimum may lie outside the grid.'
)
# Suffix of the flag set on `maa` when the incumbent values are kept (`aggregate_tuning`).
KEPT_SUFFIX = '_kept'
KEPT_NOTE = '$^{\\S}$ marks incumbent values kept because no grid point beats them by more than the margin.'
# Suffix of the flag set when the reference value lies off the grid and its nearest point stands in.
OFF_GRID_SUFFIX = '_off_grid'
OFF_GRID_NOTE = (
    '$^{\\ddagger}$ marks a reference value off the grid, read at its nearest grid point (the better of '
    'two equidistant ones), so the gain beside it is a lower bound.'
)
# Per-cell marks of the tuning table: flag suffix -> (symbol, caption note).
TUNING_MARKS: Dict[str, Tuple[str, str]] = {
    EDGE_SUFFIX: ('$^{\\dagger}$', EDGE_NOTE),
    OFF_GRID_SUFFIX: ('$^{\\ddagger}$', OFF_GRID_NOTE),
    KEPT_SUFFIX: ('$^{\\S}$', KEPT_NOTE),
}


def tuning_edges(tuned: pd.DataFrame) -> List[str]:
    """
    `pipeline: parameter=value` for every tuned value at an end of its swept range. A kept incumbent
    is never flagged: no grid point beats it by the margin, so nothing indicates an optimum past the edge.
    """
    edges = []
    for _, row in tuned.iterrows():
        for name in TUNED_PARAMETERS:
            if bool(row.get(f"{name}{EDGE_SUFFIX}", False)):
                edges.append(f"{row['pipeline']}: {name}={row[name]:g}")
    return edges


def is_tuning_campaign(frame: pd.DataFrame) -> bool:
    """
    Whether a results frame came from a per-pipeline tuning sweep.

    Both tuned parameters vary within one pipeline and `ransac_method` does not, which excludes
    `estimators.yaml`. Variation across pipelines alone is per-preset values, not a sweep.
    """
    if any(name not in frame.columns for name in TUNED_PARAMETERS):
        return False
    if 'pipeline' in frame.columns:
        groups = frame.groupby(frame['pipeline'].astype(str).str.replace(r'#\d+$', '', regex=True))
        counts = groups[list(TUNED_PARAMETERS)].nunique(dropna=True)
    else:
        counts = frame[list(TUNED_PARAMETERS)].nunique(dropna=True).to_frame().T
    if not (counts > 1).all(axis=1).any():
        return False
    return 'ransac_method' not in frame.columns or frame['ransac_method'].nunique(dropna=True) <= 1


def tuning_grid(
    frame: pd.DataFrame,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    maa_upper: float = MAA_UPPER_PX,
) -> pd.DataFrame:
    """
    One row per pipeline and grid point of a tuning campaign, with its summary (mAA among it).

    A pipeline that never applies the ratio test has one grid point per inlier threshold, with
    `filter_ratio` NaN. Several cases (Part II) are summarized per case and averaged, so each case
    weighs equally.
    """
    missing = [name for name in TUNED_PARAMETERS if name not in frame.columns]
    if missing:
        raise ValueError(f"Results do not contain {missing}; this is a tuning-campaign table only")
    # Grid points of one preset carry repeat suffixes (`sift+none#2`); group on the bare label.
    frame = frame.assign(pipeline=frame['pipeline'].astype(str).str.replace(r'#\d+$', '', regex=True))
    # LightGlue and LoFTR never apply the ratio test, so their ratio points are one configuration.
    frame['filter_ratio'] = frame['filter_ratio'].astype(float).where(~_ignores_ratio_test(frame))
    by_case = 'case' in frame.columns and frame['case'].nunique() > 1

    grid: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(['pipeline', *TUNED_PARAMETERS], sort=False, dropna=False):
        pipeline, ratio, threshold = keys
        if by_case:
            summary = _mean_of_cases(group, hea_thresholds, maa_upper)
        else:
            summary = _group_summary(group, hea_thresholds, photometric=(), maa_upper=maa_upper)
        grid.append(
            {
                'pipeline': pipeline,
                'filter_ratio': float(ratio),
                'ransac_epipolar_threshold': float(threshold),
                **summary,
            }
        )
    return pd.DataFrame(grid)


def aggregate_tuning(
    results_dir: Optional[Path] = None,
    hea_thresholds: Sequence[float] = (1.0, 3.0, 5.0),
    maa_upper: float = MAA_UPPER_PX,
    defaults: Optional[Mapping[str, Mapping[str, Any]]] = None,
    frame: Optional[pd.DataFrame] = None,
    incumbent: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """
    The selected hyperparameters per pipeline, and the gain over a reference setting.

    One row per pipeline: the grid point maximizing mAA, and the mAA at the pipeline's reference
    values. With `incumbent` (preset name -> values), a pipeline whose incumbent values are on the
    grid keeps them unless the best point beats them by more than the analysis plan's mAA margin;
    `maa_kept` flags the rows that do. `maa_gain` is never bolded. `defaults` maps a preset name to
    the reference values and falls back to `pretuning_defaults()`; a preset it does not list is
    compared at its values in the preset file. `frame` replaces reading `results_dir`; when it
    carries several cases (Part II), every summary is the mean of the per-case summaries, so each
    case weighs equally.
    """
    if frame is None:
        frame = load_results(results_dir)
    if defaults is None:
        defaults = pretuning_defaults()
    swept = tuning_grid(frame, hea_thresholds=hea_thresholds, maa_upper=maa_upper)

    rows: List[Dict[str, Any]] = []
    for pipeline, group in swept.groupby('pipeline', sort=False):
        scored = group.dropna(subset=['maa'])
        if scored.empty:
            continue
        best = scored.loc[scored['maa'].idxmax()].to_dict()
        # Flatness of the optimum: grid points within the analysis plan's mAA margin of the best one.
        n_within_margin = int((scored['maa'] >= best['maa'] - margin_for('maa')).sum())
        preset = str(pipeline).rsplit('+', 1)[0]
        kept = False
        if incumbent is not None:
            point = _grid_point(scored, incumbent.get(preset, {}))
            if point is not None and best['maa'] - point['maa'] <= margin_for('maa'):
                best, kept = point.to_dict(), True
        best['n_configurations'] = int(len(group))
        best['n_within_margin'] = n_within_margin
        best[f"maa{KEPT_SUFFIX}"] = kept
        reference = defaults.get(preset)
        if reference is None:
            reference = inherited_tuned_values().get(preset, {})
        # A winner at the end of its swept range may have its optimum outside the grid.
        for name in TUNED_PARAMETERS:
            swept_values = scored[name].dropna().unique()
            best[f"{name}{EDGE_SUFFIX}"] = bool(
                not kept
                and swept_values.size > 1
                and np.isfinite(best[name])
                and (np.isclose(best[name], swept_values.min()) or np.isclose(best[name], swept_values.max()))
            )
        best['maa_default'], on_grid = _default_maa(group, reference)
        best[f"maa_default{OFF_GRID_SUFFIX}"] = bool(np.isfinite(best['maa_default']) and not on_grid)
        best['maa_gain'] = (
            float(best['maa'] - best['maa_default']) if np.isfinite(best['maa_default']) else float('nan')
        )
        rows.append(best)

    return pd.DataFrame(rows)


def _mean_of_cases(
    group: pd.DataFrame,
    hea_thresholds: Sequence[float],
    maa_upper: float,
    photometric: Sequence[str] = (),
    floors: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """`_group_summary` per case, averaged over the cases; `n` is the total, `n_cases` the count."""
    floors = dict(floors or {})
    per_case = [
        _group_summary(
            rows, hea_thresholds, photometric=photometric, maa_upper=maa_upper, maa_lower=floors.get(case, 0.0)
        )
        # Sorted, so the mean does not depend on the order the cases were passed in.
        for case, rows in group.groupby('case', sort=True)
    ]
    summary: Dict[str, Any] = {}
    for key in per_case[0]:
        values = [entry.get(key) for entry in per_case]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            numbers = np.asarray(values, dtype=float)
            finite = numbers[np.isfinite(numbers)]
            summary[key] = int(numbers.sum()) if key == 'n' else (float(finite.mean()) if finite.size else float('nan'))
        else:
            summary[key] = values[0]
    summary['n_cases'] = len(per_case)
    return summary


def _breakdown_summary(
    group: pd.DataFrame, epsilon_floor: float = 0.0, floors: Optional[Mapping[str, float]] = None
) -> Dict[str, Any]:
    """One breakdown cell: pooled with equal weight per case when the frame carries cases."""
    if 'case' in group.columns:
        return _mean_of_cases(group, (1.0, 3.0, 5.0), MAA_UPPER_PX, photometric=PHOTOMETRIC_COLUMNS, floors=floors)
    return _group_summary(group, (1.0, 3.0, 5.0), photometric=PHOTOMETRIC_COLUMNS, maa_lower=epsilon_floor)


def inherited_tuned_values() -> Dict[str, Dict[str, float]]:
    """
    For each preset that `extends` another, the parent's `filter_ratio` and
    `ransac_epipolar_threshold`: what the preset would run at without values of its own (the
    reference estimator, before its own tuning, runs at the sift preset's).
    """
    import yaml  # noqa: PLC0415 - only the report reads the raw preset file

    from stabilo_benchmark.pipelines import DEFAULT_PRESET_FILE  # noqa: PLC0415 - avoids import cycle

    raw = yaml.safe_load(Path(DEFAULT_PRESET_FILE).read_text(encoding='utf-8')) or {}
    resolved = preset_tuned_values()
    return {
        name: resolved[str(entry['extends'])]
        for name, entry in raw.items()
        if isinstance(entry, dict) and str(entry.get('extends')) in resolved
    }


def preset_tuned_values() -> Dict[str, Dict[str, float]]:
    """Each preset's `filter_ratio` and `ransac_epipolar_threshold` in the preset file, as it stands."""
    from stabilo_benchmark.pipelines import load_presets  # noqa: PLC0415 - avoids import cycle

    return {
        name: {key: float(values[key]) for key in TUNED_PARAMETERS}
        for name, values in load_presets().items()
        if all(key in values for key in TUNED_PARAMETERS)
    }


def tuned_value_mappings(tuned: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    The selected values of a tuning table as per-detector mappings, ready for a config's `pipeline`.

    Keyed by each preset's `detector_name`, in preset-file order. A parameter the pipeline does not
    use (the ratio of LightGlue and LoFTR) keeps its preset value, so every detector has an entry.
    """
    from stabilo_benchmark.pipelines import load_presets  # noqa: PLC0415 - avoids import cycle

    presets = load_presets()
    selected = {str(row['pipeline']).rsplit('+', 1)[0]: row for _, row in tuned.iterrows()}
    mappings: Dict[str, Dict[str, float]] = {name: {} for name in TUNED_PARAMETERS}
    for preset, values in presets.items():
        if preset not in selected:
            continue
        detector = str(values['detector_name'])
        for name in TUNED_PARAMETERS:
            value = float(selected[preset][name])
            mappings[name][detector] = value if np.isfinite(value) else float(values[name])
    return mappings


def write_tuned_mappings(tuned: pd.DataFrame, path: Path, header: str = '') -> Path:
    """`tuned_value_mappings` as YAML, one block per parameter."""
    lines = [f"# {line}" if line else '#' for line in header.splitlines()]
    for name, mapping in tuned_value_mappings(tuned).items():
        lines.append(f"{name}:")
        lines += [f"  {detector}: {value:g}" for detector, value in mapping.items()]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def pretuning_defaults(path: Path = PRETUNING_CONFIG) -> Dict[str, Dict[str, float]]:
    """
    Each preset's untuned starting `filter_ratio` and `ransac_epipolar_threshold`, read from `path`.

    Empty when the config is absent, which leaves the default and gain columns blank.
    """
    from stabilo_benchmark.config import build_pipelines, load_config  # noqa: PLC0415 - avoids import cycle

    if not Path(path).exists():
        return {}
    return {
        spec.name.rsplit('+', 1)[0]: {name: float(spec.params[name]) for name in TUNED_PARAMETERS}
        for spec in build_pipelines(load_config(Path(path)))
    }


def _ignores_ratio_test(frame: pd.DataFrame) -> pd.Series:
    """Rows whose matcher does not use `filter_ratio`, as `pipelines.ignores_ratio_test` decides for a spec."""
    ignores = pd.Series(False, index=frame.index)
    if 'filter_type' in frame:
        ignores |= frame['filter_type'].astype(str).eq('none')
    if 'matcher_name' in frame:
        ignores |= frame['matcher_name'].astype(str).eq('lightglue')
    if 'detector_name' in frame:
        ignores |= frame['detector_name'].astype(str).eq('loftr')
    return ignores


def _grid_point(scored: pd.DataFrame, values: Mapping[str, Any]) -> Optional[pd.Series]:
    """The scored grid point at exactly `values` (a parameter the pipeline ignores matches any), else None."""
    selected = scored
    for name in TUNED_PARAMETERS:
        if scored[name].isna().all():
            continue
        if name not in values:
            return None
        selected = selected[np.isclose(selected[name].astype(float), float(values[name]))]
    return selected.iloc[0] if len(selected) else None


def _default_maa(group: pd.DataFrame, preset: Mapping[str, Any]) -> Tuple[float, bool]:
    """
    mAA at a pipeline's reference values, and whether that point is on the grid.

    An off-grid value is read at its nearest swept value (the better of two equidistant ones), and
    rejected (NaN) when that value is not close.
    """
    selected = group
    on_grid = True
    for name in TUNED_PARAMETERS:
        if group[name].isna().all():
            continue
        if name not in preset:
            return float('nan'), False
        target = float(preset[name])
        offsets = (selected[name].astype(float) - target).abs()
        if offsets.min() > 0.5 * max(abs(target), 1.0):
            return float('nan'), False
        on_grid = on_grid and bool(np.isclose(offsets.min(), 0.0))
        selected = selected[offsets == offsets.min()]
    values = pd.to_numeric(selected['maa'], errors='coerce').to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return (float(np.max(finite)) if finite.size else float('nan')), on_grid


# --------------------------------------------------------------------------------------
# LaTeX rendering
# --------------------------------------------------------------------------------------


# From this many columns a table is set in \small with narrower column gaps, to fit the text width;
# every table shrinks to the text width when it would still overflow (long case names do).
WIDE_TABLE_COLUMNS = 8
# From this many rows a table is set in \footnotesize, to fit one page (a breakdown over twelve offset bins).
LONG_TABLE_ROWS = 40
# A wide table shrinks to the text width only when it would overflow.
ADJUSTBOX_OPEN = r'  \begin{adjustbox}{max width=\textwidth}'

# Accuracy columns that a table drops, when asked to, once every row reaches SATURATED_ACCURACY: an
# HEA at that level separates no two configurations, and mAA integrates the same curve.
SATURATING_COLUMNS = ('hea_1px', 'hea_2px', 'hea_3px', 'hea_5px', 'hea_10px')
SATURATED_ACCURACY = 0.995
# Columns that a table drops, when asked to, once their values span at most this much.
NARROW_SPREADS = {'region_fraction': 0.01}
# Caption sentences for the columns a table may drop (`write_latex_table(omit=...)`); {value} is the
# value every row shows, or the range it spans.
NO_FAILURE_NOTE = 'No configuration failed on any {unit}.'
OMIT_NOTES = {
    'failure_rate': '',
    'region_fraction': 'The reference-free columns are computed over a median {value} of the frame.',
    'n_cases': 'Every row pools {value} cases.',
    **{name: '' for name in SATURATING_COLUMNS},
}


def _join_names(names: Sequence[str]) -> str:
    if len(names) <= 2:
        return ' and '.join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def omitted_columns(
    table: pd.DataFrame,
    columns: Mapping[str, tuple],
    present: Sequence[str],
    shown: pd.Series,
    omit: Mapping[str, str],
    unit: str = 'row',
) -> Tuple[List[str], str]:
    """
    The columns of `present` a table keeps, and the caption sentences for those it drops.

    A column of `omit` drops when it carries no information over the rows `shown`: an accuracy of
    SATURATING_COLUMNS once every value reaches SATURATED_ACCURACY (one sentence for all of them), the
    failure rate when no row failed, and any other column when every value renders the same or, for
    NARROW_SPREADS, spans at most its spread. `omit` maps each such column to its sentence, with
    {value} for the shared value; a column with no finite value is kept.
    """
    kept: List[str] = []
    notes: List[str] = []
    saturated: List[Tuple[str, float]] = []
    for name in present:
        values = pd.to_numeric(table.loc[shown, name], errors='coerce').to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if name not in omit or not values.size:
            kept.append(name)
            continue
        fmt = columns[name][1]
        rendered = sorted({fmt.format(value) for value in values})
        if name in SATURATING_COLUMNS and values.min() >= SATURATED_ACCURACY:
            saturated.append((columns[name][0], float(values.min())))
        elif name == 'failure_rate' and not values.any():
            notes.append(NO_FAILURE_NOTE.format(unit=unit))
        elif len(rendered) == 1 and omit[name]:
            notes.append(omit[name].replace('{value}', rendered[0].replace('%', r'\%')))
        elif name in NARROW_SPREADS and values.max() - values.min() <= NARROW_SPREADS[name] and omit[name]:
            span = f"{fmt.format(values.min())} to {fmt.format(values.max())}".replace('%', r'\%')
            notes.append(omit[name].replace('{value}', span))
        else:
            kept.append(name)
    if saturated:
        headers = [header for header, _ in saturated]
        lowest = min(value for _, value in saturated)
        verb = 'is' if len(headers) == 1 else 'are'
        notes.insert(0, f"{_join_names(headers)} {verb} at least {lowest:.3f} in every row.")
    return kept, ' '.join(notes)


def _constant_masking(table: pd.DataFrame, labels: Sequence[str]) -> Optional[str]:
    """The masking suffix every pipeline label shares when no Masking column shows it, else None."""
    if 'pipeline' not in labels or 'masking' in labels or 'pipeline' not in table:
        return None
    matches = table['pipeline'].astype(str).map(_MASKING_SUFFIX.search)
    suffixes = {match.group(1) if match else None for match in matches}
    return suffixes.pop() if len(suffixes) == 1 and None not in suffixes else None


_MASKING_SUFFIX = re.compile(r'\+(none|hbb|obb)(?=$|#| \()')


def _label_text(row: pd.Series, name: str, labels: Sequence[str], constant_masking: Optional[str] = None) -> str:
    """
    A label cell in display names (`names.py`); the pipeline drops its `+<masking>` suffix when a
    Masking column shows it, and a baseline row shows no masking, since it is not a configuration of
    the masking axis. A masking suffix shared by every row (`constant_masking`) is dropped too; the
    caption states it.
    """
    text = str(row[name])
    if name == 'pipeline':
        if 'masking' in labels:
            text = _without_masking(text, str(row['masking']))
        elif constant_masking is not None:
            text = _without_masking(text, constant_masking)
        text = display_label(text)
    elif name == 'masking':
        text = '--' if str(row.get('pipeline')) in BASELINE_PIPELINES else masking_name(text)
    elif name == 'detector':
        text = pipeline_name(text)
    return text


def _repeated_aborted(table: pd.DataFrame, aborted: pd.Series, labels: Sequence[str]) -> Dict[Any, List[Any]]:
    """
    The rows of a pipeline aborted under every masking mode, which differ in nothing but the mode:
    the first row's index mapped to the others'. Empty without a Masking column.
    """
    if 'masking' not in labels or 'pipeline' not in labels:
        return {}
    names = table.apply(lambda row: _without_masking(str(row['pipeline']), str(row['masking'])), axis=1)
    groups = table.index.to_series().groupby(names, sort=False).agg(list)
    return {rows[0]: rows[1:] for rows in groups if len(rows) > 1 and all(bool(aborted.loc[index]) for index in rows)}


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    columns: Optional[Mapping[str, tuple]] = None,
    label_columns: Sequence[str] = ('pipeline', 'masking'),
    caption: str = '',
    label: str = '',
    sort_by: Optional[str] = None,
    highlight_best: bool = True,
    tie_sets: Optional[Mapping[str, TieSet]] = None,
    group_columns: Sequence[str] = ('pipeline', 'masking'),
    marks: Optional[Mapping[str, Tuple[str, str]]] = None,
    omit: Optional[Mapping[str, str]] = None,
    unit: str = 'row',
    csv: bool = False,
) -> Path:
    """
    Render an aggregated frame as a booktabs table (`booktabs`; a wide one also needs `adjustbox`,
    which shrinks it to the text width only when it would overflow).

    With `tie_sets` (`tie_sets_for`), each column bolds its tie set, and a column without one bolds
    nothing; an empty mapping bolds nothing. Without `tie_sets`, `highlight_best` bolds the best
    value per column. Columns whose direction is None are never bolded.

    A row whose `status` is `infeasible` reads N/A across every column, and the caption says what N/A
    means; a pipeline with such a row under every masking mode prints one, its Masking cell `all`.
    `--` marks a quantity the row does not have. `marks` maps a flag suffix to a symbol and a
    caption note: a cell whose `<column><suffix>` flag is set carries the symbol, and the note is
    appended to the caption when any cell does. The columns of `omit` (`OMIT_NOTES`) drop when they
    carry no information, and the caption says what they would have shown (`omitted_columns`; `unit`
    names what a failure is counted over).

    With `csv`, the frame is also written unformatted, every column and in table order, to the same
    path with suffix `.csv`.
    """
    columns = dict(columns or DYNAMIC_COLUMNS)
    present = [name for name in columns if name in frame.columns]
    if not present:
        raise ValueError(f"None of the requested columns are present. Available: {sorted(frame.columns)}")

    table = frame.copy()
    if sort_by and sort_by in table.columns:
        ascending = not (columns.get(sort_by, (None, None, False))[2] or False)
        table = table.sort_values(sort_by, ascending=ascending)
    if csv:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path.with_suffix('.csv'), index=False)

    status = (
        table['status'].astype(str)
        if 'status' in table.columns
        else pd.Series(
            [
                '',
            ]
            * len(table),
            index=table.index,
        )
    )
    aborted = status.eq(RUN_INFEASIBLE)
    labels = [name for name in label_columns if name in table.columns]
    merged = _repeated_aborted(table, aborted, labels)
    repeated = {index for rows in merged.values() for index in rows}
    if omit:
        present, omitted = omitted_columns(table, columns, present, ~aborted, omit, unit)
        if omitted:
            caption = (caption + ' ' if caption else '') + omitted
    if aborted.any():
        caption = (caption + ' ' if caption else '') + (
            f"N/A marks {'a configuration' if int(aborted.sum()) == 1 else 'configurations'} that could not be run on "
            'the benchmark machine: memory was exhausted before any measurement was taken, so the run was aborted '
            'rather than averaged over the subset of the data that happened to fit.'
        )

    marks = dict(marks or {})
    for suffix, (_, note) in marks.items():
        flags = [f"{name}{suffix}" for name in present if f"{name}{suffix}" in table.columns]
        if any(table[flag].fillna(False).astype(bool).any() for flag in flags):
            caption = (caption + ' ' if caption else '') + note

    best: Dict[str, Optional[float]] = {}
    for name in present:
        direction = columns[name][2]
        if tie_sets is not None or not highlight_best or direction is None:
            best[name] = None
            continue
        values = pd.to_numeric(table[name], errors='coerce').to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        best[name] = (float(np.max(finite)) if direction else float(np.min(finite))) if finite.size else None

    def row_label(row: pd.Series) -> Any:
        values = tuple(row[name] for name in group_columns if name in table.columns)
        return values if len(values) > 1 else (values[0] if values else None)

    header = [_escape(name.replace('_', ' ').title()) for name in labels]
    header += [columns[name][0] for name in present]
    alignment = 'l' * len(labels) + 'r' * len(present)
    wide = len(labels) + len(present) >= WIDE_TABLE_COLUMNS
    long = len(table) >= LONG_TABLE_ROWS
    size = r'  \footnotesize' if long else r'  \small' if wide else ''

    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        f"  \\caption{{{caption}}}" if caption else '',
        f"  \\label{{{label}}}" if label else '',
        *([size] if size else []),
        *([r'  \setlength{\tabcolsep}{4pt}'] if wide else []),
        ADJUSTBOX_OPEN,
        f"  \\begin{{tabular}}{{{alignment}}}",
        r'    \toprule',
        '    ' + ' & '.join(header) + r' \\',
        r'    \midrule',
    ]

    constant = _constant_masking(table, labels)
    for index, row in table.iterrows():
        if index in repeated:
            continue
        cells = [_escape(_label_text(row, name, labels, constant)) for name in labels]
        if index in merged:
            cells[labels.index('masking')] = 'all'
        if bool(aborted.loc[index]):
            lines.append('    ' + ' & '.join([*cells, *([r'\textit{N/A}'] * len(present))]) + r' \\')
            continue
        for name in present:
            value = pd.to_numeric(pd.Series([row[name]]), errors='coerce').iloc[0]
            if value is None or not np.isfinite(value):
                cells.append('--')
                continue
            text = columns[name][1].format(value).replace('%', r'\%')
            if best[name] is not None and np.isclose(value, best[name]):
                text = rf'\textbf{{{text}}}'
            elif tie_sets is not None and name in tie_sets and row_label(row) in tie_sets[name].bold:
                text = rf'\textbf{{{text}}}'
            text += ''.join(symbol for suffix, (symbol, _) in marks.items() if bool(row.get(f"{name}{suffix}", False)))
            cells.append(text)
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines += [r'    \bottomrule', r'  \end{tabular}', r'  \end{adjustbox}', r'\end{table}', '']

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(line for line in lines if line != ''), encoding='utf-8')
    return path


def _escape(text: str) -> str:
    """Escape the LaTeX special characters that appear in pipeline names."""
    for character, replacement in (
        ('\\', r'\textbackslash{}'),
        ('_', r'\_'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('#', r'\#'),
        ('$', r'\$'),
    ):
        text = text.replace(character, replacement)
    return text
