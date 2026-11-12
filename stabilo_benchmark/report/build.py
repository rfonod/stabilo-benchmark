#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
build.py - Turn results directories into the manuscript's tables and figures.

Called by `stabilo-benchmark report`. The protocol is read from the results manifest. A table or
figure that does not apply to a directory is skipped with a stated reason.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from stabilo_benchmark.gcp import GcpError
from stabilo_benchmark.oracle import OracleError
from stabilo_benchmark.report.audit_figure import AuditFigureError, figure_gcp_audit
from stabilo_benchmark.report.case_figure import CaseFigureError, figure_cases
from stabilo_benchmark.report.figures import (
    figure_accuracy_vs_compute,
    figure_error_vs_offset,
    figure_estimator_thresholds,
    figure_inlier_vs_quality,
    figure_masking_comparison,
    figure_motion_alignment,
    figure_regime_summary,
    figure_sweep,
    figure_tuning_surface,
)
from stabilo_benchmark.report.names import display_label, pipeline_name
from stabilo_benchmark.report.scenes import (
    SCENE_CAPTION,
    SCENE_COLUMNS,
    characterize_scenes,
    summarize_scenes,
)
from stabilo_benchmark.report.survey_cases import build_survey_report, is_survey_file
from stabilo_benchmark.report.tables import (
    BASELINE_IDENTITY,
    BASELINE_PIPELINES,
    BREAKDOWN_COLUMNS,
    DYNAMIC_COLUMNS,
    ESTIMATOR_COLUMNS,
    GCP_COLUMNS,
    GCP_PROPOSAL_COLUMNS,
    MAA_UPPER_PX,
    OMIT_NOTES,
    SATURATING_COLUMNS,
    STATIC_COLUMNS,
    TUNING_COLUMNS,
    TUNING_MARKS,
    aggregate_budget,
    aggregate_by_offset,
    aggregate_by_persistence,
    aggregate_by_scenario,
    aggregate_cases,
    aggregate_dynamic,
    aggregate_estimators,
    aggregate_gcp_audit,
    aggregate_gcp_proposals,
    aggregate_static,
    aggregate_sweep,
    aggregate_tuning,
    average_pipeline_rows,
    campaign_offset_bins,
    case_name,
    case_order,
    columns_for_object_boxes,
    estimator_overview,
    gcp_audit_columns,
    is_budget_campaign,
    is_estimator_campaign,
    is_tuning_campaign,
    load_case_results,
    load_results,
    object_boxes_of,
    preset_tuned_values,
    pretuning_defaults,
    static_cluster_column,
    swept_run_labels,
    tie_sets_for,
    tuning_edges,
    tuning_grid,
    write_latex_table,
    write_tuned_mappings,
)
from stabilo_benchmark.results import GCP_SCORES_NAME, MANIFEST_NAME, ResultStore, infeasible_runs
from stabilo_benchmark.stats import MAA_MARGIN, minimum_p

if TYPE_CHECKING:
    import pandas as pd

AUDIT_FILE = 'AUDIT.json'
# Where `report` writes tables/ and figures/ without --output, relative to the working directory.
DEFAULT_OUTPUT = Path('reports')

STATIC_CAPTION = (
    'Part I (static benchmark): accuracy under known homographies and photometric degradation. mAA '
    'integrates HEA$_{\\varepsilon}$, the share of trials whose corner transfer error (CTE) stays within '
    '$\\varepsilon$, over $\\varepsilon \\in [0, <upper>]$ px, a failed estimate counting as a miss; IoU and '
    'OCE (object center error) compare each object\'s box with its round trip through the known homography '
    'and the estimate. <bold_note>'
)
BOLD_NOTE_PLAN = (
    'Bold marks the best value of each column and every value not distinguishable from it: a paired '
    'difference to the best, over the <n_clusters> <cluster>s, whose 95\\% cluster-bootstrap interval '
    'includes zero or whose estimate is below the equivalence margin (<margins>).'
)
# The margins clause of BOLD_NOTE_PLAN for the tables ranked on mAA; GCP_MARGINS for the GCP table.
MAA_MARGINS = f"{MAA_MARGIN:g} for mAA<pixel_margin>"
GCP_MARGINS = '<pixel_margin> for the pixel columns, none for IoU'
BOLD_NOTE_NONE = (
    'Nothing is bolded: this table is built from a single case, which is the unit of inference, and one '
    'unit supports no claim of a difference.'
)
DYNAMIC_CAPTION = (
    'Part II (dynamic benchmark): every candidate against the reference estimator. mAA, integrated over '
    '$[0, <upper>]$ px, and CTE are reference-relative; <floor_note> SSIM, ZNCC, MI and GOS are '
    'reference-free, over the static background; IoU and OCE compare the current frame\'s boxes as placed '
    'by the candidate and by the reference estimator. Align (0.5 for isotropic error) and the inlier ratio '
    'are diagnostics, never bolded. Identity is the unstabilized frame, also what a failed pipeline '
    'delivers.<reference_note> <bold_note>'
)
REFERENCE_NOTE = (
    ' The <rows> rows share the reference estimator\'s keypoint detector (<detector>, with <masking>), so '
    'their reference-relative columns are the least independent of it.'
)
FLOOR_NOTE_AUDITED = (
    'the reference estimator\'s audited residual, <lower> px, is the equivalence margin of the '
    'reference-relative pixel columns, below which a difference is not resolvable, and '
    'mAA$_{\\varepsilon_{\\mathrm{floor}}}$ integrates from it.'
)
FLOOR_NOTE_POOLED = (
    'the largest of the cases\' audited residuals, <lower> px, is the equivalence margin of the '
    'reference-relative pixel columns, below which a difference is not resolvable, and '
    'mAA$_{\\varepsilon_{\\mathrm{floor}}}$ integrates from each case\'s own residual.<unaudited>'
)
FLOOR_NOTE_MISSING = (
    'no ground-control-point audit was available for this case, so no equivalence margin protects '
    'the reference-relative columns against differences smaller than the reference estimator\'s own error.'
)
GCP_CAPTION = (
    'Part II candidates against the homography fitted to each audited frame\'s ground control points, '
    'which involves no reference estimator: CTE is the corner transfer error, IoU and OCE compare the '
    'frame\'s object boxes as placed by the candidate and by the fitted transform. CTE$_{50}$ and '
    'CTE$_{90}$ are per case and averaged over the cases; CTE$_{\\max}$ is the largest over every audited '
    'frame.<counts> <bold_note>'
)
GCP_COUNTS_SHARED = ' Every row is scored on <frames> audited frames and <points> ground control points.'
GCP_COUNTS_LISTED = ' Frames and Points count the audited frames and the ground control points of the row.'
TUNING_CAPTION = (
    'Part I: the joint sweep of the ratio-test threshold $r$ and the inlier threshold $\\theta$ '
    '(reference-frame pixels) on the held-out tuning scenes. A pipeline keeps its preset\'s values unless '
    f'a grid point beats them on mAA by more than the margin ({MAA_MARGIN:g}); Grid counts the grid points '
    'and Tied those within the margin of the best. mAA$_{\\mathrm{default}}$ is the score at <default_values>, '
    'and $\\Delta$mAA, never bolded, the gain from tuning. $r$ reads -- for matchers without a ratio test.'
)

DYNAMIC_TUNING_CAPTION = (
    'Part II: the joint sweep of $r$ and $\\theta$ (<theta_units>) on <n_cases> tuning cases from the '
    'sites of Part I\'s tuning scenes, unmasked, against the reference estimator, each case weighing '
    'equally. A pipeline keeps its Part I values unless a grid point beats them on mAA by more than the '
    f'margin ({MAA_MARGIN:g}); Grid counts the grid points and Tied those within the margin of the best. '
    'mAA$_{\\mathrm{default}}$ is the score at the Part I values and $\\Delta$mAA, never bolded, the gain '
    'from tuning. $r$ reads -- for matchers without a ratio test.'
)

ESTIMATOR_CAPTION = (
    'Part I: robust estimators at a fixed detector, <masking>, each named by its OpenCV flag with the '
    'algorithm in parentheses (USAC-Default, USAC-Parallel and USAC-Fast are three configurations of '
    'LO-RANSAC). Columns <first> to <last> are mAA over $[0, <upper>]$ px at that inlier threshold '
    '$\\theta$ (px); Inl. is the inlier ratio, a diagnostic, at the smallest and the largest threshold; '
    'Fail and $>$<upper>\\,px count, over all thresholds, the failed estimates and the trials more than '
    '<upper> px off. Nothing is bolded: the comparison of interest is along each row. Every threshold\'s '
    'full row is in \\texttt{<csv>}.'
)

# Columns the tables drop when they carry no information (`write_latex_table(omit=...)`).
STATIC_OMIT = {name: OMIT_NOTES[name] for name in (*SATURATING_COLUMNS, 'failure_rate')}
DYNAMIC_OMIT = {name: OMIT_NOTES[name] for name in ('failure_rate', 'region_fraction', 'n_cases')}
FAILURE_OMIT = {'failure_rate': OMIT_NOTES['failure_rate']}
ESTIMATOR_OMIT = {'n': 'Every row counts {value} trials over its thresholds.'}


def read_kind(results_dir: Path) -> Optional[str]:
    """The protocol recorded in the results manifest, if present."""
    return (_read_manifest(results_dir) or {}).get('kind')


def _read_manifest(results_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(results_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return dict(json.load(f))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _case_dirs(results_dir: Path) -> List[Path]:
    """Where a dynamic result's case may be: the directory holding it, then the manifest's record."""
    manifest = _read_manifest(results_dir) or {}
    recorded = ((manifest.get('metadata') or {}).get('case') or {}).get('dir')
    return [Path(results_dir).parent.parent, *([Path(recorded)] if recorded else [])]


def read_epsilon_floor(results_dir: Path) -> Optional[float]:
    """
    The reference estimator's audited 95th-percentile residual, in pixels, or None.

    It is the equivalence margin of Part II and the lower limit of `maa_floor`. The case directory
    is the one holding `<case>/results/<name>`, with the manifest's `metadata.case.dir` (recorded
    on the machine that ran the campaign) as the fallback. None when the case has no audit yet.
    """
    for case_dir in _case_dirs(results_dir):
        path = case_dir / 'oracle' / AUDIT_FILE
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            value = report['reference_estimator']['summary']['residual_p95_px']
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        floor = float(value)
        # A non-finite or negative residual is a corrupt audit.
        if floor >= 0.0 and floor == floor and floor != float('inf'):
            return floor
    return None


def _read_case_audit(results_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The case directory holding a readable AUDIT.json and its contents, or None."""
    for case_dir in _case_dirs(results_dir):
        path = case_dir / 'oracle' / AUDIT_FILE
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return case_dir, dict(json.load(f))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return None


def read_audit(results_dir: Path) -> Optional[Dict[str, Any]]:
    """The case's AUDIT.json as a dict, located like `read_epsilon_floor` locates it, or None."""
    found = _read_case_audit(results_dir)
    return found[1] if found else None


def read_annotation_floor(results_dir: Path) -> Optional[float]:
    """The GCP repeatability recorded by `audit --repeat`, in pixels, or None."""
    report = read_audit(results_dir) or {}
    value = (report.get('annotation_repeatability') or {}).get('residual_p95_px')
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def region_dilation_px(results_dir: Path) -> Optional[int]:
    """The protocol's object dilation, from the config recorded in the manifest."""
    manifest = _read_manifest(results_dir) or {}
    region = ((manifest.get('config') or {}).get('metrics') or {}).get('region') or {}
    value = region.get('mask_dilation_px')
    return int(value) if value is not None else None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def gcp_candidate_rows(results_dir: Path) -> List[Dict[str, Any]]:
    """
    One row per candidate scored against the GCP-fitted transform, from the campaign's own scores
    (`GCP_SCORES.json`, written by `audit --results`).

    Empty when the campaign's candidates have not been scored.
    """
    path = Path(results_dir) / GCP_SCORES_NAME
    report = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    rows: List[Dict[str, Any]] = []
    for candidate in report.get('candidates') or []:
        against = candidate.get('gcp_transform') or {}
        summary = against.get('summary') or {}
        if not summary.get('n_frames'):
            continue
        errors = [_float_or_none(frame.get('cte_mean')) for frame in against.get('per_frame') or []]
        errors = [error for error in errors if error is not None and math.isfinite(error)]
        rows.append(
            {
                'pipeline': candidate.get('label'),
                'masking': candidate.get('masking'),
                'cte_median': summary.get('cte_median_px'),
                'cte_p90': summary.get('cte_p90_px'),
                'cte_max': max(errors) if errors else None,
                'iou_mean': summary.get('iou_mean'),
                'oce_mean': summary.get('oce_mean_px'),
                'n_frames': summary.get('n_frames'),
                'n_points': summary.get('n_points'),
            }
        )
    return rows


def pool_gcp_rows(per_case: Any) -> Any:
    """
    One row per candidate over the cases of `gcp_candidate_rows`: summaries averaged with equal weight
    per case, counts summed, and the largest error the largest over every audited frame.
    """
    import pandas as pd  # noqa: PLC0415 - kept out of module import so `report --help` stays cheap

    rows = []
    for (pipeline, masking), group in per_case.groupby(['pipeline', 'masking'], sort=False):
        largest = pd.to_numeric(group['cte_max'], errors='coerce') if 'cte_max' in group else pd.Series(dtype=float)
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                **{c: float(group[c].astype(float).mean()) for c in ('cte_median', 'cte_p90', 'iou_mean', 'oce_mean')},
                'cte_max': float(largest.max()) if largest.notna().any() else float('nan'),
                'n_frames': float(group['n_frames'].astype(float).sum()),
                'n_points': float(group['n_points'].astype(float).sum()),
            }
        )
    return _dataframe(rows)


def gcp_table_columns(rows: Any) -> Tuple[Dict[str, tuple], str]:
    """
    The columns of a GCP table and its caption sentence on the counts: the audited frames and the
    ground control points leave the table for the caption when every row is scored on the same.
    """
    import pandas as pd  # noqa: PLC0415 - kept out of module import so `report --help` stays cheap

    counts = {name: pd.to_numeric(rows[name], errors='coerce').dropna().unique() for name in ('n_frames', 'n_points')}
    if all(len(values) == 1 for values in counts.values()):
        columns = {name: spec for name, spec in GCP_COLUMNS.items() if name not in counts}
        sentence = _caption(
            GCP_COUNTS_SHARED, frames=f"{counts['n_frames'][0]:.0f}", points=f"{counts['n_points'][0]:.0f}"
        )
        return columns, sentence
    return dict(GCP_COLUMNS), GCP_COUNTS_LISTED


def annotation_root(results_dir: Path) -> Optional[Path]:
    """
    The directory whose `images/`, `labels_hbb/` and `labels_obb/` the campaign read.

    Static campaigns record it as `metadata.scenes_dir`; a dynamic result lives in its case
    (`<case>/results/<name>`, else `metadata.case.dir`). None when no candidate exists here; only
    the scene table is then skipped.
    """
    metadata = (_read_manifest(results_dir) or {}).get('metadata') or {}
    candidates = [metadata.get('scenes_dir')]
    if metadata.get('case'):
        candidates = [str(case_dir) for case_dir in _case_dirs(results_dir)]
    for value in candidates:
        if value and (Path(value) / 'images').is_dir():
            return Path(value)
    return None


def _scene_table(results_dirs: Any, tables_dir: Path, prefix: str, kind: str) -> str:
    """
    Write the scene-characterization table over one or several result directories (the frames of
    every pooled case), or raise with a reason to record as skipped.
    """
    import pandas as pd  # noqa: PLC0415 - kept out of module import so `report --help` stays cheap

    directories = [Path(d) for d in results_dirs] if isinstance(results_dirs, (list, tuple)) else [Path(results_dirs)]
    frames = []
    for results_dir in directories:
        root = annotation_root(results_dir)
        if root is None:
            raise ValueError(f"no annotation directory found for {results_dir}")
        names = ((_read_manifest(results_dir) or {}).get('metadata') or {}).get('scenes')
        frames.append(characterize_scenes(root, names))
    frame = pd.concat(frames, ignore_index=True)
    noun = 'scenes' if kind == 'static' else 'frames'
    return str(
        write_latex_table(
            summarize_scenes(frame),
            tables_dir / f"{prefix}_scenes.tex",
            columns=SCENE_COLUMNS,
            label_columns=('statistic',),
            caption=SCENE_CAPTION.format(count=len(frame), noun=noun),
            label=f"tab:{prefix}-scenes",
            highlight_best=False,
            csv=True,
        )
    )


GCP_AUDIT_CAPTION = (
    'Part II: the reference estimator against ground control points, per case. Frames: the audited frames '
    'besides the reference frame; Points: the ground control points matched in them; Res.$_{50}$ and '
    'Res.$_{95}$: median and 95th percentile of the distance between a point\'s annotated position and its '
    'reference-frame point carried by the reference estimator, over the case and per offset bin (frames); '
    'Rep.$_{95}$: 95th percentile of the GCP repeatability, where a second session exists; $<$\\,0.5\\,px: '
    'the share of the points placed by hand (\\Cref{tab:gcp-proposals}) within half a pixel of the '
    'reference estimator\'s prediction, near which the annotation panel opens under a hidden offset. The '
    'last row pools every case; the largest per-case Res.$_{95}$ is the equivalence margin of the pooled '
    'reference-relative comparisons.'
)

GCP_PROPOSAL_CAPTION = (
    'Part II: how the ground control points of the audited frames were placed, per case. A local image fit '
    'proposes each point and the annotator reviews the standing proposals: Accepted counts those kept as '
    'they were, Corrected those moved, by the median Corr.$_{50}$ and the largest Corr.$_{\\max}$ '
    'distance, and By hand the points without a standing proposal, placed in the annotation panel. The '
    'reference frame\'s points, all placed by hand, are not counted. The last row pools every case.'
)


def _write_gcp_audit_table(produced: Dict[str, Any], tables_dir: Path, results_dirs: Sequence[Path]) -> List[Path]:
    """
    `tables/gcp_audit.tex` from the audits of the cases behind `results_dirs`, when any exists.

    Returns the audited case directories, in the order of `results_dirs`.
    """
    audits: Dict[str, Dict[str, Any]] = {}
    cells: Dict[str, str] = {}
    case_dirs: List[Path] = []
    for directory in results_dirs:
        found = _read_case_audit(directory)
        if found is None:
            continue
        case_dir, audit = found
        name = case_dir.name
        audits[name] = audit
        case_dirs.append(case_dir)
        try:
            cells[name] = str(json.loads((case_dir / 'CASE.json').read_text(encoding='utf-8')).get('cell') or '')
        except (OSError, json.JSONDecodeError):
            cells[name] = ''
    if not audits:
        produced['skipped'].append('gcp audit table: no case has an AUDIT.json')
        return []
    frame = aggregate_gcp_audit(audits, cells)
    produced['tables'].append(
        str(
            write_latex_table(
                frame,
                tables_dir / 'gcp_audit.tex',
                columns=gcp_audit_columns(frame),
                label_columns=('case', 'cell'),
                group_columns=('case',),
                caption=GCP_AUDIT_CAPTION,
                label='tab:gcp-audit',
                highlight_best=False,
            )
        )
    )
    proposals = aggregate_gcp_proposals(audits, cells)
    if proposals.empty:
        produced['skipped'].append('gcp proposal table: no audit records a proposal review')
    else:
        produced['tables'].append(
            str(
                write_latex_table(
                    proposals,
                    tables_dir / 'gcp_proposals.tex',
                    columns=GCP_PROPOSAL_COLUMNS,
                    label_columns=('case', 'cell'),
                    group_columns=('case',),
                    caption=GCP_PROPOSAL_CAPTION,
                    label='tab:gcp-proposals',
                    highlight_best=False,
                )
            )
        )
    return case_dirs


def _write_gcp_audit_figure(
    produced: Dict[str, Any],
    figures_dir: Path,
    case_dirs: Sequence[Path],
    dilation_px: Optional[int] = None,
    audit_case: Optional[str] = None,
) -> None:
    """
    `figures/gcp_audit.pdf`: one audited case illustrates the audit, every audited case is pooled.

    The illustrated case is `audit_case` (a case directory name), else the first audited case.
    """
    if not case_dirs:
        if audit_case is not None:
            raise ValueError(f"--audit-case {audit_case}: none of the pooled cases has an audit")
        return
    shown = case_dirs[0]
    if audit_case is not None:
        named = [case_dir for case_dir in case_dirs if case_dir.name == audit_case]
        if not named:
            audited = ', '.join(case_dir.name for case_dir in case_dirs)
            raise ValueError(f"--audit-case {audit_case} is not among the audited cases ({audited})")
        shown = named[0]
    try:
        path = figure_gcp_audit(shown, case_dirs, figures_dir / 'gcp_audit.pdf', dilation_px)
        produced['figures'].append(str(path))
    except (AuditFigureError, GcpError, OracleError, OSError, KeyError, ValueError) as error:
        produced['skipped'].append(f"gcp audit figure: {error}")


def _write_case_figure(produced: Dict[str, Any], figures_dir: Path, results_dirs: Sequence[Path]) -> None:
    """`figures/cases.pdf`: every case's reference frame with what its objects did (`case_figure`)."""
    case_dirs = [
        next((case_dir for case_dir in _case_dirs(directory) if (case_dir / 'CASE.json').is_file()), None)
        for directory in results_dirs
    ]
    missing = [str(directory) for directory, case_dir in zip(results_dirs, case_dirs, strict=True) if case_dir is None]
    if missing:
        produced['skipped'].append(f"case figure: no CASE.json for {', '.join(missing)}")
        return
    try:
        produced['figures'].append(str(figure_cases(case_dirs, figures_dir / 'cases.pdf')))
    except (CaseFigureError, OracleError, OSError, KeyError, ValueError) as error:
        produced['skipped'].append(f"case figure: {error}")


def reference_note(results_dir: Path, frame: pd.DataFrame) -> str:
    """
    The caption sentence naming the reference estimator's keypoint detector and masking and the
    candidate rows that share the detector, read from the manifest; empty when either is unknown.
    """
    from stabilo_benchmark.pipelines import keypoint_detector  # noqa: PLC0415 - keeps the import graph flat

    metadata = (_read_manifest(results_dir) or {}).get('metadata') or {}
    pipeline = (metadata.get('reference_estimator') or {}).get('pipeline') or {}
    detector = pipeline.get('detector_name')
    masking = str(pipeline.get('masking') or 'none')
    if not detector or 'detector_name' not in frame or 'pipeline' not in frame:
        return ''
    shared = keypoint_detector(detector)
    rows = frame[frame['detector_name'].astype(str).map(keypoint_detector) == shared]
    presets = sorted({str(name).split('#', 1)[0].split(' (', 1)[0].rsplit('+', 1)[0] for name in rows['pipeline']})
    names = [pipeline_name(preset) for preset in presets]
    if not names:
        return ''
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])}{',' if len(names) > 2 else ''} and {names[-1]}"
    masking_text = 'no masking' if masking == 'none' else f"{masking.upper()} masking"
    return _caption(REFERENCE_NOTE, detector=shared.upper(), masking=masking_text, rows=listed.replace('_', '\\_'))


def _write_estimator_table(produced: Dict[str, Any], tables_dir: Path, prefix: str, listing: pd.DataFrame) -> None:
    """The estimator campaign as one page (`estimator_overview`), with every threshold's row in a CSV."""
    csv_path = tables_dir / f"{prefix}_estimators.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    listing[[c for c in ('detector', 'masking', 'estimator', *ESTIMATOR_COLUMNS) if c in listing]].to_csv(
        csv_path, index=False
    )
    try:
        produced['figures'].append(
            str(figure_estimator_thresholds(listing, tables_dir.parent / 'figures' / f"{prefix}_estimators.pdf"))
        )
    except ValueError as error:
        produced['skipped'].append(f"estimator figure: {error}")
    overview, columns = estimator_overview(listing)
    thresholds = [header for header, _, _ in columns.values()][:-5]
    maskings = sorted({str(m) for m in listing['masking'].dropna()})
    masking = f"{' and '.join(maskings)} masking" if maskings else 'masking as configured'
    produced['tables'] += [
        str(
            write_latex_table(
                overview,
                tables_dir / f"{prefix}_estimators.tex",
                columns=columns,
                label_columns=('detector', 'masking', 'estimator'),
                group_columns=('detector', 'estimator'),
                caption=_caption(
                    ESTIMATOR_CAPTION,
                    masking=masking.replace('obb', 'oriented').replace('hbb', 'horizontal'),
                    first=thresholds[0],
                    last=thresholds[-1],
                    upper=f"{MAA_UPPER_PX:g}",
                    csv=csv_path.name.replace('_', '\\_'),
                ),
                label=f"tab:{prefix}-estimators",
                tie_sets={},
                omit=ESTIMATOR_OMIT,
            )
        ),
        str(csv_path),
    ]


REGIMES = {'offline': 'offline regime', 'realtime': 'real-time regime'}


def _single(frame: Any, column: str) -> Any:
    values = frame[column].dropna().unique() if column in frame else []
    return values[0] if len(values) == 1 else None


def threshold_units(frame: Any) -> str:
    """The pixels a campaign's inlier threshold is expressed in, as caption text."""
    if _single(frame, 'ransac_threshold_space') == 'processed':
        return 'pixels of the downsampled image'
    return 'full-resolution pixels'


def regime_label(prefix: str, frame: Any) -> str:
    """
    The regime and budget of a Part II campaign, as caption text: the regime from the campaign's
    name (`<regime>_<campaign>`), the budget from its runs. Empty when the runs differ in budget.
    """
    ratio, features = _single(frame, 'downsample_ratio'), _single(frame, 'max_features')
    if ratio is None or features is None:
        return ''
    budget = f"downsample ratio {float(ratio):g} and {int(features)} features"
    regime = REGIMES.get(prefix.split('_', 1)[0])
    return f"{regime} at {budget}" if regime else f"at {budget}"


def tag_regime(paths: Sequence[str], prefix: str, label: str) -> None:
    """
    Name the regime in the caption of every Part II table of this campaign (`<prefix>_*.tex`), so
    the offline and real-time tables cannot be read for each other: inside the opening "Part II"
    where the caption has one, as a "Part II (...):" lead where it does not. Tables shared by the
    campaigns of a case set (the reference estimator's audit) do not carry the prefix and are left
    alone.
    """
    if not label:
        return
    for name in paths:
        path = Path(name)
        if path.suffix != '.tex' or not path.name.startswith(f"{prefix}_"):
            continue
        text = path.read_text(encoding='utf-8')
        tagged = text.replace(r'\caption{Part II (', rf'\caption{{Part II ({label}; ', 1)
        if tagged == text:
            tagged = re.sub(r'\\caption\{Part II(?=[ :])', lambda _: rf'\caption{{Part II ({label})', text, count=1)
        if tagged == text:
            # A caption that does not open with the part: "The analysis plan ..." -> "Part II (...): the ...".
            tagged = re.sub(
                r'\\caption\{([A-Z])(?=[a-z])',
                lambda match: rf'\caption{{Part II ({label}): {match.group(1).lower()}',
                text,
                count=1,
            )
        path.write_text(tagged, encoding='utf-8')


def evaluation_configs(prefix: str) -> str:
    """The configs a Part II tuning campaign's values go into, by its regime (`<regime>_tuning`)."""
    regime = prefix[: -len('_tuning')] if prefix.endswith('_tuning') else ''
    if regime == 'offline':
        return 'configs/songdo/dynamic/offline_masking.yaml and offline_margin.yaml'
    if regime:
        return f"configs/songdo/dynamic/{regime}_masking.yaml"
    return 'the evaluation configs of this campaign'


def default_prefix(name: str, kind: str) -> str:
    """The table prefix of a results directory named `name`; Part II's tuning never takes Part I's tables."""
    return 'tuning_dynamic' if kind == 'dynamic' and name == 'tuning' else name


def tuning_table_name(prefix: str) -> Tuple[str, str]:
    """File stem and label of a tuning campaign's table: Part I's `tuning` keeps the bare name."""
    if prefix == 'tuning':
        return 'tuned_parameters', 'tab:tuned-parameters'
    return f"{prefix}_tuned_parameters", f"tab:{prefix}-tuned-parameters"


OWN_VALUES_NOTE = "the preset's own values (it has no separate starting values)"


def starting_values_note(presets: Sequence[str]) -> str:
    """
    The untuned starting values of `presets` (`pretuning_defaults`) as caption text: the shared ratio
    and the most common inlier threshold, with the pipelines that start elsewhere named.
    """
    listed = pretuning_defaults()
    values = {preset: listed[preset] for preset in sorted(presets) if preset in listed}
    ratios = {float(value['filter_ratio']) for value in values.values()}
    thresholds: Dict[float, List[str]] = {}
    for preset, value in values.items():
        thresholds.setdefault(float(value['ransac_epipolar_threshold']), []).append(pipeline_name(preset))
    common = max(thresholds, key=lambda threshold: (len(thresholds[threshold]), -threshold))
    others = [
        f"{threshold:g} px for {' and '.join(sorted(names)) if len(names) < 3 else ', '.join(sorted(names))}"
        for threshold, names in sorted(thresholds.items(), reverse=True)
        if threshold != common
    ]
    ratio = f"a ratio of {ratios.pop():g}" if len(ratios) == 1 else "each preset's starting ratio"
    return f"the untuned starting values, {ratio} and an inlier threshold of {common:g} px" + (
        f" ({'; '.join(others)})" if others else ''
    )


def static_tuning_caption(frame: pd.DataFrame) -> str:
    """Part I's tuning caption, naming what mAA_default is scored at for the presets in `frame`."""
    listed = pretuning_defaults()
    presets = {str(name).split('#', 1)[0].split(' (', 1)[0].rsplit('+', 1)[0] for name in frame['pipeline'].unique()}
    if presets.isdisjoint(listed):
        note = OWN_VALUES_NOTE
    else:
        note = starting_values_note(sorted(presets & set(listed)))
        if not presets <= set(listed):
            note += "; a preset without them, at its own values"
    return TUNING_CAPTION.replace('<default_values>', note)


def _write_trial_figure(produced: Dict[str, Any], figures_dir: Path, prefix: str, results_dir: Path) -> None:
    """`<prefix>_trial.pdf` (`trial_figure`); skipped with the reason when the campaign cannot supply it."""
    from stabilo_benchmark.report.trial_figure import TrialFigureError, figure_static_trial  # noqa: PLC0415

    try:
        produced['figures'].append(str(figure_static_trial(results_dir, figures_dir / f"{prefix}_trial.pdf")))
    except (TrialFigureError, OSError, ValueError, KeyError) as error:
        produced['skipped'].append(f"trial figure: {error}")


def _write_tuning_figure(
    produced: Dict[str, Any],
    figures_dir: Path,
    stem: str,
    frame: pd.DataFrame,
    tuned: pd.DataFrame,
    defaults: Optional[Mapping[str, Mapping[str, Any]]],
) -> None:
    """The tuning surfaces beside the tuning table (`figure_tuning_surface`); a failure skips only the figure."""
    name = stem.replace('tuned_parameters', 'tuning_surface')
    try:
        produced['figures'].append(
            str(
                figure_tuning_surface(
                    tuning_grid(frame),
                    figures_dir / f"{name}.pdf",
                    selected=tuned,
                    defaults=defaults if defaults is not None else pretuning_defaults(),
                    default_label='untuned' if defaults is None else 'incumbent',
                )
            )
        )
    except ValueError as error:
        produced['skipped'].append(f"tuning figure: {error}")


def _write_tuning_table(
    produced: Dict[str, Any],
    tables_dir: Path,
    prefix: str,
    frame: pd.DataFrame,
    caption: str,
    defaults: Optional[Mapping[str, Mapping[str, Any]]] = None,
    mappings_header: str = '',
) -> None:
    """
    The tuning table; with `mappings_header`, also the selected values as per-detector YAML. Both
    parts keep the preset file's values unless a grid point beats them by more than the margin.
    """
    stem, label = tuning_table_name(prefix)
    try:
        tuned = aggregate_tuning(frame=frame, defaults=defaults, incumbent=preset_tuned_values())
        _write_tuning_figure(produced, tables_dir.parent / 'figures', stem, frame, tuned, defaults)
        produced['tables'].append(
            str(
                write_latex_table(
                    tuned,
                    tables_dir / f"{stem}.tex",
                    columns=TUNING_COLUMNS,
                    label_columns=('pipeline',),
                    caption=caption,
                    label=label,
                    sort_by='maa',
                    marks=TUNING_MARKS,
                    omit=FAILURE_OMIT,
                    unit='trial',
                )
            )
        )
    except ValueError as error:
        produced['skipped'].append(f"tuning table: {error}")
        return
    if mappings_header:
        produced['tables'].append(str(write_tuned_mappings(tuned, tables_dir / f"{stem}.yaml", mappings_header)))
    edges = tuning_edges(tuned)
    if edges:
        produced['warnings'].append(
            'Tuned values at an end of the swept range (widen the grid past them before adopting): ' + '; '.join(edges)
        )


def build_report(
    results_dir: Any,
    kind: Optional[str] = None,
    output: Optional[Path] = None,
    prefix: Optional[str] = None,
    audit_case: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write tables and figures for one results directory, or pool several; returns what was produced.

    `audit_case` names the case that illustrates the pooled GCP audit figure (default: the first
    audited case in `case_order`).

    Several directories are Part II's cases, pooled with equal weight per case and bolded with the
    case as the cluster. A single dynamic case bolds nothing.
    """
    paths = results_dir if isinstance(results_dir, (list, tuple)) else [results_dir]
    if len(paths) == 1 and is_survey_file(Path(paths[0])):
        root = Path(output) if output else DEFAULT_OUTPUT
        return build_survey_report(Path(paths[0]), root / 'tables', prefix=prefix or 'survey')
    if isinstance(results_dir, (list, tuple)):
        directories = [Path(d).expanduser().resolve() for d in results_dir]
        if len(directories) > 1:
            if {kind or read_kind(d) for d in directories} == {'static'}:
                return build_sweep_report(directories, output=output, prefix=prefix)
            return build_pooled_report(directories, kind=kind, output=output, prefix=prefix, audit_case=audit_case)
        results_dir = directories[0]
    results_dir = Path(results_dir).expanduser().resolve()
    kind = kind or read_kind(results_dir)
    if kind not in ('static', 'dynamic'):
        raise ValueError(
            f"Could not determine which protocol produced {results_dir}. Pass --kind static or --kind dynamic."
        )

    root = Path(output) if output else DEFAULT_OUTPUT
    tables_dir, figures_dir = root / 'tables', root / 'figures'
    prefix = prefix or default_prefix(results_dir.name, kind)

    frame = load_results(results_dir)
    object_boxes = object_boxes_of(frame)
    # Baseline rows appear in the tables only, not in the figures.
    candidates = frame[~frame['pipeline'].isin(BASELINE_PIPELINES)] if 'pipeline' in frame else frame
    produced: Dict[str, Any] = {
        'kind': kind,
        'object_boxes': object_boxes,
        'tables': [],
        'figures': [],
        'skipped': [],
        'warnings': [],
        # Also printed, besides the N/A rows in the tables.
        'infeasible': [
            str(run['params'].get('pipeline', run['run_number'])) for run in infeasible_runs(ResultStore(results_dir))
        ],
    }
    static_columns = columns_for_object_boxes(STATIC_COLUMNS, object_boxes)
    dynamic_columns = columns_for_object_boxes(DYNAMIC_COLUMNS, object_boxes)
    _check_annotated([results_dir], produced)

    if kind == 'static':
        summary = aggregate_static(results_dir)
        # Runs of one preset in a sweep are named by the parameters that vary between them.
        names = swept_run_labels(frame)
        if names:
            summary = summary.assign(pipeline=summary['pipeline'].map(lambda name: names.get(name, name)))
        clustered = static_cluster_column(frame.assign(pipeline=frame['pipeline'].map(lambda n: names.get(n, n))))
        ties = tie_sets_for(clustered, summary, static_columns, ('pipeline', 'masking'), 'site')
        n_sites = int(clustered['site'].nunique()) if 'site' in clustered else 0
        produced['tables'].append(
            str(
                write_latex_table(
                    summary,
                    tables_dir / f"{prefix}_static.tex",
                    columns=static_columns,
                    caption=_caption(
                        STATIC_CAPTION,
                        upper=f"{MAA_UPPER_PX:g}",
                        bold_note=_bold_note(ties, 'site', n_sites),
                    ),
                    label=f"tab:{prefix}-static",
                    sort_by='maa',
                    tie_sets=ties,
                    omit=STATIC_OMIT,
                    unit='trial',
                    csv=True,
                )
            )
        )
        _write_comparisons(
            produced, tables_dir, prefix, ties, static_columns, 'site', n_sites, r'\Cref{tab:%s-static}' % prefix
        )
        # Tuning sweep: the winning configuration per pipeline.
        if is_tuning_campaign(frame):
            _write_tuning_table(produced, tables_dir, prefix, frame, static_tuning_caption(frame))
        else:
            produced['skipped'].append(
                'tuning table: this campaign did not sweep both filter_ratio and '
                'ransac_epipolar_threshold at a fixed ransac_method'
            )

        # Estimator sweep: one row per estimator, one mAA column per threshold.
        if is_estimator_campaign(frame):
            try:
                _write_estimator_table(produced, tables_dir, prefix, aggregate_estimators(results_dir))
            except ValueError as error:
                produced['skipped'].append(f"estimator table: {error}")

        # A detector campaign: one of its trials, with where one pipeline's inliers lie.
        if not (is_budget_campaign(frame) or is_tuning_campaign(frame) or is_estimator_campaign(frame)):
            _write_trial_figure(produced, figures_dir, prefix, results_dir)

        # Budget sweep: accuracy against compute, one panel per preset.
        if is_budget_campaign(frame):
            try:
                produced['figures'].append(
                    str(
                        figure_accuracy_vs_compute(
                            budget := aggregate_budget(results_dir),
                            figures_dir / f"{prefix}_compute.pdf",
                            operating_points=_operating_points(budget),
                        )
                    )
                )
            except ValueError as error:
                produced['skipped'].append(f"accuracy-against-compute figure: {error}")
    else:
        floor = read_epsilon_floor(results_dir)
        produced['epsilon_floor_px'] = floor
        if floor is None:
            produced['skipped'].append(
                f"resolution floor: no {AUDIT_FILE} for this case, so no equivalence margin protects the "
                f"reference-relative columns; run `stabilo-benchmark audit` before reading the ranking"
            )
        else:
            _check_dilation(results_dir, floor, produced)
        caption = _caption(
            DYNAMIC_CAPTION,
            lower=f"{floor or 0.0:.2f}",
            upper=f"{MAA_UPPER_PX:g}",
            floor_note=FLOOR_NOTE_AUDITED if floor is not None else FLOOR_NOTE_MISSING,
            reference_note=reference_note(results_dir, candidates),
            bold_note=BOLD_NOTE_NONE,
        )
        # Runs of one preset in a sweep (the margin campaign) are named by the parameters that vary.
        names = swept_run_labels(frame)
        summary = _rename(aggregate_dynamic(results_dir, epsilon_floor=floor or 0.0), names)
        # A single case bolds nothing; the caption says why.
        produced['tables'].append(
            str(
                write_latex_table(
                    summary,
                    tables_dir / f"{prefix}_dynamic.tex",
                    columns=dynamic_columns,
                    caption=caption,
                    label=f"tab:{prefix}-dynamic",
                    sort_by='maa',
                    tie_sets={},
                    omit=DYNAMIC_OMIT,
                    unit='pair',
                )
            )
        )
        gcp_rows = gcp_candidate_rows(results_dir)
        if gcp_rows:
            produced['tables'].append(
                str(
                    write_latex_table(
                        gcp_frame := _rename(_dataframe(gcp_rows), names),
                        tables_dir / f"{prefix}_gcp.tex",
                        columns=(gcp_counts := gcp_table_columns(gcp_frame))[0],
                        caption=_caption(GCP_CAPTION, counts=gcp_counts[1], bold_note=BOLD_NOTE_NONE),
                        label=f"tab:{prefix}-gcp",
                        sort_by='cte_median',
                        tie_sets={},
                    )
                )
            )
        else:
            produced['skipped'].append(
                'GCP table: the audit has no candidates scored against the GCP-fitted transform; '
                'run `stabilo-benchmark audit <case> --results <results>`'
            )
        _write_breakdowns(
            produced,
            tables_dir,
            prefix,
            dynamic_columns,
            results_dir=results_dir,
            frame=_rename(frame, names),
            epsilon_floor=floor or 0.0,
        )
        _write_margin_outputs(produced, candidates, [results_dir], root, prefix)
        _write_repeat_spread(produced, frame, results_dir, tables_dir, prefix, floor)
        _write_capture(
            produced,
            candidates.assign(case=case_name(results_dir)),
            [results_dir],
            root,
            prefix,
            bool(names) or is_tuning_campaign(candidates),
            frame[frame['pipeline'] == BASELINE_IDENTITY].assign(case=case_name(results_dir)),
        )

    # The scene table is written for both protocols; a failure skips only this table.
    try:
        produced['tables'].append(_scene_table(results_dir, tables_dir, prefix, kind))
    except (ValueError, OSError, KeyError) as error:
        produced['skipped'].append(f"scene table: {error}")

    if kind == 'dynamic':
        tag_regime(produced['tables'], prefix, regime_label(prefix, candidates))
    for name, builder, requires in _figure_plan(kind, prefix, results_dir):
        missing = [column for column in requires if column not in candidates.columns]
        if missing:
            produced['skipped'].append(f"{name}: results lack {missing}")
            continue
        try:
            produced['figures'].append(str(builder(candidates, figures_dir / name)))
        except ValueError as error:
            produced['skipped'].append(f"{name}: {error}")

    _print_summary(produced, root)
    return produced


BY_OFFSET_CAPTION = (
    'Part II results grouped by temporal distance from the reference frame. The columns are those of the main '
    'dynamic table that the masking contrast is read on.'
)
BY_SCENARIO_CAPTION = (
    'Part II results by survey cell. Each case was assigned its cell from the video alone: '
    'moving foreground (high or low) crossed with its coherence (coherent or mixed), and a '
    'stationary control with much foreground of which little moved. The masking benefit is predicted to '
    'grow with the moving foreground and to shrink where the motion is incoherent; a benefit in the '
    'control as large as elsewhere would point to texture removal rather than moving objects. Columns '
    'are those of the main dynamic table that the masking contrast is read on.'
)
BY_PERSISTENCE_CAPTION = (
    'The stationary control at the level of the frame pair. Persistence is the overlap of the object '
    'footprints of the reference frame and the current frame in reference coordinates, split at the '
    'median over these results ({split:.2f}); a stationary pair holds a foreground that did not move. '
    'If masking helps as much on stationary pairs as on moving ones, the effect is about removing '
    'texture rather than motion.'
)
POOLED_NOTE = ' Each row pools its cases with equal weight.'
AVERAGE_NOTE = (
    ' Within each stratum, the All pipelines rows are the mean over the <n> candidate pipelines per masking '
    'mode, beside the identity and reference rows; every pipeline\'s rows are in '
    '\\texttt{<csv>}.'
)


def _rename(table: pd.DataFrame, names: Mapping[str, str]) -> pd.DataFrame:
    """`table` with its run labels replaced by the readable sweep names of `swept_run_labels`."""
    if not names or 'pipeline' not in table:
        return table
    return table.assign(pipeline=table['pipeline'].map(lambda name: names.get(str(name), name)))


def _write_repeat_spread(
    produced: Dict[str, Any],
    frame: pd.DataFrame,
    results_dir: Path,
    tables_dir: Path,
    prefix: str,
    epsilon_floor: Optional[float],
) -> None:
    """The run-to-run spread of a campaign that measured every pair more than once."""
    if 'repeat' not in frame or frame['repeat'].nunique() < 2:
        return
    from stabilo_benchmark.oracle import load_oracle  # noqa: PLC0415 - dynamic path only
    from stabilo_benchmark.report.repeats import repeat_spread, write_repeat_table  # noqa: PLC0415

    case_dir = next((d for d in _case_dirs(results_dir) if (d / 'oracle').is_dir()), None)
    size = (load_oracle(case_dir).get('frame_size') or {}) if case_dir else {}
    if not size:
        produced['skipped'].append('repeat spread: no frame size (the case has no reference estimates here)')
        return
    spread = repeat_spread(results_dir, int(size['width']), int(size['height']))
    if spread.empty:
        produced['skipped'].append('repeat spread: no pair was measured more than once with an estimate')
        return
    csv_path = tables_dir / f"{prefix}_repeat_spread.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    spread.to_csv(csv_path, index=False)
    produced['tables'].append(str(csv_path))
    produced['tables'].append(
        str(
            write_repeat_table(
                spread,
                tables_dir / f"{prefix}_repeat_spread.tex",
                int(frame['repeat'].nunique()),
                f"tab:{prefix}-repeat-spread",
                epsilon_floor,
            )
        )
    )


def _write_margin_outputs(
    produced: Dict[str, Any], frame: pd.DataFrame, results_dirs: Sequence[Path], root: Path, prefix: str
) -> None:
    """The margin sweep's benefit against suppressed area, when the results sweep `mask_margin_ratio`."""
    from stabilo_benchmark.report.margin import build_margin_outputs, is_margin_sweep  # noqa: PLC0415 - sweep only

    if is_margin_sweep(frame):
        build_margin_outputs(produced, frame, results_dirs, root, prefix)


def _write_capture(
    produced: Dict[str, Any],
    candidates: pd.DataFrame,
    results_dirs: Sequence[Path],
    root: Path,
    prefix: str,
    sweep: bool,
    identity: Optional[pd.DataFrame] = None,
) -> None:
    """The capture table and figures of a masking campaign; a sweep or a tuning grid has none."""
    from stabilo_benchmark.report.capture import build_capture_outputs  # noqa: PLC0415 - dynamic path only

    if sweep:
        produced['skipped'].append('capture: a parameter sweep or tuning grid, not a masking campaign')
        return
    build_capture_outputs(
        produced,
        candidates,
        results_dirs,
        root / 'tables',
        root / 'figures',
        prefix,
        campaign_offset_bins(results_dirs[0]),
        campaign_offset_bins(results_dirs[0], 'survey_bins'),
        identity,
    )


def _write_breakdowns(
    produced: Dict[str, Any],
    tables_dir: Path,
    prefix: str,
    columns: Mapping[str, tuple],
    results_dir: Optional[Path] = None,
    frame: Optional[pd.DataFrame] = None,
    epsilon_floor: float = 0.0,
    floors: Optional[Mapping[str, float]] = None,
) -> None:
    """
    The by-offset, by-cell and by-persistence tables of one case (`results_dir`) or of several
    (`frame` from `load_case_results`, with `results_dir` the first, for its sampling bins).
    """
    note = POOLED_NOTE if frame is not None and 'case' in frame else ''
    columns = {name: spec for name, spec in columns.items() if name in BREAKDOWN_COLUMNS}
    common = {'results_dir': results_dir, 'epsilon_floor': epsilon_floor, 'frame': frame, 'floors': floors}
    tables = (
        ('by-offset', 'offset_bin', lambda: (aggregate_by_offset(**common), BY_OFFSET_CAPTION)),
        ('by-scenario', 'scenario', lambda: (aggregate_by_scenario(**common), BY_SCENARIO_CAPTION)),
        (
            'by-persistence',
            'stratum',
            lambda: (
                table := aggregate_by_persistence(**common),
                BY_PERSISTENCE_CAPTION.format(split=table.attrs.get('splits', {}).get('persistence', float('nan'))),
            ),
        ),
    )
    for name, key, build in tables:
        try:
            table, caption = build()
        except (ValueError, KeyError) as error:
            produced['skipped'].append(f"{name} table: {error}")
            continue
        if table.empty:
            produced['skipped'].append(f"{name} table: no rows")
            continue
        stem = f"{prefix}_{name.replace('-', '_')}"
        csv_path = tables_dir / f"{stem}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        # Every pipeline in every stratum would run over several pages; the table shows their average.
        average = average_pipeline_rows(table, key)
        average_note = _caption(
            AVERAGE_NOTE, n=str(average.attrs['n_pipelines']), csv=csv_path.name.replace('_', '\\_')
        )
        produced['tables'] += [
            str(
                write_latex_table(
                    average,
                    tables_dir / f"{stem}.tex",
                    columns=columns,
                    label_columns=('pipeline', 'masking', key),
                    caption=caption + note + average_note,
                    label=f"tab:{prefix}-{name}",
                    tie_sets={},
                    omit=FAILURE_OMIT,
                    unit='pair',
                )
            ),
            str(csv_path),
        ]


def _write_contrasts(
    produced: Dict[str, Any],
    tables_dir: Path,
    prefix: str,
    candidates: pd.DataFrame,
    gcp_rows: Sequence[Mapping[str, Any]],
    n_cases: int,
    annotation_floor: Optional[float],
    bins: Sequence[Tuple[int, Optional[int]]] = (),
) -> None:
    """
    The planned masking and mechanism contrasts, with every interval and p-value in a CSV beside
    them, and the descriptive benefit per stratum (offset `bins`, survey cell, persistence).
    """
    import pandas as pd  # noqa: PLC0415 - kept out of module import so `report --help` stays cheap

    from stabilo_benchmark.report.contrasts import (  # noqa: PLC0415 - pandas-heavy, pooled path only
        POOLED,
        benefit_strata,
        case_table,
        masking_contrasts,
        mechanism_contrasts,
        write_benefit_strata_table,
        write_contrasts_table,
        write_mechanism_table,
    )

    if candidates['masking'].nunique() < 2:
        produced['skipped'].append('masking contrasts: the results hold one masking mode')
        return
    contrasts = masking_contrasts(case_table(candidates, gcp_rows), annotation_floor=annotation_floor)
    if contrasts.empty:
        produced['skipped'].append('masking contrasts: no pipeline was run under two masking modes')
        return
    if not gcp_rows:
        produced['warnings'].append(
            'Masking contrasts without the GCP outcomes: no primary contrast can be claimed until '
            '`audit --results` has scored the candidates.'
        )
    else:
        primary = contrasts[(contrasts['role'] == 'primary') & (contrasts['pipeline'] == POOLED)]
        short = primary[primary['n_clusters'] < n_cases]
        if not primary.empty and (not primary['complete'].all() or not short.empty):
            produced['warnings'].append(
                'The primary masking contrast is incomplete, so nothing is claimed: '
                + ', '.join(f"{row.outcome} on {row.n_clusters} of {n_cases} cases" for row in short.itertuples())
                + ('' if not short.empty else 'an outcome is missing')
                + '. Run `audit --results` on every case.'
            )
    csv_path = tables_dir / f"{prefix}_contrasts.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mechanism = mechanism_contrasts(candidates, bins)
    pd.concat([contrasts.assign(family='masking'), mechanism.assign(family='mechanism')], ignore_index=True).to_csv(
        csv_path, index=False
    )
    produced['tables'] += [
        str(
            write_contrasts_table(
                contrasts,
                tables_dir / f"{prefix}_contrasts.tex",
                n_cases,
                f"tab:{prefix}-contrasts",
                csv_path.name,
                annotation_floor=annotation_floor,
            )
        ),
        str(csv_path),
    ]
    if not mechanism.empty:
        mechanism.to_csv(tables_dir / f"{prefix}_mechanism.csv", index=False)
        produced['tables'] += [
            str(
                write_mechanism_table(
                    mechanism, tables_dir / f"{prefix}_mechanism.tex", n_cases, f"tab:{prefix}-mechanism"
                )
            ),
            str(tables_dir / f"{prefix}_mechanism.csv"),
        ]
    strata = benefit_strata(candidates, bins) if 'abs_offset' in candidates else pd.DataFrame()
    if strata.empty:
        produced['skipped'].append('benefit by stratum: no pipeline was run with and without oriented masking')
        return
    strata.to_csv(tables_dir / f"{prefix}_benefit_strata.csv", index=False)
    produced['tables'] += [
        str(
            write_benefit_strata_table(
                strata, tables_dir / f"{prefix}_benefit_strata.tex", n_cases, f"tab:{prefix}-benefit-strata", prefix
            )
        ),
        str(tables_dir / f"{prefix}_benefit_strata.csv"),
    ]


def build_pooled_report(
    results_dirs: List[Path],
    kind: Optional[str] = None,
    output: Optional[Path] = None,
    prefix: Optional[str] = None,
    audit_case: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Part II over all its cases: one table per quantity, equal weight per case.

    Every directory must come from the dynamic protocol. Each case's audited residual sets its own
    `maa_floor`; the equivalence margin for bolding is the largest residual among the cases.
    """
    kinds = {kind or read_kind(d) for d in results_dirs}
    if kinds != {'dynamic'}:
        raise ValueError(f"Pooling across result directories is for the dynamic protocol only; got {sorted(kinds)}")
    results_dirs = case_order(results_dirs)
    root = Path(output) if output else DEFAULT_OUTPUT
    tables_dir, figures_dir = root / 'tables', root / 'figures'
    prefix = prefix or default_prefix(results_dirs[0].name, 'dynamic')

    frame = load_case_results(results_dirs)
    # Runs of one preset in a sweep (the margin campaign) are named by the parameters that vary.
    names = swept_run_labels(frame)
    # The tuning table groups grid points on the raw run labels.
    raw_candidates = frame[~frame['pipeline'].isin(BASELINE_PIPELINES)]
    frame = _rename(frame, names)
    object_boxes = object_boxes_of(frame)
    dynamic_columns = columns_for_object_boxes(DYNAMIC_COLUMNS, object_boxes)
    candidates = frame[~frame['pipeline'].isin(BASELINE_PIPELINES)]
    produced: Dict[str, Any] = {
        'kind': 'dynamic',
        'object_boxes': object_boxes,
        'cases': [str(d) for d in results_dirs],
        'tables': [],
        'figures': [],
        'skipped': [],
        'warnings': [],
        # Also printed, besides the N/A rows in the main table.
        'infeasible': sorted(
            {str(run['params'].get('pipeline')) for d in results_dirs for run in infeasible_runs(ResultStore(d))}
        ),
    }

    _check_annotated(results_dirs, produced)

    floors: Dict[str, float] = {}
    annotation_floors: List[float] = []
    for directory in results_dirs:
        case = case_name(directory)
        floor = read_epsilon_floor(directory)
        if floor is None:
            produced['skipped'].append(f"resolution floor: no {AUDIT_FILE} for {case}")
        else:
            floors[case] = floor
            _check_dilation(directory, floor, produced)
        annotation = read_annotation_floor(directory)
        if annotation is not None:
            annotation_floors.append(annotation)
    margin = max(floors.values()) if floors else None
    annotation_floor = max(annotation_floors) if annotation_floors else None
    produced['epsilon_floor_px'] = margin
    produced['annotation_floor_px'] = annotation_floor
    n_cases = int(frame['case'].nunique())

    summary = _rename(aggregate_cases(results_dirs, epsilon_floors=floors), names)
    ties = tie_sets_for(candidates, summary, dynamic_columns, ('pipeline', 'masking'), 'case', epsilon_floor=margin)
    unaudited = n_cases - len(floors)
    floor_note = (
        _caption(
            FLOOR_NOTE_POOLED,
            unaudited=(
                f" {unaudited} of the {n_cases} cases have no audit, so their "
                'mAA$_{\\varepsilon_{\\mathrm{floor}}}$ starts at zero.'
                if unaudited
                else ''
            ),
        )
        if margin is not None
        else FLOOR_NOTE_MISSING.replace('for this case', 'for any case')
    )
    caption = _caption(
        DYNAMIC_CAPTION,
        lower=f"{margin or 0.0:.2f}",
        upper=f"{MAA_UPPER_PX:g}",
        floor_note=floor_note,
        reference_note=reference_note(results_dirs[0], candidates),
        bold_note=_bold_note(ties, 'case', n_cases, margin),
    )
    produced['tables'].append(
        str(
            write_latex_table(
                summary,
                tables_dir / f"{prefix}_dynamic.tex",
                columns={**dynamic_columns, 'n_cases': ('Cases', '{:.0f}', None)},
                caption=caption.replace('Part II (dynamic benchmark)', f"Part II (dynamic benchmark, {n_cases} cases)"),
                label=f"tab:{prefix}-dynamic",
                sort_by='maa',
                tie_sets=ties,
                omit=DYNAMIC_OMIT,
                unit='pair',
            )
        )
    )
    _write_comparisons(
        produced, tables_dir, prefix, ties, dynamic_columns, 'case', n_cases, r'\Cref{tab:%s-dynamic}' % prefix
    )

    dilations = {value for directory in results_dirs if (value := region_dilation_px(directory)) is not None}
    _write_gcp_audit_figure(
        produced,
        figures_dir,
        _write_gcp_audit_table(produced, tables_dir, results_dirs),
        dilations.pop() if len(dilations) == 1 else None,
        audit_case,
    )
    if not is_tuning_campaign(raw_candidates):
        _write_case_figure(produced, figures_dir, results_dirs)
    _write_breakdowns(
        produced, tables_dir, prefix, dynamic_columns, results_dir=results_dirs[0], frame=frame, floors=floors
    )
    try:
        produced['tables'].append(_scene_table(results_dirs, tables_dir, prefix, 'dynamic'))
    except (ValueError, OSError, KeyError) as error:
        produced['skipped'].append(f"scene table: {error}")

    # Part II's tuning sweep, pooled over the tuning cases.
    if is_tuning_campaign(raw_candidates):
        theta_units = threshold_units(raw_candidates)
        _write_tuning_table(
            produced,
            tables_dir,
            prefix,
            raw_candidates,
            DYNAMIC_TUNING_CAPTION.replace('<n_cases>', str(n_cases)).replace('<theta_units>', theta_units),
            defaults=preset_tuned_values(),
            mappings_header=(
                f"Part II's tuned values over {n_cases} tuning case(s), keyed by detector.\n"
                f"Paste both blocks into the `pipeline` section of {evaluation_configs(prefix)}."
            ),
        )

    gcp_rows = [
        {
            **row,
            'pipeline': names.get(str(row['pipeline']), row['pipeline']),
            'case': case_name(d),
        }
        for d in results_dirs
        for row in gcp_candidate_rows(d)
    ]
    if gcp_rows:
        per_case = _dataframe(gcp_rows)
        pooled = pool_gcp_rows(per_case)
        gcp_ties = {column: ts for column, ts in _gcp_tie_sets(per_case, pooled, annotation_floor).items()}
        produced['tables'].append(
            str(
                write_latex_table(
                    pooled,
                    tables_dir / f"{prefix}_gcp.tex",
                    columns=(gcp_counts := gcp_table_columns(pooled))[0],
                    caption=_caption(
                        GCP_CAPTION,
                        counts=gcp_counts[1],
                        bold_note=_bold_note(
                            gcp_ties, 'case', int(per_case['case'].nunique()), annotation_floor, against_gcp=True
                        ),
                    ),
                    label=f"tab:{prefix}-gcp",
                    sort_by='cte_median',
                    tie_sets=gcp_ties,
                )
            )
        )
    else:
        produced['skipped'].append('GCP table: no case has candidates scored against the GCP-fitted transform')

    if not names and not is_tuning_campaign(raw_candidates):
        try:
            produced['figures'].append(
                str(
                    figure_regime_summary(
                        summary,
                        figures_dir / f"{prefix}_summary.pdf",
                        gcp=pooled if gcp_rows else None,
                        maa_ties=ties.get('maa'),
                        gcp_ties=gcp_ties.get('cte_median') if gcp_rows else None,
                        annotation_floor=annotation_floor,
                    )
                )
            )
        except (ValueError, KeyError) as error:
            produced['skipped'].append(f"summary figure: {error}")

    if names:
        produced['skipped'].append('masking contrasts: a parameter sweep, not the masking campaign')
    else:
        _write_contrasts(
            produced,
            tables_dir,
            prefix,
            candidates,
            gcp_rows,
            n_cases,
            annotation_floor,
            bins=campaign_offset_bins(results_dirs[0]),
        )
    _write_margin_outputs(produced, candidates, results_dirs, root, prefix)
    _write_capture(
        produced,
        candidates,
        results_dirs,
        root,
        prefix,
        bool(names) or is_tuning_campaign(raw_candidates),
        frame[frame['pipeline'] == BASELINE_IDENTITY],
    )
    tag_regime(produced['tables'], prefix, regime_label(prefix, candidates))

    for name, builder, requires in _figure_plan('dynamic', prefix, results_dirs[0]):
        missing = [column for column in requires if column not in candidates.columns]
        if missing:
            produced['skipped'].append(f"{name}: results lack {missing}")
            continue
        try:
            produced['figures'].append(str(builder(candidates, figures_dir / name)))
        except ValueError as error:
            produced['skipped'].append(f"{name}: {error}")

    _print_summary(produced, root)
    return produced


def _gcp_tie_sets(per_case: Any, pooled: Any, annotation_floor: Optional[float]) -> Dict[str, Any]:
    """Tie sets of the pooled GCP table, with the case as the cluster."""
    from stabilo_benchmark.stats import margin_for, tie_set  # noqa: PLC0415 - keeps the import graph flat

    result: Dict[str, Any] = {}
    for column, (_, _, direction) in GCP_COLUMNS.items():
        if direction is None:
            continue
        table = per_case.pivot_table(index='case', columns=['pipeline', 'masking'], values=column, aggfunc='mean')
        table.columns = [tuple(c) for c in table.columns]
        values = pooled[column].astype(float)
        best_row = pooled.loc[values.idxmax() if direction else values.idxmin()]
        best = (best_row['pipeline'], best_row['masking'])
        if best not in table.columns or table.shape[0] < 2:
            continue
        result[column] = tie_set(table, best, bool(direction), margin=margin_for(column, None, annotation_floor))
    return result


COMPARISONS_CAPTION = (
    'The analysis plan applied, and the evidence behind the bold of <table>. Each row compares one '
    'configuration with the best in that outcome, on the <cluster>s where both were measured. '
    '$\\Delta$ is the Hodges--Lehmann estimate of the per-<cluster> paired difference in the '
    "column's own units, signed so that a positive value means the best configuration is ahead; the "
    'interval is a \\SI{95}{\\percent} percentile cluster bootstrap of that estimate over '
    '<draws> resamples of the <n_clusters> <cluster>s; and $p_{\\mathrm{Holm}}$ is the exact '
    'two-sided sign-flip permutation $p$, adjusted by Holm within the outcome. A tick marks a row '
    'that is not distinguishable from the best, meaning the interval includes zero or the estimate '
    'falls below the equivalence margin, and those are exactly the rows the results table bolds.<listed>'
)
COMPARISONS_LISTED = (
    ' Only mAA, the ranking quantity, is listed here; the comparisons behind every other bolded column '
    'are in \\texttt{<csv>}.'
)
# The outcome a comparisons table lists in full; every other outcome goes to its CSV only.
LISTED_OUTCOME = 'maa'


def _write_comparisons(
    produced: Dict[str, Any],
    tables_dir: Path,
    prefix: str,
    tie_sets: Mapping[str, Any],
    columns: Mapping[str, tuple],
    cluster: str,
    n_clusters: int,
    of_table: str,
) -> None:
    """Write the per-comparison listing behind a results table, when there is one to write."""
    from stabilo_benchmark.report.tables import (  # noqa: PLC0415 - keeps the import graph flat
        comparisons_frame,
        write_comparisons_table,
    )
    from stabilo_benchmark.stats import DEFAULT_DRAWS  # noqa: PLC0415 - keeps the import graph flat

    if not tie_sets:
        produced['skipped'].append(f"{prefix} comparisons: no {cluster}s to compare over")
        return
    frame = comparisons_frame(tie_sets, columns)
    if frame.empty:
        produced['skipped'].append(f"{prefix} comparisons: every outcome had a single configuration")
        return
    csv_path = tables_dir / f"{prefix}_comparisons.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    # Every outcome of a long table would run over several pages; the ranking quantity is listed.
    listed = frame[frame['column'] == LISTED_OUTCOME] if LISTED_OUTCOME in set(frame['column']) else frame
    note = _caption(COMPARISONS_LISTED, csv=csv_path.name.replace('_', '\\_')) if len(listed) < len(frame) else ''
    produced['tables'] += [
        str(
            write_comparisons_table(
                listed,
                tables_dir / f"{prefix}_comparisons.tex",
                caption=_caption(
                    COMPARISONS_CAPTION,
                    table=of_table,
                    cluster=cluster,
                    draws=f"{DEFAULT_DRAWS:,}".replace(',', '\\,'),
                    n_clusters=str(n_clusters),
                    listed=note,
                ),
                label=f"tab:{prefix}-comparisons",
            )
        ),
        str(csv_path),
    ]


def _bold_note(
    ties: Dict[str, Any],
    cluster: str,
    n_clusters: int,
    pixel_margin: Optional[float] = None,
    against_gcp: bool = False,
) -> str:
    """
    The caption sentence that says what bold means in this table. `against_gcp` states the margins
    of the table scored against the GCP-fitted transform: the GCP repeatability, no mAA.
    """
    if not ties:
        return BOLD_NOTE_NONE if cluster == 'case' else 'Nothing is bolded: no clusters were available to compare over.'
    if against_gcp:
        margins = (
            _caption(GCP_MARGINS, pixel_margin=f"{pixel_margin:.2f} px, the GCP repeatability,")
            if pixel_margin
            else 'none, since no repeat annotation session gives a GCP repeatability'
        )
    else:
        pixel = f", {pixel_margin:.2f} px for the reference-relative pixel columns" if pixel_margin else ''
        margins = _caption(MAA_MARGINS, pixel_margin=pixel)
    note = _caption(BOLD_NOTE_PLAN, cluster=cluster, n_clusters=str(n_clusters), margins=margins)
    if cluster == 'case':
        note += f" With {n_clusters} cases the smallest attainable sign-flip $p$ is {minimum_p(n_clusters):.4f}."
    return note


def _check_annotated(results_dirs: Sequence[Path], produced: Dict[str, Any]) -> None:
    """Warn for every campaign that ran on frames without annotations (`masking.is_annotated`)."""
    for results_dir in results_dirs:
        metadata = (_read_manifest(results_dir) or {}).get('metadata') or {}
        if metadata.get('annotated', (metadata.get('case') or {}).get('annotated', True)):
            continue
        where = ' and its evaluation region includes the objects' if metadata.get('case') else ''
        produced['warnings'].append(f"{results_dir} ran without annotations: IoU and OCE are undefined{where}")


def _check_dilation(results_dir: Path, floor: float, produced: Dict[str, Any]) -> None:
    """
    Warn when the audited residual reaches the evaluation region's dilation.

    The region places current-frame objects with the reference transform, so a larger residual makes
    the reference-free metrics depend on the reference estimator.
    """
    tau = region_dilation_px(results_dir)
    if tau is not None and floor >= tau:
        produced['warnings'].append(
            f"the audited residual ({floor:.2f} px) reaches the region dilation ({tau} px) for "
            f"{results_dir}: the reference-free metrics are no longer independent of the reference estimator"
        )


# Axis labels for sweep keys the manuscript plots; any other key is labeled by its last component.
SWEEP_LABELS = {'perturbation.geometric.rotation_deg': 'In-plane rotation [deg]'}


def build_sweep_report(
    results_dirs: List[Path],
    output: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Part I campaigns that differ in one config key (a `--set` sweep): mAA against that key.

    Writes `<prefix>_sweep.tex` (one column per sweep point) and `<prefix>_sweep.pdf` (one line per
    pipeline). The prefix defaults to the campaigns' common name stem (`rotation` for `rotation_*deg`).
    """
    import pandas as pd  # noqa: PLC0415 - only the sweep path needs it here

    root = Path(output) if output else DEFAULT_OUTPUT
    tables_dir, figures_dir = root / 'tables', root / 'figures'
    names = [d.name for d in results_dirs]
    prefix = prefix or os.path.commonprefix(names).rstrip('_0123456789') or 'sweep'

    key, frame = aggregate_sweep(results_dirs)
    single_masking = frame['masking'].nunique() == 1
    frame = frame.assign(
        label=frame['pipeline'].str.replace(r'\+[a-z]+$', '', regex=True) if single_masking else frame['pipeline']
    )
    values = sorted(frame['value'].unique())
    infeasible = sorted(
        {str(run['params'].get('pipeline')) for d in results_dirs for run in infeasible_runs(ResultStore(d))}
    )
    produced: Dict[str, Any] = {
        'kind': 'static',
        'sweep_key': key,
        'tables': [],
        'figures': [],
        'skipped': [],
        'warnings': [],
        'infeasible': infeasible,
    }

    table = frame.pivot_table(index='label', columns='value', values='maa').reset_index()
    table.columns = ['pipeline', *[f"{v:g}" for v in table.columns[1:]]]
    table = table.sort_values(f"{values[0]:g}", ascending=False)
    columns = {f"{v:g}": (f"{v:g}", '{:.3f}', None) for v in values}
    label = SWEEP_LABELS.get(key, key.rsplit('.', 1)[-1].replace('_', ' '))
    produced['tables'].append(
        str(
            write_latex_table(
                table,
                tables_dir / f"{prefix}_sweep.tex",
                columns=columns,
                label_columns=('pipeline',),
                caption=f"Part I: mAA over $[0, {MAA_UPPER_PX:g}]$\\,px against {label.lower()}, "
                "one campaign per column."
                + (
                    " Not shown, resource-infeasible on the benchmark machine: "
                    + ", ".join(sorted({pipeline_name(n.split('+')[0]).replace('_', '\\_') for n in infeasible}))
                    + "."
                    if infeasible
                    else ""
                ),
                label=f"tab:{prefix}-sweep",
                highlight_best=False,
                tie_sets={},
                csv=True,
            )
        )
    )
    try:
        produced['figures'].append(
            str(
                figure_sweep(
                    pd.DataFrame(frame).assign(label=frame['label'].map(display_label)),
                    figures_dir / f"{prefix}_sweep.pdf",
                    x='value',
                    y='maa',
                    series='label',
                    x_label=label,
                    y_label='mAA',
                    x_ticks=values,
                )
            )
        )
    except ValueError as error:
        produced['skipped'].append(f"sweep figure: {error}")
    _print_summary(produced, root)
    return produced


# Part II's regimes, ringed in Part I's compute figure when a clone holds their configs.
REGIME_CONFIGS = {
    'Part II offline': Path(__file__).resolve().parents[2] / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml',
    'Part II real-time': Path(__file__).resolve().parents[2]
    / 'configs'
    / 'songdo'
    / 'dynamic'
    / 'realtime_masking.yaml',
}


def _operating_points(budget: Any) -> Dict[str, tuple]:
    """
    The budgets ringed in the compute figure: Part I's operating point, the shipped (downsample_ratio,
    max_features) of the `base` preset, and each Part II regime's budget that lies on the sweep.
    """
    from stabilo_benchmark.config import ConfigError, load_config  # noqa: PLC0415 - avoids import cycle
    from stabilo_benchmark.pipelines import load_presets  # noqa: PLC0415 - avoids import cycle

    points: Dict[str, tuple] = {}
    base = load_presets().get('base', {})
    if 'downsample_ratio' in base and 'max_features' in base:
        points['Part I'] = (float(base['downsample_ratio']), int(base['max_features']))
    swept = (
        set()
        if budget.empty
        else set(zip(budget['downsample_ratio'].astype(float), budget['max_features'], strict=True))
    )
    for name, path in REGIME_CONFIGS.items():
        try:
            pipeline = load_config(path)['pipeline']
            point = (float(pipeline['downsample_ratio']), int(pipeline['max_features']))
        except (ConfigError, KeyError, OSError, TypeError, ValueError):
            continue
        if any(abs(point[0] - ratio) < 1e-9 and point[1] == features for ratio, features in swept):
            points[name] = point
    return points


def _dataframe(rows: List[Dict[str, Any]]) -> Any:
    import pandas as pd  # noqa: PLC0415 - only the pooled path needs it here

    return pd.DataFrame(rows)


def _figure_plan(kind: str, prefix: str, results_dir: Optional[Path] = None) -> List[tuple]:
    """Which figures to attempt for a protocol, and the columns each needs."""
    if kind == 'dynamic':
        offsets = partial(figure_error_vs_offset, bins=campaign_offset_bins(results_dir) if results_dir else None)
        return [
            (f"{prefix}_inlier_vs_quality.pdf", figure_inlier_vs_quality, ('inlier_ratio', 'cte_mean', 'masking')),
            (f"{prefix}_masking.pdf", figure_masking_comparison, ('masking', 'ssim')),
            (f"{prefix}_error_vs_offset.pdf", offsets, ('abs_offset', 'cte_mean', 'masking')),
            (f"{prefix}_motion_alignment.pdf", figure_motion_alignment, ('motion_alignment', 'masking')),
        ]
    # Static: no offsets, reference estimator, motion axis or photometric scores.
    static_masking = partial(
        figure_masking_comparison,
        metrics=('iou_mean', 'oce_mean'),
        metric_labels={'iou_mean': 'IoU', 'oce_mean': 'OCE [px]'},
    )
    # Part I's error is against the known homography, not a reference estimate.
    static_inliers = partial(figure_inlier_vs_quality, quality_label='Corner transfer error [px]')
    return [
        (f"{prefix}_inlier_vs_quality.pdf", static_inliers, ('inlier_ratio', 'cte_mean', 'masking')),
        (f"{prefix}_masking.pdf", static_masking, ('masking', 'iou_mean', 'oce_mean')),
    ]


def _caption(template: str, **values: str) -> str:
    """
    Fill <name> placeholders in a caption.

    Not str.format, whose fields would clash with LaTeX braces.
    """
    # Two passes: a substituted value may contain a placeholder.
    for _ in range(2):
        for name, value in values.items():
            template = template.replace(f"<{name}>", value)
    return template


def _print_summary(produced: Dict[str, Any], root: Path) -> None:
    print(f"\n\033[92mTables and figures written under {root}\033[0m")
    floor = produced.get('epsilon_floor_px')
    if floor is not None:
        print(
            f"  equivalence margin {floor:.2f} px (audited residual); mAA over [0, {MAA_UPPER_PX:g}] px, "
            f"mAA_floor over [{floor:.2f}, {MAA_UPPER_PX:g}] px"
        )
    for table in produced['tables']:
        print(f"  table   {table}")
    for figure in produced['figures']:
        print(f"  figure  {figure}")
    for reason in produced['skipped']:
        print(f"\033[33m  skipped {reason}\033[0m")
    for warning in produced.get('warnings') or []:
        print(f"\033[93m  warning {warning}\033[0m")
    aborted = produced.get('infeasible') or []
    if aborted:
        print(
            f"\033[33m  {len(aborted)} run(s) were resource-infeasible on the benchmark machine and are "
            f"reported as N/A: {', '.join(aborted)}\033[0m"
        )
