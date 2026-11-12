#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
capture.py - Consensus capture in a Part II campaign, read from its measurements.

A measurement is captured when its `capture_fraction` reaches CAPTURE_THRESHOLD: where the pair's
largest group of objects that moved together lies, the estimate follows at least that share of the
group's displacement. It is defined on pairs whose group holds at least
`scene.MIN_DISPLACED_OBJECTS` objects, and only where the pipeline returned a transform.

    <prefix>_capture.tex/.csv            per pipeline and masking mode: the pairs with a group, how
                                         many were captured and in how many cases, the error of the
                                         captured, and their inlier statistics, also ranked among the
                                         pipeline's estimates at the same case and offset bin (the
                                         table lists the configurations ever captured and names the
                                         largest capture fraction of the others; the CSV holds every
                                         configuration, the other pairs' statistics and the counts at
                                         stricter thresholds)
    <prefix>_capture_scatter.pdf         every measurement's capture fraction against the offset, a
                                         panel per pipeline
    <prefix>_capture_by_offset.csv/.pdf  the captured share per offset bin and configuration
    <prefix>_capture_by_case.csv         the pairs, those with a group and the captured per case
    <prefix>_capture_populations.tex/.csv
                                         per configuration ever captured, what the count holds:
                                         pulls within the survey's offsets, breakdowns, and the rest,
                                         with the error at the group, its direction, and the inliers
                                         against the same case's nearest offsets
    <prefix>_capture_example.pdf         the captured unmasked measurement with the largest capture
                                         fraction within the survey's offsets that re-registers to
                                         its stored estimate, beside the same pipeline under
                                         oriented-box masking
    <prefix>_graphical_abstract.pdf/.png the same pair in the journal's graphical-abstract format
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from stabilo_benchmark.metrics.diagnostics import BREAKDOWN_PX, CAPTURE_THRESHOLD, group_error
from stabilo_benchmark.report.contrasts import preset_of
from stabilo_benchmark.report.figures import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    MASKING_LABELS,
    MASKING_ORDER,
    PALETTE,
    _masking_series,
    _save,
    _style_for,
    apply_style,
    plt,
)
from stabilo_benchmark.report.names import PIPELINE_NAMES, pipeline_name
from stabilo_benchmark.report.tables import case_name, write_latex_table
from stabilo_benchmark.results import GCP_SCORES_NAME
from stabilo_benchmark.sampling import bin_label, offset_bin_label

CAPTURE_COLUMNS: Dict[str, tuple] = {
    'n_pairs': ('Pairs', '{:.0f}', None),
    'n_captured': ('Captured', '{:.0f}', None),
    'captured_share': ('Share', '{:.1%}', None),
    'n_cases_captured': ('Cases', '{:.0f}', None),
    'capture_fraction_max': (r'$\lambda_{\max}$', '{:.2f}', None),
    'cte_captured': (r'CTE$_{50}$ [px]', '{:.1f}', None),
    'n_captured_audited': ('Audited', '{:.0f}', None),
    'cte_gcp_captured': (r'CTE$^{\mathrm{GCP}}_{50}$ [px]', '{:.1f}', None),
    'inliers_captured': ('Inliers', '{:.0f}', None),
    'inliers_rank_captured': ('Inlier rank', '{:.2f}', None),
    'ratio_captured': ('Ratio', '{:.2f}', None),
    'ratio_rank_captured': ('Ratio rank', '{:.2f}', None),
    'on_moving_captured': ('On moving', '{:.0%}', None),
}

# Stricter capture thresholds whose counts the CSV adds, for the threshold's sensitivity; at 1 an
# estimate registers the group fully or errs past its displacement.
SENSITIVITY_THRESHOLDS = (0.25, 0.5, 1.0)

CAPTURE_CAPTION = (
    'Consensus capture over <cases><listed>. A pair counts when at least <min_objects> of the reference '
    "frame's objects moved together; an estimate on it is "
    "captured when it follows at least <threshold> of that group's displacement ($\\lambda \\geq <lambda>$). "
    'Cases: cases with a capture; CTE$_{50}$: median corner error of the captured against the reference '
    'estimator, and CTE$^{\\mathrm{GCP}}_{50}$ against the ground control points on the Audited frames; '
    'Inliers and Ratio: median inlier count and ratio of the captured, with their median percentile among '
    'the same configuration\'s estimates at the same case and offset bin (0.5 typical, 1 the most); On '
    "moving: median share of their inliers on objects that moved. Descriptive; nothing is bolded.<others> "
    'Every configuration, per offset bin: \\texttt{<csv>}.'
)
CAPTURE_OTHERS = ' The other <n> configurations are captured on no pair, with a largest $\\lambda$ of <largest>.'
CAPTURE_MASKED = ' No masked estimate exceeds $\\lambda = <masked>$.'
CAPTURE_IDENTITY = (
    ' The identity transform reaches $\\lambda = <identity>$, and the smallest displacement of any group '
    'is <shift>\\,px.'
)
CAPTURE_OMIT = {'n_pairs': 'Every row counts {value} such pairs.'}

# Offsets on either side of a captured estimate whose inliers it is compared with.
NEAREST_OFFSETS = 2
POPULATIONS = ('pull', 'breakdown', 'other')

POPULATION_COLUMNS: Dict[str, tuple] = {
    'n_captured': ('Captured', '{:.0f}', None),
    'n_cases': ('Cases', '{:.0f}', None),
    'offset_max': ('Offset$_{\\max}$', '{:.0f}', None),
    'capture_fraction_max': (r'$\lambda_{\max}$', '{:.2f}', None),
    'error_group': (r'Error$_{50}$ [px]', '{:.1f}', None),
    'error_angle_max': (r'Angle$_{\max}$ [deg]', '{:.1f}', None),
    'cte_median': (r'CTE$_{50}$ [px]', '{:.1f}', None),
    'cte_min': (r'CTE$_{\min}$ [px]', '{:.1f}', None),
    'cte_max': (r'CTE$_{\max}$ [px]', '{:.1f}', None),
    'inliers_median': ('Inliers', '{:.0f}', None),
    'inliers_max': (r'Inliers$_{\max}$', '{:.0f}', None),
    'ratio_max': (r'Ratio$_{\max}$', '{:.2f}', None),
    'inliers_nearest': ('Inliers/nearest', '{:.2f}', None),
    'ratio_nearest': ('Ratio/nearest', '{:.2f}', None),
}
POPULATION_CAPTION = (
    'What the capture count holds, over <cases>, for every configuration captured on at least one pair. '
    "Pull: a captured estimate within the survey's offsets of up to <horizon> frames; breakdown: one whose "
    'corner error against the reference estimator reaches <breakdown>\\,px; other: the rest. Error$_{50}$: '
    "median length of the error at the group; Angle$_{\\max}$: largest angle between that error and the "
    "group's displacement; Inliers/nearest and Ratio/nearest: median inlier count and ratio over those of "
    'the same configuration and case at the nearest <nearest> offsets on either side that are neither '
    'captured nor broken down. Descriptive; nothing is bolded.'
)

# Panels per row of the capture scatter: ten pipelines fill two rows.
PANEL_COLUMNS = 5

# Ranked candidates tried for the example figure before giving up on reproducing one.
EXAMPLE_ATTEMPTS = 10
# A re-registration reproduces a stored estimate when their mean corner transfer differs by less than this.
REPRODUCTION_TOLERANCE_PX = 0.05


def captured(frame: pd.DataFrame, threshold: float = CAPTURE_THRESHOLD) -> pd.Series:
    """Measurements whose estimate follows at least `threshold` of the group's displacement."""
    return pd.to_numeric(frame['capture_fraction'], errors='coerce') >= threshold


def with_group(frame: pd.DataFrame) -> pd.DataFrame:
    """The measurements whose pair has a group and whose pipeline returned a transform."""
    if 'capture_fraction' not in frame:
        return frame.iloc[0:0]
    return frame[np.isfinite(pd.to_numeric(frame['capture_fraction'], errors='coerce'))]


def inlier_ranks(candidates: pd.DataFrame, bins: Sequence[Tuple[int, Optional[int]]]) -> pd.DataFrame:
    """
    Each measurement's percentile (mid-rank) of inlier count and inlier ratio among the estimates of
    its pipeline and masking mode on the pairs of its case and offset bin, every pair counted.
    """
    labels = candidates['abs_offset'].map(lambda offset: offset_bin_label(int(offset), bins))
    keys = ['pipeline', 'masking', labels.rename('_bin')] + (['case'] if 'case' in candidates else [])
    ranks = pd.DataFrame(index=candidates.index)
    for column, name in (('n_inliers', 'inliers_rank'), ('inlier_ratio', 'ratio_rank')):
        values = pd.to_numeric(candidates[column], errors='coerce') if column in candidates else np.nan
        frame = candidates.assign(_value=values)
        ranks[name] = frame.groupby(keys, dropna=False)['_value'].rank(pct=True, method='average')
    return ranks


def capture_table(candidates: pd.DataFrame, bins: Optional[Sequence[Tuple[int, Optional[int]]]] = None) -> pd.DataFrame:
    """
    One row per pipeline and masking mode, pooled over cases (counts, not per-case means). The inlier
    ranks need the campaign's offset `bins`.
    """
    rows: List[Dict[str, Any]] = []
    if bins is not None and 'abs_offset' in candidates:
        candidates = candidates.join(inlier_ranks(candidates, bins))
    defined = with_group(candidates)
    for (pipeline, masking), measured in candidates.groupby(['pipeline', 'masking'], sort=False):
        group = defined[(defined['pipeline'] == pipeline) & (defined['masking'] == masking)]
        broken = broken_down(measured)
        hit = captured(group)
        took, other = group[hit], group[~hit]
        rows.append(
            {
                'pipeline': pipeline,
                'masking': masking,
                'n_pairs': len(group),
                'n_captured': int(hit.sum()),
                'captured_share': float(hit.mean()) if len(group) else float('nan'),
                'n_cases_captured': int(took['case'].nunique()) if 'case' in took else int(bool(len(took))),
                'cases_captured': ' '.join(sorted(map(str, took['case'].unique()))) if 'case' in took else '',
                'capture_fraction_max': _maximum(group, 'capture_fraction'),
                'cte_captured': _median(took, 'cte_mean'),
                'n_captured_audited': int(took['cte_gcp'].notna().sum()) if 'cte_gcp' in took else 0,
                'cte_gcp_captured': _median(took, 'cte_gcp'),
                'inliers_captured': _median(took, 'n_inliers'),
                'inliers_rank_captured': _median(took, 'inliers_rank'),
                'inliers_other': _median(other, 'n_inliers'),
                'ratio_captured': _median(took, 'inlier_ratio'),
                'ratio_rank_captured': _median(took, 'ratio_rank'),
                'ratio_other': _median(other, 'inlier_ratio'),
                'on_moving_captured': _median(took, 'inliers_on_moving'),
                **{
                    f"n_captured_at_{threshold:g}": int(captured(group, threshold).sum())
                    for threshold in SENSITIVITY_THRESHOLDS
                },
                'n_measurements': len(measured),
                'n_broken_down': int(broken.sum()),
                'n_broken_down_with_group': int(broken.reindex(group.index, fill_value=False).sum()),
                'group_shift_min': _minimum(group, 'coherent_shift_px'),
            }
        )
    return pd.DataFrame(rows)


def gcp_frame_errors(results_dirs: Sequence[Path]) -> pd.DataFrame:
    """
    Every candidate's corner transfer error against the GCP-fitted transform on each audited frame
    (`GCP_SCORES.json`, written by `audit --results`), keyed by case, pipeline, masking and offset.
    """
    rows: List[Dict[str, Any]] = []
    for results_dir in results_dirs:
        path = Path(results_dir) / GCP_SCORES_NAME
        if not path.is_file():
            continue
        case = case_name(Path(results_dir))
        for candidate in json.loads(path.read_text(encoding='utf-8')).get('candidates') or []:
            for frame in (candidate.get('gcp_transform') or {}).get('per_frame') or []:
                if frame.get('offset') is None:
                    continue
                rows.append(
                    {
                        'case': case,
                        'pipeline': candidate.get('label'),
                        'masking': candidate.get('masking'),
                        'abs_offset': abs(int(frame['offset'])),
                        'cte_gcp': _number(frame.get('cte_mean')),
                    }
                )
    return pd.DataFrame(rows, columns=['case', 'pipeline', 'masking', 'abs_offset', 'cte_gcp'])


def with_gcp_errors(candidates: pd.DataFrame, results_dirs: Sequence[Path]) -> pd.DataFrame:
    """`candidates` with `cte_gcp` on the measurements whose current frame was audited (NaN elsewhere)."""
    keys = ['case', 'pipeline', 'masking', 'abs_offset']
    errors = gcp_frame_errors(results_dirs)
    if errors.empty or not all(key in candidates for key in keys):
        return candidates.assign(cte_gcp=np.nan)
    merged = candidates.merge(errors.drop_duplicates(keys), on=keys, how='left', validate='many_to_one')
    merged.index = candidates.index
    return merged


def capture_by_case(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per case (with its survey cell), pipeline and masking mode: the pairs, those with a group, and the captured."""
    keys = [key for key in ('case', 'cell') if key in candidates] + ['pipeline', 'masking']
    fraction = pd.to_numeric(candidates['capture_fraction'], errors='coerce')
    frame = candidates.assign(_group=np.isfinite(fraction), _captured=captured(candidates))
    return (
        frame.groupby(keys, sort=False, dropna=False)
        .agg(n_measurements=('_group', 'size'), n_pairs=('_group', 'sum'), n_captured=('_captured', 'sum'))
        .reset_index()
    )


def capture_by_offset(candidates: pd.DataFrame, bins: Sequence[Tuple[int, Optional[int]]]) -> pd.DataFrame:
    """The captured share per pipeline, masking mode and offset bin, over the bins holding a pair with a group."""
    defined = with_group(candidates)
    if defined.empty:
        return pd.DataFrame()
    labels = defined['abs_offset'].map(lambda offset: offset_bin_label(int(offset), bins))
    defined = defined.assign(offset_bin=labels, _captured=captured(defined))
    order = {bin_label(low, high): rank for rank, (low, high) in enumerate(bins)}
    table = (
        defined.groupby(['pipeline', 'masking', 'offset_bin'], sort=False)
        .agg(n_pairs=('_captured', 'size'), n_captured=('_captured', 'sum'))
        .reset_index()
    )
    table['captured_share'] = table['n_captured'] / table['n_pairs']
    table['bin_rank'] = table['offset_bin'].map(order)
    return table.sort_values(['pipeline', 'masking', 'bin_rank'], kind='stable').reset_index(drop=True)


def broken_down(frame: pd.DataFrame) -> pd.Series:
    """Measurements whose corner error against the reference estimator reaches BREAKDOWN_PX."""
    if 'cte_mean' not in frame:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame['cte_mean'], errors='coerce') >= BREAKDOWN_PX


def populations(frame: pd.DataFrame, horizon: Optional[int] = None) -> pd.Series:
    """
    What each captured measurement is: a `pull`, within the offsets up to `horizon`; a `breakdown`,
    at any offset; or `other`. Empty for a measurement that is not captured.
    """
    kinds = pd.Series('', index=frame.index, dtype=object)
    hit = captured(frame)
    kinds[hit] = 'other'
    if horizon is not None:
        kinds[hit & (pd.to_numeric(frame['abs_offset'], errors='coerce') <= horizon)] = 'pull'
    kinds[hit & broken_down(frame)] = 'breakdown'
    return kinds


def nearest_offset_ratios(frame: pd.DataFrame, neighbors: int = NEAREST_OFFSETS) -> pd.DataFrame:
    """
    Each captured measurement's inlier count and ratio over the means of its case's nearest offsets
    on either side that are neither captured nor broken down. Inliers fall with the offset, so the
    comparison is local. `frame` holds one configuration.
    """
    rows: List[Dict[str, Any]] = []
    for _, case in frame.groupby('case', sort=False):
        ordered = case.sort_values('abs_offset', kind='stable')
        hit = captured(ordered)
        clean = ordered[~hit & ~broken_down(ordered)]
        for index, measurement in ordered[hit].iterrows():
            offset = measurement['abs_offset']
            before = clean[clean['abs_offset'] < offset].tail(neighbors)
            after = clean[clean['abs_offset'] > offset].head(neighbors)
            nearest = pd.concat([before, after])
            if len(nearest):
                rows.append(
                    {
                        'index': index,
                        'inliers_nearest': measurement['n_inliers'] / nearest['n_inliers'].mean(),
                        'ratio_nearest': measurement['inlier_ratio'] / nearest['inlier_ratio'].mean(),
                    }
                )
    return pd.DataFrame(rows, columns=['index', 'inliers_nearest', 'ratio_nearest']).set_index('index')


def group_errors(measurements: pd.DataFrame, results_dirs: Sequence[Path]) -> pd.DataFrame:
    """
    `diagnostics.group_error` of each measurement, its group found again from the case's frames, as
    the campaign found it. Rounded, since the frames are read through interpolation that differs in
    its last bits between machines.
    """
    from stabilo_benchmark.oracle import load_oracle, oracle_matrices  # noqa: PLC0415
    from stabilo_benchmark.results import ResultStore  # noqa: PLC0415
    from stabilo_benchmark.runners.dynamic import pair_object_motion  # noqa: PLC0415
    from stabilo_benchmark.sampling import FramePair  # noqa: PLC0415

    directories = {_case_of(directory): Path(directory) for directory in results_dirs}
    rows: List[Dict[str, Any]] = []
    for case, measured in measurements.groupby('case', sort=False):
        with open(ResultStore(directories[str(case)]).manifest_path, 'r', encoding='utf-8') as handle:
            case_dir = _case_dir(directories[str(case)], json.load(handle))
        matrices = oracle_matrices(load_oracle(case_dir))
        first = measured.drop_duplicates('pair')
        pairs = [
            FramePair(int(ref), int(cur))
            for ref, cur in zip(first['reference_index'], first['current_index'], strict=True)
        ]
        motions = pair_object_motion(case_dir, pairs, matrices)
        for index, measurement in measured.iterrows():
            motion = motions[str(measurement['pair'])]
            length, angle = group_error(
                np.asarray(measurement['homography'], np.float64),
                matrices[str(measurement['pair'])],
                motion.current_centers[motion.group],
                motion.displacement[motion.group],
            )
            rows.append({'index': index, 'error_group': round(length, 3), 'error_angle': round(angle, 3)})
    return pd.DataFrame(rows, columns=['index', 'error_group', 'error_angle']).set_index('index')


def capture_populations(
    candidates: pd.DataFrame, horizon: Optional[int] = None, errors: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    One row per configuration ever captured and kind of captured estimate (`populations`). `errors`
    holds `group_errors` of the captured measurements, when the frames could be read.
    """
    rows: List[Dict[str, Any]] = []
    for (pipeline, masking), measured in candidates.groupby(['pipeline', 'masking'], sort=False):
        kinds = populations(measured, horizon)
        ratios = nearest_offset_ratios(measured)
        for kind in POPULATIONS:
            part = measured[kinds == kind]
            if part.empty:
                continue
            relative = ratios.reindex(part.index)
            located = errors.reindex(part.index) if errors is not None else pd.DataFrame(index=part.index)
            rows.append(
                {
                    'pipeline': pipeline,
                    'masking': masking,
                    'population': kind,
                    'n_captured': len(part),
                    'n_cases': int(part['case'].nunique()) if 'case' in part else 1,
                    'offset_max': int(part['abs_offset'].max()),
                    'capture_fraction_max': _maximum(part, 'capture_fraction'),
                    'error_group': _median(located, 'error_group'),
                    'error_angle_max': _maximum(located, 'error_angle'),
                    'cte_median': _median(part, 'cte_mean'),
                    'cte_min': _minimum(part, 'cte_mean'),
                    'cte_max': _maximum(part, 'cte_mean'),
                    'inliers_median': _median(part, 'n_inliers'),
                    'inliers_max': _maximum(part, 'n_inliers'),
                    'ratio_max': _maximum(part, 'inlier_ratio'),
                    'inliers_nearest': _median(relative, 'inliers_nearest'),
                    'ratio_nearest': _median(relative, 'ratio_nearest'),
                }
            )
    return pd.DataFrame(rows)


def write_populations_table(
    table: pd.DataFrame, path: Path, label: str, n_cases: int, horizon: Optional[int] = None
) -> Path:
    """The kinds of captured estimate, as a table and a CSV beside it."""
    caption = (
        POPULATION_CAPTION.replace('<cases>', 'the one case' if n_cases == 1 else f"the {n_cases} cases")
        .replace('<horizon>', str(horizon))
        .replace('<breakdown>', f"{BREAKDOWN_PX:g}")
        .replace('<nearest>', {1: 'one', 2: 'two', 3: 'three'}.get(NEAREST_OFFSETS, str(NEAREST_OFFSETS)))
    )
    return write_latex_table(
        table,
        path,
        columns=POPULATION_COLUMNS,
        label_columns=('pipeline', 'masking', 'population'),
        caption=caption,
        label=label,
        tie_sets={},
        highlight_best=False,
        csv=True,
    )


def write_capture_table(
    table: pd.DataFrame, path: Path, label: str, n_cases: int, csv_name: str, identity_largest: Optional[float] = None
) -> Path:
    """
    The configurations captured on at least one pair, and a caption sentence on the others: how many,
    their largest capture fraction, and the largest of any masked estimate when none is captured.
    Every configuration is listed when none is captured. `identity_largest` adds what a transform
    without features reaches, beside the smallest displacement of any group.
    """
    hit = pd.to_numeric(table['n_captured'], errors='coerce').fillna(0) > 0
    shown, others = (table[hit], table[~hit]) if hit.any() else (table, table.iloc[0:0])
    note = ''
    if len(others):
        note = CAPTURE_OTHERS.replace('<n>', str(len(others))).replace(
            '<largest>', f"{_maximum(others, 'capture_fraction_max'):.2f}"
        )
    masked = table[table['masking'] != 'none']
    if len(masked) and not (pd.to_numeric(masked['n_captured'], errors='coerce').fillna(0) > 0).any():
        note += CAPTURE_MASKED.replace('<masked>', f"{_maximum(masked, 'capture_fraction_max'):.2f}")
    if identity_largest is not None and np.isfinite(identity_largest):
        shift = _minimum(table, 'group_shift_min')
        note += CAPTURE_IDENTITY.replace('<identity>', f"{identity_largest:.2f}").replace('<shift>', f"{shift:.0f}")
    caption = (
        CAPTURE_CAPTION.replace('<min_objects>', _min_objects())
        .replace('<cases>', 'the one case' if n_cases == 1 else f"the {n_cases} cases")
        .replace('<threshold>', f"{CAPTURE_THRESHOLD:.0%}".replace('%', '\\%'))
        .replace('<listed>', ', the configurations captured on at least one pair' if hit.any() else '')
        .replace('<lambda>', f"{CAPTURE_THRESHOLD:g}")
        .replace('<others>', note)
        .replace('<csv>', csv_name.replace('_', '\\_'))
    )
    return write_latex_table(
        shown,
        path,
        columns=CAPTURE_COLUMNS,
        caption=caption,
        label=label,
        tie_sets={},
        highlight_best=False,
        omit=CAPTURE_OMIT,
    )


def _ordered_presets(presets: Sequence[str]) -> List[str]:
    """Presets in the order of `names.PIPELINE_NAMES` (classical, then learned), unknown ones last."""
    known = list(PIPELINE_NAMES)
    return sorted(dict.fromkeys(presets), key=lambda preset: known.index(preset) if preset in known else len(known))


def figure_capture_by_offset(table: pd.DataFrame, path: Path) -> Path:
    """
    The captured share against the offset bin, one line per configuration ever captured, in one
    panel; the configurations with the most captures when they outnumber the palette. A `regime`
    column draws several regimes in the panel, a configuration keeping its color and marker and a
    regime its line style; a regime after the first is drawn on top with open markers, which leave a
    coinciding marker visible. Without regimes, the legend's title states a masking mode shared by
    every line.
    """
    if table.empty or not (table['n_captured'] > 0).any():
        raise ValueError('no measurement was captured')
    apply_style()
    table = table if 'regime' in table else table.assign(regime='')
    totals = table.groupby(['regime', 'pipeline', 'masking'], sort=False)['n_captured'].sum()
    shown = [key for key in totals.sort_values(ascending=False, kind='stable').index if totals[key] > 0]
    dropped = max(0, len(shown) - len(PALETTE))
    shown = shown[: len(PALETTE)]
    configurations = list(dict.fromkeys(key[1:] for key in shown))
    shown.sort(key=lambda key: configurations.index(key[1:]))
    regimes = list(dict.fromkeys(key[0] for key in shown))
    modes = {masking for _, _, masking in shown}
    bins = sorted(table['offset_bin'].unique(), key=lambda name: int(str(name).split('-')[0]))
    position = {name: index for index, name in enumerate(bins)}
    figure, axes = plt.subplots(figsize=(5.2, 2.3))
    for regime, pipeline, masking in shown:
        chosen = (table['regime'] == regime) & (table['pipeline'] == pipeline) & (table['masking'] == masking)
        series = table[chosen].assign(position=table['offset_bin'].map(position)).sort_values('position')
        style = _style_for(configurations.index((pipeline, masking)))
        rank = regimes.index(regime)
        words = [pipeline_name(preset_of(pipeline))]
        words += [MASKING_LABELS[masking].lower()] if len(modes) > 1 else []
        words += [regime] if regime else []
        axes.plot(
            series['position'],
            series['captured_share'],
            color=style['color'],
            marker=style['marker'],
            markerfacecolor='none' if rank else style['color'],
            linestyle=style['linestyle'] if len(regimes) == 1 else ('-', '--', ':')[rank],
            markersize=6 if rank else 3.5,
            linewidth=1.3 if rank else 1.1,
            zorder=3 + rank,
            label=', '.join(words),
        )
    axes.set_ylim(-0.02, min(1.02, 1.15 * float(table['captured_share'].max()) + 0.02))
    axes.set_xticks(range(len(bins)), bins, rotation=45, ha='right')
    axes.set_xlabel('Offset [frames]')
    axes.set_ylabel('Captured share')
    if dropped:
        axes.set_title(f"{dropped} more in the CSV", fontsize=8, color=INK_SECONDARY)
    title = MASKING_LABELS[next(iter(modes))] if len(modes) == 1 and not any(regimes) else None
    axes.legend(loc='upper center', fontsize=7, title=title, title_fontsize=7)
    figure.tight_layout()
    return _save(figure, path)


def figure_capture_scatter(
    candidates: pd.DataFrame, path: Path, horizon: Optional[int] = None, threshold: float = CAPTURE_THRESHOLD
) -> Path:
    """
    Every measurement with a group: its capture fraction against the offset, a panel per pipeline
    (`PANEL_COLUMNS` to a row), a marker per masking mode, the threshold dashed and the survey's
    offsets (up to `horizon`) shaded. The capture fraction is drawn on a symmetric log scale, linear
    within the threshold, so a partial pull, a complete capture and a breakdown far from zero are all
    legible; the axis spans every measurement.
    """
    data = with_group(candidates)
    if 'abs_offset' not in data or data.empty:
        raise ValueError('no measurement on a pair with a moving group')
    apply_style()
    data = data.assign(preset=data['pipeline'].map(preset_of))
    pipelines = _ordered_presets(data['preset'])
    columns = min(PANEL_COLUMNS, len(pipelines))
    rows = int(np.ceil(len(pipelines) / columns))
    fractions = pd.to_numeric(data['capture_fraction'])
    upper = max(1.5, 1.3 * float(fractions.max()))
    lower = min(-threshold, 1.3 * float(fractions.min()))
    # An axis reaching far below zero spends height on it, which the panels get back.
    panel = 2.1 if lower < -1.0 else 1.55
    height = panel * rows + 0.55
    figure, grid = plt.subplots(rows, columns, figsize=(7.2, height), sharex=True, sharey=True, squeeze=False)
    modes = _masking_series(data)
    ticks = [t for t in (-10.0, -1.0, 0.0, threshold, 1.0, 5.0, 10.0) if lower <= t <= upper]
    for slot, axes in enumerate(grid.flat):
        if slot >= len(pipelines):
            axes.set_visible(False)
            continue
        subset = data[data['preset'] == pipelines[slot]]
        if horizon:
            axes.axvspan(0.8, horizon, color=INK_MUTED, alpha=0.12, linewidth=0)
        axes.axhline(threshold, color=INK_SECONDARY, linestyle='--', linewidth=0.7)
        for mode in modes:
            series = subset[subset['masking'] == mode]
            style = _style_for(MASKING_ORDER.index(mode), scatter=True)
            axes.scatter(
                pd.to_numeric(series['abs_offset']),
                pd.to_numeric(series['capture_fraction']),
                s=4,
                marker=style['marker'],
                color=style['color'],
                alpha=0.6,
                linewidths=0,
                zorder=3 if mode == 'none' else 2,
            )
        hits = int(captured(subset, threshold).sum())
        axes.set_xscale('log')
        axes.set_yscale('symlog', linthresh=threshold)
        axes.set_ylim(lower, upper)
        axes.set_yticks(ticks, [f"{t:g}" for t in ticks])
        axes.minorticks_off()
        axes.set_title(f"{pipeline_name(pipelines[slot])} ({hits})", fontsize=7.5, color=INK_SECONDARY)
        if slot % columns == 0:
            axes.set_ylabel(r'Capture fraction $\lambda$')
        if slot >= len(pipelines) - columns:
            axes.set_xlabel('Offset [frames]')
            axes.tick_params(labelbottom=True)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle='none',
            color=_style_for(MASKING_ORDER.index(mode), scatter=True)['color'],
            marker=_style_for(MASKING_ORDER.index(mode), scatter=True)['marker'],
            label=MASKING_LABELS[mode],
        )
        for mode in modes
    ]
    handles.append(plt.Line2D([], [], color=INK_SECONDARY, linestyle='--', label=f"threshold {threshold:g}"))
    if horizon:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=INK_MUTED, alpha=0.25, label=f"offsets 1 to {horizon}"))
    figure.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0), fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.3 / height))
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# The example pair
# --------------------------------------------------------------------------------------


def example_candidates(candidates: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    Captured unmasked measurements of every pipeline, largest capture fraction first; only those at
    offsets up to `horizon` when any is.
    """
    pool = with_group(candidates)
    pool = pool[(pool['masking'] == 'none') & captured(pool)]
    if horizon is not None:
        near = pool[pd.to_numeric(pool['abs_offset'], errors='coerce') <= horizon]
        pool = near if len(near) else pool
    return pool.sort_values('capture_fraction', ascending=False, kind='stable')


def figure_capture_example(
    candidates: pd.DataFrame,
    results_dirs: Sequence[Path],
    path: Path,
    horizon: Optional[int] = None,
    abstract_path: Optional[Path] = None,
) -> Path:
    """
    One captured pair, re-registered: the unmasked estimate beside the oriented-box-masked one, from
    offsets up to `horizon` (the survey's) when a capture lies there. With `abstract_path`, the same
    pair is also drawn as a graphical abstract (`_draw_graphical_abstract`).

    The ranked candidates are re-registered until one reproduces its stored estimate within
    REPRODUCTION_TOLERANCE_PX (a learned detector on another machine, or a parallel estimator, may
    not), so the drawn inliers are those of the reported measurement.
    """
    ranked = example_candidates(candidates, horizon)
    if ranked.empty:
        raise ValueError('no measurement was captured')
    directories = {_case_of(directory): Path(directory) for directory in results_dirs}
    reasons: List[str] = []
    for row in ranked.head(EXAMPLE_ATTEMPTS).itertuples():
        results_dir = directories.get(str(row.case))
        if results_dir is None:
            reasons.append(f"{row.case}: results not given")
            continue
        try:
            panels = _reregister(results_dir, row)
        except (ValueError, OSError, KeyError) as error:
            reasons.append(f"{row.case} {row.pipeline} {row.pair}: {error}")
            continue
        if abstract_path is not None:
            _draw_graphical_abstract(panels, abstract_path)
        return _draw_example(panels, path)
    raise ValueError('no captured measurement could be reproduced here (' + '; '.join(reasons[:3]) + ')')


def _case_of(results_dir: Path) -> str:
    from stabilo_benchmark.report.tables import case_name  # noqa: PLC0415 - avoids import cycle

    return case_name(Path(results_dir))


def _reregister(results_dir: Path, row: Any) -> Dict[str, Any]:
    """Register the row's pair again with its pipeline, unmasked and with oriented boxes."""
    from stabilo_benchmark.oracle import (  # noqa: PLC0415
        boxes_for,
        load_frame,
        load_frame_labels,
        load_oracle,
        oracle_matrices,
    )
    from stabilo_benchmark.pipelines import PipelineSpec  # noqa: PLC0415
    from stabilo_benchmark.results import ResultStore  # noqa: PLC0415
    from stabilo_benchmark.runners.dynamic import object_index_map, pair_object_motion  # noqa: PLC0415
    from stabilo_benchmark.sampling import FramePair  # noqa: PLC0415
    from stabilo_benchmark.seeding import seed_estimator  # noqa: PLC0415

    store = ResultStore(results_dir)
    with open(store.manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    case_dir = _case_dir(results_dir, manifest)
    pair = FramePair(int(row.reference_index), int(row.current_index))
    reference_matrix = oracle_matrices(load_oracle(case_dir)).get(str(row.pair))
    if reference_matrix is None:
        raise ValueError('no reference estimate')
    motion = pair_object_motion(case_dir, [pair], {str(row.pair): reference_matrix})[str(row.pair)]
    reference = load_frame(case_dir, pair.reference_index)
    current = load_frame(case_dir, pair.current_index)
    height, width = reference.shape[:2]
    reference_labels = load_frame_labels(case_dir, pair.reference_index, width, height)
    current_labels = load_frame_labels(case_dir, pair.current_index, width, height)
    seed = int((manifest.get('config') or {}).get('seed', 0))

    stored = [run for run in store.iter_runs() if run.get('records')]
    unmasked = next((run for run in stored if run['params'].get('pipeline') == row.pipeline), None)
    if unmasked is None:
        raise ValueError('the unmasked run is missing')
    # The same pipeline under each masking mode: every other parameter equal.
    runs = {
        run['params'].get('masking'): run
        for run in stored
        if _shared(run['params']) == _shared(unmasked['params']) or run is unmasked
    }
    panels: Dict[str, Any] = {
        'reference': reference,
        'reference_labels': reference_labels,
        'reference_matrix': np.asarray(reference_matrix, dtype=np.float64),
        'motion': motion,
        'object_index': object_index_map(reference_labels, (height, width)),
        'case': str(row.case) if hasattr(row, 'case') else case_dir.name,
        'offset': abs(pair.current_index - pair.reference_index),
        'runs': [],
    }
    for masking in [mode for mode in ('none', 'obb', 'hbb') if mode in runs][:2]:
        run = runs[masking]
        params = dict(run['params'])
        spec = PipelineSpec(name=params.pop('pipeline'), masking=params.pop('masking'), params=params)
        record = next(r for r in run['records'] if r.get('pair') == row.pair and int(r.get('repeat', 0)) == 0)
        stored = record.get('homography')
        if stored is None:
            raise ValueError(f"no stored estimate under {masking}")
        stabilizer = spec.build()
        box_format = spec.box_format or 'xywh'
        stabilizer.set_ref_frame(reference, boxes_for(reference_labels, spec.masking), box_format=box_format)
        seed_estimator(seed, case_dir.name, str(row.pair), 0)
        stabilizer.stabilize(current, boxes_for(current_labels, spec.masking), box_format=box_format)
        estimate = np.asarray(stabilizer.get_cur_trans_matrix(), dtype=np.float64)
        drift = _corner_difference(estimate, np.asarray(stored, dtype=np.float64), width, height)
        if not drift < REPRODUCTION_TOLERANCE_PX:
            raise ValueError(f"re-registration differs from the stored estimate by {drift:.3f} px")
        inliers = np.asarray(stabilizer.cur_inliers, dtype=bool).ravel()
        points = np.asarray(stabilizer.ref_pts, dtype=np.float64).reshape(-1, 2)
        panels['runs'].append({'spec': spec, 'record': record, 'estimate': estimate, 'inliers': points[inliers]})
    return panels


def _shared(params: Mapping[str, Any]) -> Dict[str, Any]:
    """A run's parameters other than its name and masking."""
    return {key: value for key, value in params.items() if key not in ('pipeline', 'masking')}


def _case_dir(results_dir: Path, manifest: Mapping[str, Any]) -> Path:
    recorded = ((manifest.get('metadata') or {}).get('case') or {}).get('dir')
    for candidate in (Path(results_dir).parent.parent, *([Path(recorded)] if recorded else [])):
        if (candidate / 'oracle').is_dir():
            return candidate
    raise ValueError('the case directory is not here')


def _corner_difference(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float64)
    pa, pb = corners @ a.T, corners @ b.T
    return float(np.mean(np.linalg.norm(pa[:, :2] / pa[:, 2:] - pb[:, :2] / pb[:, 2:], axis=1)))


def _crop(panels: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    """A window around the moving group, padded and widened to the frame's aspect."""
    motion = panels['motion']
    height, width = panels['reference'].shape[:2]
    corners = panels['reference_labels'].corners()[motion.group].reshape(-1, 2)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    size = np.maximum((hi - lo) * 1.4, [width / 4, height / 4])
    size[0] = max(size[0], size[1] * width / height)
    size[1] = max(size[1], size[0] * height / width)
    center = (lo + hi) / 2
    x0, y0 = np.clip(center - size / 2, 0, [width - size[0], height - size[1]])
    return int(x0), int(y0), int(min(width, x0 + size[0])), int(min(height, y0 + size[1]))


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.c_[points, np.ones(len(points))] @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def _example_geometry(panels: Mapping[str, Any]) -> Dict[str, Any]:
    """The crop, the error grid of every run and the magnification shared by the example's panels."""
    x0, y0, x1, y1 = _crop(panels)
    grid_x, grid_y = np.meshgrid(np.linspace(x0, x1, 9)[1:-1], np.linspace(y0, y1, 6)[1:-1])
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    current_grid = _project(np.linalg.inv(panels['reference_matrix']), grid)
    errors = [_project(run['estimate'], current_grid) - grid for run in panels['runs']]
    largest = max(float(np.max(np.linalg.norm(error, axis=1))) for error in errors)
    return {
        'crop': (x0, y0, x1, y1),
        'gray': cv2.cvtColor(panels['reference'], cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1],
        'grid': grid,
        'errors': errors,
        'scale': max(1.0, round((x1 - x0) / 10 / max(largest, 1e-9))),
    }


def _draw_panel(
    axes: Any,
    panels: Mapping[str, Any],
    run: Mapping[str, Any],
    error: np.ndarray,
    geometry: Mapping[str, Any],
    sizes: Mapping[str, float],
) -> None:
    """One example panel: the crop, every box (the moving group in color, with its motion), inliers, error grid."""
    x0, y0, x1, y1 = geometry['crop']
    motion = panels['motion']
    axes.imshow(geometry['gray'], cmap='gray', vmin=-80, vmax=255, extent=(x0, x1, y1, y0))
    for index, quad in enumerate(panels['reference_labels'].corners()):
        closed = np.vstack([quad, quad[:1]])
        moving = bool(motion.group[index])
        axes.plot(*closed.T, color=PALETTE[1] if moving else INK_MUTED, linewidth=sizes['box'] * (2 if moving else 1))
    group = np.flatnonzero(motion.group)
    axes.quiver(
        *motion.reference_centers[group].T,
        *motion.displacement[group].T,
        color=PALETTE[1],
        angles='xy',
        scale_units='xy',
        scale=1,
        width=sizes['arrow'],
    )
    points = run['inliers']
    inside = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    columns = np.clip(points[:, 0].astype(int), 0, panels['object_index'].shape[1] - 1)
    rows = np.clip(points[:, 1].astype(int), 0, panels['object_index'].shape[0] - 1)
    covering = panels['object_index'][rows, columns]
    on_moving = covering >= 0
    on_moving[on_moving] = motion.moved[covering[on_moving]]
    axes.scatter(*points[inside & ~on_moving].T, s=sizes['inlier'], color=PALETTE[0], linewidths=0)
    axes.scatter(*points[inside & on_moving].T, s=1.75 * sizes['inlier'], color=PALETTE[1], linewidths=0)
    axes.quiver(
        *geometry['grid'].T,
        *(error * geometry['scale']).T,
        color=INK_PRIMARY,
        angles='xy',
        scale_units='xy',
        scale=1,
        width=1.25 * sizes['arrow'],
    )
    axes.set_xlim(x0, x1)
    axes.set_ylim(y1, y0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)


def _example_handles(scale: float, markersize: float = 3.0) -> List[Any]:
    return [
        plt.Line2D([], [], color=PALETTE[1], linewidth=1.2, label='objects that moved together, and their motion'),
        plt.Line2D(
            [], [], color=PALETTE[0], marker='o', linestyle='none', markersize=markersize, label='inlier elsewhere'
        ),
        plt.Line2D(
            [],
            [],
            color=PALETTE[1],
            marker='o',
            linestyle='none',
            markersize=markersize,
            label='inlier on a moved object',
        ),
        plt.Line2D([], [], color=INK_PRIMARY, linewidth=1.2, label=f"estimate error x{scale:g}"),
    ]


def _draw_example(panels: Mapping[str, Any], path: Path) -> Path:
    apply_style()
    geometry = _example_geometry(panels)
    x0, y0, x1, y1 = geometry['crop']
    count = len(panels['runs'])
    figure, axes_row = plt.subplots(1, count, figsize=(3.6 * count, 3.6 * (y1 - y0) / (x1 - x0) + 0.6), squeeze=False)
    sizes = {'box': 0.4, 'arrow': 0.004, 'inlier': 4.0}
    for axes, run, error in zip(axes_row[0], panels['runs'], geometry['errors'], strict=True):
        _draw_panel(axes, panels, run, error, geometry, sizes)
        record = run['record']
        axes.set_title(
            f"{MASKING_LABELS[run['spec'].masking]}: CTE {record['cte_mean']:.1f} px, "
            f"capture fraction {record['capture_fraction']:.2f}\n"
            f"{_number(record.get('n_inliers')):.0f} inliers, "
            f"{_number(record.get('inliers_on_moving')):.0%} on objects that moved",
            fontsize=8,
            color=INK_SECONDARY,
        )
    name = pipeline_name(preset_of(panels['runs'][0]['spec'].name))
    figure.suptitle(
        f"{name}, {panels['case']}, offset {panels['offset']} frames".replace('_', ' '), fontsize=9, color=INK_PRIMARY
    )
    figure.legend(
        handles=_example_handles(geometry['scale']), loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=7
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.97))
    return _save(figure, path)


# The journal's graphical abstract: at least 1328 x 531 px (w x h), readable at 13 x 5 cm.
ABSTRACT_SIZE_IN = (6.64, 2.655)
ABSTRACT_DPI = 400


def _draw_graphical_abstract(panels: Mapping[str, Any], path: Path) -> Path:
    """
    The example pair as a graphical abstract: the two estimates side by side, each titled by what it
    follows (the moving objects when captured, the ground otherwise) and by the inlier count that
    does not tell them apart. Written as PDF and, beside it, as PNG at ABSTRACT_DPI.
    """
    apply_style()
    geometry = _example_geometry(panels)
    figure, axes_row = plt.subplots(1, len(panels['runs']), figsize=ABSTRACT_SIZE_IN, squeeze=False)
    sizes = {'box': 0.35, 'arrow': 0.005, 'inlier': 2.5}
    for axes, run, error in zip(axes_row[0], panels['runs'], geometry['errors'], strict=True):
        _draw_panel(axes, panels, run, error, geometry, sizes)
        # The panels keep the frame's aspect; anchored at the bottom of their slot, they sit on the
        # legend and the unused height ends above the titles, where the tight bounding box trims it.
        axes.set_anchor('S')
        record = run['record']
        fraction = float(record['capture_fraction'])
        follows = 'follows the moving vehicles' if fraction >= CAPTURE_THRESHOLD else 'follows the ground'
        mode = 'No masking' if run['spec'].masking == 'none' else f"{MASKING_LABELS[run['spec'].masking]}"
        axes.set_title(
            f"{mode}: {follows}\n"
            f"{_number(record.get('n_inliers')):.0f} inliers, corner error {record['cte_mean']:.1f} px",
            fontsize=8,
            color=INK_PRIMARY,
        )
    figure.legend(
        handles=_example_handles(geometry['scale'], markersize=2.5),
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=5.5,
        handlelength=1.4,
        columnspacing=1.0,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.6)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix('.png'), dpi=ABSTRACT_DPI)
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Report entry point
# --------------------------------------------------------------------------------------


def build_capture_outputs(
    produced: Dict[str, Any],
    candidates: pd.DataFrame,
    results_dirs: Sequence[Path],
    tables_dir: Path,
    figures_dir: Path,
    prefix: str,
    bins: Sequence[Tuple[int, Optional[int]]],
    survey_bins: Optional[Sequence[Tuple[int, Optional[int]]]] = None,
    identity: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write the capture table, the per-offset share, the populations and the example pair, or say why
    not. The example is drawn from the offsets of `survey_bins`, the ones the case selection reads
    coherence at. `identity` holds the identity transform's measurements, which the CSV lists.
    """
    if 'capture_fraction' not in candidates:
        produced['skipped'].append('capture: the results predate the capture fraction')
        return
    n_cases = int(candidates['case'].nunique()) if 'case' in candidates else 1
    candidates = with_gcp_errors(candidates, results_dirs)
    table = capture_table(candidates, bins)
    horizon = max((int(high) for _, high in survey_bins or () if high is not None), default=None)
    csv_path = tables_dir / f"{prefix}_capture.csv"
    offsets_csv = tables_dir / f"{prefix}_capture_by_offset.csv"
    cases_csv = tables_dir / f"{prefix}_capture_by_case.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = capture_table(identity) if identity is not None and len(identity) else table.iloc[0:0]
    pd.concat([table, baseline], ignore_index=True).to_csv(csv_path, index=False)
    capture_by_case(candidates).to_csv(cases_csv, index=False)
    by_offset = capture_by_offset(candidates, bins)
    by_offset.drop(columns='bin_rank', errors='ignore').to_csv(offsets_csv, index=False)
    produced['tables'] += [
        str(
            write_capture_table(
                table,
                tables_dir / f"{prefix}_capture.tex",
                f"tab:{prefix}-capture",
                n_cases,
                offsets_csv.name,
                _maximum(baseline, 'capture_fraction_max') if len(baseline) else None,
            )
        ),
        str(csv_path),
        str(offsets_csv),
        str(cases_csv),
    ]
    _write_populations(produced, candidates, results_dirs, tables_dir, prefix, n_cases, horizon)
    for name, build in (
        (f"{prefix}_capture_by_offset.pdf", lambda target: figure_capture_by_offset(by_offset, target)),
        (f"{prefix}_capture_scatter.pdf", lambda target: figure_capture_scatter(candidates, target, horizon)),
        (
            f"{prefix}_capture_example.pdf",
            lambda target: figure_capture_example(
                candidates, results_dirs, target, horizon, figures_dir / f"{prefix}_graphical_abstract.pdf"
            ),
        ),
    ):
        try:
            produced['figures'].append(str(build(figures_dir / name)))
        except ValueError as error:
            produced['skipped'].append(f"{name}: {error}")


def _write_populations(
    produced: Dict[str, Any],
    candidates: pd.DataFrame,
    results_dirs: Sequence[Path],
    tables_dir: Path,
    prefix: str,
    n_cases: int,
    horizon: Optional[int],
) -> None:
    """The populations table, with the error at the group when the case frames can be read."""
    hit = with_group(candidates)
    hit = hit[captured(hit)]
    if hit.empty:
        produced['skipped'].append(f"{prefix}_capture_populations: no measurement was captured")
        return
    errors = None
    try:
        errors = group_errors(hit, results_dirs)
    except (ValueError, OSError, KeyError) as error:
        produced['skipped'].append(f"{prefix}_capture_populations, the error at the group: {error}")
    table = capture_populations(candidates, horizon, errors)
    path = tables_dir / f"{prefix}_capture_populations.tex"
    produced['tables'] += [
        str(write_populations_table(table, path, f"tab:{prefix}-capture-populations", n_cases, horizon)),
        str(path.with_suffix('.csv')),
    ]


def _median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float('nan')
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float('nan')


def _number(value: Any) -> float:
    return float('nan') if value is None else float(value)


def _maximum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float('nan')
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else float('nan')


def _minimum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float('nan')
    values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else float('nan')


def _min_objects() -> str:
    from stabilo_benchmark.scene import MIN_DISPLACED_OBJECTS  # noqa: PLC0415

    words = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
    return words.get(MIN_DISPLACED_OBJECTS, str(MIN_DISPLACED_OBJECTS))
