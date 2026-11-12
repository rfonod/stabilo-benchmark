#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
contrasts.py - The planned masking contrasts of Part II, read across cases.

Masking is a within-pair factor and the case is the independent unit, so every contrast is a
paired difference formed within a case and read across cases with `stats.paired_comparison`
(Hodges-Lehmann estimate, cluster bootstrap interval, exact sign-flip p).

Masking contrasts: oriented against none (primary), horizontal against none and oriented against
horizontal (secondary), on the outcomes

    corner transfer error against the GCP-fitted transform     (audited frames)
    ZNCC and gradient-orientation agreement over the region     (all pairs, reference-free)
    IoU and OCE against the GCP-fitted transform               (audited frames)

Each contrast is read once over all pipelines (per case, the mean benefit over one set of
pipelines measured on every outcome in every case), with Holm across the outcomes, and once per
pipeline, with Holm across pipelines within an outcome. An outcome favors the first mode when its
interval excludes zero, its estimate reaches the margin and its adjusted p is at most `ALPHA`; it
harms it under the same rule with the unadjusted p, which can only withhold a claim. The primary
contrast is claimed when the pooled reading is complete (every outcome, every case), at least one
outcome favors oriented masking and none harms it; per-pipeline rows carry no claim.

Mechanism contrasts on the per-pair oriented-against-none benefit in the reference-free outcomes,
each read over all pipelines (per case, the mean over one set of pipelines measured in every case,
with Holm across the contrast's outcomes) and per pipeline (Holm across pipelines):

    coverage      Spearman correlation of the benefit with the current frame's object coverage
                  (predicted positive);
    persistence   benefit in moving pairs minus benefit in stationary pairs at matched coverage:
                  within a case, pairs are split at the case's median coverage and, within each
                  half, at that half's median persistence; the two halves are averaged
                  (predicted positive);
    alignment     motion alignment without masking minus with oriented masking (predicted
                  positive).

The persistence contrast is repeated at matched offset (`persistence_offset`): pairs are split at the
median persistence within each offset bin and the gaps averaged over the bins. Persistence falls with
offset, so this reading separates the two; it is a sensitivity reading and never favors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stabilo_benchmark.report.names import pipeline_name
from stabilo_benchmark.stats import holm_adjust, paired_comparison

ALPHA = 0.05

# Mechanism contrasts read for sensitivity only: reported with their intervals, never favored.
SENSITIVITY_CONTRASTS = ('persistence_offset',)

# (first, second, role): the benefit is that of the first mode over the second.
MASKING_CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ('obb', 'none', 'primary'),
    ('hbb', 'none', 'secondary'),
    ('obb', 'hbb', 'secondary'),
)

# Outcome -> (header, higher is better, margin source). A margin source of 'annotation' takes the
# GCP repeatability; None means no margin.
OUTCOMES: Dict[str, Tuple[str, bool, Optional[str]]] = {
    'cte_gcp': (r'CTE$^{\mathrm{GCP}}_{50}$', False, 'annotation'),
    'zncc': ('ZNCC', True, None),
    'gradient': ('GOS', True, None),
    'iou_gcp': (r'IoU$^{\mathrm{GCP}}$', True, None),
    'oce_gcp': (r'OCE$^{\mathrm{GCP}}$', False, 'annotation'),
}
# Display of a difference: (scale, format, unit suffix for the header).
DISPLAY: Dict[str, Tuple[float, str, str]] = {
    'cte_gcp': (1.0, '{:+.2f}', ' [px]'),
    'oce_gcp': (1.0, '{:+.2f}', ' [px]'),
    'zncc': (1e3, '{:+.2f}', r' [$10^{-3}$]'),
    'gradient': (1e3, '{:+.2f}', r' [$10^{-3}$]'),
    'iou_gcp': (1e3, '{:+.2f}', r' [$10^{-3}$]'),
    'motion_alignment': (1.0, '{:+.3f}', ''),
    'coverage': (1.0, '{:+.2f}', ''),
}
# Per-pair outcomes the mechanism contrasts read; reference-free, so not tied to the reference estimator.
PAIR_OUTCOMES = ('zncc', 'gradient')

MASKING_LABELS = {'none': 'none', 'hbb': 'HBB', 'obb': 'OBB'}

# The `pipeline` of a contrast read over all pipelines at once.
POOLED = 'all pipelines'


def preset_of(pipeline: Any) -> str:
    """`sift+obb` and `sift+obb#2` -> `sift`."""
    return str(pipeline).split('#', 1)[0].rsplit('+', 1)[0]


def case_table(frame: pd.DataFrame, gcp_rows: Iterable[Mapping[str, Any]] = ()) -> pd.DataFrame:
    """
    One row per case, pipeline and masking mode, with every outcome summarized within the case.

    `frame` is `load_case_results` output without the baselines; `gcp_rows` carry `case`,
    `pipeline`, `masking`, `cte_median`, `iou_mean` and `oce_mean` against the GCP-fitted transform.
    """
    frame = frame.assign(preset=frame['pipeline'].map(preset_of))
    reference_free = (
        frame.groupby(['case', 'preset', 'masking'], sort=False)[[c for c in PAIR_OUTCOMES if c in frame]]
        .mean(numeric_only=True)
        .reset_index()
    )
    gcp = pd.DataFrame(list(gcp_rows))
    if gcp.empty:
        return reference_free.assign(cte_gcp=np.nan, iou_gcp=np.nan, oce_gcp=np.nan)
    gcp = gcp.assign(preset=gcp['pipeline'].map(preset_of)).rename(
        columns={'cte_median': 'cte_gcp', 'iou_mean': 'iou_gcp', 'oce_mean': 'oce_gcp'}
    )
    return reference_free.merge(
        gcp[['case', 'preset', 'masking', 'cte_gcp', 'iou_gcp', 'oce_gcp']],
        on=['case', 'preset', 'masking'],
        how='left',
    )


def _benefits(table: pd.DataFrame, outcome: str, first: str, second: str) -> pd.DataFrame:
    """Per case and pipeline, the benefit of `first` over `second` in `outcome` (positive favors `first`)."""
    higher_is_better = OUTCOMES[outcome][1]
    wide = table.pivot_table(index=['preset', 'case'], columns='masking', values=outcome, aggfunc='mean')
    if first not in wide or second not in wide:
        return pd.DataFrame(columns=['preset', 'case', 'benefit'])
    benefit = wide[first] - wide[second]
    benefit = benefit if higher_is_better else -benefit
    return benefit.rename('benefit').reset_index()


def _pooled_benefits(per_outcome: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Per outcome and case, the mean benefit over the pipelines measured on every outcome in every
    case that outcome has data for, so every outcome averages the same pipelines.
    """
    wide: Dict[str, pd.DataFrame] = {}
    common: Optional[set] = None
    for outcome, benefits in per_outcome.items():
        table = benefits.pivot_table(index='case', columns='preset', values='benefit', aggfunc='mean')
        table = table.dropna(axis=0, how='all')
        wide[outcome] = table
        complete = set(table.columns[table.notna().all(axis=0)])
        common = complete if common is None else common & complete
    if not common:
        return {}
    ordered = [preset for preset in next(iter(wide.values())).columns if preset in common]
    return {outcome: table[ordered].mean(axis=1) for outcome, table in wide.items()}


def _decide(family: List[Dict[str, Any]]) -> None:
    """Holm-adjust a family of comparisons; mark it in favor (adjusted p) or against (unadjusted p)."""
    for row, adjusted in zip(family, holm_adjust([r['p_value'] for r in family]), strict=True):
        row['p_holm'] = adjusted
        row['favors'] = bool(row['distinguishable'] and row['estimate'] > 0 and adjusted <= ALPHA)
        row['harms'] = bool(row['distinguishable'] and row['estimate'] < 0 and row['p_value'] <= ALPHA)


def masking_contrasts(table: pd.DataFrame, annotation_floor: Optional[float] = None) -> pd.DataFrame:
    """
    Every planned masking contrast, over all pipelines and per pipeline, per outcome, across cases.

    Long format: `contrast`, `role`, `pipeline` (`POOLED` for the reading over all pipelines),
    `outcome`, `n_clusters`, `n_agree` (cases favoring the first mode), `estimate`, `ci_low`,
    `ci_high`, `p_value`, `p_holm`, `margin`, `favors` / `harms` (see the module docstring),
    `complete` (pooled primary rows: every outcome read on every case) and `claim`.
    """
    rows: List[Dict[str, Any]] = []
    presets = list(dict.fromkeys(table['preset']))
    n_cases = int(table['case'].nunique())
    for first, second, role in MASKING_CONTRASTS:
        base_contrast = {'contrast': f"{first} vs {second}", 'role': role}
        per_outcome: Dict[str, pd.DataFrame] = {}
        margins: Dict[str, float] = {}
        for outcome, (_, _, margin_source) in OUTCOMES.items():
            if outcome not in table or table[outcome].isna().all():
                continue
            margins[outcome] = float(annotation_floor) if margin_source == 'annotation' and annotation_floor else 0.0
            per_outcome[outcome] = benefits = _benefits(table, outcome, first, second)
            family: List[Dict[str, Any]] = []
            for preset in presets:
                values = benefits.loc[benefits['preset'] == preset, 'benefit'].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                family.append(
                    {
                        **base_contrast,
                        'outcome': outcome,
                        'pipeline': preset,
                        'n_agree': int((values > 0).sum()),
                        **paired_comparison(values, margin=margins[outcome]).to_dict(),
                    }
                )
            _decide(family)
            rows.extend(family)
        pooled_family: List[Dict[str, Any]] = []
        for outcome, pooled in _pooled_benefits(per_outcome).items():
            values = pooled.to_numpy(dtype=float)
            pooled_family.append(
                {
                    **base_contrast,
                    'outcome': outcome,
                    'pipeline': POOLED,
                    'n_agree': int((values > 0).sum()),
                    **paired_comparison(values, margin=margins[outcome]).to_dict(),
                }
            )
        _decide(pooled_family)
        rows.extend(pooled_family)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result['complete'] = False
    result['claim'] = False
    primary = (result['role'] == 'primary') & (result['pipeline'] == POOLED)
    # The Holm family is every outcome on every case: a reading missing one (no GCP audit yet, a case
    # without audited frames) claims nothing.
    complete = set(result.loc[primary, 'outcome']) == set(OUTCOMES) and bool(
        (result.loc[primary, 'n_clusters'] == n_cases).all()
    )
    result.loc[primary, 'complete'] = complete
    if complete:
        claimed = bool(result.loc[primary, 'favors'].any() and not result.loc[primary, 'harms'].any())
        result.loc[primary, 'claim'] = claimed
    return result


def _pair_benefits(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Per case, pipeline and pair: the oriented-against-none benefit, with the pair's descriptors."""
    frame = frame.assign(preset=frame['pipeline'].map(preset_of))
    keys = ['case', 'preset', 'pair']
    values = frame.pivot_table(index=keys, columns='masking', values=outcome, aggfunc='mean')
    if 'obb' not in values or 'none' not in values:
        return pd.DataFrame(columns=[*keys, 'benefit', 'coverage_cur', 'persistence'])
    sign = 1.0 if OUTCOMES[outcome][1] else -1.0
    benefit = (sign * (values['obb'] - values['none'])).rename('benefit')
    descriptors = frame.groupby(keys)[[c for c in ('coverage_cur', 'persistence', 'abs_offset') if c in frame]].mean()
    return pd.concat([benefit, descriptors], axis=1).reset_index()


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float('nan')
    return float(x[valid].rank().corr(y[valid].rank()))


def _persistence_gap(pairs: pd.DataFrame) -> float:
    """Mean benefit in moving minus stationary pairs, within coverage halves, averaged over the halves."""
    pairs = pairs.dropna(subset=['benefit', 'coverage_cur', 'persistence'])
    if len(pairs) < 4:
        return float('nan')
    halves = pairs['coverage_cur'] >= pairs['coverage_cur'].median()
    gaps = []
    for half in (pairs[halves], pairs[~halves]):
        if len(half) < 2:
            continue
        stationary = half['persistence'] >= half['persistence'].median()
        if stationary.all() or not stationary.any():
            continue
        gaps.append(half.loc[~stationary, 'benefit'].mean() - half.loc[stationary, 'benefit'].mean())
    return float(np.mean(gaps)) if gaps else float('nan')


def _persistence_gap_by_offset(pairs: pd.DataFrame, bins: Sequence[Tuple[int, Optional[int]]]) -> float:
    """Mean benefit in moving minus stationary pairs, within offset bins, averaged over the bins."""
    from stabilo_benchmark.sampling import offset_bin_label  # noqa: PLC0415 - light import

    pairs = pairs.dropna(subset=['benefit', 'persistence', 'abs_offset'])
    labels = pd.Series([offset_bin_label(offset, bins) for offset in pairs['abs_offset']], index=pairs.index)
    gaps = []
    for _, group in pairs[labels.notna()].groupby(labels[labels.notna()], sort=False):
        if len(group) < 2:
            continue
        stationary = group['persistence'] >= group['persistence'].median()
        if stationary.all() or not stationary.any():
            continue
        gaps.append(group.loc[~stationary, 'benefit'].mean() - group.loc[stationary, 'benefit'].mean())
    return float(np.mean(gaps)) if gaps else float('nan')


def mechanism_contrasts(frame: pd.DataFrame, bins: Sequence[Tuple[int, Optional[int]]] = ()) -> pd.DataFrame:
    """
    The three mechanism contrasts, over all pipelines and per pipeline, each formed within a case and
    read across cases, plus the persistence contrast at matched offset when the sampling `bins` are
    given and the frame records `abs_offset`.

    Long format: `pipeline` (`POOLED` for the reading over all pipelines), `contrast` (`coverage`,
    `persistence`, `persistence_offset`, `alignment`), `outcome`, and the `paired_comparison` fields;
    `n_agree` counts cases in the predicted (positive) direction.
    """
    per_case: List[Dict[str, Any]] = []
    for outcome in PAIR_OUTCOMES:
        if outcome not in frame:
            continue
        pairs = _pair_benefits(frame, outcome)
        if 'coverage_cur' not in pairs or 'persistence' not in pairs:
            continue
        for (case, preset), group in pairs.groupby(['case', 'preset'], sort=False):
            per_case.append(
                {
                    'case': case,
                    'pipeline': preset,
                    'contrast': 'coverage',
                    'outcome': outcome,
                    'value': _spearman(group['benefit'], group['coverage_cur']),
                }
            )
            per_case.append(
                {
                    'case': case,
                    'pipeline': preset,
                    'contrast': 'persistence',
                    'outcome': outcome,
                    'value': _persistence_gap(group),
                }
            )
            if bins and 'abs_offset' in group:
                per_case.append(
                    {
                        'case': case,
                        'pipeline': preset,
                        'contrast': 'persistence_offset',
                        'outcome': outcome,
                        'value': _persistence_gap_by_offset(group, bins),
                    }
                )
    if 'motion_alignment' in frame:
        alignment = (
            frame.assign(preset=frame['pipeline'].map(preset_of))
            .groupby(['case', 'preset', 'masking'])['motion_alignment']
            .mean()
            .unstack('masking')
        )
        if 'none' in alignment and 'obb' in alignment:
            for (case, preset), value in (alignment['none'] - alignment['obb']).items():
                per_case.append(
                    {
                        'case': case,
                        'pipeline': preset,
                        'contrast': 'alignment',
                        'outcome': 'motion_alignment',
                        'value': float(value),
                    }
                )
    if not per_case:
        return pd.DataFrame()
    per_case_frame = pd.DataFrame(per_case)
    order = {POOLED: -1, **{preset: i for i, preset in enumerate(dict.fromkeys(frame['pipeline'].map(preset_of)))}}
    rows: List[Dict[str, Any]] = []
    for (contrast, outcome), family_frame in per_case_frame.groupby(['contrast', 'outcome'], sort=False):
        family = [
            _mechanism_row(preset, contrast, outcome, group['value'])
            for preset, group in family_frame.groupby('pipeline', sort=False)
        ]
        family = [row for row in family if row is not None]
        _decide_mechanism(family)
        rows.extend(family)
    for contrast, contrast_frame in per_case_frame.groupby('contrast', sort=False):
        per_outcome = {
            outcome: group.rename(columns={'pipeline': 'preset', 'value': 'benefit'})
            for outcome, group in contrast_frame.groupby('outcome', sort=False)
        }
        family = [
            _mechanism_row(POOLED, contrast, outcome, pooled)
            for outcome, pooled in _pooled_benefits(per_outcome).items()
        ]
        family = [row for row in family if row is not None]
        _decide_mechanism(family)
        rows.extend(family)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values('pipeline', key=lambda s: s.map(order), kind='stable').reset_index(drop=True)


# Strata of the descriptive benefit table, in table order: (key, heading).
BENEFIT_STRATA = (('offset_bin', 'Offset bin [frames]'), ('cell', 'Survey cell'), ('stratum', 'Persistence'))


def benefit_strata(frame: pd.DataFrame, bins: Sequence[Tuple[int, Optional[int]]]) -> pd.DataFrame:
    """
    The oriented-against-none benefit of the average pipeline per stratum, descriptive.

    `frame` is `load_case_results` output without the baselines. Strata: the sampling offset bins
    `bins`, the survey cells (whole cases), and the persistence halves (pairs split at the median
    persistence over all pairs, as in the by-persistence table). Per case and stratum, the
    pair-level benefit in ZNCC and GOS is averaged over the stratum's pairs and then over the
    pipelines measured under both modes in every case, and the reference-relative CTE per mode is
    the median over the pairs, averaged over those pipelines. Each row is the mean over the cases
    with pairs in the stratum; `<outcome>_positive` counts the cases with a positive benefit.
    """
    from stabilo_benchmark.sampling import bin_label, offset_bin_label  # noqa: PLC0415 - light import

    frame = frame.assign(preset=frame['pipeline'].map(preset_of))
    for column, missing in (('cell', None), ('persistence', np.nan)):
        if column not in frame:
            frame = frame.assign(**{column: missing})
    keys = ['case', 'preset', 'pair']
    pairs = frame.groupby(keys).agg(
        abs_offset=('abs_offset', 'first'), cell=('cell', 'first'), persistence=('persistence', 'mean')
    )
    for outcome, column in (('zncc', 'zncc'), ('gradient', 'gradient'), ('cte', 'cte_mean')):
        if column not in frame:
            continue
        wide = frame.pivot_table(index=keys, columns='masking', values=column, aggfunc='mean')
        if 'obb' not in wide or 'none' not in wide:
            return pd.DataFrame()
        if outcome == 'cte':
            pairs['cte_none'], pairs['cte_obb'] = wide['none'], wide['obb']
        else:
            pairs[outcome] = wide['obb'] - wide['none']
    pairs = pairs.reset_index()
    outcomes = [name for name in PAIR_OUTCOMES if name in pairs]
    if not outcomes:
        return pd.DataFrame()
    n_cases = pairs['case'].nunique()
    measured = pairs.dropna(subset=outcomes).groupby('preset')['case'].nunique()
    pairs = pairs[pairs['preset'].isin(measured[measured == n_cases].index)]
    split = float(pd.to_numeric(pairs['persistence'], errors='coerce').median())
    pairs = pairs.assign(
        offset_bin=[offset_bin_label(offset, bins) for offset in pairs['abs_offset']],
        stratum=[
            ('stationary' if value >= split else 'moving') if np.isfinite(value) else None
            for value in pd.to_numeric(pairs['persistence'], errors='coerce')
        ],
    )
    order = {'offset_bin': [bin_label(low, high) for low, high in bins]}
    rows: List[Dict[str, Any]] = []
    for key, heading in BENEFIT_STRATA:
        strata = pairs[pairs[key].notna()]
        labels = order.get(key) or sorted(strata[key].unique(), key=str)
        for label in labels:
            group = strata[strata[key] == label]
            if group.empty:
                continue
            per_preset = group.groupby(['case', 'preset']).agg(
                **{name: (name, 'mean') for name in outcomes},
                **{name: (name, 'median') for name in ('cte_none', 'cte_obb') if name in group},
            )
            per_case = per_preset.groupby('case').mean()
            row: Dict[str, Any] = {'heading': heading, 'stratum': str(label), 'n_cases': int(len(per_case))}
            for name in per_case.columns:
                row[name] = float(per_case[name].mean())
            for name in outcomes:
                row[f"{name}_positive"] = int((per_case[name] > 0).sum())
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs['persistence_split'] = split
    return result


BENEFIT_STRATA_CAPTION = (
    'Part II: the oriented-against-none benefit of the average pipeline by stratum, over {n_cases} cases, '
    'descriptive. $\\Delta$ZNCC and $\\Delta$GOS: the per-pair benefit averaged over the stratum\'s pairs, '
    'the pipelines and then the cases, with the number of cases in which it is positive; CTE$_{{50}}$: the '
    'median disagreement with the reference estimator without masking and with oriented masking. The '
    'persistence halves split the pairs at the median persistence ({split:.2f}). The planned tests are '
    'those of {contrasts} and {mechanism}.'
)


def write_benefit_strata_table(strata: pd.DataFrame, path: Path, n_cases: int, label: str, prefix: str) -> Path:
    """`benefit_strata` as one table, a block per kind of stratum."""
    scale, fmt, unit = DISPLAY['zncc']
    outcomes = [name for name in PAIR_OUTCOMES if name in strata]
    caption = BENEFIT_STRATA_CAPTION.format(
        n_cases=n_cases,
        split=strata.attrs.get('persistence_split', float('nan')),
        contrasts=f"\\Cref{{tab:{prefix}-contrasts}}",
        mechanism=f"\\Cref{{tab:{prefix}-mechanism}}",
    )
    headers = [r'CTE$_{50}^{\mathrm{none}}$ [px]', r'CTE$_{50}^{\mathrm{OBB}}$ [px]']
    headers += [f"$\\Delta${OUTCOMES[name][0]}{unit}" for name in outcomes]
    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r'  \small',
        r'  \begin{adjustbox}{max width=\textwidth}',
        f"  \\begin{{tabular}}{{lr{'r' * (2 + len(outcomes))}}}",
        r'    \toprule',
        '    Stratum & Cases & ' + ' & '.join(headers) + r' \\',
    ]
    for heading, block in strata.groupby('heading', sort=False):
        lines += [r'    \midrule', f"    \\multicolumn{{{4 + len(outcomes)}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for _, row in block.iterrows():
            cells = [str(row['stratum']).replace('-', '--'), str(row['n_cases'])]
            cells += [
                f"{row[name]:.2f}" if np.isfinite(row.get(name, np.nan)) else '--' for name in ('cte_none', 'cte_obb')
            ]
            cells += [
                f"{fmt.format(scale * row[name])} ({row[f'{name}_positive']}/{row['n_cases']})"
                if np.isfinite(row[name])
                else '--'
                for name in outcomes
            ]
            lines.append('    ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'  \end{adjustbox}', r'\end{table}', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def _mechanism_row(pipeline: str, contrast: str, outcome: str, values: pd.Series) -> Optional[Dict[str, Any]]:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return {
        'pipeline': pipeline,
        'contrast': contrast,
        'outcome': outcome,
        'n_agree': int((finite > 0).sum()),
        **paired_comparison(finite).to_dict(),
    }


def _decide_mechanism(family: List[Dict[str, Any]]) -> None:
    """Holm-adjust a family of mechanism contrasts; each is predicted positive."""
    for row, adjusted in zip(family, holm_adjust([r['p_value'] for r in family]), strict=True):
        row['p_holm'] = adjusted
        planned = row['contrast'] not in SENSITIVITY_CONTRASTS
        row['favors'] = bool(planned and row['ci_low'] > 0 and adjusted <= ALPHA)


# --------------------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------------------

CONTRASTS_CAPTION = (
    'Part II: the planned masking contrasts over {n_cases} cases. Each cell is the Hodges--Lehmann '
    'estimate of the per-case paired benefit of the first mode over the second (positive favors the first; '
    'pixels for CTE and OCE) and the number of cases in which it is positive. CTE, IoU and OCE are against '
    'the ground control points on the audited frames, ZNCC and GOS over every pair. A contrast is read over '
    'all pipelines (Holm across the outcomes), then per pipeline (Holm across the pipelines). Bold: a '
    'benefit with a 95\\% cluster-bootstrap interval above zero, an estimate reaching the margin (the GCP '
    'repeatability{margin} for CTE and OCE) and a Holm-adjusted $p \\leq {alpha:g}$; underlined: the same '
    'in favor of the second mode at the unadjusted $p$. The smallest attainable $p$ is {min_p:.4f}. The '
    'primary contrast (OBB against none, all pipelines) is claimed when an outcome is bold and none is '
    'underlined. Intervals and $p$-values: {csv}.'
)
MECHANISM_CAPTION = (
    'Part II: the mechanism contrasts over {n_cases} cases, each formed within a case, on the per-pair '
    'oriented-against-none benefit. Coverage: Spearman correlation of the benefit with the current '
    'frame\'s object coverage; Persistence: benefit on pairs whose foreground moved minus on pairs whose '
    'foreground stayed, at matched coverage; Persistence$^{{k}}$: the same at matched offset, a sensitivity '
    'reading never bolded; Alignment: $A$ without masking minus with oriented masking. All are predicted '
    'positive. Each cell: Hodges--Lehmann estimate across cases [95\\% cluster-bootstrap interval] (cases '
    'positive); bold: interval above zero and Holm-adjusted $p \\leq {alpha:g}$ (across ZNCC and GOS over '
    'all pipelines, across the pipelines per pipeline).'
)


def _cell(
    row: Optional[Mapping[str, Any]], outcome: str, bold: bool, interval: bool = False, underline: bool = False
) -> str:
    if row is None or not np.isfinite(row['estimate']):
        return '--'
    scale, fmt, _ = DISPLAY[outcome]
    text = fmt.format(scale * row['estimate'])
    if interval and np.isfinite(row['ci_low']):
        text += ' [' + fmt.format(scale * row['ci_low']) + ', ' + fmt.format(scale * row['ci_high']) + ']'
    text = rf'\textbf{{{text}}}' if bold else text
    text = rf'\underline{{{text}}}' if underline else text
    return f"{text} ({row['n_agree']}/{row['n_clusters']})"


def _display(preset: str) -> str:
    return pipeline_name(preset).replace('_', r'\_')


def write_contrasts_table(
    contrasts: pd.DataFrame,
    path: Path,
    n_cases: int,
    label: str,
    csv_name: str,
    annotation_floor: Optional[float] = None,
) -> Path:
    """The masking contrasts as one table: a block per contrast, a row per pipeline, a column per outcome."""
    from stabilo_benchmark.stats import minimum_p  # noqa: PLC0415 - keeps the module header light

    outcomes = [name for name in OUTCOMES if name in set(contrasts['outcome'])]
    caption = CONTRASTS_CAPTION.format(
        n_cases=n_cases,
        margin=f", {annotation_floor:.2f} px" if annotation_floor else '',
        alpha=ALPHA,
        min_p=minimum_p(n_cases),
        csv=r'\texttt{' + csv_name.replace('_', r'\_') + '}',
    )
    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r'  \footnotesize',
        r'  \setlength{\tabcolsep}{3pt}',
        r'  \begin{adjustbox}{max width=\textwidth}',
        f"  \\begin{{tabular}}{{l{'r' * len(outcomes)}c}}",
        r'    \toprule',
        '    Pipeline & ' + ' & '.join(OUTCOMES[name][0] + DISPLAY[name][2] for name in outcomes) + r' & Claim \\',
    ]
    for first, second, role in MASKING_CONTRASTS:
        name = f"{first} vs {second}"
        block = contrasts[contrasts['contrast'] == name]
        if block.empty:
            continue
        title = f"{MASKING_LABELS[first]} against {MASKING_LABELS[second]} ({role})"
        lines += [r'    \midrule', f"    \\multicolumn{{{len(outcomes) + 2}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        ordered = pd.concat([block[block['pipeline'] == POOLED], block[block['pipeline'] != POOLED]])
        for preset, group in ordered.groupby('pipeline', sort=False):
            by_outcome = {row['outcome']: row for _, row in group.iterrows()}
            cells = [
                _cell(
                    by_outcome.get(outcome),
                    outcome,
                    bool(by_outcome.get(outcome, {}).get('favors', False)),
                    underline=bool(by_outcome.get(outcome, {}).get('harms', False)),
                )
                for outcome in outcomes
            ]
            pooled = preset == POOLED
            mark = '--'
            if role == 'primary' and pooled:
                if not bool(group['complete'].iloc[0]):
                    mark = 'n/a'
                else:
                    mark = r'\cmark' if bool(group['claim'].iloc[0]) else ''

            name = r'\textit{All pipelines}' if pooled else _display(preset)
            lines.append(f"    {name} & " + ' & '.join(cells) + f" & {mark} \\\\")
            if pooled and len(ordered['pipeline'].unique()) > 1:
                lines.append(f"    \\cmidrule(lr){{2-{len(outcomes) + 2}}}")
    lines += [r'    \bottomrule', r'  \end{tabular}', r'  \end{adjustbox}', r'\end{table}', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_mechanism_table(mechanism: pd.DataFrame, path: Path, n_cases: int, label: str) -> Path:
    """The mechanism contrasts, a row over all pipelines and then a row per pipeline."""
    columns: Sequence[Tuple[str, str, str]] = (
        ('coverage', 'zncc', r'Coverage$_{\mathrm{ZNCC}}$ ($\rho$)'),
        ('coverage', 'gradient', r'Coverage$_{\mathrm{GOS}}$ ($\rho$)'),
        ('persistence', 'zncc', r'Persistence$_{\mathrm{ZNCC}}$ [$10^{-3}$]'),
        ('persistence', 'gradient', r'Persistence$_{\mathrm{GOS}}$ [$10^{-3}$]'),
        ('persistence_offset', 'zncc', r'Persistence$^{k}_{\mathrm{ZNCC}}$ [$10^{-3}$]'),
        ('persistence_offset', 'gradient', r'Persistence$^{k}_{\mathrm{GOS}}$ [$10^{-3}$]'),
        ('alignment', 'motion_alignment', 'Alignment'),
    )
    present = [c for c in columns if ((mechanism['contrast'] == c[0]) & (mechanism['outcome'] == c[1])).any()]
    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        f"  \\caption{{{MECHANISM_CAPTION.format(n_cases=n_cases, alpha=ALPHA)}}}",
        f"  \\label{{{label}}}",
        r'  \footnotesize',
        r'  \setlength{\tabcolsep}{3pt}',
        r'  \begin{adjustbox}{max width=\textwidth}',
        f"  \\begin{{tabular}}{{l{'r' * len(present)}}}",
        r'    \toprule',
        '    Pipeline & ' + ' & '.join(c[2] for c in present) + r' \\',
        r'    \midrule',
    ]
    pipelines = list(dict.fromkeys(mechanism['pipeline']))
    for preset, group in mechanism.groupby('pipeline', sort=False):
        cells = []
        for contrast, outcome, _ in present:
            match = group[(group['contrast'] == contrast) & (group['outcome'] == outcome)]
            row = match.iloc[0].to_dict() if not match.empty else None
            display = 'coverage' if contrast == 'coverage' else outcome
            cells.append(_cell(row, display, bool(row and row['favors']), interval=True))
        pooled = preset == POOLED
        name = r'\textit{All pipelines}' if pooled else _display(preset)
        lines.append(f"    {name} & " + ' & '.join(cells) + r' \\')
        if pooled and len(pipelines) > 1:
            lines.append(f"    \\cmidrule(lr){{2-{len(present) + 1}}}")
    lines += [r'    \bottomrule', r'  \end{tabular}', r'  \end{adjustbox}', r'\end{table}', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path
