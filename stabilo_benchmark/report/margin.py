#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
margin.py - The mask-margin sweep: masking benefit against the area the mask suppresses.

The margin campaign runs HBB and OBB masks at several `mask_margin_ratio` values; the unmasked
score of the same pair comes from the case's masking campaign of the same regime (beside
`<case>/results/offline_margin`, `<case>/results/offline_masking`: the same sampling plan and seed,
so the same pairs). The benefit is the reference-free score under the
mask minus that without it, per pair; the suppressed area is the mean share of the two frames the
feature mask withholds (`suppressed_ref`, `suppressed_cur`). Both are averaged within a case, then
across cases with equal weight; the interval is the cluster bootstrap over cases.

If oriented boxes at a larger margin beat horizontal boxes at equal suppressed area, the advantage
is the geometry; if the two curves coincide, it is the area.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stabilo_benchmark.report.contrasts import preset_of
from stabilo_benchmark.stats import cluster_bootstrap_ci, hodges_lehmann

OUTCOMES = {'zncc': 'ZNCC', 'gradient': 'GOS'}

CAPTION = (
    'Part II: masking benefit against suppressed area over {n_cases} cases. Each row is one box '
    'geometry at one mask margin $m$ (each box side grown by the factor $1+m$), pooled over the '
    '{n_pipelines} pipelines. Suppressed is the mean share of the reference and current frames the '
    'feature mask withholds from detection; $\\Delta$ZNCC and $\\Delta$GOS are the gain over no masking '
    'on the same pairs, in units of $10^{{-3}}$, as the Hodges--Lehmann estimate across cases with its '
    '95\\% cluster bootstrap interval. If oriented boxes at a larger margin beat horizontal boxes at '
    'equal suppressed area, the advantage is the geometry; if they coincide, it is the area. Per-pipeline '
    'values are in {csv}.'
)


def is_margin_sweep(frame: pd.DataFrame) -> bool:
    """Whether a results frame varies `mask_margin_ratio` within a pipeline and masking mode."""
    if 'mask_margin_ratio' not in frame or 'suppressed_cur' not in frame:
        return False
    masked = frame[frame['masking'].isin(('hbb', 'obb'))]
    return bool(masked.groupby(masked['pipeline'].map(preset_of))['mask_margin_ratio'].nunique().max() > 1)


def masking_campaign_of(results_dir: Path) -> Path:
    """The masking campaign beside a margin sweep: its name with `margin` replaced by `masking`."""
    results_dir = Path(results_dir)
    return results_dir.parent / results_dir.name.replace('margin', 'masking')


def unmasked_scores(results_dir: Path) -> Optional[pd.DataFrame]:
    """The case's unmasked measurements from its masking campaign, or None when it has none."""
    from stabilo_benchmark.report.tables import load_results  # noqa: PLC0415 - avoids an import cycle

    directory = masking_campaign_of(results_dir)
    if not directory.is_dir():
        return None
    frame = load_results(directory)
    return frame[frame['masking'] == 'none']


def per_case_benefits(frame: pd.DataFrame, unmasked: pd.DataFrame) -> pd.DataFrame:
    """
    One row per case, pipeline, geometry and margin: mean suppressed share and mean benefit.

    Both frames carry `case`; pairs are matched on case, pipeline and pair key.
    """
    keys = ['case', 'preset', 'pair']
    masked = frame[frame['masking'].isin(('hbb', 'obb'))].assign(preset=lambda f: f['pipeline'].map(preset_of))
    baseline = (
        unmasked.assign(preset=unmasked['pipeline'].map(preset_of))
        .groupby(keys)[[c for c in OUTCOMES if c in unmasked]]
        .mean()
        .add_suffix('_none')
    )
    joined = masked.join(baseline, on=keys, how='inner')
    joined = joined.assign(suppressed=(joined['suppressed_ref'] + joined['suppressed_cur']) / 2.0)
    for outcome in OUTCOMES:
        if outcome in joined and f"{outcome}_none" in joined:
            joined[f"benefit_{outcome}"] = joined[outcome] - joined[f"{outcome}_none"]
    columns = ['suppressed', *[f"benefit_{o}" for o in OUTCOMES if f"benefit_{o}" in joined]]
    result = (
        joined.groupby(['case', 'preset', 'masking', 'mask_margin_ratio'])[columns]
        .mean()
        .reset_index()
        .assign(n_pairs=joined.groupby(['case', 'preset', 'masking', 'mask_margin_ratio']).size().to_numpy())
    )
    # Measurements of the sweep with no unmasked score of the same pair (a sampling mismatch).
    result.attrs['unmatched'] = int(len(masked) - len(joined))
    return result


def pooled_benefits(per_case: pd.DataFrame) -> pd.DataFrame:
    """
    One row per geometry and margin: each case's mean over pipelines, then the Hodges-Lehmann
    estimate and cluster bootstrap interval across cases.
    """
    benefit_columns = [c for c in per_case if c.startswith('benefit_')]
    by_case = per_case.groupby(['masking', 'mask_margin_ratio', 'case'])[['suppressed', *benefit_columns]].mean()
    rows: List[Dict[str, Any]] = []
    for (masking, margin), group in by_case.groupby(level=['masking', 'mask_margin_ratio']):
        row: Dict[str, Any] = {
            'masking': masking,
            'mask_margin_ratio': float(margin),
            'n_cases': int(len(group)),
            'suppressed': float(group['suppressed'].mean()),
        }
        for column in benefit_columns:
            values = group[column].to_numpy(dtype=float)
            low, high = cluster_bootstrap_ci(values)
            row[column] = hodges_lehmann(values)
            row[f"{column}_low"], row[f"{column}_high"] = low, high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['masking', 'mask_margin_ratio'], kind='stable').reset_index(drop=True)


def write_margin_table(pooled: pd.DataFrame, path: Path, n_cases: int, n_pipelines: int, label: str, csv: str) -> Path:
    """The pooled sweep, one row per geometry and margin."""
    outcomes = [o for o in OUTCOMES if f"benefit_{o}" in pooled]
    lines = [
        '% Generated by stabilo-benchmark; do not edit by hand.',
        r'\begin{table}[htbp]',
        r'  \centering',
        '  \\caption{'
        + CAPTION.format(n_cases=n_cases, n_pipelines=n_pipelines, csv=r'\texttt{' + csv.replace('_', r'\_') + '}')
        + '}',
        f"  \\label{{{label}}}",
        r'  \small',
        f"  \\begin{{tabular}}{{lrr{'r' * len(outcomes)}}}",
        r'    \toprule',
        '    Boxes & $m$ & Suppressed & ' + ' & '.join(f"$\\Delta${OUTCOMES[o]}" for o in outcomes) + r' \\',
        r'    \midrule',
    ]
    for _, row in pooled.iterrows():
        cells = []
        for outcome in outcomes:
            estimate, low, high = (1e3 * row[f"benefit_{outcome}{suffix}"] for suffix in ('', '_low', '_high'))
            cells.append(f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]" if np.isfinite(estimate) else '--')
        lines.append(
            f"    {str(row['masking']).upper()} & {row['mask_margin_ratio']:.2f} & "
            f"{100 * row['suppressed']:.1f}\\% & " + ' & '.join(cells) + r' \\'
        )
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def figure_margin_benefit(pooled: pd.DataFrame, path: Path, outcomes: Sequence[str] = tuple(OUTCOMES)) -> Path:
    """Benefit against suppressed area, one panel per outcome, one line per box geometry through its margins."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - after the backend is set by figures

    from stabilo_benchmark.report.figures import (  # noqa: PLC0415 - matplotlib loads only when drawing
        INK_SECONDARY,
        MASKING_LABELS,
        MASKING_ORDER,
        _save,
        _style_for,
        apply_style,
    )

    outcomes = [o for o in outcomes if f"benefit_{o}" in pooled]
    if not outcomes:
        raise ValueError('no benefit column to plot')
    apply_style()
    figure, axes = plt.subplots(1, len(outcomes), figsize=(3.3 * len(outcomes), 2.8), squeeze=False)
    for axis, outcome in zip(axes[0], outcomes, strict=True):
        for masking in ('hbb', 'obb'):
            rows = pooled[pooled['masking'] == masking].sort_values('mask_margin_ratio')
            if rows.empty:
                continue
            style = _style_for(MASKING_ORDER.index(masking))
            x = 100 * rows['suppressed'].to_numpy(dtype=float)
            y = 1e3 * rows[f"benefit_{outcome}"].to_numpy(dtype=float)
            error = 1e3 * np.vstack(
                [
                    rows[f"benefit_{outcome}"] - rows[f"benefit_{outcome}_low"],
                    rows[f"benefit_{outcome}_high"] - rows[f"benefit_{outcome}"],
                ]
            )
            axis.errorbar(x, y, yerr=np.clip(error, 0, None), capsize=2, label=MASKING_LABELS[masking], **style)
            # Margin labels above the horizontal series and below the oriented one, clear of each other.
            offset = (4, 4) if masking == 'hbb' else (4, -10)
            for xi, yi, margin in zip(x, y, rows['mask_margin_ratio'], strict=True):
                axis.annotate(
                    f"m={margin:g}",
                    (xi, yi),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=7,
                    color=style['color'],
                )
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=':')
        axis.set_xlabel('Suppressed area [% of the frame]')
        axis.set_ylabel(f"{OUTCOMES[outcome]} gain over no masking [$10^{{-3}}$]")
    axes[0][0].legend(loc='best')
    figure.tight_layout()
    return _save(figure, path)


def build_margin_outputs(
    produced: Dict[str, Any], frame: pd.DataFrame, results_dirs: Sequence[Path], root: Path, prefix: str
) -> None:
    """Table, figure and per-pipeline CSV of a margin sweep; skipped with a reason when no unmasked runs exist."""
    unmasked = []
    for results_dir in results_dirs:
        scores = unmasked_scores(results_dir)
        if scores is None:
            produced['skipped'].append(
                f"margin benefit: no {masking_campaign_of(results_dir).name} campaign beside {results_dir} "
                'for the unmasked scores'
            )
            return
        case = Path(results_dir).parent.parent.name
        unmasked.append(scores.assign(case=case))
    data = frame if 'case' in frame else frame.assign(case=Path(results_dirs[0]).parent.parent.name)
    per_case = per_case_benefits(data, pd.concat(unmasked, ignore_index=True))
    if per_case.empty:
        produced['skipped'].append('margin benefit: no pair of the sweep was found in the masking campaign')
        return
    if per_case.attrs.get('unmatched'):
        produced['warnings'].append(
            f"margin benefit: {per_case.attrs['unmatched']} measurement(s) have no unmasked score of the same pair "
            f"in the {masking_campaign_of(results_dirs[0]).name} campaign; the two campaigns should sample the "
            'same pairs'
        )
    pooled = pooled_benefits(per_case)
    tables, figures = Path(root) / 'tables', Path(root) / 'figures'
    csv = tables / f"{prefix}_benefit.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(csv, index=False)
    n_cases, n_pipelines = int(per_case['case'].nunique()), int(per_case['preset'].nunique())
    produced['tables'] += [
        str(
            write_margin_table(
                pooled, tables / f"{prefix}_benefit.tex", n_cases, n_pipelines, f"tab:{prefix}-benefit", csv.name
            )
        ),
        str(csv),
    ]
    try:
        produced['figures'].append(str(figure_margin_benefit(pooled, figures / f"{prefix}_benefit.pdf")))
    except ValueError as error:
        produced['skipped'].append(f"margin benefit figure: {error}")
