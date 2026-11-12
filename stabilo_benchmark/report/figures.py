#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
figures.py - Manuscript figures, generated from saved results.

- No dual-axis plots; two quantities get adjacent panels sharing an x-axis.
- Every series has its own marker, line style or hatch as well as a color.
- Categorical hues come from a color-vision-deficiency-validated palette. Scatter plots use only
  the first three slots (distinguishable in every pair); bars and lines may use all eight
  (distinguishable in adjacent pairs). Re-validate the palette if the hex values change.
- Error distributions are shown as medians with an interquartile band.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use('Agg')  # figures are written to files, never displayed; keeps this import-safe
import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection
from matplotlib import colors as mcolors  # noqa: E402 - must follow the backend selection
from matplotlib import ticker  # noqa: E402 - must follow the backend selection
from matplotlib.lines import Line2D  # noqa: E402 - must follow the backend selection

from stabilo_benchmark.report.names import pipeline_name  # noqa: E402 - after the backend selection

# Validated categorical palette (light mode). Slots 1-3 are the all-pairs-safe subset used
# for scatter plots; the full order is safe for adjacent pairs (bars, lines).
PALETTE: List[str] = [
    '#2a78d6',  # 1 blue
    '#eb6834',  # 2 orange
    '#1baf7a',  # 3 aqua
    '#eda100',  # 4 yellow
    '#e87ba4',  # 5 magenta
    '#008300',  # 6 green
    '#4a3aa7',  # 7 violet
    '#e34948',  # 8 red
]
SCATTER_SLOTS = 3

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]
HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++', 'ooo', '***']

INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'

# Fixed order and display names for the masking axis, so every figure in the paper
# assigns the same color to the same strategy regardless of which appear in a given plot.
MASKING_ORDER = ['none', 'hbb', 'obb']
MASKING_LABELS = {'none': 'No masking', 'hbb': 'HBB masking', 'obb': 'OBB masking'}


def apply_style() -> None:
    """Publication defaults: serif text to match the manuscript body, recessive chrome."""
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': 'serif',
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.edgecolor': '#c3c2b7',
            'axes.labelcolor': INK_PRIMARY,
            'axes.grid': True,
            'axes.grid.axis': 'y',
            'grid.color': GRIDLINE,
            'grid.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.color': INK_MUTED,
            'ytick.color': INK_MUTED,
            'text.color': INK_PRIMARY,
            'legend.frameon': False,
            'lines.linewidth': 1.6,
            'lines.markersize': 4.5,
        }
    )


def _style_for(index: int, scatter: bool = False) -> Dict[str, Any]:
    """Color, marker, and line style for series `index`, cycling nothing."""
    limit = SCATTER_SLOTS if scatter else len(PALETTE)
    if index >= limit:
        raise ValueError(
            f"Series {index + 1} exceeds the {limit} validated palette slot(s) for this chart type. "
            f"Fold the remaining series into 'Other', or split the figure into small multiples - "
            f"generating another hue would break the color-blind separation guarantees."
        )
    return {'color': PALETTE[index], 'marker': MARKERS[index], 'linestyle': LINESTYLES[index]}


def _finite(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Rows where every named column is finite, so a plot never silently drops a series."""
    out = frame
    for column in columns:
        out = out[np.isfinite(pd.to_numeric(out[column], errors='coerce'))]
    return out


def _masking_series(frame: pd.DataFrame) -> List[str]:
    """The masking modes present, in the canonical order."""
    present = set(frame['masking'].unique())
    return [mode for mode in MASKING_ORDER if mode in present]


def _save(figure: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------------
# The motivation figure
# --------------------------------------------------------------------------------------


# Hexagonal cells across and along the inlier figure's density panels.
HEXBIN_GRID = (18, 34)


def figure_inlier_vs_quality(
    frame: pd.DataFrame,
    path: Path,
    quality: str = 'cte_mean',
    quality_label: str = 'Disagreement with reference estimator [px]',
) -> Path:
    """
    Inlier ratio against alignment quality: the pooled density, and the within-group coefficients.

    Left, a panel per masking strategy counting frame pairs and pipelines in hexagonal cells, on one
    logarithmic color scale, so a lone estimate stays visible beside the bulk and no strategy is
    drawn over another. Right, the Spearman coefficient computed within
    each pipeline and case (within each pipeline when the frame carries no case), one point per
    group, per masking strategy, with its median printed beside it and the share of positive
    coefficients under the axis; a single coefficient over the pooled cloud would measure how
    estimators and cases differ in inlier ratio rather than whether it tracks accuracy.
    """
    apply_style()
    data = _finite(frame, 'inlier_ratio', quality)
    if data.empty:
        raise ValueError('No finite inlier ratio and quality to draw')
    # The error axis is logarithmic, so an exact zero (an estimate equal to the truth) sits at its floor.
    errors = pd.to_numeric(data[quality]).clip(lower=1e-3)
    modes = _masking_series(data)
    figure = plt.figure(figsize=(7.2, 3.4))
    grid = figure.add_gridspec(1, 2, width_ratios=(3, 2))
    clouds = grid[0].subgridspec(1, len(modes), wspace=0.16).subplots(sharex=True, sharey=True, squeeze=False)[0]
    spread = figure.add_subplot(grid[1])
    extent = (0.0, 1.0, math.log10(errors.min()) - 0.05, math.log10(errors.max()) + 0.05)
    density = mcolors.LogNorm(vmin=1.0, vmax=float(len(data)))
    cells = []

    tick_labels = [MASKING_LABELS[mode].replace(' masking', '') for mode in modes]
    for index, mode in enumerate(modes):
        subset = data[data['masking'] == mode]
        style = _style_for(index, scatter=True)
        coefficients = within_group_spearman(subset, quality)
        cloud = clouds[index]
        cells.append(
            cloud.hexbin(
                subset['inlier_ratio'],
                errors[subset.index],
                gridsize=HEXBIN_GRID,
                extent=extent,
                yscale='log',
                mincnt=1,
                norm=density,
                cmap=_tint_map(style['color']),
                linewidths=0.1,
                edgecolors='face',
            )
        )
        cloud.set_title(MASKING_LABELS[mode], fontsize=8, color=INK_SECONDARY)
        cloud.set_xticks((0.0, 0.5, 1.0), ('0', '0.5', '1'))
        cloud.grid(False)
        if coefficients.size:
            jitter = np.random.default_rng(index).uniform(-0.15, 0.15, coefficients.size)
            spread.scatter(
                index + jitter,
                coefficients,
                s=12,
                alpha=0.6,
                color=style['color'],
                marker=style['marker'],
                edgecolors='none',
            )
            median = float(np.median(coefficients))
            spread.plot([index - 0.3, index + 0.3], [median, median], color=INK_PRIMARY, linewidth=1.2)
            spread.text(index + 0.32, median, f"{median:+.2f}", va='center', fontsize=7, color=INK_PRIMARY)
            tick_labels[index] += f"\n{int((coefficients > 0).sum())}/{coefficients.size} > 0"

    density.vmax = max(float(np.max(cell.get_array())) for cell in cells)
    clouds[len(modes) // 2].set_xlabel('RANSAC inlier ratio')
    clouds[0].set_ylabel(quality_label)
    clouds[0].set_yscale('log')
    clouds[0].set_xlim(0.0, 1.0)
    spread.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, linestyle=':')
    spread.set_xticks(range(len(modes)), tick_labels)
    spread.set_xlim(-0.6, len(modes) - 0.4)
    spread.set_ylim(-1.05, 1.05)
    group = 'pipeline and case' if 'case' in data else 'pipeline'
    spread.set_ylabel(f"Spearman $\\rho$ within {group}")
    figure.tight_layout()
    return _save(figure, path)


def _tint_map(color: str) -> mcolors.Colormap:
    """A sequential map from a pale tint of `color` to `color`, so that its lowest count stays visible."""
    tint = tuple(0.3 * channel + 0.7 for channel in mcolors.to_rgb(color))
    return mcolors.LinearSegmentedColormap.from_list('tint', [tint, color])


def within_group_spearman(subset: pd.DataFrame, quality: str) -> np.ndarray:
    """The finite Spearman coefficients of inlier ratio against `quality`, one per pipeline and case."""
    keys = [key for key in ('pipeline', 'case') if key in subset]
    groups = subset.groupby(keys) if keys else [(None, subset)]
    values = [_spearman(group['inlier_ratio'], group[quality]) for _, group in groups]
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation, implemented directly to avoid a scipy dependency."""
    x = pd.to_numeric(a, errors='coerce').rank().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors='coerce').rank().to_numpy(dtype=float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------------------
# Masking comparison
# --------------------------------------------------------------------------------------


def figure_masking_comparison(
    frame: pd.DataFrame,
    path: Path,
    metrics: Sequence[str] = ('ssim', 'zncc', 'gradient'),
    metric_labels: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Grouped bars: each masking strategy's score, one panel per metric.

    Bars are hatched and directly labeled.
    """
    apply_style()
    labels = {
        **{'ssim': 'SSIM', 'zncc': 'ZNCC', 'mi': 'MI [bits]', 'gradient': 'GOS'},
        **(metric_labels or {}),
    }
    metrics = [metric for metric in metrics if metric in frame.columns]
    if not metrics:
        raise ValueError(f"None of the requested metrics are present: {sorted(frame.columns)}")

    modes = _masking_series(frame)
    figure, axes_row = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics) + 0.8, 3.0), squeeze=False)

    for panel, metric in enumerate(metrics):
        axes = axes_row[0][panel]
        values, errors = [], []
        for mode in modes:
            subset = pd.to_numeric(frame.loc[frame['masking'] == mode, metric], errors='coerce')
            finite = subset[np.isfinite(subset)]
            values.append(float(np.median(finite)) if finite.size else np.nan)
            errors.append(
                [
                    values[-1] - float(np.percentile(finite, 25)) if finite.size else 0.0,
                    float(np.percentile(finite, 75)) - values[-1] if finite.size else 0.0,
                ]
            )

        positions = np.arange(len(modes))
        for index, (position, value) in enumerate(zip(positions, values, strict=True)):
            axes.bar(
                position,
                value,
                width=0.62,
                color=PALETTE[index],
                hatch=HATCHES[index],
                edgecolor='white',
                linewidth=1.0,
            )
        axes.errorbar(
            positions, values, yerr=np.array(errors).T, fmt='none', ecolor=INK_SECONDARY, elinewidth=1.0, capsize=3
        )
        for position, value in zip(positions, values, strict=True):
            if np.isfinite(value):
                axes.annotate(
                    f"{value:.3f}",
                    (position, value),
                    textcoords='offset points',
                    xytext=(0, 4),
                    ha='center',
                    fontsize=7,
                    color=INK_SECONDARY,
                )

        axes.set_xticks(positions)
        axes.set_xticklabels([MASKING_LABELS[mode].replace(' masking', '') for mode in modes])
        axes.set_title(labels.get(metric, metric))
        if panel == 0:
            axes.set_ylabel('Median over frame pairs (bars: IQR)')

    figure.suptitle('Background alignment by masking strategy', y=1.02)
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Temporal behavior
# --------------------------------------------------------------------------------------


def figure_error_vs_offset(
    frame: pd.DataFrame,
    path: Path,
    metric: str = 'cte_mean',
    metric_label: str = 'Disagreement with reference estimator [px]',
    bins: Optional[Sequence[Tuple[int, Optional[int]]]] = None,
) -> Path:
    """
    Registration error against temporal distance from the reference frame.

    Median line with an interquartile band per offset bin, one line per masking mode, on a
    logarithmic offset axis. `bins` are the campaign's sampling bins ([low, high] in frames, as in
    the by-offset table); without them the data range is split into bins that double in width.
    """
    apply_style()
    data = _finite(frame, 'abs_offset', metric)
    if data.empty:
        raise ValueError(f"No finite '{metric}' values with an offset to plot")

    figure, axes = plt.subplots(figsize=(5.0, 3.2))
    if bins is None:
        top = float(data['abs_offset'].max())
        edges = [1.0]
        while edges[-1] <= top:
            edges.append(2.0 * edges[-1])
        bins = [(int(low), int(high) - 1) for low, high in zip(edges[:-1], edges[1:], strict=True)]
    bounds = [(float(low), float(high) if high is not None else float(data['abs_offset'].max())) for low, high in bins]
    centers = [math.sqrt(low * high) for low, high in bounds]

    for index, mode in enumerate(_masking_series(data)):
        subset = data[data['masking'] == mode]
        style = _style_for(index, scatter=True)
        median, low, high = [], [], []
        for start, stop in bounds:
            window = pd.to_numeric(
                subset.loc[(subset['abs_offset'] >= start) & (subset['abs_offset'] <= stop), metric], errors='coerce'
            )
            finite = window[np.isfinite(window)]
            median.append(float(np.median(finite)) if finite.size else np.nan)
            low.append(float(np.percentile(finite, 25)) if finite.size else np.nan)
            high.append(float(np.percentile(finite, 75)) if finite.size else np.nan)

        axes.plot(centers, median, label=MASKING_LABELS[mode], **style)
        axes.fill_between(centers, low, high, color=style['color'], alpha=0.15, linewidth=0)

    axes.set_xscale('log')
    axes.set_xlabel('Temporal distance from reference frame [frames]')
    axes.set_ylabel(metric_label)
    axes.set_title('Disagreement with the reference by temporal distance')
    axes.legend(loc='upper left')
    return _save(figure, path)


def figure_motion_alignment(frame: pd.DataFrame, path: Path) -> Path:
    """
    Empirical CDF of the motion-alignment ratio, by masking strategy.

    The ratio is parallel / (parallel + perpendicular) error relative to the dominant motion axis;
    0.5, isotropic error, is marked.
    """
    apply_style()
    data = _finite(frame, 'motion_alignment')
    if data.empty:
        raise ValueError(
            "No finite 'motion_alignment' values. The decomposition needs oriented labels "
            "(the motion axis comes from the objects' long axes) and a reference estimate."
        )

    modes = _masking_series(data)
    figure, axes = plt.subplots(figsize=(5.0, 3.2))

    for index, mode in enumerate(modes):
        values = pd.to_numeric(data.loc[data['masking'] == mode, 'motion_alignment'], errors='coerce')
        finite = values[np.isfinite(values)].to_numpy()
        if finite.size == 0:
            continue
        style = _style_for(index, scatter=True)
        ordered = np.sort(finite)
        axes.plot(
            ordered,
            np.arange(1, ordered.size + 1) / ordered.size,
            label=f"{MASKING_LABELS[mode]}  (median {np.median(finite):.2f})",
            color=style['color'],
            linestyle=style['linestyle'],
            marker='',
        )

    axes.axvline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 2)))
    axes.annotate(
        'isotropic noise',
        (0.5, 0.04),
        textcoords='offset points',
        xytext=(5, 0),
        fontsize=7,
        color=INK_SECONDARY,
    )
    axes.set_xlim(0, 1)
    axes.set_xlabel(r'Error along motion axis:  $e_\parallel / (e_\parallel + e_\perp)$')
    axes.set_ylabel('Empirical CDF over frame pairs')
    axes.set_title('Direction of the registration error by masking strategy')
    axes.legend(loc='upper left')
    return _save(figure, path)


def figure_sweep(
    frame: pd.DataFrame,
    path: Path,
    x: str,
    y: str = 'hea_3px',
    series: str = 'pipeline',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: str = '',
    x_ticks: Optional[Sequence[float]] = None,
) -> Path:
    """
    One-parameter sweep: `y` against `x`, one line per pipeline, in alphabetical order so a
    pipeline keeps its color across figures.

    Raises when there are more series than validated palette slots.
    """
    apply_style()
    data = _finite(frame, x, y)
    figure, axes = plt.subplots(figsize=(5.0, 3.2))

    for index, name in enumerate(sorted(data[series].unique())):
        subset = data[data[series] == name].sort_values(x)
        axes.plot(subset[x], subset[y], label=str(name), **_style_for(index))

    if x_ticks is not None:
        axes.set_xticks(list(x_ticks))
    axes.set_xlabel(x_label or x.replace('_', ' '))
    axes.set_ylabel(y_label or y.replace('_', ' '))
    if title:
        axes.set_title(title)
    axes.legend(loc='best', ncol=2)
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Accuracy against compute
# --------------------------------------------------------------------------------------

# Panels per row of the accuracy-against-compute figure: ten presets fill two rows.
COMPUTE_COLUMNS = 5
# Ring markers of the operating points in the compute figure, in order.
RING_MARKERS = ('o', 's', 'D')
# Line style per masking mode in the compute figure, where color is the feature budget.
COMPUTE_MASKING_STYLES = {'none': '-', 'hbb': ':', 'obb': '--'}


def figure_accuracy_vs_compute(
    frame: pd.DataFrame,
    path: Path,
    operating_points: Optional[Mapping[str, Tuple[float, int]]] = None,
    y: str = 'maa',
    y_label: str = 'mAA',
    floor: float = 1e-3,
) -> Path:
    """
    mAA against median registration time over a budget grid, one panel per preset.

    In each panel a line joins the downsampling ratios at one feature budget, colored and marked by
    budget and styled by masking mode (`COMPUTE_MASKING_STYLES`), so the budget, the axis that sets
    the accuracy, reads first and a masking mode that departs from the unmasked line shows as a
    split; each of `operating_points` (name -> ratio, max_features) is ringed, with its own marker
    (`RING_MARKERS`, in order). The panels share both axes, time on a log scale. mAA is drawn on a
    log scale of 1 - mAA, inverted and labeled in mAA, so a collapse and the last hundredth before
    saturation are both legible. Small multiples rather than one panel because the presets
    outnumber the validated palette slots.
    """
    apply_style()
    data = _finite(frame, 'time_median', y)
    if data.empty:
        raise ValueError("No finite (time_median, mAA) points to plot")
    # Distance from a perfect score, clipped so a perfect score still has a position.
    data = data.assign(_miss=np.clip(1.0 - pd.to_numeric(data[y]), floor, 1.0))

    presets = list(dict.fromkeys(data['preset']))
    budgets = sorted(data['max_features'].unique())
    if len(budgets) > len(PALETTE):
        raise ValueError(f"{len(budgets)} feature budgets exceed the {len(PALETTE)} palette slots")
    modes = _masking_series(data)
    rings = list((operating_points or {}).items())
    if len(rings) > len(RING_MARKERS):
        raise ValueError(f"{len(rings)} operating points exceed the {len(RING_MARKERS)} ring markers")

    columns = min(COMPUTE_COLUMNS, len(presets))
    rows = math.ceil(len(presets) / columns)
    figure, grid = plt.subplots(
        rows, columns, figsize=(7.2, 1.55 * rows + 0.6), sharex=True, sharey=True, squeeze=False
    )
    for slot, axes in enumerate(grid.flat):
        if slot >= len(presets):
            axes.set_visible(False)
            continue
        preset = presets[slot]
        for b, budget in enumerate(budgets):
            style = _style_for(b)
            for mode in modes:
                subset = data[
                    (data['preset'] == preset) & (data['masking'] == mode) & (data['max_features'] == budget)
                ].sort_values('downsample_ratio')
                if subset.empty:
                    continue
                axes.plot(
                    subset['time_median'],
                    subset['_miss'],
                    color=style['color'],
                    linestyle=COMPUTE_MASKING_STYLES.get(mode, '-'),
                    marker=style['marker'],
                    markersize=2.4,
                    linewidth=0.8,
                    alpha=0.9,
                )
                for ring, (_, (ratio, features)) in zip(RING_MARKERS, rings, strict=False):
                    at = subset[np.isclose(subset['downsample_ratio'], ratio) & (subset['max_features'] == features)]
                    axes.plot(
                        at['time_median'],
                        at['_miss'],
                        linestyle='none',
                        marker=ring,
                        markersize=6.5,
                        markerfacecolor='none',
                        markeredgecolor=INK_PRIMARY,
                        markeredgewidth=0.7,
                    )
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_ylim(1.0 + 0.2, floor)
        ticks = [10.0**-k for k in range(int(round(-math.log10(floor))) + 1)]
        axes.set_yticks(ticks, [f"{1.0 - t:g}" for t in ticks])
        axes.minorticks_off()
        axes.set_title(pipeline_name(preset), fontsize=8, color=INK_SECONDARY)
        axes.grid(True, axis='both', which='major')
        if slot % columns == 0:
            axes.set_ylabel(y_label)
        if slot >= len(presets) - columns:
            axes.set_xlabel('Median time [s]')
            axes.tick_params(labelbottom=True)

    handles = [
        Line2D([], [], color=_style_for(b)['color'], marker=_style_for(b)['marker'], label=f"{int(budget)} features")
        for b, budget in enumerate(budgets)
    ]
    handles += [
        Line2D([], [], color=INK_SECONDARY, linestyle=COMPUTE_MASKING_STYLES.get(mode, '-'), label=MASKING_LABELS[mode])
        for mode in modes
    ]
    for ring, (name, (ratio, features)) in zip(RING_MARKERS, rings, strict=False):
        handles.append(
            Line2D(
                [],
                [],
                linestyle='none',
                marker=ring,
                markersize=6.5,
                markerfacecolor='none',
                markeredgecolor=INK_PRIMARY,
                label=f"{name} ({ratio:g} / {features})",
            )
        )
    # At most four entries per legend row, so the legend is no wider than the panels.
    legend_columns = min(len(handles), 4)
    legend_rows = math.ceil(len(handles) / legend_columns)
    figure.legend(handles=handles, loc='upper center', ncol=legend_columns, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 1 - (0.15 + 0.2 * legend_rows) / (1.55 * rows + 0.6)))
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Tuning surfaces and estimator thresholds
# --------------------------------------------------------------------------------------

# mAA below the best grid point that separates the classes of the tuning figure; the first edge is
# the analysis plan's mAA margin, so the lightest class is the set tied with the best.
TUNING_GAP_EDGES = (0.01, 0.02, 0.05, 0.10)
# One-hue sequential ramp (blue steps 100, 250, 400, 550, 700), light for a small gap.
TUNING_GAP_COLORS = ('#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b')
TUNING_COLUMNS = 5


def _tuning_classes(gap: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """The class of each gap below the best: 0 for at most `edges[0]`, ..., len(edges) above the last."""
    return np.searchsorted(np.asarray(edges, dtype=float), gap, side='left').astype(float)


def _grid_position(
    values: Mapping[str, Any], thresholds: Sequence[float], ratios: Sequence[float]
) -> Optional[Tuple[float, float]]:
    """The (column, row) cell of a tuning panel at `values`, or None when they are off the grid."""
    try:
        column = list(thresholds).index(float(values['ransac_epipolar_threshold']))
        row = 0 if np.isnan(ratios[0]) else list(ratios).index(float(values['filter_ratio']))
    except (KeyError, TypeError, ValueError):
        return None
    return float(column), float(row)


def figure_tuning_surface(
    grid: pd.DataFrame,
    path: Path,
    selected: Optional[pd.DataFrame] = None,
    defaults: Optional[Mapping[str, Mapping[str, float]]] = None,
    margin: float = TUNING_GAP_EDGES[0],
    default_label: str = 'untuned',
) -> Path:
    """
    How far each grid point of a tuning campaign lies below its pipeline's best, one panel per pipeline.

    `grid` is `tables.tuning_grid` output. Each cell is a (ratio, inlier threshold) point, classed
    by its mAA gap to the pipeline's best point; the lightest class is within `margin` of it, so a
    panel that is light throughout is a surface the data cannot tell apart. `selected` (the tuning
    table's rows) rings the values kept, and `defaults` (preset -> values) marks the reference
    point (the untuned starting values in Part I, the incumbent in Part II; `default_label`) with a
    triangle when it is on the grid. A pipeline that does not apply the ratio test has
    one row.
    """
    from stabilo_benchmark.report.names import PIPELINE_NAMES  # noqa: PLC0415 - keeps the import graph flat

    apply_style()
    data = _finite(grid, 'maa')
    if data.empty:
        raise ValueError('No scored grid point to draw')
    data = data.assign(preset=data['pipeline'].astype(str).str.split('+', n=1).str[0])
    order = {name: rank for rank, name in enumerate(PIPELINE_NAMES)}
    presets = sorted(dict.fromkeys(data['preset']), key=lambda name: (order.get(name, len(order)), name))
    thresholds = sorted(data['ransac_epipolar_threshold'].astype(float).unique())
    edges = (margin, *TUNING_GAP_EDGES[1:])
    columns = min(TUNING_COLUMNS, len(presets))
    rows = math.ceil(len(presets) / columns)
    figure, axes_grid = plt.subplots(rows, columns, figsize=(7.2, 1.7 * rows + 0.45), squeeze=False)
    colormap = mpl.colors.ListedColormap(TUNING_GAP_COLORS)
    step = max(1, math.ceil(len(thresholds) / 7))

    for slot, axes in enumerate(axes_grid.flat):
        if slot >= len(presets):
            axes.set_visible(False)
            continue
        preset = presets[slot]
        group = data[data['preset'] == preset]
        ratios = sorted(group['filter_ratio'].dropna().astype(float).unique()) or [float('nan')]
        best = float(pd.to_numeric(group['maa']).max())
        cells = np.full((len(ratios), len(thresholds)), np.nan)
        for _, row in group.iterrows():
            r = 0 if np.isnan(ratios[0]) else ratios.index(float(row['filter_ratio']))
            cells[r, thresholds.index(float(row['ransac_epipolar_threshold']))] = best - float(row['maa'])
        classes = np.where(np.isfinite(cells), _tuning_classes(np.nan_to_num(cells, nan=0.0), edges), np.nan)
        axes.imshow(
            classes,
            cmap=colormap,
            vmin=-0.5,
            vmax=len(TUNING_GAP_COLORS) - 0.5,
            origin='lower',
            aspect='auto',
            interpolation='none',
        )
        tied = int(np.sum(np.isfinite(cells) & (cells <= margin)))
        axes.set_title(
            f"{pipeline_name(preset)}\n{tied} of {int(np.isfinite(cells).sum())} tied",
            fontsize=7,
            color=INK_SECONDARY,
            linespacing=1.1,
        )

        marks = []
        if defaults and preset in defaults:
            marks.append((_grid_position(defaults[preset], thresholds, ratios), '^'))
        if selected is not None and 'pipeline' in selected:
            chosen = selected[selected['pipeline'].astype(str).str.split('+', n=1).str[0] == preset]
            if len(chosen):
                marks.append((_grid_position(chosen.iloc[0].to_dict(), thresholds, ratios), 'o'))
        for at, marker in marks:
            if at is not None:
                axes.plot(
                    *at,
                    marker=marker,
                    markersize=5,
                    markerfacecolor='white',
                    markeredgecolor=INK_PRIMARY,
                    markeredgewidth=0.8,
                    linestyle='none',
                )

        axes.grid(False)
        axes.set_xticks(range(0, len(thresholds), step), [f"{t:g}" for t in thresholds[::step]], fontsize=6)
        if np.isnan(ratios[0]):
            axes.set_yticks([0], ['no ratio\ntest'], fontsize=6)
        else:
            shown = range(0, len(ratios), max(1, math.ceil(len(ratios) / 6)))
            axes.set_yticks(list(shown), [f"{ratios[i]:g}" for i in shown], fontsize=6)
        axes.tick_params(length=0)
        if slot % columns == 0:
            axes.set_ylabel('Ratio $r$', fontsize=7)
        if slot >= len(presets) - columns:
            axes.set_xlabel(r'Inlier threshold $\theta$ [px]', fontsize=7)

    labels = [f"within {edges[0]:g} (tied)"]
    labels += [f"{low:g}-{high:g}" for low, high in zip(edges[:-1], edges[1:], strict=True)]
    labels += [f"> {edges[-1]:g}"]
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor='none', label=label)
        for color, label in zip(TUNING_GAP_COLORS, labels, strict=True)
    ]
    handles += [
        Line2D([], [], marker=mark, linestyle='none', markerfacecolor='white', markeredgecolor=INK_PRIMARY, label=text)
        for mark, text in (('o', 'kept'), ('^', default_label))
    ]
    figure.legend(
        handles=handles,
        loc='upper center',
        ncol=len(handles),
        fontsize=6.5,
        bbox_to_anchor=(0.5, 1.0),
        title='mAA below the best grid point',
        title_fontsize=6.5,
        handlelength=1.2,
        columnspacing=1.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.42 / (1.7 * rows + 0.45)), h_pad=0.6, w_pad=0.4)
    return _save(figure, path)


def _estimator_family(name: str) -> Tuple[str, str]:
    """`USAC-Fast (LO-RANSAC)` -> (`LO-RANSAC`, `USAC-Fast`); a bare name is its own family."""
    if '(' in name:
        flag, algorithm = name.rstrip(')').split(' (', 1)
        return algorithm, flag
    return name, name


def figure_estimator_thresholds(summary: pd.DataFrame, path: Path, y: str = 'maa') -> Path:
    """
    mAA against the inlier threshold, one line per estimator, one panel per detector (and masking).

    `summary` is `tables.aggregate_estimators` output. Color follows the algorithm and the line style
    the OpenCV flag within it, so the three LO-RANSAC configurations share a hue. The panels share a
    linear mAA axis over the data's range.
    """
    apply_style()
    data = _finite(summary, 'ransac_epipolar_threshold', y)
    if data.empty:
        raise ValueError('No finite (threshold, mAA) points to plot')
    data = data.assign(_score=pd.to_numeric(data[y]))
    keys = ['detector', *(['masking'] if data['masking'].nunique() > 1 else [])]
    panels = list(dict.fromkeys(map(tuple, data[keys].astype(str).to_numpy())))
    estimators = list(dict.fromkeys(data['estimator'].astype(str)))
    families = list(dict.fromkeys(_estimator_family(name)[0] for name in estimators))
    if len(families) > len(PALETTE):
        raise ValueError(f"{len(families)} estimator families exceed the {len(PALETTE)} palette slots")

    figure, axes_row = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 0.4, 2.9), sharey=True, squeeze=False)
    low, high = float(data['_score'].min()), float(data['_score'].max())
    pad = 0.05 * max(high - low, 0.01)
    axes_row[0][0].set_ylim(low - pad, min(1.0, high + pad))
    handles: Dict[str, Line2D] = {}
    for axes, panel in zip(axes_row[0], panels, strict=True):
        selected = data[(data[keys].astype(str).to_numpy() == np.array(panel)).all(axis=1)]
        for name in estimators:
            family, flag = _estimator_family(name)
            members = [n for n in estimators if _estimator_family(n)[0] == family]
            color = PALETTE[families.index(family)]
            style = LINESTYLES[members.index(name)]
            marker = MARKERS[families.index(family)]
            line = selected[selected['estimator'].astype(str) == name].sort_values('ransac_epipolar_threshold')
            if line.empty:
                continue
            (handle,) = axes.plot(
                line['ransac_epipolar_threshold'].astype(float),
                line['_score'],
                color=color,
                linestyle=style,
                marker=marker,
                markersize=3.5,
                linewidth=1.2,
                label=name,
            )
            handles.setdefault(name, handle)
        axes.set_xscale('log', base=2)
        thresholds = sorted(selected['ransac_epipolar_threshold'].astype(float).unique())
        axes.set_xticks(thresholds, [f"{t:g}" for t in thresholds])
        axes.minorticks_off()
        axes.grid(True, axis='both', which='major')
        title = pipeline_name(panel[0]) + (f", {panel[1].upper()} masking" if len(panel) > 1 else '')
        axes.set_title(title, fontsize=8, color=INK_SECONDARY)
        axes.set_xlabel(r'Inlier threshold $\theta$ [px]')
    axes_row[0][0].set_ylabel('mAA')
    figure.legend(
        handles=list(handles.values()),
        loc='upper center',
        ncol=3,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=2.4,
    )
    rows = math.ceil(len(handles) / 3)
    figure.tight_layout(rect=(0, 0, 1, 1 - (0.12 + 0.13 * rows) / 2.9))
    return _save(figure, path)


# --------------------------------------------------------------------------------------
# Part II at a glance
# --------------------------------------------------------------------------------------

# Vertical offset of each masking mode within a pipeline's row of the summary figure.
SUMMARY_OFFSETS = {'none': -0.22, 'hbb': 0.0, 'obb': 0.22}
# mAA values marked on the summary's logarithmic scale of 1 - mAA, and where a perfect score sits on it.
SUMMARY_MAA_TICKS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
SUMMARY_MISS_FLOOR = 1e-3
BASELINES = ('identity', 'reference')


def _preset(label: Any) -> str:
    return str(label).split('#', 1)[0].split(' (', 1)[0].rsplit('+', 1)[0]


def figure_regime_summary(
    summary: pd.DataFrame,
    path: Path,
    gcp: Optional[pd.DataFrame] = None,
    maa_ties: Optional[Any] = None,
    gcp_ties: Optional[Any] = None,
    annotation_floor: Optional[float] = None,
    title: str = '',
) -> Path:
    """
    One campaign's pipelines on one page: mAA against the reference estimator (left, on a log scale
    of 1 - mAA, inverted and labeled in mAA, so the leaders separate on an axis that still reaches the
    identity) and the corner error against the GCP-fitted transform (right, median as a marker and
    the 90th percentile as the end of its line, on a log axis), a row per pipeline and a marker per
    masking mode.

    `summary` and `gcp` are the pooled tables' rows (`pipeline`, `masking`, and `maa`, or
    `cte_median` and `cte_p90`). A marker is filled when its configuration is in the column's tie
    set (`maa_ties`, `gcp_ties`: `stats.TieSet`), open otherwise, so the figure carries the analysis
    plan's verdict. Baseline rows are drawn as dotted vertical lines; `annotation_floor` (the GCP
    repeatability) as a dashed one.
    """
    from stabilo_benchmark.report.names import masking_name  # noqa: PLC0415 - keeps the import graph flat

    apply_style()
    rows = summary.assign(preset=summary['pipeline'].map(_preset))
    candidates = rows[~rows['preset'].isin(BASELINES)]
    if candidates.empty:
        raise ValueError('No candidate rows to draw')
    best = candidates.groupby('preset', sort=False)['maa'].max().sort_values(ascending=True)
    order = list(best.index)
    panels = 2 if gcp is not None and len(gcp) else 1
    figure, axes_row = plt.subplots(
        1, panels, figsize=(7.2, 0.27 * len(order) + 0.95), sharey=True, squeeze=False, gridspec_kw={'wspace': 0.06}
    )
    maa_axes = axes_row[0][0]
    modes = _masking_series(candidates)

    def bold(ties: Any, pipeline: Any, masking: Any) -> bool:
        return ties is not None and (pipeline, masking) in ties.bold

    for mode in modes:
        style = _style_for(MASKING_ORDER.index(mode), scatter=True)
        chosen = candidates[candidates['masking'] == mode]
        y = chosen['preset'].map({name: i for i, name in enumerate(order)}) + SUMMARY_OFFSETS[mode]
        filled = np.array([bold(maa_ties, p, m) for p, m in zip(chosen['pipeline'], chosen['masking'], strict=True)])
        for mask, face in ((filled, style['color']), (~filled, 'white')):
            maa_axes.scatter(
                (1.0 - pd.to_numeric(chosen['maa'])[mask]).clip(lower=SUMMARY_MISS_FLOOR),
                y[mask],
                s=22,
                marker=style['marker'],
                facecolors=face,
                edgecolors=style['color'],
                linewidths=0.9,
                zorder=3,
            )
    for _, row in rows[rows['preset'] == 'identity'].iterrows():
        maa_axes.axvline(1.0 - float(row['maa']), color=INK_MUTED, linestyle=':', linewidth=1.0)
        maa_axes.annotate(
            'identity',
            (1.0 - float(row['maa']), len(order) - 0.6),
            fontsize=6.5,
            color=INK_SECONDARY,
            xytext=(2, 0),
            textcoords='offset points',
        )
    maa_axes.set_yticks(range(len(order)), [pipeline_name(name) for name in order])
    maa_axes.set_ylim(-0.6, len(order) - 0.4)
    drawn = pd.to_numeric(rows[rows['preset'].isin([*order, 'identity'])]['maa']).dropna()
    maa_axes.set_xscale('log')
    best_miss = max(SUMMARY_MISS_FLOOR, 1.0 - float(drawn.max()))
    maa_axes.set_xlim(1.15 * (1.0 - float(drawn.min())), min(0.008, 0.8 * best_miss))
    maa_axes.set_xticks([1.0 - tick for tick in SUMMARY_MAA_TICKS], [f'{tick:g}' for tick in SUMMARY_MAA_TICKS])
    maa_axes.minorticks_off()
    maa_axes.set_xlabel('mAA against the reference estimator')
    maa_axes.grid(True, axis='x')
    maa_axes.grid(False, axis='y')

    if panels == 2:
        gcp_axes = axes_row[0][1]
        frame = gcp.assign(preset=gcp['pipeline'].map(_preset))
        for mode in modes:
            style = _style_for(MASKING_ORDER.index(mode), scatter=True)
            chosen = frame[(frame['masking'] == mode) & frame['preset'].isin(order)]
            y = chosen['preset'].map({name: i for i, name in enumerate(order)}) + SUMMARY_OFFSETS[mode]
            median = pd.to_numeric(chosen['cte_median'])
            gcp_axes.hlines(y, median, pd.to_numeric(chosen['cte_p90']), color=style['color'], linewidth=1.0)
            filled = np.array(
                [bold(gcp_ties, p, m) for p, m in zip(chosen['pipeline'], chosen['masking'], strict=True)]
            )
            for mask, face in ((filled, style['color']), (~filled, 'white')):
                gcp_axes.scatter(
                    median[mask],
                    y[mask],
                    s=22,
                    marker=style['marker'],
                    facecolors=face,
                    edgecolors=style['color'],
                    linewidths=0.9,
                    zorder=3,
                )
            gcp_axes.scatter(
                pd.to_numeric(chosen['cte_p90']), y, s=10, marker='|', color=style['color'], linewidths=1.0, zorder=3
            )
        if annotation_floor is not None:
            gcp_axes.axvline(annotation_floor, color=INK_SECONDARY, linestyle='--', linewidth=0.9)
            gcp_axes.annotate(
                'GCP repeatability',
                (annotation_floor, len(order) - 0.6),
                fontsize=6.5,
                color=INK_SECONDARY,
                xytext=(2, 0),
                textcoords='offset points',
            )
        gcp_axes.set_xscale('log')
        # 1-2-5 ticks with plain labels: the default labels every minor tick when the span is under a decade.
        gcp_axes.xaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        gcp_axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:g}'))
        gcp_axes.xaxis.set_minor_formatter(ticker.NullFormatter())
        gcp_axes.set_xlabel(
            'Corner error against the GCP-fitted transform [px]\nmedian (marker) to 90th percentile (bar)'
        )
        gcp_axes.grid(True, axis='x', which='major')
        gcp_axes.grid(False, axis='y')

    handles = [
        Line2D(
            [],
            [],
            linestyle='none',
            marker=_style_for(MASKING_ORDER.index(mode), scatter=True)['marker'],
            color=_style_for(MASKING_ORDER.index(mode), scatter=True)['color'],
            label=MASKING_LABELS.get(mode, masking_name(mode)),
        )
        for mode in modes
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle='none',
            marker='o',
            markerfacecolor='white',
            markeredgecolor=INK_SECONDARY,
            label='open: outside the tie set',
        )
    )
    figure.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=7, bbox_to_anchor=(0.5, 1.0))
    if title:
        figure.suptitle(title, y=1.04, fontsize=9)
    height = 0.27 * len(order) + 0.95
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.5 / height, top=1 - 0.35 / height)
    return _save(figure, path)
