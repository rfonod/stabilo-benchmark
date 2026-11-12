#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
case_figure.py - Part II's cases at a glance: each reference frame with what its objects did.

One panel per case, ordered by survey cell: the reference frame with every annotated object's OBB,
colored by what the object did in one later frame of the case, displaced with the largest group
that moved together, displaced otherwise, found where it was, or not found. The later frame is the
sampled frame closest to the longest offset the survey reads (the case's shortest sampled offset when
it records no survey bins). Objects are found again as the
benchmark finds them (`runners.dynamic.pair_object_motion`: the survey's matching, with the
reference estimator's transform as the platform motion), and each arrow is a displacement at true
scale in reference coordinates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from stabilo_benchmark.oracle import load_frame, load_frame_labels, load_oracle, oracle_matrices, pair_key
from stabilo_benchmark.report.figures import INK_SECONDARY, PALETTE, _save, apply_style
from stabilo_benchmark.report.survey_cases import CELL_ORDER
from stabilo_benchmark.sampling import FramePair, plan_bins, representative_offsets

# Longest side of each frame as rendered; the PDF embeds it as a raster.
FRAME_RENDER_PX = 700
# Share of white mixed into the grayscale frame so the overlays carry the color.
LIGHTEN = 0.25
COLUMNS = 3
# Layout in inches: the figure width (the manuscript's text width), the gap between panels, and the
# height of each panel's two-line title and of the legend.
FIGURE_WIDTH_IN = 6.4
GAP_IN = 0.05
TITLE_IN = 0.24
LEGEND_IN = 0.22

GROUP_COLOR = PALETTE[1]
DISPLACED_COLOR = PALETTE[0]
STATIC_COLOR = '#ffffff'
GONE_COLOR = '#898781'
# Drawing order and line width per fate, so the displaced objects lie on top.
FATES = (
    ('not found', GONE_COLOR, 0.25),
    ('found where it was', STATIC_COLOR, 0.3),
    ('displaced otherwise', DISPLACED_COLOR, 0.5),
    ('displaced with the largest group', GROUP_COLOR, 0.5),
)


class CaseFigureError(RuntimeError):
    """Raised when a case lacks what the figure needs: frames, labels, reference estimates."""


def target_offset(case: Dict[str, Any]) -> Optional[int]:
    """The longest offset the survey reads, from the case's recorded `survey_bins`; None when it records none."""
    bins = (case.get('sampling') or {}).get('survey_bins')
    return representative_offsets(plan_bins({'survey_bins': bins}, 'survey_bins'))[-1] if bins else None


def case_pair(case_dir: Path) -> FramePair:
    """The case's pair whose offset is closest to `target_offset`, among those with a reference estimate."""
    case = json.loads((Path(case_dir) / 'CASE.json').read_text(encoding='utf-8'))
    reference = int(case['reference_index'])
    matrices = oracle_matrices(load_oracle(case_dir))
    offsets = sorted(
        int(frame) - reference
        for frame in case.get('frames', [])
        if int(frame) != reference and matrices.get(pair_key(FramePair(reference, int(frame)))) is not None
    )
    if not offsets:
        raise CaseFigureError(f"{case_dir}: no pair with a reference estimate")
    # Without survey bins, the shortest sampled offset.
    wanted = target_offset(case) or 0
    return FramePair(reference, reference + min(offsets, key=lambda offset: (abs(abs(offset) - wanted), offset)))


def case_fates(case_dir: Path) -> Dict[str, Any]:
    """The case's reference frame, its objects' corners and fates at `case_pair`, and their displacements."""
    from stabilo_benchmark.runners.dynamic import pair_object_motion  # noqa: PLC0415 - keeps the import graph flat

    case_dir = Path(case_dir)
    pair = case_pair(case_dir)
    matrices = oracle_matrices(load_oracle(case_dir))
    motion = pair_object_motion(case_dir, [pair], matrices)[pair_key(pair)]
    frame = load_frame(case_dir, pair.reference_index)
    height, width = frame.shape[:2]
    labels = load_frame_labels(case_dir, pair.reference_index, width, height)
    corners = labels.corners() if len(labels) else np.zeros((0, 4, 2))
    fates = np.where(
        motion.group,
        FATES[3][0],
        np.where(motion.moved, FATES[2][0], np.where(motion.found, FATES[1][0], FATES[0][0])),
    )
    case = json.loads((case_dir / 'CASE.json').read_text(encoding='utf-8'))
    return {
        'name': case_dir.name,
        'cell': str(case.get('cell') or ''),
        'pair': pair,
        'frame': frame,
        'corners': corners,
        'fates': fates,
        'centers': motion.reference_centers,
        'displacement': motion.displacement,
        'counts': {
            'found': int(motion.found.sum()),
            'displaced': int(motion.moved.sum()),
            'together': int(motion.group.sum()),
        },
    }


def cell_order(case_dirs: Sequence[Path]) -> List[Path]:
    """The cases in survey-cell order (the design cells, then the control), stable within a cell."""
    order = {cell: rank for rank, cell in enumerate(CELL_ORDER)}

    def rank(case_dir: Path) -> int:
        try:
            cell = json.loads((Path(case_dir) / 'CASE.json').read_text(encoding='utf-8')).get('cell')
        except (OSError, json.JSONDecodeError):
            cell = None
        return order.get(cell, len(order))

    return sorted(case_dirs, key=rank)


def _draw_case(axis: plt.Axes, data: Dict[str, Any]) -> None:
    frame = data['frame']
    height, width = frame.shape[:2]
    scale = min(1.0, FRAME_RENDER_PX / max(height, width))
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    shown = np.round((1 - LIGHTEN) * gray + LIGHTEN * 255).astype(np.uint8)
    axis.imshow(shown, cmap='gray', vmin=0, vmax=255, extent=(0, width, height, 0), interpolation='none')
    for fate, color, line_width in FATES:
        chosen = data['fates'] == fate
        if chosen.any():
            axis.add_collection(
                PolyCollection(
                    data['corners'][chosen], closed=True, facecolor='none', edgecolor=color, linewidths=line_width
                )
            )
    for fate, color, _ in FATES[2:]:
        for i in np.flatnonzero(data['fates'] == fate):
            x, y = data['centers'][i]
            dx, dy = data['displacement'][i]
            axis.annotate(
                '',
                xy=(x + dx, y + dy),
                xytext=(x, y),
                arrowprops={
                    'arrowstyle': '-|>',
                    'color': color,
                    'lw': 0.5,
                    'mutation_scale': 3,
                    'shrinkA': 0,
                    'shrinkB': 0,
                },
            )
    counts = data['counts']
    subtitle = ', '.join(part for part in (data['cell'], f"+{data['pair'].offset} frames") if part)
    axis.set_title(f"{data['name']}\n{subtitle}", fontsize=6, color=INK_SECONDARY, pad=2, linespacing=1.1)
    axis.text(
        0.01,
        0.02,
        f"{counts['found']} found, {counts['displaced']} displaced, {counts['together']} together",
        transform=axis.transAxes,
        fontsize=5.5,
        color='white',
        va='bottom',
        bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 1, 'linewidth': 0},
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_xticks([])
    axis.set_yticks([])


def figure_cases(case_dirs: Sequence[Path], output_path: Path) -> Path:
    """Write the case gallery to `output_path`, one panel per case in survey-cell order."""
    if not case_dirs:
        raise CaseFigureError('no case to draw')
    cases = [case_fates(case_dir) for case_dir in cell_order(case_dirs)]
    height, width = cases[0]['frame'].shape[:2]
    columns = min(COLUMNS, len(cases))
    rows = int(np.ceil(len(cases) / columns))
    panel_width = (FIGURE_WIDTH_IN - (columns - 1) * GAP_IN) / columns
    panel_height = panel_width * height / width
    figure_height = LEGEND_IN + rows * (TITLE_IN + panel_height) + (rows - 1) * GAP_IN
    apply_style()
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, figure_height))
    for index, data in enumerate(cases):
        row, column = divmod(index, columns)
        top = LEGEND_IN + row * (TITLE_IN + panel_height + GAP_IN) + TITLE_IN
        axis = figure.add_axes(
            (
                column * (panel_width + GAP_IN) / FIGURE_WIDTH_IN,
                1 - (top + panel_height) / figure_height,
                panel_width / FIGURE_WIDTH_IN,
                panel_height / figure_height,
            )
        )
        _draw_case(axis, data)
    handles = [
        Patch(facecolor='#b4b3ad', edgecolor=color, linewidth=1.0, label=fate) for fate, color, _ in reversed(FATES)
    ]
    figure.legend(
        handles=handles,
        loc='upper center',
        ncol=len(handles),
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.1,
    )
    return _save(figure, output_path)
