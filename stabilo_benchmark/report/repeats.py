#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
repeats.py - Run-to-run spread of a campaign measured more than once per pair (`repeats` > 1).

For each pipeline and pair, the spread is the largest corner transfer disagreement between the
homographies of any two repeats, in reference-frame pixels; a pair whose repeats disagree on
whether the estimate failed is counted separately, since no distance describes it. The spread is
read against the case's audited resolution floor.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from stabilo_benchmark.metrics.geometric import corner_disagreement
from stabilo_benchmark.results import ResultStore

# Below this a spread counts as no difference (float64 round trip at 4K).
IDENTICAL_PX = 1e-9

COLUMNS: Dict[str, tuple] = {
    'n_pairs': ('Pairs', '{:.0f}', None),
    'spread_median': (r'Spread$_{50}$ [px]', '{:.4f}', None),
    'spread_p95': (r'Spread$_{95}$ [px]', '{:.4f}', None),
    'spread_max': (r'Spread$_{\max}$ [px]', '{:.4f}', None),
    'identical': ('Identical', '{:.1%}', None),
    'failure_flips': ('Fail flips', '{:.0f}', None),
}

CAPTION = (
    'Run-to-run spread over {repeats} repeats of every pair on one case. For each pair the spread is the '
    'largest corner transfer disagreement between the estimates of any two repeats, in reference-frame '
    'pixels, summarized over pairs; Identical is the share of pairs whose repeats agree to {identical:g} px, '
    'and Fail flips counts pairs on which the repeats disagree about whether the estimate failed. {floor}'
)


def repeat_spread(results_dir: Path, width: int, height: int) -> pd.DataFrame:
    """One row per run that measured some pair more than once."""
    rows: List[Dict[str, Any]] = []
    for run in ResultStore(Path(results_dir)).iter_runs():
        by_pair: Dict[str, List[Mapping[str, Any]]] = {}
        for record in run.get('records', []):
            by_pair.setdefault(str(record.get('pair')), []).append(record)
        spreads, flips = [], 0
        for records in by_pair.values():
            if len(records) < 2:
                continue
            failed = {bool(r.get('failed')) for r in records}
            if len(failed) > 1:
                flips += 1
                continue
            matrices = [r['homography'] for r in records if r.get('homography') is not None]
            if len(matrices) < 2:
                continue
            spreads.append(
                max(corner_disagreement(a, b, width, height)['cte_mean'] for a, b in combinations(matrices, 2))
            )
        if not spreads and not flips:
            continue
        values = np.asarray(spreads, dtype=float)
        params = run.get('params', {})
        rows.append(
            {
                'pipeline': params.get('pipeline'),
                'masking': params.get('masking'),
                'n_pairs': int(values.size + flips),
                'spread_median': float(np.median(values)) if values.size else float('nan'),
                'spread_p95': float(np.percentile(values, 95)) if values.size else float('nan'),
                'spread_max': float(values.max()) if values.size else float('nan'),
                'identical': float(np.mean(values <= IDENTICAL_PX)) if values.size else float('nan'),
                'failure_flips': int(flips),
            }
        )
    return pd.DataFrame(rows)


def write_repeat_table(
    spread: pd.DataFrame, path: Path, repeats: int, label: str, epsilon_floor: Optional[float] = None
) -> Path:
    """The spread table, against the audited floor when one exists."""
    from stabilo_benchmark.report.tables import write_latex_table  # noqa: PLC0415 - avoids an import cycle

    floor = (
        f"The audited resolution floor of this case is {epsilon_floor:.2f} px."
        if epsilon_floor is not None
        else 'This case has no audit, so no resolution floor is available for comparison.'
    )
    return write_latex_table(
        spread,
        path,
        columns=COLUMNS,
        caption=CAPTION.format(repeats=repeats, identical=IDENTICAL_PX, floor=floor),
        label=label,
        sort_by='spread_max',
        highlight_best=False,
    )
