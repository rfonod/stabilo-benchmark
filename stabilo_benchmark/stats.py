#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stats.py - The analysis plan: when two groups in a table count as different.

Comparisons are paired within a cluster and read across clusters: the site in Part I (two drones
over one intersection are one site), the case in Part II. For one group against the best group in
a column:

  estimate      Hodges-Lehmann estimate of the per-cluster paired differences, in the column's
                units;
  ci_low/high   percentile cluster-bootstrap interval of that estimate;
  p_value       exact two-sided sign-flip permutation test on the per-cluster differences,
                enumerated up to EXACT_FLIP_LIMIT clusters and sampled beyond;
  distinguishable
                the interval excludes zero AND |estimate| >= margin, the column's equivalence
                margin (`margin_for`).

`tie_set` is what a table bolds: the best group and every group not distinguishable from it.
With n clusters the smallest two-sided p is 2 / 2^n (`minimum_p`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import PurePath
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

# Above this many clusters the sign-flip test is sampled rather than enumerated.
EXACT_FLIP_LIMIT = 16
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 0
# One percent of the mAA range: the smallest difference in the ranking quantity worth claiming.
MAA_MARGIN = 0.01


@dataclass(frozen=True)
class PairedComparison:
    """One group against the best group in a column, across clusters."""

    n_clusters: int
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    margin: float

    @property
    def distinguishable(self) -> bool:
        if self.n_clusters < 2 or not np.isfinite(self.estimate):
            return False
        excludes_zero = self.ci_low > 0.0 or self.ci_high < 0.0
        return excludes_zero and abs(self.estimate) >= self.margin

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'estimate': self.estimate,
            'ci_low': self.ci_low,
            'ci_high': self.ci_high,
            'p_value': self.p_value,
            'margin': self.margin,
            'distinguishable': self.distinguishable,
        }


@dataclass(frozen=True)
class TieSet:
    """The best group of a column and every group not distinguishable from it."""

    best: Hashable
    tied: Set[Hashable]
    comparisons: Dict[Hashable, PairedComparison]

    @property
    def bold(self) -> Set[Hashable]:
        return {self.best, *self.tied}


def hodges_lehmann(values: Sequence[float]) -> float:
    """The median of the pairwise means (Walsh averages) of a sample; NaN for an empty one."""
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    i, j = np.triu_indices(x.size)
    return float(np.median((x[i] + x[j]) / 2.0))


# Walsh averages held at once by `_hodges_lehmann_rows` (8 bytes each).
WALSH_BLOCK = 4_000_000


def _hodges_lehmann_rows(samples: np.ndarray) -> np.ndarray:
    """`hodges_lehmann` of every row of a 2-D array of finite values, the same value as row by row."""
    i, j = np.triu_indices(samples.shape[1])
    rows = max(1, WALSH_BLOCK // i.size)
    return np.concatenate(
        [
            np.median((block[:, i] + block[:, j]) / 2.0, axis=1)
            for block in (samples[start : start + rows] for start in range(0, samples.shape[0], rows))
        ]
    )


def sign_flip_p(values: Sequence[float], draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> float:
    """
    Exact two-sided sign-flip permutation p-value for the mean of paired differences.

    Every 2^n assignment of signs is enumerated up to `EXACT_FLIP_LIMIT` clusters; beyond that
    `draws` random flips are used. The observed configuration is included in the reference
    set, so the smallest attainable value with n clusters is 2 / 2^n.
    """
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float('nan')
    observed = abs(x.sum())
    if n <= EXACT_FLIP_LIMIT:
        signs = np.array(list(product((-1.0, 1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(int(draws), n))
    sums = np.abs(signs @ x)
    # The matrix product and `x.sum()` round differently; a tolerance relative to the data keeps the
    # observed configuration (and its mirror) in the count at any magnitude.
    tolerance = 1e-9 * float(np.abs(x).sum())
    return float(np.mean(sums >= observed - tolerance))


def minimum_p(n_clusters: int) -> float:
    """The smallest two-sided sign-flip p-value `n_clusters` can produce."""
    return 2.0 / (2.0**n_clusters) if n_clusters > 0 else float('nan')


def cluster_bootstrap_ci(
    values: Sequence[float],
    statistic=hodges_lehmann,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> tuple:
    """Percentile bootstrap interval of `statistic` over clusters resampled with replacement."""
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    resampled = x[rng.integers(0, x.size, size=(int(draws), x.size))]
    if statistic is hodges_lehmann:
        estimates = _hodges_lehmann_rows(resampled)
    else:
        estimates = np.array([statistic(row) for row in resampled])
    alpha = (1.0 - level) / 2.0
    return float(np.percentile(estimates, 100 * alpha)), float(np.percentile(estimates, 100 * (1 - alpha)))


def paired_comparison(
    differences: Sequence[float],
    margin: float = 0.0,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Estimate, interval and test for one set of per-cluster paired differences."""
    x = np.asarray(list(differences), dtype=np.float64)
    x = x[np.isfinite(x)]
    low, high = cluster_bootstrap_ci(x, draws=draws, seed=seed)
    return PairedComparison(
        n_clusters=int(x.size),
        estimate=hodges_lehmann(x),
        ci_low=low,
        ci_high=high,
        p_value=sign_flip_p(x, draws=draws, seed=seed),
        margin=float(margin),
    )


def tie_set(
    cluster_table: pd.DataFrame,
    best: Hashable,
    higher_is_better: bool,
    margin: float = 0.0,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> TieSet:
    """
    The groups a column may bold.

    `cluster_table` has one row per cluster and one column per group, holding the group's
    statistic on that cluster alone. `best` is the group the pooled table ranks first. Each
    other group is compared with it on the clusters where both are finite; a group whose
    difference to the best is not distinguishable joins the tie set.
    """
    if best not in cluster_table.columns:
        raise ValueError(f"best group {best!r} is not a column of the cluster table")
    sign = 1.0 if higher_is_better else -1.0
    reference = pd.to_numeric(cluster_table[best], errors='coerce')
    comparisons: Dict[Hashable, PairedComparison] = {}
    tied: Set[Hashable] = set()
    for group in cluster_table.columns:
        if group == best:
            continue
        other = pd.to_numeric(cluster_table[group], errors='coerce')
        # Positive means the best group is better than this one, in the column's direction.
        differences = (sign * (reference - other)).to_numpy(dtype=float)
        comparison = paired_comparison(differences, margin=margin, draws=draws, seed=seed)
        comparisons[group] = comparison
        if not comparison.distinguishable:
            tied.add(group)
    return TieSet(best=best, tied=tied, comparisons=comparisons)


# Image extensions a scene name may carry (results record the file name).
SCENE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def site_of(scene: str) -> str:
    """
    The cluster a Part I scene belongs to.

    The scene names of the manuscript's campaign end in a drone token (`..._<site>_D<n>`, before an
    image extension if the name carries one), and two drones over one site are two views of the
    same intersection, so the drone token is stripped. A name without one clusters on itself, which
    is the conservative choice: it never merges scenes that are not known to share a site.
    """
    name = str(scene)
    if PurePath(name).suffix.lower() in SCENE_SUFFIXES:
        name = PurePath(name).stem
    head, sep, tail = name.rpartition('_')
    if sep and len(tail) >= 2 and tail[0] in 'Dd' and tail[1:].isdigit():
        return head
    return name


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Holm's step-down adjustment of a family of p-values, controlling the family-wise error.

    The comparisons of one column form one family (every group against the best), and a
    per-group claim read off that family is made on the adjusted value. NaN entries are left
    NaN and do not count toward the family size.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return adjusted.tolist()
    order = finite[np.argsort(p[finite], kind='stable')]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def describe(comparisons: Mapping[Hashable, PairedComparison]) -> pd.DataFrame:
    """
    The comparisons of one column as a table, for a supplementary listing.

    `p_holm` is the sign-flip p-value adjusted within the column's family by `holm_adjust`.
    """
    rows = [{'group': group, **comparison.to_dict()} for group, comparison in comparisons.items()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['p_holm'] = holm_adjust(frame['p_value'].to_numpy(dtype=float))
    return frame


def margin_for(column: str, epsilon_floor: Optional[float] = None, annotation_floor: Optional[float] = None) -> float:
    """
    The equivalence margin of a table column under the analysis plan.

    mAA uses one percent of its range. Reference-relative pixel metrics (corner error, object
    center error) use the audited residual when one is given; metrics against the ground
    control points use the click repeatability when one is given; everything else uses zero,
    which leaves the bootstrap interval as the only criterion.
    """
    if column.startswith('maa'):
        return MAA_MARGIN
    if column.startswith('cte_') or column.startswith('oce_'):
        if annotation_floor is not None:
            return float(annotation_floor)
        if epsilon_floor is not None:
            return float(epsilon_floor)
    return 0.0
