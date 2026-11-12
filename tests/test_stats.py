#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""The analysis plan's arithmetic: the one invariant that decides what a table may bold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stabilo_benchmark import stats
from stabilo_benchmark.stats import (
    cluster_bootstrap_ci,
    describe,
    hodges_lehmann,
    holm_adjust,
    minimum_p,
    sign_flip_p,
    site_of,
    tie_set,
)


class TestSignFlip:
    def test_nine_clusters_bottom_out_at_two_in_five_hundred_twelve(self):
        # With nine clusters the smallest p is 2/512.
        assert sign_flip_p([1.0] * 9) == minimum_p(9) == 2 / 512
        assert sign_flip_p([1.0] * 8 + [-1.0]) < 0.05
        assert sign_flip_p([1.0] * 7 + [-1.0] * 2) > 0.05

    def test_the_observed_configuration_counts_at_any_magnitude(self):
        # Differences of thousands of pixels: rounding in the enumeration must not drop the observed sum.
        for seed in range(200):
            x = np.random.default_rng(seed).uniform(1e3, 1e4, 5)
            assert sign_flip_p(x) == minimum_p(5)
            # Flipping a small opposing difference gives a larger sum, which the tolerance must not absorb.
            assert sign_flip_p(np.concatenate([x, [-1e-3]])) == 2 * minimum_p(6)

    def test_hodges_lehmann_is_the_median_of_walsh_averages(self):
        assert hodges_lehmann([1.0, 2.0, 10.0]) == np.median([1.0, 1.5, 5.5, 2.0, 6.0, 10.0])


class TestBootstrap:
    def test_the_interval_is_the_estimator_applied_draw_by_draw(self):
        # The Hodges-Lehmann interval is computed on all draws at once; it must equal the estimator
        # applied to each draw, bit for bit, at odd and even pair counts and with ties.
        def draw_by_draw(values):
            return hodges_lehmann(values)

        for n in (2, 3, 8, 9, 29):
            x = np.round(np.random.default_rng(n).normal(size=n), 2)
            assert cluster_bootstrap_ci(x, draws=3000) == cluster_bootstrap_ci(x, draw_by_draw, 3000)

    def test_the_draws_are_held_in_blocks_with_the_same_result(self, monkeypatch):
        x = np.random.default_rng(1).normal(size=9)
        whole = cluster_bootstrap_ci(x, draws=1000)
        monkeypatch.setattr(stats, 'WALSH_BLOCK', 100)
        assert cluster_bootstrap_ci(x, draws=1000) == whole


class TestTieSet:
    def test_a_consistent_gap_separates_and_a_null_gap_ties(self):
        rng = np.random.default_rng(0)
        clusters = [f"case{i}" for i in range(9)]
        best = rng.normal(0.9, 0.02, size=9)
        table = pd.DataFrame(
            {'obb': best, 'none': best - 0.1, 'hbb': best + rng.normal(0.0, 0.005, size=9)}, index=clusters
        )
        result = tie_set(table, 'obb', higher_is_better=True, margin=0.01, draws=2000)
        assert result.tied == {'hbb'}
        assert result.comparisons['none'].distinguishable
        assert not result.comparisons['hbb'].distinguishable

    def test_a_single_cluster_can_never_distinguish(self):
        table = pd.DataFrame({'a': [0.9], 'b': [0.1]}, index=['only'])
        assert tie_set(table, 'a', higher_is_better=True).tied == {'b'}


class TestSite:
    def test_the_drone_token_is_stripped_and_nothing_else_is(self):
        assert site_of('2022-10-07_PM4_B1_D1') == site_of('2022-10-07_PM4_B1_D10') == '2022-10-07_PM4_B1'
        assert site_of('b1_000000_d1') == 'b1_000000'
        assert site_of('scene_07') == 'scene_07'

    def test_an_image_extension_does_not_hide_the_drone_token(self):
        # Results record scene file names.
        assert site_of('2022-10-07_PM4_S_D9.jpg') == site_of('2022-10-07_PM4_S_D10.jpg') == '2022-10-07_PM4_S'
        assert site_of('scene_07.png') == 'scene_07'


class TestHolm:
    def test_step_down_adjustment_matches_the_textbook_example(self):
        # Holm (1979): sort ascending, multiply the k-th smallest by (m - k + 1), enforce monotonicity.
        assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
        assert holm_adjust([0.2]) == [0.2]
        assert holm_adjust([]) == []

    def test_nan_entries_stay_nan_and_do_not_count_toward_the_family(self):
        adjusted = holm_adjust([0.01, float('nan'), 0.04])
        assert adjusted[0] == 0.02 and adjusted[2] == 0.04 and adjusted[1] != adjusted[1]

    def test_describe_carries_the_adjusted_column(self):
        table = pd.DataFrame({'best': [1.0, 1.1, 0.9, 1.2], 'a': [0.5, 0.6, 0.4, 0.7], 'b': [1.0, 1.1, 0.9, 1.2]})
        listing = describe(tie_set(table, 'best', higher_is_better=True, draws=500).comparisons)
        assert list(listing['p_holm']) == holm_adjust(listing['p_value'])
        assert (listing['p_holm'] >= listing['p_value']).all()
