#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the run-to-run spread of a repeated campaign."""

from __future__ import annotations

import numpy as np
import pytest

from stabilo_benchmark.report.repeats import repeat_spread, write_repeat_table
from stabilo_benchmark.results import ResultStore, build_manifest


def shifted(dx: float) -> list:
    return [[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def write_repeats(tmp_path, runs):
    store = ResultStore(tmp_path / 'results')
    store.initialize(build_manifest({'kind': 'dynamic', 'name': 'repeats', 'seed': 0}, 'h', {1: {'pipeline': 'x'}}))
    for number, (pipeline, records) in enumerate(runs.items(), start=1):
        store.write_run(
            number, {'run_number': number, 'params': {'pipeline': pipeline, 'masking': 'obb'}, 'records': records}
        )
    return store.directory


class TestRepeatSpread:
    def test_identical_repeats_have_no_spread(self, tmp_path):
        records = [{'pair': 'a', 'repeat': r, 'failed': False, 'homography': shifted(1.0)} for r in range(3)]
        (row,) = repeat_spread(write_repeats(tmp_path, {'sift+obb': records}), 640, 480).to_dict('records')
        assert row['spread_max'] == 0.0 and row['identical'] == 1.0 and row['failure_flips'] == 0

    def test_the_spread_is_the_largest_disagreement_between_repeats(self, tmp_path):
        records = [
            {'pair': 'a', 'repeat': r, 'failed': False, 'homography': shifted(dx)} for r, dx in enumerate((0, 0.1, 0.3))
        ]
        (row,) = repeat_spread(write_repeats(tmp_path, {'xfeat+obb': records}), 640, 480).to_dict('records')
        assert row['spread_max'] == pytest.approx(0.3)
        assert row['identical'] == 0.0

    def test_a_failure_in_some_repeats_is_counted_not_measured(self, tmp_path):
        records = [
            {'pair': 'a', 'repeat': 0, 'failed': False, 'homography': shifted(0.0)},
            {'pair': 'a', 'repeat': 1, 'failed': True},
            {'pair': 'b', 'repeat': 0, 'failed': False, 'homography': shifted(0.0)},
            {'pair': 'b', 'repeat': 1, 'failed': False, 'homography': shifted(0.0)},
        ]
        (row,) = repeat_spread(write_repeats(tmp_path, {'disk_lightglue+obb': records}), 640, 480).to_dict('records')
        assert row['failure_flips'] == 1 and row['n_pairs'] == 2

    def test_a_single_measurement_per_pair_gives_no_row(self, tmp_path):
        records = [{'pair': 'a', 'repeat': 0, 'failed': False, 'homography': shifted(0.0)}]
        assert repeat_spread(write_repeats(tmp_path, {'sift+obb': records}), 640, 480).empty

    def test_the_table_states_the_floor(self, tmp_path):
        records = [{'pair': 'a', 'repeat': r, 'failed': False, 'homography': shifted(0.1 * r)} for r in range(3)]
        spread = repeat_spread(write_repeats(tmp_path, {'xfeat+obb': records}), 640, 480)
        rendered = write_repeat_table(spread, tmp_path / 't.tex', 3, 'tab:r', epsilon_floor=0.83).read_text(
            encoding='utf-8'
        )
        assert '0.83 px' in rendered and np.isclose(spread['spread_max'].iloc[0], 0.2)
