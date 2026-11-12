#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Frame-pair sampling: reference resolution, the three modes, and determinism."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from stabilo_benchmark.config import load_config
from stabilo_benchmark.sampling import (
    DEFAULT_OFFSET_BINS,
    FramePair,
    SamplingError,
    bin_label,
    frames_to_extract,
    offset_bin_label,
    offset_bins,
    plan,
    plan_bins,
    plan_span,
    representative_offsets,
    resolve_reference,
    sample_pairs,
    summarize,
)

FRAME_COUNT = 1000


class TestReferenceResolution:
    def test_first_is_frame_zero(self):
        assert resolve_reference('first', FRAME_COUNT) == 0

    def test_middle_halves_the_sequence(self):
        assert resolve_reference('middle', FRAME_COUNT) == 500

    def test_explicit_index_is_used(self):
        assert resolve_reference(250, FRAME_COUNT) == 250

    def test_out_of_range_index_raises(self):
        with pytest.raises(SamplingError, match='outside'):
            resolve_reference(FRAME_COUNT, FRAME_COUNT)

    def test_unknown_keyword_raises(self):
        with pytest.raises(SamplingError, match='Invalid reference'):
            resolve_reference('last', FRAME_COUNT)

    def test_a_single_frame_cannot_be_benchmarked(self):
        with pytest.raises(SamplingError, match='at least two frames'):
            resolve_reference('first', 1)


class TestGridMode:
    def test_stride_selects_evenly(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 100}, 'first', 0)
        assert [pair.current_index for pair in pairs] == [100, 200, 300, 400, 500, 600, 700, 800, 900]

    def test_offsets_are_multiples_of_the_stride(self):
        # The grid is anchored at the reference: offsets are multiples of the stride.
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 50}, 'first', 0)
        assert all(pair.offset % 50 == 0 for pair in pairs)
        assert pairs[0].offset == 50

    def test_a_case_runs_forward_from_a_reference_chosen_inside_the_recording(self):
        # Nothing before the reference is sampled.
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 100, 'max_pairs': 3}, 450, 0)
        assert [pair.current_index for pair in pairs] == [550, 650, 750]
        assert all(pair.offset > 0 for pair in pairs)

    def test_reference_frame_is_never_paired_with_itself(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 1}, 'first', 0)
        assert all(pair.current_index != pair.reference_index for pair in pairs)

    def test_max_pairs_caps_the_plan(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 10, 'max_pairs': 7}, 'first', 0)
        assert len(pairs) == 7

    def test_start_and_stop_bound_the_range(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 50, 'start': 200, 'stop': 500}, 'first', 0)
        assert all(200 <= pair.current_index <= 500 for pair in pairs)

    def test_zero_stride_raises(self):
        with pytest.raises(SamplingError, match=">= 1"):
            sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 0}, 'first', 0)

    def test_available_whitelist_is_honoured(self):
        available = [0, 100, 200, 300]
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 1}, 'first', 0, available=available)
        assert [pair.current_index for pair in pairs] == [100, 200, 300]


class TestRandomMode:
    def test_draws_the_requested_count_without_replacement(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'random', 'count': 25}, 'first', 0)
        indices = [pair.current_index for pair in pairs]
        assert len(indices) == 25
        assert len(set(indices)) == 25

    def test_same_seed_reproduces_the_draw(self):
        a = sample_pairs(FRAME_COUNT, {'mode': 'random', 'count': 20}, 'first', 4)
        b = sample_pairs(FRAME_COUNT, {'mode': 'random', 'count': 20}, 'first', 4)
        assert [p.current_index for p in a] == [p.current_index for p in b]

    def test_different_seeds_differ(self):
        a = sample_pairs(FRAME_COUNT, {'mode': 'random', 'count': 20}, 'first', 1)
        b = sample_pairs(FRAME_COUNT, {'mode': 'random', 'count': 20}, 'first', 2)
        assert [p.current_index for p in a] != [p.current_index for p in b]

    def test_count_above_the_supply_is_clamped(self):
        pairs = sample_pairs(10, {'mode': 'random', 'count': 500}, 'first', 0)
        assert len(pairs) == 9


class TestStratifiedMode:
    def test_each_bin_contributes(self):
        config = {'mode': 'stratified', 'bins': [[1, 100], [101, 500], [501, 999]], 'per_bin': 5}
        pairs = sample_pairs(FRAME_COUNT, config, 'first', 0)
        offsets = [pair.abs_offset for pair in pairs]
        assert sum(1 for o in offsets if o <= 100) == 5
        assert sum(1 for o in offsets if 101 <= o <= 500) == 5
        assert sum(1 for o in offsets if o >= 501) == 5

    def test_a_reference_inside_the_recording_is_followed_not_surrounded(self):
        # Offsets count forward, as in grid mode: a bin of one offset holds one frame.
        config = {'mode': 'stratified', 'bins': [[1, 1], [2, 3], [4, 100]], 'per_bin': 5}
        pairs = sample_pairs(FRAME_COUNT, config, 450, 0)
        assert all(pair.offset > 0 for pair in pairs)
        assert [pair.offset for pair in pairs][:3] == [1, 2, 3] and len(pairs) == 8

    def test_missing_bins_raises(self):
        with pytest.raises(SamplingError, match="'bins' is required"):
            sample_pairs(FRAME_COUNT, {'mode': 'stratified'}, 'first', 0)

    def test_malformed_bin_raises(self):
        with pytest.raises(SamplingError, match='min_offset, max_offset'):
            sample_pairs(FRAME_COUNT, {'mode': 'stratified', 'bins': [[1, 2, 3]]}, 'first', 0)

    def test_bins_outside_the_sequence_raise(self):
        with pytest.raises(SamplingError, match='selected no frames'):
            sample_pairs(50, {'mode': 'stratified', 'bins': [[10000, 20000]]}, 'first', 0)


class TestFramePair:
    def test_offset_is_signed_and_absolute_is_not(self):
        assert FramePair(500, 400).offset == -100
        assert FramePair(500, 400).abs_offset == 100

    def test_dict_carries_both(self):
        payload = FramePair(0, 250).to_dict()
        assert payload == {'reference_index': 0, 'current_index': 250, 'offset': 250, 'abs_offset': 250}


class TestHelpers:
    def test_unknown_mode_raises(self):
        with pytest.raises(SamplingError, match='Unknown sampling mode'):
            sample_pairs(FRAME_COUNT, {'mode': 'spiral'}, 'first', 0)

    def test_frames_to_extract_includes_the_reference_once(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 100}, 'first', 0)
        needed = frames_to_extract(pairs)
        assert needed[0] == 0
        assert len(needed) == len(set(needed))
        assert len(needed) == len(pairs) + 1

    def test_plan_returns_pairs_and_frames_together(self):
        pairs, frames = plan(FRAME_COUNT, {'mode': 'grid', 'stride': 200}, 'first', 0)
        assert len(frames) == len(pairs) + 1

    def test_summary_records_the_span(self):
        pairs = sample_pairs(FRAME_COUNT, {'mode': 'grid', 'stride': 100}, 'first', 0)
        summary = summarize(pairs, fps=30.0)
        assert summary['n_pairs'] == len(pairs)
        assert summary['reference_index'] == 0
        assert summary["span_seconds"] == pytest.approx(900 / 30.0)

    def test_summary_of_no_pairs(self):
        assert summarize([]) == {'n_pairs': 0}

    def test_offset_bins_group_by_distance(self):
        pairs = [FramePair(0, index) for index in (10, 50, 200, 800)]
        grouped = offset_bins(pairs, [100, 500])
        assert len(grouped['0-100']) == 2
        assert len(grouped['101-500']) == 1
        assert len(grouped['>500']) == 1


class TestPlanSpan:
    def test_a_grid_reaches_stride_times_pairs(self):
        assert plan_span({'mode': 'grid', 'stride': 50, 'max_pairs': 200}) == 10000

    def test_a_grid_without_a_cap_is_unbounded(self):
        assert plan_span({'mode': 'grid', 'stride': 50}) is None

    def test_a_stratified_plan_reaches_its_widest_bin(self):
        assert plan_span({'mode': 'stratified', 'bins': [[1, 100], [500, 3000], [101, 499]]}) == 3000

    def test_a_random_plan_is_unbounded(self):
        assert plan_span({'mode': 'random', 'count': 10}) is None

    def test_the_span_matches_the_frames_a_grid_samples(self):
        config = {'mode': 'grid', 'stride': 50, 'max_pairs': 20}
        pairs = sample_pairs(FRAME_COUNT * 2, config, reference=0)
        assert max(pair.current_index for pair in pairs) == plan_span(config)


class TestPlanBins:
    def test_the_plan_s_bins_are_returned(self):
        assert plan_bins({'bins': [[1, 15], [16, 30]]}) == ((1, 15), (16, 30))

    @pytest.mark.parametrize('key', ['survey_bins', 'audit_bins'])
    def test_survey_and_audit_bins_fall_back_to_the_analysis_bins(self, key):
        assert plan_bins({'bins': [[1, 15]]}, key) == ((1, 15),)
        assert plan_bins({'bins': [[1, 15]], key: [[1, 99]]}, key) == ((1, 99),)

    def test_a_grid_plan_uses_the_defaults(self):
        assert plan_bins({'mode': 'grid', 'stride': 50}) == DEFAULT_OFFSET_BINS
        assert plan_bins(None) == DEFAULT_OFFSET_BINS

    def test_offsets_are_labeled_by_their_bin(self):
        bins = ((1, 15), (16, 30), (31, None))
        assert offset_bin_label(15, bins) == '1-15'
        assert offset_bin_label(-16, bins) == '16-30'
        assert offset_bin_label(10_000, bins) == '>30'
        assert offset_bin_label(0, bins) is None

    def test_the_shipped_plan_fills_every_bin(self):
        # Every offset of a bin narrower than `per_bin`, `per_bin` pairs in every other.
        config = load_config(
            Path(__file__).resolve().parents[1] / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml'
        )
        sampling = config['sampling']
        pairs = sample_pairs(3601, sampling, reference=0, seed=0)
        bins = plan_bins(sampling)
        counts = Counter(offset_bin_label(pair.current_index, bins) for pair in pairs)
        assert len(counts) == len(bins)
        for low, high in bins:
            assert counts[bin_label(low, high)] == min(sampling['per_bin'], high - low + 1)
        assert max(pair.current_index for pair in pairs) <= plan_span(sampling)

    def test_the_shipped_survey_reads_short_offsets_of_the_plan(self):
        sampling = load_config(
            Path(__file__).resolve().parents[1] / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml'
        )['sampling']
        survey = plan_bins(sampling, 'survey_bins')
        assert set(survey) <= set(plan_bins(sampling))
        assert representative_offsets(survey) == [1, 2, 5, 11, 22]
