#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the planned masking and mechanism contrasts of Part II."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stabilo_benchmark.report.contrasts import (
    OUTCOMES,
    POOLED,
    benefit_strata,
    case_table,
    masking_contrasts,
    mechanism_contrasts,
    preset_of,
    write_benefit_strata_table,
    write_contrasts_table,
    write_mechanism_table,
)
from stabilo_benchmark.stats import minimum_p

CASES = [f"case{i}" for i in range(9)]


def outcome_table(obb_better=9, presets=('sift', 'orb'), improved=tuple(OUTCOMES), harmed=()):
    """
    Per-case outcomes under none/hbb/obb. For `sift`, OBB beats none on every `improved` outcome in the
    first `obb_better` cases and loses in the rest, and loses on every `harmed` outcome in every case;
    `orb` shows no masking effect at all.
    """
    # Per unit gain: (no-masking value, change toward better).
    steps = {'cte_gcp': (1.0, -0.1), 'zncc': (0.9, 0.001), 'gradient': (0.8, 0.001), 'iou_gcp': (0.95, 0.001)}
    steps['oce_gcp'] = (0.5, -0.05)
    rows = []
    for i, case in enumerate(CASES):
        for preset in presets:
            for masking in ('none', 'hbb', 'obb'):
                row = {'case': case, 'preset': preset, 'masking': masking}
                for outcome, (value, step) in steps.items():
                    gain = 0.0
                    if preset == 'sift' and masking == 'obb':
                        if outcome in harmed:
                            gain = -(1 + i)
                        elif outcome in improved:
                            gain = (1.0 if i < obb_better else -1.0) * (1 + i)
                    row[outcome] = value + step * gain
                rows.append(row)
    return pd.DataFrame(rows)


def flip(table, case, outcomes):
    """Reverse the OBB-against-none effect of `sift` in one case on the given outcomes."""
    table = table.copy()
    rows = (table['preset'] == 'sift') & (table['case'] == case) & (table['masking'] == 'obb')
    for outcome in outcomes:
        none = table.loc[(table['preset'] == 'sift') & (table['case'] == case) & (table['masking'] == 'none'), outcome]
        table.loc[rows, outcome] = 2 * none.to_numpy() - table.loc[rows, outcome].to_numpy()
    return table


def pooled_primary(result):
    return result[(result['role'] == 'primary') & (result['pipeline'] == POOLED)].set_index('outcome')


class TestMaskingContrasts:
    def test_the_benefit_is_signed_so_positive_favors_the_first_mode(self):
        result = masking_contrasts(outcome_table())
        sift = result[(result['contrast'] == 'obb vs none') & (result['pipeline'] == 'sift')]
        assert set(sift['outcome']) == set(OUTCOMES)
        assert (sift['estimate'] > 0).all()  # lower CTE and OCE count as a benefit too
        assert (sift['n_agree'] == 9).all()

    def test_a_benefit_over_all_pipelines_is_claimed(self):
        result = masking_contrasts(outcome_table())
        pooled = pooled_primary(result)
        assert set(pooled.index) == set(OUTCOMES) and (pooled['n_agree'] == 9).all()
        assert pooled['claim'].all()
        assert not result.loc[result['pipeline'] != POOLED, 'claim'].any()

    def test_one_outcome_in_favor_suffices(self):
        pooled = pooled_primary(masking_contrasts(outcome_table(improved=('zncc',))))
        assert pooled.loc['zncc', 'favors'] and not pooled.drop('zncc')['favors'].any()
        assert pooled['claim'].all()

    def test_an_outcome_against_blocks_the_claim(self):
        pooled = pooled_primary(masking_contrasts(outcome_table(improved=('zncc',), harmed=('cte_gcp',))))
        assert pooled.loc['zncc', 'favors'] and pooled.loc['cte_gcp', 'harms']
        assert not pooled['claim'].any()

    def test_holm_runs_across_the_outcomes_of_the_pooled_reading(self):
        pooled = pooled_primary(masking_contrasts(outcome_table()))
        assert pooled['p_holm'].to_numpy() == pytest.approx(len(OUTCOMES) * minimum_p(9))

    def test_a_secondary_contrast_carries_no_claim(self):
        result = masking_contrasts(outcome_table())
        assert not result.loc[result['role'] == 'secondary', 'claim'].any()

    def test_seven_of_nine_cases_do_not_reach_the_level(self):
        # Here the two dissenting cases are the largest, so p is far above 0.05.
        pooled = pooled_primary(masking_contrasts(outcome_table(obb_better=7)))
        assert (pooled['p_value'] > 0.05).all()
        assert not pooled['claim'].any()

    def test_eight_of_nine_is_claimed_when_the_dissenting_case_is_the_smallest(self):
        # p = 4/512 for each outcome, 0.039 after Holm over the five.
        table = flip(outcome_table(), 'case0', OUTCOMES)
        pooled = pooled_primary(masking_contrasts(table))
        assert (pooled['n_agree'] == 8).all() and pooled['claim'].all()

    def test_a_harm_vetoes_on_its_unadjusted_p(self):
        # CTE favors no masking in 8 of 9 cases (dissent fifth smallest): p = 0.039, above 0.05 after Holm.
        table = flip(outcome_table(improved=('zncc',), harmed=('cte_gcp',)), 'case4', ('cte_gcp',))
        pooled = pooled_primary(masking_contrasts(table))
        assert pooled.loc['cte_gcp', 'p_value'] <= 0.05 < pooled.loc['cte_gcp', 'p_holm']
        assert pooled.loc['cte_gcp', 'harms'] and pooled.loc['zncc', 'favors']
        assert not pooled['claim'].any()

    def test_every_outcome_averages_the_same_pipelines(self):
        table = outcome_table(presets=('sift', 'orb', 'xfeat'))
        table.loc[(table['preset'] == 'xfeat') & (table['case'] == 'case3'), 'cte_gcp'] = np.nan
        table.loc[table['preset'] == 'xfeat', 'zncc'] += 0.5 * (table['masking'] == 'obb')
        pooled = pooled_primary(masking_contrasts(table))
        # xfeat lacks one CTE value, so it leaves the ZNCC mean too: half of sift's gain, not a third plus xfeat's.
        assert pooled.loc['zncc', 'estimate'] == pytest.approx(0.0005 * 5)

    def test_a_case_missing_from_an_outcome_blocks_the_claim(self):
        table = outcome_table()
        table.loc[table['case'] == 'case0', ['cte_gcp', 'iou_gcp', 'oce_gcp']] = np.nan
        pooled = pooled_primary(masking_contrasts(table))
        assert pooled.loc['cte_gcp', 'n_clusters'] == 8
        assert not pooled['complete'].any() and not pooled['claim'].any()

    def test_a_missing_outcome_blocks_the_claim(self):
        # Without the GCP outcomes the Holm family is incomplete, so nothing is claimed.
        table = outcome_table().assign(cte_gcp=np.nan, iou_gcp=np.nan, oce_gcp=np.nan)
        result = masking_contrasts(table)
        assert set(result['outcome']) == {'zncc', 'gradient'}
        assert not result['claim'].any()

    def test_the_click_repeatability_is_the_margin_of_the_pixel_outcomes(self):
        # A 0.1 px benefit per unit gain stays below a 10 px margin; the reference-free outcomes still carry the claim.
        pooled = pooled_primary(masking_contrasts(outcome_table(), annotation_floor=10.0))
        assert pooled.loc['cte_gcp', 'margin'] == 10.0 and not pooled.loc['cte_gcp', 'favors']
        assert pooled.loc['zncc', 'margin'] == 0.0 and pooled.loc['zncc', 'favors']
        assert pooled['claim'].all()

    def test_holm_runs_across_pipelines(self):
        result = masking_contrasts(outcome_table(presets=('sift', 'orb', 'xfeat')))
        family = result[
            (result['contrast'] == 'obb vs none') & (result['outcome'] == 'zncc') & (result['pipeline'] != POOLED)
        ]
        assert len(family) == 3 and (family['p_holm'] >= family['p_value']).all()

    def test_a_pipeline_missing_from_a_case_is_left_out_of_the_pooled_reading(self):
        table = outcome_table(presets=('sift', 'orb', 'loftr'))
        table = table[~((table['preset'] == 'loftr') & (table['case'] == 'case0'))]
        pooled = pooled_primary(masking_contrasts(table))
        # The mean over sift and orb alone: half of sift's ZNCC gain of 0.001 * (1 + i).
        assert pooled.loc['zncc', 'estimate'] == pytest.approx(0.0005 * 5)


def pair_frame(seed=0):
    """Per-pair measurements in which the OBB benefit grows with coverage and with motion."""
    rng = np.random.default_rng(seed)
    rows = []
    for case in CASES:
        for pair in range(40):
            coverage = rng.uniform(0.01, 0.1)
            persistence = rng.uniform(0, 1)
            benefit = 0.1 * coverage + 0.002 * (1 - persistence)
            for masking in ('none', 'obb'):
                gain = benefit if masking == 'obb' else 0.0
                rows.append(
                    {
                        'case': case,
                        'pipeline': f"sift+{masking}",
                        'masking': masking,
                        'pair': f"0->{pair}",
                        'coverage_cur': coverage,
                        'persistence': persistence,
                        'zncc': 0.9 + gain,
                        'gradient': 0.8 + gain,
                        'motion_alignment': 0.6 if masking == 'none' else 0.5,
                    }
                )
    return pd.DataFrame(rows)


class TestMechanismContrasts:
    def test_all_three_contrasts_point_the_predicted_way(self):
        result = mechanism_contrasts(pair_frame())
        result = result[result['pipeline'] == 'sift'].set_index(['contrast', 'outcome'])
        assert result.loc[('coverage', 'zncc'), 'estimate'] > 0.5
        assert result.loc[('persistence', 'gradient'), 'estimate'] > 0
        assert result.loc[('alignment', 'motion_alignment'), 'estimate'] == pytest.approx(0.1)
        assert result['favors'].all()
        assert (result['n_agree'] == 9).all()

    def test_each_contrast_is_also_read_over_all_pipelines_first(self):
        frame = pair_frame()
        # A second pipeline with no benefit at all halves the pooled benefit in every case.
        flat = frame.assign(pipeline=frame['pipeline'].str.replace('sift', 'orb'))
        flat.loc[flat['masking'] == 'obb', ['zncc', 'gradient']] = flat.loc[
            flat['masking'] == 'none', ['zncc', 'gradient']
        ].to_numpy()
        result = mechanism_contrasts(pd.concat([frame, flat], ignore_index=True))
        assert result['pipeline'].iloc[0] == POOLED
        pooled = result[result['pipeline'] == POOLED].set_index(['contrast', 'outcome'])
        sift = result[result['pipeline'] == 'sift'].set_index(['contrast', 'outcome'])
        assert pooled.loc[('persistence', 'zncc'), 'estimate'] == pytest.approx(
            sift.loc[('persistence', 'zncc'), 'estimate'] / 2
        )
        # Holm runs across the contrast's two outcomes over all pipelines, so 9 of 9 still passes.
        assert pooled.loc[('coverage', 'zncc'), 'p_holm'] == pytest.approx(2 * minimum_p(9))
        assert pooled.loc[('persistence', 'gradient'), 'favors']

    def test_no_effect_is_not_favored(self):
        frame = pair_frame()
        frame.loc[frame['masking'] == 'obb', ['zncc', 'gradient']] = frame.loc[
            frame['masking'] == 'none', ['zncc', 'gradient']
        ].to_numpy()
        frame['motion_alignment'] = 0.5
        result = mechanism_contrasts(frame)
        assert not result['favors'].any()

    def test_a_benefit_that_follows_offset_alone_vanishes_at_matched_offset(self):
        # Persistence falls with offset and the benefit falls with offset; within an offset bin,
        # persistence carries no information about the benefit.
        rng = np.random.default_rng(1)
        rows = []
        for case in CASES:
            for pair in range(60):
                offset = int(rng.choice([5, 50, 500]))
                persistence = {5: 0.8, 50: 0.5, 500: 0.2}[offset] + rng.uniform(-0.1, 0.1)
                benefit = {5: 0.003, 50: 0.001, 500: 0.0}[offset]
                for masking in ('none', 'obb'):
                    gain = benefit if masking == 'obb' else 0.0
                    rows.append(
                        {
                            'case': case,
                            'pipeline': f"sift+{masking}",
                            'masking': masking,
                            'pair': f"0->{pair}",
                            'abs_offset': offset,
                            'coverage_cur': 0.05,
                            'persistence': persistence,
                            'zncc': 0.9 + gain,
                            'gradient': 0.8 + gain,
                        }
                    )
        result = mechanism_contrasts(pd.DataFrame(rows), bins=BINS)
        sift = result[result['pipeline'] == 'sift'].set_index(['contrast', 'outcome'])
        assert sift.loc[('persistence', 'zncc'), 'estimate'] < -0.001
        assert sift.loc[('persistence_offset', 'zncc'), 'estimate'] == pytest.approx(0.0, abs=1e-12)

    def test_the_matched_offset_reading_is_never_favored(self):
        frame = pair_frame().assign(abs_offset=10)
        result = mechanism_contrasts(frame, bins=BINS)
        sensitivity = result[result['contrast'] == 'persistence_offset']
        assert not sensitivity.empty and (sensitivity['estimate'] > 0).all()
        assert not sensitivity['favors'].any()
        assert result.loc[result['contrast'] == 'persistence', 'favors'].all()

    def test_without_bins_there_is_no_matched_offset_reading(self):
        assert 'persistence_offset' not in set(mechanism_contrasts(pair_frame().assign(abs_offset=10))['contrast'])


class TestCaseTable:
    def test_joins_the_gcp_outcomes_by_case_pipeline_and_mode(self):
        frame = pair_frame()
        gcp = [
            {
                'case': 'case0',
                'pipeline': 'sift+obb',
                'masking': 'obb',
                'cte_median': 0.4,
                'iou_mean': 0.9,
                'oce_mean': 0.2,
            }
        ]
        table = case_table(frame, gcp).set_index(['case', 'preset', 'masking'])
        assert table.loc[('case0', 'sift', 'obb'), 'cte_gcp'] == 0.4
        assert np.isnan(table.loc[('case1', 'sift', 'obb'), 'cte_gcp'])
        assert table.loc[('case0', 'sift', 'none'), 'zncc'] == pytest.approx(0.9)

    def test_preset_of_strips_the_masking_and_repeat_suffix(self):
        assert preset_of('disk_lightglue+obb#2') == 'disk_lightglue'


class TestTables:
    def test_the_contrasts_table_marks_the_claim(self, tmp_path):
        rendered = write_contrasts_table(
            masking_contrasts(outcome_table()), tmp_path / 'c.tex', 9, 'tab:c', 'c.csv'
        ).read_text(encoding='utf-8')
        lines = rendered.splitlines()
        pooled = next(line for line in lines if 'All pipelines' in line)
        assert pooled.rstrip().endswith(r'\cmark \\') and r'\textbf' in pooled and '(9/9)' in pooled
        sift = next(line for line in lines if line.strip().startswith('SIFT'))
        assert r'\cmark' not in sift and r'\textbf' in sift
        assert 'secondary' in rendered and '0.0039' in rendered

    def test_an_incomplete_reading_is_marked_not_applicable(self, tmp_path):
        table = outcome_table().assign(cte_gcp=np.nan, iou_gcp=np.nan, oce_gcp=np.nan)
        rendered = write_contrasts_table(masking_contrasts(table), tmp_path / 'c.tex', 9, 'tab:c', 'c.csv').read_text(
            encoding='utf-8'
        )
        pooled = next(line for line in rendered.splitlines() if 'All pipelines' in line)
        assert pooled.rstrip().endswith(r'n/a \\')

    def test_a_harm_is_underlined(self, tmp_path):
        table = outcome_table(improved=('zncc',), harmed=('cte_gcp',))
        rendered = write_contrasts_table(masking_contrasts(table), tmp_path / 'c.tex', 9, 'tab:c', 'c.csv').read_text(
            encoding='utf-8'
        )
        pooled = next(line for line in rendered.splitlines() if 'All pipelines' in line)
        assert r'\underline' in pooled and not pooled.rstrip().endswith(r'\cmark \\')

    def test_the_mechanism_table_renders_every_contrast(self, tmp_path):
        rendered = write_mechanism_table(mechanism_contrasts(pair_frame()), tmp_path / 'm.tex', 9, 'tab:m').read_text(
            encoding='utf-8'
        )
        # Five contrasts over all pipelines and five for the one pipeline.
        assert rendered.count(r'\textbf') == 10
        assert rendered.index('All pipelines') < rendered.index('SIFT')


BINS = ((1, 15), (16, 120), (121, 3600))
CELLS = ['high moving / coherent'] * 2 + ['high moving / mixed'] * 2 + ['low moving / coherent'] * 2
CELLS += ['low moving / mixed'] * 2 + ['stationary control']


def strata_frame():
    """
    Per-pair results in which oriented masking gains 0.01 ZNCC on `sift` at offsets up to 120 frames,
    on moving pairs only, and nothing beyond; `orb` gains nothing anywhere.
    """
    rows = []
    for case, cell in zip(CASES, CELLS, strict=True):
        for pair, offset in enumerate((5, 10, 60, 100, 500, 2000)):
            persistence = 0.1 if pair % 2 else 0.9
            for preset in ('sift', 'orb'):
                for masking in ('none', 'obb'):
                    gain = 0.01 if (preset, masking) == ('sift', 'obb') and offset <= 120 and persistence < 0.5 else 0.0
                    rows.append(
                        {
                            'case': case,
                            'cell': cell,
                            'pipeline': f"{preset}+{masking}",
                            'masking': masking,
                            'pair': f"0->{offset}",
                            'abs_offset': offset,
                            'persistence': persistence,
                            'zncc': 0.9 + gain,
                            'gradient': 0.8,
                            'cte_mean': 1.0 if masking == 'obb' else 2.0,
                        }
                    )
    return pd.DataFrame(rows)


class TestBenefitStrata:
    def test_the_benefit_of_the_average_pipeline_is_read_per_offset_bin(self):
        strata = benefit_strata(strata_frame(), BINS).set_index('stratum')
        # Half the pairs of the first two bins move, and orb halves sift's gain again.
        assert strata.loc['1-15', 'zncc'] == pytest.approx(0.01 / 4)
        assert strata.loc['1-15', 'zncc_positive'] == 9 and strata.loc['121-3600', 'zncc'] == pytest.approx(0.0)
        assert strata.loc['121-3600', 'zncc_positive'] == 0
        assert strata.loc['1-15', 'cte_none'] == pytest.approx(2.0) and strata.loc['1-15', 'cte_obb'] == 1.0

    def test_cells_hold_their_cases_whole_and_persistence_splits_the_pairs(self):
        strata = benefit_strata(strata_frame(), BINS)
        cells = strata[strata['heading'] == 'Survey cell'].set_index('stratum')['n_cases']
        assert cells.to_dict() == {
            'high moving / coherent': 2,
            'high moving / mixed': 2,
            'low moving / coherent': 2,
            'low moving / mixed': 2,
            'stationary control': 1,
        }
        persistence = strata[strata['heading'] == 'Persistence'].set_index('stratum')['zncc']
        assert persistence['moving'] > 0 and persistence['stationary'] == pytest.approx(0.0)

    def test_a_pipeline_missing_from_a_case_is_left_out(self):
        frame = strata_frame()
        frame = frame[~((frame['pipeline'].str.startswith('orb')) & (frame['case'] == 'case0'))]
        strata = benefit_strata(frame, BINS).set_index('stratum')
        assert strata.loc['1-15', 'zncc'] == pytest.approx(0.01 / 2)

    def test_the_table_renders_a_block_per_kind_of_stratum(self, tmp_path):
        strata = benefit_strata(strata_frame(), BINS)
        rendered = write_benefit_strata_table(strata, tmp_path / 'b.tex', 9, 'tab:b', 'masking').read_text(
            encoding='utf-8'
        )
        for heading in ('Offset bin [frames]', 'Survey cell', 'Persistence'):
            assert f"\\textit{{{heading}}}" in rendered
        assert '1--15 & 9' in rendered and '(9/9)' in rendered and r'\Cref{tab:masking-contrasts}' in rendered
