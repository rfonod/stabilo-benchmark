#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the mask-margin sweep: masking benefit against suppressed area."""

from __future__ import annotations

import pandas as pd
import pytest

from stabilo_benchmark.report.margin import (
    figure_margin_benefit,
    is_margin_sweep,
    masking_campaign_of,
    per_case_benefits,
    pooled_benefits,
    write_margin_table,
)

CASES = [f"case{i}" for i in range(4)]
MARGINS = (0.0, 0.15, 0.3)


def sweep(geometry_gain=0.0):
    """
    A margin sweep in which the benefit grows with the suppressed area alone, plus `geometry_gain`
    for oriented boxes; HBB suppresses more area than OBB at every margin.
    """
    masked, unmasked = [], []
    for case in CASES:
        for pair in range(5):
            key = f"000000->{pair + 1:06d}"
            unmasked.append(
                {'case': case, 'pipeline': 'sift+none', 'masking': 'none', 'pair': key, 'zncc': 0.9, 'gradient': 0.8}
            )
            for run, (masking, margin) in enumerate((m, r) for m in ('hbb', 'obb') for r in MARGINS):
                area = (0.04 if masking == 'hbb' else 0.03) * (1 + margin) ** 2
                gain = 0.1 * area + (geometry_gain if masking == 'obb' else 0.0)
                masked.append(
                    {
                        'case': case,
                        'pipeline': f"sift+{masking}" + (f"#{run % 3 + 1}" if run % 3 else ''),
                        'masking': masking,
                        'mask_margin_ratio': margin,
                        'pair': key,
                        'suppressed_ref': area,
                        'suppressed_cur': area,
                        'zncc': 0.9 + gain,
                        'gradient': 0.8 + gain,
                    }
                )
    return pd.DataFrame(masked), pd.DataFrame(unmasked)


class TestMarginBenefit:
    def test_detects_a_margin_sweep(self):
        masked, _ = sweep()
        assert is_margin_sweep(masked)
        assert not is_margin_sweep(masked[masked['mask_margin_ratio'] == 0.0])

    def test_the_benefit_is_paired_against_the_unmasked_score_of_the_same_pair(self):
        masked, unmasked = sweep()
        per_case = per_case_benefits(masked, unmasked)
        row = per_case[(per_case['masking'] == 'obb') & (per_case['mask_margin_ratio'] == 0.0)].iloc[0]
        assert row['suppressed'] == pytest.approx(0.03)
        assert row['benefit_zncc'] == pytest.approx(0.1 * 0.03)
        assert row['n_pairs'] == 5

    def test_area_alone_puts_both_geometries_on_one_curve(self):
        # Benefit is a function of area only, so benefit / area is the same for both geometries.
        pooled = pooled_benefits(per_case_benefits(*sweep()))
        ratios = pooled['benefit_zncc'] / pooled['suppressed']
        assert ratios.to_numpy() == pytest.approx(0.1)

    def test_a_geometry_advantage_lifts_the_oriented_curve(self):
        pooled = pooled_benefits(per_case_benefits(*sweep(geometry_gain=0.002))).set_index(
            ['masking', 'mask_margin_ratio']
        )
        assert pooled.loc[('obb', 0.3), 'benefit_gradient'] > 0.1 * pooled.loc[('obb', 0.3), 'suppressed']

    def test_a_pair_missing_from_the_masking_campaign_is_not_scored(self):
        masked, unmasked = sweep()
        per_case = per_case_benefits(masked, unmasked[unmasked['pair'] != '000000->000001'])
        assert (per_case['n_pairs'] == 4).all()
        assert per_case.attrs['unmatched'] == len(CASES) * 2 * len(MARGINS)


class TestOutputs:
    def test_table_and_figure(self, tmp_path):
        pooled = pooled_benefits(per_case_benefits(*sweep()))
        rendered = write_margin_table(pooled, tmp_path / 't.tex', 4, 1, 'tab:m', 'm.csv').read_text(encoding='utf-8')
        assert rendered.count(r'\\') >= 7 and 'OBB & 0.30' in rendered
        assert figure_margin_benefit(pooled, tmp_path / 'f.pdf').exists()


def test_a_margin_sweep_reads_the_masking_campaign_of_its_regime(tmp_path):
    assert masking_campaign_of(tmp_path / 'results' / 'offline_margin') == tmp_path / 'results' / 'offline_masking'
