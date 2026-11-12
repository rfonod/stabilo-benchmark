#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Aggregation, LaTeX rendering, and figure generation from synthetic result directories."""

from __future__ import annotations

import itertools
import json
import re
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from stabilo_benchmark.config import build_pipelines, load_config
from stabilo_benchmark.gcp import summarize_residuals
from stabilo_benchmark.pipelines import load_presets, ransac_name
from stabilo_benchmark.report.build import (
    STATIC_OMIT,
    _bold_note,
    build_report,
    default_prefix,
    evaluation_configs,
    gcp_candidate_rows,
    gcp_table_columns,
    pool_gcp_rows,
    read_epsilon_floor,
    read_kind,
    reference_note,
    regime_label,
    starting_values_note,
    static_tuning_caption,
    tag_regime,
    threshold_units,
)
from stabilo_benchmark.report.figures import (
    PALETTE,
    SCATTER_SLOTS,
    TUNING_GAP_EDGES,
    _estimator_family,
    _grid_position,
    _style_for,
    _tuning_classes,
    figure_accuracy_vs_compute,
    figure_error_vs_offset,
    figure_inlier_vs_quality,
    figure_masking_comparison,
    figure_motion_alignment,
    figure_regime_summary,
    figure_tuning_surface,
    within_group_spearman,
)
from stabilo_benchmark.report.names import PIPELINE_NAMES, display_label
from stabilo_benchmark.report.scenes import (
    SCENE_STATISTICS,
    characterize_scenes,
    summarize_scenes,
)
from stabilo_benchmark.report.tables import (
    DYNAMIC_COLUMNS,
    ESTIMATOR_COLUMNS,
    GCP_COLUMNS,
    LONG_TABLE_ROWS,
    OMIT_NOTES,
    STATIC_COLUMNS,
    TUNING_COLUMNS,
    TUNING_MARKS,
    aggregate_budget,
    aggregate_by_offset,
    aggregate_by_scenario,
    aggregate_cases,
    aggregate_dynamic,
    aggregate_estimators,
    aggregate_gcp_audit,
    aggregate_gcp_proposals,
    aggregate_static,
    aggregate_sweep,
    aggregate_tuning,
    average_pipeline_rows,
    case_name,
    case_order,
    columns_for_object_boxes,
    comparisons_frame,
    estimator_overview,
    gcp_audit_columns,
    infeasible_rows,
    inherited_tuned_values,
    is_budget_campaign,
    is_estimator_campaign,
    is_tuning_campaign,
    load_case_results,
    load_results,
    object_boxes_of,
    preset_tuned_values,
    pretuning_defaults,
    sweep_key,
    sweep_value,
    swept_run_labels,
    tie_sets_for,
    tuning_edges,
    write_comparisons_table,
    write_latex_table,
    write_tuned_mappings,
)
from stabilo_benchmark.results import GCP_SCORES_NAME, RUN_INFEASIBLE, ResultStore, build_manifest
from stabilo_benchmark.runners.audit import summarize_by_offset
from stabilo_benchmark.stats import tie_set

RNG = np.random.default_rng(0)


def synthetic_frame(n: int = 240) -> pd.DataFrame:
    """A frame shaped like real dynamic results, with a masking effect built in."""
    rows = []
    for masking, quality in (('none', 6.0), ('hbb', 2.0), ('obb', 1.2)):
        for index in range(n // 3):
            rows.append(
                {
                    'run_number': 1,
                    'pipeline': f"sift+{masking}",
                    'masking': masking,
                    'detector_name': 'sift',
                    'matcher_name': 'bf',
                    'abs_offset': int(50 * (index % 20 + 1)),
                    'cte_mean': float(abs(RNG.normal(quality, quality / 3))),
                    'ssim': float(np.clip(RNG.normal(1.0 - quality / 20, 0.02), 0, 1)),
                    'zncc': float(np.clip(RNG.normal(1.0 - quality / 25, 0.02), 0, 1)),
                    'mi': float(abs(RNG.normal(3.0 - quality / 10, 0.1))),
                    'gradient': float(np.clip(RNG.normal(1.0 - quality / 15, 0.02), 0, 1)),
                    'iou_mean': float(np.clip(RNG.normal(1.0 - quality / 30, 0.02), 0, 1)),
                    'oce_mean': float(abs(RNG.normal(quality / 10, 0.05))),
                    'inlier_ratio': float(np.clip(RNG.normal(0.55, 0.1), 0, 1)),
                    'motion_alignment': float(np.clip(RNG.normal(0.5 + quality / 25, 0.08), 0, 1)),
                    'time_stabilize_s': float(abs(RNG.normal(0.2, 0.02))),
                    'region_area_fraction': 0.80 + 0.05 * (index % 3),
                    'object_boxes': 'obb',
                    'failed': False,
                }
            )
    return pd.DataFrame(rows)


def write_results(tmp_path, kind='dynamic', frame=None, sampling=None):
    """Materialize a synthetic frame as a result directory the aggregators can read."""
    frame = frame if frame is not None else synthetic_frame()
    results_dir = tmp_path / 'results'
    config = {'kind': kind, 'name': 'synthetic', 'seed': 0}
    if sampling is not None:
        config['sampling'] = sampling
    store = ResultStore(results_dir)
    store.initialize(build_manifest(config, 'hash', {1: {'pipeline': 'x'}}))

    for number, (pipeline, group) in enumerate(frame.groupby('pipeline', sort=False), start=1):
        params = {
            'pipeline': pipeline,
            'masking': group['masking'].iloc[0],
            'detector_name': group['detector_name'].iloc[0],
            'matcher_name': group['matcher_name'].iloc[0],
        }
        records = group.drop(columns=['pipeline', 'masking', 'detector_name', 'matcher_name', 'run_number'])
        store.write_run(number, {'run_number': number, 'params': params, 'records': records.to_dict('records')})
    store.finalize(3, 3)
    return results_dir


# Shipped values for the fixture's presets, independent of cfg/pipelines.yaml; on the grid, not the winner.
FIXTURE_DEFAULTS = {
    'sift': {'filter_ratio': 0.9, 'ransac_epipolar_threshold': 3.0},
    'orb': {'filter_ratio': 0.9, 'ransac_epipolar_threshold': 2.0},
}


def write_tuning_results(
    tmp_path,
    best=('sift', 0.8, 2.0),
    kind='static',
    name='tuning',
    n_records=20,
    ratios=(0.8, 0.9),
    thresholds=(2.0, 3.0),
):
    """
    A campaign shaped like `configs/songdo/static/tuning.yaml`: one run per grid point.

    `best` names the (pipeline, filter_ratio, ransac_epipolar_threshold) that is given the
    lowest errors, so a test can assert the selection found it rather than merely produced
    a row. A dynamic one is written as a case's `results/<name>`.
    """
    results_dir = tmp_path / name if kind == 'static' else tmp_path / 'results' / name
    store = ResultStore(results_dir)
    store.initialize(build_manifest({'kind': kind, 'name': name, 'seed': 0}, 'hash', {1: {'pipeline': 'x'}}))

    number = 0
    for pipeline in ('sift', 'orb'):
        for repeat, (ratio, threshold) in enumerate(itertools.product(ratios, thresholds)):
            number += 1
            winner = (pipeline, ratio, threshold) == tuple(best)
            error = 0.2 if winner else 3.0
            store.write_run(
                number,
                {
                    'run_number': number,
                    'params': {
                        # Run labels as `config.build_pipelines` writes them: `sift+none`, `sift+none#2`, ...
                        'pipeline': f"{pipeline}+none" + (f"#{repeat + 1}" if repeat else ''),
                        'masking': 'none',
                        'detector_name': pipeline,
                        'matcher_name': 'bf',
                        'filter_ratio': ratio,
                        'ransac_epipolar_threshold': threshold,
                        'ransac_method': 38,
                    },
                    'records': [
                        {'cte_mean': error, 'object_boxes': 'obb', 'failed': False, 'time_stabilize_s': 0.2}
                        for _ in range(n_records)
                    ],
                },
            )
    store.finalize(number, number)
    return results_dir


def write_estimator_results(tmp_path, methods=(4, 8, 32, 33, 35, 36, 38)):
    """
    A static campaign shaped like `configs/songdo/static/estimators.yaml`: one run per estimator.

    Pipeline names repeat as `build_pipelines` makes them (`orb+obb`, `orb+obb#2`, ...).
    """
    results_dir = tmp_path / 'estimators'
    store = ResultStore(results_dir)
    store.initialize(
        build_manifest({'kind': 'static', 'name': 'estimators', 'seed': 0}, 'hash', {1: {'pipeline': 'x'}})
    )

    for number, method in enumerate(methods, start=1):
        suffix = '' if number == 1 else f"#{number}"
        store.write_run(
            number,
            {
                'run_number': number,
                'params': {
                    'pipeline': f"orb+obb{suffix}",
                    'masking': 'obb',
                    'detector_name': 'orb',
                    'matcher_name': 'bf',
                    'ransac_method': method,
                    'ransac_epipolar_threshold': 2.0,
                },
                'records': [
                    {
                        'cte_mean': 0.3 + 0.01 * method,
                        'object_boxes': 'obb',
                        'failed': False,
                        'time_stabilize_s': 0.02,
                        'inlier_ratio': 0.47,
                    }
                    for _ in range(20)
                ],
            },
        )
    store.finalize(len(methods), len(methods))
    return results_dir


def write_audit(results_dir, residual_p95_px=0.83):
    """Place an AUDIT.json where `read_epsilon_floor` looks for it: <case>/oracle/."""
    case_dir = results_dir.parent.parent
    path = case_dir / 'oracle' / 'AUDIT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({'reference_estimator': {'summary': {'residual_p95_px': residual_p95_px}}}), encoding='utf-8'
    )
    return path


class TestMaaRanking:
    def test_static_and_dynamic_summaries_carry_maa(self, tmp_path):
        assert 'maa' in aggregate_static(write_results(tmp_path / 'static', kind='static')).columns
        assert 'maa' in aggregate_dynamic(write_results(tmp_path / 'dynamic')).columns

    def test_maa_recovers_the_built_in_masking_order(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path)).set_index('masking')
        assert summary.loc['obb', 'maa'] > summary.loc['hbb', 'maa'] > summary.loc['none', 'maa']

    def test_the_headline_integrates_from_zero_and_the_floor_is_a_sensitivity_column(self, tmp_path):
        # The audited residual is the equivalence margin, not the lower limit of the headline mAA.
        summary = aggregate_dynamic(write_results(tmp_path), epsilon_floor=1.5)
        assert (summary['maa_lower_px'] == 0.0).all()
        assert (summary['maa_floor_px'] == 1.5).all()
        assert 'maa_floor' in summary.columns and DYNAMIC_COLUMNS['maa_floor'][2] is None

    def test_raising_the_floor_merges_orderings_but_never_reverses_them(self, tmp_path):
        # mAA is normalized by the interval width, so a raised lower limit can raise a score; it never
        # reverses the ranking.
        results_dir = write_results(tmp_path)
        summary = aggregate_dynamic(results_dir, epsilon_floor=1.5).set_index('masking')
        floored, unfloored = summary['maa_floor'], summary['maa']
        assert list(floored.sort_values().index) == list(unfloored.sort_values().index)
        assert ((floored >= 0.0) & (floored <= 1.0)).all()

    def test_tables_sort_on_maa_not_on_a_single_threshold(self, tmp_path):
        build_report(write_results(tmp_path), output=tmp_path / 'out')
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        body = [line for line in rendered.splitlines() if line.strip().startswith('SIFT &')]
        assert [line.split('&')[1].strip() for line in body] == ['OBB', 'HBB', 'none']

    def test_the_pipeline_cell_does_not_repeat_the_masking_column(self, tmp_path):
        build_report(write_results(tmp_path), output=tmp_path / 'out')
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        assert 'sift+' not in rendered and 'SIFT & OBB' in rendered


class TestEpsilonFloor:
    def test_reads_the_audited_residual(self, tmp_path):
        results_dir = write_results(tmp_path / 'case' / 'results')
        assert read_epsilon_floor(write_audit(results_dir) and results_dir) == pytest.approx(0.83)

    def test_missing_audit_returns_none(self, tmp_path):
        assert read_epsilon_floor(write_results(tmp_path)) is None

    def test_corrupt_audit_returns_none_rather_than_a_zero_floor(self, tmp_path):
        results_dir = write_results(tmp_path / 'case' / 'results')
        path = write_audit(results_dir)
        path.write_text('{"reference_estimator": {}}', encoding='utf-8')
        assert read_epsilon_floor(results_dir) is None

    def test_build_warns_when_no_audit_backs_the_ranking(self, tmp_path):
        produced = build_report(write_results(tmp_path), output=tmp_path / 'out')
        assert produced['epsilon_floor_px'] is None
        assert any('resolution floor' in reason for reason in produced['skipped'])

    def test_build_uses_the_audited_floor_and_says_so_in_the_caption(self, tmp_path):
        results_dir = write_results(tmp_path / 'case' / 'results')
        write_audit(results_dir)
        produced = build_report(results_dir, output=tmp_path / 'out')
        assert produced['epsilon_floor_px'] == pytest.approx(0.83)
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        assert '0.83' in rendered and 'not resolvable' in rendered and 'equivalence margin' in rendered


class TestTuningTable:
    def test_detects_a_tuning_campaign(self, tmp_path):
        assert is_tuning_campaign(load_results(write_tuning_results(tmp_path)))

    def test_a_varying_ransac_method_is_not_a_tuning_campaign(self, tmp_path):
        # An estimator sweep is not a tuning campaign.
        frame = pd.DataFrame(
            {'filter_ratio': [0.8, 0.9], 'ransac_epipolar_threshold': [2.0, 3.0], 'ransac_method': [8, 38]}
        )
        assert not is_tuning_campaign(frame)

    def test_per_preset_values_are_not_a_tuning_campaign(self):
        # An evaluation campaign over tuned presets: the parameters differ between pipelines only.
        frame = pd.DataFrame(
            {
                'pipeline': ['sift+none', 'sift+obb', 'orb+none', 'orb+obb'],
                'filter_ratio': [0.7, 0.7, 0.85, 0.85],
                'ransac_epipolar_threshold': [0.5, 0.5, 4.0, 4.0],
                'ransac_method': [38] * 4,
            }
        )
        assert not is_tuning_campaign(frame)

    def test_selects_the_grid_point_maximizing_maa(self, tmp_path):
        table = aggregate_tuning(write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS).set_index('pipeline')
        assert table.loc['sift+none', 'filter_ratio'] == pytest.approx(0.8)
        assert table.loc['sift+none', 'ransac_epipolar_threshold'] == pytest.approx(2.0)
        assert table.loc['sift+none', 'n_configurations'] == 4

    def test_one_row_per_preset_across_suffixed_run_labels(self, tmp_path):
        table = aggregate_tuning(write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS)
        assert sorted(table['pipeline']) == ['orb+none', 'sift+none']

    def test_counts_the_grid_points_tied_with_the_winner(self, tmp_path):
        # The fixture's winner has 0.2 px errors and every other point 3.0 px, far outside the margin.
        table = aggregate_tuning(write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS).set_index('pipeline')
        assert table.loc['sift+none', 'n_within_margin'] == 1
        assert table.loc['orb+none', 'n_within_margin'] == 4

    def test_an_incumbent_within_the_margin_is_kept(self, tmp_path):
        # Every orb point scores alike, so orb keeps its incumbent values, edges and all.
        table = aggregate_tuning(
            write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS, incumbent=FIXTURE_DEFAULTS
        ).set_index('pipeline')
        orb = table.loc['orb+none']
        assert (orb['filter_ratio'], orb['ransac_epipolar_threshold']) == pytest.approx((0.9, 2.0))
        assert orb['maa_kept'] and orb['maa_gain'] == pytest.approx(0.0)
        assert not orb['filter_ratio_at_edge'] and not orb['ransac_epipolar_threshold_at_edge']

    def test_an_incumbent_beaten_by_more_than_the_margin_is_replaced(self, tmp_path):
        table = aggregate_tuning(
            write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS, incumbent=FIXTURE_DEFAULTS
        ).set_index('pipeline')
        sift = table.loc['sift+none']
        assert (sift['filter_ratio'], sift['ransac_epipolar_threshold']) == pytest.approx((0.8, 2.0))
        assert not sift['maa_kept']

    def test_an_incumbent_off_the_grid_is_not_kept(self, tmp_path):
        incumbent = {'orb': {'filter_ratio': 0.85, 'ransac_epipolar_threshold': 2.0}}
        table = aggregate_tuning(
            write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS, incumbent=incumbent
        ).set_index('pipeline')
        assert not table.loc['orb+none', 'maa_kept']

    def test_the_ratio_is_not_applicable_to_lightglue(self, tmp_path):
        frame = load_results(write_tuning_results(tmp_path))
        frame.loc[frame['pipeline'].str.startswith('orb'), 'matcher_name'] = 'lightglue'
        (tmp_path / 'frame').mkdir()
        with patch('stabilo_benchmark.report.tables.load_results', return_value=frame):
            table = aggregate_tuning(tmp_path / 'frame', defaults=FIXTURE_DEFAULTS).set_index('pipeline')
        assert np.isnan(table.loc['orb+none', 'filter_ratio'])
        assert table.loc['orb+none', 'n_configurations'] == 2
        assert np.isfinite(table.loc['orb+none', 'maa_default'])
        assert table.loc['sift+none', 'n_configurations'] == 4

    def test_pretuning_defaults_come_from_the_pinned_config_not_the_presets(self):
        defaults = pretuning_defaults()
        assert defaults['sift'] == {'filter_ratio': 0.9, 'ransac_epipolar_threshold': 3.0}
        assert defaults['orb'] == {'filter_ratio': 0.9, 'ransac_epipolar_threshold': 2.0}
        assert len(defaults) == 10

    def test_the_caption_names_the_values_the_default_column_is_scored_at(self):
        def caption(*names):
            return static_tuning_caption(pd.DataFrame({'pipeline': list(names)}))

        assert 'a ratio of 0.9' in caption('sift+none', 'sift+none#2')
        assert 'a ratio of 0.9' not in caption('oracle+obb', 'oracle+obb#2')
        assert "own values" in caption('oracle+obb') and "own values" in caption('sift+none', 'oracle+obb')
        assert '<default_values>' not in caption('sift+none')

    def test_reports_the_gain_over_the_shipped_preset(self, tmp_path):
        table = aggregate_tuning(write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS).set_index('pipeline')
        assert table.loc['sift+none', 'maa_gain'] > 0
        assert table.loc['sift+none', 'maa_default'] < table.loc['sift+none', 'maa']

    def test_gain_is_never_bolded_as_a_best_value(self, tmp_path):
        # The gain column is never bolded.
        assert TUNING_COLUMNS['maa_gain'][2] is None
        rendered = write_latex_table(
            aggregate_tuning(write_tuning_results(tmp_path), defaults=FIXTURE_DEFAULTS),
            tmp_path / 'tuned.tex',
            columns=TUNING_COLUMNS,
            label_columns=('pipeline',),
        ).read_text(encoding='utf-8')
        assert r'\textbf' in rendered  # mAA is highlighted ...
        assert not any(r'\textbf' in line.split('&')[5] for line in rendered.splitlines() if '&' in line)

    def test_flags_a_winner_at_an_end_of_its_swept_range(self, tmp_path):
        grid = {'ratios': (0.7, 0.8, 0.9), 'thresholds': (1.0, 2.0, 3.0)}
        inside = aggregate_tuning(write_tuning_results(tmp_path / 'inside', **grid), defaults=FIXTURE_DEFAULTS)
        inside = inside.set_index('pipeline')
        assert not inside.loc['sift+none', 'filter_ratio_at_edge']
        assert not inside.loc['sift+none', 'ransac_epipolar_threshold_at_edge']

        edge = aggregate_tuning(
            write_tuning_results(tmp_path / 'edge', best=('sift', 0.9, 2.0), **grid), defaults=FIXTURE_DEFAULTS
        ).set_index('pipeline')
        assert edge.loc['sift+none', 'filter_ratio_at_edge']
        assert not edge.loc['sift+none', 'ransac_epipolar_threshold_at_edge']
        assert 'sift+none: filter_ratio=0.9' in tuning_edges(edge.reset_index())

    def test_an_edge_value_carries_a_dagger_and_the_caption_says_why(self, tmp_path):
        tuned = aggregate_tuning(
            write_tuning_results(tmp_path, best=('sift', 0.9, 2.0), ratios=(0.7, 0.8, 0.9), thresholds=(1.0, 2.0, 3.0)),
            defaults=FIXTURE_DEFAULTS,
        )
        rendered = write_latex_table(
            tuned, tmp_path / 'tuned.tex', columns=TUNING_COLUMNS, label_columns=('pipeline',), marks=TUNING_MARKS
        ).read_text(encoding='utf-8')
        sift = next(line for line in rendered.splitlines() if line.strip().startswith('SIFT &'))
        assert sift.split('&')[1].strip() == r'0.90$^{\dagger}$'
        assert r'$^{\dagger}$' not in sift.split('&')[2]
        assert 'outside the grid' in rendered

    def test_a_default_off_the_grid_is_marked_and_reads_its_better_neighbor(self, tmp_path):
        # A default threshold of 2.5 on a 2.0 / 3.0 grid: never presented as the score at 2.5.
        defaults = {name: {'filter_ratio': 0.8, 'ransac_epipolar_threshold': 2.5} for name in ('sift', 'orb')}
        tuned = aggregate_tuning(write_tuning_results(tmp_path), defaults=defaults).set_index('pipeline')
        assert tuned.loc['sift+none', 'maa_default_off_grid']
        assert tuned.loc['sift+none', 'maa_default'] == tuned.loc['sift+none', 'maa']  # the 2.0 neighbor wins
        on_grid = aggregate_tuning(write_tuning_results(tmp_path / 'on'), defaults=FIXTURE_DEFAULTS)
        assert not on_grid['maa_default_off_grid'].any()
        rendered = write_latex_table(
            tuned.reset_index(),
            tmp_path / 't.tex',
            columns=TUNING_COLUMNS,
            label_columns=('pipeline',),
            marks=TUNING_MARKS,
        ).read_text(encoding='utf-8')
        assert r'$^{\ddagger}$' in rendered and 'lower bound' in rendered

    def test_build_warns_about_edge_values(self, tmp_path):
        produced = build_report(
            write_tuning_results(tmp_path, best=('sift', 0.9, 2.0), ratios=(0.7, 0.8, 0.9), thresholds=(1.0, 2.0, 3.0)),
            kind='static',
            output=tmp_path / 'out',
        )
        assert any('sift+none: filter_ratio=0.9' in warning for warning in produced['warnings'])

    def test_tuned_values_paste_into_the_evaluation_config(self, tmp_path):
        # Every preset of offline_masking.yaml gets its own selected pair; LightGlue and LoFTR keep their ratio.
        config = load_config(
            Path(__file__).resolve().parent.parent / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml'
        )
        presets = config['pipeline']['preset']
        tuned = pd.DataFrame(
            {
                'pipeline': [f"{name}+none" for name in presets],
                'filter_ratio': [np.nan if name in ('disk_lightglue', 'loftr') else 0.8 for name in presets],
                'ransac_epipolar_threshold': [1.0 + i for i in range(len(presets))],
            }
        )
        path = write_tuned_mappings(tuned, tmp_path / 'tuned.yaml', 'header')
        config['pipeline'].update(yaml.safe_load(path.read_text(encoding='utf-8')))
        runs = {spec.name: spec.params for spec in build_pipelines(config) if spec.masking == 'obb'}
        assert len(runs) == len(presets)
        for i, name in enumerate(presets):
            params = runs[f"{name}+obb"]
            assert params['ransac_epipolar_threshold'] == 1.0 + i
        assert runs['sift+obb']['filter_ratio'] == 0.8
        assert runs['disk_lightglue+obb']['filter_ratio'] == load_presets()['disk_lightglue']['filter_ratio']

    def test_build_writes_the_tuned_parameter_table(self, tmp_path):
        produced = build_report(write_tuning_results(tmp_path), kind='static', output=tmp_path / 'out')
        assert (tmp_path / 'out' / 'tables' / 'tuned_parameters.tex').exists()
        assert any('tuned_parameters' in table for table in produced['tables'])

    def test_another_tuning_campaign_does_not_overwrite_part_one_table(self, tmp_path):
        produced = build_report(write_tuning_results(tmp_path, name='sweep'), kind='static', output=tmp_path / 'out')
        tables = tmp_path / 'out' / 'tables'
        assert not (tables / 'tuned_parameters.tex').exists()
        rendered = (tables / 'sweep_tuned_parameters.tex').read_text(encoding='utf-8')
        assert r'\label{tab:sweep-tuned-parameters}' in rendered
        assert any('sweep_tuned_parameters' in table for table in produced['tables'])

    def test_a_preset_without_pinned_defaults_is_compared_at_its_parents_values(self, tmp_path):
        frame = load_results(write_tuning_results(tmp_path))
        frame['pipeline'] = frame['pipeline'].str.replace('sift', 'oracle')
        parents = {'oracle': {'filter_ratio': 0.9, 'ransac_epipolar_threshold': 3.0}}
        with patch('stabilo_benchmark.report.tables.inherited_tuned_values', return_value=parents):
            table = aggregate_tuning(frame=frame, defaults={}).set_index('pipeline')
        assert np.isfinite(table.loc['oracle+none', 'maa_default'])
        assert np.isnan(table.loc['orb+none', 'maa_default'])

    def test_the_reference_estimator_inherits_from_sift(self):
        assert inherited_tuned_values()['oracle'] == preset_tuned_values()['sift']

    def test_cases_weigh_equally_whatever_their_pair_counts(self, tmp_path):
        small = write_tuning_results(tmp_path / 'case_a', kind='dynamic', n_records=5)
        large = write_tuning_results(tmp_path / 'case_b', best=('sift', 0.9, 3.0), kind='dynamic', n_records=45)
        frame = load_case_results([small, large])
        table = aggregate_tuning(frame=frame, defaults=FIXTURE_DEFAULTS).set_index('pipeline')
        # Per case, each winner scores 0.96 at its own point and 0.40 at the other's; pooled by pair,
        # case_b's 45 pairs would pick its own point.
        assert table.loc['sift+none', 'maa'] == pytest.approx((0.96 + 0.40) / 2)
        assert table.loc['sift+none', 'n_cases'] == 2

    def test_the_pooled_dynamic_report_writes_part_two_tuning(self, tmp_path):
        directories = [write_tuning_results(tmp_path / f"case_{i}", kind='dynamic', n_records=5) for i in range(3)]
        produced = build_report(directories, output=tmp_path / 'out', prefix='offline_tuning')
        rendered = (tmp_path / 'out' / 'tables' / 'offline_tuning_tuned_parameters.tex').read_text(encoding='utf-8')
        assert '3 tuning cases' in rendered and 'Part I values' in rendered
        assert not (tmp_path / 'out' / 'tables' / 'tuned_parameters.tex').exists()
        assert any('offline_tuning_tuned_parameters' in table for table in produced['tables'])

    def test_a_plain_static_campaign_skips_it_with_a_reason(self, tmp_path):
        frame = synthetic_frame().drop(columns=['abs_offset', 'motion_alignment'])
        produced = build_report(write_results(tmp_path, kind='static', frame=frame), output=tmp_path / 'out')
        assert any('tuning table' in reason for reason in produced['skipped'])
        assert not (tmp_path / 'out' / 'tables' / 'tuned_parameters.tex').exists()


class TestAggregation:
    def test_dynamic_summary_has_one_row_per_pipeline(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        assert len(summary) == 3
        assert set(summary['masking']) == {'none', 'hbb', 'obb'}

    def test_medians_and_tails_are_both_reported(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        assert (summary['cte_p90'] >= summary['cte_median']).all()

    def test_the_built_in_masking_effect_is_recovered(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path)).set_index('masking')
        assert summary.loc['obb', 'cte_median'] < summary.loc['hbb', 'cte_median']
        assert summary.loc['hbb', 'cte_median'] < summary.loc['none', 'cte_median']
        assert summary.loc['obb', 'ssim'] > summary.loc['none', 'ssim']

    def test_the_median_region_fraction_is_reported(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        assert summary['region_fraction'].to_numpy() == pytest.approx(0.85)

    def test_static_summary_carries_hea_columns(self, tmp_path):
        summary = aggregate_static(write_results(tmp_path, kind='static'))
        assert 'hea_1px' in summary.columns
        assert 'failure_rate' in summary.columns

    def test_by_offset_groups_by_temporal_distance(self, tmp_path):
        summary = aggregate_by_offset(write_results(tmp_path), bins=[(1, 200), (201, 600), (601, None)])
        assert set(summary['offset_bin']) == {'1-200', '201-600', '>600'}

    def test_by_offset_reads_the_campaign_s_sampling_bins(self, tmp_path):
        sampling = {'mode': 'stratified', 'bins': [[1, 100], [101, 400], [401, 1000]]}
        summary = aggregate_by_offset(write_results(tmp_path, sampling=sampling))
        # The fixture's offsets run 50..1000; every bin the plan declares appears, in plan order.
        assert list(dict.fromkeys(summary['offset_bin'])) == ['1-100', '101-400', '401-1000']

    def test_by_offset_without_bins_falls_back_to_the_defaults(self, tmp_path):
        summary = aggregate_by_offset(write_results(tmp_path))
        assert set(summary['offset_bin']) <= {'1-100', '101-500', '501-1000', '1001-5000', '>5000'}

    def test_by_offset_needs_offsets(self, tmp_path):
        frame = synthetic_frame().drop(columns=['abs_offset'])
        with pytest.raises(ValueError, match='abs_offset'):
            aggregate_by_offset(write_results(tmp_path, frame=frame))

    def test_empty_directory_raises(self, tmp_path):
        results_dir = tmp_path / 'empty'
        ResultStore(results_dir).initialize(build_manifest({'kind': 'dynamic', 'name': 'e', 'seed': 0}, 'h', {}))
        with pytest.raises(ValueError, match='No measurements'):
            aggregate_dynamic(results_dir)


class TestLatex:
    def test_writes_a_booktabs_table(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        path = write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS, caption='C', label='tab:x')
        text = path.read_text(encoding='utf-8')
        assert r'\toprule' in text and r'\bottomrule' in text
        assert r'\caption{C}' in text and r'\label{tab:x}' in text
        assert 'Generated by stabilo-benchmark' in text

    def test_bolds_the_best_value_per_column(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        assert r'\textbf{' in write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS).read_text(
            encoding='utf-8'
        )

    def test_diagnostic_column_is_never_marked_best(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        columns = {'inlier_ratio': DYNAMIC_COLUMNS['inlier_ratio']}
        assert r'\textbf{' not in write_latex_table(summary, tmp_path / 't.tex', columns=columns).read_text(
            encoding='utf-8'
        )

    def test_region_fraction_is_never_bolded_and_escapes_the_percent_sign(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        columns = {'region_fraction': DYNAMIC_COLUMNS['region_fraction']}
        text = write_latex_table(summary, tmp_path / 't.tex', columns=columns).read_text(encoding='utf-8')
        assert r'\textbf{' not in text
        assert r'85.0\%' in text

    def test_underscores_in_names_are_escaped(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        summary.loc[0, 'pipeline'] = 'sift_bf+obb'
        assert r'sift\_bf' in write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS).read_text(
            encoding='utf-8'
        )

    def test_nan_renders_as_a_dash(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        summary.loc[0, 'ssim'] = np.nan
        assert '--' in write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS).read_text(
            encoding='utf-8'
        )

    def test_static_columns_render(self, tmp_path):
        summary = aggregate_static(write_results(tmp_path, kind='static'))
        assert write_latex_table(summary, tmp_path / 's.tex', columns=STATIC_COLUMNS).exists()

    def test_csv_holds_the_unformatted_frame_in_table_order(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS, sort_by='maa', csv=True)
        written = pd.read_csv(tmp_path / 't.csv')
        assert list(written.columns) == list(summary.columns)
        assert list(written['pipeline']) == list(summary.sort_values('maa', ascending=False)['pipeline'])
        assert not (tmp_path / 'u.csv').exists()
        write_latex_table(summary, tmp_path / 'u.tex', columns=DYNAMIC_COLUMNS)
        assert not (tmp_path / 'u.csv').exists()

    def test_no_matching_columns_raises(self, tmp_path):
        with pytest.raises(ValueError, match='None of the requested columns'):
            write_latex_table(pd.DataFrame({'a': [1]}), tmp_path / 't.tex', columns=DYNAMIC_COLUMNS)

    def test_a_long_table_is_set_smaller_to_fit_one_page(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        short = write_latex_table(summary, tmp_path / 's.tex', columns=DYNAMIC_COLUMNS).read_text(encoding='utf-8')
        assert r'\small' in short and r'\footnotesize' not in short
        rows = pd.concat([summary] * (LONG_TABLE_ROWS // len(summary) + 1), ignore_index=True)
        long = write_latex_table(rows, tmp_path / 'l.tex', columns=DYNAMIC_COLUMNS).read_text(encoding='utf-8')
        assert r'\footnotesize' in long and r'\small' not in long
        assert r'\begin{adjustbox}' in long

    def test_label_cells_use_display_names(self, tmp_path):
        summary = aggregate_dynamic(write_results(tmp_path))
        summary.loc[0, ['pipeline', 'masking']] = ['rsift+obb', 'obb']
        summary.loc[1, ['pipeline', 'masking']] = ['disk_lightglue+hbb', 'hbb']
        text = write_latex_table(summary, tmp_path / 't.tex', columns=DYNAMIC_COLUMNS).read_text(encoding='utf-8')
        assert 'RootSIFT & OBB' in text and 'DISK+LightGlue & HBB' in text
        assert 'rsift' not in text and 'disk\\_lightglue' not in text


class TestDisplayNames:
    @pytest.mark.parametrize(
        ('label', 'shown'),
        [
            ('rsift', 'RootSIFT'),
            ('rsift+obb', 'RootSIFT / OBB'),
            ('keynet_lightglue+none#3', 'KeyNet+LightGlue / none#3'),
            ('sift+obb (margin 0.15)', 'SIFT / OBB (margin 0.15)'),
            ('All pipelines', 'All pipelines'),
            ('custom_preset+obb', 'custom_preset+obb'),
        ],
    )
    def test_run_labels(self, label, shown):
        assert display_label(label) == shown

    def test_every_shipped_preset_has_a_display_name(self):
        presets = set(load_presets()) - {'base'}
        assert presets <= set(PIPELINE_NAMES)


class TestPalette:
    def test_scatter_is_limited_to_the_all_pairs_safe_subset(self):
        # Scatter plots use at most three palette slots.
        _style_for(SCATTER_SLOTS - 1, scatter=True)
        with pytest.raises(ValueError, match='validated palette slot'):
            _style_for(SCATTER_SLOTS, scatter=True)

    def test_bars_and_lines_may_use_the_full_order(self):
        for index in range(len(PALETTE)):
            _style_for(index)
        with pytest.raises(ValueError, match='validated palette slot'):
            _style_for(len(PALETTE))

    def test_every_series_gets_a_distinct_marker_and_linestyle(self):
        # Every series has a distinct marker or line style.
        styles = [_style_for(index) for index in range(len(PALETTE))]
        assert len({style['marker'] for style in styles}) == len(PALETTE)
        assert len({str(style['linestyle']) for style in styles}) == len(PALETTE)


class TestFigures:
    def test_inlier_vs_quality_renders(self, tmp_path):
        assert figure_inlier_vs_quality(synthetic_frame(), tmp_path / 'a.pdf').exists()

    def test_inlier_vs_quality_refuses_a_frame_with_nothing_to_draw(self, tmp_path):
        frame = synthetic_frame().assign(inlier_ratio=np.nan)
        with pytest.raises(ValueError):
            figure_inlier_vs_quality(frame, tmp_path / 'a.pdf')

    def test_the_regime_summary_draws_a_perfect_score_on_its_log_axis(self, tmp_path):
        summary = pd.DataFrame(
            [
                {'pipeline': 'sift+none', 'masking': 'none', 'maa': 1.0},
                {'pipeline': 'sift+obb', 'masking': 'obb', 'maa': 0.97},
                {'pipeline': 'identity', 'masking': 'none', 'maa': 0.3},
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert figure_regime_summary(summary, tmp_path / 'a.pdf').exists()

    def test_the_inlier_correlation_is_formed_within_pipeline_and_case(self):
        # Within each case the inlier ratio falls as the error rises; across cases both rise together,
        # so a coefficient over the pooled cloud would be positive.
        rows = [
            {'pipeline': 'sift+obb', 'case': case, 'inlier_ratio': offset + 0.1 * i, 'cte_mean': offset - 0.1 * i}
            for case, offset in (('a', 0.0), ('b', 5.0))
            for i in range(5)
        ]
        coefficients = within_group_spearman(pd.DataFrame(rows), 'cte_mean')
        assert coefficients.tolist() == pytest.approx([-1.0, -1.0])

    def test_masking_comparison_renders(self, tmp_path):
        assert figure_masking_comparison(synthetic_frame(), tmp_path / 'b.pdf').exists()

    def test_error_vs_offset_renders(self, tmp_path):
        assert figure_error_vs_offset(synthetic_frame(), tmp_path / 'c.pdf').exists()

    def test_error_vs_offset_reads_the_campaign_bins(self, tmp_path):
        bins = [(1, 15), (16, 30), (31, 60), (61, 120), (121, 240), (241, None)]
        assert figure_error_vs_offset(synthetic_frame(), tmp_path / 'd.pdf', bins=bins).exists()

    def test_motion_alignment_renders(self, tmp_path):
        assert figure_motion_alignment(synthetic_frame(), tmp_path / 'd.pdf').exists()

    def test_missing_metric_raises_a_clear_error(self, tmp_path):
        frame = synthetic_frame().drop(columns=['motion_alignment'])
        with pytest.raises(KeyError):
            figure_motion_alignment(frame, tmp_path / 'e.pdf')

    def test_all_nan_metric_raises_rather_than_drawing_an_empty_axis(self, tmp_path):
        frame = synthetic_frame().assign(motion_alignment=np.nan)
        with pytest.raises(ValueError, match='No finite'):
            figure_motion_alignment(frame, tmp_path / 'f.pdf')


class TestBuildReport:
    def test_kind_is_read_from_the_manifest(self, tmp_path):
        assert read_kind(write_results(tmp_path)) == 'dynamic'

    def test_dynamic_report_writes_tables_and_figures(self, tmp_path):
        produced = build_report(write_results(tmp_path), output=tmp_path / 'out')
        assert produced['kind'] == 'dynamic'
        assert produced['tables'] and produced['figures']
        assert (tmp_path / 'out' / 'tables').is_dir()
        assert (tmp_path / 'out' / 'figures').is_dir()

    def test_static_report_skips_inapplicable_figures_with_a_reason(self, tmp_path):
        frame = synthetic_frame().drop(columns=['abs_offset', 'motion_alignment'])
        produced = build_report(write_results(tmp_path, kind='static', frame=frame), output=tmp_path / 'out')
        assert produced['kind'] == 'static'
        assert produced['tables']
        assert list((tmp_path / 'out' / 'tables').glob('*_static.csv'))

    def test_static_masking_figure_plots_the_object_metrics(self, tmp_path):
        # Static results have no photometric scores; the masking figure uses the object metrics.
        photometric = ['ssim', 'zncc', 'gradient', 'mi']
        frame = synthetic_frame().drop(columns=['abs_offset', 'motion_alignment', *photometric], errors='ignore')
        produced = build_report(write_results(tmp_path, kind='static', frame=frame), output=tmp_path / 'out')
        assert any(path.endswith('_masking.pdf') for path in produced['figures']), produced['skipped']

    def test_unknown_kind_raises(self, tmp_path):
        results_dir = write_results(tmp_path)
        (results_dir / 'MANIFEST.json').unlink()
        with pytest.raises(ValueError, match='Could not determine'):
            build_report(results_dir, output=tmp_path / 'out')


class TestTuningFigure:
    """The tuning surfaces: every grid point classed by its gap to the pipeline's best."""

    def test_the_tuning_report_writes_the_surface_beside_the_table(self, tmp_path):
        produced = build_report(write_tuning_results(tmp_path), output=tmp_path / 'out')
        assert any(path.endswith('tuning_surface.pdf') for path in produced['figures']), produced['skipped']

    def test_a_gap_equal_to_the_margin_is_tied(self):
        classes = _tuning_classes(np.array([0.0, 0.01, 0.0101, 0.03, 0.2]), TUNING_GAP_EDGES)
        assert classes.tolist() == [0, 0, 1, 2, 4]

    def test_a_pipeline_without_a_ratio_test_is_drawn_as_one_row(self, tmp_path):
        grid = pd.DataFrame(
            {
                'pipeline': ['sift+none'] * 4 + ['disk_lightglue+none'] * 2,
                'filter_ratio': [0.8, 0.8, 0.9, 0.9, np.nan, np.nan],
                'ransac_epipolar_threshold': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                'maa': [0.95, 0.96, 0.90, 0.97, 0.80, 0.81],
            }
        )
        selected = pd.DataFrame({'pipeline': ['sift+none'], 'filter_ratio': [0.9], 'ransac_epipolar_threshold': [2.0]})
        path = figure_tuning_surface(grid, tmp_path / 'surface.pdf', selected=selected, defaults=FIXTURE_DEFAULTS)
        assert path.stat().st_size > 0

    def test_off_grid_values_are_not_marked(self):
        assert _grid_position({'filter_ratio': 0.9, 'ransac_epipolar_threshold': 2.5}, [1.0, 2.0], [0.9]) is None
        assert _grid_position({'ransac_epipolar_threshold': 2.0}, [1.0, 2.0], [np.nan]) == (1.0, 0.0)

    def test_the_estimator_report_writes_the_threshold_figure(self, tmp_path):
        produced = build_report(write_estimator_results(tmp_path), output=tmp_path / 'out')
        assert any(path.endswith('_estimators.pdf') for path in produced['figures']), produced['skipped']

    def test_lo_ransac_configurations_share_one_family(self):
        families = {_estimator_family(ransac_name(code))[0] for code in (32, 33, 35)}
        assert families == {'LO-RANSAC'}
        assert _estimator_family('LMEDS') == ('LMEDS', 'LMEDS')


class TestEstimatorCampaign:
    """The estimator axis reaches the table, by name."""

    def test_detects_a_sweep_over_estimators(self, tmp_path):
        assert is_estimator_campaign(load_results(write_estimator_results(tmp_path)))

    def test_does_not_mistake_a_tuning_sweep_for_one(self, tmp_path):
        # Tuning and estimator campaigns are recognized disjointly.
        frame = load_results(write_tuning_results(tmp_path))
        assert is_tuning_campaign(frame)
        assert not is_estimator_campaign(frame)

    def test_one_row_per_estimator(self, tmp_path):
        summary = aggregate_estimators(write_estimator_results(tmp_path))
        assert len(summary) == 7
        assert summary['ransac_method'].nunique() == 7

    def test_estimators_are_reported_by_name_not_by_opencv_code(self, tmp_path):
        summary = aggregate_estimators(write_estimator_results(tmp_path))
        names = set(summary['estimator'])
        assert 'USAC-MAGSAC (MAGSAC++)' in names
        assert 'USAC-Accurate (GC-RANSAC)' in names
        # Estimators are shown by name, not by code.
        assert not any(name.isdigit() for name in names)

    def test_no_estimator_is_labeled_degensac(self, tmp_path):
        summary = aggregate_estimators(write_estimator_results(tmp_path))
        assert not any('DEGENSAC' in name.upper() for name in summary['estimator'])

    def test_the_rendered_table_names_the_estimators(self, tmp_path):
        path = write_latex_table(
            aggregate_estimators(write_estimator_results(tmp_path)),
            tmp_path / 'estimators.tex',
            columns=ESTIMATOR_COLUMNS,
            label_columns=('detector', 'masking', 'estimator'),
            sort_by='maa',
        )
        text = path.read_text(encoding='utf-8')
        assert 'USAC-MAGSAC (MAGSAC++)' in text
        assert 'GC-RANSAC' in text

    def test_inlier_ratio_is_never_bolded(self, tmp_path):
        # The inlier-ratio column is never bolded.
        assert ESTIMATOR_COLUMNS['inlier_ratio'][2] is None

    def test_the_report_writes_an_estimator_table(self, tmp_path):
        produced = build_report(write_estimator_results(tmp_path), output=tmp_path / 'out')
        assert any('estimators.tex' in table for table in produced['tables'])
        assert any('estimators.csv' in table for table in produced['tables'])

    def test_the_overview_has_a_row_per_estimator_and_a_column_per_threshold(self):
        summary = pd.DataFrame(
            [
                {
                    'detector': 'orb',
                    'masking': 'obb',
                    'estimator': name,
                    'ransac_epipolar_threshold': threshold,
                    'maa': maa,
                    'hea_5px': 0.9 if name == 'RHO' else 1.0,
                    'failure_rate': 0.05 if name == 'RHO' else 0.0,
                    'inlier_ratio': 0.2 * threshold,
                    'n': 100,
                }
                for name, maa in (('RANSAC', 0.9), ('RHO', 0.8))
                for threshold in (0.5, 1.0, 4.0)
            ]
        )
        overview, columns = estimator_overview(summary)
        assert len(overview) == 2 and 'masking' not in overview
        assert list(columns)[:3] == ['maa_at_0.5', 'maa_at_1', 'maa_at_4']
        rho = overview.set_index('estimator').loc['RHO']
        # Over three thresholds: 15 failures, and 30 misses of which 15 did not fail.
        assert (rho['n'], rho['failures'], rho['beyond']) == (300, 15, 15)
        assert rho['inlier_low'] == pytest.approx(0.1) and rho['inlier_high'] == pytest.approx(0.8)
        assert all(spec[2] is None for spec in columns.values())


class TestObjectBoxes:
    def test_obb_headers_are_left_as_they_are(self):
        assert columns_for_object_boxes(DYNAMIC_COLUMNS, 'obb') == DYNAMIC_COLUMNS

    def test_hbb_headers_are_marked(self):
        columns = columns_for_object_boxes(STATIC_COLUMNS, 'hbb')
        assert columns['iou_mean'][0] == r'IoU$_{\mathrm{HBB}}$'
        assert columns['oce_mean'][0] == r'OCE$_{\mathrm{HBB}}$ [px]'
        assert columns['cte_median'] == STATIC_COLUMNS['cte_median']

    def test_results_that_do_not_record_the_geometry_are_refused(self):
        with pytest.raises(ValueError, match='object_boxes'):
            object_boxes_of(synthetic_frame(30).drop(columns='object_boxes'))

    def test_mixed_geometries_are_refused(self):
        frame = synthetic_frame(30)
        frame['object_boxes'] = ['obb', 'hbb'] * 15
        with pytest.raises(ValueError, match='mix'):
            object_boxes_of(frame)

    def test_the_report_marks_an_hbb_campaign(self, tmp_path):
        frame = synthetic_frame()
        frame['object_boxes'] = 'hbb'
        produced = build_report(write_results(tmp_path, frame=frame), output=tmp_path / 'out')
        assert produced['object_boxes'] == 'hbb'
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        assert r'IoU$_{\mathrm{HBB}}$' in rendered


class TestSceneCharacterization:
    """
    The table that describes the data: object counts and covered area per scene, for both box
    geometries.
    """

    ROOT = Path(__file__).resolve().parent.parent / 'data' / 'sample' / 'static'

    def test_one_row_per_image(self):
        frame = characterize_scenes(self.ROOT)
        images = sorted((self.ROOT / 'images').iterdir())
        assert len(frame) == len(images)
        assert list(frame['scene']) == [path.stem for path in images]

    def test_a_horizontal_mask_never_covers_less_than_its_oriented_conversion(self):
        # The HBB area is at least the OBB area.
        frame = characterize_scenes(self.ROOT)
        assert (frame['coverage_hbb'] >= frame['coverage_obb'] - 1e-9).all()

    def test_coverage_is_a_union_not_a_sum(self):
        # Coverage is the union of the boxes.
        frame = characterize_scenes(self.ROOT)
        assert (frame['coverage_obb'] > 0).all()
        assert (frame['coverage_obb'] < 100).all()

    def test_names_select_a_subset_by_stem_or_filename(self):
        images = sorted((self.ROOT / 'images').iterdir())
        chosen = images[0]
        assert list(characterize_scenes(self.ROOT, [chosen.stem])['scene']) == [chosen.stem]
        assert list(characterize_scenes(self.ROOT, [chosen.name])['scene']) == [chosen.stem]

    def test_the_summary_is_five_ordered_quantiles(self):
        summary = summarize_scenes(characterize_scenes(self.ROOT))
        assert list(summary['statistic']) == list(SCENE_STATISTICS)
        for column in ('n_objects', 'coverage_obb', 'coverage_hbb'):
            assert summary[column].is_monotonic_increasing

    def test_an_empty_scene_set_raises_rather_than_reporting_nothing(self):
        with pytest.raises(ValueError):
            summarize_scenes(pd.DataFrame())


def add_infeasible_run(results_dir, pipeline='loftr+none', masking='none', detector='loftr'):
    """Append one aborted run to an existing result directory, as a runner would write it."""
    store = ResultStore(Path(results_dir))
    number = 1 + max((int(path.stem) for path in Path(results_dir).glob('[0-9]*.json')), default=0)
    store.write_run(
        number,
        {
            'run_number': number,
            'params': {
                'pipeline': pipeline,
                'masking': masking,
                'detector_name': detector,
                'matcher_name': 'bf',
                'downsample_ratio': 1.0,
            },
            'records': [],
            'status': RUN_INFEASIBLE,
            'resource_error': {'type': 'OutOfMemoryError', 'message': 'CUDA out of memory', 'where': 'scene_a.jpg'},
        },
    )
    return results_dir


class TestInfeasibleRunsReachTheTable:
    """
    Runs a machine could not hold appear in the tables as N/A rows, distinct from `--` and from a
    failure rate.
    """

    def test_an_aborted_run_becomes_a_row(self, tmp_path):
        results_dir = add_infeasible_run(write_results(tmp_path, kind='static'))
        summary = aggregate_static(results_dir)
        aborted = summary[summary['status'] == RUN_INFEASIBLE] if 'status' in summary else summary.iloc[0:0]
        assert list(aborted['pipeline']) == ['loftr+none']
        assert int(aborted['n'].iloc[0]) == 0
        assert not np.isfinite(pd.to_numeric(aborted.get('maa', pd.Series([np.nan])), errors='coerce')).any()

    def test_it_does_not_disturb_the_measured_rows(self, tmp_path):
        results_dir = write_results(tmp_path, kind='static')
        before = aggregate_static(results_dir)
        after = aggregate_static(add_infeasible_run(results_dir))
        measured = after[after['status'].isna()] if 'status' in after else after
        assert len(measured) == len(before)
        assert np.allclose(measured['maa'].to_numpy(), before['maa'].to_numpy())

    def test_the_dynamic_aggregate_carries_them_too(self, tmp_path):
        summary = aggregate_dynamic(add_infeasible_run(write_results(tmp_path, kind='dynamic')))
        assert RUN_INFEASIBLE in set(summary['status'].dropna())

    def test_the_pooled_aggregate_carries_one_row_per_configuration(self, tmp_path):
        directories = [add_infeasible_run(d) for d in multi_case_results(tmp_path)]
        pooled = aggregate_cases(directories)
        aborted = pooled[pooled['status'] == RUN_INFEASIBLE]
        assert list(aborted['pipeline']) == ['loftr+none']
        assert int(aborted['n_cases'].iloc[0]) == 0
        assert len(pooled) - 1 == len(aggregate_cases(multi_case_results(tmp_path / 'clean')))

    def test_a_configuration_measured_in_some_cases_keeps_its_measured_row(self, tmp_path):
        directories = multi_case_results(tmp_path)
        add_infeasible_run(directories[0], pipeline='sift+obb', masking='obb', detector='sift')
        pooled = aggregate_cases(directories)
        assert 'status' not in pooled or pooled['status'].isna().all()

    def test_the_pooled_report_reads_na_and_lists_them(self, tmp_path):
        directories = [add_infeasible_run(d) for d in multi_case_results(tmp_path / 'cases')]
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        text = (tmp_path / 'out' / 'tables' / 'pooled_dynamic.tex').read_text(encoding='utf-8')
        assert 'LoFTR & none' in text and r'\textit{N/A}' in text
        assert 'could not be run on the benchmark machine' in text
        assert produced['infeasible'] == ['loftr+none']

    def test_the_table_reads_na_and_the_caption_explains_it(self, tmp_path):
        results_dir = add_infeasible_run(write_results(tmp_path, kind='static'))
        path = write_latex_table(
            aggregate_static(results_dir),
            tmp_path / 'static.tex',
            columns=STATIC_COLUMNS,
            caption='Part I ranking.',
            sort_by='maa',
        )
        text = path.read_text(encoding='utf-8')
        assert 'LoFTR & none' in text
        assert r'\textit{N/A}' in text
        assert 'could not be run' in text
        # N/A and `--` stay distinct.
        row = next(line for line in text.splitlines() if 'LoFTR' in line)
        assert '--' not in row

    def test_a_pipeline_aborted_under_every_mode_prints_one_row(self, tmp_path):
        results_dir = write_results(tmp_path, kind='static')
        for masking in ('none', 'hbb', 'obb'):
            add_infeasible_run(results_dir, pipeline=f"loftr+{masking}", masking=masking)
        summary = aggregate_static(results_dir)
        path = write_latex_table(summary, tmp_path / 'static.tex', columns=STATIC_COLUMNS, csv=True)
        text = path.read_text(encoding='utf-8')
        assert text.count('LoFTR') == 1 and 'LoFTR & all' in text
        assert 'N/A marks configurations' in text
        # The CSV keeps a row per configuration.
        assert pd.read_csv(path.with_suffix('.csv'))['pipeline'].str.startswith('loftr').sum() == 3

    def test_a_table_without_any_says_nothing_about_them(self, tmp_path):
        path = write_latex_table(
            aggregate_static(write_results(tmp_path, kind='static')),
            tmp_path / 'static.tex',
            columns=STATIC_COLUMNS,
            caption='Part I ranking.',
        )
        text = path.read_text(encoding='utf-8')
        assert 'could not be run' not in text
        assert 'N/A' not in text

    def test_all_runs_infeasible_is_an_error_that_says_why(self, tmp_path):
        results_dir = tmp_path / 'empty'
        store = ResultStore(results_dir)
        store.initialize(build_manifest({'kind': 'static', 'name': 'x', 'seed': 0}, 'hash', {1: {'pipeline': 'x'}}))
        add_infeasible_run(results_dir)
        with pytest.raises(ValueError, match='resource-infeasible'):
            load_results(results_dir)

    def test_rows_are_labeled_from_the_run_parameters(self, tmp_path):
        results_dir = tmp_path / 'r'
        store = ResultStore(results_dir)
        store.initialize(build_manifest({'kind': 'static', 'name': 'x', 'seed': 0}, 'hash', {1: {'pipeline': 'x'}}))
        add_infeasible_run(results_dir, pipeline='dedode+obb', masking='obb', detector='dedode')
        rows = infeasible_rows(results_dir)
        assert rows == [
            {
                'pipeline': 'dedode+obb',
                'masking': 'obb',
                'detector': 'dedode',
                'matcher': 'bf',
                'status': RUN_INFEASIBLE,
                'n': 0,
                'infeasible_reason': 'OutOfMemoryError',
            }
        ]


# --------------------------------------------------------------------------------------
# The analysis plan decides what is bolded
# --------------------------------------------------------------------------------------


def multi_case_results(tmp_path, n_cases=9, effect=4.0, seed=1):
    """
    Nine cases laid out as the dynamic runner writes them, with a built-in masking effect.

    `effect` is the CTE gap between `none` and `obb` in every case; zero makes them tie.
    """
    rng = np.random.default_rng(seed)
    directories = []
    for case in range(n_cases):
        rows = []
        base = 1.0 + rng.uniform(0.0, 1.0)
        for masking, shift in (('none', effect), ('hbb', effect / 2), ('obb', 0.0)):
            for index in range(20):
                rows.append(
                    {
                        'run_number': 1,
                        'pipeline': f"sift+{masking}",
                        'masking': masking,
                        'detector_name': 'sift',
                        'matcher_name': 'bf',
                        'pair': f"000000->{50 * (index + 1):06d}",
                        'abs_offset': 50 * (index + 1),
                        'cte_mean': float(abs(rng.normal(base + shift, 0.3))),
                        'ssim': float(np.clip(rng.normal(0.8 - shift / 20, 0.02), 0, 1)),
                        'zncc': float(np.clip(rng.normal(0.8 - shift / 20, 0.02), 0, 1)),
                        'mi': 1.0,
                        'gradient': 0.5,
                        'iou_mean': float(np.clip(rng.normal(0.9 - shift / 30, 0.02), 0, 1)),
                        'oce_mean': float(abs(rng.normal(shift / 10 + 0.1, 0.05))),
                        'inlier_ratio': 0.5,
                        'motion_alignment': 0.5,
                        'time_stabilize_s': 0.1,
                        'object_boxes': 'obb',
                        'failed': False,
                    }
                )
        results_dir = write_results(tmp_path / f"case{case}" / 'results', frame=pd.DataFrame(rows))
        write_audit(results_dir, residual_p95_px=0.5)
        directories.append(results_dir)
    return directories


class TestPoolingOrder:
    def test_the_tables_do_not_depend_on_the_order_the_cases_are_given_in(self, tmp_path):
        directories = multi_case_results(tmp_path / 'cases')
        build_report(directories, output=tmp_path / 'given', prefix='pooled')
        build_report(directories[3:] + directories[:3], output=tmp_path / 'rotated', prefix='pooled')
        tables = sorted((tmp_path / 'given' / 'tables').iterdir())
        assert any(path.name == 'pooled_comparisons.csv' for path in tables)
        for path in tables:
            assert path.read_bytes() == (tmp_path / 'rotated' / 'tables' / path.name).read_bytes(), path.name

    def test_cases_are_ordered_field_by_field_on_their_names(self, tmp_path):
        directories = [tmp_path / name / 'results' / 'campaign' for name in ('D10_PM5', 'D9_AM3', 'D1_PM1')]
        assert [case_name(d) for d in case_order(directories)] == ['D1_PM1', 'D10_PM5', 'D9_AM3']


class TestBoldingFollowsThePlan:
    def test_a_single_case_bolds_nothing(self, tmp_path):
        build_report(write_results(tmp_path / 'case' / 'results'), output=tmp_path / 'out')
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        assert '\\textbf' not in rendered and 'Nothing is bolded' in rendered

    def test_a_consistent_effect_is_bolded_alone_and_a_null_effect_ties(self, tmp_path):
        directories = multi_case_results(tmp_path / 'effect')
        produced = build_report(directories, output=tmp_path / 'out_effect', prefix='pooled')
        rendered = (tmp_path / 'out_effect' / 'tables' / 'pooled_dynamic.tex').read_text(encoding='utf-8')
        rows = {line.split('&')[1].strip(): line for line in rendered.splitlines() if line.strip().startswith('SIFT &')}
        maa_cell = lambda row: row.split('&')[2].strip()  # noqa: E731 - a one-line accessor
        assert maa_cell(rows['OBB']).startswith('\\textbf')
        assert not maa_cell(rows['none']).startswith('\\textbf')
        assert '9 cases' in rendered and '0.0039' in rendered
        assert produced['epsilon_floor_px'] == pytest.approx(0.5)

        tied = multi_case_results(tmp_path / 'null', effect=0.0, seed=2)
        build_report(tied, output=tmp_path / 'out_null', prefix='pooled')
        rendered = (tmp_path / 'out_null' / 'tables' / 'pooled_dynamic.tex').read_text(encoding='utf-8')
        rows = {line.split('&')[1].strip(): line for line in rendered.splitlines() if line.strip().startswith('SIFT &')}
        assert all(maa_cell(rows[mode]).startswith('\\textbf') for mode in ('none', 'HBB', 'OBB'))

    def test_the_pooled_report_writes_the_breakdowns_and_the_contrasts(self, tmp_path):
        directories = multi_case_results(tmp_path / 'effect')
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        tables = tmp_path / 'out' / 'tables'
        for name in ('by_offset', 'contrasts', 'mechanism', 'benefit_strata'):
            assert (tables / f"pooled_{name}.tex").exists(), name
        assert (tables / 'pooled_contrasts.csv').exists()
        assert 'pools its cases' in (tables / 'pooled_by_offset.tex').read_text(encoding='utf-8')
        # The fixture's cases carry no survey cell: the by-cell table is skipped, never median-split.
        assert any('by-scenario' in reason and '--cell' in reason for reason in produced['skipped'])
        assert any('GCP outcomes' in warning for warning in produced['warnings'])

    def test_the_pooled_margin_is_the_largest_residual_and_each_case_keeps_its_own_floor(self, tmp_path):
        directories = multi_case_results(tmp_path / 'cases')
        for index, directory in enumerate(directories):
            write_audit(directory, residual_p95_px=0.4 + 0.1 * index)
        (directories[-1].parent.parent / 'oracle' / 'AUDIT.json').unlink()
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        caption = (tmp_path / 'out' / 'tables' / 'pooled_dynamic.tex').read_text(encoding='utf-8')
        assert produced['epsilon_floor_px'] == pytest.approx(1.1)
        assert 'largest of the cases\' audited' in caption and '1.10 px' in caption
        assert 'each case\'s own residual' in caption
        assert '1 of the 9 cases have no audit' in caption

    def test_the_direction_of_the_error_is_never_bolded(self, tmp_path):
        directories = multi_case_results(tmp_path)
        frame = load_case_results(directories)
        ties = tie_sets_for(frame, aggregate_cases(directories), DYNAMIC_COLUMNS, ('pipeline', 'masking'), 'case')
        assert 'maa' in ties and 'motion_alignment' not in ties

    def test_the_caption_names_the_reference_detector_and_the_rows_sharing_it(self, tmp_path):
        frame = pd.concat(
            [
                synthetic_frame(30).assign(pipeline=f"{name}+{mode}", detector_name=name)
                for name in ('sift', 'rsift', 'orb')
                for mode in ('none',)
            ],
            ignore_index=True,
        )
        results_dir = write_results(tmp_path / 'case' / 'results', frame=frame)
        manifest = json.loads((results_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
        manifest.setdefault('metadata', {})['reference_estimator'] = {
            'pipeline': {'detector_name': 'sift', 'masking': 'obb'}
        }
        (results_dir / 'MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
        note = reference_note(results_dir, frame)
        assert "reference estimator's keypoint detector (SIFT, with OBB masking)" in note
        assert 'RootSIFT and SIFT rows' in note and 'ORB' not in note
        assert reference_note(write_results(tmp_path / 'other' / 'results'), frame) == ''

    def test_the_gcp_table_states_its_own_margins(self):
        ties = {'cte_median': object()}
        note = _bold_note(ties, 'case', 9, 0.42, against_gcp=True)
        assert '0.42 px, the GCP repeatability, for the pixel columns' in note and 'mAA' not in note
        assert 'no repeat annotation session' in _bold_note(ties, 'case', 9, None, against_gcp=True)
        assert '0.01 for mAA, 0.42 px for the reference-relative' in _bold_note(ties, 'case', 9, 0.42)

    def test_pooled_breakdowns_do_not_depend_on_the_order_of_the_cases(self, tmp_path):
        directories = multi_case_results(tmp_path / 'cases', n_cases=4)
        cells = ['low moving / mixed', 'high moving / coherent', 'stationary control', 'high moving / mixed']
        cell_of = {directory.parent.parent.name: cell for directory, cell in zip(directories, cells, strict=True)}

        def by_cell(order):
            frame = load_case_results([directories[i] for i in order])
            return aggregate_by_scenario(frame=frame.assign(cell=frame['case'].map(cell_of), coverage_ref=0.1))

        forward, backward = by_cell([0, 1, 2, 3]), by_cell([3, 2, 1, 0])
        pd.testing.assert_frame_equal(forward, backward)
        assert list(dict.fromkeys(forward['scenario'])) == [
            'high moving / coherent',
            'high moving / mixed',
            'low moving / mixed',
            'stationary control',
        ]

    def test_breakdowns_show_the_average_pipeline_and_write_every_pipeline_to_csv(self, tmp_path):
        directories = multi_case_results(tmp_path / 'cases')
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        tables = tmp_path / 'out' / 'tables'
        assert str(tables / 'pooled_by_offset.csv') in produced['tables']
        rendered = (tables / 'pooled_by_offset.tex').read_text(encoding='utf-8')
        rows = [line for line in rendered.splitlines() if '&' in line and 'Pipeline' not in line]
        assert rows and all(line.strip().startswith('All pipelines') for line in rows)
        assert 'mean over the 1 candidate pipelines' in rendered and 'pooled\\_by\\_offset.csv' in rendered

    def test_the_average_pipeline_keeps_baselines_and_sweep_values_apart(self):
        table = pd.DataFrame(
            [
                {'pipeline': 'sift+obb (margin 0)', 'masking': 'obb', 'bin': '1-15', 'zncc': 0.8},
                {'pipeline': 'orb+obb (margin 0)', 'masking': 'obb', 'bin': '1-15', 'zncc': 0.6},
                {'pipeline': 'sift+obb (margin 0.3)', 'masking': 'obb', 'bin': '1-15', 'zncc': 0.9},
                {'pipeline': 'sift+none', 'masking': 'none', 'bin': '1-15', 'zncc': 0.5},
                {'pipeline': 'identity', 'masking': 'none', 'bin': '1-15', 'zncc': 0.1},
            ]
        )
        average = average_pipeline_rows(table, 'bin').set_index('pipeline')
        assert average.loc['All pipelines (margin 0)', 'zncc'] == pytest.approx(0.7)
        assert average.loc['All pipelines (margin 0.3)', 'zncc'] == pytest.approx(0.9)
        assert average.loc['All pipelines', 'masking'] == 'none' and average.loc['identity', 'zncc'] == 0.1
        assert list(average.index)[0] == 'All pipelines'  # none before the masked modes
        assert average_pipeline_rows(table, 'bin').attrs['n_pipelines'] == 2

    def test_pooled_rows_weight_cases_equally(self, tmp_path):
        directories = multi_case_results(tmp_path)
        pooled = aggregate_cases(directories).set_index('masking')
        assert (pooled['n_cases'] == 9).all()
        per_case = pooled.attrs['per_case']
        expected = per_case[per_case['masking'] == 'obb']['cte_median'].mean()
        assert pooled.loc['obb', 'cte_median'] == pytest.approx(expected)


class TestComparisonsTable:
    """The per-comparison listing behind a results table's bold."""

    @staticmethod
    def _ties(effect, seed=0, margin=0.01):
        rng = np.random.default_rng(seed)
        best = rng.normal(0.90, 0.02, size=9)
        table = pd.DataFrame(
            {
                ('sift', 'obb'): best,
                ('sift', 'none'): best - effect,
                ('disk_lightglue', 'obb'): best + rng.normal(0.0, 0.004, size=9),
            },
            index=[f"case{i}" for i in range(9)],
        )
        return {'maa': tie_set(table, ('sift', 'obb'), True, margin=margin, draws=1000)}

    COLUMNS = {'maa': ('mAA$[0,5]$', '{:.3f}', True)}

    def test_every_comparison_becomes_a_row_under_its_best(self):
        frame = comparisons_frame(self._ties(0.12), self.COLUMNS)
        assert list(frame['reference']) == [True, False, False]
        assert frame.iloc[0]['group'] == 'best: sift / obb'
        assert set(frame[~frame['reference']]['group']) == {'sift / none', 'disk_lightglue / obb'}

    def test_holm_is_applied_within_the_outcome_and_never_lowers_a_p_value(self):
        frame = comparisons_frame(self._ties(0.12), self.COLUMNS)
        rows = frame[~frame['reference']]
        assert (rows['p_holm'] >= rows['p_value']).all()

    def test_the_tie_marker_agrees_with_what_the_results_table_bolds(self, tmp_path):
        ties = self._ties(0.12)
        frame = comparisons_frame(ties, self.COLUMNS)
        bolded = ties['maa'].bold
        for _, row in frame[~frame['reference']].iterrows():
            tied_here = not bool(row['distinguishable'])
            in_bold = row['group'].replace(' / ', ',') in {','.join(b) for b in bolded if isinstance(b, tuple)}
            assert tied_here == in_bold

        rendered = write_comparisons_table(frame, tmp_path / 'c.tex', caption='c', label='tab:c').read_text(
            encoding='utf-8'
        )
        ticks = sum(1 for line in rendered.splitlines() if '\\cmark' in line)
        assert ticks == int((~frame['reference'] & ~frame['distinguishable'].astype(bool)).sum())

    def test_math_in_a_header_survives_and_names_are_displayed(self, tmp_path):
        frame = comparisons_frame(self._ties(0.12), self.COLUMNS)
        two = pd.concat([frame, frame.assign(outcome='CTE$_{50}$', column='cte_median')], ignore_index=True)
        rendered = write_comparisons_table(two, tmp_path / 'c.tex', caption='c', label='tab:c').read_text(
            encoding='utf-8'
        )
        assert 'mAA$[0,5]$' in rendered and 'DISK+LightGlue / OBB' in rendered and 'best: SIFT / OBB' in rendered
        assert 'mAA\\$' not in rendered and 'Outcome &' in rendered

    def test_one_outcome_needs_no_outcome_column(self, tmp_path):
        frame = comparisons_frame(self._ties(0.12), self.COLUMNS)
        rendered = write_comparisons_table(frame, tmp_path / 'c.tex', caption='c', label='tab:c').read_text(
            encoding='utf-8'
        )
        assert 'Outcome' not in rendered and 'mAA$[0,5]$' not in rendered
        assert '\\begin{tabular}{lrrrc}' in rendered and 'best: SIFT / OBB & -- & -- & -- &' in rendered

    def test_the_pooled_dynamic_report_writes_it(self, tmp_path):
        directories = multi_case_results(tmp_path / 'effect')
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        path = tmp_path / 'out' / 'tables' / 'pooled_comparisons.tex'
        assert path.exists() and str(path) in produced['tables']
        rendered = path.read_text(encoding='utf-8')
        assert '\\label{tab:pooled-comparisons}' in rendered
        assert 'Hodges--Lehmann' in rendered and 'Holm' in rendered
        caption = rendered[rendered.index('\\caption{') : rendered.index('\\label{')]
        assert '\\Cref{tab:pooled-dynamic}' in caption and '9 cases' in caption
        assert not re.search(r'<\w+>|\{\{|\{(table|cluster|draws|n_clusters)\}', caption)

    def test_the_table_lists_maa_and_the_csv_every_outcome(self, tmp_path):
        directories = multi_case_results(tmp_path / 'effect')
        produced = build_report(directories, output=tmp_path / 'out', prefix='pooled')
        tables = tmp_path / 'out' / 'tables'
        listing = pd.read_csv(tables / 'pooled_comparisons.csv')
        assert str(tables / 'pooled_comparisons.csv') in produced['tables']
        assert {'maa', 'cte_median', 'zncc'} <= set(listing['column'])
        rendered = (tables / 'pooled_comparisons.tex').read_text(encoding='utf-8')
        assert 'Outcome' not in rendered and 'Only mAA, the ranking quantity' in rendered
        assert 'pooled\\_comparisons.csv' in rendered


def write_budget_results(tmp_path, presets=('sift', 'orb'), ratios=(0.25, 0.5, 1.0), budgets=(2000, 8000)):
    """A static campaign shaped like `configs/songdo/static/budget.yaml`: one run per grid point and masking."""
    results_dir = tmp_path / 'budget'
    store = ResultStore(results_dir)
    store.initialize(build_manifest({'kind': 'static', 'name': 'budget', 'seed': 0}, 'hash', {1: {'pipeline': 'x'}}))
    number = 0
    for preset in presets:
        for masking in ('none', 'obb'):
            for repeat, (ratio, budget) in enumerate(itertools.product(ratios, budgets)):
                number += 1
                store.write_run(
                    number,
                    {
                        'run_number': number,
                        'params': {
                            'pipeline': f"{preset}+{masking}" + (f"#{repeat + 1}" if repeat else ''),
                            'masking': masking,
                            'detector_name': preset,
                            'matcher_name': 'bf',
                            'downsample_ratio': ratio,
                            'max_features': budget,
                        },
                        'records': [
                            {
                                'cte_mean': 2.0 * (1.1 - ratio),
                                'object_boxes': 'obb',
                                'failed': False,
                                'time_stabilize_s': 0.1 * ratio * budget / 2000,
                            }
                            for _ in range(10)
                        ],
                    },
                )
    store.finalize(number, number)
    return results_dir


def write_sweep_results(tmp_path, degrees=(0, 30, 60)):
    """Static campaigns that differ only in `perturbation.geometric.rotation_deg`, one per magnitude."""
    directories = []
    for deg in degrees:
        results_dir = tmp_path / f"rotation_{deg}deg"
        config = {
            'kind': 'static',
            'name': f"rotation_{deg}deg",
            'seed': 0,
            'perturbation': {'geometric': {'rotation_deg': [deg, deg], 'scale_ratio': 0.02}},
        }
        store = ResultStore(results_dir)
        store.initialize(build_manifest(config, 'hash', {1: {'pipeline': 'x'}}))
        for number, preset in enumerate(('sift', 'xfeat'), start=1):
            # xfeat degrades with rotation, sift does not.
            error = 0.3 if preset == 'sift' else 0.3 + deg / 10
            store.write_run(
                number,
                {
                    'run_number': number,
                    'params': {'pipeline': f"{preset}+obb", 'masking': 'obb', 'detector_name': preset},
                    'records': [
                        {'cte_mean': error, 'object_boxes': 'obb', 'failed': False, 'time_stabilize_s': 0.1}
                        for _ in range(10)
                    ],
                },
            )
        store.finalize(2, 2)
        directories.append(results_dir)
    return directories


class TestBudgetCampaign:
    def test_a_budget_sweep_is_recognized(self, tmp_path):
        assert is_budget_campaign(load_results(write_budget_results(tmp_path)))
        assert not is_budget_campaign(load_results(write_estimator_results(tmp_path)))

    def test_one_row_per_grid_point_with_its_parameters(self, tmp_path):
        summary = aggregate_budget(write_budget_results(tmp_path))
        assert len(summary) == 2 * 2 * 3 * 2
        assert set(summary['preset']) == {'sift', 'orb'}
        row = summary[(summary['preset'] == 'orb') & (summary['downsample_ratio'] == 1.0)]
        assert set(row['max_features']) == {2000, 8000}
        assert (row['maa'] > summary[summary['downsample_ratio'] == 0.25]['maa'].max()).all()

    def test_the_compute_figure_renders_and_the_report_writes_it(self, tmp_path):
        results_dir = write_budget_results(tmp_path)
        assert figure_accuracy_vs_compute(
            aggregate_budget(results_dir), tmp_path / 'c.pdf', {'offline': (1.0, 8000), 'real-time': (0.18, 2000)}
        ).exists()
        produced = build_report(results_dir, output=tmp_path / 'out')
        assert any(path.endswith('budget_compute.pdf') for path in produced['figures']), produced['skipped']

    def test_no_finite_points_raises(self, tmp_path):
        summary = aggregate_budget(write_budget_results(tmp_path)).assign(maa=np.nan)
        with pytest.raises(ValueError, match='No finite'):
            figure_accuracy_vs_compute(summary, tmp_path / 'd.pdf')


class TestSweepReport:
    def test_the_one_varying_key_is_found(self):
        configs = [{'name': f"r{d}", 'perturbation': {'geometric': {'rotation_deg': [d, d], 'x': 1}}} for d in (0, 5)]
        assert sweep_key(configs) == 'perturbation.geometric.rotation_deg'

    def test_two_varying_keys_are_refused(self):
        configs = [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]
        with pytest.raises(ValueError, match='exactly one'):
            sweep_key(configs)

    def test_a_degenerate_range_reads_as_its_value(self):
        assert sweep_value({'p': {'r': [45, 45]}}, 'p.r') == 45.0
        with pytest.raises(ValueError, match='range'):
            sweep_value({'p': {'r': [0, 45]}}, 'p.r')

    def test_rows_carry_the_sweep_value(self, tmp_path):
        key, frame = aggregate_sweep(write_sweep_results(tmp_path))
        assert key == 'perturbation.geometric.rotation_deg'
        xfeat = frame[frame['pipeline'] == 'xfeat+obb'].sort_values('value')
        assert list(xfeat['value']) == [0.0, 30.0, 60.0]
        assert xfeat['maa'].is_monotonic_decreasing

    def test_several_static_directories_make_a_sweep_report(self, tmp_path):
        produced = build_report(write_sweep_results(tmp_path), output=tmp_path / 'out')
        assert produced['sweep_key'] == 'perturbation.geometric.rotation_deg'
        assert (tmp_path / 'out' / 'tables' / 'rotation_sweep.tex').exists()
        assert (tmp_path / 'out' / 'figures' / 'rotation_sweep.pdf').exists()
        table = (tmp_path / 'out' / 'tables' / 'rotation_sweep.tex').read_text(encoding='utf-8')
        assert ' 0 & 30 & 60 ' in table
        assert (tmp_path / 'out' / 'tables' / 'rotation_sweep.csv').exists()


def audit_report(residuals_by_offset, repeat=None):
    """An AUDIT.json payload: one audited frame per offset, with its point residuals."""
    per_frame = [
        {'frame_index': offset, 'offset': offset, 'residuals_px': list(values)}
        for offset, values in residuals_by_offset.items()
    ]
    bins = [(1, 120), (121, 480), (481, 1920), (1921, 3600)]
    report = {
        'reference_estimator': {
            'per_frame': per_frame,
            'summary': summarize_residuals(per_frame),
            'summary_by_offset': summarize_by_offset(per_frame, bins),
        }
    }
    if repeat is not None:
        report['annotation_repeatability'] = {'residual_p95_px': repeat}
    return report


class TestGcpCandidateTable:
    @staticmethod
    def _scores(directory: Path, errors) -> Path:
        directory.mkdir(parents=True)
        frames = [{'offset': k, 'cte_mean': e} for k, e in enumerate(errors, start=1)]
        finite = np.array([e for e in errors if np.isfinite(e)])
        summary = {
            'n_frames': len(frames),
            'n_points': 12 * len(frames),
            'cte_median_px': float(np.median(finite)),
            'cte_p90_px': float(np.percentile(finite, 90)),
            'iou_mean': 0.99,
            'oce_mean_px': 0.1,
        }
        candidate = {'label': 'orb+none', 'masking': 'none', 'gcp_transform': {'per_frame': frames, 'summary': summary}}
        (directory / GCP_SCORES_NAME).write_text(json.dumps({'candidates': [candidate]}), encoding='utf-8')
        return directory

    def test_the_largest_error_is_the_largest_audited_frame(self, tmp_path):
        (row,) = gcp_candidate_rows(self._scores(tmp_path / 'a', [0.4, 0.6, 19.4, float('nan'), 4469.4]))
        assert row['cte_max'] == pytest.approx(4469.4)

    def test_pooling_averages_the_percentiles_and_keeps_the_largest_frame(self, tmp_path):
        rows = [
            {**row, 'case': name}
            for name, errors in (('a', [0.4, 0.6, 19.4, 4469.4]), ('b', [0.5, 0.7, 0.9, 1.1]))
            for row in gcp_candidate_rows(self._scores(tmp_path / name, errors))
        ]
        (pooled,) = pool_gcp_rows(pd.DataFrame(rows)).to_dict('records')
        assert pooled['cte_max'] == pytest.approx(4469.4)
        assert pooled['cte_p90'] == pytest.approx(np.mean([r['cte_p90'] for r in rows]))
        assert pooled['n_frames'] == 8

    def test_the_largest_error_is_never_bolded(self):
        assert GCP_COLUMNS['cte_max'][2] is None


class TestGcpAuditTable:
    AUDITS = {
        'case_a': audit_report({10: [0.2, 0.4, 0.6], 300: [0.5, 0.7], 2000: [1.0, 1.2]}, repeat=0.8),
        'case_b': audit_report({50: [0.1, 0.3], 1000: [0.9, 2.0, 0.4]}),
    }

    def test_one_row_per_case_and_a_pooled_row(self):
        table = aggregate_gcp_audit(self.AUDITS, {'case_a': 'high moving / coherent'}).set_index('case')
        assert list(table.index) == ['case_a', 'case_b', 'all cases']
        assert table.loc['case_a', 'n_points'] == 7 and table.loc['case_a', 'cell'] == 'high moving / coherent'
        assert np.isnan(table.loc['case_b', 'repeat_p95_px'])
        assert table.loc['all cases', 'repeat_p95_px'] == pytest.approx(0.8)

    def test_the_pooled_row_recomputes_percentiles_over_all_points(self):
        table = aggregate_gcp_audit(self.AUDITS).set_index('case')
        points = [0.2, 0.4, 0.6, 0.5, 0.7, 1.0, 1.2, 0.1, 0.3, 0.9, 2.0, 0.4]
        assert table.loc['all cases', 'n_points'] == len(points)
        assert table.loc['all cases', 'residual_p95_px'] == pytest.approx(np.percentile(points, 95))
        assert table.loc['all cases', 'p95_1-120'] == pytest.approx(np.percentile([0.2, 0.4, 0.6, 0.1, 0.3], 95))
        assert table.loc['all cases', 'share_within_half_px'] == pytest.approx(np.mean(np.array(points) < 0.5))

    def test_with_proposals_only_the_hand_placed_points_check_the_panel(self):
        review = {'n_points': 7, 'n_accepted': 4, 'n_corrected': 1, 'n_placed_by_hand': 2, 'corrections_px': [0.5]}
        audits = {
            'case_a': {
                **self.AUDITS['case_a'],
                'proposal_review': {**review, 'n_hand_scored': 2, 'n_hand_within_half_px': 0},
            },
            'case_b': self.AUDITS['case_b'],
        }
        table = aggregate_gcp_audit(audits).set_index('case')
        assert table.loc['case_a', 'share_within_half_px'] == 0.0
        assert table.loc['case_b', 'share_within_half_px'] == pytest.approx(3 / 5)
        assert table.loc['all cases', 'share_within_half_px'] == pytest.approx(3 / 7)

    def test_the_proposal_table_pools_counts_and_corrections(self):
        reviews = {
            'case_a': {
                'n_points': 7,
                'n_accepted': 4,
                'n_corrected': 2,
                'n_placed_by_hand': 1,
                'corrections_px': [0.5, 1.5],
            },
            'case_b': {
                'n_points': 5,
                'n_accepted': 4,
                'n_corrected': 1,
                'n_placed_by_hand': 0,
                'corrections_px': [0.2],
            },
        }
        audits = {name: {**self.AUDITS[name], 'proposal_review': review} for name, review in reviews.items()}
        audits['case_c'] = self.AUDITS['case_b']
        table = aggregate_gcp_proposals(audits).set_index('case')
        assert list(table.index) == ['case_a', 'case_b', 'all cases']
        assert table.loc['all cases', 'n_points'] == 12 and table.loc['all cases', 'n_corrected'] == 3
        assert table.loc['all cases', 'correction_median_px'] == pytest.approx(0.5)
        assert aggregate_gcp_proposals({'case_b': self.AUDITS['case_b']}).empty

    def test_bins_a_case_did_not_audit_are_blank(self):
        table = aggregate_gcp_audit(self.AUDITS).set_index('case')
        assert np.isnan(table.loc['case_b', 'p95_121-480'])
        columns = list(gcp_audit_columns(table.reset_index()))
        assert columns.index('p95_1-120') < columns.index('p95_1921-3600') < columns.index('repeat_p95_px')

    def test_the_pooled_report_writes_it(self, tmp_path):
        directories = []
        for name, audit in self.AUDITS.items():
            directories.append(write_tuning_results(tmp_path / name, kind='dynamic', n_records=5))
            (tmp_path / name / 'oracle').mkdir()
            # Audited elsewhere: the recorded path does not exist here.
            (tmp_path / name / 'oracle' / 'AUDIT.json').write_text(
                json.dumps({**audit, 'case': f"/elsewhere/{name}"}), encoding='utf-8'
            )
            (tmp_path / name / 'CASE.json').write_text(json.dumps({'cell': f"cell of {name}"}), encoding='utf-8')
        build_report(directories, output=tmp_path / 'out', prefix='pooled')
        rendered = (tmp_path / 'out' / 'tables' / 'gcp_audit.tex').read_text(encoding='utf-8')
        assert r'\label{tab:gcp-audit}' in rendered and 'all cases' in rendered and 'case\\_b' in rendered
        assert 'cell of case\\_a' in rendered
        assert not (tmp_path / 'out' / 'tables' / 'gcp_proposals.tex').exists()


class TestSweptRunLabels:
    def test_runs_of_a_sweep_are_named_by_what_varies(self, tmp_path):
        names = swept_run_labels(load_results(write_tuning_results(tmp_path)))
        assert names['sift+none'] == 'sift+none (ratio 0.8, thr. 2)'
        assert names['sift+none#4'] == 'sift+none (ratio 0.9, thr. 3)'
        assert len(set(names.values())) == len(names)

    def test_a_campaign_without_repeated_labels_keeps_its_names(self):
        assert swept_run_labels(synthetic_frame()) == {}

    def test_the_static_table_uses_them(self, tmp_path):
        build_report(write_tuning_results(tmp_path), kind='static', output=tmp_path / 'out')
        rendered = (tmp_path / 'out' / 'tables' / 'tuning_static.tex').read_text(encoding='utf-8')
        # The Masking column shows `none`, so the name drops its `+none`.
        assert 'SIFT (ratio 0.9, thr. 3) & none' in rendered and '\\#' not in rendered

    def test_the_dynamic_table_uses_them_and_leaves_the_baselines_alone(self, tmp_path):
        frame = synthetic_frame()
        sweep = frame.assign(pipeline=frame['pipeline'] + '#2', mask_margin_ratio=0.3)
        frame = pd.concat([frame.assign(mask_margin_ratio=0.0), sweep], ignore_index=True)
        baseline = frame.iloc[:5].assign(pipeline='reference', masking='none', mask_margin_ratio=np.nan)
        directory = write_results(tmp_path, frame=pd.concat([frame, baseline], ignore_index=True))
        build_report(directory, output=tmp_path / 'out')
        rendered = (tmp_path / 'out' / 'tables' / 'results_dynamic.tex').read_text(encoding='utf-8')
        assert 'SIFT (margin 0.3) & OBB' in rendered and 'SIFT (margin 0) & OBB' in rendered
        # A baseline is not a configuration of the masking axis, so it shows no masking.
        assert 'Reference & --' in rendered and 'Reference & none' not in rendered
        assert 'margin nan' not in rendered


def test_part_ii_tuning_never_writes_part_i_tuning_tables():
    assert default_prefix('tuning', 'static') == 'tuning'
    assert default_prefix('tuning', 'dynamic') == 'tuning_dynamic'
    assert default_prefix('masking', 'dynamic') == 'masking'


def test_a_tuning_campaign_names_the_configs_of_its_own_regime():
    assert evaluation_configs('offline_tuning') == 'configs/songdo/dynamic/offline_masking.yaml and offline_margin.yaml'
    assert evaluation_configs('realtime_tuning') == 'configs/songdo/dynamic/realtime_masking.yaml'


def test_the_tuning_caption_states_the_threshold_units():
    frame = pd.DataFrame({'ransac_threshold_space': ['processed'] * 2})
    assert threshold_units(frame) == 'pixels of the downsampled image'
    assert threshold_units(frame.drop(columns='ransac_threshold_space')) == 'full-resolution pixels'


def test_every_table_of_a_campaign_names_its_regime_and_budget(tmp_path):
    frame = pd.DataFrame({'downsample_ratio': [0.18] * 2, 'max_features': [2000] * 2})
    label = regime_label('realtime_masking', frame)
    assert label == 'real-time regime at downsample ratio 0.18 and 2000 features'
    assert regime_label('masking', frame) == 'at downsample ratio 0.18 and 2000 features'
    assert regime_label('realtime_masking', frame.assign(max_features=[2000, 8000])) == ''
    captions = {
        'realtime_masking_dynamic.tex': r'\caption{Part II (dynamic benchmark, 9 cases): performance}',
        'realtime_masking_contrasts.tex': r'\caption{Part II: the planned masking contrasts}',
        'realtime_masking_by_offset.tex': r'\caption{Part II results grouped by offset}',
        'realtime_masking_comparisons.tex': r'\caption{The analysis plan applied}',
        'realtime_masking_repeat_spread.tex': r'\caption{CTE spread}',
        'gcp_audit.tex': r'\caption{Part II: the reference estimator}',
    }
    for name, caption in captions.items():
        (tmp_path / name).write_text(caption, encoding='utf-8')
    tag_regime([str(tmp_path / name) for name in captions], 'realtime_masking', label)
    read = {name: (tmp_path / name).read_text(encoding='utf-8') for name in captions}
    dynamic = rf'\caption{{Part II ({label}; dynamic benchmark, 9 cases): performance}}'
    assert read['realtime_masking_dynamic.tex'] == dynamic
    assert read['realtime_masking_contrasts.tex'] == rf'\caption{{Part II ({label}): the planned masking contrasts}}'
    assert read['realtime_masking_by_offset.tex'] == rf'\caption{{Part II ({label}) results grouped by offset}}'
    assert read['realtime_masking_comparisons.tex'] == rf'\caption{{Part II ({label}): the analysis plan applied}}'
    assert read['realtime_masking_repeat_spread.tex'] == captions['realtime_masking_repeat_spread.tex']
    assert read['gcp_audit.tex'] == captions['gcp_audit.tex']


class TestCaseName:
    """A dynamic result is pooled under its case, wherever it was written."""

    def test_the_manifest_names_the_case(self, tmp_path):
        directory = tmp_path / 'reruns' / 'results' / 'offline_masking'
        directory.mkdir(parents=True)
        (directory / 'MANIFEST.json').write_text(
            json.dumps({'metadata': {'case': {'dir': 'data/songdo/dynamic/x'}}}), encoding='utf-8'
        )
        assert case_name(directory) == 'x'

    def test_without_a_manifest_the_case_holds_its_results(self, tmp_path):
        directory = tmp_path / 'case_a' / 'results' / 'offline_masking'
        directory.mkdir(parents=True)
        assert case_name(directory) == 'case_a'
        assert case_name(tmp_path / 'elsewhere') == str(tmp_path / 'elsewhere')


class TestColumnsWithoutInformation:
    """A column that shows the same value in every row leaves the table for one caption sentence."""

    def frame(self, **columns):
        base = {'pipeline': ['sift+none', 'orb+none'], 'masking': ['none', 'none'], 'maa': [0.98, 0.97]}
        return pd.DataFrame({**base, **columns})

    def render(self, tmp_path, frame, omit, unit='trial'):
        path = write_latex_table(frame, tmp_path / 't.tex', columns=STATIC_COLUMNS, caption='C.', omit=omit, unit=unit)
        return path.read_text(encoding='utf-8')

    def test_saturated_hea_columns_share_one_sentence(self, tmp_path):
        text = self.render(tmp_path, self.frame(hea_1px=[0.999, 1.0], hea_3px=[1.0, 1.0]), STATIC_OMIT)
        assert 'HEA$_{1}$ &' not in text and 'HEA$_{3}$ &' not in text
        assert 'HEA$_{1}$ and HEA$_{3}$ are at least 0.999 in every row.' in text

    def test_an_hea_that_separates_configurations_stays(self, tmp_path):
        text = self.render(tmp_path, self.frame(hea_1px=[0.95, 1.0]), STATIC_OMIT)
        assert '& HEA$_{1}$' in text and 'at least' not in text

    def test_no_failure_is_said_once_and_a_failure_keeps_the_column(self, tmp_path):
        text = self.render(tmp_path, self.frame(failure_rate=[0.0, 0.0]), STATIC_OMIT)
        assert 'Fail' not in text and 'No configuration failed on any trial.' in text
        text = self.render(tmp_path, self.frame(failure_rate=[0.0, 0.01]), STATIC_OMIT)
        assert 'Fail' in text and 'No configuration failed' not in text

    def test_a_shared_count_and_a_narrow_region_are_stated(self, tmp_path):
        frame = self.frame(n_cases=[9, 9], region_fraction=[0.565, 0.567])
        columns = {**DYNAMIC_COLUMNS, 'n_cases': ('Cases', '{:.0f}', None)}
        omit = {name: OMIT_NOTES[name] for name in ('n_cases', 'region_fraction')}
        text = write_latex_table(frame, tmp_path / 't.tex', columns=columns, caption='C.', omit=omit).read_text(
            encoding='utf-8'
        )
        assert 'Cases' not in text.split('\\midrule')[0].split('caption')[0]
        assert 'Every row pools 9 cases.' in text
        assert 'over a median 56.5\\% to 56.7\\% of the frame' in text

    def test_columns_not_offered_are_kept(self, tmp_path):
        text = self.render(tmp_path, self.frame(failure_rate=[0.0, 0.0]), None)
        assert 'Fail' in text

    def test_infeasible_rows_do_not_hold_a_column_in_the_table(self, tmp_path):
        frame = self.frame(failure_rate=[0.0, float('nan')]).assign(status=['', 'infeasible'])
        text = self.render(tmp_path, frame, STATIC_OMIT)
        assert 'No configuration failed on any trial.' in text and r'\textit{N/A}' in text


def test_the_gcp_counts_leave_the_table_when_every_row_shares_them():
    rows = pd.DataFrame({'n_frames': [72, 72], 'n_points': [1064, 1064]})
    columns, sentence = gcp_table_columns(rows)
    assert 'n_frames' not in columns and 'n_points' not in columns
    assert sentence == ' Every row is scored on 72 audited frames and 1064 ground control points.'
    columns, sentence = gcp_table_columns(rows.assign(n_points=[1064, 1000]))
    assert 'n_points' in columns and 'Frames and Points' in sentence


def test_the_starting_values_name_the_pipelines_that_start_elsewhere():
    note = starting_values_note(['orb', 'sift', 'rsift', 'brisk', 'akaze', 'xfeat'])
    assert note.startswith('the untuned starting values, a ratio of 0.9 and an inlier threshold of 2 px')
    assert '3 px for RootSIFT and SIFT' in note and '2.5 px for AKAZE and BRISK' in note


def test_reports_default_to_reports_in_the_working_directory(tmp_path, monkeypatch):
    results_dir = write_results(tmp_path / 'case' / 'results')
    monkeypatch.chdir(tmp_path / 'case')
    produced = build_report(results_dir)
    assert produced['tables'] and all(
        Path(path).resolve().is_relative_to((tmp_path / 'case' / 'reports' / 'tables').resolve())
        for path in produced['tables']
    )
