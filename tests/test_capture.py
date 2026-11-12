#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Consensus capture as the report reads it: counts, the per-offset share, the example pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as image
import numpy as np
import pandas as pd
import pytest

from stabilo_benchmark.report import capture
from stabilo_benchmark.report.capture import (
    build_capture_outputs,
    capture_by_offset,
    capture_table,
    figure_capture_by_offset,
)
from stabilo_benchmark.report.tables import campaign_offset_bins, load_case_results
from stabilo_benchmark.runners.dynamic import run_dynamic

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = REPO_ROOT / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
SMOKE_CONFIG = REPO_ROOT / 'configs' / 'sample' / 'dynamic.yaml'
BINS = ((1, 1), (2, 3), (4, 7))


def measurements():
    """Two cases; orb+none captured on two pairs of one case, never under masking."""
    rows = []
    for case in ('a', 'b'):
        for masking in ('none', 'obb'):
            for offset, fraction in ((1, 0.9), (2, 0.7), (5, 0.1), (6, float('nan'))):
                hit = masking == 'none' and case == 'a' and fraction >= 0.5
                rows.append(
                    {
                        'case': case,
                        'pipeline': f"orb+{masking}",
                        'masking': masking,
                        'abs_offset': offset,
                        'capture_fraction': fraction if hit or fraction != fraction else 0.05,
                        'cte_mean': 30.0 if hit else 0.4,
                        'n_inliers': 800 if hit else 600,
                        'inlier_ratio': 0.4 if hit else 0.3,
                        'inliers_on_moving': 0.6 if hit else 0.0,
                    }
                )
    return pd.DataFrame(rows)


class TestCaptureTable:
    def test_pairs_without_a_group_are_not_counted(self):
        table = capture_table(measurements()).set_index('pipeline')
        assert table.loc['orb+none', 'n_pairs'] == 6
        assert table.loc['orb+obb', 'n_pairs'] == 6

    def test_captures_are_counted_with_their_cases(self):
        table = capture_table(measurements()).set_index('pipeline')
        assert table.loc['orb+none', 'n_captured'] == 2
        assert table.loc['orb+none', 'captured_share'] == pytest.approx(2 / 6)
        assert table.loc['orb+none', 'n_cases_captured'] == 1
        assert table.loc['orb+obb', 'n_captured'] == 0 and table.loc['orb+obb', 'n_cases_captured'] == 0

    def test_inlier_statistics_of_captured_and_other_pairs_are_separate(self):
        row = capture_table(measurements()).set_index('pipeline').loc['orb+none']
        assert (row['inliers_captured'], row['inliers_other']) == (800, 600)
        assert row['cte_captured'] == pytest.approx(30.0) and row['on_moving_captured'] == pytest.approx(0.6)
        assert np.isnan(capture_table(measurements()).set_index('pipeline').loc['orb+obb', 'cte_captured'])

    def test_the_captured_are_ranked_within_their_case_and_offset_bin(self):
        frame = measurements()
        ranks = capture.inlier_ranks(frame, BINS)
        hit = (frame['masking'] == 'none') & (frame['case'] == 'a') & (frame['abs_offset'] == 1)
        # Alone in its pipeline, case and bin, a measurement takes the top rank.
        assert ranks.loc[hit, 'inliers_rank'].tolist() == [1.0]
        row = capture_table(frame, BINS).set_index('pipeline').loc['orb+none']
        assert row['inliers_rank_captured'] == pytest.approx(1.0) and row['ratio_rank_captured'] == pytest.approx(1.0)

    def test_a_rank_compares_estimates_of_one_bin(self):
        frame = pd.DataFrame(
            {
                'case': 'a',
                'pipeline': 'orb+none',
                'masking': 'none',
                'abs_offset': [2, 3, 3, 5],
                'n_inliers': [100, 300, 200, 50],
                'inlier_ratio': [0.1, 0.3, 0.2, 0.05],
            }
        )
        ranks = capture.inlier_ranks(frame, BINS)
        assert ranks['inliers_rank'].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3, 1.0])

    def test_the_largest_capture_fraction_and_the_captured_cases_are_reported(self):
        table = capture_table(measurements()).set_index('pipeline')
        assert table.loc['orb+none', 'capture_fraction_max'] == pytest.approx(0.9)
        assert table.loc['orb+obb', 'capture_fraction_max'] == pytest.approx(0.05)
        assert (table.loc['orb+none', 'cases_captured'], table.loc['orb+obb', 'cases_captured']) == ('a', '')

    def test_each_case_counts_its_pairs_and_captures(self):
        table = capture.capture_by_case(measurements()).set_index(['case', 'pipeline'])
        assert table.loc[('a', 'orb+none'), ['n_measurements', 'n_pairs', 'n_captured']].tolist() == [4, 3, 2]
        assert table.loc[('b', 'orb+none'), 'n_captured'] == 0

    def test_captured_measurements_on_audited_frames_carry_their_gcp_error(self, tmp_path):
        directories = []
        for case, frames in (('a', {'orb+none': [(1, 19.4), (5, 0.3)], 'orb+obb': [(1, 0.5)]}), ('b', {})):
            directory = tmp_path / case / 'results' / 'realtime_masking'
            directory.mkdir(parents=True)
            candidates = [
                {
                    'label': label,
                    'masking': label.split('+')[1],
                    'gcp_transform': {'per_frame': [{'offset': k, 'cte_mean': e} for k, e in entries]},
                }
                for label, entries in frames.items()
            ]
            (directory / 'GCP_SCORES.json').write_text(json.dumps({'candidates': candidates}), encoding='utf-8')
            directories.append(directory)
        frame = capture.with_gcp_errors(measurements(), directories)
        assert len(frame) == len(measurements()) and frame['cte_gcp'].notna().sum() == 3
        table = capture_table(frame).set_index('pipeline')
        assert table.loc['orb+none', ['n_captured_audited', 'cte_gcp_captured']].tolist() == [1, pytest.approx(19.4)]
        assert table.loc['orb+obb', 'n_captured_audited'] == 0

    def test_without_gcp_scores_no_measurement_is_audited(self, tmp_path):
        frame = capture.with_gcp_errors(measurements(), [tmp_path])
        assert frame['cte_gcp'].isna().all()
        assert capture_table(frame).set_index('pipeline').loc['orb+none', 'n_captured_audited'] == 0

    def test_the_counts_at_stricter_thresholds_are_kept(self):
        row = capture_table(measurements()).set_index('pipeline').loc['orb+none']
        assert (row['n_captured_at_0.25'], row['n_captured_at_0.5'], row['n_captured_at_1']) == (2, 2, 0)
        frame = measurements().assign(
            capture_fraction=lambda f: f['capture_fraction'].mask(f['capture_fraction'] >= 0.5, 0.3)
        )
        row = capture_table(frame).set_index('pipeline').loc['orb+none']
        assert (row['n_captured'], row['n_captured_at_0.25'], row['n_captured_at_0.5']) == (2, 2, 0)
        frame = measurements().assign(
            capture_fraction=lambda f: f['capture_fraction'].mask(f['capture_fraction'] >= 0.5, 6.0)
        )
        assert capture_table(frame).set_index('pipeline').loc['orb+none', 'n_captured_at_1'] == 2

    def test_the_table_lists_the_captured_and_names_the_others(self, tmp_path):
        text = capture.write_capture_table(capture_table(measurements()), tmp_path / 'c.tex', 'tab:c', 2, 'c.csv')
        rows = [line for line in text.read_text(encoding='utf-8').splitlines() if line.strip().startswith('ORB')]
        assert len(rows) == 1 and '& none &' in rows[0]
        assert 'the configurations captured on at least one pair' in text.read_text(encoding='utf-8')
        assert 'The other 1 configurations are captured on no pair, with a largest $\\lambda$ of 0.05.' in (
            text.read_text(encoding='utf-8')
        )
        assert 'No masked estimate exceeds $\\lambda = 0.05$.' in text.read_text(encoding='utf-8')
        assert 'Every row counts 6 such pairs.' in text.read_text(encoding='utf-8')

    def test_without_a_capture_every_configuration_is_listed(self, tmp_path):
        table = capture_table(measurements().assign(capture_fraction=0.0))
        text = capture.write_capture_table(table, tmp_path / 'c.tex', 'tab:c', 2, 'c.csv').read_text(encoding='utf-8')
        assert sum(line.strip().startswith('ORB') for line in text.splitlines()) == 2
        assert 'at least one pair' not in text and 'The other' not in text

    @pytest.mark.parametrize(('n_cases', 'words'), [(1, 'over the one case,'), (9, 'over the 9 cases,')])
    def test_the_caption_counts_the_cases(self, tmp_path, n_cases, words):
        path = capture.write_capture_table(capture_table(measurements()), tmp_path / 'c.tex', 'tab:c', n_cases, 'c.csv')
        assert words in path.read_text(encoding='utf-8')


class TestByOffset:
    def test_each_bin_counts_its_own_pairs(self):
        table = capture_by_offset(measurements(), BINS)
        none = table[table['pipeline'] == 'orb+none'].set_index('offset_bin')
        assert none.loc['1-1', 'n_captured'] == 1 and none.loc['1-1', 'n_pairs'] == 2
        assert none.loc['4-7', 'n_captured'] == 0 and none.loc['4-7', 'n_pairs'] == 2
        assert list(none.index) == ['1-1', '2-3', '4-7']

    def test_nothing_captured_draws_nothing(self, tmp_path):
        frame = measurements().assign(capture_fraction=0.0)
        with pytest.raises(ValueError, match='no measurement was captured'):
            figure_capture_by_offset(capture_by_offset(frame, BINS), tmp_path / 'f.pdf')

    def test_the_scatter_axis_spans_every_measurement(self, tmp_path, monkeypatch):
        axes = []
        monkeypatch.setattr(capture, '_save', lambda figure, path: axes.extend(figure.axes) or path)
        frame = measurements()
        frame.loc[frame['capture_fraction'].notna().idxmax(), 'capture_fraction'] = -12.0
        capture.figure_capture_scatter(frame, tmp_path / 's.pdf')
        assert axes[0].get_ylim()[0] < -12.0

    def test_a_capture_is_drawn_per_pipeline(self, tmp_path):
        path = figure_capture_by_offset(capture_by_offset(measurements(), BINS), tmp_path / 'f.pdf')
        assert path.is_file() and path.stat().st_size > 0

    def test_regimes_share_a_panel_and_a_configuration_its_color(self, tmp_path, monkeypatch):
        axes = []
        monkeypatch.setattr(capture, '_save', lambda figure, path: axes.extend(figure.axes) or path)
        table = capture_by_offset(measurements(), BINS).drop(columns='bin_rank')
        both = pd.concat([table.assign(regime='real time'), table.assign(regime='offline')], ignore_index=True)
        capture.figure_capture_by_offset(both, tmp_path / 'f.pdf')
        assert len(axes) == 1
        lines = {line.get_label(): line for line in axes[0].get_lines()}
        pairs = {label.rsplit(', ', 1)[0] for label in lines}
        assert len(lines) == 2 * len(pairs)
        for name in pairs:
            first, second = lines[f"{name}, real time"], lines[f"{name}, offline"]
            assert first.get_color() == second.get_color() and first.get_linestyle() != second.get_linestyle()
            assert second.get_markerfacecolor() == 'none' and first.get_markerfacecolor() != 'none'
            assert second.get_zorder() > first.get_zorder()
            assert first.get_markevery() is None

    def test_only_a_figure_without_regimes_titles_its_legend_with_the_shared_mode(self, tmp_path, monkeypatch):
        axes = []
        monkeypatch.setattr(capture, '_save', lambda figure, path: axes.extend(figure.axes) or path)
        table = capture_by_offset(measurements(), BINS).drop(columns='bin_rank')
        capture.figure_capture_by_offset(table, tmp_path / 'f.pdf')
        capture.figure_capture_by_offset(table.assign(regime='offline'), tmp_path / 'g.pdf')
        assert [panel.get_legend().get_title().get_text() for panel in axes] == ['No masking', '']

    def test_the_scatter_draws_every_measurement_with_a_group(self, tmp_path):
        path = capture.figure_capture_scatter(measurements(), tmp_path / 's.pdf', horizon=3)
        assert path.is_file() and path.stat().st_size > 0
        # Every pipeline has a panel, captured or not; only measurements without a group draw nothing.
        assert capture.figure_capture_scatter(measurements().assign(capture_fraction=0.0), tmp_path / 'z.pdf')
        with pytest.raises(ValueError, match='no measurement on a pair with a moving group'):
            capture.figure_capture_scatter(measurements().assign(capture_fraction=np.nan), tmp_path / 'n.pdf')


def series(rows):
    """One configuration's measurements from (case, offset, capture fraction, error, inliers, ratio)."""
    columns = ['case', 'abs_offset', 'capture_fraction', 'cte_mean', 'n_inliers', 'inlier_ratio']
    return pd.DataFrame(rows, columns=columns).assign(pipeline='orb+none', masking='none')


class TestPopulations:
    def test_kinds_of_captured_estimate(self):
        frame = series(
            [
                ('a', 5, 0.40, 8.0, 500, 0.4),  # pull
                ('a', 6, 0.02, 1.0, 500, 0.4),  # not captured
                ('a', 7, 0.40, 500.0, 90, 0.05),  # a breakdown within the horizon is still one
                ('a', 60, 0.60, 900.0, 90, 0.05),  # breakdown
                ('a', 61, -0.60, 900.0, 90, 0.05),  # broken down, not captured
                ('a', 70, 0.15, 1.5, 700, 0.4),  # captured late, with a small error
            ]
        )
        expected = ['pull', '', 'breakdown', 'breakdown', '', 'other']
        assert capture.populations(frame, horizon=31).tolist() == expected

    def test_without_a_horizon_nothing_is_a_pull(self):
        frame = series([('a', 5, 0.40, 8.0, 500, 0.4), ('a', 60, 0.60, 900.0, 90, 0.05)])
        assert capture.populations(frame).tolist() == ['other', 'breakdown']

    def test_the_nearest_offsets_skip_the_captured_and_the_broken(self):
        frame = series(
            [
                ('a', 1, 0.00, 1.0, 100, 0.20),
                ('a', 2, 0.00, 1.0, 300, 0.40),
                ('a', 3, 0.50, 5.0, 200, 0.30),  # against offsets 1, 2, 6 and 7
                ('a', 4, 0.00, 500.0, 10, 0.01),  # broken down: never a neighbor
                ('a', 5, 0.20, 5.0, 999, 0.99),  # captured: never a neighbor
                ('a', 6, 0.00, 1.0, 200, 0.30),
                ('a', 7, 0.00, 1.0, 200, 0.30),
            ]
        )
        ratios = capture.nearest_offset_ratios(frame)
        assert ratios.loc[2, 'inliers_nearest'] == pytest.approx(1.0)
        assert ratios.loc[2, 'ratio_nearest'] == pytest.approx(1.0)

    def test_the_nearest_offsets_stay_within_a_case(self):
        frame = series([('a', 1, 0.50, 5.0, 200, 0.30), ('a', 2, 0.00, 1.0, 100, 0.30), ('b', 2, 0.00, 1.0, 900, 0.90)])
        assert capture.nearest_offset_ratios(frame).loc[0, 'inliers_nearest'] == pytest.approx(2.0)

    def test_each_kind_is_a_row_of_the_table(self):
        frame = series(
            [
                ('a', 3, 0.50, 5.0, 200, 0.30),
                ('a', 4, 0.00, 1.0, 200, 0.30),
                ('b', 5, 0.30, 9.0, 300, 0.40),
                ('b', 6, 0.00, 1.0, 300, 0.40),
                ('b', 60, 2.00, 900.0, 90, 0.05),
                ('b', 61, 0.00, 1.0, 180, 0.10),
            ]
        )
        table = capture.capture_populations(frame, horizon=31).set_index('population')
        assert table.loc['pull', ['n_captured', 'n_cases', 'offset_max']].tolist() == [2, 2, 5]
        assert table.loc['pull', 'cte_median'] == pytest.approx(7.0)
        assert table.loc['breakdown', 'inliers_nearest'] == pytest.approx(90 / 240)
        assert 'other' not in table.index
        assert np.isnan(table.loc['pull', 'error_group'])

    def test_a_configuration_never_captured_has_no_row(self):
        frame = series([('a', 3, 0.05, 1.0, 200, 0.30)])
        assert capture.capture_populations(frame, horizon=31).empty

    def test_the_error_at_the_group_joins_its_measurements(self):
        frame = series([('a', 3, 0.50, 5.0, 200, 0.30), ('a', 4, 0.40, 6.0, 200, 0.30), ('a', 5, 0.0, 1.0, 200, 0.3)])
        errors = pd.DataFrame({'error_group': [2.0, 4.0], 'error_angle': [3.0, 9.0]}, index=[0, 1])
        row = capture.capture_populations(frame, 31, errors).iloc[0]
        assert row['error_group'] == pytest.approx(3.0) and row['error_angle_max'] == pytest.approx(9.0)

    def test_the_caption_states_its_definitions(self, tmp_path):
        frame = series([('a', 3, 0.50, 5.0, 200, 0.30), ('a', 4, 0.00, 1.0, 200, 0.30)])
        path = capture.write_populations_table(
            capture.capture_populations(frame, 31), tmp_path / 'p.tex', 'tab:p', 9, 31
        )
        text = path.read_text(encoding='utf-8')
        assert 'over the 9 cases' in text and 'up to 31 frames' in text and 'reaches 100\\,px' in text
        assert 'nearest two offsets' in text
        assert path.with_suffix('.csv').is_file()

    def test_breakdowns_are_counted_over_every_measurement(self):
        frame = measurements()
        frame.loc[frame['capture_fraction'].isna(), 'cte_mean'] = 500.0
        frame.loc[frame['capture_fraction'].notna().idxmax(), 'cte_mean'] = 500.0
        row = capture_table(frame).set_index('pipeline').loc['orb+none']
        assert row['n_measurements'] == 8
        assert (row['n_broken_down'], row['n_broken_down_with_group']) == (3, 1)

    def test_the_caption_names_what_the_identity_reaches(self, tmp_path):
        table = capture_table(measurements().assign(coherent_shift_px=5.2))
        text = capture.write_capture_table(table, tmp_path / 'c.tex', 'tab:c', 2, 'c.csv', 0.0686)
        assert 'identity transform reaches $\\lambda = 0.07$' in text.read_text(encoding='utf-8')
        assert 'smallest displacement of any group is 5\\,px' in text.read_text(encoding='utf-8')


class TestExampleChoice:
    def test_the_largest_capture_fraction_within_the_horizon_comes_first(self):
        frame = measurements()
        assert capture.example_candidates(frame)['abs_offset'].tolist() == [1, 2]
        frame.loc[(frame['abs_offset'] == 5) & (frame['masking'] == 'none'), 'capture_fraction'] = 2.0
        assert capture.example_candidates(frame)['abs_offset'].iloc[0] == 5
        assert capture.example_candidates(frame, horizon=3)['abs_offset'].tolist() == [1, 2]

    def test_without_a_capture_within_the_horizon_every_capture_is_a_candidate(self):
        frame = measurements()
        assert capture.example_candidates(frame, horizon=0)['abs_offset'].tolist() == [1, 2]


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    """The sample campaign, with the unmasked ORB run marked captured wherever its pair has a group."""
    root = tmp_path_factory.mktemp('runs')
    args = argparse.Namespace(overwrite=False, resume=False, regrid=False, verbosity=0, results_dir=root)
    directory = run_dynamic(CASE_DIR, SMOKE_CONFIG, args)
    marked = 0
    for path in sorted(directory.glob('0*.json')):
        run = json.loads(path.read_text(encoding='utf-8'))
        if run['params'].get('pipeline') == 'orb+none':
            for record in run['records']:
                if record['capture_fraction'] == record['capture_fraction']:
                    record['capture_fraction'] = 0.9
                    marked += 1
            path.write_text(json.dumps(run), encoding='utf-8')
    assert marked, 'the sample case has no pair with a group'
    return directory


def candidates_of(directory):
    frame = load_case_results([directory])
    return frame[~frame['pipeline'].isin(('identity', 'reference'))]


@pytest.mark.slow
@pytest.mark.skipif(not (CASE_DIR / 'oracle').is_dir(), reason='the bundled dynamic sample case is not present')
class TestExample:
    def test_the_example_reproduces_its_stored_estimate_and_is_drawn(self, results, tmp_path):
        produced = {'tables': [], 'figures': [], 'skipped': []}
        bins = campaign_offset_bins(results)
        build_capture_outputs(
            produced, candidates_of(results), [results], tmp_path / 'tables', tmp_path / 'figures', 'x', bins
        )
        names = {Path(path).name for path in produced['figures']}
        assert {'x_capture_example.pdf', 'x_capture_by_offset.pdf'} <= names, produced['skipped']
        assert (tmp_path / 'figures' / 'x_graphical_abstract.pdf').is_file()
        assert (tmp_path / 'figures' / 'x_graphical_abstract.png').is_file()
        assert (tmp_path / 'tables' / 'x_capture.tex').is_file()
        assert (tmp_path / 'tables' / 'x_capture_by_case.csv').is_file()

    def test_the_populations_read_the_error_at_the_group_from_the_frames(self, results, tmp_path):
        produced = {'tables': [], 'figures': [], 'skipped': []}
        frame = load_case_results([results])
        build_capture_outputs(
            produced,
            candidates_of(results),
            [results],
            tmp_path / 'tables',
            tmp_path / 'figures',
            'x',
            campaign_offset_bins(results),
            identity=frame[frame['pipeline'] == 'identity'],
        )
        table = pd.read_csv(tmp_path / 'tables' / 'x_capture_populations.csv')
        assert len(table) and table['error_group'].notna().all(), produced['skipped']
        assert table['error_angle_max'].between(0, 180).all()
        assert 'identity' in set(pd.read_csv(tmp_path / 'tables' / 'x_capture.csv')['pipeline'])

    def test_an_estimate_that_does_not_reproduce_is_not_drawn(self, results, tmp_path, monkeypatch):
        monkeypatch.setattr(capture, 'REPRODUCTION_TOLERANCE_PX', -1.0)
        produced = {'tables': [], 'figures': [], 'skipped': []}
        bins = campaign_offset_bins(results)
        build_capture_outputs(
            produced, candidates_of(results), [results], tmp_path / 'tables', tmp_path / 'figures', 'x', bins
        )
        assert any('capture_example' in reason and 'reproduced' in reason for reason in produced['skipped'])


def synthetic_panels():
    """Panels as `_reregister` returns them: three objects, two of which moved together, two runs."""
    quads = np.array([[[x, 100], [x + 40, 100], [x + 40, 120], [x, 120]] for x in (300, 360, 900)], dtype=float)
    labels = SimpleNamespace(corners=lambda: quads)
    motion = SimpleNamespace(
        group=np.array([True, True, False]),
        moved=np.array([True, True, False]),
        reference_centers=quads.mean(axis=1),
        displacement=np.array([[12.0, 0.0], [11.0, 1.0], [0.0, 0.0]]),
    )
    index = -np.ones((540, 960), dtype=int)
    index[100:120, 300:340], index[100:120, 360:400] = 0, 1
    shift = np.array([[1.0, 0.0, -11.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def run(masking, estimate, fraction):
        return {
            'spec': SimpleNamespace(masking=masking, name=f"orb+{masking}"),
            'record': {
                'cte_mean': 11.0 if masking == 'none' else 0.4,
                'capture_fraction': fraction,
                'n_inliers': 466 if masking == 'none' else 783,
                'inliers_on_moving': 0.2 if masking == 'none' else 0.0,
            },
            'estimate': estimate,
            'inliers': np.array([[310.0, 110.0], [370.0, 105.0], [500.0, 300.0], [700.0, 200.0]]),
        }

    return {
        'reference': np.full((540, 960, 3), 120, dtype=np.uint8),
        'reference_labels': labels,
        'reference_matrix': np.eye(3),
        'motion': motion,
        'object_index': index,
        'case': 'case_a',
        'offset': 10,
        'runs': [run('none', shift, 0.96), run('obb', np.eye(3), -0.07)],
    }


def test_the_example_and_its_graphical_abstract_are_drawn(tmp_path):
    panels = synthetic_panels()
    assert capture._draw_example(panels, tmp_path / 'e.pdf').stat().st_size > 0
    path = capture._draw_graphical_abstract(panels, tmp_path / 'g.pdf')
    assert path.stat().st_size > 0 and path.with_suffix('.png').is_file()
    height, width = image.imread(path.with_suffix('.png')).shape[:2]
    # The journal asks for at least 1328 x 531 pixels at a 5:2 aspect.
    assert width >= 1328 and height >= 531 and 2.2 < width / height < 2.8
