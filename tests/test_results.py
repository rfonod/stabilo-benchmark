#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Result storage, manifests, provenance, resume semantics, and seeding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from stabilo_benchmark.results import (
    FAILURE_MARKER,
    MANIFEST_NAME,
    RUN_COMPLETED,
    RUN_INFEASIBLE,
    SUCCESS_MARKER,
    WARNING_MARKER,
    Manifest,
    ResultError,
    ResultStore,
    build_manifest,
    infeasible_runs,
    package_versions,
    seed_everything,
    system_metadata,
    to_dataframe,
)

CONFIG = {'kind': 'static', 'name': 'unit', 'seed': 7, '_path': '/tmp/unit.yaml'}
RUNS = {1: {'pipeline': 'orb+none', 'masking': 'none'}, 2: {'pipeline': 'orb+obb', 'masking': 'obb'}}


def make_manifest():
    return build_manifest(CONFIG, 'deadbeef', RUNS)


class TestProvenance:
    def test_package_versions_cover_what_moves_numbers(self):
        versions = package_versions()
        for key in ('stabilo_benchmark', 'python', 'opencv', 'numpy', 'stabilo'):
            assert key in versions

    def test_absent_package_is_reported_not_raised(self):
        assert package_versions()['kornia'] is not None

    def test_system_metadata_has_the_manifest_fields(self):
        metadata = system_metadata()
        for key in ('timestamp', 'os', 'hardware', 'software'):
            assert key in metadata
        assert 'gpu' in metadata['hardware']
        assert 'git' not in metadata


class TestSeeding:
    def test_seeds_every_generator_that_can_move_a_number(self):
        seeded = seed_everything(42)
        assert seeded['numpy'] == 42
        assert seeded['opencv'] == 42  # np.random.seed does NOT reach OpenCV's RNG

    def test_reseeding_reproduces_the_same_draws(self):
        seed_everything(3)
        first = np.random.rand(5)
        seed_everything(3)
        assert np.array_equal(first, np.random.rand(5))


class TestManifest:
    def test_round_trips_through_a_dict(self):
        manifest = make_manifest()
        restored = Manifest.from_dict(manifest.to_dict())
        assert restored.runs == manifest.runs
        assert restored.config_hash == manifest.config_hash
        assert restored.seed == 7

    def test_run_keys_survive_json_as_integers(self):
        manifest = make_manifest()
        restored = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
        assert set(restored.runs) == {1, 2}

    def test_provenance_keys_are_excluded_from_the_stored_config(self):
        assert '_path' not in make_manifest().config


class TestResultStore:
    def test_writes_the_manifest_on_initialize(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        assert (tmp_path / 'results' / MANIFEST_NAME).exists()

    def test_run_records_round_trip(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(1, {'params': RUNS[1], 'records': [{'cte_mean': 0.5}]})
        assert store.has_run(1)
        assert store.read_run(1)['records'][0]['cte_mean'] == 0.5

    def test_numpy_values_serialize(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(
            1, {'params': RUNS[1], 'records': [{'a': np.float64(1.5), 'b': np.int64(2), 'c': np.array([1.0, 2.0])}]}
        )
        record = store.read_run(1)['records'][0]
        assert record == {'a': 1.5, 'b': 2, 'c': [1.0, 2.0]}

    def test_nan_survives_the_round_trip(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(1, {'params': RUNS[1], 'records': [{'cte_mean': float('nan')}]})
        assert np.isnan(store.read_run(1)['records'][0]['cte_mean'])

    def test_success_marker_on_completion(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.finalize(2, 2)
        assert (tmp_path / 'results' / SUCCESS_MARKER).exists()
        assert store.is_complete()

    def test_warning_marker_when_stopped_early(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.finalize(1, 2)
        assert (tmp_path / 'results' / WARNING_MARKER).exists()
        assert not store.is_complete()

    def test_failure_report_records_the_traceback_and_parameters(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.record_failure(2, RUNS[2], 2, RuntimeError('boom'))
        text = (tmp_path / 'results' / FAILURE_MARKER).read_text(encoding='utf-8')
        assert 'boom' in text
        assert 'orb+obb' in text

    def test_overwrite_and_resume_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(ResultError, match='mutually exclusive'):
            ResultStore(tmp_path, overwrite=True, resume=True)

    def test_resume_rejects_a_changed_grid(self, tmp_path):
        ResultStore(tmp_path / 'results').initialize(make_manifest())
        changed = build_manifest(CONFIG, 'deadbeef', {1: RUNS[1]})
        with pytest.raises(ResultError, match='differs from the one this config'):
            ResultStore(tmp_path / 'results', resume=True).initialize(changed)

    def test_resume_accepts_the_same_grid_and_records_the_resumption(self, tmp_path):
        ResultStore(tmp_path / 'results').initialize(make_manifest())
        manifest = ResultStore(tmp_path / 'results', resume=True).initialize(make_manifest())
        assert len(manifest.history) == 1
        assert 'resumed_at' in manifest.history[0]

    def test_resume_clears_stale_failure_markers(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.record_failure(1, RUNS[1], 2, RuntimeError('boom'))
        ResultStore(tmp_path / 'results', resume=True).initialize(make_manifest())
        assert not (tmp_path / 'results' / FAILURE_MARKER).exists()

    def test_overwrite_clears_the_success_marker(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.finalize(2, 2)
        ResultStore(tmp_path / 'results', overwrite=True).initialize(make_manifest())
        assert not (tmp_path / 'results' / SUCCESS_MARKER).exists()

    def test_a_plain_rerun_does_not_discard_an_interrupted_campaign(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(1, {'run_number': 1, 'params': RUNS[1], 'records': []})
        with pytest.raises(ResultError, match='--resume'):
            ResultStore(tmp_path / 'results').initialize(make_manifest())
        assert (tmp_path / 'results' / '0000001.json').exists()

    def test_overwrite_removes_the_earlier_run_records(self, tmp_path):
        # A run of a larger earlier grid must not be aggregated with the new one.
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(7, {'run_number': 7, 'params': RUNS[1], 'records': []})
        store.finalize(2, 2)
        ResultStore(tmp_path / 'results', overwrite=True).initialize(make_manifest())
        assert not list((tmp_path / 'results').glob('[0-9]*.json'))

    def test_reading_a_missing_run_raises(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        with pytest.raises(ResultError, match='not found'):
            store.read_run(99)

    def test_corrupt_run_file_raises(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.run_path(1).write_text('{not json', encoding='utf-8')
        with pytest.raises(ResultError, match='Corrupt result file'):
            store.read_run(1)

    def test_iter_runs_is_in_run_order(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        for number in (2, 1):
            store.write_run(number, {'params': RUNS[number], 'records': [], 'run_number': number})
        assert [run['run_number'] for run in store.iter_runs()] == [1, 2]


class TestDataFrame:
    def test_joins_params_onto_every_measurement(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        store.write_run(
            1,
            {
                'run_number': 1,
                'params': {'pipeline': 'orb+none', 'masking': 'none'},
                'records': [{'cte_mean': 0.5}, {'cte_mean': 1.5}],
            },
        )
        frame = to_dataframe(store)
        assert len(frame) == 2
        assert set(frame['masking']) == {'none'}
        assert frame['run_number'].tolist() == [1, 1]

    def test_empty_directory_gives_an_empty_frame(self, tmp_path):
        store = ResultStore(tmp_path / 'results')
        store.initialize(make_manifest())
        assert to_dataframe(store).empty


class TestEmptyRunsAreNotSuccess:
    """A run that measured nothing downgrades the marker to WARNING."""

    def test_all_runs_measured_gives_success(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.finalize(3, 3)
        assert (tmp_path / 'r' / 'SUCCESS.txt').exists()
        assert not (tmp_path / 'r' / 'WARNING.txt').exists()

    def test_an_empty_run_downgrades_the_marker_and_names_it(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.finalize(3, 3, ['akaze+none'])
        assert not (tmp_path / 'r' / 'SUCCESS.txt').exists()
        warning = (tmp_path / 'r' / 'WARNING.txt').read_text(encoding='utf-8')
        assert 'akaze+none' in warning
        assert 'no measurements' in warning

    def test_stopping_early_still_reports_the_count(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.finalize(1, 3)
        assert 'stopped after 1/3' in (tmp_path / 'r' / 'WARNING.txt').read_text(encoding='utf-8')


class TestInfeasibleRuns:
    """
    An infeasible run keeps the SUCCESS marker and is named in it; runs are told apart by `status`,
    not by being empty.
    """

    def test_an_infeasible_run_keeps_the_success_marker_and_is_named_in_it(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.finalize(3, 3, empty=['loftr+none'], infeasible=['loftr+none'])
        success = tmp_path / 'r' / 'SUCCESS.txt'
        assert success.exists()
        assert not (tmp_path / 'r' / WARNING_MARKER).exists()
        assert 'loftr+none' in success.read_text(encoding='utf-8')
        assert 'N/A' in success.read_text(encoding='utf-8')

    def test_an_empty_run_beside_an_infeasible_one_still_warns(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.finalize(3, 3, empty=['akaze+none', 'loftr+none'], infeasible=['loftr+none'])
        warning = (tmp_path / 'r' / WARNING_MARKER).read_text(encoding='utf-8')
        assert 'akaze+none' in warning
        assert 'loftr+none' in warning  # in the note, as infeasible
        assert not (tmp_path / 'r' / SUCCESS_MARKER).exists()

    def test_they_are_read_back_with_their_reason(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.write_run(
            1,
            {
                'run_number': 1,
                'params': {'pipeline': 'orb+none'},
                'records': [{'cte_mean': 1.0}],
                'status': RUN_COMPLETED,
            },
        )
        store.write_run(
            2,
            {
                'run_number': 2,
                'params': {'pipeline': 'loftr+none'},
                'records': [],
                'status': RUN_INFEASIBLE,
                'resource_error': {'type': 'OutOfMemoryError', 'message': 'CUDA out of memory', 'where': 'scene_a.jpg'},
            },
        )
        aborted = infeasible_runs(store)
        assert [run['params']['pipeline'] for run in aborted] == ['loftr+none']
        assert aborted[0]['resource_error']['where'] == 'scene_a.jpg'

    def test_they_contribute_no_measurements(self, tmp_path):
        store = ResultStore(tmp_path / 'r')
        store.initialize(make_manifest())
        store.write_run(
            1,
            {
                'run_number': 1,
                'params': {'pipeline': 'orb+none'},
                'records': [{'cte_mean': 1.0}],
                'status': RUN_COMPLETED,
            },
        )
        store.write_run(
            2, {'run_number': 2, 'params': {'pipeline': 'loftr+none'}, 'records': [], 'status': RUN_INFEASIBLE}
        )
        frame = to_dataframe(store)
        assert list(frame['pipeline']) == ['orb+none']


class TestGridExtension:
    CONFIG = {
        'kind': 'dynamic',
        'name': 't',
        'seed': 0,
        'sampling': {'per_bin': 2},
        'pipeline': {'filter_ratio': [0.7, 0.8]},
    }

    @staticmethod
    def _runs(ratios):
        runs = {i + 1: {'pipeline': f"sift+none#{i + 1}", 'filter_ratio': r} for i, r in enumerate(ratios)}
        runs[len(ratios) + 1] = {'pipeline': 'identity', 'detector_name': 'identity', 'baseline': True}
        return runs

    def _store(self, tmp_path, ratios):
        store = ResultStore(tmp_path)
        store.initialize(build_manifest(self.CONFIG, 'a', self._runs(ratios)))
        for number, params in self._runs(ratios).items():
            store.write_run(number, {'run_number': number, 'params': params})
        store.finalize(len(ratios) + 1, len(ratios) + 1)
        return store

    def test_stored_runs_move_to_their_place_in_the_new_grid(self, tmp_path):
        self._store(tmp_path, [0.7, 0.8])
        config = dict(self.CONFIG, pipeline={'filter_ratio': [0.6, 0.7, 0.8, 0.9]})
        store = ResultStore(tmp_path, regrid=True)
        manifest = store.initialize(build_manifest(config, 'b', self._runs([0.6, 0.7, 0.8, 0.9])))
        assert [store.has_run(n) for n in range(1, 6)] == [False, True, True, False, True]
        for number in (2, 3, 5):
            record = store.read_run(number)
            assert record['run_number'] == number and record['params'] == manifest.runs[number]
        stored = Manifest.from_dict(json.loads(store.manifest_path.read_text(encoding='utf-8')))
        assert stored.config == config and stored.config_hash == 'b' and stored.runs == manifest.runs
        assert [sorted(entry) for entry in stored.history] == [['metadata', 'resumed_at']]
        assert not store.is_complete()

    def test_a_changed_setting_outside_the_grid_is_refused(self, tmp_path):
        self._store(tmp_path, [0.7, 0.8])
        config = dict(self.CONFIG, sampling={'per_bin': 3}, pipeline={'filter_ratio': [0.6, 0.7, 0.8]})
        with pytest.raises(ResultError, match='outside the pipeline grid'):
            ResultStore(tmp_path, regrid=True).initialize(build_manifest(config, 'b', self._runs([0.6, 0.7, 0.8])))

    def test_a_removed_value_deletes_its_runs(self, tmp_path):
        self._store(tmp_path, [0.7, 0.8, 0.9])
        config = dict(self.CONFIG, pipeline={'filter_ratio': [0.6, 0.8]})
        store = ResultStore(tmp_path, regrid=True)
        manifest = store.initialize(build_manifest(config, 'b', self._runs([0.6, 0.8])))
        assert [store.has_run(n) for n in range(1, 4)] == [False, True, True]
        assert store.read_run(2)['params'] == manifest.runs[2] and store.read_run(3)['params'] == manifest.runs[3]
        assert sorted(path.name for path in tmp_path.glob('[0-9]*.json')) == ['0000002.json', '0000003.json']

    def test_a_run_whose_values_changed_is_deleted_and_runs_again(self, tmp_path):
        # A preset re-tuned in the preset file expands to new parameters under an unchanged config.
        self._store(tmp_path, [0.7, 0.8])
        store = ResultStore(tmp_path, regrid=True)
        store.initialize(build_manifest(self.CONFIG, 'a', self._runs([0.7, 0.85])))
        assert [store.has_run(n) for n in range(1, 4)] == [True, False, True]

    def test_runs_the_grid_now_collapses_keep_the_exact_match_and_drop_the_rest(self, tmp_path):
        # Stored at two ratios a pipeline ignores; the extended grid runs it once, at 0.9.
        def key(run):
            return json.dumps({k: v for k, v in run.items() if k not in ('pipeline', 'filter_ratio')}, sort_keys=True)

        stored = {1: {'pipeline': 'lg+none', 'filter_ratio': 0.7}, 2: {'pipeline': 'lg+none#2', 'filter_ratio': 0.9}}
        store = ResultStore(tmp_path)
        store.initialize(build_manifest(self.CONFIG, 'a', stored))
        for number, params in stored.items():
            store.write_run(number, {'run_number': number, 'params': params, 'records': [{'n': number}]})
        config = dict(self.CONFIG, pipeline={'filter_ratio': [0.6, 0.7, 0.8]})
        store = ResultStore(tmp_path, regrid=True, run_key=key)
        manifest = store.initialize(build_manifest(config, 'b', {1: {'pipeline': 'lg+none', 'filter_ratio': 0.9}}))
        assert store.read_run(1)['records'] == [{'n': 2}] and store.read_run(1)['params'] == manifest.runs[1]
        assert sorted(path.name for path in tmp_path.rglob('*.json')) == ['0000001.json', MANIFEST_NAME]

    def test_plain_resume_still_refuses_a_changed_grid(self, tmp_path):
        self._store(tmp_path, [0.7, 0.8])
        config = dict(self.CONFIG, pipeline={'filter_ratio': [0.6, 0.7, 0.8]})
        with pytest.raises(ResultError, match='differs'):
            ResultStore(tmp_path, resume=True).initialize(build_manifest(config, 'b', self._runs([0.6, 0.7, 0.8])))
