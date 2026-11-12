#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Part II runner: repeated measurement and reproducibility, on the bundled sample case."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from stabilo_benchmark.config import build_oracle, build_pipelines, load_config
from stabilo_benchmark.masking import Labels
from stabilo_benchmark.metrics.photometric import PHOTOMETRIC_METRICS
from stabilo_benchmark.oracle import OracleError, compute_oracle, load_oracle, oracle_matrices
from stabilo_benchmark.regions import RegionError
from stabilo_benchmark.results import RUN_COMPLETED, RUN_INFEASIBLE
from stabilo_benchmark.runners import dynamic as dynamic_runner
from stabilo_benchmark.runners.dynamic import (
    BASELINE_IDENTITY,
    BASELINE_REFERENCE,
    NO_EVALUATION_MASK,
    DynamicBenchmarkError,
    check_evaluation_mask,
    evaluate_baseline,
    load_evaluation_mask,
    plan_pairs,
    run_dynamic,
    run_pipeline,
)
from stabilo_benchmark.runners.prepare import load_case

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = REPO_ROOT / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
SMOKE_CONFIG = REPO_ROOT / 'configs' / 'sample' / 'dynamic.yaml'

pytestmark = pytest.mark.skipif(
    not (CASE_DIR / 'CASE.json').is_file() or not (CASE_DIR / 'oracle').is_dir(),
    reason='the bundled dynamic sample case is not present',
)


@pytest.fixture(scope='module')
def sample():
    """The sample case, its frame pairs, and the cached reference homographies."""
    config = load_config(SMOKE_CONFIG)
    case = load_case(CASE_DIR)
    pairs = plan_pairs(case, config)
    references = oracle_matrices(load_oracle(CASE_DIR))
    spec = build_pipelines(config)[0]
    return config, pairs, references, spec


def _run(sample, repeats):
    config, pairs, references, spec = sample
    config = {**config, 'repeats': repeats}
    return run_pipeline(spec, CASE_DIR, pairs, references, config)


@pytest.fixture(scope='module')
def shared_run(sample):
    """One run per repeat count for the tests that only read it; each test gets its own copy."""
    runs = {}

    def run(repeats):
        if repeats not in runs:
            runs[repeats] = _run(sample, repeats)
        return copy.deepcopy(runs[repeats])

    return run


@pytest.mark.slow
class TestRepeats:
    def test_defaults_to_one_measurement_per_pair(self, sample, shared_run):
        _, pairs, _, _ = sample
        result = shared_run(1)
        assert result['repeats'] == 1
        assert len(result['records']) == len(pairs)
        assert {record['repeat'] for record in result['records']} == {0}

    def test_produces_one_record_per_pair_per_repeat(self, sample, shared_run):
        _, pairs, _, _ = sample
        result = shared_run(3)
        assert result['repeats'] == 3
        assert len(result['records']) == 3 * len(pairs)
        assert sorted({record['repeat'] for record in result['records']}) == [0, 1, 2]

    def test_every_pair_appears_in_every_repeat(self, sample, shared_run):
        _, pairs, _, _ = sample
        result = shared_run(3)
        for repeat in range(3):
            keys = {r['pair'] for r in result['records'] if r['repeat'] == repeat}
            assert len(keys) == len(pairs)

    def test_n_pairs_counts_pairs_not_measurements(self, sample, shared_run):
        # Repeats do not change `n_pairs`.
        _, pairs, _, _ = sample
        assert shared_run(3)['n_pairs'] == len(pairs)


@pytest.mark.slow
class TestReproducibility:
    def test_a_classical_pipeline_repeats_exactly(self, shared_run):
        # OpenCV's estimators are bit-exact across repeated calls (tests/test_seeding.py).
        result = shared_run(3)
        by_pair = {}
        for record in result['records']:
            by_pair.setdefault(record['pair'], []).append(record.get('cte_mean'))
        for values in by_pair.values():
            assert len(set(values)) == 1

    def test_two_invocations_agree(self, sample):
        metrics = ('cte_mean', 'ssim', 'zncc', 'n_inliers', 'inlier_ratio')

        def measure():
            return [{key: record.get(key) for key in metrics if key in record} for record in _run(sample, 1)['records']]

        assert measure() == measure()


@pytest.mark.slow
class TestObjectBoxes:
    def test_hbb_changes_only_the_box_scores(self, sample):
        config, pairs, references, spec = sample

        def run(object_boxes):
            metrics = {**config.get('metrics', {}), 'object_boxes': object_boxes}
            return run_pipeline(spec, CASE_DIR, pairs, references, {**config, 'metrics': metrics})['records']

        obb, hbb = run('obb'), run('hbb')
        assert {record['object_boxes'] for record in obb} == {'obb'}
        assert {record['object_boxes'] for record in hbb} == {'hbb'}
        assert [r.get('cte_mean') for r in obb] == [r.get('cte_mean') for r in hbb]
        assert [r.get('ssim') for r in obb] == [r.get('ssim') for r in hbb]
        assert [r.get('iou_mean') for r in obb] != [r.get('iou_mean') for r in hbb]


@pytest.mark.slow
class TestWithoutAnnotations:
    """A case without labels_hbb/ and labels_obb/ runs with masking 'none' only, reference estimator included."""

    @pytest.fixture
    def case(self, tmp_path):
        case = tmp_path / 'case'
        shutil.copytree(CASE_DIR, case, ignore=shutil.ignore_patterns('labels_*', 'PROVENANCE_*', 'results'))
        return case

    @staticmethod
    def _args(tmp_path, *overrides):
        return argparse.Namespace(
            results_dir=tmp_path / 'results', overwrite=True, resume=False, verbosity=0, set=list(overrides), name=None
        )

    def test_the_reference_estimator_refuses_a_mask(self, sample, case):
        config, pairs, _, _ = sample
        with pytest.raises(OracleError, match='no annotations'):
            compute_oracle(case, pairs, build_oracle(config), overwrite=True)

    def test_candidates_needing_boxes_are_refused(self, case, tmp_path):
        with pytest.raises(DynamicBenchmarkError, match='no annotations'):
            run_dynamic(case, SMOKE_CONFIG, self._args(tmp_path))

    def test_masking_none_runs_and_is_recorded(self, case, tmp_path):
        results_dir = run_dynamic(
            case, SMOKE_CONFIG, self._args(tmp_path, 'pipeline.masking=none', 'oracle.masking=none')
        )
        manifest = json.loads((results_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
        assert manifest['metadata']['case']['annotated'] is False
        record = json.loads(next(results_dir.glob('0000001.json')).read_text(encoding='utf-8'))['records'][0]
        assert record['n_objects_ref'] == 0 and record['n_objects_cur'] == 0
        assert record['cte_mean'] == record['cte_mean']  # finite against the reference estimator


@pytest.mark.slow
class TestBaselinesAndStrata:
    def test_persistence_is_recorded_per_pair_and_bounded(self, shared_run):
        records = shared_run(1)['records']
        values = [r['persistence'] for r in records]
        assert all(0.0 <= v <= 1.0 for v in values if v == v)

    def test_a_failed_pair_carries_the_identity_photometric_score(self, sample, monkeypatch):
        # A failed pair carries the identity warp's photometric scores.
        config, pairs, references, spec = sample
        identity = evaluate_baseline(BASELINE_IDENTITY, CASE_DIR, pairs, references, config)['records']
        monkeypatch.setattr(dynamic_runner, 'estimation_failed', lambda *_: True)
        failed = _run(sample, 1)['records']
        assert all(r['failed'] and r['photometric_from_identity'] for r in failed)
        assert [r['ssim'] for r in failed] == pytest.approx([r['ssim'] for r in identity])
        assert all(r['cte_mean'] != r['cte_mean'] for r in failed)  # geometric scores stay NaN

    def test_an_empty_photometric_list_records_no_photometric_score(self, sample):
        config, pairs, references, spec = sample
        metrics = {**config.get('metrics', {}), 'photometric': []}
        records = run_pipeline(spec, CASE_DIR, pairs, references, {**config, 'metrics': metrics})['records']
        assert records and not any(name in record for record in records for name in PHOTOMETRIC_METRICS)
        assert all('cte_mean' in record and 'region_area_fraction' in record for record in records)

    def test_the_suppressed_share_follows_the_mask_geometry_and_margin(self, sample):
        config, pairs, references, _ = sample
        grid = {'preset': ['orb'], 'masking': ['none', 'hbb', 'obb'], 'mask_margin_ratio': [0.0, 0.3]}
        specs = {spec.name: spec for spec in build_pipelines({**config, 'pipeline': grid})}
        share = {}
        for name, spec in specs.items():
            records = run_pipeline(spec, CASE_DIR, pairs[:2], references, config)['records']
            share[name] = (records[0]['suppressed_ref'], records[0]['suppressed_cur'])
        none, hbb, obb = (share[f"orb+{mode}"] for mode in ('none', 'hbb', 'obb'))
        hbb_wide, obb_wide = share['orb+hbb#2'], share['orb+obb#2']
        assert none == (0.0, 0.0)
        assert 0.0 < obb[0] <= hbb[0] and 0.0 < obb[1] <= hbb[1]  # an envelope covers its rotated box
        assert obb_wide[0] > obb[0] and hbb_wide[0] > hbb[0]

    def test_the_reference_baseline_agrees_with_itself(self, sample):
        config, pairs, references, spec = sample
        records = evaluate_baseline(BASELINE_REFERENCE, CASE_DIR, pairs, references, config)['records']
        assert all(r['cte_mean'] == pytest.approx(0.0, abs=1e-9) for r in records if not r['failed'])
        assert all(r['iou_mean'] == pytest.approx(1.0) for r in records if not r['failed'])
        captured = [r['capture_fraction'] for r in records if r['capture_fraction'] == r['capture_fraction']]
        assert captured and captured == pytest.approx([0.0] * len(captured), abs=1e-9)


@pytest.mark.slow
class TestMovingForeground:
    def test_every_measurement_records_the_pair_s_moving_foreground(self, sample, shared_run):
        config, pairs, references, spec = sample
        records = shared_run(1)['records']
        baseline = evaluate_baseline(BASELINE_IDENTITY, CASE_DIR, pairs, references, config)['records']
        for name in dynamic_runner.MOTION_FIELDS:
            assert [r[name] for r in records] == [r[name] for r in baseline], name
        for record in records:
            assert 0.0 <= record['coherent_share'] <= record['moving_share'] <= 1.0
            assert record['n_coherent'] <= record['n_moving']

    def test_masking_keeps_inliers_off_the_objects(self, sample):
        config, pairs, references, _ = sample
        grid = {'preset': ['orb'], 'masking': ['none', 'hbb', 'obb']}
        shares = {}
        for spec in build_pipelines({**config, 'pipeline': grid}):
            records = run_pipeline(spec, CASE_DIR, pairs, references, config)['records']
            shares[spec.masking] = [r['inliers_on_objects'] for r in records]
        assert min(shares['none']) > 0.05
        assert max(shares['obb']) == 0.0
        # The share is read on the oriented boxes, and an oriented box can reach past the horizontal
        # box it was converted from, so an HBB-masked estimate may keep an inlier or two there.
        assert max(shares['hbb']) < 0.01

    def test_a_failed_pair_has_no_capture_fraction(self, sample, monkeypatch):
        monkeypatch.setattr(dynamic_runner, 'estimation_failed', lambda *_: True)
        records = _run(sample, 1)['records']
        assert all(r['capture_fraction'] != r['capture_fraction'] for r in records)

    def test_objects_are_indexed_where_they_lie(self):
        labels = Labels(
            classes=np.zeros(2, dtype=int), hbb=np.array([[20.0, 20.0, 10.0, 10.0], [60.0, 20.0, 10.0, 10.0]]), obb=None
        )
        index = dynamic_runner.object_index_map(labels, (40, 80))
        assert index[20, 20] == 0 and index[20, 60] == 1 and index[5, 5] == -1


class TestConfigSurface:
    def test_the_shipped_dynamic_configs_declare_repeats(self):
        # `--set` can only change keys the config already has.
        for name in ('offline_masking', 'realtime_masking'):
            config = yaml.safe_load(
                (REPO_ROOT / 'configs' / 'songdo' / 'dynamic' / f"{name}.yaml").read_text(encoding='utf-8')
            )
            assert config.get('repeats') == 1


@pytest.mark.slow
class TestResourceInfeasibleRuns:
    """A resource error part way through a case aborts the run instead of keeping its partial records."""

    def test_an_allocator_failure_aborts_the_run_and_keeps_nothing(self, sample, monkeypatch):
        config, pairs, references, spec = sample
        original = dynamic_runner.load_frame
        seen = {'n': 0}

        def failing(case_dir, index):
            seen['n'] += 1
            if seen['n'] > 1:  # the reference frame loads, then the first pair exhausts memory
                raise RuntimeError('MPS backend out of memory (MPS allocated: 18.00 GB)')
            return original(case_dir, index)

        monkeypatch.setattr(dynamic_runner, 'load_frame', failing)
        result = run_pipeline(spec, CASE_DIR, pairs, references, config)
        assert result['status'] == RUN_INFEASIBLE
        assert result['records'] == []
        assert 'out of memory' in result['resource_error']['message']

    def test_an_ordinary_failure_still_fails_the_campaign(self, sample, monkeypatch):
        config, pairs, references, spec = sample
        monkeypatch.setattr(
            dynamic_runner, 'load_frame', lambda *a, **k: (_ for _ in ()).throw(ValueError('unreadable frame'))
        )
        with pytest.raises(ValueError, match='unreadable frame'):
            run_pipeline(spec, CASE_DIR, pairs, references, config)

    def test_a_healthy_run_is_marked_completed(self, shared_run):
        assert shared_run(1)['status'] == RUN_COMPLETED

    def test_every_record_carries_the_work_proxy(self, sample, shared_run):
        result = shared_run(1)
        size = result['frame_size']
        ratio = sample[3].params.get('downsample_ratio', 1.0)
        expected = round(size['width'] * ratio) * round(size['height'] * ratio)
        assert all(record['processed_pixels'] == expected for record in result['records'])


class TestCachedOracle:
    def test_the_cache_matches_only_the_oracle_the_config_names(self, monkeypatch):
        config = load_config(SMOKE_CONFIG)
        cached = build_oracle(config).to_dict()
        monkeypatch.setattr(dynamic_runner, 'load_oracle', lambda case_dir: {'pipeline': dict(cached)})
        assert dynamic_runner.cached_oracle_matches(CASE_DIR, config)
        monkeypatch.setattr(
            dynamic_runner, 'load_oracle', lambda case_dir: {'pipeline': {**cached, 'filter_ratio': 0.123}}
        )
        assert not dynamic_runner.cached_oracle_matches(CASE_DIR, config)


class TestEvaluationMask:
    SHAPE = (120, 160)
    POLYGON = [[0, 0], [80, 0], [80, 60], [0, 60]]

    @staticmethod
    def _config(**region):
        return {'_path': str(SMOKE_CONFIG), 'metrics': {'region': {'mask_dilation_px': 10, **region}}}

    def test_without_files_nothing_is_restricted(self, tmp_path):
        mask, record = load_evaluation_mask(self._config(ground_plane_roi=None, exclusion=None), tmp_path, self.SHAPE)
        assert mask is None and record == NO_EVALUATION_MASK

    def test_the_exclusion_is_read_from_the_case_and_recorded(self, tmp_path):
        (tmp_path / 'EXCLUSION.json').write_text(
            json.dumps({'shapes': [{'label': 'building', 'shape_type': 'polygon', 'points': self.POLYGON}]}),
            encoding='utf-8',
        )
        mask, record = load_evaluation_mask(self._config(exclusion='EXCLUSION.json'), tmp_path, self.SHAPE)
        assert not mask[30, 40] and mask[100, 140]
        assert record['exclusion']['file'] == 'EXCLUSION.json'
        assert len(record['exclusion']['sha256']) == 64
        assert record['exclusion']['excluded_fraction'] == pytest.approx(float((~mask).mean()), abs=1e-6)

    def test_a_configured_exclusion_missing_from_a_case_raises(self, tmp_path):
        with pytest.raises(RegionError, match='not found'):
            load_evaluation_mask(self._config(exclusion='EXCLUSION.json'), tmp_path, self.SHAPE)

    def test_a_resume_under_a_different_mask_raises(self, tmp_path):
        changed = {'ground_plane_roi': None, 'exclusion': {'file': 'EXCLUSION.json', 'sha256': '0' * 64}}
        check_evaluation_mask({'evaluation_mask': changed}, changed, tmp_path)
        check_evaluation_mask({}, NO_EVALUATION_MASK, tmp_path)  # a manifest from before the record existed
        with pytest.raises(RegionError, match='Cannot resume'):
            check_evaluation_mask({'evaluation_mask': NO_EVALUATION_MASK}, changed, tmp_path)

    def test_the_evaluation_campaigns_exclude_the_case_s_polygons(self):
        for name in ('offline_masking', 'offline_margin', 'realtime_masking'):
            config = load_config(REPO_ROOT / 'configs' / 'songdo' / 'dynamic' / f"{name}.yaml")
            assert config['metrics']['region']['exclusion'] == 'EXCLUSION.json'
