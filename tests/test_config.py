#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Config loading, validation, grid expansion, overrides, and pipeline construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from stabilo_benchmark.config import (
    ConfigError,
    apply_overrides,
    build_oracle,
    build_pipelines,
    config_hash,
    expand_grid,
    load_config,
    run_key,
    validate_config,
)
from stabilo_benchmark.pipelines import PipelineError, load_presets, make_spec, stabilo_defaults

CONFIG_ROOT = Path(__file__).resolve().parent.parent / 'configs'

DYNAMIC_CAMPAIGNS = ('offline_tuning', 'offline_masking', 'offline_margin', 'realtime_tuning', 'realtime_masking')
BUDGET_KEYS = ('downsample_ratio', 'max_features', 'ransac_threshold_space')


def dynamic_config(name):
    return load_config(CONFIG_ROOT / 'songdo' / 'dynamic' / f"{name}.yaml")


STATIC_CONFIG = {
    'kind': 'static',
    'name': 'unit',
    'seed': 0,
    'trials': 2,
    'scenes': {'dir': 'scenes'},
    'perturbation': {'geometric': {'rotation_deg': 5.0}, 'photometric': [{'name': 'blur', 'max_ksize': 3}]},
    'pipeline': {'preset': ['orb'], 'masking': ['none', 'obb']},
}

DYNAMIC_CONFIG = {
    'kind': 'dynamic',
    'name': 'unit',
    'seed': 0,
    'sampling': {'mode': 'grid', 'stride': 10},
    'oracle': {'preset': 'oracle', 'masking': 'obb'},
    'pipeline': {'preset': ['sift'], 'masking': ['none']},
}


def write(tmp_path, payload, name='config.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding='utf-8')
    return path


class TestLoading:
    def test_yaml_loads_and_records_provenance(self, tmp_path):
        config = load_config(write(tmp_path, STATIC_CONFIG))
        assert config['kind'] == 'static'
        assert config['_path'].endswith('config.yaml')

    def test_json_loads_too(self, tmp_path):
        # JSON configs load too.
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(STATIC_CONFIG), encoding='utf-8')
        assert load_config(path)['kind'] == 'static'

    def test_name_defaults_to_the_file_stem(self, tmp_path):
        payload = {key: value for key, value in STATIC_CONFIG.items() if key != 'name'}
        assert load_config(write(tmp_path, payload, 'my_campaign.yaml'))['name'] == 'my_campaign'

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match='not found'):
            load_config(tmp_path / 'absent.yaml')

    def test_malformed_yaml_raises(self, tmp_path):
        path = tmp_path / 'bad.yaml'
        path.write_text('kind: [unclosed\n', encoding='utf-8')
        with pytest.raises(ConfigError, match='Invalid YAML'):
            load_config(path)

    def test_non_mapping_raises(self, tmp_path):
        path = tmp_path / 'list.yaml'
        path.write_text('- a\n- b\n', encoding='utf-8')
        with pytest.raises(ConfigError, match='mapping at the top level'):
            load_config(path)


class TestHash:
    def test_ignores_provenance_keys(self, tmp_path):
        # Moving a config file must not change the hash of its results.
        a = load_config(write(tmp_path, STATIC_CONFIG, 'a.yaml'))
        b = dict(a, _path='/elsewhere/a.yaml')
        assert config_hash(a) == config_hash(b)

    def test_changes_with_content(self, tmp_path):
        a = load_config(write(tmp_path, STATIC_CONFIG, 'a.yaml'))
        b = dict(a, trials=99)
        assert config_hash(a) != config_hash(b)


class TestShippedConfigs:
    """Every campaign template in `configs/` loads, validates and expands to its run count."""

    @pytest.mark.parametrize('path', sorted(CONFIG_ROOT.rglob('*.yaml')), ids=lambda p: p.name)
    def test_it_loads_validates_and_expands(self, path):
        config = load_config(path)
        validate_config(config)
        specs = build_pipelines(config)
        assert specs, f"{path.name} expands to no runs"
        assert len({spec.name for spec in specs}) == len(specs), f"{path.name} has duplicate run names"
        if config['kind'] == 'dynamic':
            assert build_oracle(config)

    def test_the_dynamic_campaigns_share_one_reference_estimator(self):
        # Every campaign of a case reads the same cached reference estimates.
        oracles = {name: build_oracle(dynamic_config(name)).to_dict() for name in DYNAMIC_CAMPAIGNS}
        assert all(oracle == oracles['offline_masking'] for oracle in oracles.values())

    def test_the_evaluation_campaigns_share_their_tuned_thresholds(self):
        # Part II's offline values are written into both; the margin sweep must run the same pipelines.
        keys = ('filter_ratio', 'ransac_epipolar_threshold')
        masking, margin = (
            {key: dynamic_config(name)['pipeline'].get(key) for key in keys}
            for name in ('offline_masking', 'offline_margin')
        )
        assert masking == margin

    def test_the_margin_sweep_samples_the_masking_campaign_s_pairs(self):
        # The margin benefit is read against the masking campaign's unmasked score of the same pair.
        masking, margin = (dynamic_config(name) for name in ('offline_masking', 'offline_margin'))
        assert margin['sampling'] == masking['sampling']
        assert margin['seed'] == masking['seed'] and margin['reference'] == masking['reference']
        assert margin['metrics'] == masking['metrics']

    def test_the_dynamic_campaigns_run_the_same_pipelines(self):
        presets = {name: dynamic_config(name)['pipeline']['preset'] for name in DYNAMIC_CAMPAIGNS}
        assert all(preset == presets['offline_masking'] for preset in presets.values())

    @pytest.mark.parametrize('campaign', ['tuning', 'masking'])
    def test_the_realtime_regime_differs_from_the_offline_one_only_in_its_budget(self, campaign):
        # Same pairs (so the audited frames are among them), metrics and reference; a smaller budget,
        # with the inlier threshold in processed-image pixels in both.
        offline, realtime = (dynamic_config(f"{regime}_{campaign}") for regime in ('offline', 'realtime'))
        for key in ('sampling', 'seed', 'reference', 'metrics', 'oracle', 'store_homographies'):
            assert realtime.get(key) == offline.get(key), key
        budgets = [{key: config['pipeline'].get(key) for key in BUDGET_KEYS} for config in (offline, realtime)]
        assert budgets == [
            {'downsample_ratio': 0.5, 'max_features': 2000, 'ransac_threshold_space': 'processed'},
            {'downsample_ratio': 0.18, 'max_features': 1000, 'ransac_threshold_space': 'processed'},
        ]
        assert realtime['pipeline']['masking'] == offline['pipeline']['masking']

    def test_both_budgets_are_points_of_the_budget_campaign_the_realtime_one_its_cheapest(self):
        budget = load_config(CONFIG_ROOT / 'songdo' / 'static' / 'budget.yaml')['pipeline']
        for name in DYNAMIC_CAMPAIGNS:
            pipeline = dynamic_config(name)['pipeline']
            assert pipeline['downsample_ratio'] in budget['downsample_ratio'], name
            assert pipeline['max_features'] in budget['max_features'], name
        realtime = dynamic_config('realtime_masking')['pipeline']
        assert realtime['downsample_ratio'] == min(budget['downsample_ratio'])
        assert realtime['max_features'] == min(budget['max_features'])

    def test_the_static_templates_are_all_present(self):
        names = {path.stem for path in (CONFIG_ROOT / 'songdo' / 'static').glob('*.yaml')}
        assert {'tuning', 'threshold_scaling', 'budget', 'detectors', 'estimators', 'rotation_sweep'} <= names


class TestValidation:
    def test_valid_configs_pass(self):
        validate_config(STATIC_CONFIG)
        validate_config(DYNAMIC_CONFIG)

    def test_bad_kind_is_named(self):
        with pytest.raises(ConfigError, match="'kind' must be one of"):
            validate_config(dict(STATIC_CONFIG, kind='sideways'))

    @pytest.mark.parametrize('key', ['seed', 'trials', 'scenes', 'perturbation', 'pipeline'])
    def test_missing_static_key_is_named(self, key):
        payload = {k: v for k, v in STATIC_CONFIG.items() if k != key}
        with pytest.raises(ConfigError, match=key):
            validate_config(payload)

    @pytest.mark.parametrize('key', ['sampling', 'oracle', 'pipeline'])
    def test_missing_dynamic_key_is_named(self, key):
        payload = {k: v for k, v in DYNAMIC_CONFIG.items() if k != key}
        with pytest.raises(ConfigError, match=key):
            validate_config(payload)

    @pytest.mark.parametrize(
        'bins, message',
        [
            ([[1, 15], [10, 30]], 'ascending and non-overlapping'),
            ([[0, 15]], '1 <= min_offset'),
            ([[20, 10]], '1 <= min_offset'),
            ([[1, 15, 30]], r'\[min_offset, max_offset\]'),
            ([], 'non-empty|required'),
        ],
    )
    @pytest.mark.parametrize('key', ['bins', 'audit_bins'])
    def test_malformed_offset_bins_are_named(self, key, bins, message):
        sampling = {'mode': 'stratified', 'bins': [[1, 15]], key: bins}
        with pytest.raises(ConfigError, match=f"sampling.{key}.*{message}|{message}"):
            validate_config(dict(DYNAMIC_CONFIG, sampling=sampling))

    def test_well_formed_offset_bins_pass(self):
        sampling = {'mode': 'stratified', 'bins': [[1, 15], [16, 30]], 'audit_bins': [[1, 30]]}
        validate_config(dict(DYNAMIC_CONFIG, sampling=sampling))

    def test_bool_is_not_accepted_as_an_int(self):
        with pytest.raises(ConfigError, match="'seed' must be of type int"):
            validate_config(dict(STATIC_CONFIG, seed=True))

    def test_zero_trials_rejected(self):
        with pytest.raises(ConfigError, match=">= 1"):
            validate_config(dict(STATIC_CONFIG, trials=0))

    def test_bad_masking_mode_is_named(self):
        payload = dict(STATIC_CONFIG, pipeline={'preset': ['orb'], 'masking': ['rotated']})
        with pytest.raises(ConfigError, match='rotated'):
            validate_config(payload)

    @pytest.mark.parametrize('key', ['labels', 'label_format', 'coords', 'min_confidence'])
    def test_single_file_label_keys_are_refused_by_name(self, key):
        # Single-file label keys raise instead of being ignored.
        payload = dict(STATIC_CONFIG, scenes={'dir': 'scenes', key: 'obb'})
        with pytest.raises(ConfigError, match=key):
            validate_config(payload)

    def test_stratified_sampling_requires_bins(self):
        payload = dict(DYNAMIC_CONFIG, sampling={'mode': 'stratified'})
        with pytest.raises(ConfigError, match='bins'):
            validate_config(payload)

    def test_negative_dilation_rejected(self):
        payload = dict(DYNAMIC_CONFIG, metrics={'region': {'mask_dilation_px': -1}})
        with pytest.raises(ConfigError, match='non-negative'):
            validate_config(payload)

    @pytest.mark.parametrize(
        'exclusion', ['/abs/EXCLUSION.json', '../EXCLUSION.json', 'C:\\x\\EXCLUSION.json', '..\\EXCLUSION.json', '', 3]
    )
    def test_an_exclusion_outside_the_case_is_rejected(self, exclusion):
        payload = dict(DYNAMIC_CONFIG, metrics={'region': {'exclusion': exclusion}})
        with pytest.raises(ConfigError, match='exclusion'):
            validate_config(payload)

    @pytest.mark.parametrize('value', ['obb', 'hbb'])
    def test_object_boxes_accepts_both_geometries(self, value):
        validate_config(dict(STATIC_CONFIG, metrics={'object_boxes': value}))

    def test_object_boxes_rejects_anything_else(self):
        with pytest.raises(ConfigError, match='object_boxes'):
            validate_config(dict(DYNAMIC_CONFIG, metrics={'object_boxes': 'polygon'}))

    def test_photometric_entry_needs_a_name(self):
        payload = dict(STATIC_CONFIG, perturbation={'photometric': [{'max_ksize': 3}]})
        with pytest.raises(ConfigError, match="'name' key"):
            validate_config(payload)

    def test_repeats_is_optional(self):
        validate_config(dict(DYNAMIC_CONFIG))

    @pytest.mark.parametrize('value', [0, -1])
    def test_repeats_below_one_is_rejected(self, value):
        with pytest.raises(ConfigError, match='repeats'):
            validate_config(dict(DYNAMIC_CONFIG, repeats=value))

    def test_repeats_must_be_an_integer(self):
        with pytest.raises(ConfigError, match='repeats'):
            validate_config(dict(DYNAMIC_CONFIG, repeats=2.5))


class TestGridExpansion:
    def test_cartesian_product_of_axes(self):
        runs = expand_grid({'preset': ['orb', 'sift'], 'masking': ['none', 'hbb', 'obb']})
        assert len(runs) == 6
        assert {run['preset'] for run in runs} == {'orb', 'sift'}

    def test_scalars_are_held_fixed(self):
        runs = expand_grid({'preset': ['orb', 'sift'], 'clahe': False})
        assert all(run['clahe'] is False for run in runs)

    def test_last_key_varies_fastest(self):
        runs = expand_grid({'a': [1, 2], 'b': [10, 20]})
        assert [(run['a'], run['b']) for run in runs] == [(1, 10), (1, 20), (2, 10), (2, 20)]

    def test_per_detector_mapping_is_resolved(self):
        runs = expand_grid({'detector_name': ['orb', 'sift'], 'ransac_epipolar_threshold': {'orb': 2.0, 'sift': 3.0}})
        assert {run['detector_name']: run['ransac_epipolar_threshold'] for run in runs} == {'orb': 2.0, 'sift': 3.0}

    def test_per_detector_mapping_resolves_through_a_preset(self):
        runs = expand_grid({'preset': ['sift'], 'ransac_max_iter': {'sift': 1234, 'orb': 5678}})
        assert runs[0]['ransac_max_iter'] == 1234

    def test_missing_detector_entry_raises(self):
        with pytest.raises(ConfigError, match='no entry for detector'):
            expand_grid({'detector_name': ['kaze'], 'ransac_epipolar_threshold': {'orb': 2.0}})


class TestOverrides:
    def test_sets_an_existing_scalar(self):
        config = apply_overrides(dict(STATIC_CONFIG), ['trials=50'])
        assert config['trials'] == 50

    def test_sets_a_nested_value(self):
        config = apply_overrides(json.loads(json.dumps(STATIC_CONFIG)), ['perturbation.geometric.rotation_deg=[30,30]'])
        assert config['perturbation']['geometric']['rotation_deg'] == [30, 30]

    def test_parses_lists_as_yaml(self):
        config = apply_overrides(json.loads(json.dumps(STATIC_CONFIG)), ['pipeline.masking=[obb]'])
        assert config['pipeline']['masking'] == ['obb']

    def test_a_typo_in_the_path_raises_rather_than_creating_a_key(self):
        with pytest.raises(ConfigError, match='not an existing key'):
            apply_overrides(dict(STATIC_CONFIG), ['perturbation.geometrik.rotation_deg=5'])

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ConfigError, match='not an existing key'):
            apply_overrides(dict(STATIC_CONFIG), ['trails=50'])

    def test_missing_equals_raises(self):
        with pytest.raises(ConfigError, match='key.path=value'):
            apply_overrides(dict(STATIC_CONFIG), ['trials'])

    def test_none_is_a_no_op(self):
        assert apply_overrides(dict(STATIC_CONFIG), None)['trials'] == 2


class TestPipelineConstruction:
    def test_grid_becomes_unique_named_specs(self):
        specs = build_pipelines(STATIC_CONFIG)
        assert [spec.name for spec in specs] == ['orb+none', 'orb+obb']
        assert len({spec.name for spec in specs}) == len(specs)

    def test_repeated_label_is_disambiguated(self):
        config = dict(STATIC_CONFIG, pipeline={'preset': ['orb'], 'masking': ['obb'], 'max_features': [1000, 2000]})
        names = [spec.name for spec in build_pipelines(config)]
        assert names == ['orb+obb', 'orb+obb#2']

    def test_a_matcher_without_the_ratio_test_runs_once_per_remaining_grid_point(self):
        grid = {'preset': ['sift', 'disk_lightglue'], 'filter_ratio': [0.7, 0.8]}
        grid['ransac_epipolar_threshold'] = [1.0, 2.0]
        specs = build_pipelines(dict(STATIC_CONFIG, pipeline=grid))
        lightglue = [spec for spec in specs if spec.matcher == 'lightglue']
        assert len(specs) == 6 and len(lightglue) == 2
        assert {spec.params['filter_ratio'] for spec in lightglue} == {0.9}
        assert {spec.params['ransac_epipolar_threshold'] for spec in lightglue} == {1.0, 2.0}

    def test_runs_stored_at_every_ratio_match_the_collapsed_run_on_regrid(self):
        def run(preset, ratio):
            return make_spec({'preset': preset, 'masking': 'none', 'filter_ratio': ratio}).to_dict()

        assert run_key(run('disk_lightglue', 0.7)) == run_key(run('disk_lightglue', 0.9))
        assert run_key(run('sift', 0.7)) != run_key(run('sift', 0.9))
        assert run_key({'pipeline': 'identity', 'detector_name': 'identity', 'baseline': True})

    def test_mask_use_follows_the_masking_axis(self):
        specs = {spec.masking: spec for spec in build_pipelines(STATIC_CONFIG)}
        assert specs['none'].stabilo_kwargs()['mask_use'] is False
        assert specs['obb'].stabilo_kwargs()['mask_use'] is True

    def test_oracle_is_built_from_its_own_block(self):
        spec = build_oracle(DYNAMIC_CONFIG)
        assert spec.masking == 'obb'
        assert spec.params['downsample_ratio'] == 1.0

    def test_unknown_stabilo_key_is_rejected_not_silently_dropped(self):
        # Stabilizer ignores unknown kwargs, so the spec check must reject them.
        with pytest.raises(PipelineError, match='unknown to the installed stabilo'):
            make_spec({'preset': 'orb', 'masking': 'obb', 'ransac_epipolar_threshhold': 2.0})

    def test_setting_mask_use_directly_is_rejected(self):
        with pytest.raises(PipelineError, match="use the benchmark-level 'masking' axis"):
            make_spec({'preset': 'orb', 'masking': 'obb', 'mask_use': False})

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(PipelineError, match='Unknown pipeline preset'):
            make_spec({'preset': 'sifty', 'masking': 'none'})

    def test_grid_keys_override_the_preset(self):
        spec = make_spec({'preset': 'orb', 'masking': 'none', 'max_features': 9999})
        assert spec.params['max_features'] == 9999


class TestPresets:
    def test_every_shipped_preset_is_valid(self):
        for name in load_presets():
            make_spec({'preset': name, 'masking': 'none'})

    def test_inheritance_is_resolved(self):
        presets = load_presets()
        assert presets['sift']['detector_name'] == 'sift'
        assert presets['sift']['ransac_method'] == presets['base']['ransac_method']

    def test_presets_only_use_keys_stabilo_knows(self):
        known = set(stabilo_defaults())
        for name, params in load_presets().items():
            unknown = set(params) - known
            assert not unknown, f"preset '{name}' sets unknown key(s) {unknown}"
