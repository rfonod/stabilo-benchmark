#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
config.py - Experiment configuration: loading, validation, and grid expansion.

Every quantity that shapes a reported number is a config key, so a result can be reproduced from
its manifest. Configs are YAML (JSON is accepted as a subset). Any list-valued key under
`pipeline` is a grid axis, and the Cartesian product of the axes is the run list. A value may
also be a mapping keyed by detector name, resolved per run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from stabilo_benchmark.pipelines import (
    DEFAULT_OBJECT_BOXES,
    MASKING_MODES,
    OBJECT_BOX_GEOMETRIES,
    PipelineSpec,
    ignores_ratio_test,
    load_presets,
    make_spec,
    stabilo_defaults,
)

VALID_KINDS = ('static', 'dynamic')
VALID_REFERENCE_MODES = ('first', 'middle')
VALID_SAMPLING_MODES = ('grid', 'random', 'stratified')


class ConfigError(ValueError):
    """Raised when a configuration file is missing a key, or a key holds a bad value."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    """
    Read a YAML (or JSON) experiment config and add two derived keys:
      `_path`: absolute path of the config file
      `name` : the file stem, when the config does not set one
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML/JSON in {path}:\n{e}") from e

    if not isinstance(config, dict):
        raise ConfigError(f"Configuration file must contain a mapping at the top level: {path}")

    config = dict(config)
    config['_path'] = str(path)
    config.setdefault('name', path.stem)
    return config


def apply_overrides(config: Dict[str, Any], overrides: Optional[Sequence[str]]) -> Dict[str, Any]:
    """
    Apply `--set key.path=value` overrides to a loaded config.

    Values are parsed as YAML (`--set trials=200`, `--set pipeline.masking='[obb]'`). Overrides are
    applied before validation and are part of the hashed config. A path to a key that does not exist
    raises.
    """
    if not overrides:
        return config

    for override in overrides:
        if '=' not in override:
            raise ConfigError(f"--set expects key.path=value, got {override!r}")
        path, raw_value = override.split('=', 1)
        keys = path.strip().split('.')
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as e:
            raise ConfigError(f"Could not parse value in --set {override!r}: {e}") from e

        target: Any = config
        for key in keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise ConfigError(f"--set {path}: '{key}' is not an existing key in the config")
            target = target[key]
        if not isinstance(target, dict) or keys[-1] not in target:
            raise ConfigError(f"--set {path}: '{keys[-1]}' is not an existing key in the config")
        target[keys[-1]] = value

    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """
    Stable SHA-256 over the content of a config.

    Keys beginning with '_' (added by the loader) are excluded.
    """
    payload = {k: v for k, v in config.items() if not k.startswith('_')}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

_COMMON_REQUIRED = ('kind', 'seed', 'pipeline')
_STATIC_REQUIRED = ('trials', 'scenes', 'perturbation')
_DYNAMIC_REQUIRED = ('sampling', 'oracle')


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate a config, raising `ConfigError` naming the offending key. Values are not coerced."""
    kind = config.get('kind')
    if kind not in VALID_KINDS:
        raise ConfigError(f"'kind' must be one of {list(VALID_KINDS)}, got {kind!r}")

    required = _COMMON_REQUIRED + (_STATIC_REQUIRED if kind == 'static' else _DYNAMIC_REQUIRED)
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Config '{config.get('name', '?')}' ({kind}) is missing required key(s): {missing}")

    _require_type(config, 'seed', int)
    _require_type(config, 'pipeline', dict)

    if kind == 'static':
        _validate_static(config)
    else:
        _validate_dynamic(config)

    _validate_metrics(config.get('metrics', {}) or {})
    _validate_pipeline_grid(config['pipeline'])


def _validate_static(config: Mapping[str, Any]) -> None:
    _require_type(config, 'trials', int)
    if config['trials'] < 1:
        raise ConfigError(f"'trials' must be >= 1, got {config['trials']}")

    scenes = config['scenes']
    if not isinstance(scenes, dict):
        raise ConfigError("'scenes' must be a mapping with at least a 'dir' key")
    if 'dir' not in scenes:
        raise ConfigError("'scenes.dir' is required (directory of static aerial images)")
    retired = sorted({'labels', 'label_format', 'coords', 'min_confidence'} & set(scenes))
    if retired:
        raise ConfigError(
            f"'scenes.{retired[0]}' is not supported: labels are read from <scenes.dir>/labels_hbb and "
            f"<scenes.dir>/labels_obb, as hbb2obb writes them. Remove {', '.join(retired)}."
        )

    perturbation = config['perturbation']
    if not isinstance(perturbation, dict):
        raise ConfigError("'perturbation' must be a mapping with 'geometric' and/or 'photometric' keys")
    geometric = perturbation.get('geometric', {}) or {}
    if not isinstance(geometric, dict):
        raise ConfigError("'perturbation.geometric' must be a mapping of parameter name to range")
    photometric = perturbation.get('photometric', []) or []
    if not isinstance(photometric, list):
        raise ConfigError("'perturbation.photometric' must be a list of {name: ..., ...} entries, applied in order")
    for i, entry in enumerate(photometric):
        if not isinstance(entry, dict) or 'name' not in entry:
            raise ConfigError(f"'perturbation.photometric[{i}]' must be a mapping containing a 'name' key")


def _validate_dynamic(config: Mapping[str, Any]) -> None:
    reference = config.get('reference', 'first')
    if not (reference in VALID_REFERENCE_MODES or isinstance(reference, int)):
        raise ConfigError(
            f"'reference' must be one of {list(VALID_REFERENCE_MODES)} or a frame index, got {reference!r}"
        )

    if 'repeats' in config:
        _require_type(config, 'repeats', int)
        if config['repeats'] < 1:
            raise ConfigError(f"'repeats' must be >= 1, got {config['repeats']}")

    sampling = config['sampling']
    if not isinstance(sampling, dict):
        raise ConfigError("'sampling' must be a mapping with at least a 'mode' key")
    mode = sampling.get('mode', 'grid')
    if mode not in VALID_SAMPLING_MODES:
        raise ConfigError(f"'sampling.mode' must be one of {list(VALID_SAMPLING_MODES)}, got {mode!r}")
    if mode == 'grid' and int(sampling.get('stride', 0)) < 1:
        raise ConfigError("'sampling.stride' must be >= 1 for sampling.mode 'grid'")
    if mode == 'stratified' and not sampling.get('bins'):
        raise ConfigError(
            "'sampling.bins' (list of [min_offset, max_offset] pairs) is required for sampling.mode 'stratified'"
        )
    for key in ('bins', 'survey_bins', 'audit_bins'):
        if key in sampling:
            _validate_offset_bins(f"sampling.{key}", sampling[key])

    oracle = config['oracle']
    if not isinstance(oracle, dict):
        raise ConfigError("'oracle' must be a mapping describing the reference estimator pipeline")
    if 'preset' not in oracle and 'detector_name' not in oracle:
        raise ConfigError("'oracle' must name a 'preset' or set 'detector_name' explicitly")
    oracle_masking = oracle.get('masking', 'obb')
    if oracle_masking not in MASKING_MODES:
        raise ConfigError(f"'oracle.masking' must be one of {list(MASKING_MODES)}, got {oracle_masking!r}")


def _validate_offset_bins(name: str, bins: Any) -> None:
    """Offset bins are ascending, non-overlapping [low, high] frame pairs with 1 <= low <= high."""
    if not isinstance(bins, list) or not bins:
        raise ConfigError(f"'{name}' must be a non-empty list of [min_offset, max_offset] pairs")
    previous = 0
    for index, bounds in enumerate(bins):
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, int) for b in bounds)):
            raise ConfigError(f"'{name}[{index}]' must be [min_offset, max_offset] in frames, got {bounds!r}")
        low, high = bounds
        if not 1 <= low <= high:
            raise ConfigError(f"'{name}[{index}]' must satisfy 1 <= min_offset <= max_offset, got {bounds!r}")
        if low <= previous:
            raise ConfigError(f"'{name}' must be ascending and non-overlapping; bin {index} starts at {low}")
        previous = high


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    thresholds = metrics.get('hea_thresholds', [])
    if thresholds and not all(isinstance(t, (int, float)) and t > 0 for t in thresholds):
        raise ConfigError(f"'metrics.hea_thresholds' must be positive numbers, got {thresholds!r}")

    object_boxes = metrics.get('object_boxes', DEFAULT_OBJECT_BOXES)
    if object_boxes not in OBJECT_BOX_GEOMETRIES:
        raise ConfigError(f"'metrics.object_boxes' must be one of {list(OBJECT_BOX_GEOMETRIES)}, got {object_boxes!r}")

    region = metrics.get('region', {}) or {}
    if not isinstance(region, dict):
        raise ConfigError("'metrics.region' must be a mapping")
    dilation = region.get('mask_dilation_px', 0)
    if not isinstance(dilation, int) or dilation < 0:
        raise ConfigError(f"'metrics.region.mask_dilation_px' must be a non-negative integer, got {dilation!r}")
    exclusion = region.get('exclusion')
    # Both path flavors, by anchor: on Windows '/abs' is rooted but not absolute, on POSIX 'C:\x' is a name.
    inside_case = (
        isinstance(exclusion, str)
        and bool(exclusion)
        and not any(
            flavor(exclusion).anchor or '..' in flavor(exclusion).parts for flavor in (PurePosixPath, PureWindowsPath)
        )
    )
    if exclusion is not None and not inside_case:
        raise ConfigError(
            f"'metrics.region.exclusion' must be null or a file name inside each case directory, got {exclusion!r}"
        )


def _validate_pipeline_grid(grid: Mapping[str, Any]) -> None:
    if not grid:
        raise ConfigError("'pipeline' must define at least one key (e.g. 'preset' or 'detector_name')")
    masking = grid.get('masking', 'none')
    values = masking if isinstance(masking, list) else [masking]
    bad = [v for v in values if v not in MASKING_MODES]
    if bad:
        raise ConfigError(f"'pipeline.masking' contains invalid mode(s) {bad}; choose from {list(MASKING_MODES)}")


def _require_type(config: Mapping[str, Any], key: str, expected: type) -> None:
    """Type check that does not let Python's bool-is-an-int equivalence through."""
    value = config.get(key)
    if (isinstance(value, bool) and expected is not bool) or not isinstance(value, expected):
        raise ConfigError(f"'{key}' must be of type {expected.__name__}, got {type(value).__name__} ({value!r})")


# --------------------------------------------------------------------------------------
# Grid expansion
# --------------------------------------------------------------------------------------


def expand_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a pipeline grid into concrete parameter dicts (Cartesian product).

    List-valued keys are axes; scalars are held fixed. A mapping keyed by detector name is resolved
    after the product against the run's detector:

        ransac_epipolar_threshold: {orb: 2.0, sift: 3.0}

    Keys are iterated in file order with the last varying fastest, so a run number always identifies
    the same parameter set for a given config.
    """
    axes: List[Tuple[str, Sequence[Any]]] = []
    for key, value in grid.items():
        axes.append((key, value if isinstance(value, list) else [value]))

    runs: List[Dict[str, Any]] = []
    for combination in itertools.product(*[values for _, values in axes]):
        params = {key: value for (key, _), value in zip(axes, combination, strict=True)}
        runs.append(_resolve_per_detector(params))
    return runs


def _resolve_per_detector(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace any mapping-valued parameter with its entry for this run's detector.

    The detector comes from the grid, or from the preset when the grid does not set `detector_name`.
    """
    if not any(isinstance(v, dict) for v in params.values()):
        return params

    detector = params.get('detector_name')
    if detector is None and params.get('preset') is not None:
        presets = load_presets()
        detector = presets.get(str(params['preset']), {}).get('detector_name')
    if detector is None:
        detector = stabilo_defaults()['detector_name']

    resolved: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, dict):
            if detector not in value:
                raise ConfigError(
                    f"'pipeline.{key}' is a per-detector mapping but has no entry for detector '{detector}'. "
                    f"Available: {sorted(value)}"
                )
            resolved[key] = value[detector]
        else:
            resolved[key] = value
    return resolved


def build_pipelines(config: Mapping[str, Any], preset_file: Optional[Path] = None) -> List[PipelineSpec]:
    """Expand the config's `pipeline` grid into validated `PipelineSpec` objects."""
    presets = load_presets(preset_file)
    specs: List[PipelineSpec] = []
    seen: Dict[str, int] = {}
    collapsed = set()
    for params in expand_grid(config['pipeline']):
        spec = make_spec(params, presets)
        if 'filter_ratio' in params and ignores_ratio_test(spec):
            # Every ratio point is the same pipeline: run it once, at the preset's own value.
            spec = make_spec({k: v for k, v in params.items() if k != 'filter_ratio'}, presets)
            key = (spec.name, spec.masking, json.dumps(spec.params, sort_keys=True, default=str))
            if key in collapsed:
                continue
            collapsed.add(key)
        # Suffix repeated names (`#2`, `#3`) so every run has a unique label.
        label = f"{spec.name}+{spec.masking}"
        count = seen.get(label, 0)
        seen[label] = count + 1
        name = label if count == 0 else f"{label}#{count + 1}"
        specs.append(PipelineSpec(name=name, masking=spec.masking, params=spec.params))
    return specs


def run_key(run: Mapping[str, Any]) -> str:
    """
    A manifest run's identity when a grid is extended: its parameters without the label, and without
    `filter_ratio` where the pipeline ignores it, so runs stored at every ratio match the single
    run `build_pipelines` now makes of them.
    """
    key = {k: v for k, v in run.items() if k != 'pipeline'}
    params = {k: v for k, v in key.items() if k != 'masking'}
    if ignores_ratio_test(PipelineSpec(name=str(run.get('pipeline')), masking=str(run.get('masking')), params=params)):
        key.pop('filter_ratio', None)
    return json.dumps(key, sort_keys=True, default=str)


def build_oracle(config: Mapping[str, Any], preset_file: Optional[Path] = None) -> PipelineSpec:
    """Build the reference-estimator pipeline for a dynamic config."""
    oracle = dict(config['oracle'])
    oracle.setdefault('masking', 'obb')
    spec = make_spec(oracle, load_presets(preset_file))
    return PipelineSpec(name=f"oracle:{spec.name}+{spec.masking}", masking=spec.masking, params=spec.params)


def resolve_path(config: Mapping[str, Any], value: Any) -> Path:
    """Resolve a config-relative path against the directory holding the config file."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    base = Path(str(config['_path'])).parent if '_path' in config else Path.cwd()
    return (base / path).resolve()
