#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
pipelines.py - Registration pipeline specifications.

A pipeline is the full registration chain: detector, descriptor, matcher, robust estimator and
masking strategy. This module turns a pipeline description (a config grid point or a named
preset) into `stabilo.Stabilizer` kwargs and rejects specs stabilo would mishandle:

1. `Stabilizer` ignores unknown kwargs, so `validate_spec` rejects keys absent from stabilo's
   `default.yaml`.
2. `check_resolution` warns when a learning-based detector exceeds its pixel budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import yaml
from stabilo import Stabilizer

# Re-exported for callers of this module.
from stabilo_benchmark.environment import UNBOUND_IN_OPENCV5, opencv_binding_hint  # noqa: F401

# Masking strategy -> the `box_format` string understood by stabilo. 'none' passes no boxes.
MASKING_MODES = ('none', 'hbb', 'obb')
MASKING_BOX_FORMAT: Dict[str, Optional[str]] = {'none': None, 'hbb': 'xywh', 'obb': 'four'}

# Boxes IoU and OCE are computed on (`metrics.object_boxes`), independent of the masking axis.
OBJECT_BOX_GEOMETRIES = ('obb', 'hbb')
DEFAULT_OBJECT_BOXES = 'obb'

# Presets shipped with the package; a config may point elsewhere with `preset_file`.
DEFAULT_PRESET_FILE = Path(__file__).parent / 'cfg' / 'pipelines.yaml'


class PipelineError(ValueError):
    """Raised when a pipeline specification is invalid or unsupported."""


@lru_cache(maxsize=1)
def stabilo_defaults() -> Dict[str, Any]:
    """stabilo's `default.yaml` as a dict: the set of configurable keys."""
    try:
        from stabilo.stabilo import cfg  # noqa: PLC0415 - private module path, guarded by the fallback below

        return dict(cfg)
    except Exception:  # pragma: no cover - only hit if stabilo relocates its config
        import stabilo  # noqa: PLC0415 - only needed on the fallback path

        cfg_path = Path(stabilo.__file__).parent / 'cfg' / 'default.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return dict(yaml.safe_load(f))


@dataclass(frozen=True)
class PipelineSpec:
    """
    A fully resolved registration pipeline.

    Attributes:
        name: Identifier used in results and figures.
        masking: One of MASKING_MODES.
        params: Keyword arguments passed to `Stabilizer(**params)`; never contains `masking`.
    """

    name: str
    masking: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def box_format(self) -> Optional[str]:
        """The `box_format` string to pass alongside boxes, or None when masking is off."""
        return MASKING_BOX_FORMAT[self.masking]

    @property
    def detector(self) -> str:
        return str(self.params.get('detector_name', stabilo_defaults()['detector_name']))

    @property
    def matcher(self) -> str:
        return str(self.params.get('matcher_name', stabilo_defaults()['matcher_name']))

    def stabilo_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """
        Build the kwargs for `Stabilizer(...)`.

        `mask_use` is set from the masking mode.
        """
        kwargs = dict(self.params)
        kwargs['mask_use'] = self.masking != 'none'
        kwargs.update(overrides)
        return kwargs

    def build(self, **overrides: Any) -> Stabilizer:
        """Instantiate a `Stabilizer` for this pipeline. Benchmark mode is always on."""
        kwargs = self.stabilo_kwargs(**overrides)
        kwargs.setdefault('benchmark', True)
        return Stabilizer(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Flat, JSON-serializable description used in result records and manifests."""
        return {'pipeline': self.name, 'masking': self.masking, **self.params}


def load_presets(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load named pipeline presets from a YAML file.

    A preset maps a name to Stabilizer kwargs and may inherit from another with
    `extends: <other preset>`. Inheritance is resolved here.
    """
    path = Path(path) if path is not None else DEFAULT_PRESET_FILE
    if not path.exists():
        raise PipelineError(f"Pipeline preset file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise PipelineError(f"Pipeline preset file must contain a mapping of names to parameters: {path}")

    resolved: Dict[str, Dict[str, Any]] = {}

    def resolve(name: str, seen: Iterable[str] = ()) -> Dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name in seen:
            raise PipelineError(f"Circular 'extends' chain in pipeline presets: {' -> '.join([*seen, name])}")
        if name not in raw:
            raise PipelineError(f"Unknown pipeline preset '{name}'. Available: {sorted(raw)}")
        entry = dict(raw[name] or {})
        parent = entry.pop('extends', None)
        merged = dict(resolve(str(parent), (*seen, name))) if parent else {}
        merged.update(entry)
        resolved[name] = merged
        return merged

    for name in raw:
        resolve(name)
    return resolved


def make_spec(
    params: Dict[str, Any],
    presets: Optional[Dict[str, Dict[str, Any]]] = None,
    name: Optional[str] = None,
) -> PipelineSpec:
    """
    Turn one resolved grid point into a `PipelineSpec`.

    `params` may contain the benchmark-level keys `preset` and `masking`; everything
    else must be a valid stabilo configuration key. Keys given alongside `preset`
    override the preset's values.
    """
    params = dict(params)
    preset_name = params.pop('preset', None)
    masking = str(params.pop('masking', 'none'))

    base: Dict[str, Any] = {}
    if preset_name is not None:
        presets = presets if presets is not None else load_presets()
        if preset_name not in presets:
            raise PipelineError(f"Unknown pipeline preset '{preset_name}'. Available: {sorted(presets)}")
        base = dict(presets[preset_name])
    base.update(params)

    if name is None:
        name = str(preset_name) if preset_name is not None else _derive_name(base)

    spec = PipelineSpec(name=name, masking=masking, params=base)
    validate_spec(spec)
    return spec


def _derive_name(params: Dict[str, Any]) -> str:
    """Build a compact fallback name from the axes the paper varies."""
    defaults = stabilo_defaults()
    detector = params.get('detector_name', defaults['detector_name'])
    matcher = params.get('matcher_name', defaults['matcher_name'])
    ransac = params.get('ransac_method', defaults['ransac_method'])
    return f"{detector}-{matcher}-{ransac_name(int(ransac)).lower()}"


def ransac_name(method: int) -> str:
    """
    Name of an OpenCV robust-estimation flag, with the algorithm it runs in parentheses.

    Follows OpenCV's USAC tutorial: USAC_DEFAULT, USAC_PARALLEL and USAC_FAST are LO-RANSAC
    variants, USAC_ACCURATE is GC-RANSAC. None of them is DEGENSAC.
    """
    names = {
        4: 'LMEDS',
        8: 'RANSAC',
        16: 'RHO',
        32: 'USAC-Default (LO-RANSAC)',
        33: 'USAC-Parallel (LO-RANSAC)',
        35: 'USAC-Fast (LO-RANSAC)',
        36: 'USAC-Accurate (GC-RANSAC)',
        37: 'USAC-PROSAC (PROSAC)',
        38: 'USAC-MAGSAC (MAGSAC++)',
    }
    return names.get(method, str(method))


def validate_spec(spec: PipelineSpec) -> None:
    """
    Reject specs with an invalid masking mode, keys unknown to the installed stabilo, or `mask_use`.

    Enum values are validated by `Stabilizer` itself.
    """
    if spec.masking not in MASKING_MODES:
        raise PipelineError(
            f"Invalid masking mode '{spec.masking}' for pipeline '{spec.name}'. Choose from {list(MASKING_MODES)}"
        )

    known = set(stabilo_defaults())
    unknown = sorted(set(spec.params) - known - {'logger'})
    if unknown:
        raise PipelineError(
            f"Pipeline '{spec.name}' sets configuration key(s) unknown to the installed stabilo: {unknown}. "
            f"Stabilizer would ignore them silently. Valid keys: {sorted(known)}"
        )

    if 'mask_use' in spec.params:
        raise PipelineError(
            f"Pipeline '{spec.name}' sets 'mask_use' directly; use the benchmark-level 'masking' axis "
            f"({list(MASKING_MODES)}) instead so the mask geometry and the stabilo flag cannot disagree."
        )


def resource_cost(spec: PipelineSpec) -> tuple:
    """
    Sort key from least to most memory-hungry: learning-based detector, `downsample_ratio`,
    `max_features`.
    """
    defaults = stabilo_defaults()
    ratio = float(spec.params.get('downsample_ratio', defaults['downsample_ratio']))
    features = int(spec.params.get('max_features', defaults['max_features']))
    return (spec.detector in Stabilizer.DL_DETECTORS, ratio, features)


def order_by_cost(specs: Sequence[PipelineSpec]) -> List[PipelineSpec]:
    """
    The same grid, cheapest run first (stable within ties).

    A process killed by the kernel then loses the least completed work.
    """
    return sorted(specs, key=resource_cost)


def downsample_ceiling(spec: PipelineSpec, width: int, height: int) -> Optional[float]:
    """
    Largest `downsample_ratio` that keeps a learning-based detector inside its pixel budget.

    None for classical detectors. Budgets come from `Stabilizer.DL_PIXEL_BUDGETS`.
    """
    budget = Stabilizer.DL_PIXEL_BUDGETS.get(spec.detector)
    if budget is None:
        return None
    return float((budget / float(width * height)) ** 0.5)


def check_resolution(spec: PipelineSpec, width: int, height: int) -> Optional[str]:
    """A warning string when a DL pipeline exceeds its pixel budget, else None."""
    ceiling = downsample_ceiling(spec, width, height)
    if ceiling is None:
        return None
    ratio = float(spec.params.get('downsample_ratio', stabilo_defaults()['downsample_ratio']))
    if ratio <= ceiling:
        return None
    processed = width * height * ratio * ratio
    budget = Stabilizer.DL_PIXEL_BUDGETS[spec.detector]
    return (
        f"Pipeline '{spec.name}' runs detector '{spec.detector}' on {processed / 1e6:.2f} MP "
        f"(downsample_ratio={ratio} on {width}x{height}), above its {budget / 1e6:.2f} MP budget. "
        f"Use downsample_ratio <= {ceiling:.4f} to stay inside it."
    )


def processed_pixels(spec: PipelineSpec, width: int, height: int) -> int:
    """Pixel count the detector sees on a `width` x `height` frame, as stabilo's `cv2.resize` rounds it."""
    ratio = float(spec.params.get('downsample_ratio', stabilo_defaults()['downsample_ratio']))
    if ratio == 1.0:
        return int(width) * int(height)
    return round(width * ratio) * round(height * ratio)


def unavailable_detectors(specs: Sequence[PipelineSpec]) -> List[str]:
    """
    Names of the pipelines whose detector this OpenCV build cannot construct.

    A preflight for BRISK, KAZE and AKAZE, whose bindings OpenCV 5 removed; see `environment.py`.
    """
    missing = []
    for spec in specs:
        detector = spec.detector
        if detector in UNBOUND_IN_OPENCV5 and not hasattr(cv2, f"{detector.upper()}_create"):
            missing.append(spec.name)
    return missing


def rotation_limited(spec: PipelineSpec) -> bool:
    """
    True for upright learning-based models, which lose matches past ~30 degrees of rotation.

    Only `keynet` and the classical detectors estimate keypoint orientation.
    """
    return spec.detector in {'xfeat', 'disk', 'dedode', 'loftr'}


# Detectors that run another's keypoint detector: RootSIFT is SIFT's detector with a renormalized descriptor.
SHARED_KEYPOINTS = {'rsift': 'sift'}


def keypoint_detector(detector_name: str) -> str:
    """The detector whose keypoints `detector_name` uses: `rsift` -> `sift`, anything else itself."""
    return SHARED_KEYPOINTS.get(str(detector_name), str(detector_name))


def ignores_ratio_test(spec: PipelineSpec) -> bool:
    """True when `filter_ratio` has no effect: LightGlue and `filter_type: none` skip it, LoFTR has no matcher."""
    return spec.matcher == 'lightglue' or spec.detector == 'loftr' or spec.params.get('filter_type') == 'none'


def describe(specs: List[PipelineSpec]) -> List[Dict[str, Any]]:
    """Flat description of a list of pipelines, for manifests and result tables."""
    return [spec.to_dict() for spec in specs]
