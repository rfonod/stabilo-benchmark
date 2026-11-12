#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
static.py - Part I: the static benchmark, under an exactly known homography.

For each scene and trial, a random homography and a chain of photometric degradations are
applied; the pipeline registers the degraded, warped frame back onto the original, and the
estimate is compared with the known inverse.

- The perturbation of trial `t` of scene `s` depends on (seed, scene, trial) only, so every
  pipeline faces the same perturbations (paired trials).
- A failed estimate (stabilo's identity fallback with no inlier count, see `estimation_failed`) is recorded with NaN
  metrics and a failure flag.
- Masks are applied to both frames. The scene's boxes come from `masking.load_annotations`; the
  warped frame's boxes are the scene's carried through the known homography.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from stabilo_benchmark.config import (
    apply_overrides,
    build_pipelines,
    config_hash,
    load_config,
    resolve_path,
    run_key,
    validate_config,
)
from stabilo_benchmark.masking import (
    IMAGES_DIR,
    LabelError,
    Labels,
    corners_to_hbb,
    hbb_to_corners,
    load_annotations,
    require_annotations,
    summarize,
)
from stabilo_benchmark.metrics.diagnostics import estimation_failed, matching_stats
from stabilo_benchmark.metrics.geometric import geometric_scores, nan_object_scores, threshold_label
from stabilo_benchmark.perturbations import apply_photometric, sample_geometric, warp_image
from stabilo_benchmark.pipelines import (
    DEFAULT_OBJECT_BOXES,
    PipelineSpec,
    check_resolution,
    opencv_binding_hint,
    order_by_cost,
    processed_pixels,
    unavailable_detectors,
)
from stabilo_benchmark.regions import warp_points
from stabilo_benchmark.resources import describe_resource_error, is_resource_error, peak_memory, reset_peak_memory
from stabilo_benchmark.results import (
    RUN_COMPLETED,
    RUN_INFEASIBLE,
    ResultStore,
    build_manifest,
    describe_path,
    seed_everything,
)
from stabilo_benchmark.seeding import seed_estimator, stable_key
from stabilo_benchmark.trial_cache import CachedTrial, TrialCache, estimated_bytes, fits_on_disk

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class StaticBenchmarkError(RuntimeError):
    """Raised when the static benchmark cannot run at all (no scenes, unreadable inputs)."""


def trial_generator(seed: int, scene: str, trial: int) -> np.random.Generator:
    """The perturbation generator for one (scene, trial), independent of the pipeline."""
    return np.random.default_rng([int(seed), stable_key(scene), int(trial)])


# --------------------------------------------------------------------------------------
# Scene loading
# --------------------------------------------------------------------------------------


def find_scenes(scenes_dir: Path, names: Optional[Sequence[str]] = None) -> List[Path]:
    """
    Every readable image in `<scenes_dir>/images`, in a stable order.

    With `names`, only those scenes are kept, matched on the filename or its stem; a name that
    matches nothing raises.
    """
    images_dir = scenes_dir / IMAGES_DIR
    if not images_dir.is_dir():
        raise StaticBenchmarkError(
            f"{images_dir} not found. A scenes directory holds images/, labels_hbb/ and labels_obb/."
        )
    scenes = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not scenes:
        raise StaticBenchmarkError(f"No images found in {images_dir} (looked for {', '.join(IMAGE_SUFFIXES)})")
    if names is None:
        return scenes

    wanted = list(names)
    by_key = {}
    for scene in scenes:
        by_key.setdefault(scene.name, scene)
        by_key.setdefault(scene.stem, scene)
    missing = [name for name in wanted if name not in by_key]
    if missing:
        raise StaticBenchmarkError(
            f"{len(missing)} scene(s) in the scene list are not in {scenes_dir}: "
            f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
        )
    # De-duplicated, in directory order.
    selected = {by_key[name].name for name in wanted}
    return [scene for scene in scenes if scene.name in selected]


def read_scene_list(path: Path) -> List[str]:
    """Scene names from a text file: one per line, `#` comments and blank lines ignored."""
    path = Path(path)
    if not path.is_file():
        raise StaticBenchmarkError(f"Scene list not found: {path}")
    names = [line.split('#', 1)[0].strip() for line in path.read_text(encoding='utf-8').splitlines()]
    names = [name for name in names if name]
    if not names:
        raise StaticBenchmarkError(f"Scene list {path} contains no scene names")
    return names


def load_scene(path: Path, scenes_config: Mapping[str, Any]) -> tuple:
    """Read one scene image and its labels; returns (image, labels)."""
    image = cv2.imread(str(path))
    if image is None:
        raise StaticBenchmarkError(f"Could not read scene image: {path}")
    height, width = image.shape[:2]
    labels = load_annotations(path.parent.parent, path.stem, width, height, classes=scenes_config.get('classes'))
    return image, labels


def boxes_for_frame(labels: Labels, masking: str, matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Object boxes in the format stabilo expects, optionally transformed by a homography.

    'hbb' uses the horizontal boxes and 'obb' the oriented ones. Under a homography the corners are
    transformed first; the warped frame's horizontal box is the axis-aligned envelope of the carried
    oriented box, or of the carried horizontal box when no oriented box exists.
    """
    if masking == 'none':
        return None
    if masking == 'obb' and not labels.has_obb:
        labels.boxes_for('obb')  # raises LabelError with the actionable message
    if matrix is None:
        return labels.boxes_for(masking)
    if masking == 'obb':
        corners = labels.obb.reshape(-1, 4, 2)
    else:
        corners = labels.obb.reshape(-1, 4, 2) if labels.has_obb else hbb_to_corners(labels.hbb)
    if corners.size == 0:
        return np.zeros((0, 8 if masking == 'obb' else 4), dtype=np.float64)
    corners = warp_points(corners, matrix)
    return corners.reshape(-1, 8) if masking == 'obb' else corners_to_hbb(corners)


# --------------------------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------------------------


def run_pipeline(
    spec: PipelineSpec,
    scenes: Sequence[Path],
    config: Mapping[str, Any],
    verbosity: int = 0,
    trial_cache: Optional[TrialCache] = None,
) -> Dict[str, Any]:
    """
    Evaluate one pipeline over every scene and trial, returning its result record.

    With `trial_cache`, perturbed frames are read from it when present and written to it otherwise.
    A scene that cannot be processed is recorded as a scene-level error and the run continues.
    """
    seed = int(config['seed'])
    trials = int(config['trials'])
    scenes_config = config.get('scenes', {}) or {}
    perturbation = config.get('perturbation', {}) or {}
    geometric_config = perturbation.get('geometric', {}) or {}
    photometric_config = perturbation.get('photometric', []) or []
    metrics_config = config.get('metrics', {}) or {}
    thresholds = metrics_config.get('hea_thresholds', [1.0, 3.0, 5.0])
    object_boxes = metrics_config.get('object_boxes', DEFAULT_OBJECT_BOXES)

    records: List[Dict[str, Any]] = []
    scene_errors: Dict[str, str] = {}

    for scene_path in tqdm(scenes, desc=spec.name, unit='scene', leave=False, disable=verbosity > 2):
        scene_name = scene_path.name
        try:
            image, labels = load_scene(scene_path, scenes_config)
            height, width = image.shape[:2]

            work = processed_pixels(spec, width, height)
            warning = check_resolution(spec, width, height)
            if warning and verbosity > 0:
                tqdm.write(f"\033[93m{warning}\033[0m")

            stabilizer = spec.build(
                min_good_match_count_warning=20 if verbosity > 2 else -1,
                min_inliers_match_count_warning=10 if verbosity > 2 else -1,
            )

            reference_boxes = boxes_for_frame(labels, spec.masking)
            reference_start = time.perf_counter()
            stabilizer.set_ref_frame(image, reference_boxes, box_format=spec.box_format or 'xywh')
            reference_time = time.perf_counter() - reference_start

            # Boxes IoU and OCE are scored on (`metrics.object_boxes`), whatever the masking mode.
            reference_corners = labels.object_corners(object_boxes)
            label_summary = summarize(labels)

            for trial in range(trials):
                cached = trial_cache.load(scene_path, trial, width, height) if trial_cache else None
                if cached is None:
                    rng = trial_generator(seed, scene_name, trial)
                    perturbation_sample = sample_geometric(width, height, geometric_config, rng)
                    distorted, photometric_params = apply_photometric(image, photometric_config, rng)
                    cached = CachedTrial(
                        image=warp_image(distorted, perturbation_sample.matrix, width, height),
                        matrix=perturbation_sample.matrix,
                        geometric_params=perturbation_sample.params,
                        photometric_params=photometric_params,
                    )
                    if trial_cache:
                        trial_cache.store(scene_path, trial, cached)
                matrix = cached.matrix
                warped = cached.image
                warped_boxes = boxes_for_frame(labels, spec.masking, matrix)

                seed_estimator(seed, scene_name, trial)
                start = time.perf_counter()
                stabilizer.stabilize(warped, warped_boxes, box_format=spec.box_format or 'xywh')
                elapsed = time.perf_counter() - start

                estimate = stabilizer.get_cur_trans_matrix()
                failed = estimation_failed(stabilizer, estimate)

                record: Dict[str, Any] = {
                    'scene': scene_name,
                    'trial': trial,
                    'failed': failed,
                    'object_boxes': object_boxes,
                    'time_stabilize_s': elapsed,
                    'time_reference_s': reference_time,
                    'processed_pixels': work,
                    **cached.geometric_params,
                    **cached.photometric_params,
                    **matching_stats(stabilizer),
                    **label_summary,
                }

                if failed:
                    record.update(_nan_geometric(thresholds))
                else:
                    record.update(
                        geometric_scores(
                            matrix,
                            np.asarray(estimate, dtype=np.float64),
                            width,
                            height,
                            corners_reference=reference_corners if len(labels) else None,
                            hea_thresholds=thresholds,
                        )
                    )
                records.append(record)

        except Exception as error:  # noqa: BLE001 - one scene must not abort the campaign
            # A resource error ends the run instead (see `resources.py`).
            if is_resource_error(error):
                if verbosity > 0:
                    tqdm.write(
                        f"\033[93mInfeasible: '{spec.name}' exhausted memory on scene '{scene_name}' "
                        f"({type(error).__name__}: {error}). Reported as N/A.\033[0m"
                    )
                return {
                    'params': spec.to_dict(),
                    'records': [],
                    'scene_errors': scene_errors,
                    'n_scenes': len(scenes),
                    'n_trials': trials,
                    'status': RUN_INFEASIBLE,
                    'resource_error': describe_resource_error(error, where=scene_name),
                }
            scene_errors[scene_name] = f"{type(error).__name__}: {error}"
            if verbosity > 0:
                tqdm.write(f"\033[93mWarning: scene '{scene_name}' failed for '{spec.name}': {error}\033[0m")

    return {
        'params': spec.to_dict(),
        'records': records,
        'scene_errors': scene_errors,
        'n_scenes': len(scenes),
        'n_trials': trials,
        'status': RUN_COMPLETED,
    }


def _nan_geometric(thresholds: Sequence[float]) -> Dict[str, Any]:
    """Metric values for a trial where no transform was produced."""
    record: Dict[str, Any] = {
        'cte_mean': float('nan'),
        'cte_max': float('nan'),
        'cte_rms': float('nan'),
        'h_frobenius': float('nan'),
        **nan_object_scores(),
    }
    for threshold in thresholds:
        record[f"within_{threshold_label(threshold)}"] = False
    return record


# --------------------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------------------


def run_static(config_path: Path, args: argparse.Namespace) -> Optional[Path]:
    """Execute the static benchmark described by one config file."""
    config = load_config(config_path)
    if config.get('kind') != 'static':
        raise StaticBenchmarkError(f"{config_path} declares kind '{config.get('kind')}', expected 'static'")
    apply_overrides(config, getattr(args, 'set', None))
    if getattr(args, 'name', None):
        config['name'] = str(args.name)
    validate_config(config)

    # --scenes overrides the config's scenes directory.
    scenes_override = getattr(args, 'scenes', None)
    scenes_dir = Path(scenes_override) if scenes_override else resolve_path(config, config['scenes']['dir'])
    # Results go to <scenes>/results/<name> unless --results-dir names another root.
    results_override = getattr(args, 'results_dir', None)
    results_root = Path(results_override) if results_override else scenes_dir / 'results'
    results_dir = results_root / str(config['name'])

    store = ResultStore(
        results_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        regrid=getattr(args, 'regrid', False),
        run_key=run_key,
    )
    if store.is_complete() and not (args.overwrite or store.resume):
        print(f"\033[33mSkipping '{config['name']}': results already exist. Use --overwrite or --resume.\033[0m")
        return results_dir

    specs = order_by_cost(build_pipelines(config))
    # Preflight: stop before any frame is read if a detector cannot be constructed.
    missing = unavailable_detectors(specs)
    if missing:
        raise StaticBenchmarkError(
            f"{len(missing)} pipeline(s) name a detector that is unavailable here: "
            f"{', '.join(missing)}.\n{opencv_binding_hint()}"
        )
    # --scenes-list overrides the config's `scenes.list`.
    list_override = getattr(args, 'scenes_list', None)
    list_path = Path(list_override) if list_override else None
    if list_path is None and config['scenes'].get('list'):
        list_path = resolve_path(config, config['scenes']['list'])
    scenes = find_scenes(scenes_dir, read_scene_list(list_path) if list_path else None)
    try:
        annotated = require_annotations(scenes_dir, [spec.masking for spec in specs])
    except LabelError as error:
        raise StaticBenchmarkError(str(error)) from error

    manifest = build_manifest(
        config,
        config_hash(config),
        {index: spec.to_dict() for index, spec in enumerate(specs, start=1)},
    )
    manifest.metadata['seeds'] = seed_everything(int(config['seed']))
    manifest.metadata['scenes'] = [scene.name for scene in scenes]
    manifest.metadata['scenes_dir'] = describe_path(scenes_dir)
    manifest.metadata['scenes_list'] = describe_path(list_path) if list_path else None
    manifest.metadata['annotated'] = annotated
    store.initialize(manifest)

    print(
        f"\033[92mStatic benchmark '{config['name']}': {len(specs)} pipeline(s) x {len(scenes)} scene(s) "
        f"x {config['trials']} trial(s) = {len(specs) * len(scenes) * int(config['trials'])} evaluations\033[0m"
    )
    print(f"\033[92mResults -> {results_dir}\033[0m")
    if not annotated:
        print(f"\033[93m{scenes_dir} is not annotated: masking 'none' only, and no IoU or OCE.\033[0m")

    # Outside the campaign directory, so copying a campaign's results does not copy the frames.
    trial_cache = None
    if getattr(args, 'trial_cache', True):
        cache_root = results_root / '.cache' / str(config['name'])
        needed = estimated_bytes(scenes[0], len(scenes), int(config['trials']))
        fits, free = fits_on_disk(cache_root, needed)
        if fits:
            trial_cache = TrialCache(cache_root, int(config['seed']), config.get('perturbation', {}) or {})
            print(
                f"\033[92mTrial cache -> {cache_root} (up to {needed / 1e9:.1f} GB, "
                f"deleted when the campaign completes)\033[0m"
            )
        else:
            print(
                f"\033[93mTrial cache off: it needs about {needed / 1e9:.1f} GB and {free / 1e9:.1f} GB is free. "
                f"Every run regenerates its perturbed frames; results are the same.\033[0m"
            )

    completed = 0
    empty: List[str] = []
    infeasible: List[str] = []
    for run_number, spec in enumerate(tqdm(specs, desc='Pipeline', unit='run'), start=1):
        if store.resume and store.has_run(run_number):
            # A stored run still counts toward the marker's infeasible and empty lists.
            result = store.read_run(run_number)
        else:
            # Reset per run so each run reports its own peak (see `resources.py`).
            reset_peak_memory()
            try:
                result = run_pipeline(spec, scenes, config, verbosity=args.verbosity, trial_cache=trial_cache)
            except Exception as error:  # noqa: BLE001 - recorded, then re-raised for the caller
                store.record_failure(run_number, spec.to_dict(), len(specs), error)
                store.finalize(completed, len(specs))
                raise
            result['run_number'] = run_number
            result['peak_memory'] = peak_memory()
            store.write_run(run_number, result)
            if args.verbosity > 0:
                _print_run_summary(spec, result)
        completed += 1
        if result.get('status') == RUN_INFEASIBLE:
            infeasible.append(spec.name)
        elif not result.get('records'):
            empty.append(spec.name)

    store.finalize(completed, len(specs), empty, infeasible)
    if trial_cache is not None and completed == len(specs):
        trial_cache.clear()
    return results_dir


def _print_run_summary(spec: PipelineSpec, result: Mapping[str, Any]) -> None:
    """One-line summary of a finished run."""
    records = result.get('records', [])
    if result.get('status') == RUN_INFEASIBLE:
        where = result.get('resource_error', {}).get('where', '?')
        print(f"  {spec.name:<32} infeasible on this machine (out of memory at '{where}')")
        return
    if not records:
        print(f"  {spec.name:<32} no valid trials")
        return
    errors = np.array([r.get('cte_mean', np.nan) for r in records], dtype=float)
    failures = float(np.mean([bool(r.get('failed')) for r in records]))
    finite = errors[np.isfinite(errors)]
    median = float(np.median(finite)) if finite.size else float('nan')
    print(f"  {spec.name:<32} median CTE {median:8.3f} px | failure rate {failures:6.1%} | n={len(records)}")
