#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
trial_figure.py - One Part I trial made visible.

Top, a scene and its perturbed copy as one trial of the campaign builds it (`runners.static`: the
same generator, warp and photometric chain), with a sample of one pipeline's inlier matches drawn
between them. Bottom, a crop of the reference frame around its densest group of objects with every
inlier of the same pipeline unmasked and under OBB masks, the suppressed area shaded. The pair is
registered again with the campaign's stored parameters and seed, and drawn only when that
registration reproduces the stored trial under both maskings (`reproduction_error`); otherwise the
next stored trial is tried. A learned detector, a parallel estimator or another machine's OpenCV
build may not reproduce it.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import ConnectionPatch, Rectangle

from stabilo_benchmark.report.figures import INK_PRIMARY, INK_SECONDARY, PALETTE, _save, apply_style
from stabilo_benchmark.report.names import pipeline_name

# The pipeline drawn when the campaign holds it, and the trial.
PRESET = 'orb'
TRIAL = 0
# A re-registration reproduces a stored trial when it keeps the same inlier count and its mean corner
# error lies within this many pixels of the stored one; trials tried, from `TRIAL` on, before skipping.
REPRODUCTION_TOLERANCE_PX = 0.05
TRIALS_TRIED = 10
# Inlier matches drawn between the two frames, and the seed of their sample.
MATCH_LINES = 60
MATCH_SEED = 0
# The crop's width as a share of the frame's; its aspect is the frame's.
CROP_FRACTION = 0.3
# Longest side of each panel as rendered; the PDF embeds it as a raster.
FRAME_RENDER_PX = 1100
# Share of white mixed into the grayscale frame so the overlays carry the color.
LIGHTEN = 0.25
FIGURE_WIDTH_IN = 7.2
GAP_IN = 0.1
TITLE_IN = 0.34
LEGEND_IN = 0.3

GROUND_COLOR = PALETTE[0]
OBJECT_COLOR = PALETTE[1]
MASK_COLOR = '#0b0b0b'
MASK_ALPHA = 0.4


class TrialFigureError(RuntimeError):
    """Raised when a campaign lacks what the figure needs: the pipeline unmasked and OBB-masked, or its scenes."""


def trial_runs(results_dir: Path, preset: str) -> Dict[str, Dict[str, Any]]:
    """The stored runs of `preset` without masking and with OBB masks, keyed by masking mode."""
    from stabilo_benchmark.results import ResultStore  # noqa: PLC0415 - keeps the import graph flat

    runs: Dict[str, Dict[str, Any]] = {}
    for run in ResultStore(Path(results_dir)).iter_runs():
        params = run.get('params') or {}
        label = str(params.get('pipeline', ''))
        if run.get('records') and label.split('#', 1)[0].rsplit('+', 1)[0] == preset:
            runs.setdefault(str(params.get('masking')), run)
    missing = [mode for mode in ('none', 'obb') if mode not in runs]
    if missing:
        raise TrialFigureError(f"{results_dir}: no {preset} run with masking {', '.join(missing)}")
    return {mode: runs[mode] for mode in ('none', 'obb')}


def choose_scene(root: Path, names: Optional[Sequence[str]] = None) -> Path:
    """The scene with the most annotated objects among `names` (every scene without), the first name on a tie."""
    from stabilo_benchmark.masking import load_annotations  # noqa: PLC0415
    from stabilo_benchmark.runners.static import find_scenes  # noqa: PLC0415

    best: Tuple[int, str, Optional[Path]] = (-1, '', None)
    for scene in find_scenes(Path(root), names):
        image = cv2.imread(str(scene), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if image is None:
            continue
        # The labels are normalized, so any frame size reads the same count.
        count = len(load_annotations(Path(root), scene.stem, image.shape[1], image.shape[0]))
        if count > best[0] or (count == best[0] and scene.name < best[1]):
            best = (count, scene.name, scene)
    if best[2] is None:
        raise TrialFigureError(f"{root}: no readable scene")
    return best[2]


def perturbed_trial(image: np.ndarray, config: Mapping[str, Any], scene_name: str, trial: int) -> Dict[str, Any]:
    """The campaign's trial `trial` of a scene: warped image, homography, and sampled parameters."""
    from stabilo_benchmark.perturbations import apply_photometric, sample_geometric, warp_image  # noqa: PLC0415
    from stabilo_benchmark.runners.static import trial_generator  # noqa: PLC0415

    height, width = image.shape[:2]
    perturbation = config.get('perturbation', {}) or {}
    rng = trial_generator(int(config['seed']), scene_name, trial)
    sample = sample_geometric(width, height, perturbation.get('geometric', {}) or {}, rng)
    distorted, photometric = apply_photometric(image, perturbation.get('photometric', []) or [], rng)
    return {
        'image': warp_image(distorted, sample.matrix, width, height),
        'matrix': np.asarray(sample.matrix, dtype=np.float64),
        'params': {**sample.params, **photometric},
    }


def register(
    run: Mapping[str, Any],
    image: np.ndarray,
    labels: Any,
    perturbed: Mapping[str, Any],
    seed: int,
    scene_name: str,
    trial: int,
) -> Dict[str, Any]:
    """Register the perturbed copy onto the scene with a stored run's parameters, as the static runner does."""
    from stabilo_benchmark.metrics.geometric import corner_transfer_error  # noqa: PLC0415
    from stabilo_benchmark.pipelines import PipelineSpec  # noqa: PLC0415
    from stabilo_benchmark.runners.static import boxes_for_frame  # noqa: PLC0415
    from stabilo_benchmark.seeding import seed_estimator  # noqa: PLC0415

    params = dict(run['params'])
    spec = PipelineSpec(name=params.pop('pipeline'), masking=params.pop('masking'), params=params)
    stabilizer = spec.build()
    box_format = spec.box_format or 'xywh'
    stabilizer.set_ref_frame(image, boxes_for_frame(labels, spec.masking), box_format=box_format)
    seed_estimator(seed, scene_name, trial)
    stabilizer.stabilize(perturbed['image'], boxes_for_frame(labels, spec.masking, perturbed['matrix']), box_format)
    estimate = np.asarray(stabilizer.get_cur_trans_matrix(), dtype=np.float64)
    inliers = np.asarray(stabilizer.cur_inliers, dtype=bool).ravel()
    height, width = image.shape[:2]
    count = stabilizer.get_cur_inliers_count()
    return {
        'spec': spec,
        'n_inliers': None if count is None else int(count),
        'reference_points': np.asarray(stabilizer.ref_pts, dtype=np.float64).reshape(-1, 2)[inliers],
        'current_points': np.asarray(stabilizer.cur_pts, dtype=np.float64).reshape(-1, 2)[inliers],
        'cte_mean': corner_transfer_error(perturbed['matrix'], estimate, width, height)['cte_mean'],
    }


def stored_record(run: Mapping[str, Any], scene_name: str, trial: int) -> Optional[Mapping[str, Any]]:
    """The stored record of one scene and trial in a run, or None."""
    for record in run.get('records') or []:
        if record.get('scene') == scene_name and int(record.get('trial', -1)) == trial:
            return record
    return None


def reproduction_error(result: Mapping[str, Any], record: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why a re-registration (`register`) does not reproduce its stored record, or None when it does."""
    if record is None:
        return 'no stored record'
    if record.get('failed'):
        return 'the stored trial failed'
    stored = record.get('n_inliers')
    if stored is None or result['n_inliers'] != int(stored):
        return f"{result['n_inliers']} inliers against {stored} stored"
    drift = abs(float(result['cte_mean']) - float(record.get('cte_mean', np.nan)))
    if not drift < REPRODUCTION_TOLERANCE_PX:
        return f"mean corner error {drift:.3f} px from the stored one"
    return None


def candidate_trials(run: Mapping[str, Any], scene_name: str, first: int, count: int = TRIALS_TRIED) -> List[int]:
    """The stored trials of a scene from `first` on, at most `count`, in order."""
    trials = {
        int(record['trial'])
        for record in run.get('records') or []
        if record.get('scene') == scene_name and int(record.get('trial', -1)) >= first
    }
    return sorted(trials)[:count]


def densest_window(centers: np.ndarray, width: int, height: int, fraction: float = CROP_FRACTION) -> Tuple[int, ...]:
    """(x0, y0, x1, y1) of a window of the frame's aspect, `fraction` of its width, around the most object centers."""
    window_width, window_height = int(round(fraction * width)), int(round(fraction * height))
    if not len(centers):
        x0, y0 = (width - window_width) // 2, (height - window_height) // 2
        return x0, y0, x0 + window_width, y0 + window_height
    inside = (np.abs(centers[:, None, 0] - centers[None, :, 0]) <= window_width / 2) & (
        np.abs(centers[:, None, 1] - centers[None, :, 1]) <= window_height / 2
    )
    cx, cy = centers[int(np.argmax(inside.sum(axis=1)))]
    x0 = int(np.clip(round(cx - window_width / 2), 0, width - window_width))
    y0 = int(np.clip(round(cy - window_height / 2), 0, height - window_height))
    return x0, y0, x0 + window_width, y0 + window_height


def _shown(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """The image as drawn: grayscale, lightened, at most `FRAME_RENDER_PX` on its longest side."""
    height, width = image.shape[:2]
    scale = min(1.0, FRAME_RENDER_PX / max(height, width))
    gray = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return np.round((1 - LIGHTEN) * gray + LIGHTEN * 255).astype(np.uint8), scale


def _frame_axes(axes: plt.Axes, image: np.ndarray, window: Optional[Tuple[int, ...]] = None) -> None:
    """The frame, or its `window` cut from the full-resolution image so that a crop keeps its detail."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = window or (0, 0, width, height)
    shown, _ = _shown(image[y0:y1, x0:x1])
    axes.imshow(shown, cmap='gray', vmin=0, vmax=255, extent=(x0, x1, y1, y0), interpolation='none')
    axes.set_xlim(x0, x1)
    axes.set_ylim(y1, y0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)


def _on_objects(points: np.ndarray, corners: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Which points lie inside an annotated box."""
    from stabilo_benchmark.regions import rasterize_quads  # noqa: PLC0415

    if not len(points) or not len(corners):
        return np.zeros(len(points), dtype=bool)
    mask = rasterize_quads(corners, shape)
    x = np.clip(np.round(points[:, 0]).astype(int), 0, shape[1] - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, shape[0] - 1)
    return mask[y, x]


def _subtitle(params: Mapping[str, Any]) -> str:
    """The sampled perturbation in words, in the order the trial applies it."""
    parts = []
    if 'rotation_deg' in params:
        parts.append(f"rotation {float(params['rotation_deg']):+.1f}$^\\circ$")
    if 'scale' in params:
        parts.append(f"scale {float(params['scale']):.3f}")
    if 'translation_x_px' in params and 'translation_y_px' in params:
        parts.append(f"shift ({float(params['translation_x_px']):+.0f}, {float(params['translation_y_px']):+.0f}) px")
    photometric = []
    for key, text in (
        ('brightness_factor', 'brightness $\\times${:.2f}'),
        ('blur_ksize', 'blur {:.0f} px'),
        ('fog_density', 'haze {:.2f}'),
        ('noise_sigma', 'noise $\\sigma$ {:.1f}'),
        ('jpeg_quality', 'JPEG {:.0f}'),
    ):
        value = params.get(key)
        if value is not None and np.isfinite(float(value)):
            photometric.append(text.format(float(value)))
    return ', '.join(parts) + ('\n' + ', '.join(photometric) if photometric else '')


def figure_static_trial(results_dir: Path, path: Path, preset: str = PRESET, trial: int = TRIAL) -> Path:
    """Write the trial figure for a static campaign that ran `preset` unmasked and under OBB masks."""
    from stabilo_benchmark.masking import load_annotations  # noqa: PLC0415

    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
    config = manifest.get('config') or {}
    metadata = manifest.get('metadata') or {}
    root = Path(metadata.get('scenes_dir') or '')
    if not (root / 'images').is_dir():
        raise TrialFigureError(f"{results_dir}: the scenes directory {root} is not here")
    runs = trial_runs(results_dir, preset)
    scene = choose_scene(root, metadata.get('scenes'))
    image = cv2.imread(str(scene))
    if image is None:
        raise TrialFigureError(f"could not read {scene}")
    height, width = image.shape[:2]
    labels = load_annotations(root, scene.stem, width, height, classes=(config.get('scenes') or {}).get('classes'))
    seed = int(config['seed'])
    attempts = []
    for candidate in candidate_trials(runs['none'], scene.name, trial):
        perturbed = perturbed_trial(image, config, scene.name, candidate)
        results = {
            mode: register(run, image, labels, perturbed, seed, scene.name, candidate) for mode, run in runs.items()
        }
        reasons = {
            mode: reproduction_error(results[mode], stored_record(run, scene.name, candidate))
            for mode, run in runs.items()
        }
        if not any(reasons.values()):
            trial = candidate
            break
        attempts.append(f"trial {candidate}: " + '; '.join(f"{mode} {why}" for mode, why in reasons.items() if why))
    else:
        detail = ', '.join(attempts) if attempts else 'no stored trial'
        raise TrialFigureError(f"{scene.name}: no re-registration reproduces the stored runs ({detail})")
    corners = labels.object_corners('obb') if len(labels) else np.zeros((0, 4, 2))
    carried = _carry(corners, perturbed['matrix'])
    window = densest_window(corners.mean(axis=1) if len(corners) else np.zeros((0, 2)), width, height)

    apply_style()
    panel_width = (FIGURE_WIDTH_IN - GAP_IN) / 2
    panel_height = panel_width * height / width
    figure_height = 2 * (TITLE_IN + panel_height) + GAP_IN + LEGEND_IN
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, figure_height))

    def place(row: int, column: int) -> plt.Axes:
        top = row * (TITLE_IN + panel_height + GAP_IN) + TITLE_IN
        return figure.add_axes(
            (
                column * (panel_width + GAP_IN) / FIGURE_WIDTH_IN,
                1 - (top + panel_height) / figure_height,
                panel_width / FIGURE_WIDTH_IN,
                panel_height / figure_height,
            )
        )

    reference_axes, perturbed_axes = place(0, 0), place(0, 1)
    _frame_axes(reference_axes, image)
    _frame_axes(perturbed_axes, perturbed['image'])
    for axes, quads in ((reference_axes, corners), (perturbed_axes, carried)):
        if len(quads):
            axes.add_collection(PolyCollection(quads, closed=True, facecolor='none', edgecolor='white', linewidths=0.3))
    x0, y0, x1, y1 = window
    reference_axes.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK_PRIMARY, linewidth=0.8, linestyle='--')
    )
    name = pipeline_name(preset)
    reference_axes.set_title(f"(a) Scene {scene.stem}", fontsize=7, color=INK_SECONDARY, pad=2)
    perturbed_axes.set_title(
        f"(b) Trial {trial}: {_subtitle(perturbed['params'])}",
        fontsize=6.5,
        color=INK_SECONDARY,
        pad=2,
        linespacing=1.1,
    )

    unmasked = results['none']
    on_objects = _on_objects(unmasked['reference_points'], corners, (height, width))
    rng = np.random.default_rng(MATCH_SEED)
    drawn = rng.choice(len(on_objects), size=min(MATCH_LINES, len(on_objects)), replace=False)
    for index in drawn:
        figure.add_artist(
            ConnectionPatch(
                xyA=tuple(unmasked['reference_points'][index]),
                coordsA='data',
                axesA=reference_axes,
                xyB=tuple(unmasked['current_points'][index]),
                coordsB='data',
                axesB=perturbed_axes,
                color=OBJECT_COLOR if on_objects[index] else GROUND_COLOR,
                linewidth=0.4,
                alpha=0.7,
            )
        )

    for column, mode in enumerate(('none', 'obb')):
        axes = place(1, column)
        _frame_axes(axes, image, window)
        result = results[mode]
        points = result['reference_points']
        on = _on_objects(points, corners, (height, width))
        if mode == 'obb' and len(corners):
            margin = float(runs[mode]['params'].get('mask_margin_ratio', 0.0) or 0.0)
            centers = corners.mean(axis=1, keepdims=True)
            axes.add_collection(
                PolyCollection(
                    centers + (corners - centers) * (1.0 + margin),
                    closed=True,
                    facecolor=MASK_COLOR,
                    edgecolor='none',
                    alpha=MASK_ALPHA,
                )
            )
        if len(corners):
            axes.add_collection(
                PolyCollection(corners, closed=True, facecolor='none', edgecolor='white', linewidths=0.5)
            )
        inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        axes.scatter(*points[inside & ~on].T, s=3, color=GROUND_COLOR, linewidths=0)
        axes.scatter(*points[inside & on].T, s=5, color=OBJECT_COLOR, linewidths=0)
        heading = '(c) No masking' if mode == 'none' else '(d) OBB masking'
        axes.set_title(
            f"{heading}: {name}, {len(points)} inliers, {on.mean() if len(on) else 0.0:.0%} on objects, "
            f"CTE {result['cte_mean']:.2f} px",
            fontsize=7,
            color=INK_SECONDARY,
            pad=2,
        )

    handles = [
        plt.Line2D([], [], color=color, marker='o', markersize=3, label=label)
        for color, label in ((GROUND_COLOR, 'inlier match elsewhere'), (OBJECT_COLOR, 'inlier match on an object'))
    ]
    handles += [
        plt.Line2D([], [], color='white', marker='s', markeredgecolor=INK_SECONDARY, linestyle='none', label='OBB'),
        plt.Line2D([], [], color=MASK_COLOR, alpha=MASK_ALPHA, marker='s', linestyle='none', label='masked area'),
    ]
    figure.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    return _save(figure, path)


def _carry(corners: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """The scene's boxes in the perturbed copy, corner by corner."""
    from stabilo_benchmark.regions import warp_points  # noqa: PLC0415

    return warp_points(corners, matrix) if len(corners) else corners
