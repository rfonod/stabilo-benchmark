"""The Part I trial figure, on the bundled static sample."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from stabilo_benchmark.config import build_pipelines, load_config
from stabilo_benchmark.masking import load_annotations
from stabilo_benchmark.report.build import build_report
from stabilo_benchmark.report.trial_figure import (
    FRAME_RENDER_PX,
    TrialFigureError,
    _frame_axes,
    _subtitle,
    candidate_trials,
    choose_scene,
    densest_window,
    figure_static_trial,
    perturbed_trial,
    register,
    reproduction_error,
    trial_runs,
)
from stabilo_benchmark.results import ResultStore, build_manifest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'data' / 'sample' / 'static'
CONFIG = ROOT / 'configs' / 'sample' / 'static.yaml'


def _stored(config, spec, trial: int, cte_offset: float = 0.0) -> dict:
    """The record the static runner stores for the sample's busiest scene and `trial`, by registering it here."""
    scene = choose_scene(SAMPLE)
    image = cv2.imread(str(scene))
    labels = load_annotations(SAMPLE, scene.stem, image.shape[1], image.shape[0])
    perturbed = perturbed_trial(image, config, scene.name, trial)
    result = register({'params': spec.to_dict()}, image, labels, perturbed, int(config['seed']), scene.name, trial)
    return {
        'scene': scene.name,
        'trial': trial,
        'failed': False,
        'object_boxes': 'obb',
        'n_inliers': result['n_inliers'],
        'cte_mean': result['cte_mean'] + cte_offset,
    }


def _campaign(tmp_path: Path, maskings=('none', 'obb'), presets=('orb',), records=None) -> Path:
    """
    A static result directory holding the sample config's runs of `presets` under `maskings`.

    `records(config, spec)` gives each run's records; by default one placeholder record.
    """
    config = load_config(CONFIG)
    specs = [
        spec for spec in build_pipelines(config) if spec.masking in maskings and spec.name.rsplit('+', 1)[0] in presets
    ]
    results_dir = tmp_path / 'detectors'
    store = ResultStore(results_dir)
    runs = {number: spec.to_dict() for number, spec in enumerate(specs, start=1)}
    store.initialize(build_manifest(config, 'hash', runs))
    for number, spec in enumerate(specs, start=1):
        stored = (
            records(config, spec)
            if records
            else [{'scene': 'x', 'trial': 0, 'cte_mean': 0.1, 'failed': False, 'object_boxes': 'obb'}]
        )
        store.write_run(number, {'run_number': number, 'params': spec.to_dict(), 'records': stored})
    store.finalize(len(specs), len(specs))
    manifest = json.loads((results_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
    manifest['metadata']['scenes_dir'] = str(SAMPLE)
    (results_dir / 'MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    return results_dir


class TestHelpers:
    def test_the_window_holds_the_most_objects_and_stays_in_frame(self):
        centers = np.array([[100.0, 100.0], [120.0, 110.0], [130.0, 90.0], [900.0, 500.0]])
        x0, y0, x1, y1 = densest_window(centers, 1000, 600, fraction=0.3)
        assert (x1 - x0, y1 - y0) == (300, 180)
        assert x0 == 0 and y0 == 10
        assert densest_window(np.zeros((0, 2)), 1000, 600, fraction=0.3) == (350, 210, 650, 390)

    def test_the_subtitle_reads_the_sampled_parameters_in_order(self):
        text = _subtitle({'rotation_deg': -3.24, 'scale': 1.01, 'blur_ksize': 3, 'jpeg_quality': float('nan')})
        assert text.startswith('rotation -3.2') and 'scale 1.010' in text and 'blur 3 px' in text
        assert 'JPEG' not in text

    def test_the_scene_with_most_objects_is_chosen(self):
        counts = {
            path.stem: len(path.read_text(encoding='utf-8').splitlines())
            for path in (SAMPLE / 'labels_obb').glob('*.txt')
        }
        assert choose_scene(SAMPLE).stem == max(counts, key=counts.get)

    def test_the_trial_is_the_campaigns(self):
        config = load_config(CONFIG)
        image = np.zeros((90, 160, 3), dtype=np.uint8)
        first = perturbed_trial(image, config, 'scene.jpg', 0)
        assert np.array_equal(first['matrix'], perturbed_trial(image, config, 'scene.jpg', 0)['matrix'])
        assert not np.array_equal(first['matrix'], perturbed_trial(image, config, 'scene.jpg', 1)['matrix'])


class TestPanels:
    def test_a_crop_is_cut_from_the_full_resolution_frame(self):
        image = np.zeros((2000, 4000, 3), dtype=np.uint8)
        figure, axes = plt.subplots()
        _frame_axes(axes, image, (100, 200, 1000, 700))
        assert axes.images[0].get_array().shape == (500, 900)
        assert axes.images[0].get_extent() == [100, 1000, 700, 200]
        plt.close(figure)

    def test_a_whole_frame_is_limited_to_the_render_size(self):
        image = np.zeros((2000, 4000, 3), dtype=np.uint8)
        figure, axes = plt.subplots()
        _frame_axes(axes, image)
        assert max(axes.images[0].get_array().shape) == FRAME_RENDER_PX
        plt.close(figure)


@pytest.mark.slow
class TestFigure:
    def test_a_campaign_without_an_unmasked_run_cannot_supply_it(self, tmp_path):
        with pytest.raises(TrialFigureError, match='masking none'):
            trial_runs(_campaign(tmp_path, maskings=('obb',)), 'orb')

    def test_the_figure_is_drawn_when_the_trial_reproduces(self, tmp_path):
        campaign = _campaign(tmp_path, records=lambda config, spec: [_stored(config, spec, 0)])
        path = figure_static_trial(campaign, tmp_path / 'trial.pdf')
        assert path.stat().st_size > 0

    def test_a_trial_that_does_not_reproduce_is_not_drawn(self, tmp_path):
        campaign = _campaign(tmp_path, records=lambda config, spec: [_stored(config, spec, 0, cte_offset=1.0)])
        with pytest.raises(TrialFigureError, match='trial 0: .*corner error 1.000 px'):
            figure_static_trial(campaign, tmp_path / 'trial.pdf')
        assert not (tmp_path / 'trial.pdf').exists()

    def test_the_next_stored_trial_is_tried(self, tmp_path):
        campaign = _campaign(
            tmp_path,
            records=lambda config, spec: [_stored(config, spec, 0, cte_offset=1.0), _stored(config, spec, 1)],
        )
        assert figure_static_trial(campaign, tmp_path / 'trial.pdf').stat().st_size > 0

    def test_the_static_report_writes_it(self, tmp_path):
        campaign = _campaign(tmp_path, records=lambda config, spec: [_stored(config, spec, 0)])
        produced = build_report(campaign, output=tmp_path / 'out')
        assert any(path.endswith('detectors_trial.pdf') for path in produced['figures']), produced['skipped']

    def test_the_static_report_skips_it_with_the_reason(self, tmp_path):
        produced = build_report(_campaign(tmp_path), output=tmp_path / 'out')
        assert not any(path.endswith('detectors_trial.pdf') for path in produced['figures'])
        assert any('trial figure' in reason and 'no stored trial' in reason for reason in produced['skipped'])


class TestReproduction:
    RESULT = {'n_inliers': 120, 'cte_mean': 0.31}

    def test_a_matching_record_reproduces(self):
        assert reproduction_error(self.RESULT, {'n_inliers': 120, 'cte_mean': 0.3101}) is None

    @pytest.mark.parametrize(
        'record, reason',
        [
            (None, 'no stored record'),
            ({'failed': True, 'n_inliers': None, 'cte_mean': float('nan')}, 'failed'),
            ({'n_inliers': 119, 'cte_mean': 0.31}, '120 inliers against 119'),
            ({'n_inliers': 120, 'cte_mean': 0.41}, 'corner error 0.100 px'),
            ({'n_inliers': 120}, 'corner error nan px'),
        ],
    )
    def test_anything_else_does_not(self, record, reason):
        assert reason in reproduction_error(self.RESULT, record)

    def test_the_trials_tried_are_the_scenes_from_the_first_on(self):
        run = {'records': [{'scene': 'a', 'trial': t} for t in (3, 0, 2, 1)] + [{'scene': 'b', 'trial': 9}]}
        assert candidate_trials(run, 'a', 1) == [1, 2, 3]
        assert candidate_trials(run, 'a', 0, count=2) == [0, 1]
        assert candidate_trials(run, 'b', 10) == []
