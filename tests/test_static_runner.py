#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
End-to-end static benchmark: bit-exact determinism, paired trials across pipelines, and the
masking axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from stabilo_benchmark.masking import corners_to_hbb, hbb_to_corners, load_annotations
from stabilo_benchmark.regions import warp_points
from stabilo_benchmark.results import (
    RUN_COMPLETED,
    RUN_INFEASIBLE,
    SUCCESS_MARKER,
    ResultStore,
    infeasible_runs,
    to_dataframe,
)
from stabilo_benchmark.runners import static as static_runner
from stabilo_benchmark.runners.static import (
    StaticBenchmarkError,
    boxes_for_frame,
    find_scenes,
    read_scene_list,
    run_static,
    stable_key,
    trial_generator,
)

TIMING_COLUMNS = {'time_stabilize_s', 'time_reference_s'}


def scene_source():
    """The committed sample scenes: images/, labels_hbb/ and labels_obb/."""
    directory = Path(__file__).resolve().parent.parent / 'data' / 'sample' / 'static'
    assert (directory / 'images').is_dir(), f"sample scenes not found at {directory}"
    return directory


def write_config(tmp_path, **overrides):
    """A minimal but complete static config pointing at the sample scenes."""
    config = {
        'kind': 'static',
        'name': 'unit',
        'seed': 0,
        'trials': 2,
        'scenes': {'dir': str(scene_source())},
        'perturbation': {
            'geometric': {'translation_ratio': 0.05, 'rotation_deg': 8.0, 'scale_ratio': 0.02, 'projective': 1.0e-5},
            'photometric': [{'name': 'brightness', 'range': [0.9, 1.1]}, {'name': 'noise', 'max_sigma': 2.0}],
        },
        'metrics': {'hea_thresholds': [1.0, 3.0]},
        'pipeline': {'preset': ['orb'], 'masking': ['none']},
    }
    config.update(overrides)
    path = tmp_path / 'unit.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


# Where `args()` sends a campaign: the running test's own directory, never <scenes>/results beside
# the committed sample. Set by `results_in_tmp`.
RESULTS_ROOT = {}


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path):
    RESULTS_ROOT['path'] = tmp_path / 'results'
    yield
    RESULTS_ROOT.clear()


def args(**overrides):
    defaults = {
        'results_dir': RESULTS_ROOT.get('path'),
        'overwrite': True,
        'resume': False,
        'verbosity': 0,
        'scenes': None,
        'set': None,
        'name': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSeeding:
    def test_scene_key_is_stable_across_processes(self):
        assert stable_key('scene_a.jpg') == stable_key('scene_a.jpg')
        assert stable_key('scene_a.jpg') != stable_key('scene_b.jpg')

    def test_trial_generator_depends_on_scene_and_trial_only(self):
        a = trial_generator(0, 'x.jpg', 3).random(4)
        b = trial_generator(0, 'x.jpg', 3).random(4)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, trial_generator(0, 'x.jpg', 4).random(4))
        assert not np.array_equal(a, trial_generator(0, 'y.jpg', 3).random(4))
        assert not np.array_equal(a, trial_generator(1, 'x.jpg', 3).random(4))


class TestSceneDiscovery:
    def test_finds_the_sample_images_in_a_stable_order(self):
        scenes = find_scenes(scene_source())
        assert len(scenes) >= 3
        assert scenes == sorted(scenes)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(Exception, match='not found'):
            find_scenes(tmp_path / 'absent')

    def test_directory_without_images_raises(self, tmp_path):
        (tmp_path / 'empty' / 'images').mkdir(parents=True)
        with pytest.raises(Exception, match='No images found'):
            find_scenes(tmp_path / 'empty')


def sample_labels():
    """The first sample scene's labels."""
    image = sorted((scene_source() / 'images').glob('*.jpg'))[0]
    height, width = cv2.imread(str(image)).shape[:2]
    return load_annotations(scene_source(), image.stem, width, height)


class TestBoxesForFrame:
    def test_none_masking_passes_no_boxes(self):
        assert boxes_for_frame(sample_labels(), 'none') is None

    def test_hbb_is_the_detectors_box_and_obb_its_conversion(self):
        labels = sample_labels()
        assert np.array_equal(boxes_for_frame(labels, 'hbb'), labels.hbb)
        assert np.array_equal(boxes_for_frame(labels, 'obb'), labels.obb)

    def test_hbb_mask_of_a_rotated_frame_is_the_envelope_of_the_rotated_object(self):
        # The warped frame's HBB is the envelope of the carried OBB.
        labels = sample_labels()
        rotate = np.vstack([cv2.getRotationMatrix2D((960.0, 540.0), 30.0, 1.0), [0, 0, 1]])
        rotated = boxes_for_frame(labels, 'hbb', rotate)
        from_obb = corners_to_hbb(warp_points(labels.obb.reshape(-1, 4, 2), rotate))
        from_hbb = corners_to_hbb(warp_points(hbb_to_corners(labels.hbb), rotate))
        assert np.allclose(rotated, from_obb)
        assert rotated[:, 2:].prod(axis=1).sum() < from_hbb[:, 2:].prod(axis=1).sum()


@pytest.mark.slow
class TestResultsLocation:
    def test_results_default_to_the_scenes_directory(self, tmp_path):
        scenes = tmp_path / 'static'
        shutil.copytree(scene_source(), scenes, ignore=shutil.ignore_patterns('results'))
        results_dir = run_static(write_config(tmp_path, scenes={'dir': str(scenes)}), args(results_dir=None))
        assert results_dir == scenes / 'results' / 'unit'
        assert ResultStore(results_dir).is_complete()

    def test_results_dir_names_the_campaign_s_parent(self, tmp_path):
        results_dir = run_static(write_config(tmp_path), args(results_dir=tmp_path / 'mine'))
        assert results_dir == tmp_path / 'mine' / 'unit'
        assert ResultStore(results_dir).is_complete()


@pytest.mark.slow
class TestEndToEnd:
    def test_produces_results_a_manifest_and_a_success_marker(self, tmp_path):
        results_dir = run_static(write_config(tmp_path), args())
        store = ResultStore(results_dir)
        assert store.is_complete()
        assert (results_dir / SUCCESS_MARKER).exists()
        assert store.manifest_path.exists()

        frame = to_dataframe(store)
        assert len(frame) == 3 * 2  # 3 scenes x 2 trials
        for column in ('cte_mean', 'iou_mean', 'oce_mean', 'inlier_ratio', 'rotation_deg', 'failed'):
            assert column in frame.columns

    def test_records_the_work_proxy_and_the_runs_peak_memory(self, tmp_path):
        # Every record has `processed_pixels`; every run has `peak_memory`.
        config = write_config(tmp_path, pipeline={'preset': ['orb'], 'masking': ['none'], 'downsample_ratio': [0.5]})
        store = ResultStore(run_static(config, args()))
        frame = to_dataframe(store)
        assert (frame["processed_pixels"] == 1920 * 1080).all()  # the 3840x2160 sample scenes at 0.5
        peak = store.read_run(1)['peak_memory']
        if importlib.util.find_spec('resource') is not None:  # no host counter on Windows
            assert 'host_peak_scope' in peak

    def test_records_the_sampled_perturbation_of_every_trial(self, tmp_path):
        frame = to_dataframe(ResultStore(run_static(write_config(tmp_path), args())))
        for column in ('rotation_deg', 'translation_x_px', 'scale', 'brightness_factor', 'noise_sigma'):
            assert column in frame.columns
        assert frame['rotation_deg'].nunique() > 1

    def test_recovers_the_known_homography_accurately(self, tmp_path):
        config = write_config(tmp_path, pipeline={'preset': ['sift'], 'masking': ['obb']})
        frame = to_dataframe(ResultStore(run_static(config, args())))
        assert not frame['failed'].any()
        assert frame['cte_mean'].median() < 2.0
        assert frame['iou_mean'].median() > 0.85

    def test_repeated_run_is_bit_identical_except_for_timings(self, tmp_path):
        first = to_dataframe(ResultStore(run_static(write_config(tmp_path), args())))
        second = to_dataframe(ResultStore(run_static(write_config(tmp_path), args())))
        compared = [c for c in first.columns if c not in TIMING_COLUMNS]
        assert first[compared].equals(second[compared])

    def test_paired_trials_are_identical_across_pipelines(self, tmp_path):
        # The perturbation for (scene, trial) must not depend on which pipeline sees it.
        config = write_config(tmp_path, pipeline={'preset': ['orb', 'sift'], 'masking': ['none']})
        frame = to_dataframe(ResultStore(run_static(config, args())))
        by_pipeline = {name: group for name, group in frame.groupby('pipeline')}
        assert len(by_pipeline) == 2
        keys = ['scene', 'trial']
        left, right = (group.sort_values(keys).reset_index(drop=True) for group in by_pipeline.values())
        for column in ('rotation_deg', 'translation_x_px', 'scale', 'brightness_factor', 'noise_sigma'):
            assert np.allclose(left[column], right[column])

    def test_a_failed_trial_is_recorded_with_nan_scores(self, tmp_path, monkeypatch):
        # A failed trial records every metric as NaN.
        monkeypatch.setattr('stabilo_benchmark.runners.static.estimation_failed', lambda *_, **__: True)
        frame = to_dataframe(ResultStore(run_static(write_config(tmp_path), args())))
        assert len(frame) == 3 * 2 and frame['failed'].all()
        for column in ('cte_mean', 'iou_mean', 'iou_min', 'oce_mean', 'oce_max'):
            assert frame[column].isna().all()

    def test_all_three_masking_modes_run(self, tmp_path):
        config = write_config(tmp_path, pipeline={'preset': ['orb'], 'masking': ['none', 'hbb', 'obb']})
        frame = to_dataframe(ResultStore(run_static(config, args())))
        assert set(frame['masking']) == {'none', 'hbb', 'obb'}

    def test_completed_results_are_not_silently_overwritten(self, tmp_path):
        config = write_config(tmp_path)
        results_dir = run_static(config, args())
        marker = results_dir / SUCCESS_MARKER
        before = marker.stat().st_mtime_ns
        run_static(config, args(overwrite=False))
        assert marker.stat().st_mtime_ns == before

    def test_resume_skips_completed_runs(self, tmp_path):
        config = write_config(tmp_path, pipeline={'preset': ['orb', 'sift'], 'masking': ['none']})
        results_dir = run_static(config, args())
        first_run = results_dir / '0000001.json'
        before = first_run.stat().st_mtime_ns
        run_static(config, args(overwrite=False, resume=True))
        assert first_run.stat().st_mtime_ns == before

    def test_wrong_kind_is_rejected(self, tmp_path):
        config = write_config(tmp_path, kind='dynamic')
        with pytest.raises(Exception, match="expected 'static'"):
            run_static(config, args())

    def test_overrides_and_name_take_effect(self, tmp_path):
        results_dir = run_static(write_config(tmp_path), args(set=['trials=1'], name='overridden'))
        assert results_dir.name == 'overridden'
        assert len(to_dataframe(ResultStore(results_dir))) == 3  # 3 scenes x 1 trial

    def test_scenes_override_is_honoured(self, tmp_path):
        config = write_config(tmp_path)
        results_dir = run_static(config, args(scenes=scene_source()))
        assert ResultStore(results_dir).is_complete()


@pytest.mark.slow
class TestWithoutAnnotations:
    """Scenes without labels_hbb/ and labels_obb/ run with masking 'none' only, and record it."""

    def _scenes(self, tmp_path):
        scenes = tmp_path / 'scenes'
        shutil.copytree(scene_source() / 'images', scenes / 'images')
        return scenes

    def test_masking_none_runs_without_object_scores(self, tmp_path):
        results_dir = run_static(write_config(tmp_path), args(scenes=self._scenes(tmp_path)))
        store = ResultStore(results_dir)
        assert (
            store.is_complete()
            and json.loads(store.manifest_path.read_text(encoding='utf-8'))['metadata']['annotated'] is False
        )
        frame = to_dataframe(store)
        assert np.all(np.isfinite(frame['cte_mean'])) and set(frame['n_objects']) == {0}
        assert 'iou_mean' not in frame or frame['iou_mean'].isna().all()

    def test_a_masking_mode_needing_boxes_is_refused_before_any_run(self, tmp_path):
        config = write_config(tmp_path, pipeline={'preset': ['orb'], 'masking': ['none', 'obb']})
        with pytest.raises(StaticBenchmarkError, match='no annotations'):
            run_static(config, args(scenes=self._scenes(tmp_path)))
        assert not RESULTS_ROOT['path'].exists()

    def test_annotated_scenes_are_recorded_as_such(self, tmp_path):
        store = ResultStore(run_static(write_config(tmp_path), args()))
        assert json.loads(store.manifest_path.read_text(encoding='utf-8'))['metadata']['annotated'] is True


@pytest.mark.slow
class TestObjectBoxes:
    """IoU and OCE are scored on OBBs by default, on HBBs when asked, and never on the masking mode's boxes."""

    def _frame(self, tmp_path, object_boxes=None):
        metrics = {'hea_thresholds': [1.0, 3.0], **({'object_boxes': object_boxes} if object_boxes else {})}
        config = write_config(tmp_path, metrics=metrics, pipeline={'preset': ['orb'], 'masking': ['hbb']})
        frame = to_dataframe(ResultStore(run_static(config, args(name=object_boxes or 'default'))))
        return frame.sort_values(['scene', 'trial']).reset_index(drop=True)

    def test_default_is_obb_whatever_the_masking_mode(self, tmp_path):
        assert set(self._frame(tmp_path)['object_boxes']) == {'obb'}

    def test_hbb_changes_only_the_box_scores(self, tmp_path):
        obb, hbb = self._frame(tmp_path, 'obb'), self._frame(tmp_path, 'hbb')
        assert set(hbb['object_boxes']) == {'hbb'}
        # Same pipeline, same trials: the registration is identical, only the boxes scored differ.
        assert np.array_equal(obb['cte_mean'], hbb['cte_mean'])
        assert not np.allclose(obb['iou_mean'], hbb['iou_mean'])
        assert np.all(np.isfinite(hbb['oce_mean']))


class TestSceneLists:
    """A named subset of a scene directory, which keeps the tuning split disjoint."""

    @staticmethod
    def _scenes(directory):
        import cv2  # noqa: PLC0415 - only this class needs it

        (directory / 'images').mkdir(parents=True, exist_ok=True)
        for stem in ('alpha', 'beta', 'gamma'):
            cv2.imwrite(str(directory / 'images' / f"{stem}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
        return directory

    def test_a_list_narrows_the_directory(self, tmp_path):
        directory = self._scenes(tmp_path / 'scenes')
        assert [p.stem for p in find_scenes(directory, ['gamma', 'alpha'])] == ['alpha', 'gamma']

    def test_names_match_with_or_without_the_extension(self, tmp_path):
        directory = self._scenes(tmp_path / 'scenes')
        assert find_scenes(directory, ['alpha']) == find_scenes(directory, ['alpha.jpg'])

    def test_order_follows_the_directory_not_the_list(self, tmp_path):
        # List order does not change the perturbations a scene sees.
        directory = self._scenes(tmp_path / 'scenes')
        assert find_scenes(directory, ['gamma', 'alpha']) == find_scenes(directory, ['alpha', 'gamma'])

    def test_an_unmatched_name_raises_rather_than_shrinking_the_campaign(self, tmp_path):
        directory = self._scenes(tmp_path / 'scenes')
        with pytest.raises(StaticBenchmarkError, match='not in'):
            find_scenes(directory, ['alpha', 'delta'])

    def test_comments_and_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / 'split.txt'
        path.write_text('# tuning half\nalpha\n\nbeta  # keep\n', encoding='utf-8')
        assert read_scene_list(path) == ['alpha', 'beta']

    def test_an_empty_list_raises(self, tmp_path):
        path = tmp_path / 'split.txt'
        path.write_text('# nothing but a comment\n', encoding='utf-8')
        with pytest.raises(StaticBenchmarkError, match='no scene names'):
            read_scene_list(path)

    def test_a_missing_list_raises(self, tmp_path):
        with pytest.raises(StaticBenchmarkError, match='not found'):
            read_scene_list(tmp_path / 'absent.txt')

    def test_two_lists_over_one_directory_are_disjoint_when_written_disjoint(self, tmp_path):
        directory = self._scenes(tmp_path / 'scenes')
        tune = {p.name for p in find_scenes(directory, ['alpha'])}
        evaluate = {p.name for p in find_scenes(directory, ['beta', 'gamma'])}
        assert not tune & evaluate


@pytest.mark.slow
class TestResourceInfeasibleRuns:
    """An out-of-memory error aborts its run: no measurements, status `infeasible`."""

    @staticmethod
    def _out_of_memory_after(monkeypatch, scenes_completed: int):
        """Make scene loading raise an allocator error once `scenes_completed` have succeeded."""
        original = static_runner.load_scene
        seen = {'n': 0}

        def failing(scene_path, scenes_config):
            if seen['n'] >= scenes_completed:
                raise RuntimeError('CUDA out of memory. Tried to allocate 20.00 GiB')
            seen['n'] += 1
            return original(scene_path, scenes_config)

        monkeypatch.setattr(static_runner, 'load_scene', failing)

    def test_the_run_is_aborted_and_its_partial_records_discarded(self, tmp_path, monkeypatch):
        self._out_of_memory_after(monkeypatch, scenes_completed=1)
        results_dir = run_static(write_config(tmp_path), args())
        run = ResultStore(results_dir).read_run(1)
        assert run['status'] == RUN_INFEASIBLE
        assert run['records'] == []  # the one scene that fit is not a result
        assert 'out of memory' in run['resource_error']['message']
        assert run['resource_error']['where'].endswith('.jpg')

    def test_the_campaign_still_completes_and_says_so(self, tmp_path, monkeypatch):
        self._out_of_memory_after(monkeypatch, scenes_completed=0)
        results_dir = run_static(write_config(tmp_path), args())
        marker = (results_dir / SUCCESS_MARKER).read_text(encoding='utf-8')
        assert 'N/A' in marker
        assert [run['params']['pipeline'] for run in infeasible_runs(ResultStore(results_dir))] == ['orb+none']

    def test_a_resume_still_names_a_stored_infeasible_run(self, tmp_path, monkeypatch):
        self._out_of_memory_after(monkeypatch, scenes_completed=0)
        config = write_config(tmp_path)
        results_dir = run_static(config, args())
        run_static(config, args(overwrite=False, resume=True))
        assert 'orb+none' in (results_dir / SUCCESS_MARKER).read_text(encoding='utf-8')

    def test_an_ordinary_scene_failure_still_only_costs_that_scene(self, tmp_path, monkeypatch):
        original = static_runner.load_scene
        seen = {'n': 0}

        def failing(scene_path, scenes_config):
            seen['n'] += 1
            if seen['n'] == 1:
                raise ValueError('unreadable annotation')
            return original(scene_path, scenes_config)

        monkeypatch.setattr(static_runner, 'load_scene', failing)
        results_dir = run_static(write_config(tmp_path), args())
        run = ResultStore(results_dir).read_run(1)
        assert run['status'] == RUN_COMPLETED
        assert run['records']  # the remaining scenes were measured
        assert len(run['scene_errors']) == 1

    def test_a_healthy_run_is_marked_completed(self, tmp_path):
        results_dir = run_static(write_config(tmp_path), args())
        assert ResultStore(results_dir).read_run(1)['status'] == RUN_COMPLETED
        assert infeasible_runs(ResultStore(results_dir)) == []


@pytest.mark.slow
class TestCheapestFirst:
    """The grid runs cheapest configuration first."""

    def test_the_grid_runs_from_the_smallest_budget_upward(self, tmp_path):
        config = write_config(
            tmp_path,
            trials=1,
            pipeline={'preset': ['orb'], 'masking': ['none'], 'downsample_ratio': [1.0, 0.25, 0.5]},
        )
        results_dir = run_static(config, args())
        store = ResultStore(results_dir)
        ratios = [store.read_run(n)['params']['downsample_ratio'] for n in (1, 2, 3)]
        assert ratios == [0.25, 0.5, 1.0]
