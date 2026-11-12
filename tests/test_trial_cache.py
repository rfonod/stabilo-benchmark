#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
The trial cache: records are identical with and without it, frames are built once per campaign,
stale entries are never read, and a cache that cannot be written disables itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_static_runner import TIMING_COLUMNS, args, results_in_tmp, scene_source, write_config  # noqa: F401

from stabilo_benchmark import trial_cache as trial_cache_module
from stabilo_benchmark.runners import static as static_runner
from stabilo_benchmark.runners.static import run_static
from stabilo_benchmark.trial_cache import CachedTrial, TrialCache


@pytest.fixture(autouse=True)
def no_disk_reserve(monkeypatch):
    """The cache starts whatever the machine's free space; the test of the reserve sets its own."""
    monkeypatch.setattr(trial_cache_module, 'DISK_RESERVE_BYTES', 0)


def records_of(results_dir):
    """Every run's records with the timings removed, as JSON text, in run order."""
    out = []
    for path in sorted(results_dir.glob('0*.json')):
        records = json.loads(path.read_text(encoding='utf-8'))['records']
        untimed = [{k: v for k, v in r.items() if k not in TIMING_COLUMNS} for r in records]
        out.append(json.dumps(untimed, sort_keys=True))
    assert out, f"no runs in {results_dir}"
    return out


def two_pipeline_config(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    return write_config(directory, name=name, pipeline={'preset': ['orb', 'sift'], 'masking': ['none']})


@pytest.mark.slow
class TestIdenticalResults:
    def test_cached_and_uncached_campaigns_write_identical_records(self, tmp_path):
        uncached = run_static(two_pipeline_config(tmp_path, 'plain'), args(trial_cache=False))
        cached = run_static(two_pipeline_config(tmp_path, 'cached'), args(trial_cache=True))
        assert records_of(cached) == records_of(uncached)

    def test_each_frame_is_built_once_per_campaign(self, tmp_path, monkeypatch):
        calls = []
        original = static_runner.apply_photometric

        def counting(*a, **k):
            calls.append(1)
            return original(*a, **k)

        monkeypatch.setattr(static_runner, 'apply_photometric', counting)
        config = two_pipeline_config(tmp_path, 'counted')
        run_static(config, args(trial_cache=True))
        scenes = len(static_runner.find_scenes(scene_source()))
        assert len(calls) == scenes * 2  # two trials, built by the first run, read by the second

        calls.clear()
        run_static(config, args(trial_cache=False))
        assert len(calls) == 2 * scenes * 2  # every run rebuilds every frame

    def test_the_cache_is_deleted_when_the_campaign_completes(self, tmp_path):
        config = two_pipeline_config(tmp_path, 'cleared')
        run_static(config, args(trial_cache=True))
        assert not (tmp_path / 'results' / '.cache' / 'cleared').exists()


def entry(width=8, height=6):
    return CachedTrial(
        image=np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3),
        matrix=np.eye(3) * 1.5,
        geometric_params={'rotation_deg': np.float64(1.25), 'scale': 1.0},
        photometric_params={'jpeg_quality': np.int64(80)},
    )


def scene_file(tmp_path, content=b'scene'):
    path = tmp_path / 'scene.jpg'
    path.write_bytes(content)
    return path


class TestTrialCache:
    def test_round_trip_is_exact(self, tmp_path):
        cache = TrialCache(tmp_path / 'cache', 0, {'photometric': []})
        scene = scene_file(tmp_path)
        cache.store(scene, 3, entry())
        loaded = cache.load(scene, 3, 8, 6)
        assert np.array_equal(loaded.image, entry().image)
        assert np.array_equal(loaded.matrix, entry().matrix)
        assert loaded.geometric_params == {'rotation_deg': 1.25, 'scale': 1.0}
        assert loaded.photometric_params == {'jpeg_quality': 80}

    def test_a_different_seed_config_or_scene_content_never_reads_the_entry(self, tmp_path):
        scene = scene_file(tmp_path)
        TrialCache(tmp_path / 'cache', 0, {'photometric': []}).store(scene, 0, entry())
        assert TrialCache(tmp_path / 'cache', 1, {'photometric': []}).load(scene, 0, 8, 6) is None
        assert TrialCache(tmp_path / 'cache', 0, {'photometric': [{'name': 'blur'}]}).load(scene, 0, 8, 6) is None
        scene.write_bytes(b'another scene')
        assert TrialCache(tmp_path / 'cache', 0, {'photometric': []}).load(scene, 0, 8, 6) is None

    def test_an_entry_of_the_wrong_size_is_ignored(self, tmp_path):
        cache = TrialCache(tmp_path / 'cache', 0, {})
        scene = scene_file(tmp_path)
        cache.store(scene, 0, entry())
        assert cache.load(scene, 0, 16, 6) is None

    def test_a_cache_that_cannot_be_written_disables_itself(self, tmp_path):
        blocker = tmp_path / 'not_a_directory'
        blocker.write_text('', encoding='utf-8')
        cache = TrialCache(blocker / 'cache', 0, {})
        scene = scene_file(tmp_path)
        cache.store(scene, 0, entry())
        assert cache.enabled is False
        assert cache.load(scene, 0, 8, 6) is None


class TestDiskPreflight:
    @pytest.mark.slow
    def test_a_cache_that_would_not_fit_is_never_started(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(trial_cache_module, 'DISK_RESERVE_BYTES', 10**18)
        config = two_pipeline_config(tmp_path, 'full')
        run_static(config, args(trial_cache=True))
        assert 'Trial cache off' in capsys.readouterr().out
        assert not (tmp_path / 'results' / '.cache' / 'full').exists()

    def test_the_estimate_is_frames_times_scenes_times_trials(self):
        scene = static_runner.find_scenes(scene_source())[0]
        assert trial_cache_module.estimated_bytes(scene, 3, 5) == 3840 * 2160 * 3 * 15
