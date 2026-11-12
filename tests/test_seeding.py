#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
test_seeding.py - The per-measurement seeding key, and properties of the installed OpenCV.

`TestOpenCVEstimatorDeterminism` checks that robust estimation never reads the global RNG, that
every method except `USAC_PARALLEL` is bit-exact across repeated calls, and that `USAC_PARALLEL`
stays within a bound. If these fail after an OpenCV upgrade, the reproducibility statements in
`seeding` and `runners/dynamic` need revisiting.
"""

from __future__ import annotations

import subprocess
import sys
import zlib

import cv2
import numpy as np
import pytest

from stabilo_benchmark.seeding import seed_estimator, stable_key

# Every method the estimator campaign sweeps.
HOMOGRAPHY_METHODS = (
    cv2.LMEDS,
    cv2.RANSAC,
    cv2.RHO,
    cv2.USAC_DEFAULT,
    cv2.USAC_PARALLEL,
    cv2.USAC_FAST,
    cv2.USAC_ACCURATE,
    cv2.USAC_PROSAC,
    cv2.USAC_MAGSAC,
)
# USAC_PARALLEL varies with thread scheduling; it has its own bounded test.
NONDETERMINISTIC = (cv2.USAC_PARALLEL,)
DETERMINISTIC = tuple(m for m in HOMOGRAPHY_METHODS if m not in NONDETERMINISTIC)

SEEDS = (0, 7, 999, 12345)

# Corner transfer tolerance for USAC_PARALLEL, in pixels on a 4K frame (observed ~0.0015 px).
PARALLEL_TOLERANCE_PX = 0.05

FRAME_CORNERS = np.array([[0, 0], [3840, 0], [3840, 2160], [0, 2160]], dtype=np.float64)


def _correspondences(outlier_fraction: float = 0.7, n: int = 400):
    """A hard but solvable correspondence set: a known homography plus gross outliers."""
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (n, 1, 2)).astype(np.float32)
    matrix = np.array([[1.01, 0.02, 7.0], [-0.02, 1.0, -3.0], [1e-6, 2e-6, 1.0]])
    dst = cv2.perspectiveTransform(src, matrix)
    corrupt = int(n * outlier_fraction)
    dst[:corrupt] += rng.uniform(-200, 200, (corrupt, 1, 2)).astype(np.float32)
    return src, dst


def _estimate(src, dst, method):
    matrix, _ = cv2.findHomography(
        src, dst, method=method, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999999
    )
    return np.eye(3) if matrix is None else np.asarray(matrix, dtype=np.float64)


def _transfer(matrix, points=FRAME_CORNERS):
    """Corner transfer in float64, matching `regions.warp_points` rather than OpenCV."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


class TestStableKey:
    def test_is_the_crc32_of_the_name(self):
        assert stable_key('scene_a') == zlib.crc32(b'scene_a')

    def test_survives_a_fresh_interpreter_with_a_different_hash_seed(self):
        # Run in a subprocess: the key must not depend on the process.
        code = 'from stabilo_benchmark.seeding import stable_key; print(stable_key("scene_a"))'
        outputs = {
            subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                check=True,
                env={'PYTHONHASHSEED': seed, 'PATH': '/usr/bin:/bin'},
            ).stdout.strip()
            for seed in ('0', '1', '42')
        }
        assert outputs == {str(stable_key('scene_a'))}

    def test_distinguishes_names(self):
        assert stable_key('scene_a') != stable_key('scene_b')


class TestSeedEstimatorKey:
    @staticmethod
    def _draw():
        return cv2.randu(np.zeros(4, np.float32), 0, 1).copy()

    def test_distinct_measurements_get_distinct_keys(self):
        seed_estimator(0, 'case', 'pair_0001', 0)
        first = self._draw()
        seed_estimator(0, 'case', 'pair_0001', 1)
        assert not np.array_equal(first, self._draw())

    def test_the_same_measurement_gets_the_same_key(self):
        seed_estimator(3, 'case', 'pair_0001', 0)
        first = self._draw()
        seed_estimator(3, 'case', 'pair_0001', 0)
        assert np.array_equal(first, self._draw())

    def test_matches_the_part_one_key_formula(self):
        # Key: seed + crc32(scene) + trial, modulo 2**31 - 1.
        seed, scene, trial = 5, 'songdo_a.jpg', 11
        seed_estimator(seed, scene, trial)
        through_the_helper = self._draw()

        cv2.setRNGSeed(int((seed + stable_key(scene) + trial) % (2**31 - 1)))
        assert np.array_equal(through_the_helper, self._draw())


class TestOpenCVEstimatorDeterminism:
    @pytest.mark.parametrize('method', HOMOGRAPHY_METHODS)
    def test_findhomography_ignores_the_global_seed(self, method):
        src, dst = _correspondences()
        results = []
        for seed in SEEDS:
            cv2.setRNGSeed(seed)
            results.append(_transfer(_estimate(src, dst, method)))
        # Bounded rather than exact, so USAC_PARALLEL fits the same test.
        for other in results[1:]:
            assert np.linalg.norm(results[0] - other, axis=1).max() < PARALLEL_TOLERANCE_PX

    @pytest.mark.parametrize('method', DETERMINISTIC)
    def test_repeated_identical_calls_are_bit_exact(self, method):
        src, dst = _correspondences()
        results = [_estimate(src, dst, method) for _ in range(20)]
        assert len({matrix.tobytes() for matrix in results}) == 1

    def test_the_parallel_variant_varies_but_far_below_a_pixel(self):
        # USAC_PARALLEL may differ between identical calls; bound the difference.
        src, dst = _correspondences()
        transfers = np.stack([_transfer(_estimate(src, dst, cv2.USAC_PARALLEL)) for _ in range(20)])
        spread = np.linalg.norm(transfers - transfers[0], axis=2).max()
        assert spread < PARALLEL_TOLERANCE_PX

    def test_estimation_does_not_advance_the_global_stream(self):
        # An estimation does not advance the global stream.
        src, dst = _correspondences()
        cv2.setRNGSeed(0)
        before = cv2.randu(np.zeros(4, np.float32), 0, 1).copy()

        cv2.setRNGSeed(0)
        _estimate(src, dst, cv2.USAC_MAGSAC)
        after = cv2.randu(np.zeros(4, np.float32), 0, 1).copy()

        assert np.array_equal(before, after)

    def test_position_in_a_sequence_does_not_change_an_estimate(self):
        src, dst = _correspondences()
        first = _estimate(src, dst, cv2.USAC_MAGSAC)
        for _ in range(25):
            _estimate(src, dst, cv2.RANSAC)
            cv2.randu(np.zeros((50, 50), np.float32), 0, 1)
        assert np.array_equal(first, _estimate(src, dst, cv2.USAC_MAGSAC))
