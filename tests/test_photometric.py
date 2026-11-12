#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Reference-free alignment metrics: a perfect match, a misalignment, an illumination change, and a too-small region."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stabilo_benchmark.metrics.photometric import (
    MIN_REGION_PIXELS,
    PHOTOMETRIC_METRICS,
    PhotometricError,
    masked_gradient_similarity,
    masked_mutual_information,
    masked_ssim,
    masked_zncc,
    photometric_scores,
    ssim_map,
)

HEIGHT, WIDTH = 200, 260


@pytest.fixture
def textured():
    """A textured frame: random noise blurred into structure, so gradients are meaningful."""
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur((rng.random((HEIGHT, WIDTH)) * 255).astype(np.uint8), (7, 7), 0)


@pytest.fixture
def region():
    return np.ones((HEIGHT, WIDTH), dtype=bool)


class TestPerfectMatch:
    def test_all_metrics_saturate_on_an_identical_pair(self, textured, region):
        assert masked_ssim(textured, textured, region) == pytest.approx(1.0, abs=1e-6)
        assert masked_zncc(textured, textured, region) == pytest.approx(1.0, abs=1e-9)
        assert masked_gradient_similarity(textured, textured, region) == pytest.approx(1.0, abs=1e-6)
        assert masked_mutual_information(textured, textured, region) > 3.0


class TestMisalignment:
    def test_every_metric_drops_when_shifted(self, textured, region):
        shifted = np.roll(textured, 6, axis=1)
        assert masked_ssim(textured, shifted, region) < 0.5
        assert masked_zncc(textured, shifted, region) < 0.5
        assert masked_gradient_similarity(textured, shifted, region) < 0.5
        assert masked_mutual_information(textured, shifted, region) < masked_mutual_information(
            textured, textured, region
        )

    def test_scores_decrease_monotonically_with_shift(self, textured, region):
        scores = [masked_zncc(textured, np.roll(textured, shift, axis=1), region) for shift in (0, 2, 5, 10)]
        assert scores == sorted(scores, reverse=True)


class TestIlluminationInvariance:
    def test_zncc_ignores_an_affine_intensity_change(self, textured, region):
        # A brightness difference between two frames must not masquerade as misalignment.
        brighter = np.clip(textured.astype(np.float64) * 0.8 + 25, 0, 255).astype(np.uint8)
        assert masked_zncc(textured, brighter, region) > 0.99

    def test_gradient_similarity_ignores_it_too(self, textured, region):
        brighter = np.clip(textured.astype(np.float64) * 0.8 + 25, 0, 255).astype(np.uint8)
        assert masked_gradient_similarity(textured, brighter, region) > 0.99


class TestRegionRestriction:
    def test_a_disagreement_outside_the_region_is_ignored(self, textured):
        corrupted = textured.copy()
        corrupted[:60, :] = 0

        inside = np.zeros((HEIGHT, WIDTH), dtype=bool)
        inside[80:, :] = True
        assert masked_zncc(textured, corrupted, inside) == pytest.approx(1.0, abs=1e-9)

        everywhere = np.ones((HEIGHT, WIDTH), dtype=bool)
        assert masked_zncc(textured, corrupted, everywhere) < 1.0

    def test_ssim_erodes_the_region_by_the_window_radius(self, textured):
        # The region is eroded by the window radius.
        corrupted = textured.copy()
        corrupted[:100, :] = 0
        boundary = np.zeros((HEIGHT, WIDTH), dtype=bool)
        boundary[100:, :] = True
        assert masked_ssim(textured, corrupted, boundary) == pytest.approx(1.0, abs=1e-3)

    def test_statistics_are_taken_over_the_region_only(self, textured):
        # An excluded bright block must not shift the means that normalize ZNCC.
        with_block = textured.copy()
        with_block[:50, :] = 255
        region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        region[60:, :] = True
        assert masked_zncc(textured, with_block, region) == pytest.approx(1.0, abs=1e-9)


class TestDegenerateInput:
    def test_a_tiny_region_is_nan_not_a_flattering_score(self, textured):
        tiny = np.zeros((HEIGHT, WIDTH), dtype=bool)
        tiny[0:5, 0:5] = True
        for metric in PHOTOMETRIC_METRICS.values():
            assert np.isnan(metric(textured, textured, tiny))

    def test_the_threshold_is_where_it_says_it_is(self, textured):
        region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        region.reshape(-1)[: MIN_REGION_PIXELS - 1] = True
        assert np.isnan(masked_zncc(textured, textured, region))
        region.reshape(-1)[: MIN_REGION_PIXELS + 4000] = True
        assert not np.isnan(masked_zncc(textured, textured, region))

    def test_a_constant_region_carries_no_information(self, region):
        flat = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
        assert np.isnan(masked_zncc(flat, flat, region))
        assert np.isnan(masked_gradient_similarity(flat, flat, region))

    def test_mismatched_frame_shapes_raise(self, textured, region):
        with pytest.raises(PhotometricError, match='shapes differ'):
            masked_zncc(textured, textured[:100], region)

    def test_mismatched_region_shape_raises(self, textured):
        with pytest.raises(PhotometricError, match='does not match'):
            masked_zncc(textured, textured, np.ones((10, 10), dtype=bool))


class TestColorHandling:
    def test_color_and_grayscale_agree(self, region):
        rng = np.random.default_rng(1)
        gray = cv2.GaussianBlur((rng.random((HEIGHT, WIDTH)) * 255).astype(np.uint8), (5, 5), 0)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert masked_zncc(color, color, region) == pytest.approx(masked_zncc(gray, gray, region), abs=1e-6)


class TestScoresDict:
    def test_returns_every_requested_metric(self, textured, region):
        scores = photometric_scores(textured, textured, region, metrics=['ssim', 'zncc'])
        assert set(scores) == {'ssim', 'zncc'}

    def test_defaults_to_all_metrics(self, textured, region):
        assert set(photometric_scores(textured, textured, region)) == set(PHOTOMETRIC_METRICS)

    def test_an_empty_list_requests_none(self, textured, region):
        # `metrics.photometric: []` (Part II tuning) must not fall back to every metric.
        assert photometric_scores(textured, textured, region, metrics=[]) == {}

    def test_an_unknown_metric_raises_rather_than_being_skipped(self, textured, region):
        with pytest.raises(PhotometricError, match='Unknown photometric metric'):
            photometric_scores(textured, textured, region, metrics=['ssim', 'psnr'])


class TestReferenceImplementations:
    """The fast SSIM and MI reproduce skimage's SSIM and numpy's joint histogram."""

    @staticmethod
    def _pair(seed=1):
        rng = np.random.default_rng(seed)
        a = cv2.GaussianBlur((rng.random((HEIGHT, WIDTH)) * 255).astype(np.uint8), (5, 5), 0)
        shift = np.float32([[1, 0, 1.3], [0, 1, -0.7]])
        b = cv2.warpAffine(a, shift, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 6, a.shape)
        return a, np.clip(b + noise, 0, 255).astype(np.uint8)

    def test_ssim_map_matches_skimage(self):
        skimage_metrics = pytest.importorskip('skimage.metrics')
        a, b = (image.astype(np.float64) for image in self._pair())
        _, expected = skimage_metrics.structural_similarity(a, b, win_size=7, data_range=255.0, full=True)
        assert np.allclose(ssim_map(a, b, 7), expected, rtol=0, atol=1e-12)

    def test_mutual_information_matches_the_float_histogram(self, region):
        a, b = self._pair()
        fast = masked_mutual_information(a, b, region)
        slow = masked_mutual_information(a.astype(np.float64), b.astype(np.float64), region)
        assert fast == slow

    def test_the_top_intensity_falls_in_the_last_bin(self, region):
        a = np.full((HEIGHT, WIDTH), 255, np.uint8)
        a[: HEIGHT // 2] = 3
        assert masked_mutual_information(a, a, region) == masked_mutual_information(
            a.astype(np.float64), a.astype(np.float64), region
        )

    def test_gradient_similarity_matches_its_definition(self, region):
        a, b = (image.astype(np.float64) for image in self._pair(seed=3))
        ax, ay = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        bx, by = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_a, magnitude_b = np.hypot(ax, ay), np.hypot(bx, by)
        weights = np.sqrt(magnitude_a * magnitude_b)
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine = np.where(magnitude_a * magnitude_b > 0, (ax * bx + ay * by) / (magnitude_a * magnitude_b), 0.0)
        expected = np.sum(weights * cosine) / weights.sum()
        assert masked_gradient_similarity(a, b, region) == pytest.approx(expected, rel=1e-12, abs=1e-15)
