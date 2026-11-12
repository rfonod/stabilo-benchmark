#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Geometric and photometric perturbations: ranges, determinism, and recorded parameters."""

from __future__ import annotations

import numpy as np
import pytest

from stabilo_benchmark.perturbations import (
    DEFAULT_GEOMETRIC,
    PHOTOMETRIC_OPS,
    PerturbationError,
    apply_photometric,
    describe_spec,
    sample_geometric,
    warp_image,
)

WIDTH, HEIGHT = 320, 240


def image(seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8)


class TestGeometric:
    def test_matrix_is_invertible_and_the_inverse_is_exact(self):
        sample = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(0))
        assert np.allclose(sample.matrix @ sample.inverse, np.eye(3))

    def test_ranges_are_respected(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            sample = sample_geometric(WIDTH, HEIGHT, DEFAULT_GEOMETRIC, rng)
            assert abs(sample.params['translation_x_px']) <= 0.10 * WIDTH + 1e-9
            assert abs(sample.params['translation_y_px']) <= 0.10 * HEIGHT + 1e-9
            assert abs(sample.params['rotation_deg']) <= 15.0 + 1e-9
            assert 0.95 - 1e-9 <= sample.params['scale'] <= 1.05 + 1e-9

    def test_two_element_range_is_used_as_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            sample = sample_geometric(WIDTH, HEIGHT, {'rotation_deg': [20.0, 30.0]}, rng)
            assert 20.0 <= sample.params['rotation_deg'] <= 30.0

    def test_a_degenerate_range_pins_the_value(self):
        # [v, v] is a constant, not a draw.
        rng = np.random.default_rng(2)
        for _ in range(10):
            sample = sample_geometric(WIDTH, HEIGHT, {'rotation_deg': [45.0, 45.0]}, rng)
            assert sample.params['rotation_deg'] == pytest.approx(45.0)

    def test_zero_perturbation_gives_the_identity(self):
        config = {'translation_ratio': 0.0, 'rotation_deg': 0.0, 'scale_ratio': 0.0, 'projective': 0.0}
        sample = sample_geometric(WIDTH, HEIGHT, config, np.random.default_rng(0))
        assert np.allclose(sample.matrix, np.eye(3), atol=1e-12)

    def test_same_generator_seed_reproduces_the_matrix(self):
        a = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(5))
        b = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(5))
        assert np.array_equal(a.matrix, b.matrix)

    def test_independent_of_global_numpy_state(self):
        # Draws do not depend on the global numpy state.
        np.random.seed(1)
        a = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(9))
        np.random.seed(999)
        np.random.rand(1000)
        b = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(9))
        assert np.array_equal(a.matrix, b.matrix)

    def test_every_parameter_is_recorded(self):
        params = sample_geometric(WIDTH, HEIGHT, {}, np.random.default_rng(0)).params
        assert set(params) == {
            'translation_x_px',
            'translation_y_px',
            'rotation_deg',
            'scale',
            'projective_p1',
            'projective_p2',
        }

    def test_malformed_range_raises(self):
        with pytest.raises(PerturbationError, match='exactly two elements'):
            sample_geometric(WIDTH, HEIGHT, {'rotation_deg': [1.0, 2.0, 3.0]}, np.random.default_rng(0))

    def test_unordered_range_raises(self):
        with pytest.raises(PerturbationError, match='ordered'):
            sample_geometric(WIDTH, HEIGHT, {'rotation_deg': [10.0, 1.0]}, np.random.default_rng(0))


class TestWarpImage:
    def test_identity_leaves_the_image_unchanged(self):
        source = image()
        assert np.array_equal(warp_image(source, np.eye(3)), source)

    def test_size_is_preserved_by_default(self):
        assert warp_image(image(), np.eye(3)).shape == (HEIGHT, WIDTH, 3)


class TestPhotometric:
    def test_empty_spec_is_a_no_op(self):
        source = image()
        result, sampled = apply_photometric(source, [], np.random.default_rng(0))
        assert np.array_equal(result, source)
        assert sampled == {}

    def test_every_registered_op_runs_and_reports_its_draw(self):
        source = image()
        for name in PHOTOMETRIC_OPS:
            result, sampled = apply_photometric(source, [{'name': name}], np.random.default_rng(0))
            assert result.shape == source.shape
            assert result.dtype == np.uint8
            assert sampled, f"'{name}' recorded no parameters"

    def test_unknown_op_raises_with_the_available_names(self):
        with pytest.raises(PerturbationError, match='Unknown photometric perturbation'):
            apply_photometric(image(), [{'name': 'vignette'}], np.random.default_rng(0))

    def test_chain_is_applied_in_the_listed_order(self):
        # The chain is applied in the listed order.
        source = image()
        forward, _ = apply_photometric(
            source, [{'name': 'noise', 'max_sigma': 20}, {'name': 'blur', 'max_ksize': 9}], np.random.default_rng(3)
        )
        reverse, _ = apply_photometric(
            source, [{'name': 'blur', 'max_ksize': 9}, {'name': 'noise', 'max_sigma': 20}], np.random.default_rng(3)
        )
        assert not np.array_equal(forward, reverse)

    def test_all_sampled_parameters_are_merged(self):
        _, sampled = apply_photometric(
            image(),
            [{'name': 'brightness'}, {'name': 'noise', 'max_sigma': 2}, {'name': 'jpeg', 'quality': [60, 90]}],
            np.random.default_rng(0),
        )
        assert {'brightness_factor', 'noise_sigma', 'jpeg_quality'} <= set(sampled)

    def test_jpeg_quality_stays_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            _, sampled = apply_photometric(image(), [{'name': 'jpeg', 'quality': [40, 95]}], rng)
            assert 40 <= sampled['jpeg_quality'] <= 95

    def test_jpeg_actually_degrades_the_image(self):
        source = image()
        result, _ = apply_photometric(source, [{'name': 'jpeg', 'quality': [10, 10]}], np.random.default_rng(0))
        assert not np.array_equal(result, source)

    def test_blur_kernel_is_odd(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            _, sampled = apply_photometric(image(), [{'name': 'blur', 'max_ksize': 11}], rng)
            assert sampled['blur_ksize'] % 2 == 1

    def test_zero_noise_leaves_the_image_untouched(self):
        source = image()
        result, sampled = apply_photometric(source, [{'name': 'noise', 'max_sigma': 0.0}], np.random.default_rng(0))
        assert sampled['noise_sigma'] == 0.0
        assert np.array_equal(result, source)

    def test_saturation_is_a_no_op_on_grayscale(self):
        gray = image()[:, :, 0]
        result, sampled = apply_photometric(gray, [{'name': 'saturation'}], np.random.default_rng(0))
        assert np.array_equal(result, gray)
        assert sampled['saturation_factor'] == 1.0

    def test_fog_brightens_toward_white(self):
        source = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        result, sampled = apply_photometric(source, [{'name': 'fog', 'max_density': 0.5}], np.random.default_rng(1))
        assert sampled['fog_density'] > 0
        assert result.mean() > source.mean()

    def test_same_generator_seed_reproduces_the_chain(self):
        spec = [{'name': 'brightness'}, {'name': 'blur', 'max_ksize': 7}, {'name': 'noise', 'max_sigma': 5}]
        a, _ = apply_photometric(image(), spec, np.random.default_rng(11))
        b, _ = apply_photometric(image(), spec, np.random.default_rng(11))
        assert np.array_equal(a, b)


class TestDescribeSpec:
    def test_summarizes_both_families(self):
        lines = describe_spec({'geometric': {}, 'photometric': [{'name': 'blur', 'max_ksize': 5}]})
        assert any('geometric' in line for line in lines)
        assert any('blur' in line for line in lines)

    def test_handles_range_valued_geometry(self):
        assert describe_spec({'geometric': {'rotation_deg': [30, 30]}})
