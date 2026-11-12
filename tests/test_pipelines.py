#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Pipeline specifications: estimator names, processed pixels, and the OpenCV detector checks."""

from __future__ import annotations

import cv2
import pytest

from stabilo_benchmark import environment
from stabilo_benchmark.pipelines import (
    UNBOUND_IN_OPENCV5,
    make_spec,
    opencv_binding_hint,
    processed_pixels,
    ransac_name,
    unavailable_detectors,
)

# The nine codes `configs/songdo/static/estimators.yaml` sweeps, with the name each renders as.
EXPECTED_NAMES = {
    cv2.LMEDS: 'LMEDS',
    cv2.RANSAC: 'RANSAC',
    cv2.RHO: 'RHO',
    cv2.USAC_DEFAULT: 'USAC-Default (LO-RANSAC)',
    cv2.USAC_PARALLEL: 'USAC-Parallel (LO-RANSAC)',
    cv2.USAC_FAST: 'USAC-Fast (LO-RANSAC)',
    cv2.USAC_ACCURATE: 'USAC-Accurate (GC-RANSAC)',
    cv2.USAC_PROSAC: 'USAC-PROSAC (PROSAC)',
    cv2.USAC_MAGSAC: 'USAC-MAGSAC (MAGSAC++)',
}


class TestProcessedPixels:
    """The work proxy equals the frame size stabilo's resize produces."""

    @pytest.mark.parametrize(('width', 'height'), [(3840, 2160), (1920, 1080), (1001, 777)])
    @pytest.mark.parametrize('ratio', [0.18, 0.30, 0.50, 0.55, 1.00])
    def test_it_matches_what_cv2_resize_produces(self, width, height, ratio):
        import numpy as np  # noqa: PLC0415

        resized = cv2.resize(np.zeros((height, width), np.uint8), (0, 0), fx=ratio, fy=ratio)
        spec = make_spec({'preset': 'orb', 'masking': 'none', 'downsample_ratio': ratio})
        assert processed_pixels(spec, width, height) == resized.shape[0] * resized.shape[1]


class TestRansacName:
    @pytest.mark.parametrize(('method', 'expected'), sorted(EXPECTED_NAMES.items()))
    def test_names_every_swept_method(self, method, expected):
        assert ransac_name(method) == expected

    def test_no_method_is_called_degensac(self):
        # None of OpenCV's homography estimators is DEGENSAC.
        assert not any('DEGENSAC' in ransac_name(method).upper() for method in EXPECTED_NAMES)

    def test_the_three_lo_ransac_variants_are_distinguishable(self):
        # Same algorithm, distinct labels.
        names = {ransac_name(m) for m in (cv2.USAC_DEFAULT, cv2.USAC_PARALLEL, cv2.USAC_FAST)}
        assert len(names) == 3
        assert all('LO-RANSAC' in name for name in names)

    def test_an_unknown_code_falls_back_to_its_number(self):
        assert ransac_name(999) == '999'


class TestDetectorAvailability:
    """
    OpenCV 5 dropped the Python bindings for BRISK, KAZE and AKAZE; a grid naming them is refused
    before it runs.
    """

    AFFECTED = ('brisk', 'kaze', 'akaze')

    def test_the_affected_set_matches_what_stabilo_guards(self):
        assert set(UNBOUND_IN_OPENCV5) == set(self.AFFECTED)

    def test_available_detectors_are_never_reported_missing(self):
        specs = [make_spec({'preset': preset, 'masking': 'none'}) for preset in ('orb', 'sift', 'rsift')]
        assert unavailable_detectors(specs) == []

    def test_verdict_follows_the_installed_opencv(self):
        specs = [make_spec({'preset': preset, 'masking': 'none'}) for preset in self.AFFECTED]
        expected = [] if hasattr(cv2, 'AKAZE_create') else [spec.name for spec in specs]
        assert unavailable_detectors(specs) == expected

    def test_the_hint_names_the_installed_distribution_to_reinstall(self, monkeypatch):
        # The hint names the distribution that provides the imported cv2.
        monkeypatch.setattr(
            environment, 'installed_cv2_distributions', lambda: [('opencv-python-headless', '5.0.0.86')]
        )
        hint = opencv_binding_hint()
        assert "pip install -U 'opencv-python-headless<5.0.0'" in hint
        assert 'the last one installed wins' in hint

    def test_a_pipeline_reports_under_its_own_name(self, monkeypatch):
        monkeypatch.delattr(cv2, 'AKAZE_create', raising=False)
        spec = make_spec({'preset': 'akaze', 'masking': 'obb'})
        assert unavailable_detectors([spec]) == [spec.name]
        assert 'akaze' in spec.name

    def test_the_installed_opencv_satisfies_the_declared_dependency(self):
        """
        The installed `cv2` is 4.x with all three detectors.

        Four distributions install the same `cv2` package, so the declared bound can be satisfied while
        the imported build is 5.x. The failure message names the distribution to reinstall.
        """
        major = int(cv2.__version__.split('.')[0])
        assert major == 4, opencv_binding_hint()
        assert environment.missing_detectors() == [], opencv_binding_hint()


class TestTheEnvironmentGate:
    """
    `environment.check_opencv`, which `cli.main` runs before dispatching a handler.

    Strict for the commands that construct a `Stabilizer`, advisory for the others, and downgraded to
    a warning by `ALLOW_VARIABLE`.
    """

    @staticmethod
    def _break_opencv(monkeypatch):
        for detector in UNBOUND_IN_OPENCV5:
            monkeypatch.delattr(cv2, f'{detector.upper()}_create', raising=False)
        monkeypatch.delenv(environment.ALLOW_VARIABLE, raising=False)

    def test_a_healthy_environment_passes_quietly(self, capsys):
        assert environment.check_opencv(strict=True) is True
        assert capsys.readouterr().err == ''

    def test_a_running_command_is_refused(self, monkeypatch):
        self._break_opencv(monkeypatch)
        with pytest.raises(environment.EnvironmentCheckError) as excinfo:
            environment.check_opencv(strict=True)
        assert 'BRISK' in str(excinfo.value)
        assert environment.ALLOW_VARIABLE in str(excinfo.value)

    def test_a_command_that_runs_no_pipeline_only_warns(self, monkeypatch, capsys):
        self._break_opencv(monkeypatch)
        assert environment.check_opencv(strict=False) is False
        assert 'BRISK' in capsys.readouterr().err

    def test_the_escape_downgrades_the_refusal_to_a_warning(self, monkeypatch, capsys):
        self._break_opencv(monkeypatch)
        monkeypatch.setenv(environment.ALLOW_VARIABLE, '1')
        assert environment.check_opencv(strict=True) is False
        assert 'BRISK' in capsys.readouterr().err

    def test_the_escape_does_not_reach_the_grid_preflight(self, monkeypatch):
        # The variable does not bypass the grid preflight.
        self._break_opencv(monkeypatch)
        monkeypatch.setenv(environment.ALLOW_VARIABLE, '1')
        spec = make_spec({'preset': 'akaze', 'masking': 'none'})
        assert unavailable_detectors([spec]) == [spec.name]

    def test_the_distribution_build_component_still_matches_the_module(self):
        # `opencv-python 4.14.0.94` imports as `4.14.0`; a naive equality never marks it.
        assert environment._provides_imported_cv2(f'{cv2.__version__}.94')
        assert not environment._provides_imported_cv2('9.9.9.1')
