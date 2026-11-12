#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
environment.py - Checks on the installed environment, run before a command starts.

Imports nothing heavier than `cv2`, so `cli.main` can call it on every invocation.

OpenCV 5.0 dropped the Python bindings for BRISK, KAZE and AKAZE. `opencv-python`,
`opencv-python-headless`, `opencv-contrib-python` and `opencv-contrib-python-headless` all
install the same `cv2` package, so the `<5.0` bound on one of them does not stop another from
providing OpenCV 5. `check_opencv` detects the missing bindings and names the distribution to
reinstall.
"""

from __future__ import annotations

import os
import sys
from importlib import metadata
from typing import List, Optional, Tuple

import cv2

# The detectors OpenCV 5.0 left without Python bindings.
UNBOUND_IN_OPENCV5 = ('brisk', 'kaze', 'akaze')

# Every distribution that installs a top-level `cv2`.
CV2_DISTRIBUTIONS = (
    'opencv-python',
    'opencv-python-headless',
    'opencv-contrib-python',
    'opencv-contrib-python-headless',
)

# Set to downgrade the check to a warning, for grids without these detectors;
# `pipelines.unavailable_detectors` still refuses a grid that needs them.
ALLOW_VARIABLE = 'STABILO_BENCHMARK_ALLOW_OPENCV5'


class EnvironmentCheckError(RuntimeError):
    """The installed environment cannot run the command that was asked for."""


def missing_detectors() -> List[str]:
    """Which of `UNBOUND_IN_OPENCV5` the imported `cv2` cannot construct."""
    return [name for name in UNBOUND_IN_OPENCV5 if not hasattr(cv2, f"{name.upper()}_create")]


def installed_cv2_distributions() -> List[Tuple[str, str]]:
    """The `cv2`-providing distributions present in this environment, as (name, version)."""
    found = []
    for name in CV2_DISTRIBUTIONS:
        try:
            found.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    return found


def _provides_imported_cv2(distribution_version: str) -> bool:
    """
    Whether this distribution's version is the one `cv2.__version__` reports.

    Compared on the first three components: the distributions carry a fourth, build-level
    component that the module does not (`opencv-python 4.14.0.94` imports as `4.14.0`).
    """
    return distribution_version.split('.')[:3] == cv2.__version__.split('.')[:3]


def opencv_binding_hint() -> str:
    """
    Why the detectors are missing and what to reinstall on this machine.

    The distribution whose version matches the imported `cv2` is the one to reinstall; the others are
    listed too.
    """
    distributions = installed_cv2_distributions()
    lines = [
        f"OpenCV {cv2.__version__} provides no Python bindings for "
        f"{', '.join(d.upper() for d in UNBOUND_IN_OPENCV5)}; they were dropped in 5.0, and three "
        f"of the eleven detector presets need them.",
    ]

    if distributions:
        lines.append('')
        lines.append('These distributions all install the same `cv2`, and the last one installed wins:')
        for name, version in distributions:
            marker = '  <- the one that imports' if _provides_imported_cv2(version) else ''
            lines.append(f"    {name:<32} {version}{marker}")

    offenders = [name for name, version in distributions if not version.startswith('4.')]
    targets = offenders or ['opencv-python']
    pins = ' '.join(f"'{name}<5.0.0'" for name in targets)
    lines.append('')
    lines.append(f"Reinstall below 5.0, in this environment:\n    pip install -U {pins}")
    lines.append(
        'Then re-check:\n'
        "    python -c \"import cv2; print(cv2.__version__, "
        "[hasattr(cv2, f'{d}_create') for d in ('BRISK','KAZE','AKAZE')])\""
    )
    return '\n'.join(lines)


def check_opencv(strict: bool = True, stream: Optional[object] = None) -> bool:
    """
    Verify that this OpenCV can construct the detectors the benchmark's axis names.

    Args:
        strict: Raise when they are missing; False warns and continues.
        stream: Where a warning goes (default `sys.stderr`).

    Returns:
        True when the bindings are all present.

    Raises:
        EnvironmentCheckError: under `strict`, unless `ALLOW_VARIABLE` is set.
    """
    if not missing_detectors():
        return True

    if strict and not os.environ.get(ALLOW_VARIABLE):
        raise EnvironmentCheckError(
            f"{opencv_binding_hint()}\n\nTo run anyway, with a grid that names none of them, set {ALLOW_VARIABLE}=1."
        )

    print(f"Warning: {opencv_binding_hint()}", file=stream or sys.stderr)
    return False
