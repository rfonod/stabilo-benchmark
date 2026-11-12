#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
seeding.py - Process-independent seeding for each measurement.

`results.seed_everything` seeds numpy, OpenCV and torch once per run; this module provides the
per-measurement keys used by both runners.

OpenCV's robust estimators do not read the global RNG that `cv2.setRNGSeed` sets, and every
method is bit-exact across repeated identical calls except `USAC_PARALLEL`, whose result depends
on thread scheduling (~0.0015 px of corner transfer at 4K). `seed_estimator` therefore does not
control RANSAC sampling; it covers OpenCV routines that do draw from the global stream.
`tests/test_seeding.py` checks these properties against the installed OpenCV.
"""

from __future__ import annotations

import zlib
from typing import Union

import cv2

# Largest seed `cv2.setRNGSeed` accepts without wrapping; the key is reduced modulo this.
_RNG_MODULUS = 2**31 - 1


def stable_key(text: str) -> int:
    """
    Process-independent integer key for a name (a scene, a case, a frame pair).

    CRC32, since Python's `hash` is salted per process.
    """
    return int(zlib.crc32(text.encode('utf-8')))


def seed_estimator(seed: int, *parts: Union[str, int]) -> None:
    """
    Seed OpenCV's global RNG from the identity of the measurement about to be taken.

    `parts` identifies the measurement: (scene, trial) in Part I, (case, frame pair, repeat) in
    Part II. Strings are folded in through `stable_key` and integers directly, summed modulo
    2**31 - 1.
    """
    total = int(seed) + sum(stable_key(part) if isinstance(part, str) else int(part) for part in parts)
    cv2.setRNGSeed(int(total % _RNG_MODULUS))
