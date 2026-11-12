#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
photometric.py - Reference-free alignment quality on the static background.

Each metric compares the reference frame with the current frame warped onto it, over the region
from `regions.metric_region`:

  SSIM      mean structural similarity, over the region eroded by the window radius.
  ZNCC      zero-mean normalized cross-correlation; invariant to affine intensity change.
  MI        mutual information of the joint intensity histogram, in bits.
  Gradient  magnitude-weighted cosine between the two image gradients.

Every metric returns NaN when the region has fewer than `MIN_REGION_PIXELS` pixels.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import cv2
import numpy as np

MIN_REGION_PIXELS = 1024
SSIM_WINDOW = 7
# SSIM constants of Wang et al. (2004), as in skimage's `structural_similarity`.
SSIM_K1, SSIM_K2 = 0.01, 0.03


class PhotometricError(ValueError):
    """Raised when inputs to a photometric metric are inconsistent in shape or type."""


def _gray(image: np.ndarray) -> np.ndarray:
    """Grayscale image in its own dtype."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _prepare(image: np.ndarray) -> np.ndarray:
    """Grayscale float64 view of an image, without rescaling the [0, 255] range."""
    return _gray(image).astype(np.float64)


def ssim_map(a: np.ndarray, b: np.ndarray, window: int = SSIM_WINDOW, data_range: float = 255.0) -> np.ndarray:
    """
    Per-pixel SSIM of two float64 images over uniform `window` x `window` neighborhoods.

    The definition of skimage's `structural_similarity` with its defaults (uniform window, sample
    covariance, mirrored borders), computed with OpenCV box filters.
    """

    def mean(image: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(image, cv2.CV_64F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)

    n = window * window
    covariance_norm = n / (n - 1.0)
    mean_a, mean_b = mean(a), mean(b)
    var_a = covariance_norm * (mean(a * a) - mean_a * mean_a)
    var_b = covariance_norm * (mean(b * b) - mean_b * mean_b)
    cov_ab = covariance_norm * (mean(a * b) - mean_a * mean_b)
    c1, c2 = (SSIM_K1 * data_range) ** 2, (SSIM_K2 * data_range) ** 2
    numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2)
    denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
    return numerator / denominator


def _check(reference: np.ndarray, warped: np.ndarray, region: np.ndarray) -> Optional[str]:
    """Validate shapes and region size; returns a reason string when unusable."""
    if reference.shape[:2] != warped.shape[:2]:
        raise PhotometricError(f"Frame shapes differ: {reference.shape[:2]} vs {warped.shape[:2]}")
    if region.shape != reference.shape[:2]:
        raise PhotometricError(f"Region shape {region.shape} does not match frame shape {reference.shape[:2]}")
    if int(region.sum()) < MIN_REGION_PIXELS:
        return f"region has {int(region.sum())} px, below the {MIN_REGION_PIXELS} px minimum"
    return None


def masked_zncc(reference: np.ndarray, warped: np.ndarray, region: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation over the region, in [-1, 1]."""
    if _check(reference, warped, region) is not None:
        return float('nan')
    a, b = _prepare(reference)[region], _prepare(warped)[region]
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denominator <= 0:
        return float('nan')  # a constant region carries no alignment information
    return float(np.sum(a_centered * b_centered) / denominator)


def masked_ssim(reference: np.ndarray, warped: np.ndarray, region: np.ndarray, window: int = SSIM_WINDOW) -> float:
    """
    Mean structural similarity over the region, in [-1, 1].

    The region is eroded by the window radius first, so no window reaches an excluded pixel.
    """
    if _check(reference, warped, region) is not None:
        return float('nan')

    scores = ssim_map(_prepare(reference), _prepare(warped), window)

    radius = window // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    # Explicit zero border, so the region also erodes at the frame edge.
    interior = cv2.erode(region.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if int(interior.sum()) < MIN_REGION_PIXELS:
        return float('nan')
    return float(scores[interior].mean())


def masked_mutual_information(reference: np.ndarray, warped: np.ndarray, region: np.ndarray, bins: int = 64) -> float:
    """
    Mutual information of the two intensity distributions over the region, in bits.

    Unbounded above; comparable between pipelines on one frame pair.
    """
    if _check(reference, warped, region) is not None:
        return float('nan')

    a, b = _gray(reference)[region], _gray(warped)[region]
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        # The bins of `histogram2d` over [0, 255], by integer arithmetic: no bin edge between two
        # integers is itself an integer, so every value falls in the same bin.
        index_a = np.minimum(a.astype(np.int64) * bins // 255, bins - 1)
        index_b = np.minimum(b.astype(np.int64) * bins // 255, bins - 1)
        counts = np.bincount(index_a * bins + index_b, minlength=bins * bins)
        histogram = counts.reshape(bins, bins).astype(np.float64)
    else:
        histogram, _, _ = np.histogram2d(
            a.astype(np.float64), b.astype(np.float64), bins=bins, range=[[0, 255], [0, 255]]
        )
    joint = histogram / histogram.sum()
    marginal_a, marginal_b = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = joint / (marginal_a * marginal_b)
        terms = np.where(joint > 0, joint * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return float(terms.sum())


def masked_gradient_similarity(reference: np.ndarray, warped: np.ndarray, region: np.ndarray) -> float:
    """
    Magnitude-weighted agreement of gradient orientation over the region, in [-1, 1].

    The cosine between the two gradients at each pixel is weighted by the geometric mean of their
    magnitudes, so flat areas contribute nothing.
    """
    if _check(reference, warped, region) is not None:
        return float('nan')

    a, b = _prepare(reference), _prepare(warped)
    ax, ay = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
    bx, by = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)

    # Only region pixels are summed, so only they are computed. The weighted cosine
    # sqrt(|a||b|) * dot / (|a||b|) is dot / sqrt(|a||b|).
    ax, ay, bx, by = ax[region], ay[region], bx[region], by[region]
    product = np.sqrt((ax * ax + ay * ay) * (bx * bx + by * by))  # |a| |b|
    weights = np.sqrt(product)
    total = weights.sum()
    if total <= 0:
        return float('nan')

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted = np.where(product > 0, (ax * bx + ay * by) / weights, 0.0)
    return float(weighted.sum() / total)


PHOTOMETRIC_METRICS: Dict[str, Callable[..., float]] = {
    'ssim': masked_ssim,
    'zncc': masked_zncc,
    'mi': masked_mutual_information,
    'gradient': masked_gradient_similarity,
}


def photometric_scores(
    reference: np.ndarray,
    warped: np.ndarray,
    region: np.ndarray,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the requested photometric metrics over one region; an unknown name raises.

    `None` requests every metric; an empty sequence requests none.
    """
    names = list(PHOTOMETRIC_METRICS) if metrics is None else list(metrics)
    unknown = [name for name in names if name not in PHOTOMETRIC_METRICS]
    if unknown:
        raise PhotometricError(f"Unknown photometric metric(s) {unknown}. Available: {sorted(PHOTOMETRIC_METRICS)}")

    if names:
        reference, warped = _gray(reference), _gray(warped)  # once, not once per metric
    return {name: PHOTOMETRIC_METRICS[name](reference, warped, region) for name in names}
