#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
perturbations.py - Synthetic geometric and photometric perturbations for the static benchmark.

A known homography and a photometric degradation are applied to a real aerial frame; the inverse
of the homography is the ground truth. Every range comes from the config, and every sampled value
is returned with the perturbed image.

Range syntax, accepted for every numeric parameter:
    0.10          symmetric uniform draw from [-0.10, +0.10] (or [1-r, 1+r] for factors)
    [lo, hi]      uniform draw from [lo, hi]
    [v, v]        a fixed value

Draws come from an explicit `numpy.random.Generator`, not the global numpy state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_GEOMETRIC: Dict[str, Any] = {
    'translation_ratio': 0.10,
    'rotation_deg': 15.0,
    'scale_ratio': 0.05,
    'projective': 5.0e-5,
}


class PerturbationError(ValueError):
    """Raised for an unknown perturbation name or a malformed range."""


# --------------------------------------------------------------------------------------
# Range handling
# --------------------------------------------------------------------------------------


def _uniform(rng: np.random.Generator, value: Any, symmetric_about: float = 0.0) -> float:
    """
    Draw one sample from a config range.

    A scalar `v` means uniform on [symmetric_about - v, symmetric_about + v]; `symmetric_about` is 0
    for additive quantities (translation, rotation) and 1 for multiplicative ones (scale, brightness).
    """
    if value is None:
        return float(symmetric_about)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise PerturbationError(f"A range must have exactly two elements [lo, hi], got {value!r}")
        low, high = float(value[0]), float(value[1])
        if low > high:
            raise PerturbationError(f"Range [lo, hi] must be ordered, got {value!r}")
        return float(rng.uniform(low, high)) if high > low else low
    magnitude = float(value)
    return float(rng.uniform(symmetric_about - magnitude, symmetric_about + magnitude))


# --------------------------------------------------------------------------------------
# Geometric perturbation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometricPerturbation:
    """A sampled homography together with the parameters that generated it."""

    matrix: np.ndarray
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def inverse(self) -> np.ndarray:
        """
        The ground-truth transform a pipeline should recover.

        `matrix` maps reference -> perturbed; the estimate is compared with its inverse.
        """
        return np.linalg.inv(self.matrix)


def sample_geometric(
    width: int,
    height: int,
    config: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeometricPerturbation:
    """
    Sample a homography imitating the frame-to-frame motion of a hovering drone.

    Composed as T @ R @ S @ P: translation, rotation about the frame center, uniform scale, and a
    small projective term.
    """
    rng = rng if rng is not None else np.random.default_rng()
    config = {**DEFAULT_GEOMETRIC, **(config or {})}

    tx = _uniform(rng, config['translation_ratio']) * width
    ty = _uniform(rng, config['translation_ratio']) * height
    angle = _uniform(rng, config['rotation_deg'])
    scale = _uniform(rng, config['scale_ratio'], symmetric_about=1.0)
    p1 = _uniform(rng, config['projective'])
    p2 = _uniform(rng, config['projective'])

    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    rotation = np.vstack([cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0), [0.0, 0.0, 1.0]])
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    projective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [p1, p2, 1.0]])

    matrix = translation @ rotation @ scaling @ projective
    params = {
        'translation_x_px': tx,
        'translation_y_px': ty,
        'rotation_deg': angle,
        'scale': scale,
        'projective_p1': p1,
        'projective_p2': p2,
    }
    return GeometricPerturbation(matrix=matrix, params=params)


def warp_image(
    image: np.ndarray, matrix: np.ndarray, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    """Warp an image by a homography, keeping its size unless one is given."""
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, np.asarray(matrix, dtype=np.float64), (width or w, height or h))


# --------------------------------------------------------------------------------------
# Photometric perturbations
# --------------------------------------------------------------------------------------


def _brightness(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Multiplicative intensity change: time of day, exposure drift, cloud shadow."""
    factor = _uniform(rng, params.get('range', 0.25), symmetric_about=1.0)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0), {'brightness_factor': factor}


def _contrast(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Contrast change about mid-gray, which flattens or exaggerates local gradients."""
    factor = _uniform(rng, params.get('range', 0.20), symmetric_about=1.0)
    return cv2.convertScaleAbs(image, alpha=factor, beta=128.0 * (1.0 - factor)), {'contrast_factor': factor}


def _saturation(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Color saturation change."""
    factor = _uniform(rng, params.get('range', 0.05), symmetric_about=1.0)
    if image.ndim != 3:
        return image, {'saturation_factor': 1.0}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), {'saturation_factor': factor}


def _blur(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Gaussian blur: defocus and the residual motion blur of a hovering platform."""
    max_ksize = int(params.get('max_ksize', 5))
    ksize = int(rng.integers(0, max(max_ksize, 1)))
    ksize = 2 * (ksize // 2) + 1  # odd kernel size, as cv2.GaussianBlur requires
    return cv2.GaussianBlur(image, (ksize, ksize), 0), {'blur_ksize': ksize}


def _fog(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Additive white veil: haze, thin cloud, and the contrast loss of a long air column."""
    density = _uniform(rng, [0.0, float(params.get('max_density', 0.10))])
    veil = np.full_like(image, 255)
    return cv2.addWeighted(image, 1.0 - density, veil, density, 0.0), {'fog_density': density}


def _noise(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Additive Gaussian sensor noise."""
    sigma = _uniform(rng, [0.0, float(params.get('max_sigma', 5.0))])
    if sigma <= 0:
        return image, {'noise_sigma': 0.0}
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy, {'noise_sigma': sigma}


def _shot_noise(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Poisson (photon) noise, whose variance follows intensity: the low-light regime."""
    scale = float(params.get('scale', 60.0))
    if scale <= 0:
        return image, {'shot_noise_scale': 0.0}
    normalized = image.astype(np.float64) / 255.0
    noisy = rng.poisson(normalized * scale) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8), {'shot_noise_scale': scale}


def _jpeg(image: np.ndarray, rng: np.random.Generator, **params: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """JPEG compression by an encode/decode round trip."""
    quality = int(round(_uniform(rng, params.get('quality', [50, 95]))))
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image, {'jpeg_quality': float('nan')}
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return decoded, {'jpeg_quality': quality}


PHOTOMETRIC_OPS: Dict[str, Callable[..., Tuple[np.ndarray, Dict[str, float]]]] = {
    'brightness': _brightness,
    'contrast': _contrast,
    'saturation': _saturation,
    'blur': _blur,
    'fog': _fog,
    'noise': _noise,
    'shot_noise': _shot_noise,
    'jpeg': _jpeg,
}


def apply_photometric(
    image: np.ndarray,
    spec: Optional[Sequence[Mapping[str, Any]]],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply a configured chain of photometric perturbations, in the order listed.

    Returns the perturbed image and a flat dict of every sampled parameter.
    """
    rng = rng if rng is not None else np.random.default_rng()
    sampled: Dict[str, float] = {}
    if not spec:
        return image, sampled

    result = image
    for entry in spec:
        params = dict(entry)
        name = str(params.pop('name'))
        if name not in PHOTOMETRIC_OPS:
            raise PerturbationError(f"Unknown photometric perturbation '{name}'. Available: {sorted(PHOTOMETRIC_OPS)}")
        result, drawn = PHOTOMETRIC_OPS[name](result, rng, **params)
        sampled.update(drawn)

    return result, sampled


def describe_spec(config: Mapping[str, Any]) -> List[str]:
    """Human-readable summary of a perturbation config, for logs and figure captions."""
    lines = []
    geometric = {**DEFAULT_GEOMETRIC, **(config.get('geometric') or {})}
    lines.append(
        "geometric: translation +/-{translation_ratio:.0%}, rotation +/-{rotation_deg}deg, "
        "scale +/-{scale_ratio:.0%}, projective +/-{projective:g}".format(**geometric)
        if all(not isinstance(v, (list, tuple)) for v in geometric.values())
        else f"geometric: {geometric}"
    )
    for entry in config.get('photometric') or []:
        params = {k: v for k, v in entry.items() if k != 'name'}
        lines.append(f"photometric: {entry['name']} {params}")
    return lines
