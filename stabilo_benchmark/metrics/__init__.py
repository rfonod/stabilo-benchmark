#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
metrics - What the benchmark measures.

  geometric    error against a known or reference homography: CTE, HEA, mAA, IoU, OCE.
  photometric  reference-free alignment over the static background: SSIM, ZNCC, MI, gradient.
  diagnostics  keypoint, match and inlier statistics, and the motion-alignment decomposition.

Ranking uses the geometric and photometric families only.
"""

from stabilo_benchmark.metrics.diagnostics import (
    estimation_failed,
    matching_stats,
    motion_decomposition,
)
from stabilo_benchmark.metrics.geometric import (
    OBJECT_STATISTICS,
    box_centers,
    corner_transfer,
    corner_transfer_error,
    hea,
    homography_frobenius_error,
    object_center_error,
    object_iou,
    quad_iou,
)
from stabilo_benchmark.metrics.photometric import (
    PHOTOMETRIC_METRICS,
    masked_gradient_similarity,
    masked_mutual_information,
    masked_ssim,
    masked_zncc,
    photometric_scores,
)

__all__ = [
    'OBJECT_STATISTICS',
    'PHOTOMETRIC_METRICS',
    'box_centers',
    'corner_transfer',
    'corner_transfer_error',
    'estimation_failed',
    'hea',
    'homography_frobenius_error',
    'masked_gradient_similarity',
    'masked_mutual_information',
    'masked_ssim',
    'masked_zncc',
    'matching_stats',
    'motion_decomposition',
    'object_center_error',
    'object_iou',
    'photometric_scores',
    'quad_iou',
]
