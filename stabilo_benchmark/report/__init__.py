#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
report - Manuscript tables and figures, generated from saved results.

No number in the manuscript is typed by hand. Every table and figure is produced from a
result directory, so a re-run of the campaign updates the paper, and any figure can be
traced back to the manifest that produced it.
"""

from stabilo_benchmark.report.figures import (
    figure_error_vs_offset,
    figure_inlier_vs_quality,
    figure_masking_comparison,
    figure_motion_alignment,
    figure_sweep,
)
from stabilo_benchmark.report.tables import (
    aggregate_dynamic,
    aggregate_static,
    write_latex_table,
)

__all__ = [
    'aggregate_dynamic',
    'aggregate_static',
    'figure_error_vs_offset',
    'figure_inlier_vs_quality',
    'figure_masking_comparison',
    'figure_motion_alignment',
    'figure_sweep',
    'write_latex_table',
]
