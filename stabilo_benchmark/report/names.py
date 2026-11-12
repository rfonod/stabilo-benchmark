#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
names.py - Display names for the identifiers that reach a table cell or a figure label.

Run labels are `<preset>+<masking>`, optionally followed by a repeat suffix (`#3`) or the swept
parameters (` (margin 0.15)`). Results, CSV files and code keep the identifiers; only rendered
output uses these names.
"""

import re
from typing import Any

PIPELINE_NAMES = {
    'orb': 'ORB',
    'sift': 'SIFT',
    'rsift': 'RootSIFT',
    'brisk': 'BRISK',
    'kaze': 'KAZE',
    'akaze': 'AKAZE',
    'xfeat': 'XFeat',
    'disk_lightglue': 'DISK+LightGlue',
    'dedode_lightglue': 'DeDoDe+LightGlue',
    'keynet_lightglue': 'KeyNet+LightGlue',
    'loftr': 'LoFTR',
    'oracle': 'Reference estimator',
    'identity': 'Identity',
    'reference': 'Reference',
}

MASKING_NAMES = {'none': 'none', 'hbb': 'HBB', 'obb': 'OBB'}

_LABEL = re.compile(r'^(?P<preset>[^+#\s(]+)(?:\+(?P<masking>none|hbb|obb))?(?P<rest>.*)$')


def pipeline_name(preset: Any) -> str:
    """The display name of a preset identifier; an unknown identifier is returned unchanged."""
    return PIPELINE_NAMES.get(str(preset), str(preset))


def masking_name(masking: Any) -> str:
    """`hbb` -> `HBB`; an unknown mode is returned unchanged."""
    return MASKING_NAMES.get(str(masking), str(masking))


def display_label(label: Any) -> str:
    """
    A run label for display: `rsift+obb` -> `RootSIFT / OBB`, `disk_lightglue` -> `DISK+LightGlue`,
    `sift+obb (margin 0.15)` -> `SIFT / OBB (margin 0.15)`. A repeat suffix is kept; text that is not
    a run label is returned unchanged.
    """
    text = str(label)
    match = _LABEL.match(text)
    if not match or match.group('preset') not in PIPELINE_NAMES:
        return text
    name = pipeline_name(match.group('preset'))
    if match.group('masking'):
        name += f" / {masking_name(match.group('masking'))}"
    return name + match.group('rest')
