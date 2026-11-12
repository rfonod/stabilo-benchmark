#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
scenes.py - Characterize the image set a campaign ran on, from its annotations.

The annotations are located through the manifest. Coverage is the union of the object
footprints (as in `masking.summarize`), reported for both the horizontal and the oriented boxes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stabilo_benchmark.masking import IMAGES_DIR, load_annotations
from stabilo_benchmark.regions import rasterize_quads

SCENE_STATISTICS = ('minimum', 'q1', 'median', 'q3', 'maximum')

SCENE_COLUMNS = {
    'n_objects': ('Objects', '{:.0f}', None),
    'coverage_obb': (r'OBB area [\%]', '{:.2f}', None),
    'coverage_hbb': (r'HBB area [\%]', '{:.2f}', None),
}

SCENE_CAPTION = (
    'Object content of the {count} {noun} the campaign ran on, as quantiles over {noun}. '
    'Area is the fraction of the frame covered by the union of the boxes, for the oriented '
    'boxes and for the horizontal boxes they were converted from, before any mask margin; the difference '
    'between the two area columns is the additional area a horizontal mask suppresses.'
)


def _coverage(corners: np.ndarray, width: int, height: int) -> float:
    """Fraction of the frame covered by the union of the quadrilaterals."""
    if len(corners) == 0:
        return 0.0
    mask = rasterize_quads(corners, (height, width))
    return float(np.count_nonzero(mask)) / float(width * height)


def characterize_scenes(root: Path, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    One row per image in `<root>/images`: object count and covered area, both geometries.

    `names` selects a subset, matched on the filename or its stem; a name that matches nothing is
    skipped.
    """
    import cv2  # noqa: PLC0415 - kept out of module import so `report --help` stays cheap

    root = Path(root)
    images_dir = root / IMAGES_DIR
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} does not exist")

    paths = sorted(p for p in images_dir.iterdir() if p.is_file() and not p.name.startswith('.'))
    if names is not None:
        wanted = {str(name) for name in names}
        paths = [p for p in paths if p.name in wanted or p.stem in wanted]

    rows: List[Dict[str, Any]] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = load_annotations(root, path.stem, width, height)
        rows.append(
            {
                'scene': path.stem,
                'n_objects': len(labels),
                'coverage_obb': 100.0 * _coverage(labels.object_corners('obb'), width, height),
                'coverage_hbb': 100.0 * _coverage(labels.object_corners('hbb'), width, height),
            }
        )
    return pd.DataFrame(rows)


def summarize_scenes(frame: pd.DataFrame) -> pd.DataFrame:
    """Quartiles of `characterize_scenes`' per-scene rows, one row per statistic."""
    if frame.empty:
        raise ValueError('No scenes were characterized')

    quantiles = {'minimum': 0.0, 'q1': 0.25, 'median': 0.5, 'q3': 0.75, 'maximum': 1.0}
    rows = [
        {'statistic': name, **{column: float(frame[column].quantile(q)) for column in SCENE_COLUMNS}}
        for name, q in quantiles.items()
    ]
    return pd.DataFrame(rows)
