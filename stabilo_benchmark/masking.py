#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
masking.py - Object label IO and the none / HBB / OBB masking axis.

This module selects the label geometry handed to stabilo for feature suppression; stabilo
rasterizes it with the pipeline's `mask_margin_ratio`. The evaluation region, which uses the
protocol's `mask_dilation_px`, is in `regions.py`.

Image directories follow hbb2obb's layout: `images/`, `labels_hbb/` and `labels_obb/` side by
side, read together by `load_annotations`. A directory with neither label directory is
unannotated: every frame has no objects, and only masking `none` can run on it
(`require_annotations`).

Label formats (YOLO TXT, one object per line), in normalized or absolute coordinates:

    columns  format     layout
    -------  ---------  ----------------------------------------------------------
    5        hbb        <class> <xc> <yc> <w> <h>
    6        hbb        <class> <xc> <yc> <w> <h> <conf>          detector output
    6        xywha      <class> <xc> <yc> <w> <h> <angle_deg>     explicit only
    9        obb        <class> <x1> <y1> ... <x4> <y4>
    10       obb        <class> <x1> <y1> ... <x4> <y4> <conf>
    odd >= 7 polygon    <class> <x1> <y1> ... <xn> <yn>           YOLO segmentation

Six columns are read as HBB + confidence under `auto`; `xywha` must be named. A polygon becomes
its minimum-area rotated rectangle (OBB) and its axis-aligned extent (HBB).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from stabilo_benchmark.pipelines import MASKING_BOX_FORMAT, MASKING_MODES, OBJECT_BOX_GEOMETRIES

HBB_COLUMNS = 5
HBB_CONF_COLUMNS = 6
XYWHA_COLUMNS = 6
OBB_COLUMNS = 9
OBB_CONF_COLUMNS = 10
MIN_POLYGON_COLUMNS = 7  # class + three vertices
VALID_COORDS = ('auto', 'normalized', 'absolute')
VALID_LABEL_FORMATS = ('auto', 'hbb', 'obb', 'xywha', 'polygon')


class LabelError(ValueError):
    """Raised when a label file cannot be parsed, or lacks the geometry a mode needs."""


@dataclass(frozen=True)
class Labels:
    """
    Object annotations for one frame, in absolute pixel coordinates.

    Attributes:
        classes: (N,) integer class ids.
        hbb: (N, 4) axis-aligned boxes as [xc, yc, w, h].
        obb: (N, 8) oriented boxes as [x1, y1, ..., x4, y4], or None when the source file only
            contained HBBs; `masking='obb'` then raises.
        confidence: (N,) detector scores when the source carried them, else None.
        source: Path the labels were read from.
    """

    classes: np.ndarray
    hbb: np.ndarray
    obb: Optional[np.ndarray]
    confidence: Optional[np.ndarray] = None
    source: Optional[Path] = None

    def __len__(self) -> int:
        return int(self.hbb.shape[0])

    @property
    def has_obb(self) -> bool:
        return self.obb is not None

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None

    def boxes_for(self, masking: str) -> Optional[np.ndarray]:
        """
        The box array to pass to `set_ref_frame` / `stabilize` for a masking mode.

        Returns None for 'none', so stabilo builds no mask.
        """
        if masking not in MASKING_MODES:
            raise LabelError(f"Invalid masking mode '{masking}'. Choose from {list(MASKING_MODES)}")
        if masking == 'none':
            return None
        if masking == 'hbb':
            return self.hbb.copy()
        if self.obb is None:
            raise LabelError(
                f"masking='obb' requires oriented labels, but {self.source or 'the label source'} contains "
                f"only horizontal boxes. Convert them with hbb2obb (see the README's 'Annotating images')."
            )
        return self.obb.copy()

    def corners(self) -> np.ndarray:
        """(N, 4, 2) corner coordinates: OBB corners when available, else HBB corners."""
        if self.obb is not None:
            return self.obb.reshape(-1, 4, 2)
        return hbb_to_corners(self.hbb)

    def object_corners(self, geometry: str) -> np.ndarray:
        """
        (N, 4, 2) corners of the boxes IoU and OCE are computed on.

        Unlike `corners()`, raises instead of falling back to HBB when oriented boxes are missing.
        """
        if geometry not in OBJECT_BOX_GEOMETRIES:
            raise LabelError(f"Invalid object box geometry '{geometry}'. Choose from {list(OBJECT_BOX_GEOMETRIES)}")
        if geometry == 'hbb':
            return hbb_to_corners(self.hbb)
        if self.obb is None:
            raise LabelError(
                f"metrics.object_boxes='obb' requires oriented labels, but {self.source or 'the label source'} "
                f"contains only horizontal boxes. Convert them with hbb2obb, or set metrics.object_boxes: hbb."
            )
        return self.obb.reshape(-1, 4, 2)

    def subset(self, keep: Sequence[bool]) -> 'Labels':
        """Filter objects (e.g. to drop a class from the evaluation) keeping both geometries."""
        keep = np.asarray(keep, dtype=bool)
        return Labels(
            classes=self.classes[keep],
            hbb=self.hbb[keep],
            obb=None if self.obb is None else self.obb[keep],
            confidence=None if self.confidence is None else self.confidence[keep],
            source=self.source,
        )


def empty_labels(source: Optional[Path] = None, with_obb: bool = True) -> Labels:
    """Labels for a frame containing no objects."""
    return Labels(
        classes=np.zeros((0,), dtype=int),
        hbb=np.zeros((0, 4), dtype=np.float64),
        obb=np.zeros((0, 8), dtype=np.float64) if with_obb else None,
        confidence=None,
        source=source,
    )


def hbb_to_corners(hbb: np.ndarray) -> np.ndarray:
    """(N, 4) [xc, yc, w, h] -> (N, 4, 2) corners, ordered TL, TR, BR, BL."""
    hbb = np.asarray(hbb, dtype=np.float64).reshape(-1, 4)
    xc, yc, w, h = hbb[:, 0], hbb[:, 1], hbb[:, 2], hbb[:, 3]
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return np.stack(
        [np.stack([x1, y1], -1), np.stack([x2, y1], -1), np.stack([x2, y2], -1), np.stack([x1, y2], -1)], axis=1
    )


def corners_to_hbb(corners: np.ndarray) -> np.ndarray:
    """(N, 4, 2) corners -> (N, 4) [xc, yc, w, h] axis-aligned enclosing box."""
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    center = (lo + hi) / 2
    size = hi - lo
    return np.concatenate([center, size], axis=1)


def xywha_to_corners(boxes: np.ndarray) -> np.ndarray:
    """(N, 5) [xc, yc, w, h, angle_deg] -> (N, 4, 2) corners, angle measured counter-clockwise."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    xc, yc, w, h, angle = (boxes[:, i] for i in range(5))
    cos, sin = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    dx, dy = w / 2, h / 2
    # Local corner offsets, TL TR BR BL, rotated into place around the box center.
    offsets = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    local = offsets[None, :, :] * np.stack([dx, dy], axis=-1)[:, None, :]
    rotated_x = local[..., 0] * cos[:, None] - local[..., 1] * sin[:, None]
    rotated_y = local[..., 0] * sin[:, None] + local[..., 1] * cos[:, None]
    return np.stack([rotated_x + xc[:, None], rotated_y + yc[:, None]], axis=-1)


def polygon_to_corners(polygons: Sequence[np.ndarray]) -> np.ndarray:
    """(N,) polygons of (V, 2) vertices -> (N, 4, 2) minimum-area rotated rectangles."""
    if not len(polygons):
        return np.zeros((0, 4, 2), dtype=np.float64)
    boxes = [cv2.boxPoints(cv2.minAreaRect(np.asarray(p, dtype=np.float32))) for p in polygons]
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4, 2)


def load_labels(
    path: Path,
    width: int,
    height: int,
    label_format: str = 'auto',
    coords: str = 'auto',
    classes: Optional[Sequence[int]] = None,
    min_confidence: Optional[float] = None,
) -> Labels:
    """
    Read a YOLO label file into absolute-pixel `Labels`.

    Args:
        path: Label file. A missing file yields empty labels.
        width, height: Frame size, needed to denormalize relative coordinates.
        label_format: One of VALID_LABEL_FORMATS; 'auto' infers from the column count.
        coords: 'auto' (infer), 'normalized', or 'absolute'.
        classes: Optional whitelist of class ids to keep.
        min_confidence: Optional score floor; raises when the source carries no confidences.
    """
    path = Path(path)
    if label_format not in VALID_LABEL_FORMATS:
        raise LabelError(f"Invalid label_format '{label_format}'. Choose from {list(VALID_LABEL_FORMATS)}")
    if coords not in VALID_COORDS:
        raise LabelError(f"Invalid coords '{coords}'. Choose from {list(VALID_COORDS)}")

    rows = _read_rows(path)
    if rows is None:
        return empty_labels(path, with_obb=label_format not in ('hbb', 'xywha'))

    widths = {len(row) for row in rows}
    resolved = _resolve_format(path, label_format, widths)

    class_ids = np.array([int(row[0]) for row in rows], dtype=int)
    confidence: Optional[np.ndarray] = None

    horizontal: Optional[np.ndarray] = None
    if resolved == 'polygon':
        vertices = [np.asarray(row[1:], dtype=np.float64).reshape(-1, 2) for row in rows]
        vertices = [_denormalize(v, width, height, coords) for v in vertices]
        corners = polygon_to_corners(vertices)
        # The HBB is the polygon's own extent, not the extent of its rotated rectangle.
        horizontal = np.array(
            [np.concatenate([(v.min(0) + v.max(0)) / 2, v.max(0) - v.min(0)]) for v in vertices]
        ).reshape(-1, 4)
    else:
        geometry = np.asarray([row[1:] for row in rows], dtype=np.float64)
        if resolved == 'hbb' and geometry.shape[1] == HBB_CONF_COLUMNS - 1:
            geometry, confidence = geometry[:, :4], geometry[:, 4]
        elif resolved == 'obb' and geometry.shape[1] == OBB_CONF_COLUMNS - 1:
            geometry, confidence = geometry[:, :8], geometry[:, 8]

        if resolved == 'xywha':
            # The angle is not scaled by the frame size.
            spatial = _denormalize(geometry[:, :4].reshape(-1, 2, 2), width, height, coords).reshape(-1, 4)
            corners = xywha_to_corners(np.column_stack([spatial, geometry[:, 4]]))
        else:
            spatial = _denormalize(geometry.reshape(len(rows), -1, 2), width, height, coords)
            corners = hbb_to_corners(spatial.reshape(-1, 4)) if resolved == 'hbb' else spatial

    if horizontal is None:
        horizontal = corners_to_hbb(corners)
    obb = None if resolved == 'hbb' else corners.reshape(-1, 8)
    labels = Labels(classes=class_ids, hbb=horizontal, obb=obb, confidence=confidence, source=path)

    if classes is not None:
        labels = labels.subset(np.isin(labels.classes, list(classes)))
    if min_confidence is not None:
        if labels.confidence is None:
            raise LabelError(
                f"min_confidence={min_confidence} was requested, but {path} carries no confidence column "
                f"({resolved} labels with {sorted(widths)[0]} columns). Drop the threshold, or supply "
                f"detector output that scores its boxes."
            )
        labels = labels.subset(labels.confidence >= float(min_confidence))
    return labels


def _read_rows(path: Path) -> Optional[List[List[float]]]:
    """
    Parse a label file into per-line float lists, or None when it is missing or empty.

    Parsed line by line because polygon rows are ragged.
    """
    if not path.exists():
        return None
    content = path.read_text(encoding='utf-8').strip()
    if not content:
        return None

    rows: List[List[float]] = []
    for number, line in enumerate(content.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            rows.append([float(field) for field in fields])
        except ValueError as e:
            raise LabelError(f"{path}:{number}: could not parse '{line.strip()}' as numbers") from e
    return rows or None


def _resolve_format(path: Path, label_format: str, widths: set) -> str:
    """Settle the label layout from the requested format and the observed column counts."""
    ragged = len(widths) > 1
    columns = sorted(widths)[0]

    if label_format != 'auto':
        if label_format == 'polygon':
            if ragged or (columns >= MIN_POLYGON_COLUMNS and columns % 2 == 1):
                return 'polygon'
            raise LabelError(
                f"{path}: label_format 'polygon' expects an odd column count >= {MIN_POLYGON_COLUMNS} "
                f"(class + at least three vertices), found {sorted(widths)}"
            )
        if ragged:
            raise LabelError(
                f"{path}: label_format '{label_format}' expects a fixed column count, found {sorted(widths)}"
            )
        expected = {
            'hbb': (HBB_COLUMNS, HBB_CONF_COLUMNS),
            'obb': (OBB_COLUMNS, OBB_CONF_COLUMNS),
            'xywha': (XYWHA_COLUMNS,),
        }[label_format]
        if columns not in expected:
            raise LabelError(
                f"{path}: label_format '{label_format}' expects "
                f"{' or '.join(str(e) for e in expected)} columns, found {columns}"
            )
        return label_format

    if ragged:
        return 'polygon'
    if columns in (HBB_COLUMNS, HBB_CONF_COLUMNS):
        # Six columns are HBB + confidence, never xywha.
        return 'hbb'
    if columns in (OBB_COLUMNS, OBB_CONF_COLUMNS):
        return 'obb'
    if columns >= MIN_POLYGON_COLUMNS and columns % 2 == 1:
        return 'polygon'
    raise LabelError(
        f"Cannot infer label format of {path}: found {columns} columns, which matches none of "
        f"{HBB_COLUMNS}/{HBB_CONF_COLUMNS} (HBB), {OBB_COLUMNS}/{OBB_CONF_COLUMNS} (OBB), or an odd "
        f"count >= {MIN_POLYGON_COLUMNS} (polygon). Set scenes.label_format explicitly."
    )


def _denormalize(points: np.ndarray, width: int, height: int, coords: str) -> np.ndarray:
    """
    Scale (..., 2) xy points to pixels when they are normalized.

    'auto' treats the points as normalized when every coordinate is <= 1.
    """
    points = np.asarray(points, dtype=np.float64)
    if coords == 'absolute':
        return points
    if coords == 'auto' and float(np.nanmax(np.abs(points))) > 1.0:
        return points
    return points * np.array([width, height], dtype=np.float64)


IMAGES_DIR = 'images'
HBB_DIR = 'labels_hbb'
OBB_DIR = 'labels_obb'


def is_annotated(root: Path) -> bool:
    """
    Whether `<root>` holds object annotations: True with both `labels_hbb/` and `labels_obb/`,
    False with neither. One without the other raises, since the oriented labels are the conversion
    of the horizontal ones.
    """
    root = Path(root)
    hbb, obb = ((root / name).is_dir() for name in (HBB_DIR, OBB_DIR))
    if hbb != obb:
        present, missing = (HBB_DIR, OBB_DIR) if hbb else (OBB_DIR, HBB_DIR)
        raise LabelError(
            f"{root} holds {present}/ but no {missing}/. Annotate {root / IMAGES_DIR} with both hbb2obb "
            f"commands (see the README's 'Annotating images'), or remove {present}/ to run without masks."
        )
    return hbb


def require_annotations(root: Path, maskings: Sequence[str]) -> bool:
    """
    Whether `<root>` is annotated (`is_annotated`); raises when it is not and a masking mode other
    than `none` needs its boxes.
    """
    annotated = is_annotated(root)
    needing = sorted({mode for mode in maskings if mode != 'none'})
    if needing and not annotated:
        raise LabelError(
            f"{root} has no annotations ({HBB_DIR}/, {OBB_DIR}/), which masking {', '.join(needing)} needs. "
            f"Annotate {root / IMAGES_DIR} with hbb2obb, or run with masking none only (see the README's "
            f"'Annotating images')."
        )
    return annotated


def load_annotations(root: Path, stem: str, width: int, height: int, classes: Optional[Sequence[int]] = None) -> Labels:
    """
    A frame's objects from `<root>/labels_hbb/<stem>.txt` and `<root>/labels_obb/<stem>.txt`.

    The two files are paired row by row, as hbb2obb writes them; a count or class mismatch raises.
    A missing file inside the label directories is a frame with no objects, and so is every frame
    of an unannotated `<root>` (`is_annotated`).
    """
    root = Path(root)
    if not is_annotated(root):
        return empty_labels()
    horizontal = load_labels(root / HBB_DIR / f"{stem}.txt", width, height, label_format='hbb')
    oriented = load_labels(root / OBB_DIR / f"{stem}.txt", width, height, label_format='obb')
    if len(horizontal) != len(oriented) or not np.array_equal(horizontal.classes, oriented.classes):
        raise LabelError(
            f"{root / HBB_DIR / stem}.txt holds {len(horizontal)} box(es) and {root / OBB_DIR / stem}.txt "
            f"{len(oriented)}, or their classes differ row by row. The oriented labels must be hbb2obb's "
            f"conversion of exactly these horizontal labels; re-run it."
        )
    labels = Labels(classes=oriented.classes, hbb=horizontal.hbb, obb=oriented.obb, source=oriented.source)
    if classes is not None:
        labels = labels.subset(np.isin(labels.classes, list(classes)))
    return labels


def box_format_for(masking: str) -> Optional[str]:
    """The stabilo `box_format` string for a masking mode, or None when masking is off."""
    if masking not in MASKING_MODES:
        raise LabelError(f"Invalid masking mode '{masking}'. Choose from {list(MASKING_MODES)}")
    return MASKING_BOX_FORMAT[masking]


def summarize(labels: Labels, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
    """
    Per-frame label statistics recorded with the results.

    The key set is the same with and without objects. With `width` and `height`, `coverage` is the
    fraction of the frame covered by the union of the boxes.
    """
    unique, counts = np.unique(labels.classes, return_counts=True)
    summary: Dict[str, Any] = {
        'n_objects': len(labels),
        'total_box_area_px': 0.0,
        'mean_box_area_px': 0.0,
        'coverage': float('nan') if width is None or height is None else 0.0,
        'classes': {int(k): int(v) for k, v in zip(unique, counts, strict=True)},
    }
    if len(labels) == 0:
        return summary

    corners = labels.corners()
    areas = np.abs(_shoelace(corners))
    summary['total_box_area_px'] = float(areas.sum())
    summary['mean_box_area_px'] = float(areas.mean())
    if width is not None and height is not None:
        from stabilo_benchmark.regions import rasterize_quads  # noqa: PLC0415 - avoids a circular import

        occupied = rasterize_quads(corners, (int(height), int(width)))
        summary['coverage'] = float(np.count_nonzero(occupied) / (int(width) * int(height)))
    return summary


def _shoelace(corners: np.ndarray) -> np.ndarray:
    """Signed polygon area of each (4, 2) quadrilateral, via the shoelace formula."""
    x, y = corners[..., 0], corners[..., 1]
    return 0.5 * (np.sum(x * np.roll(y, -1, axis=-1), axis=-1) - np.sum(y * np.roll(x, -1, axis=-1), axis=-1))
