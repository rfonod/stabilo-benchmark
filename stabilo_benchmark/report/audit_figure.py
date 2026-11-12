#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
audit_figure.py - The ground-control-point audit of the reference estimator, as one figure.

(a) One case's reference frame with its ground control points, the excluded static structure, every
    object's OBB, the reference estimator's feature mask and the evaluation region's dilation around
    each object, an inset magnifying one object at the frame's resolution, and
    each point's residual at the case's longest audited offset as an exaggerated vector.
(b) Three of those points (smallest, median and largest residual), magnified: the reference frame
    beside the audited frame resampled into reference coordinates by the reference estimator, with
    the reference point and the carried current-frame point. Their distance is the residual.
(c) Every audited point of every audited case against its time offset, pooled per offset as a box,
    with the equivalence margin.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from stabilo_benchmark.gcp import GCP_DIR, load_gcp_set
from stabilo_benchmark.oracle import load_frame, load_frame_labels, load_oracle, oracle_matrices
from stabilo_benchmark.regions import EXCLUSION_FILE, load_exclusion, rasterize_quads, warp_points
from stabilo_benchmark.report.figures import (
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE,
    _save,
    apply_style,
)
from stabilo_benchmark.report.tables import audit_bin_ranges

AUDIT_FILE = Path('oracle') / 'AUDIT.json'
# Residual vectors in (a) are drawn this many times their length.
VECTOR_EXAGGERATION = 250
# Half-width of the magnified windows in (b), in source pixels.
CROP_HALF_PX = 10
# Longest side of the frame as rendered in (a), in pixels; the PDF embeds it as a raster.
FRAME_RENDER_PX = 2400
# Share of gray mixed into the frame in (a) so the overlays carry the color.
DESATURATION = 0.45
# Half-width of the boxes in (c), in natural-log units of the offset.
BOX_HALF_WIDTH = 0.065

POINT_COLOR = PALETTE[0]
CARRIED_COLOR = PALETTE[1]
OBJECT_COLOR = PALETTE[2]
EXCLUDED_COLOR = '#3d3c39'
NOT_SCORED_COLOR = PALETTE[3]
NOT_SCORED_FILL = (*[int(NOT_SCORED_COLOR[i : i + 2], 16) / 255 for i in (1, 3, 5)], 0.5)
RAW_OBB_STYLE = (0, (2.5, 1.5))
# The magnified window in (a): its widths to try as a share of the frame, and its height over width.
INSET_WIDTHS = (0.24, 0.22, 0.2, 0.18)
INSET_ASPECT = 0.68


class AuditFigureError(RuntimeError):
    """Raised when a case lacks what the figure needs: frames, points, reference estimates, an audit."""


def read_case_audit(case_dir: Path) -> Optional[Dict[str, Any]]:
    """The case's `oracle/AUDIT.json`, or None when it has not been audited."""
    path = Path(case_dir) / AUDIT_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _fps(case_dir: Path) -> Optional[float]:
    """The case's frame rate, or None for a case staged from frames without one."""
    case = json.loads((Path(case_dir) / 'CASE.json').read_text(encoding='utf-8'))
    fps = (case.get('video') or {}).get('fps')
    return float(fps) if fps else None


def _when(offset: int, fps: Optional[float]) -> str:
    return f"{offset / fps:.0f} s" if fps else f"{offset} frames"


def _tick(value: float) -> str:
    """A bin edge for its label: one significant digit below one, whole numbers above."""
    return f"{value:.1g}" if 0 < value < 1 else f"{value:.0f}"


def _desaturate(image_bgr: np.ndarray, amount: float) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    gray = rgb.mean(axis=2, keepdims=True)
    return np.clip((1 - amount) * rgb + amount * gray, 0, 255).astype(np.uint8)


def _resampled_window(image: np.ndarray, homography: np.ndarray, origin: Tuple[int, int], size: int) -> np.ndarray:
    """The window at `origin` of `image` resampled into reference coordinates by `homography` (current -> reference)."""
    shift = np.array([[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]])
    return cv2.warpPerspective(image, shift @ homography, (size, size), flags=cv2.INTER_LINEAR)


def _audited_points(case_dir: Path) -> Dict[str, Any]:
    """Reference points, the longest audited frame, its points carried into reference coordinates, and residuals."""
    case_dir = Path(case_dir)
    reference_index = int(json.loads((case_dir / 'CASE.json').read_text(encoding='utf-8'))['reference_index'])
    points = load_gcp_set(case_dir / GCP_DIR)
    if reference_index not in points:
        raise AuditFigureError(f"{case_dir}: the reference frame {reference_index} has no ground control points")
    matrices = oracle_matrices(load_oracle(case_dir))
    audited = [
        index
        for index in sorted(points)
        if index != reference_index and matrices.get(f"{reference_index:06d}->{index:06d}") is not None
    ]
    if not audited:
        raise AuditFigureError(f"{case_dir}: no audited frame has a reference estimate")
    frame_index = audited[-1]
    homography = matrices[f"{reference_index:06d}->{frame_index:06d}"]
    reference = points[reference_index].as_dict()
    current = points[frame_index].as_dict()
    labels = [label for label in points[reference_index].labels if label in current]
    clicked = np.array([reference[label] for label in labels], dtype=np.float64)
    carried = warp_points(np.array([current[label] for label in labels], dtype=np.float64), homography)
    return {
        'reference_index': reference_index,
        'reference_points': points[reference_index],
        'frame_index': frame_index,
        'homography': homography,
        'labels': labels,
        'clicked': clicked,
        'carried': carried,
        'residuals': np.linalg.norm(carried - clicked, axis=1),
    }


def _panel_frame(
    ax: plt.Axes, case_dir: Path, data: Dict[str, Any], when: str, dilation_px: Optional[int] = None
) -> None:
    image = load_frame(case_dir, data['reference_index'])
    height, width = image.shape[:2]
    scale = min(1.0, FRAME_RENDER_PX / max(height, width))
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ax.imshow(_desaturate(shown, DESATURATION), extent=(0, width, height, 0), interpolation='antialiased')

    handles: List[Any] = []
    exclusion_path = Path(case_dir) / EXCLUSION_FILE
    if exclusion_path.is_file():
        excluded = load_exclusion(exclusion_path, (height, width))
        if excluded.any():
            small = cv2.resize(
                excluded.astype(np.uint8), (shown.shape[1], shown.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            overlay = np.zeros((*small.shape, 4), dtype=np.float32)
            overlay[small > 0] = (*[int(EXCLUDED_COLOR[i : i + 2], 16) / 255 for i in (1, 3, 5)], 0.6)
            ax.imshow(overlay, extent=(0, width, height, 0), interpolation='nearest')
            grid_x = np.linspace(0, width, small.shape[1])
            grid_y = np.linspace(0, height, small.shape[0])
            plt.rcParams['hatch.color'] = 'white'
            plt.rcParams['hatch.linewidth'] = 0.4
            ax.contourf(grid_x, grid_y, small, levels=[0.5, 1.5], colors='none', hatches=['//////'])
            ax.contour(grid_x, grid_y, small, levels=[0.5], colors='white', linewidths=0.7)
            handles.append(
                Patch(
                    facecolor=EXCLUDED_COLOR, alpha=0.7, edgecolor='white', hatch='//////', label='Excluded structure'
                )
            )

    labels = load_frame_labels(case_dir, data['reference_index'], width, height)
    corners = labels.object_corners('obb')
    feature = _feature_mask(case_dir, labels)
    if len(corners):
        if dilation_px:
            grown = rasterize_quads(corners, (height, width), int(dilation_px))
            small = cv2.resize(
                grown.astype(np.uint8), (shown.shape[1], shown.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            overlay = np.zeros((*small.shape, 4), dtype=np.float32)
            overlay[small > 0] = NOT_SCORED_FILL
            ax.imshow(overlay, extent=(0, width, height, 0), interpolation='nearest')
            handles.append(
                Patch(facecolor=NOT_SCORED_FILL, edgecolor='none', label=f"OBB + {dilation_px} px (not scored)")
            )
        ax.add_collection(
            PolyCollection(
                corners, facecolors='none', edgecolors=INK_PRIMARY, linewidths=0.35, linestyles=RAW_OBB_STYLE
            )
        )
        handles.append(Line2D([], [], color=INK_PRIMARY, linewidth=0.9, linestyle=RAW_OBB_STYLE, label='OBB (hbb2obb)'))
    if feature is not None:
        quads, factor, geometry = feature
        ax.add_collection(PolyCollection(quads, facecolors='none', edgecolors=OBJECT_COLOR, linewidths=0.6))
        handles.append(
            Line2D([], [], color=OBJECT_COLOR, linewidth=1.0, label=f"Feature mask ({geometry.upper()} ×{factor:g})")
        )

    clicked, carried = data['clicked'], data['carried']
    reference_all = data['reference_points'].points
    ax.scatter(
        reference_all[:, 0],
        reference_all[:, 1],
        s=16,
        marker='o',
        facecolors=POINT_COLOR,
        edgecolors='white',
        linewidths=0.7,
        zorder=6,
    )
    handles.append(
        Line2D([], [], marker='o', linestyle='', markerfacecolor=POINT_COLOR, markeredgecolor='white', label='GCP')
    )
    delta = (carried - clicked) * VECTOR_EXAGGERATION
    ax.quiver(
        clicked[:, 0],
        clicked[:, 1],
        delta[:, 0],
        delta[:, 1],
        angles='xy',
        scale_units='xy',
        scale=1,
        color=CARRIED_COLOR,
        width=0.004,
        headwidth=3.2,
        headlength=3.6,
        headaxislength=3.2,
        zorder=5,
    )
    handles.append(
        Line2D([], [], color=CARRIED_COLOR, linewidth=1.4, label=f"Residual at {when} (×{VECTOR_EXAGGERATION})")
    )
    # A 1 px key, drawn in data coordinates so it shares the vectors' exaggeration, on a backing box
    # in the corner that hides the fewest points and vector tips.
    box_w, box_h = VECTOR_EXAGGERATION * 2.25, height * 0.085
    pad = width * 0.012
    ends = np.vstack([clicked, clicked + delta])

    def hidden(corner: Tuple[float, float]) -> int:
        x, y = corner
        inside = (ends[:, 0] >= x) & (ends[:, 0] <= x + box_w) & (ends[:, 1] >= y) & (ends[:, 1] <= y + box_h)
        return int(np.sum(inside))

    taken: List[Tuple[float, float, float, float]] = []
    if len(corners):
        # Every GCP and every point along every residual vector stays uncovered when possible.
        along = (clicked[None] + np.linspace(0, 1, 16)[:, None, None] * delta[None]).reshape(-1, 2)
        inset_box = _inset(ax, image, corners, feature, dilation_px, np.vstack([reference_all, along]), pad)
        if inset_box is not None:
            taken.append(inset_box)
    key_corners = [
        corner
        for corner in (
            (pad, height - pad - box_h),
            (width - pad - box_w, height - pad - box_h),
            (pad, pad),
            (width - pad - box_w, pad),
        )
        if not any(_overlaps((*corner, box_w, box_h), other) for other in taken)
    ]
    box_x, box_y = min(key_corners, key=hidden)
    ax.add_patch(
        FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle=f"round,pad=0,rounding_size={box_h * 0.25}",
            facecolor='white',
            edgecolor='none',
            alpha=0.88,
            zorder=6,
        )
    )
    key_x, key_y = box_x + box_w * 0.08, box_y + box_h / 2
    ax.annotate(
        '',
        xy=(key_x + VECTOR_EXAGGERATION, key_y),
        xytext=(key_x, key_y),
        arrowprops={'arrowstyle': '-|>', 'color': CARRIED_COLOR, 'lw': 1.4, 'shrinkA': 0, 'shrinkB': 0},
        zorder=7,
    )
    ax.text(key_x + VECTOR_EXAGGERATION * 1.12, key_y, '1 px', color=INK_PRIMARY, fontsize=7, va='center', zorder=7)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=(len(handles) + 1) // 2,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=0.9,
        fontsize=7,
    )


def _feature_mask(case_dir: Path, labels: Any) -> Optional[Tuple[np.ndarray, float, str]]:
    """
    The reference estimator's feature-suppression quadrilaterals, as stabilo builds them from the
    oracle cache's recorded masking mode and `mask_margin_ratio`: each box scaled about its center.
    None when the cache records no box masking.
    """
    pipeline = load_oracle(case_dir).get('pipeline') or {}
    geometry, ratio = pipeline.get('masking'), pipeline.get('mask_margin_ratio')
    if geometry not in ('obb', 'hbb') or ratio is None:
        return None
    quads = labels.object_corners(geometry)
    if not len(quads):
        return None
    factor = 1.0 + float(ratio)
    centers = quads.mean(axis=1, keepdims=True)
    return centers + (quads - centers) * factor, factor, geometry


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """Whether two (x, y, w, h) boxes intersect."""
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _inset(
    ax: plt.Axes,
    image: np.ndarray,
    corners: np.ndarray,
    feature: Optional[Tuple[np.ndarray, float, str]],
    dilation_px: Optional[int],
    avoid: np.ndarray,
    pad: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    A magnified window around the object of median length, at the frame's own resolution: its OBB,
    the reference estimator's feature mask and the area the evaluation region drops around it.

    Placed in the corner of `ax` that covers the fewest of the points in `avoid` and not the window
    itself; returns the inset's (x, y, w, h) in data coordinates, or None when no window fits.
    """
    height, width = image.shape[:2]
    lengths = np.max(np.linalg.norm(np.diff(corners, axis=1, append=corners[:, :1]), axis=2), axis=1)
    win_w = 2.6 * float(np.median(lengths))
    win_h = win_w * INSET_ASPECT
    centers = corners.mean(axis=1)
    inside = (
        (centers[:, 0] - win_w / 2 >= 0)
        & (centers[:, 0] + win_w / 2 <= width)
        & (centers[:, 1] - win_h / 2 >= 0)
        & (centers[:, 1] + win_h / 2 <= height)
    )
    candidates = np.flatnonzero(inside)
    if not candidates.size:
        return None
    target = candidates[np.argmin(np.abs(lengths[candidates] - np.median(lengths)))]
    window = (centers[target, 0] - win_w / 2, centers[target, 1] - win_h / 2, win_w, win_h)

    def cost(box: Tuple[float, float, float, float]) -> float:
        x, y, w, h = box
        covered = (avoid[:, 0] >= x) & (avoid[:, 0] <= x + w) & (avoid[:, 1] >= y) & (avoid[:, 1] <= y + h)
        return float(np.sum(covered)) + (1e6 if _overlaps(box, window) else 0.0)

    # The largest inset, corner by corner, that covers nothing; else the smallest that covers least.
    boxes = [
        (x, y, w, w * INSET_ASPECT)
        for w in (share * width for share in INSET_WIDTHS)
        for x, y in (
            (pad, pad),
            (width - pad - w, pad),
            (pad, height - pad - w * INSET_ASPECT),
            (width - pad - w, height - pad - w * INSET_ASPECT),
        )
    ]
    clear = [box for box in boxes if cost(box) == 0]
    box_x, box_y, box_w, box_h = clear[0] if clear else min(boxes[-4:], key=cost)
    inset = ax.inset_axes(
        [box_x / width, 1 - (box_y + box_h) / height, box_w / width, box_h / height], transform=ax.transAxes
    )
    inset.set_zorder(8)
    x0, y0 = int(np.floor(window[0])), int(np.floor(window[1]))
    x1, y1 = int(np.ceil(window[0] + win_w)), int(np.ceil(window[1] + win_h))
    crop = image[y0:y1, x0:x1]
    inset.imshow(
        _desaturate(crop, DESATURATION), extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation='nearest'
    )
    local = corners - np.array([x0, y0])
    if dilation_px:
        grown = rasterize_quads(local, (y1 - y0, x1 - x0), int(dilation_px))
        overlay = np.zeros((*grown.shape, 4), dtype=np.float32)
        overlay[grown] = NOT_SCORED_FILL
        inset.imshow(overlay, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), interpolation='nearest')
    if feature is not None:
        inset.add_collection(PolyCollection(feature[0], facecolors='none', edgecolors=OBJECT_COLOR, linewidths=0.8))
    inset.add_collection(
        PolyCollection(corners, facecolors='none', edgecolors=INK_PRIMARY, linewidths=0.9, linestyles=RAW_OBB_STYLE)
    )
    inset.set_xlim(window[0], window[0] + win_w)
    inset.set_ylim(window[1] + win_h, window[1])
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.2)
    ax.indicate_inset_zoom(inset, edgecolor='white', alpha=0.9, linewidth=0.8)
    return box_x, box_y, box_w, box_h


def _crop_axes(ax: plt.Axes, window: np.ndarray, origin: Tuple[int, int]) -> None:
    size = window.shape[0]
    x0, y0 = origin
    ax.imshow(
        cv2.cvtColor(window, cv2.COLOR_BGR2RGB),
        extent=(x0 - 0.5, x0 + size - 0.5, y0 + size - 0.5, y0 - 0.5),
        interpolation='nearest',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK_MUTED)
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def _panel_crops(axes: np.ndarray, case_dir: Path, data: Dict[str, Any], when: str) -> None:
    reference_image = load_frame(case_dir, data['reference_index'])
    current_image = load_frame(case_dir, data['frame_index'])
    order = np.argsort(data['residuals'])
    picks = [('smallest', order[0]), ('median', order[len(order) // 2]), ('largest', order[-1])]
    size = 2 * CROP_HALF_PX + 1
    for row, (name, index) in enumerate(picks):
        clicked, carried = data['clicked'][index], data['carried'][index]
        origin = (int(round(clicked[0])) - CROP_HALF_PX, int(round(clicked[1])) - CROP_HALF_PX)
        x0, y0 = origin
        left, right = axes[row]
        _crop_axes(left, reference_image[y0 : y0 + size, x0 : x0 + size], origin)
        _crop_axes(right, _resampled_window(current_image, data['homography'], origin, size), origin)
        for ax in (left, right):
            ax.scatter(*clicked, s=70, marker='+', color=POINT_COLOR, linewidths=1.6, zorder=4)
        right.scatter(*carried, s=40, marker='x', color=CARRIED_COLOR, linewidths=1.6, zorder=5)
        right.plot(*np.stack([clicked, carried]).T, color=CARRIED_COLOR, linewidth=0.8, zorder=3)
        for ax, text in ((left, name), (right, f"{data['residuals'][index]:.2f} px")):
            ax.text(
                0.05,
                0.05,
                text,
                transform=ax.transAxes,
                fontsize=6.5,
                color='white',
                va='bottom',
                bbox={'boxstyle': 'round,pad=0.2', 'facecolor': INK_PRIMARY, 'alpha': 0.6, 'linewidth': 0},
            )
        if row == 0:
            left.set_title('Reference', fontsize=7, color=INK_SECONDARY, pad=3)
            right.set_title(f"{when}, warped", fontsize=7, color=INK_SECONDARY, pad=3)
    axes[-1][0].legend(
        handles=[
            Line2D([], [], marker='+', linestyle='', color=POINT_COLOR, markersize=7, label='Reference point'),
            Line2D([], [], marker='x', linestyle='', color=CARRIED_COLOR, markersize=6, label='Carried point'),
        ],
        loc='upper left',
        bbox_to_anchor=(0.0, -0.06),
        ncol=2,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def _panel_offsets(ax: plt.Axes, case_dirs: Sequence[Path]) -> None:
    rng = np.random.default_rng(0)
    audited = [(audit, _fps(case_dir)) for case_dir in case_dirs if (audit := read_case_audit(case_dir))]
    if not audited:
        raise AuditFigureError('No audited case to plot')
    # Seconds when every case knows its frame rate, else frame offsets.
    in_seconds = all(fps for _, fps in audited)
    margins: List[float] = []
    bins: List[Tuple[int, Optional[int]]] = []
    # Points are pooled per frame offset over the cases and placed at the median frame rate.
    scale = float(np.median([fps for _, fps in audited])) if in_seconds else 1.0
    pooled: Dict[int, List[float]] = {}
    for audit, _ in audited:
        estimator = audit['reference_estimator']
        margins.append(float(estimator['summary']['residual_p95_px']))
        bins = sorted(set(bins) | set(audit_bin_ranges(estimator.get('summary_by_offset') or {})))
        for frame in estimator['per_frame']:
            pooled.setdefault(int(frame['offset']), []).extend(float(v) for v in frame['residuals_px'])
    offsets = [offset / scale for offset in sorted(pooled)]
    for when, offset in zip(offsets, sorted(pooled), strict=True):
        values = np.asarray(pooled[offset], dtype=float)
        spread = np.exp(rng.uniform(-BOX_HALF_WIDTH * 1.5, BOX_HALF_WIDTH * 1.5, size=len(values)))
        ax.scatter(when * spread, values, s=4, color=INK_MUTED, alpha=0.35, linewidths=0, zorder=2)
        p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        left, right = when * np.exp(-BOX_HALF_WIDTH), when * np.exp(BOX_HALF_WIDTH)
        ax.add_patch(
            Rectangle(
                (left, q1),
                right - left,
                q3 - q1,
                facecolor='white',
                edgecolor=INK_PRIMARY,
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
                gid='offset-box',
            )
        )
        ax.plot([left, right], [median, median], color=POINT_COLOR, linewidth=1.8, zorder=4, solid_capstyle='butt')
        for low, high in ((p5, q1), (q3, p95)):
            ax.plot([when, when], [low, high], color=INK_PRIMARY, linewidth=0.8, zorder=3)
        for cap in (p5, p95):
            ax.plot(
                [when * np.exp(-BOX_HALF_WIDTH / 2), when * np.exp(BOX_HALF_WIDTH / 2)],
                [cap, cap],
                color=INK_PRIMARY,
                linewidth=0.8,
                zorder=3,
            )
    unit = 's' if in_seconds else 'frames'
    x_min, x_max = min(offsets) / 1.6, max((bins[-1][1] or bins[-1][0]) / scale, *offsets) * 1.1
    for position, (low, high) in enumerate(bins):
        start, end = max((low - 1) / scale, x_min), (high or low) / scale
        first = _tick((low - 1) / scale)
        if position % 2 == 0:
            ax.axvspan(start, end, color=GRIDLINE, alpha=0.45, linewidth=0, zorder=0)
        ax.text(
            np.sqrt(start * end),
            0.98,
            f"{first}–{_tick(high / scale)} {unit}" if high else f">{first} {unit}",
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=6.5,
            color=INK_MUTED,
        )
    margin = max(margins)
    ax.axhline(margin, color=CARRIED_COLOR, linestyle='--', linewidth=1.1, zorder=1)
    ax.text(
        0.01,
        margin,
        f"equivalence margin {margin:.2f} px",
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=7,
        color=CARRIED_COLOR,
    )
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None)
    ax.set_xlabel(f"{'Time' if in_seconds else 'Offset'} since the reference frame [{unit}]")
    ax.set_ylabel('GCP residual [px]')
    n_cases = len(margins)
    ax.legend(
        handles=[
            Line2D([], [], marker='o', linestyle='', color=INK_MUTED, alpha=0.6, markersize=3, label='Point'),
            Line2D([], [], color=POINT_COLOR, linewidth=1.8, label='Median'),
            Patch(facecolor='white', edgecolor=INK_PRIMARY, linewidth=0.8, label='Quartiles'),
            Line2D([], [], color=INK_PRIMARY, linewidth=0.8, marker='_', markersize=6, label='5th–95th pct.'),
        ],
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        fontsize=7,
        title=f"{n_cases} case{'s' if n_cases != 1 else ''}",
        title_fontsize=7,
        alignment='left',
    )


def figure_gcp_audit(case_dir: Path, case_dirs: Sequence[Path], path: Path, dilation_px: Optional[int] = None) -> Path:
    """
    The audit figure: `case_dir` illustrates (a) and (b); (c) pools every audited case of `case_dirs`.

    `dilation_px`, the evaluation region's object dilation, shades the area excluded around each object.
    """
    apply_style()
    case_dir = Path(case_dir)
    data = _audited_points(case_dir)
    when = _when(data['frame_index'] - data['reference_index'], _fps(case_dir))

    figure = plt.figure(figsize=(7.0, 5.15))
    outer = figure.add_gridspec(2, 1, height_ratios=[2.55, 1.35], hspace=0.32)
    top = outer[0].subgridspec(1, 2, width_ratios=[4.45, 1.55], wspace=0.07)
    frame_ax = figure.add_subplot(top[0])
    crops = top[1].subgridspec(3, 2, wspace=0.06, hspace=0.12)
    crop_axes = np.array([[figure.add_subplot(crops[row, column]) for column in range(2)] for row in range(3)])
    offset_ax = figure.add_subplot(outer[1])

    _panel_frame(frame_ax, case_dir, data, when, dilation_px)
    _panel_crops(crop_axes, case_dir, data, when)
    _panel_offsets(offset_ax, case_dirs)

    for ax, letter in ((frame_ax, 'a'), (crop_axes[0][0], 'b'), (offset_ax, 'c')):
        ax.text(
            -0.02 if ax is not crop_axes[0][0] else -0.05,
            1.0 if ax is not crop_axes[0][0] else 1.16,
            f"({letter})",
            transform=ax.transAxes,
            ha='right',
            va='bottom' if ax is offset_ax else 'top',
            fontsize=9,
            fontweight='bold',
            color=INK_PRIMARY,
        )
    return _save(figure, path)
