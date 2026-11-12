#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
survey.py - Choose Part II's cases from candidate recordings, before any benchmark runs.

    stabilo-benchmark survey <videos>/*.MP4 -c configs/songdo/dynamic/offline_masking.yaml

A candidate case is a segment: a reference frame and the horizon its sampling plan reaches. The
survey reads it the way the benchmark will, as pairs (reference, reference + k), with one k per
bin of the plan's `survey_bins` (the short offsets, while the objects in view at the reference are
still shared with it), and asks where the reference frame's objects are k frames later:

    segment            cover%  static  moving  coher.  texture  cell
    -----------------  ------  ------  ------  ------  -------  -----------------------
    D1_AM2_B3@000000     2.41    1.12    0.81    0.52     4100  high moving / coherent
    ...

`moving` is the share of the frame covered by reference objects found displaced (the foreground
that can carry a homography with it), `coher.` the part of it that moves with one displacement,
`static` the part found where it was. Cells are a 2x2 on moving x coherence at the pool medians,
plus stationary controls, segments whose foreground stays put. A cell that cannot be filled is
reported, never filled from a neighboring cell. A recording's segments start at its first frame
(`--anchor start`) or every `--scan-stride` frames inside it (`--anchor scan`).

Frames are decoded sequentially, as `prepare` decodes them, and annotated with hbb2obb. Staged
frames that decode to identical bytes keep their labels, so a repeated survey only re-reads them.
`--pool` adds the candidates of earlier surveys of the same plan, so shards of one pool, read in
parallel, are assigned cells and selected as one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from stabilo_benchmark import scene
from stabilo_benchmark.annotation import annotate as run_hbb2obb
from stabilo_benchmark.masking import HBB_DIR, IMAGES_DIR, OBB_DIR, load_annotations
from stabilo_benchmark.results import describe_path, system_metadata
from stabilo_benchmark.scene import anchored_motion, describe, platform_motion, summarize_fates

VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
SURVEY_MANIFEST = 'SURVEY.json'
JPEG_QUALITY = 95  # as `prepare` writes case frames
CONFIDENCE_DIR = 'labels_confidence'
PROVENANCE_FILES = ('PROVENANCE_hbb.txt', 'PROVENANCE_obb.txt')
STAGING_RECORD = 'STAGED.json'

# Frames between scan points inside a recording, with `anchor='scan'`.
DEFAULT_SCAN_STRIDE = 900  # 30 s at 29.97 fps

# Minimum frames between two segments selected from one recording.
DEFAULT_MIN_SEPARATION = 3000  # ~100 s

# At most this many segments from one recording, so a cell is not filled by one site.
DEFAULT_MAX_PER_VIDEO = 2

# Where a recording's candidate segments start: its first frame only, or every `stride` frames.
ANCHOR_MODES = ('start', 'scan')

# The two design axes, read from each segment's summary.
MOVING, COHERENCE = 'displaced', 'coherence'

# A stationary control is a segment whose found foreground stays put: stationarity (static over
# static + displaced) in the pool's top decile and above its median, with at least the pool's
# median static foreground.
CONTROL_STATIONARITY_QUANTILE = 0.9

DESIGN_CELLS = (
    'high moving / coherent',
    'high moving / mixed',
    'low moving / coherent',
    'low moving / mixed',
)
CONTROL_CELL = 'stationary control'
# Too few displaced objects for coherence to mean anything (`scene.MIN_DISPLACED_OBJECTS`).
UNCLASSIFIED = 'unclassified'


class SurveyError(RuntimeError):
    """Raised when a candidate cannot be characterized: unreadable source, no labels, too short."""


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


def segment_frames(anchor: int, offsets: Sequence[int]) -> List[int]:
    """The frames a segment is read from: its reference, then one per offset."""
    return [int(anchor)] + [int(anchor) + int(k) for k in offsets]


def video_frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SurveyError(f"Could not open video: {video}")
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def stage_video_frames(video: Path, indices: Sequence[int], images_dir: Path) -> bool:
    """
    Decode `indices` sequentially into `images_dir` as `<index:06d>.jpg`.

    Returns True when the staged frames have the bytes labels beside them were made from. A staging
    record (`STAGING_RECORD` beside `images_dir`: the source's path, size and modification time,
    the indices, the JPEG quality, the OpenCV version and each frame's SHA-256) that matches lets a
    repeated survey skip decoding; frames deleted since (`--discard-frames`) are decoded again and
    checked against the recorded digests.
    """
    wanted = sorted({int(i) for i in indices})
    images_dir.mkdir(parents=True, exist_ok=True)
    record_path = images_dir.parent / STAGING_RECORD
    record = _staging_record(video, wanted)
    previous = _read_json(record_path)
    digests = previous.get('sha256') if isinstance(previous, dict) else None
    if digests is None or {key: previous.get(key) for key in record} != record:
        digests = None
    names = {f"{i:06d}.jpg" for i in wanted}
    if digests is not None and {p.name for p in images_dir.glob('*.jpg')} == names:
        return True
    record_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SurveyError(f"Could not open video: {video}")
    unchanged = True
    decoded = 0
    staged: List[str] = []
    try:
        index, target, last = 0, set(wanted), wanted[-1]
        while index <= last:
            if index not in target:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])[1].tobytes()
            out = images_dir / f"{index:06d}.jpg"
            digest = hashlib.sha256(encoded).hexdigest()
            if not out.is_file() or out.read_bytes() != encoded:
                out.write_bytes(encoded)
                unchanged = unchanged and digests is not None and digests[len(staged)] == digest
            staged.append(digest)
            decoded += 1
            index += 1
    finally:
        capture.release()
    if decoded < len(wanted):
        raise SurveyError(f"{video} ended before frame {last}; decoded {decoded} of {len(wanted)} frames")
    stale = {p.name for p in images_dir.glob('*.jpg')} - names
    for name in stale:
        (images_dir / name).unlink()
        unchanged = False
    record_path.write_text(json.dumps({**record, 'sha256': staged}, indent=1) + '\n', encoding='utf-8')
    return unchanged


def discard_frames(images_dir: Path) -> None:
    """Delete staged frames once described; their labels and the staging record stay."""
    for path in Path(images_dir).glob('*.jpg'):
        path.unlink()


def _staging_record(video: Path, indices: Sequence[int]) -> Dict[str, Any]:
    stat = Path(video).stat()
    return {
        'source': describe_path(video),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'frames': [int(i) for i in indices],
        'jpeg_quality': JPEG_QUALITY,
        'opencv': cv2.__version__,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def directory_frames(directory: Path) -> Dict[int, Path]:
    """The frames of an extracted set, by the frame index in their filename."""
    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise SurveyError(f"No images found in {directory} (looked for {', '.join(IMAGE_SUFFIXES)})")
    indexed = {_frame_index(p): p for p in images}
    indexed.pop(-1, None)
    return dict(sorted(indexed.items()))


def images_dir_of(source: Path) -> Path:
    """The frame directory of a directory source: `<source>/images` when it has one."""
    return source / IMAGES_DIR if (source / IMAGES_DIR).is_dir() else source


def is_annotated(images_dir: Path, complete: bool = False) -> bool:
    """
    Whether hbb2obb has written labels beside this frame directory; with `complete`, also that both
    commands finished (each writes its provenance file last).
    """
    root = Path(images_dir).parent
    if not all((root / name).is_dir() for name in (HBB_DIR, OBB_DIR)):
        return False
    return not complete or all((root / name).is_file() for name in PROVENANCE_FILES)


def ensure_labels(
    images_dir: Path, annotate: bool = True, device: Optional[str] = None, staged: bool = False, stale: bool = False
) -> None:
    """
    Run hbb2obb over `images_dir` unless labels beside it still describe its frames.

    `staged` marks a directory the survey writes, whose labels count only once both commands have
    finished; `stale`, frames just rewritten, whose old labels are removed before annotating.
    """
    if is_annotated(images_dir, complete=staged) and not stale:
        return
    if not annotate:
        raise SurveyError(f"No {HBB_DIR}/ and {OBB_DIR}/ beside {images_dir}, and --no-annotate was given.")
    if staged:
        root = Path(images_dir).parent
        for name in (HBB_DIR, OBB_DIR, CONFIDENCE_DIR):
            if (root / name).is_dir():
                shutil.rmtree(root / name)
        for name in PROVENANCE_FILES:
            (root / name).unlink(missing_ok=True)
    run_hbb2obb(images_dir, device=device, overwrite=True)


def _frame_index(path: Path) -> int:
    """The frame index encoded in a filename (its last run of digits), or -1 when there is none."""
    runs = re.findall(r'\d+', Path(path).stem)
    return int(runs[-1]) if runs else -1


# --------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------

# Layout directory names; a candidate passed as `<clip>/images` is named for the clip.
LAYOUT_DIR_NAMES = ('images', 'frames', 'img', 'imgs')


def candidate_name(source: Path) -> str:
    """The name a recording appears under in the survey."""
    source = Path(source)
    if source.is_file():
        return source.stem
    if source.name.lower() in LAYOUT_DIR_NAMES and source.parent.name:
        return source.parent.name
    return source.name


def unique_names(sources: Sequence[Path]) -> Dict[Path, str]:
    """
    A survey name per source: `candidate_name`, or, when two sources share it (hovers cut into
    files named per session), the path below the sources' common directory joined with '_'.
    """
    names = {source: candidate_name(source) for source in sources}
    if len(set(names.values())) == len(names):
        return names
    resolved = {source: Path(source).resolve() for source in sources}
    root = Path(os.path.commonpath([str(path.parent) for path in resolved.values()]))
    return {
        source: '_'.join(path.relative_to(root).with_suffix('').parts)
        if path.is_file()
        else '_'.join(path.relative_to(root).parts) or candidate_name(source)
        for source, path in resolved.items()
    }


def group_of(name: str, pattern: Optional[str]) -> str:
    """The group a candidate belongs to: the first capture group of `pattern` in its name, else the name."""
    if not pattern:
        return name
    compiled = re.compile(pattern)
    if compiled.groups < 1:
        raise SurveyError(f"--group-pattern needs a capture group, got {pattern!r}")
    match = compiled.search(name)
    return match.group(1) if match else name


# --------------------------------------------------------------------------------------
# Reading a segment
# --------------------------------------------------------------------------------------


def describe_segment(frames: Dict[int, Path], anchor: int, offsets: Sequence[int]) -> Dict[str, Any]:
    """
    Descriptors of one segment: the reference frame's own (`scene.describe`), and the fate of its
    objects at each offset (`scene.anchored_motion`), averaged with equal weight per offset.
    """
    reference_path = frames[anchor]
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise SurveyError(f"Could not read frame: {reference_path}")
    height, width = reference.shape[:2]
    reference_labels = load_annotations(reference_path.parent.parent, reference_path.stem, width, height)
    stats = describe(reference, reference_labels, width, height)

    per_offset: List[Dict[str, Any]] = []
    for offset in offsets:
        path = frames.get(anchor + int(offset))
        if path is None:
            continue
        current = cv2.imread(str(path))
        if current is None or current.shape[:2] != reference.shape[:2]:
            continue
        current_labels = load_annotations(path.parent.parent, path.stem, width, height)
        motion = platform_motion(reference, current, reference_labels, current_labels)
        fates = anchored_motion(reference, current, reference_labels, current_labels, motion=motion)
        per_offset.append({'offset': int(offset), 'n_objects': len(current_labels), **fates})
    if not per_offset:
        raise SurveyError(f"No frame at any survey offset after {reference_path}")

    summary = summarize_fates(per_offset)
    found = summary['static'] + summary['displaced']
    summary['stationarity'] = summary['static'] / found if found > 0 else float('nan')
    return {
        **stats.to_dict(),
        **summary,
        'frame_size': [width, height],
        'per_offset': per_offset,
    }


def scan_source(
    source: Path,
    work_dir: Path,
    offsets: Sequence[int],
    horizon: int = 0,
    anchor: str = 'start',
    stride: int = DEFAULT_SCAN_STRIDE,
    annotate: bool = True,
    device: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    discard: bool = False,
) -> List[Dict[str, Any]]:
    """
    Rate one recording: one segment at its first frame (`anchor='start'`), or one every `stride`
    frames inside it (`'scan'`, clear of its first and last seconds). A segment is kept only if the
    recording holds `horizon` frames after it, the span a case prepared from it samples. All of a
    recording's frames are annotated in one hbb2obb run; with `discard`, frames decoded from a
    video are deleted once its segments are described.
    """
    if anchor not in ANCHOR_MODES:
        raise SurveyError(f"anchor must be one of {ANCHOR_MODES}, got {anchor!r}")
    if not offsets:
        raise SurveyError('No survey offset: the sampling plan has no bounded bin within the horizon')
    source = Path(source)
    name = name or candidate_name(source)
    reach = max(int(horizon), *(int(k) for k in offsets))

    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        total = video_frame_count(source)
        anchors = _anchors(total - 1, reach, anchor, stride, str(source))
        images_dir = work_dir / name / IMAGES_DIR
        indices = sorted({i for a in anchors for i in segment_frames(a, offsets)})
        unchanged = stage_video_frames(source, indices, images_dir)
        ensure_labels(images_dir, annotate, device, staged=True, stale=not unchanged)
        frames = directory_frames(images_dir)
    elif source.is_dir():
        images_dir = images_dir_of(source)
        available = directory_frames(images_dir)
        anchors = [a for a in _anchors(max(available), reach, anchor, stride, str(source)) if a in available]
        if not anchors:
            raise SurveyError(f"{source} has no frame with {reach} frames after it")
        ensure_labels(images_dir, annotate, device)
        frames = available
    else:
        raise SurveyError(f"Not a video or a directory of frames: {source}")

    segments = []
    for start in anchors:
        segments.append(
            {
                'name': f"{name}@{start:06d}",
                'video': name,
                'group': group if group is not None else name,
                'source': describe_path(source),
                'frame': int(start),
                'labels_dir': describe_path(images_dir.parent),
                **describe_segment(frames, start, offsets),
            }
        )
    if discard and source.is_file():
        discard_frames(images_dir)
    return segments


def _anchors(final: int, reach: int, anchor: str, stride: int, label: str) -> List[int]:
    """Segment starts whose `reach` frames fit before frame `final`."""
    if anchor == 'start':
        if final < reach:
            raise SurveyError(f"{label} ends at frame {final}; a case needs {reach} frames after its first")
        return [0]
    # Skip the first and last few seconds (take-off and landing).
    margin = min(int(0.05 * (final + 1)), 300)
    end = final - margin - reach
    if end < margin:
        raise SurveyError(f"{label} ends at frame {final}; a case needs {reach} frames, clear of {margin} at each end")
    return list(range(margin, end + 1, max(1, int(stride))))


# --------------------------------------------------------------------------------------
# Meeting the design
# --------------------------------------------------------------------------------------


def assign_cells(
    candidates: Sequence[Dict[str, Any]],
    moving_split: Optional[float] = None,
    coherence_split: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Flag the stationary controls, then place every other candidate in a moving x coherence cell.

    Splits default to the medians of the candidates that are neither controls nor unclassified, and
    are returned; pass them to pin them across surveys. A candidate whose coherence is undefined (too
    few displaced objects to tell one structure from several) is `unclassified` and in no cell.
    """
    if not candidates:
        return {'cells': {}, 'splits': {}, 'controls': []}

    stationarity = np.array([c.get('stationarity', np.nan) for c in candidates], dtype=float)
    static = np.array([c.get('static', np.nan) for c in candidates], dtype=float)
    stationarity_floor = float(np.nanquantile(stationarity, CONTROL_STATIONARITY_QUANTILE))
    stationarity_median = float(np.nanmedian(stationarity))
    static_floor = float(np.nanmedian(static))
    for candidate, share, still in zip(candidates, stationarity, static, strict=True):
        # Strictly above the median as well, so a pool that does not differ in stationarity has none.
        stationary = share >= stationarity_floor and share > stationarity_median
        candidate['control'] = bool(stationary and still >= static_floor)

    rest = [c for c in candidates if not c['control'] and np.isfinite(c[COHERENCE])]
    explicit = {'moving': moving_split is not None, 'coherence': coherence_split is not None}
    moving_values = np.array([c[MOVING] for c in rest], dtype=float)
    coherence_values = np.array([c[COHERENCE] for c in rest], dtype=float)
    if moving_split is None:
        moving_split = float(np.nanmedian(moving_values)) if rest else 0.0
    if coherence_split is None:
        coherence_split = float(np.nanmedian(coherence_values)) if rest else 0.0

    cells: Dict[str, List[str]] = {}
    for candidate in candidates:
        if candidate['control']:
            candidate['cell'] = CONTROL_CELL
        elif not np.isfinite(candidate[COHERENCE]):
            candidate['cell'] = UNCLASSIFIED
        else:
            moving = 'high moving' if candidate[MOVING] >= moving_split else 'low moving'
            coherent = candidate[COHERENCE] >= coherence_split
            candidate['cell'] = f"{moving} / {'coherent' if coherent else 'mixed'}"
        cells.setdefault(candidate['cell'], []).append(candidate['name'])

    degenerate = [
        axis
        for axis, values, split in (
            ('moving', moving_values, moving_split),
            ('coherence', coherence_values, coherence_split),
        )
        if len(values) and _spread(values, split) <= 1e-9
    ]
    return {
        'cells': cells,
        'controls': cells.get(CONTROL_CELL, []),
        'splits': {
            'moving': moving_split,
            'coherence': coherence_split,
            'control_stationarity': stationarity_floor,
            'control_static': static_floor,
            'basis': _split_basis(explicit),
        },
        'degenerate_axes': degenerate,
    }


def select_cases(
    points: Sequence[Dict[str, Any]],
    strata: Dict[str, Any],
    per_cell: int = 2,
    controls: int = 1,
    max_per_video: int = DEFAULT_MAX_PER_VIDEO,
    min_separation: int = DEFAULT_MIN_SEPARATION,
    max_per_group: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pick segments until the design is filled, and report the cells that could not be.

    Within a cell, candidates rank by `margin`, their distance from the split boundaries in units
    of the pool's spread, the weaker axis deciding; controls rank by how far their stationarity
    clears the control floor. At most `max_per_video` segments come from one recording, never two
    within `min_separation` frames, and at most `max_per_group` per cell share a `group` (a site,
    say). An unfilled cell is not padded from another.
    """
    splits = strata.get('splits', {})
    moving_split = float(splits.get('moving', 0.0))
    coherence_split = float(splits.get('coherence', 0.0))
    rest = [p for p in points if p.get('cell') in DESIGN_CELLS]
    moving_scale = _spread([p[MOVING] for p in rest], moving_split)
    coherence_scale = _spread([p[COHERENCE] for p in rest], coherence_split)
    stationarity_floor = float(splits.get('control_stationarity', 0.0))
    stationarity_scale = _spread([p.get('stationarity', np.nan) for p in points], stationarity_floor)

    for point in points:
        if point.get('cell') == CONTROL_CELL:
            point['margin'] = (point['stationarity'] - stationarity_floor) / stationarity_scale
        elif point.get('cell') in DESIGN_CELLS:
            point['margin'] = min(
                abs(point[MOVING] - moving_split) / moving_scale,
                abs(point[COHERENCE] - coherence_split) / coherence_scale,
            )
        else:
            point['margin'] = float('nan')

    targets = {cell: per_cell for cell in DESIGN_CELLS}
    if controls:
        targets[CONTROL_CELL] = controls

    chosen: List[Dict[str, Any]] = []
    taken_per_video: Dict[str, int] = {}
    taken_frames: Dict[str, List[int]] = {}
    taken_per_group: Dict[Tuple[str, str], int] = {}
    for cell, wanted in targets.items():
        ranked = sorted((p for p in points if p.get('cell') == cell), key=lambda p: -p['margin'])
        for point in ranked:
            if sum(1 for c in chosen if c['cell'] == cell) >= wanted:
                break
            video = point['video']
            if taken_per_video.get(video, 0) >= max_per_video:
                continue
            if any(abs(point['frame'] - f) < min_separation for f in taken_frames.get(video, [])):
                continue
            group = (cell, str(point.get('group', video)))
            if max_per_group is not None and taken_per_group.get(group, 0) >= max_per_group:
                continue
            taken_per_group[group] = taken_per_group.get(group, 0) + 1
            chosen.append(point)
            taken_per_video[video] = taken_per_video.get(video, 0) + 1
            taken_frames.setdefault(video, []).append(point['frame'])

    filled = {cell: sum(1 for c in chosen if c['cell'] == cell) for cell in targets}
    return {
        'targets': targets,
        'filled': filled,
        'unfilled': {cell: wanted - filled[cell] for cell, wanted in targets.items() if filled[cell] < wanted},
        'selected': chosen,
        'constraints': {
            'max_per_video': max_per_video,
            'min_separation': min_separation,
            'max_per_group': max_per_group,
        },
    }


def _split_basis(explicit: Dict[str, bool]) -> str:
    """Where the splits came from: 'pool median', 'explicit', or which axis was given."""
    if all(explicit.values()):
        return 'explicit'
    if not any(explicit.values()):
        return 'pool median'
    return ', '.join(f"{axis} {'explicit' if given else 'pool median'}" for axis, given in explicit.items())


def _spread(values: Sequence[float], center: float) -> float:
    """Median absolute deviation from the split, floored so a degenerate pool cannot divide by zero."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 1.0
    return max(float(np.median(np.abs(finite - center))), 1e-9)


# --------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------


def resolve_plan(args: argparse.Namespace) -> Tuple[int, List[int]]:
    """
    The horizon a segment must leave room for and the offsets it is read at (one per bin of the
    sampling plan's `survey_bins`, else of its `bins`), from the dynamic config (`-c`); `--horizon`
    overrides the horizon.
    """
    from stabilo_benchmark.config import ConfigError, load_config  # noqa: PLC0415 - avoids import cycle
    from stabilo_benchmark.sampling import plan_bins, plan_span, representative_offsets  # noqa: PLC0415

    config_path = getattr(args, 'config', None)
    if config_path is None:
        raise SurveyError('Pass -c <dynamic config>: its sampling plan sets the offsets a case is read at')
    try:
        sampling = load_config(Path(config_path)).get('sampling') or {}
    except ConfigError as error:
        raise SurveyError(str(error)) from error
    horizon = getattr(args, 'horizon', None)
    if horizon is None:
        horizon = plan_span(sampling)
        if horizon is None:
            raise SurveyError(f"The sampling plan in {config_path} is not bounded; pass --horizon")
    if horizon < 0:
        raise SurveyError(f"--horizon must be >= 0, got {horizon}")
    offsets = representative_offsets(plan_bins(sampling, 'survey_bins'), horizon)
    if not offsets:
        raise SurveyError(f"The sampling plan in {config_path} has no bounded bin within {horizon} frames")
    return int(horizon), offsets


def survey_thresholds() -> Dict[str, Any]:
    """The constants a candidate's descriptors depend on (`scene`), recorded with every survey."""
    return {
        'platform_long_side': scene.PLATFORM_LONG_SIDE,
        'object_patch': list(scene.OBJECT_PATCH),
        'appearance_threshold': scene.APPEARANCE_THRESHOLD,
        'size_tolerance': scene.SIZE_TOLERANCE,
        'displaced_widths': scene.DISPLACED_WIDTHS,
        'coherent_widths': scene.COHERENT_WIDTHS,
        'coherent_share': scene.COHERENT_SHARE,
        'min_displaced_objects': scene.MIN_DISPLACED_OBJECTS,
    }


def pooled_candidates(paths: Sequence[Path], reading: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    The candidates of earlier surveys (`--pool`), each read the same way (`reading`): horizon, offsets,
    anchoring, group pattern and thresholds. A survey read otherwise is refused, since its
    descriptors would not be comparable with the rest of the pool.
    """
    candidates: List[Dict[str, Any]] = []
    for path in paths:
        report = _read_json(Path(path))
        if not isinstance(report, dict) or not isinstance(report.get('candidates'), list):
            raise SurveyError(f"{path} is not a survey's {SURVEY_MANIFEST}")
        differs = [key for key, value in reading.items() if report.get(key) != value]
        if differs:
            raise SurveyError(f"{path} was read with another {', '.join(differs)}; pool only surveys of one plan")
        candidates.extend(report['candidates'])
    return candidates


def run_survey(sources: Sequence[Path], args: argparse.Namespace) -> Path:
    """CLI entry point: characterize every candidate, assign cells, and select the cases."""
    work_dir = Path(args.work_dir)
    horizon, offsets = resolve_plan(args)
    anchor = getattr(args, 'anchor', 'start')
    group_pattern = getattr(args, 'group_pattern', None)
    pool = [Path(path) for path in getattr(args, 'pool', None) or []]
    if not sources and not pool:
        raise SurveyError('Pass candidate recordings, or earlier surveys to pool with --pool')
    reading = {
        'horizon': horizon,
        'offsets': offsets,
        'anchor': anchor,
        'scan_stride': args.scan_stride if anchor == 'scan' else None,
        'group_pattern': group_pattern,
        'thresholds': survey_thresholds(),
    }
    names = unique_names([Path(source) for source in sources])

    candidates = pooled_candidates(pool, reading)
    for number, source in enumerate(sources, start=1):
        name = names[Path(source)]
        try:
            found = scan_source(
                Path(source),
                work_dir,
                offsets,
                horizon=horizon,
                anchor=anchor,
                stride=args.scan_stride,
                annotate=not args.no_annotate,
                device=getattr(args, 'device', None),
                name=name,
                group=group_of(name, group_pattern),
                discard=getattr(args, 'discard_frames', False),
            )
        except SurveyError as error:
            print(f"\033[93m[{number}/{len(sources)}] Skipping {source}: {error}\033[0m", flush=True)
            continue
        candidates.extend(found)
        print(f"[{number}/{len(sources)}] {name}: {len(found)} segment(s)", flush=True)

    if not candidates:
        raise SurveyError('No candidate could be characterized.')
    repeated = sorted(name for name, count in Counter(c['name'] for c in candidates).items() if count > 1)
    if repeated:
        raise SurveyError(f"Segments described twice: {', '.join(repeated[:5])}; pool disjoint surveys")

    strata = assign_cells(candidates, args.moving_split, args.coherence_split)
    plan = select_cases(
        candidates,
        strata,
        per_cell=args.per_cell,
        controls=args.controls,
        max_per_video=args.max_per_video,
        min_separation=args.min_separation,
        max_per_group=getattr(args, 'max_per_group', None) if group_pattern else None,
    )
    report: Dict[str, Any] = {
        'timestamp': system_metadata()['timestamp'],
        'config': describe_path(Path(args.config)),
        **reading,
        'pooled': [describe_path(path) for path in pool],
        'n_candidates': len(candidates),
        'strata': strata,
        'selection': plan,
        'candidates': candidates,
    }
    out = Path(args.output) if args.output else work_dir / SURVEY_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    _print_plan(candidates, strata, plan, out, args.config)
    return out


# --------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------


def _print_plan(
    candidates: Sequence[Dict[str, Any]], strata: Dict[str, Any], plan: Dict[str, Any], out: Path, config: Path
) -> None:
    """The selection, the pool it drew from, and the cells it could not fill."""
    splits = strata['splits']
    print(
        f"\n\033[92mRead {len(candidates)} segment(s) across "
        f"{len({c['video'] for c in candidates})} recording(s)\033[0m"
    )
    print(
        f"Splits ({splits['basis']}): moving {100 * splits['moving']:.3f}% of the frame, "
        f"coherence {splits['coherence']:.3f}; controls: stationarity >= {splits['control_stationarity']:.3f} "
        f"and static >= {100 * splits['control_static']:.3f}%"
    )
    for axis in strata.get('degenerate_axes', []):
        print(
            f"\033[93mEvery candidate has essentially the same {axis}, so the split on it is arbitrary and\n"
            f"the cells it produces are not real strata. The pool needs segments that differ in {axis}.\033[0m"
        )

    selected = plan['selected']
    if selected:
        width = max(len(c['name']) for c in selected)
        header = (
            f"\n{'segment':<{width}}  {'cover%':>6}  {'static':>6}  {'moving':>6}  {'coher.':>6}  "
            f"{'still':>5}  {'texture':>7}  {'margin':>6}  cell"
        )
        print(f"\033[92m{header}\033[0m")
        print('-' * (len(header) - 1))
        for point in sorted(selected, key=lambda c: (c['cell'], -c['margin'])):
            print(
                f"{point['name']:<{width}}  {100 * point['coverage']:>6.2f}  {100 * point['static']:>6.2f}  "
                f"{100 * point[MOVING]:>6.2f}  {_fmt(point[COHERENCE], '6.2f')}  "
                f"{_fmt(point['stationarity'], '5.2f')}  {point['texture']:>7.0f}  {point['margin']:>6.2f}  "
                f"{point['cell']}"
            )

    print('\nDesign:')
    for cell, wanted in plan['targets'].items():
        got = plan['filled'][cell]
        mark = '\033[92mok\033[0m' if got >= wanted else '\033[91mSHORT\033[0m'
        print(f"  {cell:<24} {got}/{wanted}  {mark}  (pool: {len(strata['cells'].get(cell, []))})")

    unclassified = len(strata['cells'].get(UNCLASSIFIED, []))
    if unclassified:
        print(f"  {unclassified} candidate(s) unclassified: too few displaced objects for coherence")

    if plan['unfilled']:
        print(
            "\n\033[91mThe design is not filled. Cells are never padded from a neighboring cell; "
            "add recordings covering:\033[0m"
        )
        for cell, missing in plan['unfilled'].items():
            print(f"\033[91m  {missing} more: {cell}\033[0m")

    print(f"\n\033[92mSurvey -> {out}\033[0m")
    if selected:
        print('Next, per selected segment (see SURVEY.json for every one):')
        example = selected[0]
        print(
            f"  stabilo-benchmark prepare {example['source']} -o <case dir> --reference {example['frame']} "
            f"--cell '{example['cell']}' -c {config}"
        )


def _fmt(value: Optional[float], spec: str) -> str:
    """Right-aligned number, or a dash when it is undefined."""
    if value is None or not np.isfinite(value):
        return format('-', f">{spec.split('.', 1)[0]}")
    return format(value, f">{spec}")
