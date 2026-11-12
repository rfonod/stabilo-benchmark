#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
prepare.py - Build a dynamic benchmark case from a hovering-drone video.

A case holds the sampled frames, their annotations and a manifest of where they came from, so
every pipeline reads identical inputs.

    <case>/
      CASE.json               video provenance, sampling plan, frame list
      images/000123.jpg       sampled frames, named by their original frame index
      labels_hbb/000123.txt   written by hbb2obb-detect
      labels_obb/000123.txt   written by hbb2obb
      gcp/                    optional, hand-annotated ground control points
      oracle/                 written later by `stabilo-benchmark oracle`
      results/                written later by `stabilo-benchmark dynamic`

`prepare` writes the frames and the manifest; the labels come from hbb2obb (see `annotation.py`).

  <video>              Decode the sampled frames from a recording, sequentially (seeking with
                       `CAP_PROP_POS_FRAMES` is not frame-accurate for several codecs).
  --from-frames <dir>  Adopt the frames of a directory of extracted frames (every one, or those
                       `--frames` names), renamed by the index in their filename. `-c` supplies
                       the reference and the offset bins that `annotate-gcp`, `audit` and
                       `report` read; the frames themselves are taken as provided.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

from stabilo_benchmark.config import load_config
from stabilo_benchmark.masking import HBB_DIR, IMAGES_DIR, OBB_DIR
from stabilo_benchmark.results import describe_path, system_metadata
from stabilo_benchmark.sampling import resolve_reference, sample_pairs

CASE_MANIFEST = 'CASE.json'
# The sampling plan's offset bins, kept in CASE.json for frames adopted as provided.
OFFSET_BIN_KEYS = ('bins', 'survey_bins', 'audit_bins')
# What is made from a case's frames, and so describes the old ones once --overwrite rebuilds them.
DERIVED_FROM_FRAMES = (
    HBB_DIR,
    OBB_DIR,
    'labels_confidence',
    'PROVENANCE_hbb.txt',
    'PROVENANCE_obb.txt',
    'EXCLUSION.json',
    'gcp',
    'gcp_repeat',
    'gcp_manual',
    'oracle',
    'results',
)


class PrepareError(RuntimeError):
    """Raised when a case cannot be prepared: unreadable video, unusable frames."""


def clear_for_rebuild(case_dir: Path) -> None:
    """
    Make way for `--overwrite`: refuse while anything made from the case's current frames remains
    (labels, points, polygons, caches, results), then delete the frames themselves.
    """
    derived = [name for name in DERIVED_FROM_FRAMES if (case_dir / name).exists()]
    if derived:
        raise PrepareError(
            f"{case_dir} holds {', '.join(derived)}, made from its current frames; --overwrite rebuilds the "
            "frames only, so move or delete those first"
        )
    shutil.rmtree(case_dir / IMAGES_DIR, ignore_errors=True)


def probe_video(path: Path) -> Dict[str, Any]:
    """Frame count, fps, and frame size of a video, without decoding it all."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PrepareError(f"Could not open video: {path}")
    try:
        info = {
            'path': describe_path(path),
            'frame_count': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': float(capture.get(cv2.CAP_PROP_FPS)),
            'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if info['frame_count'] < 2:
        raise PrepareError(f"{path} reports {info['frame_count']} frame(s); need at least two")
    return info


def extract_frames(video: Path, indices: Sequence[int], frames_dir: Path, quality: int = 95) -> List[int]:
    """
    Write the requested frames as JPEGs named by their original frame index.

    Decodes from the start and stops after the last requested frame. Returns the indices actually
    extracted, which may be fewer than requested when the container's frame count is too high.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return []

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PrepareError(f"Could not open video: {video}")

    extracted: List[int] = []
    try:
        target = set(wanted)
        last = wanted[-1]
        index = 0
        with tqdm(total=len(wanted), desc='Extracting frames', unit='frame') as progress:
            while index <= last:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in target:
                    out = frames_dir / f"{index:06d}.jpg"
                    if not cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
                        raise PrepareError(f"Failed to write {out}")
                    extracted.append(index)
                    progress.update(1)
                index += 1
    finally:
        capture.release()

    if len(extracted) < len(wanted):
        missing = sorted(set(wanted) - set(extracted))
        print(
            f"\033[93mWarning: {len(missing)} requested frame(s) were past the end of the decodable "
            f"stream and were skipped (first missing: {missing[0]}). The container reported more "
            f"frames than it contains; the sampling plan has been trimmed accordingly.\033[0m"
        )
    return extracted


# --------------------------------------------------------------------------------------
# Cases built from frames that are already extracted
# --------------------------------------------------------------------------------------

# The last run of digits anywhere in the stem.
_TRAILING_DIGITS = re.compile(r'(\d+)(?!.*\d)')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def frame_index_from_name(stem: str) -> int:
    """
    The frame index encoded in a filename: its last run of digits.

    Covers `000500.jpg` and `2022-10-07_D1_PM5_B1.MP4_000500.jpg`.
    """
    match = _TRAILING_DIGITS.search(stem)
    if match is None:
        raise PrepareError(
            f"Cannot read a frame index from '{stem}'. A frame must carry its index as the last "
            f"digits of its filename, e.g. 000500.jpg or 2022-10-07_D1_PM5_B1.MP4_000500.jpg."
        )
    return int(match.group(1))


def collect_source_frames(source: Path) -> Dict[int, Path]:
    """
    Map frame index -> image path for a directory of already-extracted frames.

    A nested `images/` subdirectory is entered automatically.
    """
    source = Path(source).expanduser().absolute()
    if (source / 'images').is_dir():
        source = source / 'images'
    if not source.is_dir():
        raise PrepareError(f"Frame directory not found: {source}")

    entries: Dict[int, Path] = {}
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index = frame_index_from_name(path.stem)
        if index in entries:
            raise PrepareError(
                f"Two frames claim index {index}: '{entries[index].name}' and '{path.name}'. "
                f"A case addresses frames by index, so they must be unique; prepare one "
                f"recording at a time."
            )
        entries[index] = path

    if len(entries) < 2:
        raise PrepareError(
            f"{source} holds {len(entries)} usable frame(s) ({', '.join(IMAGE_SUFFIXES)}); "
            f"a case needs at least two so there is a pair to register."
        )
    return entries


def adopt_frames(
    entries: Dict[int, Path], frames_dir: Path, scale: float = 1.0, quality: int = 95
) -> Tuple[List[int], int, int]:
    """
    Copy the source frames into the case, renamed to their index, and return (indices, w, h).

    At `scale == 1.0` files are copied byte for byte; otherwise they are resized with `INTER_AREA`.
    """
    if not scale > 0:
        raise PrepareError(f"--scale must be positive, got {scale}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    size: Optional[Tuple[int, int]] = None
    written: List[int] = []
    for index in tqdm(sorted(entries), desc='Adopting frames', unit='frame'):
        source_path = entries[index]
        target = frames_dir / f"{index:06d}.jpg"
        if scale == 1.0 and source_path.suffix.lower() in ('.jpg', '.jpeg'):
            shutil.copyfile(source_path, target)
            image = cv2.imread(str(target))
        else:
            image = cv2.imread(str(source_path))
            if image is None:
                raise PrepareError(f"Could not read frame: {source_path}")
            if scale != 1.0:
                height, width = image.shape[:2]
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if image is None:
            raise PrepareError(f"Could not read frame: {source_path}")

        frame_size = (image.shape[1], image.shape[0])
        if size is None:
            size = frame_size
        elif frame_size != size:
            raise PrepareError(
                f"Frame {index} is {frame_size[0]}x{frame_size[1]} but earlier frames are "
                f"{size[0]}x{size[1]}. A case must hold one frame size: a homography between "
                f"differently sized frames is not the quantity this benchmark measures."
            )
        written.append(index)

    assert size is not None  # guaranteed: collect_source_frames requires >= 2 frames
    return written, size[0], size[1]


def run_prepare_from_frames(source: Path, args: argparse.Namespace) -> Path:
    """Prepare a case from a directory of already-extracted frames, and return it."""
    source = Path(source).expanduser().absolute()
    case_dir = Path(args.output).expanduser().absolute() if args.output else source.parent / f"case_{source.name}"

    if case_dir.exists() and (case_dir / CASE_MANIFEST).exists() and not args.overwrite:
        print(f"\033[33mCase already prepared at {case_dir}. Use --overwrite to rebuild.\033[0m")
        return case_dir

    entries = collect_source_frames(source)
    selected = getattr(args, 'frames', None)
    if selected:
        missing = sorted(set(selected) - set(entries))
        if missing:
            raise PrepareError(f"Frame(s) {', '.join(map(str, missing))} not found in {source}")
        entries = {index: entries[index] for index in sorted(set(selected))}
        if len(entries) < 2:
            raise PrepareError('A case needs at least two frames; --frames names fewer')
    if args.max_frames is not None:
        keep = sorted(entries)[: int(args.max_frames)]
        entries = {index: entries[index] for index in keep}

    # A config gives the reference (unless --reference does) and the offset bins of its sampling plan.
    config = load_config(Path(args.config)) if getattr(args, 'config', None) else {}
    reference = args.reference if args.reference is not None else config.get('reference', 'first')
    plan = config.get('sampling') or {}

    original = cv2.imread(str(entries[sorted(entries)[0]]))
    if original is None:
        raise PrepareError(f"Could not read frame: {entries[sorted(entries)[0]]}")
    source_size = (original.shape[1], original.shape[0])

    scale = float(getattr(args, 'scale', 1.0) or 1.0)
    # No video to probe: frame_count is inferred from the indices.
    frame_count = max(entries) + 1
    reference_index = resolve_reference(reference, frame_count)
    if reference_index not in entries:
        raise PrepareError(
            f"The reference frame {reference_index} is not among the frames in {source} "
            f"(present: {', '.join(str(i) for i in sorted(entries)[:8])}...). Pass --reference "
            f"with one of those indices."
        )

    resized = ''
    if scale != 1.0:
        resized = f" scaled by {scale} -> {round(source_size[0] * scale)}x{round(source_size[1] * scale)}"
    print(f"\033[92mPreparing case from {len(entries)} frame(s) in {source}\033[0m")
    print(
        f"\033[92m  {source_size[0]}x{source_size[1]}{resized}; reference frame {reference_index} -> {case_dir}\033[0m"
    )

    if (case_dir / CASE_MANIFEST).exists():
        own = (case_dir / IMAGES_DIR).resolve()
        if any(path.resolve().is_relative_to(own) for path in entries.values()):
            raise PrepareError(f"{source} holds {case_dir}'s own frames, which --overwrite replaces; adopt a copy")
        clear_for_rebuild(case_dir)
    extracted, width, height = adopt_frames(entries, case_dir / IMAGES_DIR, scale=scale, quality=int(args.jpeg_quality))

    # The `video` block describes the source recording; `available` says whether its file is here.
    info = {
        'path': describe_path(source),
        'available': False,
        'source_kind': 'frames',
        'video_name': getattr(args, 'video_name', None),
        'frame_count': frame_count,
        'frame_count_inferred': True,
        'fps': float(args.fps) if getattr(args, 'fps', None) else None,
        'width': width,
        'height': height,
        'original_width': source_size[0],
        'original_height': source_size[1],
        'scale': scale,
        'frame_files': {str(index): entries[index].name for index in extracted},
    }
    sampling = {'mode': 'as-provided', 'source': describe_path(source), 'n_frames': len(extracted)}
    sampling.update({key: plan[key] for key in OFFSET_BIN_KEYS if key in plan})

    return _finalize_case(
        case_dir, info, reference_index, sampling, int(args.seed), extracted, cell=getattr(args, 'cell', None)
    )


def _finalize_case(
    case_dir: Path,
    info: Dict[str, Any],
    reference_index: int,
    sampling: Dict[str, Any],
    seed: int,
    extracted: Sequence[int],
    cell: Optional[str] = None,
) -> Path:
    """
    Write CASE.json and report what to run next.

    `cell` is the survey cell (`--cell`), which the dynamic runner records with every measurement.
    """
    manifest = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'video': info,
        'reference_index': reference_index,
        'sampling': sampling,
        'seed': seed,
        'frames': list(extracted),
        'n_frames': len(extracted),
        'cell': cell,
        'metadata': system_metadata(),
    }
    (case_dir / CASE_MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')

    print(f"\033[92mCase ready: {case_dir}\033[0m")
    print(f"\033[92m  {len(extracted)} frames in {case_dir / IMAGES_DIR}\033[0m")
    print(
        "\033[92m  Next: annotate the images with hbb2obb (README, 'Annotating images'), then "
        "stabilo-benchmark oracle <case>\033[0m"
    )
    return case_dir


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_prepare(video: Path, args: argparse.Namespace) -> Path:
    """Prepare one dynamic benchmark case and return its directory."""
    video = Path(video).expanduser().absolute()
    case_dir = Path(args.output).expanduser().absolute() if args.output else video.parent / f"case_{video.stem}"

    if case_dir.exists() and (case_dir / CASE_MANIFEST).exists() and not args.overwrite:
        print(f"\033[33mCase already prepared at {case_dir}. Use --overwrite to rebuild.\033[0m")
        return case_dir

    info = probe_video(video)

    # Sampling comes from the dynamic config when given, else from the CLI flags. An explicit
    # --reference overrides the config's.
    if getattr(args, 'config', None):
        config = load_config(Path(args.config))
        sampling = config.get('sampling', {}) or {}
        reference = args.reference if args.reference is not None else config.get('reference', 'first')
        seed = int(config.get('seed', 0))
    else:
        sampling = {'mode': 'grid', 'stride': int(args.stride), 'max_pairs': args.max_frames}
        reference = args.reference if args.reference is not None else 'first'
        seed = int(args.seed)

    pairs = sample_pairs(info['frame_count'], sampling, reference=reference, seed=seed)
    reference_index = resolve_reference(reference, info['frame_count'])
    wanted = sorted({reference_index, *(pair.current_index for pair in pairs)})

    print(
        f"\033[92mPreparing case for {video.name}: {info['frame_count']} frames @ {info['fps']:.2f} fps, "
        f"{info['width']}x{info['height']}\033[0m"
    )
    print(f"\033[92mReference frame {reference_index}; extracting {len(wanted)} frame(s) -> {case_dir}\033[0m")

    if (case_dir / CASE_MANIFEST).exists():
        clear_for_rebuild(case_dir)
    extracted = extract_frames(video, wanted, case_dir / IMAGES_DIR, quality=int(args.jpeg_quality))
    if reference_index not in extracted:
        raise PrepareError(f"The reference frame {reference_index} could not be extracted from {video}")

    info['available'] = True
    info['source_kind'] = 'video'
    return _finalize_case(case_dir, info, reference_index, sampling, seed, extracted, cell=getattr(args, 'cell', None))


def load_case(case_dir: Path) -> Dict[str, Any]:
    """Read a prepared case's manifest, with a clear error when the directory is not one."""
    path = Path(case_dir) / CASE_MANIFEST
    if not path.exists():
        raise PrepareError(
            f"{case_dir} is not a prepared case ({CASE_MANIFEST} is missing). "
            f"Build one with: stabilo-benchmark prepare <video> -o {case_dir}"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
