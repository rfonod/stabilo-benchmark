#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
oracle.py - The reference estimator (pseudo-ground truth) for the dynamic benchmark.

A strong pipeline from Part I is run once per frame pair at full resolution with a large
feature budget and OBB masking, and its homographies are cached. Every candidate is scored
against that one cache.

It is not ground truth. `audit` measures its error against ground control points; every pair
records its own diagnostics; a pair it fails on is recorded as failed and excluded from
reference-based scoring, while the reference-free metrics are still computed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from stabilo_benchmark.masking import IMAGES_DIR, LabelError, Labels, load_annotations, require_annotations
from stabilo_benchmark.metrics.diagnostics import estimation_failed, matching_stats
from stabilo_benchmark.pipelines import PipelineSpec
from stabilo_benchmark.results import system_metadata
from stabilo_benchmark.sampling import FramePair

ORACLE_DIR = 'oracle'
ORACLE_FILE = 'ORACLE.json'


class OracleError(RuntimeError):
    """Raised when the reference estimator cannot be computed or its cache is unusable."""


def pair_key(pair: FramePair) -> str:
    """Stable cache key for a frame pair, e.g. '000000->001500'."""
    return f"{pair.reference_index:06d}->{pair.current_index:06d}"


def frame_path(case_dir: Path, index: int) -> Path:
    return Path(case_dir) / IMAGES_DIR / f"{index:06d}.jpg"


def load_frame(case_dir: Path, index: int) -> np.ndarray:
    path = frame_path(case_dir, index)
    image = cv2.imread(str(path))
    if image is None:
        raise OracleError(f"Could not read frame {index} at {path}")
    return image


def load_frame_labels(case_dir: Path, index: int, width: int, height: int) -> Labels:
    """
    A frame's horizontal and oriented boxes.

    The masking mode chooses which one stabilo receives; metric regions and the motion axis always
    use the oriented boxes.
    """
    return load_annotations(case_dir, f"{index:06d}", width, height)


def boxes_for(labels: Labels, masking: str) -> Optional[np.ndarray]:
    """Object boxes in the format stabilo expects for a masking mode."""
    return labels.boxes_for(masking)


def compute_oracle(
    case_dir: Path,
    pairs: Sequence[FramePair],
    spec: PipelineSpec,
    overwrite: bool = False,
    verbosity: int = 0,
) -> Dict[str, Any]:
    """
    Estimate and cache one reference homography per frame pair.

    The reference frame's features are computed once and reused for every pair.
    """
    case_dir = Path(case_dir)
    output_dir = case_dir / ORACLE_DIR
    output_path = output_dir / ORACLE_FILE

    if output_path.exists() and not overwrite:
        print(f"\033[33mReference estimates already cached at {output_path}. Use --overwrite to recompute.\033[0m")
        return load_oracle(case_dir)

    if not pairs:
        raise OracleError("No frame pairs to estimate")

    try:
        require_annotations(case_dir, [spec.masking])
    except LabelError as error:
        raise OracleError(str(error)) from error

    reference_index = pairs[0].reference_index
    reference_frame = load_frame(case_dir, reference_index)
    height, width = reference_frame.shape[:2]
    reference_labels = load_frame_labels(case_dir, reference_index, width, height)

    stabilizer = spec.build(
        min_good_match_count_warning=20 if verbosity > 1 else -1,
        min_inliers_match_count_warning=10 if verbosity > 1 else -1,
    )
    box_format = spec.box_format or 'xywh'
    stabilizer.set_ref_frame(reference_frame, boxes_for(reference_labels, spec.masking), box_format=box_format)

    homographies: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    failures: List[str] = []

    for pair in tqdm(pairs, desc='Reference estimator', unit='pair'):
        key = pair_key(pair)
        frame = load_frame(case_dir, pair.current_index)
        labels = load_frame_labels(case_dir, pair.current_index, width, height)

        start = time.perf_counter()
        stabilizer.stabilize(frame, boxes_for(labels, spec.masking), box_format=box_format)
        elapsed = time.perf_counter() - start

        matrix = stabilizer.get_cur_trans_matrix()
        failed = estimation_failed(stabilizer, matrix)
        if failed:
            failures.append(key)
            homographies[key] = None
        else:
            homographies[key] = np.asarray(matrix, dtype=np.float64).tolist()

        diagnostics[key] = {
            **pair.to_dict(),
            **matching_stats(stabilizer),
            'failed': failed,
            'time_s': elapsed,
        }

    if failures:
        print(
            f"\033[93mThe reference estimator failed on {len(failures)}/{len(pairs)} pair(s). Those pairs are "
            f"excluded from reference-based metrics; reference-free photometric metrics still apply.\033[0m"
        )

    payload = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'pipeline': spec.to_dict(),
        'reference_index': reference_index,
        'frame_size': {'width': width, 'height': height},
        'n_pairs': len(pairs),
        'n_failed': len(failures),
        'failed_pairs': failures,
        'homographies': homographies,
        'diagnostics': diagnostics,
        'metadata': system_metadata(),
        'caveat': (
            'Pseudo-ground truth from a reference estimator, not true ground truth. Its own '
            'error is measured by stabilo-benchmark audit against hand-annotated ground control '
            'points; do not claim differences below that residual.'
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f"\033[92mReference estimates -> {output_path}\033[0m")
    return payload


def load_oracle(case_dir: Path) -> Dict[str, Any]:
    """Read the cached reference estimates, with an actionable error when absent."""
    path = Path(case_dir) / ORACLE_DIR / ORACLE_FILE
    if not path.exists():
        raise OracleError(
            f"No reference estimates found at {path}. Compute them first with:\n  stabilo-benchmark oracle {case_dir}"
        )
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def oracle_matrices(payload: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
    """The cached homographies as numpy arrays, with None kept for failed pairs."""
    return {
        key: (None if value is None else np.asarray(value, dtype=np.float64))
        for key, value in payload.get('homographies', {}).items()
    }
