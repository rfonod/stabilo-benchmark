#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
sampling.py - Frame-pair selection for the dynamic benchmark.

Each measurement is a pair (reference frame, current frame) whose homography is estimated
directly, never by chaining adjacent transforms. The reference defaults to the first frame;
'middle' and an explicit index are alternatives. All modes are deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class SamplingError(ValueError):
    """Raised for an invalid sampling configuration or an out-of-range reference index."""


@dataclass(frozen=True)
class FramePair:
    """One benchmark measurement: register `current_index` onto `reference_index`."""

    reference_index: int
    current_index: int

    @property
    def offset(self) -> int:
        """Signed temporal distance in frames; negative when the current frame precedes."""
        return self.current_index - self.reference_index

    @property
    def abs_offset(self) -> int:
        return abs(self.offset)

    def to_dict(self) -> Dict[str, int]:
        return {
            'reference_index': self.reference_index,
            'current_index': self.current_index,
            'offset': self.offset,
            'abs_offset': self.abs_offset,
        }


def resolve_reference(reference: Any, frame_count: int) -> int:
    """Turn the config's `reference` value ('first', 'middle', or an index) into an index."""
    if frame_count < 2:
        raise SamplingError(f"A dynamic benchmark needs at least two frames, got {frame_count}")
    if reference == 'first':
        return 0
    if reference == 'middle':
        return frame_count // 2
    if isinstance(reference, (int, np.integer)):
        index = int(reference)
        if not 0 <= index < frame_count:
            raise SamplingError(f"Reference frame index {index} is outside [0, {frame_count - 1}]")
        return index
    raise SamplingError(f"Invalid reference {reference!r}; use 'first', 'middle', or a frame index")


def sample_pairs(
    frame_count: int,
    config: Mapping[str, Any],
    reference: Any = 'first',
    seed: int = 0,
    available: Optional[Sequence[int]] = None,
) -> List[FramePair]:
    """
    Build the deterministic list of frame pairs for one dynamic case.

    Args:
        frame_count: Number of frames in the sequence.
        config: The `sampling` block: `mode` plus mode-specific keys.
        reference: 'first', 'middle', or an explicit index.
        seed: Seeds the generator used by the 'random' and 'stratified' modes.
        available: Optional whitelist of frame indices that actually exist on disk
            (a prepared case holds only the sampled frames, not the whole video).

    Modes:
        grid        Every `stride`-th frame after the reference (the default).
        random      `count` frames drawn without replacement.
        stratified  A fixed number of frames drawn from each `[min_offset, max_offset]` bin.

    Grid and stratified plans run forward from the reference: frames before it are never sampled.
    """
    reference_index = resolve_reference(reference, frame_count)
    mode = str(config.get('mode', 'grid'))
    rng = np.random.default_rng(seed)

    candidates = [i for i in range(frame_count) if i != reference_index]
    if available is not None:
        allowed = {int(i) for i in available}
        candidates = [i for i in candidates if i in allowed]
    if not candidates:
        raise SamplingError("No candidate frames available for sampling (is the reference the only frame?)")

    if mode == 'grid':
        indices = _sample_grid(candidates, reference_index, config)
    elif mode == 'random':
        indices = _sample_random(candidates, config, rng)
    elif mode == 'stratified':
        indices = _sample_stratified(candidates, reference_index, config, rng)
    else:
        raise SamplingError(f"Unknown sampling mode '{mode}'; choose from ('grid', 'random', 'stratified')")

    max_pairs = config.get('max_pairs')
    if max_pairs is not None:
        indices = indices[: int(max_pairs)]

    return [FramePair(reference_index=reference_index, current_index=int(i)) for i in sorted(indices)]


def _sample_grid(candidates: Sequence[int], reference_index: int, config: Mapping[str, Any]) -> List[int]:
    """
    The frames `reference + k * stride` for k >= 1, within the optional absolute `start` / `stop`
    bounds. Frames before the reference are never sampled.
    """
    stride = int(config.get('stride', 50))
    if stride < 1:
        raise SamplingError(f"'stride' must be >= 1, got {stride}")
    start = int(config.get('start', reference_index))
    stop = config.get('stop')
    return [
        i
        for i in candidates
        if i > reference_index
        and (i - reference_index) % stride == 0
        and i >= start
        and (stop is None or i <= int(stop))
    ]


def _sample_random(candidates: Sequence[int], config: Mapping[str, Any], rng: np.random.Generator) -> List[int]:
    count = int(config.get('count', min(100, len(candidates))))
    count = min(count, len(candidates))
    return [int(i) for i in rng.choice(np.asarray(candidates), size=count, replace=False)]


def _sample_stratified(
    candidates: Sequence[int], reference_index: int, config: Mapping[str, Any], rng: np.random.Generator
) -> List[int]:
    bins: Sequence[Sequence[int]] = config.get('bins') or []
    if not bins:
        raise SamplingError("'bins' is required for stratified sampling: a list of [min_offset, max_offset] pairs")
    per_bin = int(config.get('per_bin', 10))

    selected: List[int] = []
    for bin_index, bounds in enumerate(bins):
        if len(bounds) != 2:
            raise SamplingError(f"'bins[{bin_index}]' must be [min_offset, max_offset], got {bounds!r}")
        low, high = int(bounds[0]), int(bounds[1])
        in_bin = [i for i in candidates if low <= i - reference_index <= high]
        if not in_bin:
            continue
        take = min(per_bin, len(in_bin))
        selected.extend(int(i) for i in rng.choice(np.asarray(in_bin), size=take, replace=False))
    if not selected:
        raise SamplingError(f"Stratified sampling selected no frames; do the bins {list(bins)} cover the sequence?")
    return selected


def plan_span(config: Mapping[str, Any]) -> Optional[int]:
    """
    How many frames past the reference a sampling plan reaches, or None when it is not bounded.

    A grid reaches `stride * max_pairs`; a stratified plan its widest bin. A random plan, or a grid
    without `max_pairs`, takes whatever the recording holds.
    """
    mode = str(config.get('mode', 'grid'))
    if mode == 'grid':
        max_pairs = config.get('max_pairs')
        return None if max_pairs is None else int(config.get('stride', 50)) * int(max_pairs)
    if mode == 'stratified':
        bins = config.get('bins') or []
        return max((int(bounds[1]) for bounds in bins), default=None)
    return None


# Offset bins for results whose sampling plan names none (a grid): [low, high] in frames, open-ended last.
DEFAULT_OFFSET_BINS: Tuple[Tuple[int, Optional[int]], ...] = (
    (1, 100),
    (101, 500),
    (501, 1000),
    (1001, 5000),
    (5001, None),
)


def plan_bins(config: Optional[Mapping[str, Any]], key: str = 'bins') -> Tuple[Tuple[int, Optional[int]], ...]:
    """
    The offset bins a sampling plan declares under `key` ('bins', 'survey_bins' or 'audit_bins').

    Falls back to the analysis bins for 'survey_bins' and 'audit_bins', and to DEFAULT_OFFSET_BINS
    without either.
    """
    config = config or {}
    bins = config.get(key) or (config.get('bins') if key != 'bins' else None)
    if not bins:
        return DEFAULT_OFFSET_BINS
    return tuple((int(low), int(high)) for low, high in bins)


def representative_offsets(bins: Sequence[Tuple[int, Optional[int]]], horizon: Optional[int] = None) -> List[int]:
    """
    One offset per bounded bin, at its geometric center, for bins that end within `horizon`.

    `survey` reads a candidate case at these offsets of its `survey_bins`, one frame per bin, with
    equal weight.
    """
    offsets = []
    for low, high in bins:
        if high is None or (horizon is not None and high > horizon):
            continue
        offset = int(round(float(np.sqrt(max(1, int(low)) * int(high)))))
        if offset not in offsets:
            offsets.append(offset)
    return offsets


def bin_label(low: int, high: Optional[int]) -> str:
    return f">{low - 1}" if high is None else f"{low}-{high}"


def offset_bin_label(offset: int, bins: Sequence[Tuple[int, Optional[int]]]) -> Optional[str]:
    """The label of the bin holding `abs(offset)`, or None when no bin does."""
    distance = abs(int(offset))
    for low, high in bins:
        if low <= distance and (high is None or distance <= high):
            return bin_label(low, high)
    return None


def offset_bins(pairs: Sequence[FramePair], edges: Sequence[int]) -> Dict[str, List[FramePair]]:
    """
    Group pairs by absolute temporal offset, for reporting error against distance.

    `edges` are inclusive upper bounds in frames; a final open-ended bin catches the rest.
    """
    grouped: Dict[str, List[FramePair]] = {}
    sorted_edges = sorted(int(e) for e in edges)
    for pair in pairs:
        label = f">{sorted_edges[-1]}" if sorted_edges else 'all'
        low = 0
        for edge in sorted_edges:
            if pair.abs_offset <= edge:
                label = f"{low}-{edge}"
                break
            low = edge + 1
        grouped.setdefault(label, []).append(pair)
    return grouped


def summarize(pairs: Sequence[FramePair], fps: Optional[float] = None) -> Dict[str, Any]:
    """Summary of a sampling plan, recorded in the manifest."""
    if not pairs:
        return {'n_pairs': 0}
    offsets = np.array([pair.abs_offset for pair in pairs], dtype=float)
    summary: Dict[str, Any] = {
        'n_pairs': len(pairs),
        'reference_index': pairs[0].reference_index,
        'offset_min': int(offsets.min()),
        'offset_max': int(offsets.max()),
        'offset_median': float(np.median(offsets)),
    }
    if fps:
        summary['span_seconds'] = float(offsets.max() / fps)
    return summary


def frames_to_extract(pairs: Sequence[FramePair]) -> List[int]:
    """Sorted, de-duplicated frame indices a prepared case must contain for these pairs."""
    needed = {pair.reference_index for pair in pairs} | {pair.current_index for pair in pairs}
    return sorted(needed)


def plan(
    frame_count: int, config: Mapping[str, Any], reference: Any = 'first', seed: int = 0
) -> Tuple[List[FramePair], List[int]]:
    """Convenience wrapper: the pair list and the frame indices needed to realize it."""
    pairs = sample_pairs(frame_count, config, reference=reference, seed=seed)
    return pairs, frames_to_extract(pairs)
