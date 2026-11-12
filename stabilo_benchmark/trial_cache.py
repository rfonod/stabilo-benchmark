#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
trial_cache.py - Build each perturbed frame of a static campaign once and reuse it in every run.

A trial's perturbation depends on the seed, the scene and the trial index, not on the
pipeline, so every run of a campaign would otherwise regenerate the same frames. Each entry
holds the warped, degraded frame, its homography and the sampled parameters, so records are
identical with and without the cache.

Entries are keyed per scene by the seed, the scene name and image digest, the perturbation
config, and the numpy and OpenCV versions. Each frame is stored uncompressed (25 MB at 4K).
The directory is deleted when the campaign completes; a failed write disables the cache and
the campaign continues without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

# Part of every key; bump when the stored layout changes.
CACHE_FORMAT = 1


@dataclass(frozen=True)
class CachedTrial:
    """One perturbed frame and everything a record needs from its generation."""

    image: np.ndarray
    matrix: np.ndarray
    geometric_params: Dict[str, Any]
    photometric_params: Dict[str, Any]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """numpy scalars as their Python equivalents, which JSON can hold exactly."""
    return value.item() if isinstance(value, np.generic) else value


# Free space that must remain after the cache's estimated footprint.
DISK_RESERVE_BYTES = 10 * 1024**3


def estimated_bytes(scene_path: Path, n_scenes: int, trials: int) -> int:
    """The cache's size for a campaign, from the first scene's frame size."""
    image = cv2.imread(str(scene_path))
    frame = image.nbytes if image is not None else 3840 * 2160 * 3
    return int(frame) * int(n_scenes) * int(trials)


def fits_on_disk(root: Path, needed: int) -> tuple:
    """Whether `needed` bytes fit under `root` with `DISK_RESERVE_BYTES` to spare, and the free space."""
    probe = Path(root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    already = sum(f.stat().st_size for f in Path(root).rglob('*.npz')) if Path(root).exists() else 0
    return free + already - needed >= DISK_RESERVE_BYTES, free


class TrialCache:
    """Perturbed frames of one campaign, on disk, keyed by what produced them."""

    def __init__(self, root: Path, seed: int, perturbation: Mapping[str, Any]):
        self.root = Path(root)
        self.enabled = True
        self._base = {
            'format': CACHE_FORMAT,
            'seed': int(seed),
            'perturbation': perturbation,
            'numpy': np.__version__,
            'opencv': cv2.__version__,
        }
        self._scene_dirs: Dict[str, Path] = {}

    def _scene_dir(self, scene_path: Path) -> Path:
        key = str(scene_path)
        if key not in self._scene_dirs:
            description = {**self._base, 'scene': scene_path.name, 'scene_sha256': _file_digest(scene_path)}
            canonical = json.dumps(description, sort_keys=True, default=str)
            name = f"{scene_path.stem}-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
            directory = self.root / name
            if self.enabled:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    key_file = directory / 'KEY.json'
                    if not key_file.exists():
                        key_file.write_text(
                            json.dumps(description, indent=2, sort_keys=True, default=str), encoding='utf-8'
                        )
                except OSError as error:
                    self._disable(error)
            self._scene_dirs[key] = directory
        return self._scene_dirs[key]

    def load(self, scene_path: Path, trial: int, width: int, height: int) -> Optional[CachedTrial]:
        """The cached trial, or None when it is absent, unreadable or of the wrong size."""
        if not self.enabled:
            return None
        path = self._scene_dir(scene_path) / f"trial_{int(trial):05d}.npz"
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                image = data['image']
                matrix = data['matrix']
                params = json.loads(str(data['params']))
        except (OSError, ValueError, KeyError):
            return None
        if image.shape[:2] != (height, width):
            return None
        return CachedTrial(image, matrix, params['geometric'], params['photometric'])

    def store(self, scene_path: Path, trial: int, entry: CachedTrial) -> None:
        """Write one trial atomically; on failure, stop caching rather than stop the campaign."""
        if not self.enabled:
            return
        directory = self._scene_dir(scene_path)
        if not self.enabled:
            return
        path = directory / f"trial_{int(trial):05d}.npz"
        params = {
            'geometric': {k: _plain(v) for k, v in entry.geometric_params.items()},
            'photometric': {k: _plain(v) for k, v in entry.photometric_params.items()},
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, 'wb') as handle:
                np.savez(handle, image=entry.image, matrix=entry.matrix, params=np.array(json.dumps(params)))
            os.replace(temporary, path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            self._disable(error)

    def clear(self) -> None:
        """Delete the campaign's cache; called once every run has completed."""
        shutil.rmtree(self.root, ignore_errors=True)
        parent = self.root.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def _disable(self, error: OSError) -> None:
        self.enabled = False
        print(
            f"\033[93mTrial cache disabled ({error}); the campaign continues and regenerates "
            f"every perturbed frame.\033[0m"
        )
