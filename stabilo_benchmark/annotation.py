#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
annotation.py - The two hbb2obb commands that annotate an image directory.

    <root>/
      images/              the frames
      labels_hbb/          hbb2obb-detect: horizontal boxes, absolute pixels
      labels_obb/          hbb2obb: oriented boxes, one per horizontal box, same order
      labels_confidence/   hbb2obb: per-box conversion score (0.0 marks an HBB fallback)
      PROVENANCE_hbb.txt   hbb2obb-detect: command, settings, checkpoint SHA-256
      PROVENANCE_obb.txt   hbb2obb: the same for the conversion

The detector is hbb2obb's registered `geotrax` model; the conversion settings are those selected
for Songdo Vision OBB. The README prints the same two commands, and `tests/test_samples.py`
checks that they match.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

DETECT_ARGS = ['--model', 'geotrax', '--no_confidence', '--save_provenance']
# fmt: off
CONVERT_ARGS = [
    '--sam_models', 'sam_l',
    '--imgsz', '1024',
    '--scale_factors', '0.02',
    '--opening_kernel_percentage', '-0.1',
    '--fragment_ratio', '0.1',
    '--confidence_dir',
    '--save_provenance',
]
# fmt: on


class AnnotationError(RuntimeError):
    """Raised when an hbb2obb command is missing or fails."""


def commands(images_dir: Path, device: Optional[str] = None, overwrite: bool = False) -> List[List[str]]:
    """The hbb2obb-detect and hbb2obb command lines for an image directory."""
    device_args = ['--device', str(device)] if device else []
    detect = ['hbb2obb-detect', str(images_dir), *DETECT_ARGS, *device_args, *(['--overwrite'] if overwrite else [])]
    convert = ['hbb2obb', str(images_dir), *CONVERT_ARGS, *device_args]
    return [detect, convert]


def annotate(images_dir: Path, device: Optional[str] = None, overwrite: bool = False) -> None:
    """Run both commands; labels land beside `images_dir`."""
    for command in commands(Path(images_dir), device=device, overwrite=overwrite):
        executable = _executable(command[0])
        result = subprocess.run([executable, *command[1:]], check=False)
        if result.returncode != 0:
            raise AnnotationError(f"{' '.join(command)} exited with status {result.returncode}")


def _executable(name: str) -> str:
    """The console script beside this interpreter first, so an unactivated venv still finds it."""
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside)
    found = shutil.which(name)
    if found is None:
        raise AnnotationError(f"'{name}' was not found. It is installed with hbb2obb: pip install hbb2obb")
    return found
