#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
resources.py - Separate "this machine cannot run it" from "this pipeline got it wrong".

A resource error (out of memory on the host or the device) aborts its run, which is recorded as
`infeasible`: no measurements, an explicit status, and N/A in the tables. An estimation failure
is different and still counts as a miss.

Classification matches exception types and messages and treats an unrecognized error as a
pipeline failure. A process killed by the kernel cannot be caught; the runners order the grid
cheapest-first for that case.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Substrings of the messages that CUDA, MPS, OpenCV, numpy, and the host allocator use when
# an allocation fails. Lowercased before matching.
_RESOURCE_MESSAGES = (
    'out of memory',
    'insufficient memory',
    'not enough memory',
    'failed to allocate',
    'can not allocate',
    'cannot allocate',
    'unable to allocate',
    'allocation failed',
    'invalid buffer size',  # what the MPS allocator says when a tensor exceeds the device
    'no kernel image is available',  # a CUDA build that cannot run on this device at all
)


def is_resource_error(error: BaseException) -> bool:
    """
    Whether this exception means the machine ran out of a resource, not that the pipeline failed.

    `MemoryError` (which numpy's allocation failure subclasses) is unambiguous. Everything
    else arrives as a `RuntimeError` from torch or a `cv2.error`, both of which carry the
    reason only in their message, so the message is what is matched.
    """
    if isinstance(error, MemoryError):
        return True
    name = type(error).__name__
    if name in {'OutOfMemoryError', 'OutOfResourcesError'}:  # torch.cuda / triton
        return True
    message = str(error).lower()
    return any(needle in message for needle in _RESOURCE_MESSAGES)


# Linux only: writing '5' to clear_refs resets the peak resident set size (VmHWM), per proc(5).
# Set by `reset_peak_memory` when the write succeeds.
_CLEAR_REFS = Path('/proc/self/clear_refs')
_STATUS = Path('/proc/self/status')
_host_peak_is_per_run = False


def reset_peak_memory() -> None:
    """
    Start a new peak-memory window; the runners call it before each run.

    Resets CUDA's peak statistics and, on Linux, the host peak (VmHWM). macOS's `ru_maxrss`
    cannot be reset and is reported with scope `process`. Also releases the device memory
    torch's caching allocator holds from earlier runs, so feasibility does not depend on run order.
    """
    global _host_peak_is_per_run  # noqa: PLW0603 - one flag for the process, set once it is known

    try:
        _CLEAR_REFS.write_text('5', encoding='utf-8')
        _host_peak_is_per_run = True
    except OSError:
        _host_peak_is_per_run = False

    torch = sys.modules.get('torch')
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gc.collect()  # drop unreferenced models first, so their blocks are returnable
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def peak_memory() -> Dict[str, Any]:
    """
    Peak host memory and, when torch is already loaded, device memory, in MB.

    `host_peak_scope` is `run` when `reset_peak_memory` could reset the host counter (Linux) and
    `process` otherwise. CUDA peaks are per run. MPS exposes no peak, so its current allocation is
    recorded as `device_allocated_mb`. Entries that cannot be read are omitted; torch is never
    imported here.
    """
    usage: Dict[str, Any] = {}

    host = _host_peak_from_status() if _host_peak_is_per_run else None
    if host is not None:
        usage['host_peak_mb'] = host
        usage['host_peak_scope'] = 'run'
    else:
        try:
            import resource  # noqa: PLC0415 - unix only, and not worth a module-level import

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is bytes on macOS and kilobytes on Linux; the platform is the only tell.
            usage['host_peak_mb'] = peak / 1e6 if sys.platform == 'darwin' else peak / 1e3
            usage['host_peak_scope'] = 'process'
        except Exception:
            pass

    torch = sys.modules.get('torch')
    if torch is not None:
        try:
            if torch.cuda.is_available():
                usage['device'] = 'cuda'
                usage['device_peak_mb'] = torch.cuda.max_memory_allocated() / 1e6
                usage['device_reserved_peak_mb'] = torch.cuda.max_memory_reserved() / 1e6
            elif torch.backends.mps.is_available():
                usage['device'] = 'mps'
                usage['device_allocated_mb'] = torch.mps.driver_allocated_memory() / 1e6
        except Exception:
            pass

    return usage


def _host_peak_from_status() -> Optional[float]:
    """VmHWM from /proc/self/status in MB, or None where it cannot be read."""
    try:
        for line in _STATUS.read_text(encoding='utf-8').splitlines():
            if line.startswith('VmHWM:'):
                return float(line.split()[1]) / 1e3  # reported in kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def describe_resource_error(error: BaseException, where: Optional[str] = None) -> Dict[str, Any]:
    """The record written into an infeasible run: what failed, where, and the memory in use."""
    description: Dict[str, Any] = {
        'type': type(error).__name__,
        'message': str(error),
        'peak_memory': peak_memory(),
    }
    if where is not None:
        description['where'] = where
    return description
