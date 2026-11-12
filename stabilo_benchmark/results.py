#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
results.py - Result storage, run manifests, provenance, and resume.

    MANIFEST.json      config, its hash, the resolved run grid, versions, hardware,
                       seeds; written before the first run and rewritten, with history, on resume
    <run:07>.json      one file per run
    SUCCESS.txt        every planned run completed
    WARNING.txt        finished early, or finished with a run that measured nothing
    FAILURE.txt        raised; contains the traceback and the failing run's parameters
    GCP_SCORES.json    the candidates scored against the case's ground control points, written by
                       `audit --results`; removed whenever runs are written, since it describes them

Each run record carries a `status`: `completed`, or `infeasible` when the machine ran out of
memory and the partial records were discarded (see `resources.py`). Infeasible runs keep the
SUCCESS marker, which names them, and `report` renders them as N/A.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import psutil

from stabilo_benchmark import __version__

RUN_COMPLETED = 'completed'
RUN_INFEASIBLE = 'infeasible'

SUCCESS_MARKER = 'SUCCESS.txt'
WARNING_MARKER = 'WARNING.txt'
FAILURE_MARKER = 'FAILURE.txt'
MANIFEST_NAME = 'MANIFEST.json'
GCP_SCORES_NAME = 'GCP_SCORES.json'


class ResultError(RuntimeError):
    """Raised when results cannot be written, or a resume is inconsistent with the config."""


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def package_versions() -> Dict[str, str]:
    """Versions of every package whose behavior can move a benchmark number."""
    versions = {
        'stabilo_benchmark': __version__,
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'opencv': cv2.__version__,
        'numpy': np.__version__,
    }
    for name in ('stabilo', 'torch', 'kornia', 'shapely', 'pandas'):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = 'not installed'
    return versions


def describe_path(path: Path) -> str:
    """
    A path as recorded in a written file: relative when it is inside the working tree, absolute
    otherwise. Symbolic links are kept, so a link into the tree records the in-tree path rather
    than where the data happen to be stored on this machine.
    """
    path = Path(path).expanduser().absolute()
    try:
        return str(path.relative_to(Path.cwd().absolute()))
    except ValueError:
        return str(path)


def system_metadata() -> Dict[str, Any]:
    """OS, hardware, and accelerator description, for the manifest."""
    metadata: Dict[str, Any] = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'os': {'system': platform.system(), 'platform': platform.platform(), 'release': platform.release()},
        'hardware': {
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / 1024**3, 2),
        },
        'software': package_versions(),
    }

    # torch is imported lazily to keep CLI start-up fast.
    try:
        import torch  # noqa: PLC0415 - deliberate: keeps CLI start-up fast

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            metadata['hardware']['gpu'] = {
                'count': count,
                'names': [torch.cuda.get_device_name(i) for i in range(count)],
                'memory_gb': [
                    round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2) for i in range(count)
                ],
            }
        else:
            metadata['hardware']['gpu'] = {'count': 0, 'names': [], 'memory_gb': []}
    except Exception:
        metadata['hardware']['gpu'] = {'count': 0, 'names': [], 'memory_gb': []}

    return metadata


def seed_everything(seed: int) -> Dict[str, int]:
    """
    Seed numpy, OpenCV and torch, and report what was seeded.

    The three generators are independent: numpy drives perturbation sampling and frame-pair
    selection, torch the learning-based detectors and matchers. OpenCV's robust estimators do not
    read its global RNG (see `seeding.py`). CUDA kernels may still be nondeterministic.
    """
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    seeded = {'numpy': seed, 'opencv': seed}
    try:
        import torch  # noqa: PLC0415 - deliberate: keeps CLI start-up fast

        torch.manual_seed(seed)
        seeded['torch'] = seed
    except Exception:
        pass
    return seeded


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


@dataclass
class Manifest:
    """The complete description of one benchmark invocation."""

    kind: str
    name: str
    config: Dict[str, Any]
    config_hash: str
    seed: int
    runs: Dict[int, Dict[str, Any]]
    metadata: Dict[str, Any]
    history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'kind': self.kind,
            'name': self.name,
            'config_hash': self.config_hash,
            'seed': self.seed,
            'config': self.config,
            'runs': {str(k): v for k, v in self.runs.items()},
            'metadata': self.metadata,
            'history': self.history,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'Manifest':
        return cls(
            kind=payload['kind'],
            name=payload['name'],
            config=payload.get('config', {}),
            config_hash=payload['config_hash'],
            seed=payload['seed'],
            runs={int(k): v for k, v in payload.get('runs', {}).items()},
            metadata=payload.get('metadata', {}),
            history=payload.get('history', []),
        )


def build_manifest(
    config: Mapping[str, Any], config_hash_value: str, runs: Mapping[int, Mapping[str, Any]]
) -> Manifest:
    """Assemble a manifest for a fresh invocation."""
    return Manifest(
        kind=str(config['kind']),
        name=str(config['name']),
        config={k: v for k, v in config.items() if not k.startswith('_')},
        config_hash=config_hash_value,
        seed=int(config['seed']),
        runs={int(k): dict(v) for k, v in runs.items()},
        metadata=system_metadata(),
        history=[],
    )


# --------------------------------------------------------------------------------------
# Result directory
# --------------------------------------------------------------------------------------


class ResultStore:
    """
    Writes and reads one benchmark's result directory.

    Without `overwrite` or `resume`, a directory holding SUCCESS.txt is left untouched.
    """

    def __init__(
        self,
        directory: Path,
        overwrite: bool = False,
        resume: bool = False,
        regrid: bool = False,
        run_key: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ):
        if overwrite and (resume or regrid):
            raise ResultError("'overwrite' and 'resume' (or 'regrid') are mutually exclusive")
        self.regrid = regrid
        self.run_key = run_key or _canonical
        resume = resume or regrid
        self.directory = Path(directory)
        self.overwrite = overwrite
        self.resume = resume

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def is_complete(self) -> bool:
        return (self.directory / SUCCESS_MARKER).exists()

    def run_path(self, run_number: int) -> Path:
        return self.directory / f"{run_number:07}.json"

    def has_run(self, run_number: int) -> bool:
        return self.run_path(run_number).exists()

    def initialize(self, manifest: Manifest) -> Manifest:
        """
        Create the directory and write the manifest, validating a resume against it.

        On resume, a stored run grid that differs from the config's raises; with `regrid`, a config
        that changes only the pipeline grid is accepted (`adopt_grid`). Overwriting removes the
        earlier invocation's run records with its markers, so none can outlive a changed grid; without
        either flag, a directory already holding run records raises rather than discarding them.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        earlier_runs = sorted(self.directory.glob('[0-9]*.json'))

        if self.resume and self.manifest_path.exists():
            stored = Manifest.from_dict(_read_json(self.manifest_path))
            if self.regrid and stored.runs != manifest.runs:
                stored = self.adopt_grid(stored, manifest)
            if stored.runs != manifest.runs:
                raise ResultError(
                    f"Cannot resume {self.directory}: the stored run grid differs from the one this config "
                    f"expands to ({len(stored.runs)} stored vs {len(manifest.runs)} current). "
                    f"Delete {MANIFEST_NAME} to start fresh, or restore the original config."
                )
            resumption = {'resumed_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'metadata': manifest.metadata}
            stored.history = [*stored.history, resumption]
            manifest = stored
            for marker in (FAILURE_MARKER, WARNING_MARKER):
                (self.directory / marker).unlink(missing_ok=True)
        elif self.overwrite:
            for marker in (SUCCESS_MARKER, WARNING_MARKER, FAILURE_MARKER):
                (self.directory / marker).unlink(missing_ok=True)
            for path in earlier_runs:
                path.unlink()
        elif earlier_runs:
            raise ResultError(
                f"{self.directory} already holds {len(earlier_runs)} run record(s) from an earlier invocation. "
                f"Resume it with --resume (-r), or discard them with --overwrite (-o)."
            )

        # Scores against the ground control points describe the runs as they were.
        (self.directory / GCP_SCORES_NAME).unlink(missing_ok=True)
        _write_json(self.manifest_path, manifest.to_dict())
        return manifest

    def adopt_grid(self, stored: Manifest, manifest: Manifest) -> Manifest:
        """
        Adopt a changed pipeline grid: renumber the stored run records still in it to their place in
        `manifest`, delete the others, and return the manifest to resume with: the new config and
        grid with the stored metadata and history, as if the new grid had been run from the start.

        Raises unless the configs are equal outside `pipeline`, so a kept run is measured exactly
        as a new one would be. Runs match on `run_key`, by default their parameters without the
        label, which numbers the grid (`sift+none#12`); a moved record takes the new run's
        parameters. A stored run the new grid does not contain (a removed value, or a preset whose
        values changed) is deleted, and of stored runs that `run_key` makes equivalent beyond the
        new grid's slots (points the grid now runs once) the one whose parameters match exactly is
        kept.
        """
        reason = grid_change_error(stored.config, manifest.config)
        if reason:
            raise ResultError(f"Cannot regrid {self.directory}: {reason}")
        targets: Dict[str, List[int]] = {}
        for number in sorted(manifest.runs):
            targets.setdefault(self.run_key(manifest.runs[number]), []).append(number)
        exact = {_canonical(run): number for number, run in manifest.runs.items()}
        mapping: Dict[int, int] = {}
        dropped: List[int] = []
        for number in sorted(stored.runs, key=lambda n: (_canonical(stored.runs[n]) not in exact, n)):
            free = targets.get(self.run_key(stored.runs[number]))
            if not free:
                dropped.append(number)
                continue
            match = exact.get(_canonical(stored.runs[number]))
            mapping[number] = free.pop(free.index(match) if match in free else 0)
        for number in dropped:
            self.run_path(number).unlink(missing_ok=True)
        staged = []
        for old, new in mapping.items():
            if self.run_path(old).exists():
                temporary = self.directory / f".regrid-{old:07d}.json"
                self.run_path(old).rename(temporary)
                staged.append((temporary, new))
        for temporary, new in staged:
            record = _read_json(temporary)
            if 'run_number' in record:
                record['run_number'] = new
            if isinstance(record.get('params'), dict):
                record['params'] = {**record['params'], **manifest.runs[new]}
            _write_json(self.run_path(new), record)
            temporary.unlink()
        (self.directory / SUCCESS_MARKER).unlink(missing_ok=True)
        return Manifest(
            kind=manifest.kind,
            name=manifest.name,
            config=manifest.config,
            config_hash=manifest.config_hash,
            seed=manifest.seed,
            runs=manifest.runs,
            metadata=stored.metadata,
            history=stored.history,
        )

    def write_run(self, run_number: int, record: Mapping[str, Any]) -> Path:
        path = self.run_path(run_number)
        _write_json(path, record)
        return path

    def read_run(self, run_number: int) -> Dict[str, Any]:
        return _read_json(self.run_path(run_number))

    def iter_runs(self) -> Iterable[Dict[str, Any]]:
        """Yield every stored run record in run-number order."""
        for path in sorted(self.directory.glob('[0-9]*.json')):
            yield _read_json(path)

    def finalize(self, completed: int, planned: int, empty: Sequence[str] = (), infeasible: Sequence[str] = ()) -> None:
        """
        Write the SUCCESS or WARNING marker describing how the invocation ended.

        A run that recorded no measurements downgrades the marker to WARNING and is named in it, except
        a resource-infeasible run, which SUCCESS names instead.
        """
        empty = [name for name in empty if name not in set(infeasible)]
        infeasible = list(infeasible)
        note = (
            f"{len(infeasible)} run(s) were resource-infeasible on this machine and are reported "
            f"as N/A: {', '.join(infeasible)}.\n"
            if infeasible
            else ''
        )
        if completed >= planned and not empty:
            message = f"SUCCESS: all {planned} benchmark run(s) completed.\n{note}"
            (self.directory / SUCCESS_MARKER).write_text(message, encoding='utf-8')
        elif completed >= planned:
            message = (
                f"WARNING: all {planned} benchmark run(s) executed, but {len(empty)} recorded no "
                f"measurements at all: {', '.join(empty)}.\nSee each run's `scene_errors` for why.\n{note}"
            )
            (self.directory / WARNING_MARKER).write_text(message, encoding='utf-8')
        else:
            message = f"WARNING: benchmark stopped after {completed}/{planned} run(s).\n{note}"
            (self.directory / WARNING_MARKER).write_text(message, encoding='utf-8')

    def record_failure(self, run_number: int, params: Mapping[str, Any], planned: int, error: BaseException) -> None:
        """Append a failure report with the traceback."""
        lines = [
            f"FAILURE: the benchmark failed at run {run_number}/{planned}.",
            f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"Error: {type(error).__name__}: {error}",
            '',
            'Traceback:',
            traceback.format_exc(),
            'Run parameters:',
            *[f"  {k:<28} {v}" for k, v in params.items()],
            '',
        ]
        with open(self.directory / FAILURE_MARKER, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays, which metric functions return throughout."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            # NaN is written as the literal NaN, which json.load reads back.
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def grid_change_error(stored: Mapping[str, Any], current: Mapping[str, Any]) -> Optional[str]:
    """Why `current` cannot take over the runs of `stored`, or None: they must be equal outside `pipeline`."""
    differing = sorted(
        key for key in set(stored) | set(current) if key != 'pipeline' and stored.get(key) != current.get(key)
    )
    if differing:
        return f"settings outside the pipeline grid differ ({', '.join(differing)})"
    return None


def _canonical(params: Mapping[str, Any]) -> str:
    """A run's parameters without its label, which numbers the grid and so moves when it grows."""
    return json.dumps({k: v for k, v in params.items() if k != 'pipeline'}, sort_keys=True, default=str)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, cls=_NumpyEncoder)
        tmp.replace(path)  # atomic: an interrupted write never leaves a half-parsed result
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ResultError(f"Result file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ResultError(f"Corrupt result file {path}: {e}") from e


def infeasible_runs(store: ResultStore) -> List[Dict[str, Any]]:
    """
    The runs a machine could not hold, in run-number order.

    They carry no measurements, so `to_dataframe` does not see them; the tables read them from here.
    """
    return [
        {
            'run_number': run.get('run_number'),
            'params': run.get('params', {}),
            'resource_error': run.get('resource_error', {}),
        }
        for run in store.iter_runs()
        if run.get('status') == RUN_INFEASIBLE
    ]


def to_dataframe(store: ResultStore) -> Any:
    """
    Flatten a result directory into a DataFrame, one row per measurement.

    Each run's `params` are joined onto each of its `records`.
    """
    rows: List[Dict[str, Any]] = []
    for run in store.iter_runs():
        params = run.get('params', {})
        run_number = run.get('run_number')
        for record in run.get('records', []):
            rows.append({'run_number': run_number, **params, **record})
    return pd.DataFrame(rows)
