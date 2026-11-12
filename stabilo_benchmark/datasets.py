#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
datasets.py - The dataset registry: what the benchmark reads, and where to get it.

`data/registry.yaml` lists each dataset's DOI, Zenodo record, license, and files with their
sizes and checksums. This module loads and validates it; `runners/fetch.py` acts on it.

An entry pins one version of a record: each Zenodo version has its own record id and DOI, and the
files' sizes and checksums are that version's. Moving to a newer version means changing `version`,
`doi`, `record` and `files` together.

A file is verified against the checksum pinned in the registry, or, when none is pinned, against
the checksum Zenodo reports. An archive may pin `extracted_size`, the bytes its entries take once
extracted, which the free-space check counts. At most one entry is marked `default: true`: the
dataset `fetch` uses when none is named. Unknown keys raise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

REGISTRY_FILENAME = 'registry.yaml'
RECEIPT_FILENAME = 'FETCHED.json'
SUPPORTED_VERSIONS = (1,)
ZENODO_API = 'https://zenodo.org/api/records'

_FILE_KEYS = {'name', 'size', 'md5', 'extract', 'extracted_size', 'optional', 'note'}
_DATASET_KEYS = {
    'title',
    'doi',
    'concept_doi',
    'version',
    'record',
    'published',
    'default',
    'license',
    'citation',
    'dest',
    'used_for',
    'files',
}


class RegistryError(ValueError):
    """Raised when the registry is malformed, or names a dataset that does not exist."""


@dataclass(frozen=True)
class FileSpec:
    """One file of one dataset."""

    name: str
    size: Optional[int] = None
    md5: Optional[str] = None
    extract: bool = False
    extracted_size: Optional[int] = None
    optional: bool = False
    note: Optional[str] = None

    @property
    def pinned(self) -> bool:
        """Whether a checksum is recorded, and so whether a fetch can be pinned to it."""
        return bool(self.md5)


@dataclass(frozen=True)
class DatasetSpec:
    """One registered dataset."""

    name: str
    title: str
    dest: str
    doi: Optional[str] = None
    concept_doi: Optional[str] = None
    version: Optional[str] = None
    record: Optional[int] = None
    published: bool = True
    default: bool = False
    license: Optional[str] = None
    citation: Optional[str] = None
    used_for: Optional[str] = None
    files: Sequence[FileSpec] = field(default_factory=tuple)

    @property
    def api_url(self) -> str:
        """The Zenodo API endpoint for this record."""
        if self.record is None:
            raise RegistryError(
                f"Dataset '{self.name}' has no Zenodo record id, so it cannot be fetched. "
                f"Set 'record:' to the numeric id of the published record."
            )
        return f"{ZENODO_API}/{int(self.record)}"

    def selected_files(self, include_optional: bool = False) -> List[FileSpec]:
        """The files a plain fetch should get; optional ones only when asked for."""
        return [f for f in self.files if include_optional or not f.optional]

    def resolve_dest(self, root: Path) -> Path:
        """Where this dataset lands, given the directory holding the registry."""
        return (Path(root) / self.dest).resolve()


def find_registry(start: Optional[Path] = None) -> Path:
    """
    Locate `data/registry.yaml`, searching from the working directory upwards, then beside the
    installed package.
    """
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / 'data' / REGISTRY_FILENAME
        if candidate.is_file():
            return candidate

    beside_package = Path(__file__).resolve().parent.parent / 'data' / REGISTRY_FILENAME
    if beside_package.is_file():
        return beside_package

    raise RegistryError(
        f"Could not find data/{REGISTRY_FILENAME} in {here} or any parent directory. "
        f"Run this from inside the stabilo-benchmark repository, or pass --registry."
    )


def load_registry(path: Optional[Path] = None) -> Dict[str, DatasetSpec]:
    """Read and validate the registry, returning dataset name -> spec."""
    path = Path(path) if path is not None else find_registry()
    if not path.is_file():
        raise RegistryError(f"Registry not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise RegistryError(f"{path}: the registry must be a mapping, found {type(raw).__name__}")

    version = raw.get('version')
    if version not in SUPPORTED_VERSIONS:
        raise RegistryError(
            f"{path}: unsupported registry 'version' {version!r}; this build understands {list(SUPPORTED_VERSIONS)}"
        )

    datasets = raw.get('datasets')
    if not isinstance(datasets, Mapping) or not datasets:
        raise RegistryError(f"{path}: 'datasets' must be a non-empty mapping of name -> dataset")

    registry = {name: _parse_dataset(name, entry, path) for name, entry in datasets.items()}
    defaults = [name for name, spec in registry.items() if spec.default]
    if len(defaults) > 1:
        raise RegistryError(f"{path}: {defaults} are all marked 'default: true'; mark at most one")
    return registry


def _parse_dataset(name: str, entry: Any, path: Path) -> DatasetSpec:
    """Validate one dataset entry, naming the offending key on failure."""
    if not isinstance(entry, Mapping):
        raise RegistryError(f"{path}: dataset '{name}' must be a mapping, found {type(entry).__name__}")

    unknown = set(entry) - _DATASET_KEYS
    if unknown:
        raise RegistryError(
            f"{path}: dataset '{name}' has unknown key(s) {sorted(unknown)}. Valid keys are {sorted(_DATASET_KEYS)}."
        )
    for required in ('title', 'dest'):
        if not entry.get(required):
            raise RegistryError(f"{path}: dataset '{name}' is missing required key '{required}'")

    dest = str(entry['dest'])
    if _escapes_data_dir(dest):
        raise RegistryError(
            f"{path}: dataset '{name}' has dest {dest!r}. It must be a relative path inside "
            f"data/, since that is the directory the ignore rules cover."
        )

    record = entry.get('record')
    if record is not None and not isinstance(record, int):
        raise RegistryError(f"{path}: dataset '{name}' has non-integer 'record' {record!r}")

    return DatasetSpec(
        name=name,
        title=str(entry['title']),
        dest=dest,
        doi=_optional_str(entry.get('doi')),
        concept_doi=_optional_str(entry.get('concept_doi')),
        version=_optional_str(entry.get('version')),
        record=record,
        published=bool(entry.get('published', True)),
        default=bool(entry.get('default', False)),
        license=_optional_str(entry.get('license')),
        citation=_optional_str(entry.get('citation')),
        used_for=_optional_str(entry.get('used_for')),
        files=tuple(_parse_file(name, item, path) for item in (entry.get('files') or [])),
    )


def _escapes_data_dir(dest: str) -> bool:
    r"""
    Whether a registry `dest` could write outside data/.

    Checked under both POSIX and Windows path flavors, since each misses the other's escapes:

      '/var/tmp/x'    rooted on Windows, where `is_absolute()` is False.
      'C:/x', 'C:x'   drive paths on Windows; a directory named 'C:' on POSIX.
      r'\\server\s\x' a UNC root on Windows; a plain name on POSIX.
      r'..\..\x'      traversal on Windows; one filename on POSIX.

    `anchor` is non-empty for every rooted, drive and UNC form, including 'C:x'.
    """
    flavours = (PurePosixPath(dest), PureWindowsPath(dest))
    return any(candidate.anchor or '..' in candidate.parts for candidate in flavours)


def _parse_file(dataset: str, entry: Any, path: Path) -> FileSpec:
    """Validate one file entry of one dataset."""
    if not isinstance(entry, Mapping):
        raise RegistryError(
            f"{path}: every item of '{dataset}.files' must be a mapping with a 'name', found {type(entry).__name__}"
        )
    unknown = set(entry) - _FILE_KEYS
    if unknown:
        raise RegistryError(
            f"{path}: file entry in '{dataset}' has unknown key(s) {sorted(unknown)}. "
            f"Valid keys are {sorted(_FILE_KEYS)}."
        )
    if not entry.get('name'):
        raise RegistryError(f"{path}: a file entry in '{dataset}' is missing 'name'")

    for key in ('size', 'extracted_size'):
        value = entry.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise RegistryError(f"{path}: '{dataset}/{entry['name']}' has invalid '{key}' {value!r}")
    extract = bool(entry.get('extract', False))
    if entry.get('extracted_size') is not None and not extract:
        raise RegistryError(
            f"{path}: '{dataset}/{entry['name']}' has 'extracted_size' but is not marked 'extract: true'"
        )

    return FileSpec(
        name=str(entry['name']),
        size=entry.get('size'),
        md5=normalize_digest(entry.get('md5')),
        extract=extract,
        extracted_size=entry.get('extracted_size'),
        optional=bool(entry.get('optional', False)),
        note=_optional_str(entry.get('note')),
    )


def _optional_str(value: Any) -> Optional[str]:
    """Trim a free-text field, mapping blank to None (YAML folds `>` blocks to a newline)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_digest(value: Any) -> Optional[str]:
    """A checksum as lowercase hex, accepting Zenodo's `md5:<hex>` form."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if ':' in text:
        text = text.split(':', 1)[1]
    return text or None


def select_datasets(registry: Mapping[str, DatasetSpec], names: Optional[Sequence[str]] = None) -> List[DatasetSpec]:
    """Look up datasets by name, or return the default one when none is named."""
    if not names:
        default = [spec for spec in registry.values() if spec.default]
        if not default:
            raise RegistryError(
                f"No dataset named and none marked 'default: true'. Registered: {sorted(registry)}. "
                f"List them with: stabilo-benchmark fetch --list"
            )
        return default
    missing = [name for name in names if name not in registry]
    if missing:
        raise RegistryError(
            f"Unknown dataset(s) {missing}. Registered: {sorted(registry)}. "
            f"List them with: stabilo-benchmark fetch --list"
        )
    return [registry[name] for name in names]


def file_digest(path: Path, algorithm: str = 'md5', chunk_size: int = 1 << 20) -> str:
    """Hex digest of a file, read in chunks so a 12 GB archive does not enter memory."""
    digest = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: FileSpec, expected_md5: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check one downloaded file against its registered size and checksum.

    `expected_md5` is the server's digest, used when the registry pins none. Returns (ok, reason).
    """
    path = Path(path)
    if not path.exists():
        return False, 'missing'

    actual_size = path.stat().st_size
    if spec.size is not None and actual_size != spec.size:
        return False, f"size {actual_size} != expected {spec.size}"

    reference = spec.md5 or normalize_digest(expected_md5)
    if reference is None:
        return True, f"present ({actual_size} bytes; no checksum to verify against)"

    actual = file_digest(path, 'md5')
    if actual != reference:
        return False, f"md5 {actual} != expected {reference}"
    return True, 'verified' if spec.md5 else 'verified against the record (unpinned in registry)'
