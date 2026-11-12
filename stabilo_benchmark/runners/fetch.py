#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
fetch.py - Download the registered datasets from Zenodo into `data/`.

    stabilo-benchmark fetch --list           what is registered, and what is already here
    stabilo-benchmark fetch songdo           download, verify, extract into data/songdo/
    stabilo-benchmark fetch                  the same for the dataset marked `default: true`
    stabilo-benchmark fetch --all            include the large optional files
    stabilo-benchmark fetch --check          re-verify what is on disk, download nothing

- Before anything is downloaded, the total size and the free space on each destination volume
  are printed; a transfer that would not fit is refused, and one over `LARGE_DOWNLOAD` bytes asks
  for confirmation on a terminal (`--yes` skips it).
- Partial downloads are kept as `<name>.part` and resumed with a Range request; verified files
  are skipped.
- Size and MD5 are checked before extraction, against the checksum pinned in `registry.yaml`
  or, when none is pinned, the record's own.
- `FETCHED.json` records the record, the files, their digests, which checksum applied, and when
  they arrived.
- The registry pins one version of each record. A destination holding another version, by the DOI
  its `FETCHED.json` records, is refused rather than extracted over, so no file of an earlier
  version survives among a later one's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from tqdm import tqdm

from stabilo_benchmark.datasets import (
    RECEIPT_FILENAME,
    DatasetSpec,
    FileSpec,
    RegistryError,
    file_digest,
    load_registry,
    normalize_digest,
    select_datasets,
    verify_file,
)

USER_AGENT = 'stabilo-benchmark/fetch'
TIMEOUT_S = 60
CHUNK = 1 << 20
# Downloads at least this large ask for confirmation on a terminal.
LARGE_DOWNLOAD = 1_000_000_000

GREEN, YELLOW, RED, DIM, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[2m', '\033[0m'


class FetchError(RuntimeError):
    """Raised when a dataset cannot be fetched or fails verification."""


# --------------------------------------------------------------------------------------
# Network access, isolated in two functions
# --------------------------------------------------------------------------------------


def fetch_record(url: str) -> Dict[str, Any]:
    """Read a Zenodo record's metadata as JSON."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - registry URL
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FetchError(
                f"Zenodo returned 404 for {url}. Either the record id is wrong, or the record is "
                f"reserved but not yet published."
            ) from e
        raise FetchError(f"Zenodo returned HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"Could not reach Zenodo ({url}): {e.reason}") from e


def download_file(url: str, target: Path, expected_size: Optional[int] = None, label: str = '') -> Path:
    """
    Download `url` to `target`, resuming a previous partial transfer when possible.

    The bytes go to `<target>.part`, renamed into place once complete.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')

    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and offset > expected_size:
        # A stale part file from a different version of the record; start over.
        partial.unlink()
        offset = 0

    headers = {'User-Agent': USER_AGENT}
    if offset:
        headers['Range'] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - registry URL
            # A server that ignores Range answers 200 with the whole file.
            if offset and response.status != 206:
                offset = 0
            mode = 'ab' if offset else 'wb'
            total = expected_size
            if total is None:
                length = response.headers.get('Content-Length')
                total = (int(length) + offset) if length else None

            with (
                open(partial, mode) as out,
                tqdm(
                    total=total,
                    initial=offset,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label or target.name,
                    leave=False,
                ) as progress,
            ):
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    progress.update(len(chunk))
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} downloading {url}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"Transfer failed for {url}: {e.reason}") from e

    partial.replace(target)
    return target


# --------------------------------------------------------------------------------------
# Planning, extraction, receipts
# --------------------------------------------------------------------------------------


def record_files(record: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map filename -> {url, size, md5} from a Zenodo record's file list."""
    files: Dict[str, Dict[str, Any]] = {}
    for entry in record.get('files') or []:
        key = entry.get('key')
        if not key:
            continue
        links = entry.get('links') or {}
        files[str(key)] = {
            'url': links.get('self') or links.get('download'),
            'size': entry.get('size'),
            'md5': normalize_digest(entry.get('checksum')),
        }
    return files


def plan_dataset(
    spec: DatasetSpec, available: Mapping[str, Dict[str, Any]], include_optional: bool = False
) -> List[Tuple[FileSpec, Dict[str, Any]]]:
    """
    Pair each wanted file with its entry in the record, raising when one is absent (usually a
    superseded record).
    """
    wanted = spec.selected_files(include_optional=include_optional)
    if not wanted:
        raise FetchError(
            f"Dataset '{spec.name}' lists no files to fetch. Fill in its 'files:' from the published record."
        )

    missing = [f.name for f in wanted if f.name not in available]
    if missing:
        raise FetchError(
            f"Record {spec.record} does not contain {missing}. It has {sorted(available)}. "
            f"If the record was superseded, update 'record' and the pinned checksums together."
        )
    return [(f, available[f.name]) for f in wanted]


def download_size(
    spec: DatasetSpec, root: Path, plan: List[Tuple[FileSpec, Dict[str, Any]]], force: bool = False
) -> Tuple[int, int]:
    """
    Bytes still to download for `plan`, and bytes its archives will take once extracted.

    A file already on disk at its expected size counts as present (it is verified by checksum
    when the fetch reaches it); a partial download counts only its missing part. Extraction counts
    the archive's pinned `extracted_size`, or, when none is pinned, the archive's own size, which
    holds for already-compressed images and underestimates compressible content.
    """
    dest = spec.resolve_dest(root)
    download = extracted = 0
    for file_spec, remote in plan:
        size = file_spec.size if file_spec.size is not None else remote.get('size')
        if size is None:
            continue
        target = dest / file_spec.name
        present = target.is_file() and target.stat().st_size == size
        if file_spec.extract and (force or not present):
            extracted += file_spec.extracted_size if file_spec.extracted_size is not None else size
        if present and not force:
            continue
        partial = target.with_suffix(target.suffix + '.part')
        done = partial.stat().st_size if partial.is_file() else 0
        download += max(size - done, 0)
    return download, extracted


def free_space(path: Path) -> Tuple[int, int]:
    """(device id, free bytes) of the volume `path` would be created on."""
    probe = Path(os.path.realpath(path))
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe.stat().st_dev, shutil.disk_usage(probe).free


def confirm_download(
    needs: List[Tuple[Path, int, int]],
    assume_yes: bool = False,
    interactive: Optional[bool] = None,
    ask: Callable[[str], str] = input,
) -> int:
    """
    Print what is about to be downloaded and where, refuse a transfer that does not fit, and ask
    before a large one when a person is at the terminal. `needs` holds (destination, bytes to
    download, bytes to extract). Returns the total download size.
    """
    volumes: Dict[int, Dict[str, Any]] = {}
    for dest, download, extracted in needs:
        device, free = free_space(dest)
        volume = volumes.setdefault(device, {'free': free, 'need': 0, 'dests': []})
        volume['need'] += download + extracted
        volume['dests'].append(dest)
    total = sum(download for _, download, _ in needs)
    extracted = sum(e for _, _, e in needs)
    if total == 0 and extracted == 0:
        return 0

    print(
        f"{YELLOW}About to download {_gb(total)}"
        f"{f', and extract about {_gb(extracted)} more' if extracted else ''}:{RESET}"
    )
    for dest, download, extract in needs:
        if download or extract:
            print(f"  {_gb(download):>9}  -> {dest}")
    short = []
    for volume in volumes.values():
        dests = ', '.join(str(d) for d in volume['dests'])
        print(f"  {_gb(volume['free']):>9}  free on the volume holding {dests}")
        if volume['need'] > volume['free']:
            short.append(f"{dests} needs {_gb(volume['need'])} and has {_gb(volume['free'])} free")
    if short:
        raise FetchError('Not enough disk space: ' + '; '.join(short) + '.')

    if interactive is None:
        interactive = sys.stdin.isatty()
    if total >= LARGE_DOWNLOAD and interactive and not assume_yes:
        try:
            answer = ask('Continue? [y/N] ')
        except EOFError:
            answer = ''
        if answer.strip().lower() not in ('y', 'yes'):
            raise FetchError('Download cancelled.')
    return total


def _gb(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 10_000_000 else f"{size / 1e6:.2f} MB"


def extract_archive(archive: Path, dest: Path) -> List[str]:
    """Extract a zip or tar archive into `dest`, refusing entries that escape it."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()

    def safe(name: str) -> bool:
        candidate = (resolved_dest / name).resolve()
        return candidate == resolved_dest or resolved_dest in candidate.parents

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            unsafe = [n for n in names if not safe(n)]
            if unsafe:
                raise FetchError(f"{archive.name} contains entries that escape {dest}: {unsafe[:3]}")
            zf.extractall(dest)  # noqa: S202 - every member was just checked against dest
        return names

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            names = tf.getnames()
            unsafe = [n for n in names if not safe(n)]
            if unsafe:
                raise FetchError(f"{archive.name} contains entries that escape {dest}: {unsafe[:3]}")
            tf.extractall(dest)  # noqa: S202 - every member was just checked against dest
        return names

    raise FetchError(f"{archive.name} is marked 'extract: true' but is neither a zip nor a tar archive")


def write_receipt(dest: Path, spec: DatasetSpec, record: Mapping[str, Any], entries: List[Dict[str, Any]]) -> Path:
    """Record what was fetched, from where, and with which digests."""
    receipt = {
        'fetched': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dataset': spec.name,
        'title': spec.title,
        'doi': spec.doi,
        'version': spec.version,
        'record': spec.record,
        'record_doi': record.get('doi'),
        'license': spec.license,
        'citation': spec.citation,
        'files': entries,
    }
    path = Path(dest) / RECEIPT_FILENAME
    path.write_text(json.dumps(receipt, indent=2, default=str), encoding='utf-8')
    return path


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def list_datasets(registry: Mapping[str, DatasetSpec], root: Path) -> None:
    """Print each registered dataset, its license, and whether it is already on disk."""
    print(f"{GREEN}Registered datasets (root: {root}){RESET}")
    for spec in registry.values():
        dest = spec.resolve_dest(root)
        receipt = dest / RECEIPT_FILENAME
        if not spec.published:
            state = f"{YELLOW}not yet published{RESET}"
        elif receipt.is_file():
            state = f"{GREEN}fetched{RESET}"
        elif dest.is_dir():
            state = f"{YELLOW}present, no receipt{RESET}"
        else:
            state = f"{DIM}not fetched{RESET}"

        pinned = sum(1 for f in spec.files if f.pinned)
        optional = sum(1 for f in spec.files if f.optional)
        print(f"\n  {spec.name}{' (default)' if spec.default else ''}  [{state}]")
        print(f"    {spec.title}")
        if spec.doi:
            version = f" ({spec.version})" if spec.version else ''
            print(f"    doi: {spec.doi}{version}   license: {spec.license or 'unspecified'}")
        print(f"    dest: {dest}")
        held = other_version(spec, dest)
        if held:
            print(f"    {RED}{held}{RESET}")
        if spec.files:
            print(
                f"    files: {len(spec.files)} ({pinned} with pinned checksums"
                f"{f', {optional} optional' if optional else ''})"
            )
        if spec.used_for:
            print(f"    {DIM}{spec.used_for}{RESET}")


def _receipt(dest: Path) -> Dict[str, Any]:
    """The previous fetch's `FETCHED.json`, or empty if there is none or it cannot be read."""
    path = Path(dest) / RECEIPT_FILENAME
    if not path.is_file():
        return {}
    try:
        receipt = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def read_receipt(dest: Path) -> Dict[str, Dict[str, Any]]:
    """The previous fetch's receipt as filename -> entry, or empty if there is none."""
    return {str(entry.get('name')): entry for entry in _receipt(dest).get('files') or []}


def other_version(spec: DatasetSpec, dest: Path) -> Optional[str]:
    """
    Why `dest` holds a version of the record other than the one the registry pins, or None.

    Read from the DOI in `FETCHED.json`; a destination without a receipt is not judged.
    """
    receipt = _receipt(dest)
    held = receipt.get('doi')
    if not held or not spec.doi or held == spec.doi:
        return None
    version = f" ({receipt['version']})" if receipt.get('version') else ''
    wanted = f" ({spec.version})" if spec.version else ''
    return f"{dest} holds DOI {held}{version}; the registry pins DOI {spec.doi}{wanted}"


def check_dataset(spec: DatasetSpec, root: Path, include_optional: bool = False) -> bool:
    """
    Re-verify a dataset already on disk, without touching the network.

    Files the registry does not pin are checked against the digest in `FETCHED.json`. A missing file
    is reported but is not a failure, since optional files may not have been fetched.
    """
    dest = spec.resolve_dest(root)
    wanted = spec.selected_files(include_optional=include_optional)
    if not dest.is_dir():
        print(f"  {spec.name}: {DIM}not fetched{RESET}")
        return True
    if not wanted:
        print(f"  {spec.name}: {YELLOW}no files registered to check{RESET}")
        return True

    held = other_version(spec, dest)
    if held:
        print(f"  {spec.name}: {RED}{held}{RESET}")
        return False

    receipt = read_receipt(dest)
    ok = True
    print(f"  {spec.name}:")
    for file_spec in wanted:
        recorded = (receipt.get(file_spec.name) or {}).get('md5')
        good, reason = verify_file(dest / file_spec.name, file_spec, expected_md5=recorded)
        if not good and reason == 'missing':
            print(f"    {DIM}--  {file_spec.name}: not fetched{RESET}")
            continue
        if good and not file_spec.pinned and recorded:
            reason = 'verified against FETCHED.json (unpinned in registry)'
        color = GREEN if good else RED
        print(f"    {color}{'ok ' if good else 'BAD'}{RESET} {file_spec.name}: {reason}")
        ok &= good
    return ok


def fetch_dataset(
    spec: DatasetSpec,
    root: Path,
    include_optional: bool = False,
    force: bool = False,
    record: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Download, verify, and extract one dataset. Returns its directory."""
    if not spec.published:
        raise FetchError(
            f"Dataset '{spec.name}' is registered but not yet published"
            f"{f' (reserved DOI {spec.doi})' if spec.doi else ''}. Nothing to fetch: fill in its "
            f"'record', 'files', and checksums once the record is public, and set "
            f"'published: true'."
        )

    dest = spec.resolve_dest(root)
    held = other_version(spec, dest)
    if held:
        raise FetchError(
            f"{held}. Move or delete {dest} first: extracting one version over another would leave "
            f"files the newer version no longer has."
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{GREEN}{spec.name}: {spec.title}{RESET}")
    print(f"  record {spec.record} -> {dest}")

    if record is None:
        record = fetch_record(spec.api_url)
    available = record_files(record)
    plan = plan_dataset(spec, available, include_optional=include_optional)

    entries: List[Dict[str, Any]] = []
    for file_spec, remote in plan:
        target = dest / file_spec.name
        good, reason = verify_file(target, file_spec, expected_md5=remote.get('md5'))
        if good and not force:
            print(f"  {DIM}skip{RESET} {file_spec.name}: {reason}")
        else:
            if not remote.get('url'):
                raise FetchError(f"Record {spec.record} gives no download link for {file_spec.name}")
            size = file_spec.size if file_spec.size is not None else remote.get('size')
            print(f"  get  {file_spec.name}{f' ({size / 1e9:.2f} GB)' if size and size > 1e9 else ''}")
            download_file(remote['url'], target, expected_size=size, label=file_spec.name)

            good, reason = verify_file(target, file_spec, expected_md5=remote.get('md5'))
            if not good:
                raise FetchError(
                    f"{file_spec.name} failed verification after download ({reason}). The file has "
                    f"been left in place for inspection; delete it and retry, or update the "
                    f"registry if the record has a new version."
                )
            print(f"  {GREEN}ok{RESET}   {file_spec.name}: {reason}")

        digest = file_digest(target, 'md5')
        entries.append(
            {
                'name': file_spec.name,
                'size': target.stat().st_size,
                'md5': digest,
                'pinned_in_registry': file_spec.pinned,
                'matched': 'registry' if file_spec.pinned else 'record',
            }
        )

        if file_spec.extract:
            print(f"  {DIM}extract{RESET} {file_spec.name}")
            names = extract_archive(target, dest)
            entries[-1]['extracted_entries'] = len(names)

    write_receipt(dest, spec, record, entries)
    unpinned = [e['name'] for e in entries if not e['pinned_in_registry']]
    if unpinned:
        print(
            f"  {DIM}note{RESET} {', '.join(unpinned)}: no checksum in the registry, so verified "
            f"against the one Zenodo reports; {RECEIPT_FILENAME} records their digests."
        )
    print(f"  {GREEN}done{RESET} receipt -> {dest / RECEIPT_FILENAME}")
    return dest


def run_fetch(args: argparse.Namespace) -> int:
    """Entry point for `stabilo-benchmark fetch`."""
    registry_path = Path(args.registry) if getattr(args, 'registry', None) else None
    registry = load_registry(registry_path)
    root = (registry_path or _registry_path_used()).parent

    if args.list:
        list_datasets(registry, root)
        return 0

    specs = select_datasets(registry, args.datasets)

    if args.check:
        print(f"{GREEN}Verifying datasets under {root}{RESET}")
        ok = all(check_dataset(spec, root, include_optional=True) for spec in specs)
        if not ok:
            print(f"{RED}Some files failed verification.{RESET}")
        return 0 if ok else 1

    pending = [spec for spec in specs if not spec.published]
    fetchable = [spec for spec in specs if spec.published]
    if not fetchable:
        names = ', '.join(spec.name for spec in pending)
        raise FetchError(
            f"Nothing to fetch: {names} {'is' if len(pending) == 1 else 'are'} registered but not "
            f"yet published. See data/registry.yaml."
        )

    for spec in fetchable:
        held = other_version(spec, spec.resolve_dest(root))
        if held:
            raise FetchError(f"{held}. Move or delete it first, then fetch again.")
    records = {spec.name: fetch_record(spec.api_url) for spec in fetchable}
    needs = []
    for spec in fetchable:
        plan = plan_dataset(spec, record_files(records[spec.name]), include_optional=args.all)
        needs.append((spec.resolve_dest(root), *download_size(spec, root, plan, force=args.force)))
    confirm_download(needs, assume_yes=getattr(args, 'yes', False))

    for spec in fetchable:
        fetch_dataset(spec, root, include_optional=args.all, force=args.force, record=records[spec.name])
    for spec in pending:
        print(f"{YELLOW}skipped {spec.name}: registered but not yet published{RESET}")

    if shutil.disk_usage(root).free < (1 << 30):
        print(f"{YELLOW}Warning: under 1 GB free on the volume holding {root}.{RESET}")
    return 0


def _registry_path_used() -> Path:
    """The registry `load_registry` would have found, for reporting its root directory."""
    from stabilo_benchmark.datasets import find_registry  # noqa: PLC0415 - avoids a cycle at import

    return find_registry()


__all__ = [
    'FetchError',
    'RegistryError',
    'check_dataset',
    'confirm_download',
    'download_size',
    'extract_archive',
    'fetch_dataset',
    'list_datasets',
    'other_version',
    'plan_dataset',
    'record_files',
    'run_fetch',
]
