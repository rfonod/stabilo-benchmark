#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""The dataset registry, and every part of `fetch` that does not touch the network."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from stabilo_benchmark.datasets import (
    RECEIPT_FILENAME,
    FileSpec,
    RegistryError,
    file_digest,
    find_registry,
    load_registry,
    normalize_digest,
    select_datasets,
    verify_file,
)
from stabilo_benchmark.runners import fetch
from stabilo_benchmark.runners.fetch import (
    FetchError,
    confirm_download,
    download_size,
    extract_archive,
    plan_dataset,
    read_receipt,
    record_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MINIMAL = {
    'version': 1,
    'datasets': {
        'example': {
            'title': 'An example dataset',
            'doi': '10.5281/zenodo.1',
            'record': 1,
            'license': 'CC-BY-4.0',
            'dest': 'zenodo/example',
            'files': [
                {'name': 'small.txt', 'size': 4, 'md5': 'abc123'},
                {'name': 'big.zip', 'size': 99, 'extract': True, 'optional': True},
            ],
        }
    },
}


def write_registry(tmp_path: Path, data) -> Path:
    path = tmp_path / 'registry.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


class TestTheShippedRegistry:
    def test_the_repository_registry_loads_and_validates(self):
        registry = load_registry(REPO_ROOT / 'data' / 'registry.yaml')
        assert [spec.name for spec in select_datasets(registry, None)] == ['songdo']

    def test_every_dataset_records_a_license_and_what_it_is_for(self):
        for name, spec in load_registry(REPO_ROOT / 'data' / 'registry.yaml').items():
            assert spec.license, f"{name} has no license; sample and full data are both redistributed"
            assert spec.used_for, f"{name} does not say what the benchmark needs it for"
            assert spec.version and spec.doi, f"{name} does not pin the version of its record"

    def test_every_extracted_archive_pins_its_extracted_size(self):
        # The free-space check counts it; an archive's own size understates compressible content.
        for name, spec in load_registry(REPO_ROOT / 'data' / 'registry.yaml').items():
            for f in spec.files:
                if f.extract:
                    assert f.extracted_size is not None, f"{name}/{f.name} pins no extracted_size"
                    assert f.extracted_size >= (f.size or 0) * 0.5, f"{name}/{f.name}: implausible extracted_size"

    def test_the_record_id_is_the_suffix_of_its_doi(self):
        songdo = load_registry(REPO_ROOT / 'data' / 'registry.yaml')['songdo']
        assert songdo.published is True
        assert songdo.doi == f'10.5281/zenodo.{songdo.record}'

    def test_an_unpublished_dataset_is_reported_rather_than_fetched(self, tmp_path):
        entry = {'title': 't', 'dest': 'd', 'doi': '10.5281/zenodo.1', 'published': False}
        data = {'version': 1, 'datasets': {'e': entry}}
        spec = load_registry(write_registry(tmp_path, data))['e']
        with pytest.raises(fetch.FetchError, match='not yet published'):
            fetch.fetch_dataset(spec, tmp_path)

    def test_found_by_searching_upwards_from_a_subdirectory(self):
        assert find_registry(REPO_ROOT / 'tests') == REPO_ROOT / 'data' / 'registry.yaml'


class TestRegistryValidation:
    def test_unknown_dataset_key_names_the_key(self, tmp_path):
        data = {'version': 1, 'datasets': {'example': {'title': 't', 'dest': 'd', 'licence': 'CC0'}}}
        with pytest.raises(RegistryError, match=r"unknown key\(s\) \['licence'\]"):
            load_registry(write_registry(tmp_path, data))

    def test_unknown_file_key_names_the_key(self, tmp_path):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd', 'files': [{'name': 'a', 'sha256': 'x'}]}}}
        with pytest.raises(RegistryError, match=r"unknown key\(s\) \['sha256'\]"):
            load_registry(write_registry(tmp_path, data))

    def test_missing_required_key_is_named(self, tmp_path):
        data = {'version': 1, 'datasets': {'e': {'title': 'only a title'}}}
        with pytest.raises(RegistryError, match="missing required key 'dest'"):
            load_registry(write_registry(tmp_path, data))

    # dest must stay inside data/. Every form is rejected under both path flavors, whatever the host.
    @pytest.mark.parametrize(
        'dest',
        [
            '/var/tmp/somewhere',  # POSIX absolute; on Windows, rooted but not "absolute"
            'C:/somewhere',  # Windows absolute; on POSIX just a directory named 'C:'
            'C:somewhere',  # Windows drive-relative; is_absolute() is False even there
            r'\\server\share\somewhere',  # UNC root on Windows
            '../../elsewhere',  # traversal, either flavor
            r'..\..\elsewhere',  # traversal Windows-style; one filename to POSIX
        ],
    )
    def test_dest_escaping_the_data_directory_is_rejected(self, tmp_path, dest):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': dest}}}
        with pytest.raises(RegistryError, match='must be a relative path inside'):
            load_registry(write_registry(tmp_path, data))

    @pytest.mark.parametrize('dest', ['songdo', 'sample', 'a/b/c', 'name.with.dots'])
    def test_ordinary_relative_dest_is_accepted(self, tmp_path, dest):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': dest}}}
        assert load_registry(write_registry(tmp_path, data))['e'].dest == dest

    def test_unsupported_version_is_rejected(self, tmp_path):
        with pytest.raises(RegistryError, match="unsupported registry 'version'"):
            load_registry(write_registry(tmp_path, {'version': 99, 'datasets': MINIMAL['datasets']}))

    def test_empty_datasets_is_rejected(self, tmp_path):
        with pytest.raises(RegistryError, match='non-empty mapping'):
            load_registry(write_registry(tmp_path, {'version': 1, 'datasets': {}}))

    def test_non_integer_record_is_rejected(self, tmp_path):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd', 'record': '13828408'}}}
        with pytest.raises(RegistryError, match='non-integer'):
            load_registry(write_registry(tmp_path, data))

    @pytest.mark.parametrize('value', [-1, '8.7 GB', 1.5, True])
    def test_invalid_extracted_size_is_named(self, tmp_path, value):
        files = [{'name': 'a.zip', 'size': 1, 'extract': True, 'extracted_size': value}]
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd', 'files': files}}}
        with pytest.raises(RegistryError, match="invalid 'extracted_size'"):
            load_registry(write_registry(tmp_path, data))

    def test_extracted_size_needs_an_extracted_file(self, tmp_path):
        files = [{'name': 'a.txt', 'size': 1, 'extracted_size': 1}]
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd', 'files': files}}}
        with pytest.raises(RegistryError, match="not marked 'extract: true'"):
            load_registry(write_registry(tmp_path, data))


class TestSelection:
    def test_optional_files_are_excluded_by_default(self, tmp_path):
        spec = load_registry(write_registry(tmp_path, MINIMAL))['example']
        assert [f.name for f in spec.selected_files()] == ['small.txt']
        assert [f.name for f in spec.selected_files(include_optional=True)] == ['small.txt', 'big.zip']

    def test_unknown_name_lists_what_is_registered(self, tmp_path):
        registry = load_registry(write_registry(tmp_path, MINIMAL))
        with pytest.raises(RegistryError, match=r"Registered: \['example'\]"):
            select_datasets(registry, ['nope'])

    @staticmethod
    def two_datasets(default_a=False, default_b=False):
        entry = MINIMAL['datasets']['example']
        return {
            'version': 1,
            'datasets': {
                'a': {**entry, 'dest': 'a', 'default': default_a},
                'b': {**entry, 'dest': 'b', 'default': default_b},
            },
        }

    def test_no_names_means_the_default(self, tmp_path):
        registry = load_registry(write_registry(tmp_path, self.two_datasets(default_b=True)))
        assert [spec.name for spec in select_datasets(registry, None)] == ['b']
        assert [spec.name for spec in select_datasets(registry, ['a'])] == ['a']

    def test_no_names_and_no_default_asks_for_a_name(self, tmp_path):
        registry = load_registry(write_registry(tmp_path, self.two_datasets()))
        with pytest.raises(RegistryError, match=r"none marked 'default: true'. Registered: \['a', 'b'\]"):
            select_datasets(registry, [])

    def test_two_defaults_raise(self, tmp_path):
        with pytest.raises(RegistryError, match='mark at most one'):
            load_registry(write_registry(tmp_path, self.two_datasets(default_a=True, default_b=True)))

    def test_a_record_less_dataset_cannot_produce_an_api_url(self, tmp_path):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd'}}}
        spec = load_registry(write_registry(tmp_path, data))['e']
        with pytest.raises(RegistryError, match='no Zenodo record id'):
            _ = spec.api_url


class TestDigests:
    def test_zenodo_algorithm_prefix_is_stripped(self):
        assert normalize_digest('md5:ABC123') == 'abc123'
        assert normalize_digest('abc123') == 'abc123'
        assert normalize_digest(None) is None

    def test_digest_matches_hashlib(self, tmp_path):
        path = tmp_path / 'f.bin'
        payload = b'stabilo' * 1000
        path.write_bytes(payload)
        assert file_digest(path) == hashlib.md5(payload).hexdigest()  # noqa: S324 - matching Zenodo


class TestVerifyFile:
    def test_missing_file(self, tmp_path):
        ok, reason = verify_file(tmp_path / 'absent.txt', FileSpec(name='absent.txt'))
        assert (ok, reason) == (False, 'missing')

    def test_size_mismatch_is_caught_before_hashing(self, tmp_path):
        path = tmp_path / 'f.txt'
        path.write_text('four', encoding='utf-8')
        ok, reason = verify_file(path, FileSpec(name='f.txt', size=999))
        assert not ok
        assert 'size 4 != expected 999' in reason

    def test_checksum_mismatch_is_reported_with_both_digests(self, tmp_path):
        path = tmp_path / 'f.txt'
        path.write_text('four', encoding='utf-8')
        ok, reason = verify_file(path, FileSpec(name='f.txt', md5='0' * 32))
        assert not ok
        assert file_digest(path) in reason

    def test_pinned_checksum_verifies(self, tmp_path):
        path = tmp_path / 'f.txt'
        path.write_text('four', encoding='utf-8')
        ok, reason = verify_file(path, FileSpec(name='f.txt', size=4, md5=file_digest(path)))
        assert ok and reason == 'verified'

    def test_unpinned_falls_back_to_the_records_checksum(self, tmp_path):
        # An unpinned entry is verified against the server's checksum.
        path = tmp_path / 'f.txt'
        path.write_text('four', encoding='utf-8')
        ok, reason = verify_file(path, FileSpec(name='f.txt'), expected_md5=f"md5:{file_digest(path)}")
        assert ok
        assert 'unpinned' in reason

    def test_nothing_to_verify_against_is_not_silently_a_pass(self, tmp_path):
        path = tmp_path / 'f.txt'
        path.write_text('four', encoding='utf-8')
        ok, reason = verify_file(path, FileSpec(name='f.txt'))
        assert ok
        assert 'no checksum' in reason


class TestRecordParsing:
    RECORD = {
        'files': [
            {
                'key': 'a.zip',
                'size': 10,
                'checksum': 'md5:aaa',
                'links': {'self': 'https://zenodo.org/api/records/1/files/a.zip/content'},
            },
            {'key': 'b.txt', 'size': 2, 'checksum': 'md5:bbb', 'links': {'download': 'https://x/b.txt'}},
            {'size': 1},  # no key: ignored rather than crashing the whole fetch
        ]
    }

    def test_files_are_keyed_by_name_with_url_size_and_digest(self):
        files = record_files(self.RECORD)
        assert set(files) == {'a.zip', 'b.txt'}
        assert files['a.zip']['md5'] == 'aaa'
        assert files['b.txt']['url'] == 'https://x/b.txt'

    def test_empty_record_yields_nothing(self):
        assert record_files({}) == {}

    def test_plan_pairs_registry_entries_with_record_entries(self, tmp_path):
        spec = load_registry(write_registry(tmp_path, MINIMAL))['example']
        available = {'small.txt': {'url': 'u', 'size': 4, 'md5': 'x'}}
        plan = plan_dataset(spec, available)
        assert [f.name for f, _ in plan] == ['small.txt']

    def test_plan_errors_when_the_record_lacks_a_registered_file(self, tmp_path):
        spec = load_registry(write_registry(tmp_path, MINIMAL))['example']
        with pytest.raises(FetchError, match='does not contain'):
            plan_dataset(spec, {'something_else.txt': {}})

    def test_plan_errors_when_no_files_are_registered(self, tmp_path):
        data = {'version': 1, 'datasets': {'e': {'title': 't', 'dest': 'd', 'record': 1}}}
        spec = load_registry(write_registry(tmp_path, data))['e']
        with pytest.raises(FetchError, match='lists no files'):
            plan_dataset(spec, {})


class TestExtraction:
    def test_a_normal_zip_extracts(self, tmp_path):
        archive = tmp_path / 'ok.zip'
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('inner/file.txt', 'hello')
        names = extract_archive(archive, tmp_path / 'out')
        assert names == ['inner/file.txt']
        assert (tmp_path / 'out' / 'inner' / 'file.txt').read_text(encoding='utf-8') == 'hello'

    def test_an_entry_escaping_the_destination_is_refused(self, tmp_path):
        # A path-traversal entry is refused before anything is extracted.
        archive = tmp_path / 'evil.zip'
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('../escaped.txt', 'nope')
        with pytest.raises(FetchError, match='escape'):
            extract_archive(archive, tmp_path / 'out')
        assert not (tmp_path / 'escaped.txt').exists()

    def test_a_non_archive_marked_for_extraction_is_an_error(self, tmp_path):
        plain = tmp_path / 'notanarchive.zip'
        plain.write_text('just text', encoding='utf-8')
        with pytest.raises(FetchError, match='neither a zip nor a tar'):
            extract_archive(plain, tmp_path / 'out')


class TestReceipts:
    def test_absent_receipt_reads_as_empty(self, tmp_path):
        assert read_receipt(tmp_path) == {}

    def test_corrupt_receipt_reads_as_empty_rather_than_raising(self, tmp_path):
        (tmp_path / RECEIPT_FILENAME).write_text('{not json', encoding='utf-8')
        assert read_receipt(tmp_path) == {}

    def test_receipt_is_keyed_by_filename(self, tmp_path):
        (tmp_path / RECEIPT_FILENAME).write_text(
            json.dumps({'files': [{'name': 'a.zip', 'md5': 'aaa'}]}), encoding='utf-8'
        )
        assert read_receipt(tmp_path)['a.zip']['md5'] == 'aaa'


class TestDownloadSize:
    @pytest.fixture
    def spec(self, tmp_path):
        return load_registry(write_registry(tmp_path, MINIMAL))['example']

    def plan(self, spec):
        return [(f, {'size': f.size}) for f in spec.selected_files(include_optional=True)]

    def test_nothing_on_disk_downloads_everything_and_extracts_the_archive(self, spec, tmp_path):
        assert download_size(spec, tmp_path, self.plan(spec)) == (4 + 99, 99)

    def test_a_file_at_its_size_is_present_unless_forced(self, spec, tmp_path):
        dest = spec.resolve_dest(tmp_path)
        dest.mkdir(parents=True)
        (dest / 'big.zip').write_bytes(b'x' * 99)
        assert download_size(spec, tmp_path, self.plan(spec)) == (4, 0)
        assert download_size(spec, tmp_path, self.plan(spec), force=True) == (103, 99)

    def test_a_pinned_extracted_size_replaces_the_archives_own(self, tmp_path):
        data = json.loads(json.dumps(MINIMAL))
        data['datasets']['example']['files'][1]['extracted_size'] = 250
        spec = load_registry(write_registry(tmp_path, data))['example']
        assert download_size(spec, tmp_path, self.plan(spec)) == (4 + 99, 250)

    def test_a_partial_download_counts_only_what_is_missing(self, spec, tmp_path):
        dest = spec.resolve_dest(tmp_path)
        dest.mkdir(parents=True)
        (dest / 'big.zip.part').write_bytes(b'x' * 90)
        assert download_size(spec, tmp_path, self.plan(spec))[0] == 4 + 9


class TestConfirmDownload:
    def test_nothing_to_download_asks_nothing(self, tmp_path):
        assert confirm_download([(tmp_path, 0, 0)], interactive=True, ask=self.refuse) == 0

    def test_a_large_download_asks_at_a_terminal(self, tmp_path, capsys):
        size = fetch.LARGE_DOWNLOAD
        with pytest.raises(FetchError, match='cancelled'):
            confirm_download([(tmp_path / 'new', size, 0)], interactive=True, ask=lambda _: 'n')
        assert '1.00 GB' in capsys.readouterr().out
        assert confirm_download([(tmp_path, size, 0)], interactive=True, ask=lambda _: 'y') == size

    def test_end_of_input_cancels(self, tmp_path):
        def eof(prompt):
            raise EOFError

        with pytest.raises(FetchError, match='cancelled'):
            confirm_download([(tmp_path, fetch.LARGE_DOWNLOAD, 0)], interactive=True, ask=eof)

    def test_yes_and_scripts_are_not_asked(self, tmp_path):
        size = fetch.LARGE_DOWNLOAD
        assert confirm_download([(tmp_path, size, 0)], assume_yes=True, interactive=True, ask=self.refuse) == size
        assert confirm_download([(tmp_path, size, 0)], interactive=False, ask=self.refuse) == size

    def test_a_download_that_does_not_fit_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch, 'free_space', lambda path: (1, 100))
        with pytest.raises(FetchError, match='Not enough disk space'):
            confirm_download([(tmp_path, 60, 0), (tmp_path / 'b', 30, 20)], assume_yes=True)

    def test_free_space_is_read_from_the_nearest_existing_directory(self, tmp_path):
        device, free = fetch.free_space(tmp_path / 'not' / 'yet' / 'there')
        assert device == tmp_path.stat().st_dev and free > 0

    @staticmethod
    def refuse(prompt):
        raise AssertionError('asked for confirmation')


class TestFetchEndToEnd:
    """`fetch_dataset` against a record served from local files."""

    @staticmethod
    def setup(tmp_path, doi='10.5281/zenodo.2', version='v2'):
        served = tmp_path / 'served'
        served.mkdir(exist_ok=True)
        archive = served / 'part.zip'
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('part/file.txt', 'hello')
        data = {
            'version': 1,
            'datasets': {
                'example': {
                    'title': 'An example dataset',
                    'version': version,
                    'doi': doi,
                    'record': 2,
                    'dest': 'example',
                    'files': [{'name': 'part.zip', 'size': archive.stat().st_size, 'extract': True}],
                }
            },
        }
        root = tmp_path / 'data'
        root.mkdir(exist_ok=True)
        spec = load_registry(write_registry(root, data))['example']
        record = {
            'doi': doi,
            'files': [
                {
                    'key': 'part.zip',
                    'size': archive.stat().st_size,
                    'checksum': 'md5:' + file_digest(archive),
                    'links': {'self': archive.as_uri()},
                }
            ],
        }
        return spec, root, record

    def test_downloads_verifies_extracts_and_records_the_version(self, tmp_path, capsys):
        spec, root, record = self.setup(tmp_path)
        dest = fetch.fetch_dataset(spec, root, record=record)
        assert (dest / 'part' / 'file.txt').read_text(encoding='utf-8') == 'hello'
        receipt = json.loads((dest / RECEIPT_FILENAME).read_text(encoding='utf-8'))
        assert (receipt['doi'], receipt['version']) == ('10.5281/zenodo.2', 'v2')
        assert receipt['files'][0]['matched'] == 'record'
        # An unpinned file is reported as checked against Zenodo, without asking for a registry edit.
        out = capsys.readouterr().out
        assert 'part.zip: no checksum in the registry, so verified against the one Zenodo reports' in out
        assert 'Pin them' not in out

    def test_the_same_version_fetches_again(self, tmp_path):
        spec, root, record = self.setup(tmp_path)
        fetch.fetch_dataset(spec, root, record=record)
        assert fetch.other_version(spec, spec.resolve_dest(root)) is None
        fetch.fetch_dataset(spec, root, record=record)

    def test_another_version_on_disk_is_refused_rather_than_overwritten(self, tmp_path, capsys):
        old, root, record = self.setup(tmp_path, doi='10.5281/zenodo.1', version='v1')
        fetch.fetch_dataset(old, root, record=record)
        new, _, record = self.setup(tmp_path)
        with pytest.raises(FetchError, match=r'holds DOI 10\.5281/zenodo\.1 \(v1\).*pins DOI 10\.5281/zenodo\.2'):
            fetch.fetch_dataset(new, root, record=record)
        assert fetch.check_dataset(new, root) is False
        assert 'holds DOI 10.5281/zenodo.1' in capsys.readouterr().out

    def test_a_copy_without_a_receipt_is_not_judged(self, tmp_path):
        spec, root, _ = self.setup(tmp_path)
        (root / 'example').mkdir()
        assert fetch.other_version(spec, root / 'example') is None
