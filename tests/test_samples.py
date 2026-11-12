#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""The committed sample data, the annotation commands, and the `--from-frames` path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pytest
import yaml

from stabilo_benchmark.annotation import CONVERT_ARGS, DETECT_ARGS, commands
from stabilo_benchmark.gcp import GCP_DIR, GCP_MANUAL_DIR, GCP_REPEAT_DIR, load_gcp_file
from stabilo_benchmark.masking import load_annotations
from stabilo_benchmark.oracle import ORACLE_DIR
from stabilo_benchmark.regions import EXCLUSION_FILE
from stabilo_benchmark.report.annotate_gcp import audit_frames
from stabilo_benchmark.runners.prepare import (
    DERIVED_FROM_FRAMES,
    OFFSET_BIN_KEYS,
    PrepareError,
    collect_source_frames,
    describe_path,
    frame_index_from_name,
    probe_video,
    run_prepare_from_frames,
)
from stabilo_benchmark.sampling import plan_bins

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / 'data' / 'sample'
STATIC_SAMPLE = SAMPLE / 'static'
CASE_NAME = '2022-10-06_D9_AM3_S2'
DYNAMIC_SAMPLE = SAMPLE / 'dynamic' / CASE_NAME
SAMPLE_CONFIG = REPO_ROOT / 'configs' / 'sample' / 'dynamic.yaml'
STATIC_CONFIG = REPO_ROOT / 'configs' / 'sample' / 'static.yaml'
RECORD = REPO_ROOT / 'data' / 'songdo'
# Every input the sample ships; CASE.json is the one file `prepare` wrote rather than copied.
COPIED = ('images', 'labels_hbb', 'labels_obb', 'labels_confidence', GCP_DIR, EXCLUSION_FILE, 'PROVENANCE_hbb.txt',
          'PROVENANCE_obb.txt')  # fmt: skip


@pytest.mark.parametrize('root', [STATIC_SAMPLE, DYNAMIC_SAMPLE], ids=['static', 'dynamic'])
class TestSampleLayout:
    """Both samples have the one layout both benchmarks read, written by hbb2obb."""

    def test_every_image_has_both_label_files(self, root):
        images = sorted((root / 'images').glob('*.jpg'))
        assert len(images) >= 3
        for image in images:
            for kind in ('labels_hbb', 'labels_obb'):
                assert (root / kind / f"{image.stem}.txt").is_file(), f"{image.name} has no {kind} file"

    def test_label_files_are_five_and_nine_column_absolute_pixels(self, root):
        for kind, columns in (('labels_hbb', 5), ('labels_obb', 9)):
            for label in sorted((root / kind).glob('*.txt')):
                raw = np.loadtxt(label, ndmin=2)
                assert raw.shape[1] == columns, f"{kind}/{label.name} is not {columns}-column"
                assert raw[:, 1:].max() > 1.0, f"{kind}/{label.name} looks normalized"

    def test_labels_load_as_one_conversion_inside_the_frame(self, root):
        for image_path in sorted((root / 'images').glob('*.jpg')):
            height, width = cv2.imread(str(image_path)).shape[:2]
            labels = load_annotations(root, image_path.stem, width, height)
            assert len(labels) > 0
            corners = labels.corners()
            assert corners[..., 0].max() <= width + 1 and corners[..., 1].max() <= height + 1

    def test_provenance_records_the_pinned_settings(self, root):
        detect = (root / 'PROVENANCE_hbb.txt').read_text(encoding='utf-8')
        assert '--model geotrax' in detect
        # hbb2obb >= 1.11.1 records the complete command and the weights' license credit.
        assert '--no_confidence' in detect
        assert 'cc-by-4.0' in detect.lower()
        convert = (root / 'PROVENANCE_obb.txt').read_text(encoding='utf-8')
        for flag, value in zip(CONVERT_ARGS[:-2:2], CONVERT_ARGS[1:-2:2], strict=True):
            assert f"{flag} {value}" in convert, f"PROVENANCE_obb.txt was not made with {flag} {value}"


@pytest.fixture(scope='module')
def text():
    """data/sample/README.md."""
    return (SAMPLE / 'README.md').read_text(encoding='utf-8')


class TestSampleReadme:
    """data/sample/README.md names every source, under its license."""

    def test_the_record_is_cited_under_cc_by(self, text):
        assert 'CC BY 4.0' in text
        assert '10.5281/zenodo.23623540' in text

    def test_every_static_scene_and_the_case_are_named(self, text):
        for image in sorted(STATIC_SAMPLE.glob('images/*.jpg')):
            assert f"`{image.stem}`" in text, f"{image.stem} is not described"
        assert f"`{CASE_NAME}`" in text


def sample_files():
    """Every copied input of the sample, relative to data/sample/ (the same path under the record)."""
    files = []
    for root in (STATIC_SAMPLE, DYNAMIC_SAMPLE):
        for name in COPIED:
            path = root / name
            if path.is_dir():
                files.extend(sorted(p for p in path.iterdir() if p.is_file()))
            elif path.is_file():
                files.append(path)
    return [path.relative_to(SAMPLE) for path in files]


@pytest.mark.skipif(not (RECORD / 'dynamic' / CASE_NAME).is_dir(), reason='needs the data record (fetch songdo)')
class TestSampleIsACopyOfTheRecord:
    """data/sample/ is a subset of the data record: every input byte for byte, at the same path."""

    def test_every_copied_file_is_identical_to_the_records(self):
        files = sample_files()
        assert len(files) > 60
        differing = [str(f) for f in files if (SAMPLE / f).read_bytes() != (RECORD / f).read_bytes()]
        assert differing == []

    def test_the_case_manifest_agrees_with_the_records(self, case):
        record = json.loads((RECORD / 'dynamic' / CASE_NAME / 'CASE.json').read_text(encoding='utf-8'))
        assert set(case['frames']) <= set(record['frames'])
        assert case['reference_index'] == record['reference_index']
        assert case['cell'] == record['cell']
        for key in ('fps', 'width', 'height'):
            assert case['video'][key] == record['video'][key], key
        for key in OFFSET_BIN_KEYS:
            assert case['sampling'][key] == record['sampling'][key], key


def without(config: dict, *paths: str) -> dict:
    """A copy of `config` without the dotted `paths`."""
    config = json.loads(json.dumps(config))
    for dotted in paths:
        *parents, leaf = dotted.split('.')
        node = config
        for key in parents:
            node = node[key]
        node.pop(leaf)
    return config


class TestSampleConfigs:
    """The sample configs are the paper's, restricted to ORB and SIFT."""

    def load(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding='utf-8'))

    def test_dynamic_is_offline_masking_restricted_to_orb_and_sift(self):
        sample = self.load(SAMPLE_CONFIG)
        paper = self.load(REPO_ROOT / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml')
        assert sample['pipeline']['preset'] == ['orb', 'sift']
        for key in ('filter_ratio', 'ransac_epipolar_threshold'):
            assert sample['pipeline'][key] == {name: paper['pipeline'][key][name] for name in ('orb', 'sift')}
        restricted = ('name', 'pipeline.preset', 'pipeline.filter_ratio', 'pipeline.ransac_epipolar_threshold')
        assert without(sample, *restricted) == without(paper, *restricted)

    def test_static_is_detectors_restricted_to_orb_and_sift(self):
        sample = self.load(STATIC_CONFIG)
        paper = self.load(REPO_ROOT / 'configs' / 'songdo' / 'static' / 'detectors.yaml')
        assert sample['pipeline']['preset'] == ['orb', 'sift']
        assert (REPO_ROOT / 'configs' / 'sample' / sample['scenes']['dir']).resolve() == STATIC_SAMPLE
        restricted = ('name', 'trials', 'scenes', 'pipeline.preset')
        assert without(sample, *restricted) == without(paper, *restricted)


class TestAnnotationCommands:
    def test_the_readme_prints_the_commands_annotation_runs(self):
        # The README and annotation.py hold the same commands.
        readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
        for command in commands(PurePosixPath('<root>/images')):  # as printed, whatever the host
            assert ' '.join(command) in readme, f"README does not show: {' '.join(command)}"

    def test_detection_writes_no_confidence_column(self):
        assert '--no_confidence' in DETECT_ARGS and '--save_provenance' in DETECT_ARGS


@pytest.mark.parametrize('root', [STATIC_SAMPLE, DYNAMIC_SAMPLE], ids=['static', 'dynamic'])
class TestProvenanceIsPortable:
    """The committed provenance files use relative paths and name no home directory."""

    def test_no_provenance_file_carries_an_absolute_path(self, root):
        for name in ('PROVENANCE_hbb.txt', 'PROVENANCE_obb.txt'):
            path = root / name
            assert path.is_file(), f"{root.name} has no {name}"
            offenders = [
                line
                for line in path.read_text(encoding='utf-8').splitlines()
                if ('/Users/' in line or '/home/' in line or line.lstrip().startswith('/'))
            ]
            assert offenders == [], f"{root.name}/{name} carries absolute paths: {offenders}"


@pytest.fixture(scope='module')
def case():
    """The committed dynamic case's manifest."""
    return json.loads((DYNAMIC_SAMPLE / 'CASE.json').read_text(encoding='utf-8'))


class TestDynamicSampleCase:
    def test_manifest_frame_list_matches_the_files_on_disk(self, case):
        on_disk = sorted(int(p.stem) for p in DYNAMIC_SAMPLE.glob('images/*.jpg'))
        assert on_disk == sorted(int(i) for i in case['frames'])
        assert case['n_frames'] == len(on_disk)

    def test_the_reference_frame_is_present(self, case):
        reference = int(case['reference_index'])
        assert reference in [int(i) for i in case['frames']]
        assert (DYNAMIC_SAMPLE / 'images' / f"{reference:06d}.jpg").is_file()

    def test_frames_all_share_one_size_which_the_manifest_records(self, case):
        sizes = {cv2.imread(str(p)).shape[:2] for p in DYNAMIC_SAMPLE.glob('images/*.jpg')}
        assert len(sizes) == 1, f"mixed frame sizes: {sizes}"
        height, width = sizes.pop()
        assert (width, height) == (int(case['video']['width']), int(case['video']['height']))

    def test_the_source_video_is_recorded_as_unavailable(self, case):
        # The case ships frames, not the recording.
        assert case['video']['source_kind'] == 'frames'
        assert case['video']['available'] is False
        assert case['video']['frame_count_inferred'] is True

    def test_the_manifest_holds_no_machine_specific_absolute_paths(self, case):
        # Source paths are relative.
        for value in (case['video']['path'], case['sampling']['source']):
            assert not Path(value).is_absolute(), f"{value} is absolute"

    def test_frame_count_spans_the_sampled_frames(self, case):
        assert int(case['video']['frame_count']) > max(int(i) for i in case['frames'])

    def test_the_case_records_its_survey_cell_and_the_configs_offset_bins(self, case):
        assert case['cell']
        plan = yaml.safe_load(SAMPLE_CONFIG.read_text(encoding='utf-8'))['sampling']
        for key in OFFSET_BIN_KEYS:
            assert case['sampling'][key] == plan[key], key

    def test_the_shortest_offsets_are_all_present(self, case):
        reference = int(case['reference_index'])
        assert {reference + 1, reference + 2, reference + 3} <= set(case['frames'])

    def test_annotate_gcp_opens_the_frames_that_carry_points(self, case):
        # Two frames per audit bin, as the protocol picks them, and the reference.
        reference = int(case['reference_index'])
        annotated = sorted(int(path.stem) for path in (DYNAMIC_SAMPLE / GCP_DIR).glob('*.json'))
        picked = audit_frames(case['frames'], reference, plan_bins(case['sampling'], 'audit_bins'))
        assert annotated == sorted([reference, *picked])
        assert len(picked) == 2 * len(case['sampling']['audit_bins'])

    def test_points_and_polygons_lie_in_the_frames(self, case):
        width, height = int(case['video']['width']), int(case['video']['height'])
        for path in sorted((DYNAMIC_SAMPLE / GCP_DIR).glob('*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            assert (payload['imageWidth'], payload['imageHeight']) == (width, height)
            points = load_gcp_file(path).points
            assert len(points) >= 12
            assert (points >= 0).all() and (points[:, 0] < width).all() and (points[:, 1] < height).all()
        exclusion = json.loads((DYNAMIC_SAMPLE / 'EXCLUSION.json').read_text(encoding='utf-8'))
        assert exclusion['imagePath'] == f"images/{int(case['reference_index']):06d}.jpg"
        polygons = [np.asarray(shape['points']) for shape in exclusion['shapes']]
        assert polygons and all(p[:, 0].max() <= width and p[:, 1].max() <= height for p in polygons)


class TestFrameIndexParsing:
    def test_a_bare_index_is_read(self):
        assert frame_index_from_name('000500') == 500

    def test_an_index_after_a_source_recording_name_is_read(self):
        # Only the trailing digits are the index.
        assert frame_index_from_name('2022-10-07_D1_PM5_B1.MP4_000500') == 500

    def test_digits_earlier_in_the_name_do_not_win(self):
        assert frame_index_from_name('2022-10-07_D7_PM5_L3.MP4_003500') == 3500

    def test_a_name_without_digits_is_an_error(self):
        with pytest.raises(PrepareError, match='Cannot read a frame index'):
            frame_index_from_name('reference')


class TestCollectSourceFrames:
    def make(self, directory: Path, names) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        for name in names:
            cv2.imwrite(str(directory / name), blank)
        return directory

    def test_frames_are_keyed_by_index(self, tmp_path):
        source = self.make(tmp_path / 'f', ['clip.MP4_000000.jpg', 'clip.MP4_000500.jpg'])
        assert sorted(collect_source_frames(source)) == [0, 500]

    def test_a_nested_images_directory_is_entered(self, tmp_path):
        self.make(tmp_path / 'case' / 'images', ['a_000000.jpg', 'a_000010.jpg'])
        assert sorted(collect_source_frames(tmp_path / 'case')) == [0, 10]

    def test_two_frames_claiming_one_index_is_an_error(self, tmp_path):
        # Frames from two recordings in one directory raise.
        source = self.make(tmp_path / 'f', ['b1_000500.jpg', 'b2_000500.jpg'])
        with pytest.raises(PrepareError, match='claim index 500'):
            collect_source_frames(source)

    def test_fewer_than_two_frames_is_an_error(self, tmp_path):
        source = self.make(tmp_path / 'f', ['only_000000.jpg'])
        with pytest.raises(PrepareError, match='at least two'):
            collect_source_frames(source)

    def test_non_images_are_ignored(self, tmp_path):
        source = self.make(tmp_path / 'f', ['a_000000.jpg', 'a_000500.jpg'])
        (source / 'notes_999.md').write_text('not a frame', encoding='utf-8')
        assert sorted(collect_source_frames(source)) == [0, 500]

    def test_a_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(PrepareError, match='not found'):
            collect_source_frames(tmp_path / 'absent')


class TestPrepareFromFrames:
    """`prepare --from-frames`: which frames are adopted, and what a config adds to them."""

    def frames(self, directory: Path, indices) -> Path:
        directory.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for index in indices:
            image = rng.integers(0, 255, (16, 24, 3), dtype=np.uint8)
            cv2.imwrite(str(directory / f"clip.MP4_{index:06d}.jpg"), image)
        return directory

    def args(self, output: Path, **values) -> argparse.Namespace:
        defaults = {
            'output': output, 'overwrite': False, 'frames': None, 'max_frames': None, 'reference': None,
            'config': None, 'scale': 1.0, 'fps': None, 'video_name': None, 'jpeg_quality': 95, 'seed': 0,
            'cell': None,
        }  # fmt: skip
        return argparse.Namespace(**{**defaults, **values})

    def prepare(self, tmp_path, monkeypatch, indices, **values) -> dict:
        monkeypatch.chdir(tmp_path)
        source = self.frames(tmp_path / 'frames', indices)
        case = run_prepare_from_frames(source, self.args(tmp_path / 'case', **values))
        return json.loads((case / 'CASE.json').read_text(encoding='utf-8'))

    def test_frames_selects_the_adopted_indices(self, tmp_path, monkeypatch):
        case = self.prepare(tmp_path, monkeypatch, [10, 11, 12, 40], frames=[10, 12, 40], reference=10)
        assert case['frames'] == [10, 12, 40]
        assert sorted(p.name for p in (tmp_path / 'case' / 'images').iterdir()) == [
            '000010.jpg', '000012.jpg', '000040.jpg',
        ]  # fmt: skip

    def test_a_frame_missing_from_the_directory_is_an_error(self, tmp_path, monkeypatch):
        with pytest.raises(PrepareError, match='99'):
            self.prepare(tmp_path, monkeypatch, [10, 11], frames=[10, 99], reference=10)

    def test_a_config_gives_the_reference_and_the_offset_bins(self, tmp_path, monkeypatch):
        plan = yaml.safe_load(SAMPLE_CONFIG.read_text(encoding='utf-8'))
        plan['reference'] = 11
        config = tmp_path / 'dynamic.yaml'
        config.write_text(yaml.safe_dump(plan), encoding='utf-8')
        case = self.prepare(tmp_path, monkeypatch, [10, 11, 12], config=config)
        assert case['reference_index'] == 11
        assert case['sampling']['mode'] == 'as-provided' and case['sampling']['source'] == 'frames'
        for key in OFFSET_BIN_KEYS:
            assert case['sampling'][key] == plan['sampling'][key], key

    def test_reference_overrides_the_configs(self, tmp_path, monkeypatch):
        case = self.prepare(tmp_path, monkeypatch, [10, 11, 12], config=SAMPLE_CONFIG, reference=12)
        assert case['reference_index'] == 12

    def test_overwrite_replaces_the_frames(self, tmp_path, monkeypatch):
        self.prepare(tmp_path, monkeypatch, [10, 11, 12], reference=10)
        args = self.args(tmp_path / 'case', frames=[10, 12], reference=10, overwrite=True)
        run_prepare_from_frames(tmp_path / 'frames', args)
        assert sorted(p.name for p in (tmp_path / 'case' / 'images').iterdir()) == ['000010.jpg', '000012.jpg']

    def test_overwrite_refuses_while_labels_of_the_old_frames_remain(self, tmp_path, monkeypatch):
        self.prepare(tmp_path, monkeypatch, [10, 11, 12], reference=10)
        (tmp_path / 'case' / 'labels_obb').mkdir()
        args = self.args(tmp_path / 'case', reference=10, overwrite=True)
        with pytest.raises(PrepareError, match='labels_obb'):
            run_prepare_from_frames(tmp_path / 'frames', args)
        assert len(list((tmp_path / 'case' / 'images').iterdir())) == 3

    def test_overwrite_never_adopts_the_frames_it_replaces(self, tmp_path, monkeypatch):
        self.prepare(tmp_path, monkeypatch, [10, 11, 12], reference=10)
        case = tmp_path / 'case'
        args = self.args(case, reference=10, overwrite=True)
        with pytest.raises(PrepareError, match='own frames'):
            run_prepare_from_frames(case / 'images', args)
        assert len(list((case / 'images').iterdir())) == 3

    def test_everything_made_from_frames_is_named(self):
        assert {GCP_DIR, GCP_REPEAT_DIR, GCP_MANUAL_DIR, ORACLE_DIR, EXCLUSION_FILE, 'results'} <= set(
            DERIVED_FROM_FRAMES
        )

    def test_without_a_config_no_bins_are_recorded(self, tmp_path, monkeypatch):
        case = self.prepare(tmp_path, monkeypatch, [0, 1, 2])
        assert case['reference_index'] == 0
        assert not set(OFFSET_BIN_KEYS) & set(case['sampling'])


class TestDescribePath:
    def test_a_path_inside_the_tree_is_recorded_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'data').mkdir()
        assert describe_path(tmp_path / 'data') == 'data'

    def test_a_path_outside_the_tree_keeps_its_absolute_form(self, tmp_path, monkeypatch):
        inside = tmp_path / 'repo'
        inside.mkdir()
        monkeypatch.chdir(inside)
        assert Path(describe_path(tmp_path)).is_absolute()

    def test_a_link_into_the_tree_records_the_link_not_its_target(self, tmp_path, monkeypatch):
        (tmp_path / 'storage').mkdir()
        (tmp_path / 'repo').mkdir()
        (tmp_path / 'repo' / 'videos').symlink_to(tmp_path / 'storage')
        monkeypatch.chdir(tmp_path / 'repo')
        assert describe_path(Path('videos') / 'a.mp4') == str(Path('videos') / 'a.mp4')

    def test_a_probed_video_is_recorded_by_its_in_tree_link(self, tmp_path, monkeypatch):
        (tmp_path / 'storage').mkdir()
        writer = cv2.VideoWriter(str(tmp_path / 'storage' / 'a.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 24))
        for _ in range(3):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
        writer.release()
        (tmp_path / 'repo').mkdir()
        (tmp_path / 'repo' / 'videos').symlink_to(tmp_path / 'storage')
        monkeypatch.chdir(tmp_path / 'repo')
        assert probe_video(Path('videos') / 'a.avi')['path'] == str(Path('videos') / 'a.avi')


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
class TestDataIgnoreRules:
    """The ignore rules under `data/` are a whitelist."""

    def ignored(self, path: str) -> bool:
        result = subprocess.run(['git', 'check-ignore', '-q', path], cwd=REPO_ROOT, capture_output=True, check=False)
        return result.returncode == 0

    def test_the_fetched_record_is_ignored(self):
        # The directory itself: a path inside it cannot be checked where `data/songdo` is a symlink, and
        # nothing below an ignored directory can be re-included.
        assert self.ignored('data/songdo')

    def test_a_new_directory_under_data_is_ignored_by_default(self):
        # Anything new under data/ stays untracked.
        assert self.ignored('data/something_nobody_anticipated/big.bin')

    def test_the_committed_sample_is_not_ignored(self):
        assert not self.ignored('data/sample/static/images/2022-10-07_PM4_Q_D8.jpg')
        assert not self.ignored(f'data/sample/dynamic/{CASE_NAME}/CASE.json')
        assert not self.ignored('data/registry.yaml')
        assert not self.ignored('data/README.md')

    def test_outputs_written_into_the_sample_are_ignored(self):
        assert self.ignored(f'data/sample/dynamic/{CASE_NAME}/oracle/ORACLE.json')
        assert self.ignored(f'data/sample/dynamic/{CASE_NAME}/results/smoke/MANIFEST.json')
        assert self.ignored('data/sample/static/results/smoke/MANIFEST.json')
        assert self.ignored(f'data/sample/dynamic/{CASE_NAME}/gcp_repeat/000300.json')
        assert self.ignored(f'data/sample/dynamic/{CASE_NAME}/gcp_manual/000300.json')

    def test_the_shipped_points_and_polygons_are_not_ignored(self):
        assert not self.ignored(f'data/sample/dynamic/{CASE_NAME}/gcp/000300.json')
        assert not self.ignored(f'data/sample/dynamic/{CASE_NAME}/EXCLUSION.json')

    def test_git_never_converts_the_samples_line_endings(self):
        # The copies stay byte-identical to the record on every platform.
        result = subprocess.run(
            ['git', 'check-attr', 'text', '--', f'data/sample/dynamic/{CASE_NAME}/labels_obb/000300.txt'],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding='utf-8', check=True,
        )  # fmt: skip
        assert result.stdout.strip().endswith('text: unset')
