#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""The survey: staging and labeling frames, reading segments, and filling the design, on known pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stabilo_benchmark import scene
from stabilo_benchmark.runners import survey
from stabilo_benchmark.runners.survey import (
    CONTROL_CELL,
    SurveyError,
    assign_cells,
    directory_frames,
    ensure_labels,
    images_dir_of,
    segment_frames,
    select_cases,
    stage_video_frames,
)

MASKING_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'songdo' / 'dynamic' / 'offline_masking.yaml'


def candidate(name: str, moving: float, coherence: float, static: float = 0.01, stationarity: float = 0.6, **extra):
    base = {
        'name': name,
        'video': name,
        'frame': 0,
        'displaced': moving,
        'coherence': coherence,
        'static': static,
        'stationarity': stationarity,
        'coverage': 0.02,
        'texture': 4000.0,
    }
    base.update(extra)
    return base


def frames(directory, stems):
    directory.mkdir(parents=True, exist_ok=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for stem in stems:
        cv2.imwrite(str(directory / f"{stem}.jpg"), image)
    return sorted(directory.glob('*.jpg'))


def video(path: Path, count: int, size=(160, 120)) -> Path:
    """An MJPG clip whose frame i is noise seeded by i, so decoded frames are told apart."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    for i in range(count):
        writer.write(np.random.default_rng(i).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


class TestCellAssignment:
    POOL = [candidate(f"c{i}", 0.001 * (i + 1), 0.1 * (i % 9 + 1)) for i in range(20)]

    def test_splits_default_to_the_medians_and_are_reported(self):
        strata = assign_cells([dict(c) for c in self.POOL])
        assert strata['splits']['moving'] == pytest.approx(np.median([c['displaced'] for c in self.POOL]))
        assert strata['splits']['basis'] == 'pool median'
        assert set(strata['cells']) == set(survey.DESIGN_CELLS)

    def test_explicit_splits_are_honored(self):
        strata = assign_cells([dict(c) for c in self.POOL], moving_split=0.5, coherence_split=0.5)
        assert set(strata['cells']) <= {'low moving / coherent', 'low moving / mixed'}
        assert strata['splits']['basis'] == 'explicit'

    def test_the_basis_names_each_axis_when_only_one_split_is_given(self):
        strata = assign_cells([dict(c) for c in self.POOL], moving_split=0.005)
        assert strata['splits']['basis'] == 'moving explicit, coherence pool median'

    def test_too_few_displaced_objects_leave_a_candidate_unclassified(self):
        pool = [dict(c) for c in self.POOL] + [candidate('empty', 0.0, float('nan'))]
        strata = assign_cells(pool)
        assert pool[-1]['cell'] == survey.UNCLASSIFIED
        assert strata['splits']['moving'] == pytest.approx(np.median([c['displaced'] for c in self.POOL]))
        plan = select_cases(pool, strata, controls=0)
        assert 'empty' not in [p['name'] for p in plan['selected']]

    def test_an_empty_pool_is_empty_not_an_error(self):
        assert assign_cells([]) == {'cells': {}, 'splits': {}, 'controls': []}


class TestStationaryControl:
    @staticmethod
    def pool():
        return [candidate(f"c{i}", 0.004, 0.3, static=0.01, stationarity=0.5 + 0.01 * i) for i in range(20)]

    def test_the_foreground_that_stays_put_is_the_control(self):
        pool = self.pool() + [candidate('queue', 0.0005, 0.3, static=0.03, stationarity=0.95)]
        strata = assign_cells(pool)
        assert 'queue' in strata['controls']
        assert pool[-1]['cell'] == CONTROL_CELL

    def test_a_still_scene_with_little_foreground_is_not_a_control(self):
        pool = self.pool() + [candidate('empty', 0.0, 0.3, static=0.0001, stationarity=1.0)]
        assert 'empty' not in assign_cells(pool)['controls']

    def test_a_pool_that_does_not_differ_in_stationarity_has_no_control(self):
        pool = [candidate(f"c{i}", 0.001 * i, 0.3, stationarity=0.7) for i in range(10)]
        assert assign_cells(pool)['controls'] == []

    def test_controls_are_left_out_of_the_splits(self):
        pool = self.pool() + [candidate('queue', 0.0, 0.3, static=0.5, stationarity=1.0)]
        strata = assign_cells(pool)
        assert strata['splits']['moving'] == pytest.approx(0.004)


def segment(video: str, frame: int, moving: float, coherence: float, **extra) -> dict:
    return candidate(f"{video}@{frame:06d}", moving, coherence, video=video, frame=frame, **extra)


class TestSelection:
    @staticmethod
    def _plan(points, **kwargs):
        strata = assign_cells(points, moving_split=0.005, coherence_split=0.5)
        return select_cases(points, strata, **kwargs)

    @staticmethod
    def pool():
        return [
            segment('A', 0, 0.020, 0.90),
            segment('A', 9000, 0.019, 0.88),
            segment('B', 0, 0.018, 0.10),
            segment('B', 9000, 0.017, 0.12),
            segment('C', 0, 0.001, 0.90),
            segment('C', 9000, 0.002, 0.92),
            segment('D', 0, 0.001, 0.10),
            segment('D', 9000, 0.002, 0.11),
            segment('E', 0, 0.000, 0.10, static=0.05, stationarity=0.99),
        ]

    def test_a_complete_pool_fills_the_design(self):
        plan = self._plan(self.pool())
        assert plan['unfilled'] == {}
        assert sum(p['cell'] == CONTROL_CELL for p in plan['selected']) == 1

    def test_a_missing_condition_is_reported_not_padded(self):
        pool = [p for p in self.pool() if p['video'] != 'C']
        plan = self._plan(pool)
        assert plan['unfilled'] == {'low moving / coherent': 2}
        assert not any(p['cell'] == 'low moving / coherent' for p in plan['selected'])

    def test_segments_far_from_the_boundary_are_preferred(self):
        pool = [segment('A', 0, 0.0051, 0.51), segment('B', 0, 0.030, 0.95)] + self.pool()[4:]
        plan = self._plan(pool, per_cell=1)
        assert [p['video'] for p in plan['selected'] if p['cell'] == 'high moving / coherent'] == ['B']

    def test_the_weaker_axis_decides_the_rank(self):
        pool = [segment('A', 0, 0.090, 0.51), segment('B', 0, 0.012, 0.95)] + self.pool()[4:]
        plan = self._plan(pool, per_cell=1)
        assert [p['video'] for p in plan['selected'] if p['cell'] == 'high moving / coherent'] == ['B']

    def test_one_recording_cannot_fill_a_cell_alone(self):
        pool = [segment('A', f * 9000, 0.020, 0.90) for f in range(5)] + self.pool()[4:]
        plan = self._plan(pool, max_per_video=2, per_cell=4)
        assert plan['filled']['high moving / coherent'] == 2

    def test_segments_too_close_in_time_are_the_same_scene(self):
        pool = [segment('A', 0, 0.020, 0.90), segment('A', 100, 0.020, 0.90)] + self.pool()[4:]
        plan = self._plan(pool, min_separation=3000)
        assert plan['filled']['high moving / coherent'] == 1

    def test_controls_can_be_switched_off(self):
        plan = self._plan(self.pool(), controls=0)
        assert CONTROL_CELL not in plan['targets']
        assert not any(p['cell'] == CONTROL_CELL for p in plan['selected'])

    def test_a_cell_takes_at_most_one_segment_per_group(self):
        pool = [segment(f"v{i}", 0, 0.02 + i * 0.001, 0.9, group='B' if i < 3 else 'C') for i in range(5)]
        pool += self.pool()[2:]
        plan = self._plan(pool, per_cell=2, controls=0, max_per_group=1)
        picked = [p for p in plan['selected'] if p['cell'] == 'high moving / coherent']
        assert sorted(p['group'] for p in picked) == ['B', 'C']

    def test_without_a_limit_one_group_can_fill_a_cell(self):
        pool = [segment(f"v{i}", 0, 0.02 + i * 0.001, 0.9, group='B') for i in range(3)] + self.pool()[2:]
        plan = self._plan(pool, per_cell=2, controls=0)
        assert sum(p['cell'] == 'high moving / coherent' for p in plan['selected']) == 2


class TestDegenerateAxes:
    def test_a_pool_with_no_spread_is_flagged(self):
        pool = [candidate(f"c{i}", 0.005, 0.1 * i) for i in range(6)]
        assert assign_cells(pool)['degenerate_axes'] == ['moving']

    def test_a_spread_pool_is_not(self):
        pool = [candidate(f"c{i}", 0.001 * i, 0.1 * i) for i in range(6)]
        assert assign_cells(pool)['degenerate_axes'] == []


class TestFrames:
    def test_a_segment_is_its_reference_then_one_frame_per_offset(self):
        assert segment_frames(100, [4, 22, 43]) == [100, 104, 122, 143]

    def test_frames_are_decoded_in_order_and_named_by_index(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 3, 17], tmp_path / 'images')
        assert sorted(p.name for p in (tmp_path / 'images').iterdir()) == ['000000.jpg', '000003.jpg', '000017.jpg']
        capture = cv2.VideoCapture(str(clip))
        for _ in range(17):
            capture.grab()
        ok, frame = capture.read()
        capture.release()
        expected = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), survey.JPEG_QUALITY])[1].tobytes()
        assert (tmp_path / 'images' / '000017.jpg').read_bytes() == expected

    def test_a_second_staging_reports_identical_frames(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is False
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is True

    def test_a_matching_staging_record_skips_decoding(self, tmp_path, monkeypatch):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 5], tmp_path / 'images')
        monkeypatch.setattr(survey.cv2, 'VideoCapture', lambda *a: pytest.fail('decoded again'))
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is True

    def test_a_changed_source_is_decoded_again(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 5], tmp_path / 'images')
        video(clip, 30, size=(176, 120))
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is False

    def test_a_frame_no_longer_wanted_is_removed_and_reported(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 5, 9], tmp_path / 'images')
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is False
        assert not (tmp_path / 'images' / '000009.jpg').exists()

    def test_discarded_frames_are_decoded_again_and_keep_their_labels(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 5], tmp_path / 'images')
        survey.discard_frames(tmp_path / 'images')
        assert not list((tmp_path / 'images').glob('*.jpg'))
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is True
        assert sorted(p.name for p in (tmp_path / 'images').iterdir()) == ['000000.jpg', '000005.jpg']

    def test_discarded_frames_of_a_changed_source_lose_their_labels(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 30)
        stage_video_frames(clip, [0, 5], tmp_path / 'images')
        survey.discard_frames(tmp_path / 'images')
        record = json.loads((tmp_path / survey.STAGING_RECORD).read_text(encoding='utf-8'))
        record['sha256'][1] = '0' * 64
        (tmp_path / survey.STAGING_RECORD).write_text(json.dumps(record), encoding='utf-8')
        assert stage_video_frames(clip, [0, 5], tmp_path / 'images') is False

    def test_a_recording_that_ends_early_is_an_error(self, tmp_path):
        clip = video(tmp_path / 'clip.avi', 10)
        with pytest.raises(SurveyError, match='ended before frame 25'):
            stage_video_frames(clip, [0, 25], tmp_path / 'images')

    def test_an_extracted_set_is_indexed_by_the_last_number_in_each_name(self, tmp_path):
        frames(tmp_path, ['clip_2022_000004', 'clip_2022_000000', 'notes'])
        assert list(directory_frames(tmp_path)) == [0, 4]

    def test_an_empty_directory_raises(self, tmp_path):
        with pytest.raises(SurveyError, match='No images'):
            directory_frames(tmp_path)

    def test_a_root_holding_images_is_entered(self, tmp_path):
        frames(tmp_path / 'clip' / 'images', ['000000'])
        assert images_dir_of(tmp_path / 'clip') == tmp_path / 'clip' / 'images'
        assert images_dir_of(tmp_path / 'clip' / 'images') == tmp_path / 'clip' / 'images'


class TestLabels:
    @pytest.fixture
    def runs(self, monkeypatch):
        calls = []

        def fake(images, device=None, overwrite=False):
            calls.append(images)
            root = Path(images).parent
            for name in ('labels_hbb', 'labels_obb'):
                (root / name).mkdir(exist_ok=True)
            for name in survey.PROVENANCE_FILES:
                (root / name).write_text('done\n', encoding='utf-8')

        monkeypatch.setattr(survey, 'run_hbb2obb', fake)
        return calls

    def test_labels_beside_a_given_frame_set_are_used_as_they_are(self, tmp_path, runs):
        frames(tmp_path / 'images', ['000000'])
        for name in ('labels_hbb', 'labels_obb'):
            (tmp_path / name).mkdir()
        ensure_labels(tmp_path / 'images')
        assert runs == []

    def test_staged_frames_reuse_labels_only_once_both_commands_finished(self, tmp_path, runs):
        frames(tmp_path / 'images', ['000000'])
        for name in ('labels_hbb', 'labels_obb'):
            (tmp_path / name).mkdir()
        ensure_labels(tmp_path / 'images', staged=True)
        assert runs == [tmp_path / 'images']
        ensure_labels(tmp_path / 'images', staged=True)
        assert len(runs) == 1

    def test_rewritten_frames_lose_their_old_labels(self, tmp_path, runs):
        frames(tmp_path / 'images', ['000000'])
        (tmp_path / 'labels_obb').mkdir()
        (tmp_path / 'labels_obb' / 'old.txt').write_text('0 1 2 3 4 5 6 7 8\n', encoding='utf-8')
        ensure_labels(tmp_path / 'images', staged=True, stale=True)
        assert runs == [tmp_path / 'images']
        assert not (tmp_path / 'labels_obb' / 'old.txt').exists()

    def test_no_annotate_names_what_is_missing(self, tmp_path, runs):
        frames(tmp_path / 'images', ['000000'])
        with pytest.raises(SurveyError, match='labels_hbb/ and labels_obb/'):
            ensure_labels(tmp_path / 'images', annotate=False)


class TestAnchors:
    def test_a_cut_hover_is_one_segment_at_its_first_frame(self):
        assert survey._anchors(3999, 3600, 'start', 900, 'clip') == [0]

    def test_a_hover_shorter_than_the_horizon_is_rejected(self):
        with pytest.raises(SurveyError, match='needs 3600 frames'):
            survey._anchors(3525, 3600, 'start', 900, 'clip')

    def test_scan_points_keep_clear_of_both_ends_and_leave_room_for_the_case(self):
        anchors = survey._anchors(19999, 3600, 'scan', 900, 'clip')
        assert anchors[0] == 300 and anchors[-1] + 3600 <= 19999 - 300
        assert np.all(np.diff(anchors) == 900)

    def test_a_recording_too_short_to_scan_is_rejected(self):
        with pytest.raises(SurveyError, match='clear of'):
            survey._anchors(3800, 3600, 'scan', 900, 'clip')


class TestResolvePlan:
    @staticmethod
    def args(**kwargs):
        return argparse.Namespace(**{'horizon': None, 'config': MASKING_CONFIG, **kwargs})

    def test_the_shipped_masking_config_is_read_at_one_offset_per_survey_bin(self):
        horizon, offsets = survey.resolve_plan(self.args())
        assert horizon == 3600
        assert offsets == [1, 2, 5, 11, 22]

    def test_without_survey_bins_every_analysis_bin_is_read(self, tmp_path):
        config = tmp_path / 'plan.yaml'
        config.write_text(
            "kind: dynamic\nname: x\nsampling:\n  mode: stratified\n  bins: [[1, 15], [16, 30], [31, 60]]\n",
            encoding='utf-8',
        )
        horizon, offsets = survey.resolve_plan(self.args(config=config))
        assert horizon == 60 and offsets == [4, 22, 43]

    def test_an_explicit_horizon_wins_and_drops_bins_beyond_it(self):
        horizon, offsets = survey.resolve_plan(self.args(horizon=15))
        assert horizon == 15 and offsets == [1, 2, 5, 11]

    def test_a_config_is_required(self):
        with pytest.raises(SurveyError, match='-c'):
            survey.resolve_plan(self.args(config=None))

    def test_an_unbounded_plan_asks_for_an_explicit_horizon(self, tmp_path):
        config = tmp_path / 'random.yaml'
        config.write_text("kind: dynamic\nname: x\nsampling:\n  mode: random\n  count: 10\n", encoding='utf-8')
        with pytest.raises(SurveyError, match='not bounded'):
            survey.resolve_plan(self.args(config=config))

    def test_a_negative_horizon_is_an_error(self):
        with pytest.raises(SurveyError):
            survey.resolve_plan(self.args(horizon=-1))


class TestNamesAndGroups:
    def test_distinct_stems_keep_their_names(self, tmp_path):
        a, b = tmp_path / 'x' / 'A1.mp4', tmp_path / 'y' / 'B1.mp4'
        for path in (a, b):
            path.parent.mkdir(parents=True)
            path.touch()
        assert survey.unique_names([a, b]) == {a: 'A1', b: 'B1'}

    def test_colliding_stems_are_named_by_their_path(self, tmp_path):
        a = tmp_path / '2022-10-06' / 'D1' / 'AM1' / 'A1.MP4'
        b = tmp_path / '2022-10-06' / 'D2' / 'AM1' / 'A1.MP4'
        for path in (a, b):
            path.parent.mkdir(parents=True)
            path.touch()
        assert survey.unique_names([a, b]) == {a: 'D1_AM1_A1', b: 'D2_AM1_A1'}

    def test_a_layout_directory_is_named_for_its_clip(self, tmp_path):
        frames(tmp_path / 'clip' / 'images', ['000000'])
        assert survey.candidate_name(tmp_path / 'clip' / 'images') == 'clip'

    def test_the_group_is_the_pattern_s_first_capture(self):
        assert survey.group_of('D1_AM1_B3', r'_([A-Z])\d+$') == 'B'
        assert survey.group_of('unmatched', r'_([A-Z])\d+$') == 'unmatched'
        assert survey.group_of('D1_AM1_B3', None) == 'D1_AM1_B3'

    def test_a_pattern_without_a_group_is_an_error(self):
        with pytest.raises(SurveyError, match='capture group'):
            survey.group_of('D1_AM1_B3', r'[A-Z]\d+$')


class TestEndToEnd:
    """Two short synthetic hovers, one with a single stream and one standing still, through `run_survey`."""

    SIZE = (320, 240)
    LENGTH, WIDTH = 26, 12

    @classmethod
    def places(cls, stream: bool, index: int):
        """Object boxes at frame `index`: rows of objects; with `stream` they move 3 px per frame."""
        quads = []
        for row, y in enumerate((60, 120, 180)):
            for column in range(4):
                x = 40 + 60 * column + (3.0 * index if stream else 0.0) + 7 * row
                quads.append([[x, y], [x + cls.LENGTH, y], [x + cls.LENGTH, y + cls.WIDTH], [x, y + cls.WIDTH]])
        return np.array(quads, dtype=float)

    @classmethod
    def clip(cls, path: Path, stream: bool, count: int = 20) -> Path:
        rng = np.random.default_rng(1)
        ground = cv2.GaussianBlur(rng.integers(40, 215, (cls.SIZE[1], cls.SIZE[0]), dtype=np.uint8), (0, 0), 2)
        looks = [rng.integers(0, 255, (cls.WIDTH, cls.LENGTH, 3), dtype=np.uint8) for _ in range(12)]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, cls.SIZE)
        for i in range(count):
            image = cv2.cvtColor(ground, cv2.COLOR_GRAY2BGR)
            for look, quad in zip(looks, cls.places(stream, i), strict=True):
                x, y = int(round(quad[0, 0])), int(round(quad[0, 1]))
                if x + cls.LENGTH <= cls.SIZE[0]:
                    image[y : y + cls.WIDTH, x : x + cls.LENGTH] = cv2.resize(look, (cls.LENGTH, cls.WIDTH))
            writer.write(image)
        writer.release()
        return path

    @pytest.fixture
    def annotations(self, monkeypatch):
        calls = []

        def fake(images, device=None, overwrite=False):
            images = Path(images)
            calls.append(images.parent.name)
            stream = images.parent.name == 'flow'
            root = images.parent
            for name in ('labels_hbb', 'labels_obb'):
                (root / name).mkdir(exist_ok=True)
            for image in images.glob('*.jpg'):
                quads = [q for q in self.places(stream, int(image.stem)) if q[1, 0] <= self.SIZE[0]]
                hbb = [f"0 {q[:, 0].mean()} {q[:, 1].mean()} {self.LENGTH} {self.WIDTH}" for q in quads]
                obb = ['0 ' + ' '.join(f"{v:.1f}" for v in q.ravel()) for q in quads]
                (root / 'labels_hbb' / f"{image.stem}.txt").write_text('\n'.join(hbb) + '\n', encoding='utf-8')
                (root / 'labels_obb' / f"{image.stem}.txt").write_text('\n'.join(obb) + '\n', encoding='utf-8')
            for name in survey.PROVENANCE_FILES:
                (root / name).write_text('done\n', encoding='utf-8')

        monkeypatch.setattr(survey, 'run_hbb2obb', fake)
        return calls

    def run(self, tmp_path, sources, **overrides):
        config = tmp_path / 'plan.yaml'
        config.write_text(
            'kind: dynamic\nname: x\n'
            'sampling:\n  mode: stratified\n  per_bin: 2\n  bins:\n    - [1, 3]\n    - [4, 12]\n',
            encoding='utf-8',
        )
        args = argparse.Namespace(
            config=config,
            horizon=None,
            work_dir=tmp_path / 'work',
            output=None,
            no_annotate=False,
            device=None,
            per_cell=1,
            controls=0,
            moving_split=None,
            coherence_split=None,
            group_pattern=None,
            max_per_group=1,
            anchor='start',
            scan_stride=900,
            max_per_video=2,
            min_separation=3000,
            pool=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return json.loads(survey.run_survey(sources, args).read_text(encoding='utf-8'))

    def test_a_stream_reads_as_moving_and_a_queue_as_static(self, tmp_path, annotations):
        sources = [self.clip(tmp_path / 'flow.avi', True), self.clip(tmp_path / 'queue.avi', False)]
        report = self.run(tmp_path, sources)
        assert report['offsets'] == [2, 7]
        flow, queue = sorted(report['candidates'], key=lambda c: c['video'])
        assert flow['displaced'] > 0 and flow['coherence'] == pytest.approx(1.0)
        assert queue['displaced'] == 0 and queue['static'] > 0
        assert [p['offset'] for p in flow['per_offset']] == [2, 7]

    def test_a_repeated_survey_reuses_the_labels(self, tmp_path, annotations):
        sources = [self.clip(tmp_path / 'flow.avi', True), self.clip(tmp_path / 'queue.avi', False)]
        self.run(tmp_path, sources)
        self.run(tmp_path, sources)
        assert sorted(annotations) == ['flow', 'queue']

    def test_the_thresholds_are_recorded(self, tmp_path, annotations):
        report = self.run(tmp_path, [self.clip(tmp_path / 'flow.avi', True)])
        assert report['thresholds'] == survey.survey_thresholds()
        assert report['thresholds']['displaced_widths'] == scene.DISPLACED_WIDTHS and report['pooled'] == []

    def test_pooled_shards_select_as_one_survey(self, tmp_path, annotations):
        flow, queue = self.clip(tmp_path / 'flow.avi', True), self.clip(tmp_path / 'queue.avi', False)
        whole = self.run(tmp_path, [flow, queue])
        shards = [tmp_path / 'a.json', tmp_path / 'b.json']
        for source, shard in zip((flow, queue), shards, strict=True):
            self.run(tmp_path, [source], output=shard)
        pooled = self.run(tmp_path, [], pool=shards, output=tmp_path / 'pooled.json')
        assert sorted(pooled['candidates'], key=lambda c: c['name']) == sorted(
            whole['candidates'], key=lambda c: c['name']
        )
        assert pooled['strata'] == whole['strata'] and pooled['selection'] == whole['selection']
        assert len(pooled['pooled']) == 2

    def test_a_survey_of_another_plan_is_not_pooled(self, tmp_path, annotations):
        shard = tmp_path / 'a.json'
        self.run(tmp_path, [self.clip(tmp_path / 'flow.avi', True)], output=shard)
        report = json.loads(shard.read_text(encoding='utf-8'))
        report['thresholds'] = {**report['thresholds'], 'displaced_widths': 1.0}
        shard.write_text(json.dumps(report), encoding='utf-8')
        with pytest.raises(SurveyError, match='another thresholds'):
            self.run(tmp_path, [], pool=[shard], output=tmp_path / 'pooled.json')

    def test_a_segment_pooled_twice_is_an_error(self, tmp_path, annotations):
        shard = tmp_path / 'a.json'
        flow = self.clip(tmp_path / 'flow.avi', True)
        self.run(tmp_path, [flow], output=shard)
        with pytest.raises(SurveyError, match='described twice'):
            self.run(tmp_path, [flow], pool=[shard], output=tmp_path / 'pooled.json')

    def test_nothing_to_read_is_an_error(self, tmp_path):
        with pytest.raises(SurveyError, match='--pool'):
            self.run(tmp_path, [])

    def test_a_short_recording_is_skipped_not_fatal(self, tmp_path, annotations, capsys):
        sources = [self.clip(tmp_path / 'flow.avi', True), self.clip(tmp_path / 'queue.avi', False, count=8)]
        report = self.run(tmp_path, sources)
        assert [c['video'] for c in report['candidates']] == ['flow']
        assert 'Skipping' in capsys.readouterr().out
