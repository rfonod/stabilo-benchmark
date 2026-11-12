#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Label parsing and the none / HBB / OBB masking axis."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stabilo_benchmark.masking import (
    LabelError,
    box_format_for,
    corners_to_hbb,
    empty_labels,
    hbb_to_corners,
    is_annotated,
    load_annotations,
    load_labels,
    require_annotations,
    summarize,
)
from stabilo_benchmark.pipelines import MASKING_BOX_FORMAT

WIDTH, HEIGHT = 1920, 1080


def write_labels(path, rows):
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return path


class TestFormatDetection:
    def test_five_columns_read_as_horizontal(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        labels = load_labels(path, WIDTH, HEIGHT)
        assert not labels.has_obb
        assert np.allclose(labels.hbb, [[960, 540, 40, 20]])

    def test_nine_columns_read_as_oriented(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['1 100 100 140 100 140 120 100 120'])
        labels = load_labels(path, WIDTH, HEIGHT)
        assert labels.has_obb
        assert labels.classes.tolist() == [1]

    def test_unexpected_column_count_raises(self, tmp_path):
        # Eight columns match no layout: an even count is never a polygon (class + xy pairs
        # is always odd), and it is neither of the box widths.
        path = write_labels(tmp_path / '000000.txt', ['0 1 2 3 4 5 6 7'])
        with pytest.raises(LabelError, match='Cannot infer label format'):
            load_labels(path, WIDTH, HEIGHT)

    def test_explicit_format_mismatch_raises(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        with pytest.raises(LabelError, match='expects 9 or 10 columns'):
            load_labels(path, WIDTH, HEIGHT, label_format='obb')

    def test_six_columns_read_as_horizontal_with_confidence(self, tmp_path):
        # YOLO detector output: HBB + confidence.
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20 0.87'])
        labels = load_labels(path, WIDTH, HEIGHT)
        assert not labels.has_obb
        assert labels.has_confidence
        assert np.allclose(labels.hbb, [[960, 540, 40, 20]])
        assert np.allclose(labels.confidence, [0.87])

    def test_six_columns_are_never_inferred_as_xywha(self, tmp_path):
        # Six columns resolve to HBB + confidence under 'auto'.
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20 45'])
        assert not load_labels(path, WIDTH, HEIGHT).has_obb
        assert load_labels(path, WIDTH, HEIGHT, label_format='xywha', coords='absolute').has_obb

    def test_ten_columns_read_as_oriented_with_confidence(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['1 100 100 140 100 140 120 100 120 0.5'])
        labels = load_labels(path, WIDTH, HEIGHT)
        assert labels.has_obb
        assert np.allclose(labels.confidence, [0.5])

    def test_missing_file_yields_empty_labels(self, tmp_path):
        labels = load_labels(tmp_path / 'absent.txt', WIDTH, HEIGHT)
        assert len(labels) == 0

    def test_empty_file_yields_empty_labels(self, tmp_path):
        path = tmp_path / '000000.txt'
        path.write_text('', encoding='utf-8')
        assert len(load_labels(path, WIDTH, HEIGHT)) == 0


class TestCoordinateScaling:
    def test_normalized_coordinates_are_denormalized(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 0.5 0.5 0.1 0.2'])
        labels = load_labels(path, WIDTH, HEIGHT)
        assert np.allclose(labels.hbb, [[960, 540, 192, 216]])

    def test_absolute_coordinates_are_left_alone(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute')
        assert np.allclose(labels.hbb, [[960, 540, 40, 20]])

    def test_explicit_normalized_overrides_the_heuristic(self, tmp_path):
        # All values exceed 1, so 'auto' would call these absolute; the explicit setting wins.
        path = write_labels(tmp_path / '000000.txt', ['0 2 2 2 2'])
        labels = load_labels(path, WIDTH, HEIGHT, coords='normalized')
        assert np.allclose(labels.hbb, [[2 * WIDTH, 2 * HEIGHT, 2 * WIDTH, 2 * HEIGHT]])

    def test_oriented_coordinates_scale_per_axis(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4'])
        labels = load_labels(path, WIDTH, HEIGHT)
        corners = labels.obb.reshape(4, 2)
        assert np.allclose(corners[:, 0], np.array([0.1, 0.3, 0.3, 0.1]) * WIDTH)
        assert np.allclose(corners[:, 1], np.array([0.2, 0.2, 0.4, 0.4]) * HEIGHT)


class TestMaskingAxis:
    def test_none_passes_no_boxes_at_all(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        assert load_labels(path, WIDTH, HEIGHT).boxes_for('none') is None

    def test_hbb_returns_four_column_boxes(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 140 100 140 120 100 120'])
        assert load_labels(path, WIDTH, HEIGHT).boxes_for('hbb').shape == (1, 4)

    def test_obb_returns_eight_column_corners(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 140 100 140 120 100 120'])
        assert load_labels(path, WIDTH, HEIGHT).boxes_for('obb').shape == (1, 8)

    def test_obb_from_horizontal_labels_raises_instead_of_degrading(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        with pytest.raises(LabelError, match='requires oriented labels'):
            load_labels(path, WIDTH, HEIGHT).boxes_for('obb')

    def test_invalid_mode_raises(self):
        with pytest.raises(LabelError, match='Invalid masking mode'):
            empty_labels().boxes_for('rotated')

    def test_box_format_mapping_matches_stabilo_expectations(self):
        assert box_format_for('none') is None
        assert box_format_for('hbb') == 'xywh'
        assert box_format_for('obb') == 'four'
        assert MASKING_BOX_FORMAT['obb'] == 'four'

    def test_boxes_for_returns_a_copy(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute')
        boxes = labels.boxes_for('hbb')
        boxes[0, 0] = 0.0
        assert labels.hbb[0, 0] == 960


class TestGeometryConversions:
    def test_hbb_corner_round_trip(self):
        boxes = np.array([[50.0, 60.0, 20.0, 10.0], [300.0, 400.0, 8.0, 44.0]])
        assert np.allclose(corners_to_hbb(hbb_to_corners(boxes)), boxes)

    def test_corners_are_ordered_top_left_first(self):
        corners = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0]]))[0]
        assert np.allclose(corners[0], [80.0, 90.0])
        assert np.allclose(corners[2], [120.0, 110.0])

    def test_horizontal_labels_derive_corners(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 40 20'])
        assert load_labels(path, WIDTH, HEIGHT, coords='absolute').corners().shape == (1, 4, 2)


class TestClassFiltering:
    def test_only_requested_classes_survive(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 10 10 4 4', '2 20 20 4 4', '1 30 30 4 4'])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute', classes=[0, 1])
        assert sorted(labels.classes.tolist()) == [0, 1]
        assert len(labels) == 2

    def test_subset_keeps_both_geometries(self, tmp_path):
        path = write_labels(
            tmp_path / '000000.txt',
            ['0 100 100 140 100 140 120 100 120', '1 200 200 240 200 240 220 200 220'],
        )
        labels = load_labels(path, WIDTH, HEIGHT, classes=[1])
        assert labels.has_obb
        assert labels.hbb.shape == (1, 4)
        assert labels.obb.shape == (1, 8)


def write_pair(root, stem, hbb_rows, obb_rows):
    for name, rows in (('labels_hbb', hbb_rows), ('labels_obb', obb_rows)):
        (root / name).mkdir(parents=True, exist_ok=True)
        write_labels(root / name / f"{stem}.txt", rows)


class TestLoadAnnotations:
    """The one reader both benchmarks use: <root>/labels_hbb and <root>/labels_obb, row by row."""

    def test_the_horizontal_box_is_the_detectors_not_the_oriented_envelope(self, tmp_path):
        # The detector box is kept as is, even when looser than the OBB's envelope.
        write_pair(tmp_path, 'f', ['0 120 110 60 30'], ['0 100 100 140 100 140 120 100 120'])
        labels = load_annotations(tmp_path, 'f', WIDTH, HEIGHT)
        assert labels.boxes_for('hbb').tolist() == [[120, 110, 60, 30]]
        assert labels.boxes_for('obb').tolist() == [[100, 100, 140, 100, 140, 120, 100, 120]]

    @pytest.mark.parametrize('missing', ['labels_hbb', 'labels_obb'])
    def test_a_missing_label_directory_is_named(self, tmp_path, missing):
        write_pair(tmp_path, 'f', ['0 120 110 60 30'], ['0 100 100 140 100 140 120 100 120'])
        for path in (tmp_path / missing).iterdir():
            path.unlink()
        (tmp_path / missing).rmdir()
        with pytest.raises(LabelError, match=missing):
            load_annotations(tmp_path, 'f', WIDTH, HEIGHT)

    def test_a_missing_file_is_a_frame_without_vehicles(self, tmp_path):
        write_pair(tmp_path, 'other', [], [])
        labels = load_annotations(tmp_path, 'f', WIDTH, HEIGHT)
        assert len(labels) == 0 and labels.boxes_for('obb').shape == (0, 8)

    def test_files_that_are_not_one_conversion_raise(self, tmp_path):
        write_pair(tmp_path, 'f', ['0 120 110 60 30', '0 300 300 20 20'], ['0 100 100 140 100 140 120 100 120'])
        with pytest.raises(LabelError, match='re-run'):
            load_annotations(tmp_path, 'f', WIDTH, HEIGHT)

    def test_rows_with_different_classes_raise(self, tmp_path):
        write_pair(tmp_path, 'f', ['1 120 110 60 30'], ['0 100 100 140 100 140 120 100 120'])
        with pytest.raises(LabelError, match='classes differ'):
            load_annotations(tmp_path, 'f', WIDTH, HEIGHT)

    def test_class_filter_keeps_both_geometries_aligned(self, tmp_path):
        write_pair(
            tmp_path,
            'f',
            ['0 120 110 60 30', '3 300 300 20 20'],
            ['0 100 100 140 100 140 120 100 120', '3 290 290 310 290 310 310 290 310'],
        )
        labels = load_annotations(tmp_path, 'f', WIDTH, HEIGHT, classes=[3])
        assert labels.hbb.tolist() == [[300, 300, 20, 20]] and labels.classes.tolist() == [3]

    def test_an_unannotated_directory_has_no_objects(self, tmp_path):
        labels = load_annotations(tmp_path, 'f', WIDTH, HEIGHT)
        assert len(labels) == 0 and labels.object_corners('obb').shape == (0, 4, 2)


class TestRequireAnnotations:
    """Masks are optional: without both label directories only masking 'none' runs."""

    def test_both_directories_or_neither(self, tmp_path):
        assert not is_annotated(tmp_path)
        write_pair(tmp_path, 'f', [], [])
        assert is_annotated(tmp_path)

    @pytest.mark.parametrize('present', ['labels_hbb', 'labels_obb'])
    def test_one_directory_alone_raises(self, tmp_path, present):
        (tmp_path / present).mkdir()
        with pytest.raises(LabelError, match=present):
            is_annotated(tmp_path)

    def test_masking_none_runs_without_annotations(self, tmp_path):
        assert require_annotations(tmp_path, ['none', 'none']) is False

    def test_a_mask_without_annotations_raises_naming_the_modes(self, tmp_path):
        with pytest.raises(LabelError, match='hbb, obb'):
            require_annotations(tmp_path, ['none', 'obb', 'hbb'])

    def test_annotated_directories_allow_every_mode(self, tmp_path):
        write_pair(tmp_path, 'f', [], [])
        assert require_annotations(tmp_path, ['none', 'hbb', 'obb']) is True


class TestSummary:
    def test_reports_counts_and_areas(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 140 100 140 120 100 120'])
        summary = summarize(load_labels(path, WIDTH, HEIGHT, coords='absolute'))
        assert summary['n_objects'] == 1
        assert summary['classes'] == {0: 1}
        assert summary['total_box_area_px'] == pytest.approx(40 * 20)

    def test_empty_labels_summarize_to_zero(self):
        assert summarize(empty_labels())['n_objects'] == 0


class TestPolygonLabels:
    """Polygon input, reduced to its minimum-area rotated rectangle."""

    TRIANGLE = '0 100 50 200 50 200 100'

    def test_ragged_rows_are_read_as_polygons(self, tmp_path):
        # Three vertices on one line, five on the next: np.loadtxt cannot read this at all.
        path = write_labels(tmp_path / '000000.txt', [self.TRIANGLE, '1 300 300 360 300 370 340 330 370 290 340'])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute')
        assert len(labels) == 2
        assert labels.has_obb
        assert labels.obb.shape == (2, 8)

    def test_oriented_hull_is_the_minimum_area_rectangle(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', [self.TRIANGLE])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute')
        corners = labels.obb.reshape(4, 2)
        # Every vertex lies inside the rectangle, within cv2.minAreaRect's float32 tolerance.
        for vertex in ((100.0, 50.0), (200.0, 50.0), (200.0, 100.0)):
            assert cv2.pointPolygonTest(corners.astype(np.float32), vertex, True) >= -1e-3

    def test_horizontal_box_comes_from_the_polygon_not_from_its_oriented_hull(self, tmp_path):
        # The HBB is the polygon's extent, not that of its rotated rectangle.
        path = write_labels(tmp_path / '000000.txt', [self.TRIANGLE])
        labels = load_labels(path, WIDTH, HEIGHT, coords='absolute')
        assert np.allclose(labels.hbb, [[150, 75, 100, 50]])

    def test_polygons_are_denormalized_like_boxes(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 0.1 0.1 0.2 0.1 0.2 0.2'])
        labels = load_labels(path, 1000, 500)
        assert np.allclose(labels.hbb, [[150, 75, 100, 50]])

    def test_explicit_polygon_format_rejects_a_box_file(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20'])
        with pytest.raises(LabelError, match='polygon'):
            load_labels(path, WIDTH, HEIGHT, label_format='polygon')

    def test_ragged_rows_are_rejected_under_an_explicit_box_format(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 960 540 40 20', '0 1 2 3 4 5 6 7 8'])
        with pytest.raises(LabelError, match='fixed column count'):
            load_labels(path, WIDTH, HEIGHT, label_format='hbb')


class TestOrientedBoxesWithAngle:
    def test_xywha_recovers_the_rotated_extents(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 40 10 45'])
        labels = load_labels(path, WIDTH, HEIGHT, label_format='xywha', coords='absolute')
        corners = labels.obb.reshape(4, 2)
        assert np.isclose(np.linalg.norm(corners[1] - corners[0]), 40.0)
        assert np.isclose(np.linalg.norm(corners[2] - corners[1]), 10.0)
        assert np.allclose(corners.mean(axis=0), [100, 100])

    def test_zero_angle_matches_a_horizontal_box(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 40 10 0'])
        labels = load_labels(path, WIDTH, HEIGHT, label_format='xywha', coords='absolute')
        assert np.allclose(labels.hbb, [[100, 100, 40, 10]])


class TestConfidenceFiltering:
    ROWS = ['0 100 100 40 20 0.9', '0 200 200 40 20 0.4']

    def test_threshold_drops_low_scoring_boxes(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', self.ROWS)
        assert len(load_labels(path, WIDTH, HEIGHT, coords='absolute')) == 2
        kept = load_labels(path, WIDTH, HEIGHT, coords='absolute', min_confidence=0.5)
        assert len(kept) == 1
        assert np.allclose(kept.hbb, [[100, 100, 40, 20]])

    def test_threshold_on_unscored_labels_raises_rather_than_doing_nothing(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', ['0 100 100 40 20'])
        with pytest.raises(LabelError, match='carries no confidence column'):
            load_labels(path, WIDTH, HEIGHT, coords='absolute', min_confidence=0.5)

    def test_class_filter_and_threshold_compose(self, tmp_path):
        path = write_labels(tmp_path / '000000.txt', self.ROWS + ['1 300 300 40 20 0.95'])
        kept = load_labels(path, WIDTH, HEIGHT, coords='absolute', classes=[0], min_confidence=0.5)
        assert kept.classes.tolist() == [0]


class TestObjectCorners:
    """The boxes IoU and OCE are scored on, with no fallback."""

    def test_obb_is_the_oriented_box_and_hbb_the_detectors_box(self, tmp_path):
        write_labels(tmp_path / 'b.txt', ['0 110 105 20 10'])
        write_labels(tmp_path / 'o.txt', ['0 100 100 120 100 120 110 100 110'])
        horizontal = load_labels(tmp_path / 'b.txt', WIDTH, HEIGHT, coords='absolute')
        oriented = load_labels(tmp_path / 'o.txt', WIDTH, HEIGHT, coords='absolute')
        labels = type(oriented)(classes=oriented.classes, hbb=horizontal.hbb, obb=oriented.obb)
        assert np.allclose(labels.object_corners('obb'), oriented.obb.reshape(-1, 4, 2))
        assert np.allclose(labels.object_corners('hbb'), hbb_to_corners(horizontal.hbb))

    def test_obb_without_oriented_labels_raises_instead_of_scoring_horizontal_boxes(self, tmp_path):
        labels = load_labels(write_labels(tmp_path / 'b.txt', ['0 110 105 20 10']), WIDTH, HEIGHT, coords='absolute')
        with pytest.raises(LabelError, match='object_boxes'):
            labels.object_corners('obb')
        assert labels.object_corners('hbb').shape == (1, 4, 2)

    def test_an_unknown_geometry_is_named(self, tmp_path):
        labels = load_labels(write_labels(tmp_path / 'b.txt', ['0 110 105 20 10']), WIDTH, HEIGHT, coords='absolute')
        with pytest.raises(LabelError, match='polygon'):
            labels.object_corners('polygon')
