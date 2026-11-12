#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Metric regions: a metric looks only at pixels that received data from the warp, hold no object in
either frame, and lie inside the ground-plane ROI when one is given. Each term is checked on its
own.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stabilo_benchmark.masking import hbb_to_corners
from stabilo_benchmark.regions import (
    RegionError,
    dominant_motion,
    evaluation_mask,
    load_exclusion,
    load_roi,
    metric_region,
    motion_axis,
    object_region,
    rasterize_quads,
    region_is_usable,
    warp_points,
    warp_validity,
)

HEIGHT, WIDTH = 240, 320
SHAPE = (HEIGHT, WIDTH)


def box(cx, cy, w, h):
    return hbb_to_corners(np.array([[float(cx), float(cy), float(w), float(h)]]))


def rotated_box(cx, cy, w, h, degrees):
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (local @ np.array([[cosine, sine], [-sine, cosine]])) + np.array([cx, cy])


class TestRasterize:
    def test_fills_the_quadrilateral(self):
        mask = rasterize_quads(box(160, 120, 40, 20), SHAPE)
        assert mask[120, 160]
        assert not mask[10, 10]
        assert mask.sum() == pytest.approx(40 * 20, rel=0.1)

    def test_dilation_grows_the_region_monotonically(self):
        areas = [rasterize_quads(box(160, 120, 40, 20), SHAPE, dilation_px=d).sum() for d in (0, 4, 8, 16)]
        assert areas == sorted(areas)
        assert areas[0] < areas[-1]

    def test_dilation_is_a_constant_distance_not_proportional(self):
        # The margin is constant in pixels, not proportional to the box.
        small = rasterize_quads(box(80, 120, 20, 20), SHAPE, dilation_px=6).sum() - 20 * 20
        large = rasterize_quads(box(220, 120, 60, 60), SHAPE, dilation_px=6).sum() - 60 * 60
        assert large / small < 3.5  # perimeter ratio is 3x; a proportional margin would be 9x

    def test_empty_input_gives_an_empty_mask(self):
        assert not rasterize_quads(np.zeros((0, 4, 2)), SHAPE).any()

    def test_quad_outside_the_frame_is_clipped_not_an_error(self):
        assert not rasterize_quads(box(-500, -500, 20, 20), SHAPE).any()


class TestWarpValidity:
    def test_identity_leaves_the_whole_frame_valid_except_the_eroded_rim(self):
        valid = warp_validity(np.eye(3), SHAPE, SHAPE, erode_px=0)
        assert valid.all()

    def test_translation_invalidates_the_uncovered_strip(self):
        shift = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        valid = warp_validity(shift, SHAPE, SHAPE, erode_px=0)
        assert not valid[:, :100].any()
        assert valid[:, 100:].all()

    def test_erosion_removes_the_interpolation_fringe(self):
        assert (
            warp_validity(np.eye(3), SHAPE, SHAPE, erode_px=3).sum()
            < warp_validity(np.eye(3), SHAPE, SHAPE, erode_px=0).sum()
        )

    def test_affine_matrix_is_accepted(self):
        affine = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
        assert warp_validity(affine, SHAPE, SHAPE, erode_px=0)[:, 10:].all()

    def test_wrong_shape_raises(self):
        with pytest.raises(RegionError, match='3x3 or 2x3'):
            warp_validity(np.eye(4), SHAPE)


class TestVehicleRegion:
    def test_excludes_vehicles_from_both_frames(self):
        reference = box(80, 120, 30, 20)
        current = box(220, 120, 30, 20)
        mask = object_region(reference, current, np.eye(3), SHAPE, dilation_px=0)
        assert mask[120, 80]
        assert mask[120, 220]
        assert not mask[10, 10]

    def test_current_vehicles_are_placed_by_the_given_transform(self):
        current = box(100, 120, 30, 20)
        shift = np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mask = object_region(None, current, shift, SHAPE, dilation_px=0)
        assert mask[120, 180]
        assert not mask[120, 100]

    def test_no_vehicles_gives_an_empty_mask(self):
        assert not object_region(None, None, None, SHAPE).any()


class TestMetricRegion:
    def test_all_three_terms_are_applied(self):
        reference = box(60, 120, 30, 20)
        current = box(260, 120, 30, 20)
        shift = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        region, stats = metric_region(
            shape=SHAPE,
            homography=shift,
            reference_corners=reference,
            current_corners=current,
            mask_transform=np.eye(3),
            dilation_px=0,
        )
        assert not region[120, 10]  # outside the warp's valid area
        assert not region[120, 60]  # reference-frame vehicle
        assert not region[120, 260]  # current-frame vehicle
        assert region[60, 200]  # plain background
        assert 0.0 < stats.area_fraction < 1.0
        assert stats.object_fraction > 0.0

    def test_roi_restricts_the_region(self):
        roi = np.zeros(SHAPE, dtype=bool)
        roi[100:140, 100:200] = True
        region, _ = metric_region(shape=SHAPE, homography=np.eye(3), roi=roi, dilation_px=0)
        assert region[120, 150]
        assert not region[10, 10]

    def test_mismatched_roi_shape_raises(self):
        with pytest.raises(RegionError, match='does not match'):
            metric_region(shape=SHAPE, homography=np.eye(3), roi=np.ones((10, 10), dtype=bool))

    def test_border_erosion_pulls_the_region_off_the_frame_edge(self):
        without, _ = metric_region(shape=SHAPE, homography=np.eye(3), border_px=0)
        with_border, _ = metric_region(shape=SHAPE, homography=np.eye(3), border_px=10)
        assert with_border.sum() < without.sum()
        assert not with_border[0, :].any()

    def test_stats_report_the_surviving_fraction(self):
        _, stats = metric_region(shape=SHAPE, homography=np.eye(3), dilation_px=0)
        assert stats.area_px == pytest.approx(HEIGHT * WIDTH, rel=0.05)
        assert 'region_area_fraction' in stats.to_dict()

    def test_a_collapsed_region_is_flagged_unusable(self):
        collapse = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]])
        _, stats = metric_region(shape=SHAPE, homography=collapse)
        assert not region_is_usable(stats, minimum_fraction=0.05)

    def test_a_full_region_is_usable(self):
        _, stats = metric_region(shape=SHAPE, homography=np.eye(3))
        assert region_is_usable(stats)


class TestRoiLoading:
    def test_text_polygon(self, tmp_path):
        path = tmp_path / 'roi.txt'
        path.write_text('10 10\n200 10\n200 150\n10 150\n', encoding='utf-8')
        roi = load_roi(path, SHAPE)
        assert roi[80, 100]
        assert not roi[200, 300]

    def test_labelme_polygon(self, tmp_path):
        path = tmp_path / 'roi.json'
        path.write_text(
            json.dumps(
                {
                    'shapes': [
                        {
                            'label': 'ground',
                            'shape_type': 'polygon',
                            'points': [[10, 10], [200, 10], [200, 150], [10, 150]],
                        }
                    ]
                }
            ),
            encoding='utf-8',
        )
        assert load_roi(path, SHAPE)[80, 100]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RegionError, match='not found'):
            load_roi(tmp_path / 'absent.json', SHAPE)

    def test_json_without_polygons_raises(self, tmp_path):
        path = tmp_path / 'roi.json'
        path.write_text(json.dumps({'shapes': [{'shape_type': 'point', 'points': [[1, 1]]}]}), encoding='utf-8')
        with pytest.raises(RegionError, match='No polygon'):
            load_roi(path, SHAPE)

    def test_polygon_missing_the_frame_raises(self, tmp_path):
        path = tmp_path / 'roi.txt'
        path.write_text('-50 -50\n-10 -50\n-10 -10\n-50 -10\n', encoding='utf-8')
        with pytest.raises(RegionError, match='covers no pixels'):
            load_roi(path, SHAPE)


class TestTrafficAxis:
    def test_single_orientation_is_recovered_exactly(self):
        boxes = np.array([rotated_box(40 * i, 100, 40, 18, 30) for i in range(8)])
        axis, concentration = motion_axis(boxes)
        assert np.degrees(np.arctan2(axis[1], axis[0])) == pytest.approx(30.0, abs=0.5)
        assert concentration > 0.99

    def test_survives_the_plus_minus_ninety_degree_wrap(self):
        # Axial mean across the +/-90 degree wrap.
        boxes = np.array([rotated_box(40 * i, 100, 40, 18, a) for i, a in enumerate([88, 89, 90, -89, -88])])
        axis, concentration = motion_axis(boxes)
        assert abs(np.degrees(np.arctan2(axis[1], axis[0]))) == pytest.approx(90.0, abs=2.0)
        assert concentration > 0.99

    def test_crossroads_reports_low_concentration(self):
        # Two perpendicular flows give a low concentration.
        boxes = np.concatenate(
            [
                np.array([rotated_box(30 * i, 100, 40, 18, 0) for i in range(8)]),
                np.array([rotated_box(100, 30 * i, 40, 18, 90) for i in range(8)]),
            ]
        )
        assert motion_axis(boxes)[1] < 0.05

    def test_no_boxes_gives_no_axis(self):
        assert motion_axis(np.zeros((0, 4, 2))) == (None, 0.0)

    def test_degenerate_boxes_give_no_axis(self):
        assert motion_axis(np.zeros((3, 4, 2))) == (None, 0.0)


class TestDominantMotion:
    def test_mean_displacement_of_corresponded_objects(self):
        reference = np.concatenate([box(50, 100, 20, 10), box(150, 100, 20, 10)])
        current = np.concatenate([box(60, 100, 20, 10), box(170, 100, 20, 10)])
        motion = dominant_motion(reference, current, matches=[(0, 0), (1, 1)])
        assert np.allclose(motion, [15.0, 0.0])

    def test_no_matches_gives_none(self):
        assert dominant_motion(box(1, 1, 1, 1), box(2, 2, 1, 1), matches=[]) is None


class TestWarpPoints:
    def test_preserves_the_leading_shape(self):
        corners = np.zeros((5, 4, 2))
        assert warp_points(corners, np.eye(3)).shape == (5, 4, 2)

    def test_identity_is_a_no_op(self):
        points = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert np.allclose(warp_points(points, np.eye(3)), points)

    def test_matches_opencv_for_a_rotation(self):
        matrix = np.vstack([cv2.getRotationMatrix2D((100.0, 100.0), 30.0, 1.0), [0, 0, 1]])
        points = np.array([[10.0, 20.0], [150.0, 90.0]])
        expected = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
        assert np.allclose(warp_points(points, matrix), expected, atol=1e-4)

    def test_empty_input_returns_empty(self):
        assert warp_points(np.zeros((0, 2)), np.eye(3)).shape == (0, 2)


class TestOverlappingQuadsDoNotCancel:
    """
    Overlapping quads stay filled.

    `cv2.fillPoly` applies the even-odd rule across a list of polygons, so overlaps would cancel.
    """

    HALF_OVERLAPPING = np.array(
        [[[0, 0], [64, 0], [64, 48], [0, 48]], [[32, 0], [96, 0], [96, 48], [32, 48]]], dtype=float
    )

    def test_the_shared_pixels_stay_filled(self):
        mask = rasterize_quads(self.HALF_OVERLAPPING, (480, 640))
        assert mask[24, 48], 'a pixel inside both quads must be inside the mask'

    def test_the_area_is_the_union(self):
        # 97 x 49 with inclusive edges, not 2 x (65 x 49) and not the even-odd difference.
        mask = rasterize_quads(self.HALF_OVERLAPPING, (480, 640))
        assert int(np.count_nonzero(mask)) == 97 * 49

    def test_fully_coincident_quads_are_one_quad(self):
        single = rasterize_quads(self.HALF_OVERLAPPING[:1], (480, 640))
        doubled = rasterize_quads(np.concatenate([self.HALF_OVERLAPPING[:1]] * 2), (480, 640))
        assert np.array_equal(single, doubled)

    def test_the_metric_region_excludes_pixels_under_two_vehicles(self):
        region = object_region(self.HALF_OVERLAPPING, self.HALF_OVERLAPPING, np.eye(3), (480, 640), dilation_px=0)
        assert region[24, 48], 'a pixel under two vehicles is still a vehicle pixel'


def _labelme(path, *polygons):
    shapes = [{'label': 'structure', 'shape_type': 'polygon', 'points': polygon} for polygon in polygons]
    path.write_text(json.dumps({'shapes': shapes}), encoding='utf-8')
    return path


class TestExclusion:
    SQUARE = [[10, 10], [110, 10], [110, 110], [10, 110]]
    SHIFTED = [[60, 60], [160, 60], [160, 160], [60, 160]]

    def test_overlapping_polygons_are_unioned_not_cancelled(self, tmp_path):
        path = _labelme(tmp_path / 'polygons.json', self.SQUARE, self.SHIFTED)
        for loader in (load_roi, load_exclusion):
            mask = loader(path, SHAPE)
            assert mask[80, 80]  # inside both polygons
            assert mask[30, 30] and mask[140, 140]

    def test_the_exclusion_is_removed_from_the_whole_frame(self, tmp_path):
        exclusion = load_exclusion(_labelme(tmp_path / 'e.json', self.SQUARE), SHAPE)
        mask = evaluation_mask(SHAPE, exclusion=exclusion)
        assert not mask[50, 50] and mask[200, 300]

    def test_the_exclusion_is_removed_from_the_roi(self, tmp_path):
        roi = load_roi(_labelme(tmp_path / 'roi.json', [[0, 0], [200, 0], [200, 200], [0, 200]]), SHAPE)
        exclusion = load_exclusion(_labelme(tmp_path / 'e.json', self.SQUARE), SHAPE)
        mask = evaluation_mask(SHAPE, roi, exclusion)
        assert not mask[50, 50] and mask[150, 150] and not mask[200, 300]

    def test_a_file_without_polygons_excludes_nothing(self, tmp_path):
        path = _labelme(tmp_path / 'e.json')
        assert not load_exclusion(path, SHAPE).any()
        with pytest.raises(RegionError, match='No polygon'):
            load_roi(path, SHAPE)

    def test_no_files_leave_the_region_unrestricted(self):
        assert evaluation_mask(SHAPE) is None

    def test_an_exclusion_covering_the_whole_roi_raises(self, tmp_path):
        roi = load_roi(_labelme(tmp_path / 'roi.json', self.SQUARE), SHAPE)
        with pytest.raises(RegionError, match='covers no pixels'):
            evaluation_mask(SHAPE, roi, roi)

    def test_the_region_shrinks_by_the_excluded_area(self, tmp_path):
        _, full = metric_region(shape=SHAPE, homography=np.eye(3))
        exclusion = load_exclusion(_labelme(tmp_path / 'e.json', self.SQUARE), SHAPE)
        _, reduced = metric_region(shape=SHAPE, homography=np.eye(3), roi=evaluation_mask(SHAPE, exclusion=exclusion))
        assert reduced.area_px == full.area_px - int(exclusion.sum())
