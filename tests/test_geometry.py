#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Geometric metrics: corner transfer error, HEA, object IoU and the object center error against closed forms."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stabilo_benchmark.masking import hbb_to_corners
from stabilo_benchmark.metrics.diagnostics import (
    capture_fraction,
    estimation_failed,
    group_error,
    inlier_object_shares,
)
from stabilo_benchmark.metrics.geometric import (
    OBJECT_STATISTICS,
    box_centers,
    corner_disagreement,
    corner_transfer,
    corner_transfer_error,
    frame_corners,
    geometric_scores,
    hea,
    homography_frobenius_error,
    maa,
    nan_object_scores,
    object_center_error,
    object_iou,
    quad_iou,
    summarize_objects,
    threshold_label,
)
from stabilo_benchmark.regions import warp_points

WIDTH, HEIGHT = 640, 480


def rotation(degrees: float, scale: float = 1.0, center=(WIDTH / 2, HEIGHT / 2)) -> np.ndarray:
    return np.vstack([cv2.getRotationMatrix2D(center, degrees, scale), [0.0, 0.0, 1.0]])


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def rotated_quad(cx: float, cy: float, w: float, h: float, degrees: float) -> np.ndarray:
    """A (4, 2) oriented box, rotated about its own center."""
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (local @ np.array([[cosine, sine], [-sine, cosine]])) + np.array([cx, cy])


class TestCornerTransfer:
    def test_exact_inverse_gives_zero_error(self):
        matrix = rotation(7.0, 1.02)
        error = corner_transfer_error(matrix, np.linalg.inv(matrix), WIDTH, HEIGHT)
        assert error['cte_mean'] < 1e-9
        assert error['cte_max'] < 1e-9

    def test_identity_pair_is_exact(self):
        error = corner_transfer_error(np.eye(3), np.eye(3), WIDTH, HEIGHT)
        assert error == {'cte_mean': 0.0, 'cte_max': 0.0, 'cte_rms': 0.0}

    def test_pure_translation_error_equals_its_magnitude(self):
        # A round trip that ends 5 px right of where it started must report exactly 5 px,
        # identically at every corner.
        matrix = rotation(11.0)
        estimate = translation(5.0, 0.0) @ np.linalg.inv(matrix)
        error = corner_transfer_error(matrix, estimate, WIDTH, HEIGHT)
        assert error['cte_mean'] == pytest.approx(5.0, abs=1e-9)
        assert error['cte_max'] == pytest.approx(5.0, abs=1e-9)
        assert error['cte_rms'] == pytest.approx(5.0, abs=1e-9)

    def test_returns_displacement_vectors_not_scalars(self):
        matrix = rotation(3.0)
        displacement = corner_transfer(matrix, translation(2.0, -3.0) @ np.linalg.inv(matrix), WIDTH, HEIGHT)
        assert displacement.shape == (4, 2)
        assert np.allclose(displacement, np.array([[2.0, -3.0]] * 4), atol=1e-9)

    def test_non_finite_estimate_yields_nan(self):
        estimate = np.full((3, 3), np.nan)
        error = corner_transfer_error(np.eye(3), estimate, WIDTH, HEIGHT)
        assert all(np.isnan(value) for value in error.values())

    def test_corner_disagreement_handles_same_direction_transforms(self):
        # Both map current -> reference; a 4 px shift reports 4 px.
        reference = rotation(5.0, 1.01)
        candidate = translation(4.0, 0.0) @ reference
        error = corner_disagreement(reference, candidate, WIDTH, HEIGHT)
        assert error['cte_mean'] == pytest.approx(4.0, abs=1e-9)

    def test_corner_disagreement_of_identical_transforms_is_zero(self):
        matrix = rotation(-13.0, 0.97)
        assert corner_disagreement(matrix, matrix, WIDTH, HEIGHT)['cte_mean'] < 1e-9

    def test_singular_transform_yields_nan(self):
        singular = np.zeros((3, 3))
        assert np.isnan(corner_disagreement(singular, np.eye(3), WIDTH, HEIGHT)['cte_mean'])


class TestHea:
    def test_counts_fraction_below_threshold(self):
        assert hea([0.5, 2.0, 7.0], [1.0])['hea_1px'] == pytest.approx(1 / 3)
        assert hea([0.5, 2.0, 7.0], [3.0])['hea_3px'] == pytest.approx(2 / 3)

    def test_nan_counts_as_a_failure_not_a_dropped_sample(self):
        assert hea([0.5, np.nan], [1.0])['hea_1px'] == pytest.approx(0.5)
        assert hea([np.nan, np.nan], [100.0])['hea_100px'] == 0.0

    def test_empty_input_is_nan(self):
        assert np.isnan(hea([], [1.0])['hea_1px'])

    def test_threshold_labels_are_filesystem_safe(self):
        assert threshold_label(1.0) == '1px'
        assert threshold_label(2.5) == '2p5px'
        assert threshold_label(10) == '10px'


class TestMaa:
    def test_agrees_with_the_integral_of_the_hea_curve(self):
        # The closed form matches numerical quadrature of the HEA curve.
        errors = [0.3, 1.7, 2.9, 4.4, np.nan]
        grid = np.linspace(0.0, 5.0, 5001)
        curve = [hea(errors, [t])[f"hea_{threshold_label(t)}"] for t in grid]
        assert maa(errors, upper=5.0) == pytest.approx(float(np.trapezoid(curve, grid)) / 5.0, abs=1e-3)

    def test_perfect_and_hopeless_estimates_bound_the_range(self):
        assert maa([0.0, 0.0], upper=5.0) == pytest.approx(1.0)
        assert maa([50.0, 90.0], upper=5.0) == pytest.approx(0.0)

    def test_nan_counts_as_a_failure_not_a_dropped_sample(self):
        assert maa([0.0, np.nan], upper=5.0) == pytest.approx(0.5)

    def test_floor_discards_area_below_the_resolution_limit(self):
        # Errors that differ only below the lower limit score identically...
        assert maa([0.1] * 4, upper=5.0, lower=1.0) == pytest.approx(maa([0.9] * 4, upper=5.0, lower=1.0))
        # ... and differ from zero.
        assert maa([0.1] * 4, upper=5.0) > maa([0.9] * 4, upper=5.0)

    def test_empty_input_is_nan(self):
        assert np.isnan(maa([], upper=5.0))

    def test_rejects_a_degenerate_range(self):
        with pytest.raises(ValueError, match='upper > lower'):
            maa([1.0], upper=2.0, lower=2.0)


class TestFrobenius:
    def test_exact_recovery_is_zero(self):
        matrix = rotation(9.0, 1.03)
        assert homography_frobenius_error(matrix, np.linalg.inv(matrix)) == pytest.approx(0.0, abs=1e-9)

    def test_scale_invariant(self):
        matrix = rotation(9.0)
        estimate = np.linalg.inv(matrix)
        assert homography_frobenius_error(matrix, estimate) == pytest.approx(
            homography_frobenius_error(matrix, 3.7 * estimate), abs=1e-9
        )

    def test_singular_truth_is_nan(self):
        assert np.isnan(homography_frobenius_error(np.zeros((3, 3)), np.eye(3)))


class TestBoxIou:
    def test_identical_boxes_score_one(self):
        boxes = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0]]))
        assert quad_iou(boxes, boxes)[0] == pytest.approx(1.0)

    def test_shifted_axis_aligned_boxes_match_the_closed_form(self):
        a = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0]]))
        b = hbb_to_corners(np.array([[110.0, 100.0, 40.0, 20.0]]))
        assert quad_iou(a, b)[0] == pytest.approx(600.0 / 1000.0, abs=1e-9)

    def test_disjoint_boxes_score_zero(self):
        a = hbb_to_corners(np.array([[10.0, 10.0, 4.0, 4.0]]))
        b = hbb_to_corners(np.array([[900.0, 900.0, 4.0, 4.0]]))
        assert quad_iou(a, b)[0] == 0.0

    def test_a_projectively_distorted_box_is_compared_as_it_is(self):
        # The IoU is the carried quadrilateral's, with no rectangle refitted.
        box = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0]]))
        trapezoid = box.copy()
        trapezoid[0, 1, 0] += 10.0  # stretch the top-right corner only: area grows by 100
        assert quad_iou(box, trapezoid)[0] == pytest.approx(800.0 / 900.0, abs=1e-9)

    def test_degenerate_quadrilateral_scores_zero_without_raising(self):
        collapsed = np.zeros((1, 4, 2))
        assert quad_iou(hbb_to_corners(np.array([[10.0, 10.0, 4.0, 4.0]])), collapsed)[0] == 0.0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match='same shape'):
            quad_iou(np.zeros((2, 4, 2)), np.zeros((3, 4, 2)))

    def test_object_iou_reports_the_distribution_not_only_the_mean(self):
        a = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0], [200.0, 200.0, 40.0, 20.0]]))
        b = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0], [230.0, 200.0, 40.0, 20.0]]))
        scores = object_iou(a, b)
        assert scores['n_objects'] == 2
        assert scores['iou_max'] == pytest.approx(1.0)
        assert scores['iou_min'] == pytest.approx(10.0 * 20.0 / (2 * 800.0 - 200.0))
        assert scores['iou_min'] < scores['iou_mean'] < scores['iou_max']
        assert set(scores) == {f"iou_{name}" for name in OBJECT_STATISTICS} | {'n_objects'}

    def test_object_iou_of_no_boxes_is_nan(self):
        empty = np.zeros((0, 4, 2))
        scores = object_iou(empty, empty)
        assert scores['n_objects'] == 0
        assert all(np.isnan(scores[f"iou_{name}"]) for name in OBJECT_STATISTICS)


class TestObjectStatistics:
    def test_statistics_match_numpy(self):
        values = np.array([4.0, 1.0, 3.0, 2.0, 10.0])
        scores = summarize_objects(values, 'x')
        assert list(scores) == [f"x_{name}" for name in OBJECT_STATISTICS]
        assert scores['x_mean'] == pytest.approx(4.0)
        assert scores['x_std'] == pytest.approx(np.std(values))
        assert (scores['x_min'], scores['x_max']) == (1.0, 10.0)
        assert (scores['x_q1'], scores['x_median'], scores['x_q3']) == pytest.approx((2.0, 3.0, 4.0))

    def test_a_single_object_has_zero_spread(self):
        scores = summarize_objects(np.array([0.7]), 'iou')
        assert scores['iou_std'] == 0.0
        assert scores['iou_q1'] == scores['iou_q3'] == 0.7

    def test_a_failed_trial_is_nan_for_every_statistic(self):
        scores = nan_object_scores()
        assert set(scores) == {f"{m}_{name}" for m in ('iou', 'oce') for name in OBJECT_STATISTICS}
        assert all(np.isnan(value) for value in scores.values())


class TestObjectCenterError:
    def test_the_center_of_an_oriented_box_is_its_own(self):
        quad = rotated_quad(120.0, 80.0, 40.0, 10.0, 30.0)
        assert box_centers(quad[None]) == pytest.approx(np.array([[120.0, 80.0]]))

    def test_a_translation_error_moves_every_center_by_that_much(self):
        boxes = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0], [500.0, 300.0, 10.0, 30.0]]))
        scores = object_center_error(boxes, boxes + np.array([3.0, 4.0]))
        assert scores['oce_mean'] == pytest.approx(5.0)
        assert scores['oce_max'] == pytest.approx(5.0)
        assert scores['oce_std'] == pytest.approx(0.0)

    def test_it_does_not_depend_on_the_object_size(self):
        # IoU depends on box size; OCE does not.
        small = hbb_to_corners(np.array([[100.0, 100.0, 8.0, 4.0]]))
        large = hbb_to_corners(np.array([[100.0, 100.0, 80.0, 40.0]]))
        shift = np.array([2.0, 0.0])
        assert object_iou(small, small + shift)['iou_mean'] < object_iou(large, large + shift)['iou_mean']
        assert object_center_error(small, small + shift) == object_center_error(large, large + shift)

    def test_a_collapsing_estimate_is_nan_not_a_distance(self):
        box = hbb_to_corners(np.array([[100.0, 100.0, 40.0, 20.0]]))
        broken = box.copy()
        broken[0, 2] = np.inf
        assert np.isnan(object_center_error(box, broken)['oce_mean'])
        assert np.isnan(object_center_error(np.zeros((0, 4, 2)), np.zeros((0, 4, 2)))['oce_max'])


class TestObjectScoresOfATrial:
    """`geometric_scores` round-trips the scene's oriented boxes through the true matrix and the estimate."""

    boxes = np.array([rotated_quad(200.0, 150.0, 40.0, 10.0, 35.0), rotated_quad(450.0, 300.0, 60.0, 25.0, -70.0)])
    true_matrix = np.array([[1.02, 0.03, 12.0], [-0.02, 0.99, -7.0], [2e-5, -1e-5, 1.0]])

    def test_a_perfect_estimate_scores_one_and_zero(self):
        scores = geometric_scores(self.true_matrix, np.linalg.inv(self.true_matrix), WIDTH, HEIGHT, self.boxes)
        assert scores['iou_mean'] == pytest.approx(1.0, abs=1e-9)
        assert scores['oce_mean'] == pytest.approx(0.0, abs=1e-9)
        assert not any(key.startswith(('miou', 'center_error')) for key in scores)

    def test_both_scores_read_the_same_carried_boxes(self):
        estimate = translation(2.0, -1.0) @ np.linalg.inv(self.true_matrix)
        scores = geometric_scores(self.true_matrix, estimate, WIDTH, HEIGHT, self.boxes)
        assert scores['oce_mean'] == pytest.approx(np.hypot(2.0, 1.0), abs=1e-9)
        assert scores['oce_max'] == pytest.approx(np.hypot(2.0, 1.0), abs=1e-9)
        assert 0.0 < scores['iou_mean'] < 1.0

    def test_where_the_center_is_read_does_not_matter_measurably(self):
        # The carried box's corner mean against the carried center point, under a projective
        # truth and an estimate that is off by a few pixels and a little perspective.
        estimate = np.array([[1.0, 2e-4, 3.0], [-1e-4, 1.0, -2.0], [3e-7, -2e-7, 1.0]]) @ np.linalg.inv(
            self.true_matrix
        )
        round_trip = warp_points(warp_points(self.boxes, self.true_matrix), estimate)
        centers = box_centers(self.boxes)
        carried_point = warp_points(warp_points(centers, self.true_matrix), estimate)
        assert np.max(np.abs(box_centers(round_trip) - carried_point)) < 1e-2

    def test_no_boxes_no_object_scores(self):
        scores = geometric_scores(self.true_matrix, np.linalg.inv(self.true_matrix), WIDTH, HEIGHT, None)
        assert 'iou_mean' not in scores and 'oce_mean' not in scores


class TestFrameCorners:
    def test_ordered_clockwise_from_top_left(self):
        assert np.array_equal(
            frame_corners(WIDTH, HEIGHT),
            np.array([[0.0, 0.0], [WIDTH, 0.0], [WIDTH, HEIGHT], [0.0, HEIGHT]]),
        )


class TestEstimationFailed:
    """A failure is stabilo's fallback (no inlier count) or a non-finite matrix, never a value."""

    class _Stabilizer:
        def __init__(self, inliers):
            self.inliers = inliers

        def get_cur_inliers_count(self):
            return self.inliers

    def test_the_benchmark_fallback_is_a_failure(self):
        assert estimation_failed(self._Stabilizer(None), np.eye(3))

    def test_an_exact_identity_with_inliers_is_an_estimate(self):
        assert not estimation_failed(self._Stabilizer(5000), np.eye(3))

    def test_a_missing_or_non_finite_matrix_is_a_failure(self):
        assert estimation_failed(self._Stabilizer(5000), None)
        assert estimation_failed(self._Stabilizer(5000), np.full((3, 3), np.nan))


class TestCaptureFraction:
    """How far an estimate follows a group that moved together, against the reference's ground."""

    # A group found 30 px to the right of where it was, with the platform's motion a small shift.
    REFERENCE = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    POINTS = np.array([[100.0, 50.0], [140.0, 52.0], [180.0, 49.0], [220.0, 51.0]])
    MOVES = np.tile([30.0, 0.0], (4, 1))

    def test_the_reference_itself_follows_nothing(self):
        assert capture_fraction(self.REFERENCE, self.REFERENCE, self.POINTS, self.MOVES) == pytest.approx(0.0)

    def test_an_estimate_fitted_to_the_group_follows_all_of_it(self):
        # It puts the moved objects back where they were: the reference's map minus their move.
        fitted = np.array([[1.0, 0.0, -34.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        assert capture_fraction(fitted, self.REFERENCE, self.POINTS, self.MOVES) == pytest.approx(1.0)

    def test_error_across_the_motion_is_not_capture(self):
        across = self.REFERENCE + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 25.0], [0.0, 0.0, 0.0]])
        assert capture_fraction(across, self.REFERENCE, self.POINTS, self.MOVES) == pytest.approx(0.0)

    def test_without_an_estimate_or_a_group_it_is_undefined(self):
        assert np.isnan(capture_fraction(None, self.REFERENCE, self.POINTS, self.MOVES))
        assert np.isnan(capture_fraction(self.REFERENCE, self.REFERENCE, np.zeros((0, 2)), np.zeros((0, 2))))

    @staticmethod
    def _shifted(dx, dy):
        matrix = np.eye(3)
        matrix[:2, 2] = dx, dy
        return matrix

    def test_a_fit_to_the_group_errs_along_its_displacement(self):
        points, moves = np.array([[10.0, 10.0], [50.0, 20.0], [90.0, 40.0]]), np.tile([6.0, 0.0], (3, 1))
        assert group_error(self._shifted(-6, 0), np.eye(3), points, moves) == pytest.approx((6.0, 0.0))

    def test_the_angle_reads_the_opposite_and_the_perpendicular_direction(self):
        points, moves = np.array([[10.0, 10.0], [50.0, 20.0]]), np.tile([6.0, 0.0], (2, 1))
        assert group_error(self._shifted(6, 0), np.eye(3), points, moves) == pytest.approx((6.0, 180.0))
        assert group_error(self._shifted(0, 6), np.eye(3), points, moves) == pytest.approx((6.0, 90.0))

    def test_no_estimate_group_or_error_has_no_direction(self):
        points, moves = np.array([[10.0, 10.0]]), np.array([[6.0, 0.0]])
        assert np.isnan(group_error(None, np.eye(3), points, moves)).all()
        assert np.isnan(group_error(np.eye(3), np.eye(3), np.zeros((0, 2)), np.zeros((0, 2)))).all()
        assert np.isnan(group_error(np.eye(3), np.eye(3), points, moves)).all()


class TestInlierObjectShares:
    class _Stabilizer:
        def __init__(self, points, inliers):
            self.ref_pts = np.asarray(points, dtype=np.float32)
            self.cur_inliers = np.asarray(inliers, dtype=bool).reshape(-1, 1)

    INDEX = np.full((10, 20), -1, dtype=np.int32)
    INDEX[2:5, 2:5] = 0
    INDEX[2:5, 12:15] = 1

    def test_inliers_are_counted_where_they_lie(self):
        points = [[3, 3], [13, 3], [8, 8], [1, 9], [3.5, 2.5]]
        shares = inlier_object_shares(self._Stabilizer(points, [1, 1, 1, 1, 0]), self.INDEX, np.array([False, True]))
        assert shares == {'inliers_on_objects': 0.5, 'inliers_on_moving': 0.25}

    def test_without_inliers_both_are_undefined(self):
        shares = inlier_object_shares(self._Stabilizer([[3, 3]], [0]), self.INDEX, np.array([True, True]))
        assert all(np.isnan(value) for value in shares.values())
