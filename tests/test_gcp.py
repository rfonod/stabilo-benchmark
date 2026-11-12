#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Ground control points and the audit arithmetic, checked against closed forms: an exact homography
gives a zero residual, and an injected displacement appears at its own magnitude.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stabilo_benchmark.gcp import (
    GcpError,
    GroundControlPoints,
    annotation_repeatability,
    fit_homography,
    load_gcp_file,
    load_gcp_set,
    proposal_review,
    propose_points,
    residuals,
    save_gcp_file,
    summarize_residuals,
)
from stabilo_benchmark.regions import warp_points
from stabilo_benchmark.report.annotate_gcp import JITTER_PX, Panel, audit_frames, jitter, predict_positions
from stabilo_benchmark.results import ResultStore, build_manifest
from stabilo_benchmark.runners.audit import (
    audit_homographies,
    audit_results,
    fitted_transforms,
    score_against_gcp_transform,
    summarize_by_offset,
    synthetic_check,
)

POINTS = np.array([[100.0, 100.0], [500.0, 120.0], [480.0, 400.0], [120.0, 380.0], [300.0, 250.0]])


def homography(degrees: float = 5.0, scale: float = 1.01, shift=(12.0, -7.0)) -> np.ndarray:
    matrix = np.vstack([cv2.getRotationMatrix2D((320.0, 240.0), degrees, scale), [0.0, 0.0, 1.0]])
    return np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]]) @ matrix


def write_labelme(path, points):
    path.write_text(
        json.dumps(
            {
                'shapes': [
                    {'label': label, 'shape_type': 'point', 'points': [[float(p[0]), float(p[1])]]}
                    for label, p in points.items()
                ]
            }
        ),
        encoding='utf-8',
    )
    return path


class TestFittedTransform:
    def test_recovers_a_known_homography_from_the_points(self):
        matrix = homography()
        many = np.vstack([POINTS, POINTS + [[50.0, 30.0]], POINTS + [[-40.0, 60.0]]])
        labels = tuple(f"p{i}" for i in range(len(many)))
        current = GroundControlPoints(frame_index=1, labels=labels, points=many)
        reference = GroundControlPoints(frame_index=0, labels=labels, points=warp_points(many, matrix))
        fit = fit_homography(reference, current)
        assert fit['n_points'] == 15 and fit['well_determined']
        # OpenCV's least-squares fit is good to ~1e-5 px here, five orders below a click.
        assert fit['fit_rms_px'] < 1e-3
        assert np.allclose(warp_points(many, fit['matrix']), warp_points(many, matrix), atol=1e-3)

    def test_fewer_than_four_shared_points_give_no_transform(self):
        labels = ('a', 'b', 'c')
        current = GroundControlPoints(frame_index=1, labels=labels, points=POINTS[:3])
        reference = GroundControlPoints(frame_index=0, labels=labels, points=POINTS[:3])
        assert fit_homography(reference, current)['matrix'] is None

    def test_candidates_are_scored_against_the_fitted_transform(self):
        matrix = homography()
        many = np.vstack([POINTS, POINTS + [[50.0, 30.0]]])
        labels = tuple(f"p{i}" for i in range(len(many)))
        points = {
            0: GroundControlPoints(frame_index=0, labels=labels, points=warp_points(many, matrix)),
            1500: GroundControlPoints(frame_index=1500, labels=labels, points=many),
        }
        fitted = fitted_transforms(points, 0)
        exact = score_against_gcp_transform({'000000->001500': matrix}, fitted, 0, None, {'width': 640, 'height': 480})
        assert exact['summary']['n_frames'] == 1 and exact['summary']['cte_median_px'] < 1e-3
        shifted = matrix.copy()
        shifted[0, 2] += 3.0
        off = score_against_gcp_transform({'000000->001500': shifted}, fitted, 0, None, {'width': 640, 'height': 480})
        assert off['summary']['cte_median_px'] == pytest.approx(3.0, abs=1e-3)

    def test_repeatability_pools_shared_points_of_shared_frames(self):
        labels = ('a', 'b', 'c')
        first = {0: GroundControlPoints(frame_index=0, labels=labels, points=POINTS[:3])}
        second = {0: GroundControlPoints(frame_index=0, labels=labels, points=POINTS[:3] + [[1.0, 0.0]])}
        summary = annotation_repeatability(first, second)
        assert summary['n_points'] == 3 and summary['residual_p95_px'] == pytest.approx(1.0)


class TestLoading:
    def test_labelme_points_load(self, tmp_path):
        path = write_labelme(tmp_path / '000000.json', {'a': (10, 20), 'b': (30, 40)})
        entry = load_gcp_file(path)
        assert entry.frame_index == 0
        assert set(entry.labels) == {'a', 'b'}
        assert np.allclose(entry.as_dict()['a'], [10, 20])

    def test_text_points_load(self, tmp_path):
        path = tmp_path / '001500.txt'
        path.write_text('# comment\nlane_a 10 20\nmanhole 30 40\n', encoding='utf-8')
        entry = load_gcp_file(path)
        assert entry.frame_index == 1500
        assert entry.labels == ('lane_a', 'manhole')

    def test_frame_index_comes_from_the_filename(self, tmp_path):
        path = write_labelme(tmp_path / '003000.json', {'a': (1, 2)})
        assert load_gcp_file(path).frame_index == 3000

    def test_non_numeric_filename_raises(self, tmp_path):
        path = write_labelme(tmp_path / 'reference.json', {'a': (1, 2)})
        with pytest.raises(GcpError, match='frame index'):
            load_gcp_file(path)

    def test_unlabeled_point_raises(self, tmp_path):
        path = write_labelme(tmp_path / '000000.json', {'': (1, 2)})
        with pytest.raises(GcpError, match='unlabeled point'):
            load_gcp_file(path)

    def test_duplicate_label_within_a_frame_raises(self, tmp_path):
        path = tmp_path / '000000.txt'
        path.write_text('a 1 2\na 3 4\n', encoding='utf-8')
        with pytest.raises(GcpError, match='reuses label'):
            load_gcp_file(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / '000000.txt'
        path.write_text('', encoding='utf-8')
        with pytest.raises(GcpError, match='No point annotations'):
            load_gcp_file(path)

    def test_malformed_text_line_raises(self, tmp_path):
        path = tmp_path / '000000.txt'
        path.write_text('a 1\n', encoding='utf-8')
        with pytest.raises(GcpError, match="expected 'label x y'"):
            load_gcp_file(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(GcpError, match='not found'):
            load_gcp_file(tmp_path / '000000.json')

    def test_set_loads_keyed_by_frame(self, tmp_path):
        write_labelme(tmp_path / '000000.json', {'a': (1, 2)})
        write_labelme(tmp_path / '001500.json', {'a': (3, 4)})
        assert set(load_gcp_set(tmp_path)) == {0, 1500}

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(GcpError, match='No ground control point files'):
            load_gcp_set(tmp_path)

    def test_two_files_for_one_frame_raise(self, tmp_path):
        write_labelme(tmp_path / '000000.json', {'a': (1, 2)})
        (tmp_path / '000000.txt').write_text('a 1 2\n', encoding='utf-8')
        with pytest.raises(GcpError, match='Two GCP files'):
            load_gcp_set(tmp_path)


class TestSaving:
    def test_round_trips_through_labelme_json(self, tmp_path):
        path = save_gcp_file(tmp_path / '000000.json', {'p01': [10.5, 20.5]}, image_size=(480, 640))
        entry = load_gcp_file(path)
        assert entry.labels == ('p01',)
        assert np.allclose(entry.points[0], [10.5, 20.5])

    def test_the_image_path_is_relative_to_the_file(self, tmp_path):
        path = save_gcp_file(tmp_path / 'gcp' / '000000.json', {'p01': [1.0, 2.0]}, tmp_path / 'images' / '000000.jpg')
        assert json.loads(path.read_text(encoding='utf-8'))['imagePath'] == '../images/000000.jpg'


class TestResiduals:
    def test_exact_homography_gives_zero_residual(self):
        matrix = homography()
        result = synthetic_check(matrix, POINTS, offset_px=0.0)
        assert result['n_points'] == len(POINTS)
        assert result['residual_max_px'] < 1e-9

    def test_injected_offset_appears_at_its_own_magnitude(self):
        for offset in (1.0, 3.5, 12.0):
            result = synthetic_check(homography(), POINTS, offset_px=offset)
            assert result['residual_mean_px'] == pytest.approx(offset, abs=1e-9)
            assert result['residual_max_px'] == pytest.approx(offset, abs=1e-9)

    def test_residual_scales_linearly_with_injected_error(self):
        one = synthetic_check(homography(), POINTS, offset_px=1.0)['residual_rms_px']
        ten = synthetic_check(homography(), POINTS, offset_px=10.0)['residual_rms_px']
        assert ten == pytest.approx(10.0 * one, rel=1e-9)

    def test_only_shared_labels_contribute(self):
        matrix = homography()
        target = warp_points(POINTS, matrix)
        reference = GroundControlPoints(0, ('a', 'b', 'c'), target[:3])
        current = GroundControlPoints(1, ('b', 'c', 'z'), np.vstack([POINTS[1:3], [[0.0, 0.0]]]))
        result = residuals(reference, current, matrix)
        assert result['n_points'] == 2
        assert result['labels'] == ['b', 'c']
        assert result['residual_max_px'] < 1e-9

    def test_matching_is_by_label_not_by_order(self):
        # The same points in a different file order must give the same residual.
        matrix = homography()
        target = warp_points(POINTS, matrix)
        labels = ('a', 'b', 'c', 'd', 'e')
        reference = GroundControlPoints(0, labels, target)
        order = [4, 2, 0, 3, 1]
        shuffled = GroundControlPoints(1, tuple(labels[i] for i in order), POINTS[order])
        assert residuals(reference, shuffled, matrix)['residual_max_px'] < 1e-9

    def test_no_shared_labels_gives_nan_not_zero(self):
        reference = GroundControlPoints(0, ('a',), np.array([[1.0, 2.0]]))
        current = GroundControlPoints(1, ('z',), np.array([[1.0, 2.0]]))
        result = residuals(reference, current, np.eye(3))
        assert result['n_points'] == 0
        assert np.isnan(result['residual_mean_px'])


class TestSummary:
    def test_pools_over_individual_points_not_frame_means(self):
        # Pooled over points, not over per-frame means.
        per_frame = [
            {'residuals_px': [1.0] * 10},
            {'residuals_px': [11.0, 11.0]},
        ]
        summary = summarize_residuals(per_frame)
        assert summary['n_points'] == 12
        assert summary['residual_median_px'] == pytest.approx(1.0)  # a frame-mean average would give 6.0

    def test_reports_the_bound_not_only_the_average(self):
        summary = summarize_residuals([{'residuals_px': list(np.linspace(0.0, 10.0, 101))}])
        assert summary['residual_p95_px'] == pytest.approx(9.5, abs=0.2)
        assert summary['residual_max_px'] == pytest.approx(10.0)

    def test_no_points_reports_zero(self):
        assert summarize_residuals([])['n_points'] == 0


class TestAuditHomographies:
    def test_audits_every_annotated_frame(self):
        matrix = homography()
        target = warp_points(POINTS, matrix)
        labels = tuple(f"p{i}" for i in range(len(POINTS)))
        points = {
            0: GroundControlPoints(0, labels, target),
            1500: GroundControlPoints(1500, labels, POINTS),
            3000: GroundControlPoints(3000, labels, POINTS),
        }
        report = audit_homographies(
            points, 0, {'000000->001500': matrix, '000000->003000': matrix}, label='reference_estimator'
        )
        assert len(report['per_frame']) == 2
        assert report['summary']['residual_max_px'] < 1e-9
        assert report['summary']['n_points'] == 2 * len(POINTS)

    def test_frames_without_a_homography_are_reported_as_skipped(self):
        labels = tuple(f"p{i}" for i in range(len(POINTS)))
        matrix = homography()
        points = {
            0: GroundControlPoints(0, labels, warp_points(POINTS, matrix)),
            1500: GroundControlPoints(1500, labels, POINTS),
            3000: GroundControlPoints(3000, labels, POINTS),
        }
        report = audit_homographies(points, 0, {'000000->001500': matrix, '000000->003000': None}, label='x')
        assert report['skipped_frames'] == [3000]
        assert len(report['per_frame']) == 1

    def test_missing_reference_annotation_raises(self):
        labels = ('a',)
        points = {1500: GroundControlPoints(1500, labels, np.array([[1.0, 2.0]]))}
        with pytest.raises(Exception, match='no ground control points'):
            audit_homographies(points, 0, {'000000->001500': np.eye(3)}, label='x')

    def test_no_auditable_frame_raises(self):
        labels = ('a',)
        points = {
            0: GroundControlPoints(0, labels, np.array([[1.0, 2.0]])),
            1500: GroundControlPoints(1500, labels, np.array([[1.0, 2.0]])),
        }
        with pytest.raises(Exception, match='No audited frame'):
            audit_homographies(points, 0, {'000000->001500': None}, label='x')


class TestAnnotationGeometry:
    """The placement panel measures what is clicked, and the prediction starts where the estimator says."""

    def test_the_panel_maps_display_pixels_to_source_pixels_exactly(self):
        panel = Panel.centered((1000.25, 640.75), zoom=4.0, size=360)
        source = np.array([1003.5, 655.0])
        u, v = panel.to_display(source)
        assert np.allclose(panel.to_source(u, v), source)
        # A click on a display pixel resolves the source to within one display pixel's worth.
        assert np.linalg.norm(panel.to_source(round(u), round(v)) - source) <= 0.5 / 4.0 * np.sqrt(2) + 1e-9
        image = np.zeros((1080, 1920), dtype=np.uint8)
        image[655, 1003] = 255
        view = panel.render(image)
        lit = np.argwhere(view[..., 0] == 255)
        assert len(lit) == 16  # one source pixel is a 4 x 4 block at zoom 4
        assert np.allclose(panel.to_source(*lit.mean(axis=0)[::-1]), (1003.0, 655.0), atol=0.5)

    def test_the_prediction_inverts_the_cached_direction_and_the_jitter_is_hidden_per_session(self):
        current_to_reference = np.array([[1.02, 0.01, 12.0], [-0.01, 0.99, -7.0], [1e-6, 0.0, 1.0]])
        current = np.array([[300.0, 200.0], [1500.0, 900.0]])
        reference = {
            'p01': current_to_reference @ np.append(current[0], 1.0),
            'p02': current_to_reference @ np.append(current[1], 1.0),
        }
        reference = {k: (v[:2] / v[2]).tolist() for k, v in reference.items()}
        predicted = predict_positions(reference, current_to_reference)
        assert np.allclose(predicted['p01'], current[0], atol=1e-6)
        assert np.allclose(predicted['p02'], current[1], atol=1e-6)
        assert predict_positions(reference, None) is None
        first, again, repeat = jitter(1500, 'p01', 'first'), jitter(1500, 'p01', 'first'), jitter(1500, 'p01', 'repeat')
        assert np.array_equal(first, again)
        assert not np.array_equal(first, repeat)
        assert np.all(np.abs(first) <= JITTER_PX) and np.all(np.abs(repeat) <= JITTER_PX)


class TestResidualByOffset:
    @staticmethod
    def frame(offset, error):
        return {'offset': offset, 'residuals_px': [error] * 12, 'frame_index': offset}

    def test_frames_are_summarized_per_configured_bin(self):
        per_frame = [self.frame(10, 0.2), self.frame(100, 0.3), self.frame(2000, 0.6)]
        summary = summarize_by_offset(per_frame, bins=((1, 120), (121, 480), (481, 1920), (1921, 3600)))
        assert list(summary) == ['1-120', '1921-3600']
        assert summary['1-120']['n_frames'] == 2


class TestAuditFrames:
    BINS = ((1, 120), (121, 480), (481, 1920), (1921, 3600))

    def test_two_interior_frames_per_bin(self):
        frames = [0, *range(10, 3601, 10)]
        chosen = audit_frames(frames, 0, self.BINS)
        assert len(chosen) == 8
        for low, high in self.BINS:
            inside = [f for f in chosen if low <= f <= high]
            assert len(inside) == 2
            assert low < min(inside) and max(inside) < high

    def test_a_sparse_bin_gives_what_it_has(self):
        assert audit_frames([0, 50, 200, 300, 400], 0, self.BINS) == [50, 200, 400]

    def test_no_frame_is_picked_twice(self):
        for n in range(1, 12):
            frames = [0, *range(121, 121 + 10 * n, 10)]
            chosen = audit_frames(frames, 0, ((121, 480),))
            assert len(chosen) == len(set(chosen)) == min(n, 2)

    def test_offsets_are_measured_from_the_reference(self):
        chosen = audit_frames([1000, *range(1010, 1121, 10)], 1000, ((1, 120),))
        assert all(1000 < f <= 1120 for f in chosen)


class TestCandidateAudit:
    """`audit --results`: a failure counts against a candidate, a gap is reported."""

    # Enough points for a well-determined fitted transform.
    GRID = np.array([[x, y] for x in (80.0, 240.0, 400.0, 560.0) for y in (60.0, 240.0, 420.0)])

    @classmethod
    def setup_case(cls, tmp_path, records):
        labels = tuple(f"p{i}" for i in range(len(cls.GRID)))
        truth = homography()  # current -> reference
        points = {0: GroundControlPoints(0, labels, cls.GRID)}
        for index in (10, 20):
            points[index] = GroundControlPoints(index, labels, warp_points(cls.GRID, np.linalg.inv(truth)))
        store = ResultStore(tmp_path / 'results')
        store.initialize(build_manifest({'kind': 'dynamic', 'name': 'r', 'seed': 0}, 'h', {1: {'pipeline': 'x'}}))
        store.write_run(1, {'run_number': 1, 'params': {'pipeline': 'sift+obb', 'masking': 'obb'}, 'records': records})
        return store.directory, points, truth

    def test_a_failed_audited_pair_is_scored_as_identity_not_dropped(self, tmp_path):
        records = [
            {'pair': '000000->000010', 'homography': homography().tolist(), 'failed': False},
            {'pair': '000000->000020', 'failed': True},
        ]
        results, points, _ = self.setup_case(tmp_path, records)
        (entry,) = audit_results(results, points, 0, frame_size={'width': 640, 'height': 480})
        assert entry['failed_frames'] == [20] and entry['missing_frames'] == []
        by_frame = {frame['frame_index']: frame for frame in entry['per_frame']}
        assert by_frame[10]['residual_max_px'] < 1e-6
        assert by_frame[20]['residual_median_px'] > 5.0  # the identity is far from the truth
        assert entry['gcp_transform']['summary']['n_frames'] == 2

    def test_a_run_that_failed_on_every_audited_frame_is_scored(self, tmp_path):
        # Leaving it out would drop the pipeline from the GCP outcomes exactly where it failed.
        records = [{'pair': '000000->000010', 'failed': True}, {'pair': '000000->000020', 'failed': True}]
        results, points, _ = self.setup_case(tmp_path, records)
        unscored = []
        (entry,) = audit_results(results, points, 0, frame_size={'width': 640, 'height': 480}, unscored=unscored)
        assert unscored == [] and entry['failed_frames'] == [10, 20]
        assert entry['gcp_transform']['summary']['n_frames'] == 2

    def test_an_audited_frame_without_a_record_is_reported(self, tmp_path):
        records = [{'pair': '000000->000010', 'homography': homography().tolist(), 'failed': False}]
        results, points, _ = self.setup_case(tmp_path, records)
        (entry,) = audit_results(results, points, 0, frame_size={'width': 640, 'height': 480})
        assert entry['missing_frames'] == [20] and entry['failed_frames'] == []

    def test_a_run_without_any_audited_frame_is_reported_unscored(self, tmp_path):
        records = [{'pair': '000000->000099', 'homography': homography().tolist(), 'failed': False}]
        results, points, _ = self.setup_case(tmp_path, records)
        unscored = []
        assert audit_results(results, points, 0, frame_size={'width': 640, 'height': 480}, unscored=unscored) == []
        assert unscored == ['sift+obb']


class _Boxes:
    """Objects as `propose_points` reads them: quadrilateral corners."""

    def __init__(self, corners):
        self._corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)

    def __len__(self):
        return len(self._corners)

    def corners(self):
        return self._corners


class TestProposals:
    """Proposals on a smoothed random texture moved by a known affine map."""

    MOTION = np.array([[np.cos(0.01), -np.sin(0.01), 7.3], [np.sin(0.01), np.cos(0.01), -4.6], [0.0, 0.0, 1.0]])

    @pytest.fixture
    def frames(self):
        rng = np.random.default_rng(0)
        texture = cv2.GaussianBlur(rng.uniform(0, 255, (600, 600)).astype(np.float32), (0, 0), 2.0)
        texture[:, 450:] = 128.0  # a featureless strip
        reference = np.clip(texture, 0, 255).astype(np.uint8)
        current = cv2.warpAffine(reference, self.MOTION[:2], (600, 600), flags=cv2.INTER_CUBIC)
        return reference, current

    def _initial(self, offset=(1.5, -2.0)):
        return np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]]) @ self.MOTION

    def test_a_textured_point_is_found_where_the_motion_puts_it(self, frames):
        points = {'p01': [200.0, 250.0], 'p02': [320.5, 180.25]}
        proposals = propose_points(*frames, points, initial=self._initial())
        for label, point in points.items():
            truth = warp_points(np.array([point]), self.MOTION)[0]
            assert proposals[label].reason == ''
            assert np.linalg.norm(proposals[label].point - truth) < 0.05
            assert proposals[label].round_trip_px < 0.05

    def test_a_point_without_texture_gets_no_proposal(self, frames):
        proposal = propose_points(*frames, {'p01': [520.0, 300.0]}, initial=self._initial())['p01']
        assert proposal.point is None and proposal.reason

    def test_a_point_near_the_edge_gets_no_proposal(self, frames):
        proposal = propose_points(*frames, {'p01': [20.0, 300.0]}, initial=self._initial())['p01']
        assert proposal.point is None and proposal.reason == 'neighborhood leaves the frame'

    def test_a_point_under_an_object_gets_no_proposal(self, frames):
        box = _Boxes([[180.0, 230.0], [220.0, 230.0], [220.0, 270.0], [180.0, 270.0]])
        proposal = propose_points(*frames, {'p01': [200.0, 250.0]}, box, None, initial=self._initial())['p01']
        assert proposal.point is None and proposal.reason == 'objects cover the neighborhood'

    def test_without_an_initial_map_the_platform_motion_starts_the_fit(self, frames):
        proposal = propose_points(*frames, {'p01': [200.0, 250.0]})['p01']
        truth = warp_points(np.array([[200.0, 250.0]]), self.MOTION)[0]
        assert np.linalg.norm(proposal.point - truth) < 0.05


class TestProposalReview:
    def test_frames_without_recorded_proposals_are_left_out(self, tmp_path):
        save_gcp_file(tmp_path / '000000.json', {'p01': [1.0, 1.0]})
        save_gcp_file(tmp_path / '000005.json', {'p01': [2.0, 2.0]}, metadata={'session': 'first'})
        assert proposal_review(tmp_path, 0) == {'n_frames': 0, 'n_points': 0}

    def test_the_reference_frame_is_not_reviewed(self, tmp_path):
        recorded = {'proposals': {'p01': {'point': [1.0, 1.0]}}}
        save_gcp_file(tmp_path / '000000.json', {'p01': [1.0, 1.0]}, metadata=recorded)
        save_gcp_file(tmp_path / '000005.json', {'p01': [2.0, 2.0]}, metadata=recorded)
        review = proposal_review(tmp_path, 0)
        assert review['frames'] == [5] and review['n_corrected'] == 1
        assert review['correction_max_px'] == pytest.approx(np.sqrt(2))
        assert review['corrections_px'] == [pytest.approx(np.sqrt(2))]

    def test_hand_placed_points_are_scored_against_the_prediction(self, tmp_path):
        recorded = {'proposals': {'p01': {'point': [2.0, 2.0]}, 'p02': {'point': None}, 'p03': {'point': None}}}
        points = {'p01': [2.0, 2.0], 'p02': [5.0, 5.0], 'p03': [9.0, 9.0]}
        save_gcp_file(tmp_path / '000005.json', points, metadata=recorded)
        per_frame = [{'frame_index': 5, 'labels': ['p01', 'p02', 'p03'], 'residuals_px': [0.1, 0.3, 0.8]}]
        review = proposal_review(tmp_path, 0, per_frame)
        assert (review['n_placed_by_hand'], review['n_hand_scored'], review['n_hand_within_half_px']) == (2, 2, 1)
        assert 'n_hand_scored' not in proposal_review(tmp_path, 0)
