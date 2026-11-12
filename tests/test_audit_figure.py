"""The ground-control-point audit and its figure, on a copy of the bundled dynamic sample."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stabilo_benchmark.gcp import GCP_DIR, GroundControlPoints, load_gcp_file, save_gcp_file
from stabilo_benchmark.oracle import load_frame_labels
from stabilo_benchmark.report.audit_figure import _audited_points, _feature_mask, _panel_offsets, figure_gcp_audit
from stabilo_benchmark.report.build import _write_gcp_audit_figure, gcp_candidate_rows
from stabilo_benchmark.results import GCP_SCORES_NAME, ResultStore, build_manifest
from stabilo_benchmark.runners.audit import audit_homographies, run_audit

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
_SAMPLE_CASE = json.loads((SAMPLE / 'CASE.json').read_text(encoding='utf-8'))
# The sample's reference frame, and the nearest and farthest frames after it.
REF = _SAMPLE_CASE['reference_index']
NEAR, FAR = min(f for f in _SAMPLE_CASE['frames'] if f != REF), max(_SAMPLE_CASE['frames'])
WIDTH, HEIGHT = _SAMPLE_CASE['video']['width'], _SAMPLE_CASE['video']['height']
SHIFT = np.array([2.0, -1.5])


def _translation(shift):
    return np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])


@pytest.fixture
def case(tmp_path):
    """The sample case with synthetic GCPs on the reference, near and far frames, cached estimates and an audit."""
    case_dir = tmp_path / 'case'
    case_dir.mkdir()
    shutil.copy(SAMPLE / 'CASE.json', case_dir / 'CASE.json')
    for name in ('images', 'labels_hbb', 'labels_obb'):
        shutil.copytree(SAMPLE / name, case_dir / name)
    rng = np.random.default_rng(0)
    # Spread over the frame, as the protocol places them.
    grid = np.stack(np.meshgrid(np.linspace(0.05, 0.95, 4) * WIDTH, np.linspace(0.1, 0.9, 3) * HEIGHT), axis=-1)
    grid = grid.reshape(-1, 2)
    labels = [f"p{i + 1:02d}" for i in range(len(grid))]
    save_gcp_file(case_dir / GCP_DIR / f"{REF:06d}.json", dict(zip(labels, grid.tolist(), strict=True)))
    points = {REF: GroundControlPoints(REF, tuple(labels), grid)}
    # The cached estimate maps current to reference coordinates by SHIFT; the clicks miss it by noise.
    for index in (NEAR, FAR):
        current = grid - SHIFT + rng.normal(0, 0.4, grid.shape)
        save_gcp_file(case_dir / GCP_DIR / f"{index:06d}.json", dict(zip(labels, current.tolist(), strict=True)))
        points[index] = GroundControlPoints(index, tuple(labels), current)
    homographies = {f"{REF:06d}->{index:06d}": _translation(SHIFT) for index in (NEAR, FAR)}
    (case_dir / 'oracle').mkdir()
    (case_dir / 'oracle' / 'ORACLE.json').write_text(
        json.dumps(
            {
                'reference_index': REF,
                'pipeline': {'masking': 'obb', 'mask_margin_ratio': 0.2},
                'homographies': {key: value.tolist() for key, value in homographies.items()},
            }
        ),
        encoding='utf-8',
    )
    audit = {'reference_estimator': audit_homographies(points, REF, homographies, 'reference_estimator')}
    (case_dir / 'oracle' / 'AUDIT.json').write_text(json.dumps(audit), encoding='utf-8')
    return case_dir


def test_the_illustrated_residuals_are_the_audited_ones(case):
    data = _audited_points(case)
    audit = json.loads((case / 'oracle' / 'AUDIT.json').read_text(encoding='utf-8'))['reference_estimator']
    longest = next(frame for frame in audit['per_frame'] if frame['frame_index'] == FAR)
    assert data['frame_index'] == FAR
    assert data['labels'] == longest['labels']
    np.testing.assert_allclose(data['residuals'], longest['residuals_px'])


def test_the_figure_is_written_in_frames_without_a_frame_rate(case, tmp_path):
    metadata = json.loads((case / 'CASE.json').read_text(encoding='utf-8'))
    metadata['video']['fps'] = None
    (case / 'CASE.json').write_text(json.dumps(metadata), encoding='utf-8')
    path = figure_gcp_audit(case, [case], tmp_path / 'gcp_audit.pdf')
    assert path.is_file() and path.stat().st_size > 0


def test_the_figure_is_written_with_exclusion_polygons_and_a_frame_rate(case, tmp_path):
    metadata = json.loads((case / 'CASE.json').read_text(encoding='utf-8'))
    metadata['video']['fps'] = 29.97
    (case / 'CASE.json').write_text(json.dumps(metadata), encoding='utf-8')
    polygon = [[1200.0, 50.0], [1700.0, 50.0], [1700.0, 400.0], [1200.0, 400.0]]
    (case / 'EXCLUSION.json').write_text(
        json.dumps({'shapes': [{'label': 'structure', 'shape_type': 'polygon', 'points': polygon}]}), encoding='utf-8'
    )
    path = figure_gcp_audit(case, [case], tmp_path / 'gcp_audit.png', dilation_px=10)
    assert path.is_file() and path.stat().st_size > 0


def test_a_case_without_points_is_skipped_not_raised(case, tmp_path):
    shutil.rmtree(case / GCP_DIR)
    produced = {'figures': [], 'skipped': []}
    _write_gcp_audit_figure(produced, tmp_path / 'figures', [case])
    assert produced['figures'] == [] and produced['skipped'][0].startswith('gcp audit figure')


def test_the_audit_records_paths_relative_to_the_case(case, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    audit = json.loads(run_audit(case, argparse.Namespace(gcp=None, results=None)).read_text(encoding='utf-8'))
    assert audit['case'] == case.name and audit['gcp_dir'] == GCP_DIR


def test_the_audit_reports_how_the_proposals_were_reviewed(case, monkeypatch, tmp_path):
    path = case / GCP_DIR / f"{NEAR:06d}.json"
    points = load_gcp_file(path).as_dict()
    proposals = {label: {'point': point.tolist()} for label, point in points.items()}
    proposals['p01'] = {'point': (points['p01'] + [0.3, 0.4]).tolist()}
    proposals['p02'] = {'point': None, 'reason': 'neighborhood leaves the frame'}
    save_gcp_file(path, points, metadata={'session': 'first', 'proposals': proposals})
    monkeypatch.chdir(tmp_path)
    audit = json.loads(run_audit(case, argparse.Namespace(gcp=None, results=None)).read_text(encoding='utf-8'))
    review = audit['proposal_review']
    assert (review['n_points'], review['n_accepted'], review['n_corrected'], review['n_placed_by_hand']) == (
        len(points),
        len(points) - 2,
        1,
        1,
    )
    assert review['correction_median_px'] == pytest.approx(0.5)


def test_cases_with_slightly_different_frame_rates_share_one_box_per_offset(case, tmp_path):
    other = tmp_path / 'other'
    shutil.copytree(case, other)
    for directory, fps in ((case, 29.97002997), (other, 29.96995628)):
        metadata = json.loads((directory / 'CASE.json').read_text(encoding='utf-8'))
        metadata['video']['fps'] = fps
        (directory / 'CASE.json').write_text(json.dumps(metadata), encoding='utf-8')
    figure, ax = plt.subplots()
    _panel_offsets(ax, [case, other])
    boxes = [patch for patch in ax.patches if patch.get_gid() == 'offset-box']
    assert len(boxes) == 2
    plt.close(figure)


def test_the_feature_mask_is_scaled_about_each_center_as_recorded(case):
    labels = load_frame_labels(case, REF, WIDTH, HEIGHT)
    quads, factor, geometry = _feature_mask(case, labels)
    raw = labels.object_corners('obb')
    assert geometry == 'obb' and factor == pytest.approx(1.2)
    np.testing.assert_allclose(quads.mean(axis=1), raw.mean(axis=1))
    diagonal = np.linalg.norm(quads[:, 0] - quads[:, 2], axis=1)
    np.testing.assert_allclose(diagonal, 1.2 * np.linalg.norm(raw[:, 0] - raw[:, 2], axis=1))


def _campaign(case, name, shift):
    """A results directory beside the case's others, one candidate registering every audited frame by `shift`."""
    oracle = json.loads((case / 'oracle' / 'ORACLE.json').read_text(encoding='utf-8'))
    (case / 'oracle' / 'ORACLE.json').write_text(
        json.dumps({**oracle, 'frame_size': {'width': WIDTH, 'height': HEIGHT}}), encoding='utf-8'
    )
    store = ResultStore(case / 'results' / name)
    store.initialize(build_manifest({'kind': 'dynamic', 'name': name, 'seed': 0}, 'h', {1: {'pipeline': 'x'}}))
    records = [
        {'pair': f"{REF:06d}->{index:06d}", 'homography': _translation(shift).tolist(), 'failed': False}
        for index in (NEAR, FAR)
    ]
    store.write_run(1, {'run_number': 1, 'params': {'pipeline': 'sift+obb', 'masking': 'obb'}, 'records': records})
    return store.directory


class TestCandidateScoresStayWithTheirCampaign:
    def test_two_campaigns_of_one_case_keep_their_own_scores(self, case):
        exact, off = _campaign(case, 'offline_masking', SHIFT), _campaign(case, 'realtime_masking', SHIFT + 5.0)
        for results in (exact, off):
            run_audit(case, argparse.Namespace(gcp=None, results=str(results)))
        assert 'candidates' not in json.loads((case / 'oracle' / 'AUDIT.json').read_text(encoding='utf-8'))
        (first,), (second,) = gcp_candidate_rows(exact), gcp_candidate_rows(off)
        assert first['pipeline'] == second['pipeline'] == 'sift+obb'
        assert first['cte_median'] < 1.0 < 5.0 < second['cte_median']

    def test_writing_runs_removes_the_scores_of_the_runs_they_replace(self, case):
        results = _campaign(case, 'offline_masking', SHIFT)
        run_audit(case, argparse.Namespace(gcp=None, results=str(results)))
        assert (results / GCP_SCORES_NAME).is_file()
        ResultStore(results, resume=True).initialize(
            build_manifest({'kind': 'dynamic', 'name': 'offline_masking', 'seed': 0}, 'h', {1: {'pipeline': 'x'}})
        )
        assert not (results / GCP_SCORES_NAME).exists() and gcp_candidate_rows(results) == []


def test_the_named_case_illustrates_the_audit(case, monkeypatch, tmp_path):
    other = tmp_path / 'other'
    shutil.copytree(case, other)
    shown = []
    monkeypatch.setattr(
        'stabilo_benchmark.report.build.figure_gcp_audit', lambda case_dir, *_: shown.append(case_dir) or case_dir
    )
    _write_gcp_audit_figure({'figures': [], 'skipped': []}, tmp_path / 'figures', [case, other])
    _write_gcp_audit_figure({'figures': [], 'skipped': []}, tmp_path / 'figures', [case, other], audit_case='other')
    assert shown == [case, other]


def test_a_named_case_without_an_audit_raises(case, tmp_path):
    with pytest.raises(ValueError, match='not among the audited cases'):
        _write_gcp_audit_figure({'figures': [], 'skipped': []}, tmp_path / 'figures', [case], audit_case='missing')
