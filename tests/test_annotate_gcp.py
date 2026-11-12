"""Re-placing single ground control points, with the interactive panel scripted."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from stabilo_benchmark.gcp import (
    GCP_DIR,
    GCP_MANUAL_DIR,
    GCP_REPEAT_DIR,
    Proposal,
    load_gcp_file,
    proposal_review,
    save_gcp_file,
)
from stabilo_benchmark.report import annotate_gcp
from stabilo_benchmark.report.annotate_gcp import AnnotateError, Placement, placement_conflicts, run_annotate

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
_SAMPLE_CASE = json.loads((SAMPLE / 'CASE.json').read_text(encoding='utf-8'))
# The sample's reference frame and three later ones: the nearest, one in the middle, the farthest.
REF = _SAMPLE_CASE['reference_index']
_LATER = sorted(frame for frame in _SAMPLE_CASE['frames'] if frame != REF)
NEAR, MID, FAR = _LATER[0], _LATER[len(_LATER) // 2], _LATER[-1]
REFERENCE = {'p01': [100.0, 100.0], 'p02': [400.0, 300.0], 'p03': [900.0, 700.0]}
CURRENT = {'p01': [101.0, 99.0], 'p02': [405.0, 295.0], 'p03': [902.0, 698.0]}


def name(frame: int, suffix: str = '.json') -> str:
    return f"{frame:06d}{suffix}"


@pytest.fixture
def case(tmp_path):
    case_dir = tmp_path / 'case'
    case_dir.mkdir()
    shutil.copy(SAMPLE / 'CASE.json', case_dir / 'CASE.json')
    shutil.copytree(SAMPLE / 'images', case_dir / 'images')
    save_gcp_file(case_dir / GCP_DIR / name(REF), REFERENCE, metadata={'session': 'first'})
    save_gcp_file(case_dir / GCP_DIR / name(NEAR), CURRENT, metadata={'session': 'first'})
    return case_dir


def _script(monkeypatch, placements):
    queue = list(placements)
    monkeypatch.setattr(annotate_gcp, 'place_point', lambda *args, **kwargs: queue.pop(0))
    return queue


def _args(frames, labels, propose=False):
    return argparse.Namespace(frames=frames, labels=labels, overwrite=False, zoom=4.0, repeat=False, propose=propose)


def test_only_the_named_points_change(case, monkeypatch):
    queue = _script(
        monkeypatch,
        [
            Placement('placed', np.array([403.0, 296.0])),  # reference p02
            Placement('skip'),  # reference p03 keeps its point
            Placement('placed', np.array([404.0, 296.0])),  # the later frame's p02
            Placement('skip'),  # the later frame's p03 is not visible
        ],
    )
    run_annotate(case, _args([NEAR, REF], ['p02', 'p03']))
    assert not queue
    reference = load_gcp_file(case / GCP_DIR / name(REF))
    assert reference.labels == ('p01', 'p02', 'p03')
    np.testing.assert_allclose(reference.points, [REFERENCE['p01'], [403.0, 296.0], REFERENCE['p03']])
    current = load_gcp_file(case / GCP_DIR / name(NEAR))
    assert current.labels == ('p01', 'p02')
    np.testing.assert_allclose(current.points, [CURRENT['p01'], [404.0, 296.0]])
    metadata = json.loads((case / GCP_DIR / name(NEAR)).read_text(encoding='utf-8'))['stabilo_benchmark']
    assert metadata == {'session': 'first', 'redone_labels': ['p02', 'p03']}


def test_an_unknown_label_raises_before_anything_opens(case, monkeypatch):
    _script(monkeypatch, [])
    with pytest.raises(AnnotateError, match='p09'):
        run_annotate(case, _args([NEAR], ['p09']))


def test_labels_need_frames(case, monkeypatch):
    _script(monkeypatch, [])
    with pytest.raises(AnnotateError, match='--frames'):
        run_annotate(case, _args(None, ['p02']))


def test_a_replaced_point_drops_where_it_was_not_answered(case, monkeypatch):
    save_gcp_file(case / GCP_DIR / name(FAR), CURRENT, metadata={'session': 'first'})
    _script(
        monkeypatch,
        [
            Placement('placed', np.array([700.0, 500.0])),  # reference p02 moved to a new feature
            Placement('placed', np.array([703.0, 497.0])),  # the near frame's p02
            Placement('finish'),  # the far frame left unanswered
        ],
    )
    run_annotate(case, _args([REF, NEAR, FAR], ['p02']))
    np.testing.assert_allclose(load_gcp_file(case / GCP_DIR / name(REF)).as_dict()['p02'], [700.0, 500.0])
    np.testing.assert_allclose(load_gcp_file(case / GCP_DIR / name(NEAR)).as_dict()['p02'], [703.0, 497.0])
    assert load_gcp_file(case / GCP_DIR / name(FAR)).labels == ('p01', 'p03')


def test_the_repeat_session_is_guided_by_the_first_sessions_new_point(case, monkeypatch):
    save_gcp_file(case / GCP_REPEAT_DIR / name(REF), REFERENCE, metadata={'session': 'repeat'})
    moved = dict(REFERENCE, p02=[700.0, 500.0])
    save_gcp_file(case / GCP_DIR / name(REF), moved, metadata={'session': 'first'})
    seen = []

    def place(image, prompt, start=None, **kwargs):
        seen.append(start)
        return Placement('placed', np.array([701.0, 499.0]))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    args = _args([REF], ['p02'])
    args.repeat = True
    run_annotate(case, args)
    assert np.linalg.norm(seen[0] - np.array([700.0, 500.0])) <= np.sqrt(2) * annotate_gcp.JITTER_PX
    np.testing.assert_allclose(load_gcp_file(case / GCP_REPEAT_DIR / name(REF)).as_dict()['p02'], [701.0, 499.0])


def test_replacing_shows_every_other_point_and_the_old_one_in_the_full_frame(case, monkeypatch):
    calls = []

    def place(image, prompt, start=None, **kwargs):
        calls.append((start, kwargs))
        return Placement('placed', np.array([700.0, 500.0]))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    run_annotate(case, _args([REF], ['p02']))
    start, kwargs = calls[0]
    assert start is None
    assert set(kwargs['placed_points']) == {'p01', 'p03'}
    assert list(kwargs['old_points'].values()) == [REFERENCE['p02']]


def _propose(monkeypatch, proposals):
    """Script `frame_proposals`: a mapping from label to proposed point (None: no standing proposal)."""
    calls = []

    def frame_proposals(case_dir, reference_index, reference_image, index, image, points):
        calls.append((index, list(points)))
        return {
            label: Proposal(None, reason='scripted')
            if proposals.get(label) is None
            else Proposal(np.asarray(proposals[label], dtype=float), 0.99, 0.01)
            for label in points
        }

    monkeypatch.setattr(annotate_gcp, 'frame_proposals', frame_proposals)
    return calls


def _accept_or(monkeypatch, corrections):
    """Script `place_point`: Enter on a placed point, else the label's correction or hand placement."""
    seen = []

    def place(image, prompt, start=None, initial=None, **kwargs):
        label = prompt.split("'")[1]
        seen.append((label, start, initial))
        if label in corrections:
            return Placement('placed', np.asarray(corrections[label], dtype=float))
        return Placement('placed', np.asarray(initial, dtype=float))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    return seen


def test_a_proposal_opens_placed_and_is_recorded_with_the_frame(case, monkeypatch):
    calls = _propose(monkeypatch, {'p01': [101.0, 99.0], 'p02': [405.0, 295.0], 'p03': None})
    seen = _accept_or(monkeypatch, {'p02': [405.5, 295.0], 'p03': [902.0, 698.0]})
    run_annotate(case, _args([MID], None, propose=True))
    assert calls == [(MID, ['p01', 'p02', 'p03'])]
    np.testing.assert_allclose(seen[0][1], [101.0, 99.0])
    np.testing.assert_allclose(seen[0][2], [101.0, 99.0])
    assert seen[2][2] is None
    metadata = json.loads((case / GCP_DIR / name(MID)).read_text(encoding='utf-8'))['stabilo_benchmark']
    assert metadata['proposals']['p01']['point'] == [101.0, 99.0]
    assert metadata['proposals']['p03'] == {
        'point': None,
        'correlation': None,
        'round_trip_px': None,
        'reason': 'scripted',
    }
    assert metadata['proposal']['radius_px'] > 0
    review = proposal_review(case / GCP_DIR, REF)
    assert {k: review[k] for k in ('n_points', 'n_accepted', 'n_corrected', 'n_placed_by_hand')} == {
        'n_points': 3,
        'n_accepted': 1,
        'n_corrected': 1,
        'n_placed_by_hand': 1,
    }
    assert review['correction_median_px'] == pytest.approx(0.5)


def test_without_proposals_nothing_is_placed_before_a_click(case, monkeypatch):
    calls = _propose(monkeypatch, {})
    seen = []

    def place(image, prompt, start=None, initial=None, **kwargs):
        seen.append(initial)
        return Placement('placed', np.array([1.0, 1.0]))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    run_annotate(case, _args([MID], None, propose=False))
    assert calls == [] and seen == [None, None, None]
    metadata = json.loads((case / GCP_DIR / name(MID)).read_text(encoding='utf-8'))['stabilo_benchmark']
    assert 'proposals' not in metadata


def test_the_repeat_session_never_proposes(case, monkeypatch):
    save_gcp_file(case / GCP_REPEAT_DIR / name(REF), REFERENCE, metadata={'session': 'repeat'})
    calls = _propose(monkeypatch, {'p01': [101.0, 99.0]})
    seen = []

    def place(image, prompt, start=None, initial=None, **kwargs):
        seen.append(initial)
        return Placement('placed', np.array([1.0, 1.0]))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    args = _args([NEAR], None, propose=True)
    args.repeat = True
    run_annotate(case, args)
    assert calls == [] and seen == [None, None, None]


def test_the_manual_session_places_the_first_sessions_frames_on_its_reference_points(case, monkeypatch):
    calls = _propose(monkeypatch, {'p01': [101.0, 99.0]})
    seen = []

    def place(image, prompt, start=None, initial=None, **kwargs):
        seen.append(initial)
        return Placement('placed', np.array([1.0, 1.0]))

    monkeypatch.setattr(annotate_gcp, 'place_point', place)
    args = _args(None, None, propose=True)
    args.manual = True
    run_annotate(case, args)
    assert calls == [] and seen == [None, None, None]
    assert sorted(p.name for p in (case / GCP_MANUAL_DIR).glob('*.json')) == [name(REF), name(NEAR)]
    reference = load_gcp_file(case / GCP_MANUAL_DIR / name(REF)).as_dict()
    assert {label: point.tolist() for label, point in reference.items()} == {
        label: list(map(float, point)) for label, point in REFERENCE.items()
    }
    assert (
        json.loads((case / GCP_MANUAL_DIR / name(NEAR)).read_text(encoding='utf-8'))['stabilo_benchmark']['session']
        == 'manual'
    )
    assert load_gcp_file(case / GCP_DIR / name(NEAR)).as_dict().keys() == CURRENT.keys()


def test_repeat_and_manual_are_separate_sessions(case):
    args = _args(None, None)
    args.repeat = args.manual = True
    with pytest.raises(annotate_gcp.AnnotateError, match='separate sessions'):
        run_annotate(case, args)


def test_a_redone_point_records_its_proposal_beside_the_earlier_ones(case, monkeypatch):
    earlier = {'p01': {'point': [101.0, 99.0], 'correlation': 0.99, 'round_trip_px': 0.01, 'reason': ''}}
    save_gcp_file(case / GCP_DIR / name(NEAR), CURRENT, metadata={'session': 'first', 'proposals': earlier})
    _propose(monkeypatch, {'p02': [404.0, 296.0]})
    _accept_or(monkeypatch, {})
    run_annotate(case, _args([NEAR], ['p02'], propose=True))
    metadata = json.loads((case / GCP_DIR / name(NEAR)).read_text(encoding='utf-8'))['stabilo_benchmark']
    assert set(metadata['proposals']) == {'p01', 'p02'}
    np.testing.assert_allclose(load_gcp_file(case / GCP_DIR / name(NEAR)).as_dict()['p02'], [404.0, 296.0])


def test_reference_points_on_objects_or_structure_are_named(tmp_path):
    case_dir = tmp_path / 'case'
    shutil.copytree(SAMPLE, case_dir, ignore=shutil.ignore_patterns('oracle', 'results', 'gcp*'))
    rows = (case_dir / 'labels_obb' / name(REF, '.txt')).read_text(encoding='utf-8').split('\n')
    obb = np.array(rows[0].split()[1:9], dtype=float).reshape(4, 2)
    on_object = obb.mean(axis=0)
    (case_dir / 'EXCLUSION.json').write_text(
        json.dumps(
            {
                'shapes': [
                    {'label': 'structure', 'shape_type': 'polygon', 'points': [[0, 0], [60, 0], [60, 60], [0, 60]]}
                ]
            }
        ),
        encoding='utf-8',
    )
    conflicts = placement_conflicts(case_dir, REF, {'p01': on_object, 'p02': [30.0, 30.0], 'p03': [-5.0, 10.0]})
    assert conflicts == {
        'p01': f"within {annotate_gcp.CLEARANCE_PX} px of an object",
        'p02': 'inside an exclusion polygon',
        'p03': 'outside the frame',
    }


def test_the_clearance_is_the_shipped_evaluation_dilation():
    configs = sorted((Path(__file__).resolve().parents[1] / 'configs' / 'songdo' / 'dynamic').glob('*.yaml'))
    dilations = {
        yaml.safe_load(path.read_text(encoding='utf-8'))['metrics']['region']['mask_dilation_px']
        for path in configs
        if 'mask_dilation_px' in path.read_text(encoding='utf-8')
    }
    assert dilations == {annotate_gcp.CLEARANCE_PX}
