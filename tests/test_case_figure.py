"""The case gallery, on copies of the bundled dynamic sample."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from stabilo_benchmark.report.build import _write_case_figure
from stabilo_benchmark.report.case_figure import case_fates, case_pair, cell_order, figure_cases, target_offset
from stabilo_benchmark.runners.dynamic import pair_object_motion
from stabilo_benchmark.sampling import FramePair

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
_SAMPLE_CASE = json.loads((SAMPLE / 'CASE.json').read_text(encoding='utf-8'))
REF = _SAMPLE_CASE['reference_index']
# The sample's frames after the reference, nearest first.
LATER = sorted(frame for frame in _SAMPLE_CASE['frames'] if frame != REF)


def _case(root: Path, name: str, cell=None, failed=()) -> Path:
    """
    The sample case under `name`, without survey bins (each test records its own), with identity
    reference estimates except for the `failed` frames.
    """
    case_dir = root / name
    case_dir.mkdir(parents=True)
    metadata = json.loads((SAMPLE / 'CASE.json').read_text(encoding='utf-8'))
    metadata['cell'] = cell
    metadata['sampling'].pop('survey_bins', None)
    (case_dir / 'CASE.json').write_text(json.dumps(metadata), encoding='utf-8')
    for kind in ('images', 'labels_hbb', 'labels_obb'):
        shutil.copytree(SAMPLE / kind, case_dir / kind)
    reference = metadata['reference_index']
    homographies = {
        f"{reference:06d}->{frame:06d}": None if frame in failed else np.eye(3).tolist()
        for frame in metadata['frames']
        if frame != reference
    }
    (case_dir / 'oracle').mkdir()
    (case_dir / 'oracle' / 'ORACLE.json').write_text(
        json.dumps({'reference_index': reference, 'homographies': homographies}), encoding='utf-8'
    )
    return case_dir


@pytest.fixture
def case(tmp_path):
    return _case(tmp_path, 'case', cell='high moving / coherent')


def _record_survey_bins(case_dir: Path, bins) -> None:
    metadata = json.loads((case_dir / 'CASE.json').read_text(encoding='utf-8'))
    metadata['sampling']['survey_bins'] = bins
    (case_dir / 'CASE.json').write_text(json.dumps(metadata), encoding='utf-8')


def test_the_target_is_the_longest_offset_the_survey_reads(case):
    _record_survey_bins(case, [[1, 1], [2, 3], [4, 7], [8, 15], [16, 31]])
    assert target_offset(json.loads((case / 'CASE.json').read_text(encoding='utf-8'))) == 22


def test_the_pair_is_the_sampled_frame_closest_to_the_target(case):
    # The target, sqrt(1000 * 1600) = 1265 frames, falls between sampled offsets.
    _record_survey_bins(case, [[1000, 1600]])
    nearest = min(LATER, key=lambda frame: abs(frame - REF - 1265))
    assert nearest - REF != 1265
    assert case_pair(case) == FramePair(REF, nearest)


def test_without_survey_bins_the_shortest_sampled_offset_is_drawn(case):
    assert target_offset(json.loads((case / 'CASE.json').read_text(encoding='utf-8'))) is None
    assert case_pair(case) == FramePair(REF, LATER[0])


def test_a_pair_whose_reference_estimate_failed_is_passed_over(tmp_path):
    assert case_pair(_case(tmp_path, 'case', failed=(LATER[0],))) == FramePair(REF, LATER[1])


def test_the_fates_are_the_benchmarks_own(case):
    data = case_fates(case)
    key = f"{REF:06d}->{LATER[0]:06d}"
    motion = pair_object_motion(case, [data['pair']], {key: np.eye(3)})[key]
    assert len(data['fates']) == len(motion.found)
    assert data['counts'] == {
        'found': int(motion.found.sum()),
        'displaced': int(motion.moved.sum()),
        'together': int(motion.group.sum()),
    }
    assert np.array_equal(data['fates'] == 'not found', ~motion.found)
    assert np.array_equal(data['fates'] == 'displaced with the largest group', motion.group)


def test_cases_are_drawn_in_cell_order_and_stable_within_a_cell(tmp_path):
    cells = {
        'a': 'stationary control',
        'b': 'low moving / mixed',
        'c': 'high moving / coherent',
        'd': None,
        'e': 'low moving / mixed',
    }
    dirs = []
    for name, cell in cells.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / 'CASE.json').write_text(json.dumps({'cell': cell}), encoding='utf-8')
        dirs.append(tmp_path / name)
    assert [d.name for d in cell_order(dirs)] == ['c', 'b', 'e', 'a', 'd']


def test_the_figure_is_written_for_several_cases(tmp_path):
    cases = [_case(tmp_path, f"case{i}", cell='low moving / mixed') for i in range(4)]
    path = figure_cases(cases, tmp_path / 'figures' / 'cases.pdf')
    assert path.is_file() and path.stat().st_size > 0


def test_the_report_finds_each_case_from_its_results_directory(tmp_path):
    case = _case(tmp_path, 'case')
    results = case / 'results' / 'offline_masking'
    results.mkdir(parents=True)
    produced = {'figures': [], 'skipped': []}
    _write_case_figure(produced, tmp_path / 'figures', [results])
    assert produced == {'figures': [str(tmp_path / 'figures' / 'cases.pdf')], 'skipped': []}


def test_a_results_directory_without_a_case_is_skipped_not_drawn(tmp_path):
    results = tmp_path / 'elsewhere' / 'results' / 'offline_masking'
    results.mkdir(parents=True)
    produced = {'figures': [], 'skipped': []}
    _write_case_figure(produced, tmp_path / 'figures', [results])
    assert produced['figures'] == [] and 'no CASE.json' in produced['skipped'][0]
