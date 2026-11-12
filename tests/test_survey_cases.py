#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""The table of selected cases, written by `report` from a SURVEY.json."""

from __future__ import annotations

import json

import pytest

from stabilo_benchmark.report.build import build_report
from stabilo_benchmark.report.survey_cases import case_rows, is_survey_file, write_cases_table


def candidate(video, cell, displaced, coherence, day='2022-10-06'):
    drone, session, hover = video.split('_')
    return {
        'video': video,
        'source': f"data/songdo/videos/{day}/{drone}/{session}/{hover}.MP4",
        'cell': cell,
        'displaced': displaced,
        'coherence': coherence,
        'stationarity': 0.5,
        'n_objects': 40,
        'frame': 1800,
    }


SURVEY = {
    'n_candidates': 12,
    'anchor': 'scan',
    'scan_stride': 900,
    'offsets': [1, 2, 5, 11, 22],
    'thresholds': {'displaced_widths': 0.1, 'coherent_widths': 0.1, 'coherent_share': 0.25, 'min_displaced_objects': 4},
    'candidates': [{'video': f"v{i // 2}"} for i in range(12)],
    'strata': {
        'cells': {'unclassified': ['a', 'b']},
        'splits': {'moving': 0.004, 'coherence': 0.3, 'basis': 'pool median'},
    },
    'selection': {
        'unfilled': {'low moving / coherent': 1},
        'selected': [
            candidate('D4_PM5_G1', 'stationary control', 0.0005, float('nan')),
            candidate('D9_PM5_R1', 'high moving / coherent', 0.0072, 0.42),
        ],
    },
}


def test_rows_follow_the_design_order_and_carry_the_day():
    rows = case_rows(SURVEY)
    assert [row['case'] for row in rows] == ['2022-10-06_D9_PM5_R1', '2022-10-06_D4_PM5_G1']


def test_the_caption_states_the_pool_the_splits_and_unfilled_cells(tmp_path):
    text = write_cases_table(SURVEY, tmp_path / 'cases.tex').read_text(encoding='utf-8')
    assert 'from 12 candidate reference frames in 6 hovers of 2022-10-06' in text and 'moving 0.40\\%' in text
    assert 'left 2 candidates unclassified' in text and 'low moving / coherent (1)' in text
    assert 'points every 900 frames' in text and 'the frames 1, 2, 5, 11 and 22 frames later' in text
    assert 'displaced by more than 0.1 median object widths' in text
    assert 'within 0.1 widths or 25\\% of its displacement' in text
    # The control has no coherence.
    assert '2022-10-06\\_D4\\_PM5\\_G1 & stationary control & 1800 & 0.05 & -- & 0.50 & 40' in text


def test_a_survey_without_recorded_thresholds_is_refused(tmp_path):
    bare = {key: value for key, value in SURVEY.items() if key != 'thresholds'}
    with pytest.raises(ValueError, match='records no thresholds'):
        write_cases_table(bare, tmp_path / 'cases.tex')


def test_a_survey_of_first_frames_says_so(tmp_path):
    first = {**SURVEY, 'anchor': 'start', 'candidates': [{'video': f"v{i}"} for i in range(12)]}
    text = write_cases_table(first, tmp_path / 'cases.tex').read_text(encoding='utf-8')
    assert 'from 12 hovers of 2022-10-06' in text and 'referenced at its first frame' in text


def test_report_on_a_survey_file_writes_the_table(tmp_path):
    path = tmp_path / 'SURVEY.json'
    path.write_text(json.dumps(SURVEY), encoding='utf-8')
    assert is_survey_file(path) and not is_survey_file(tmp_path)
    produced = build_report([path], output=tmp_path / 'out')
    assert produced['kind'] == 'survey'
    assert (tmp_path / 'out' / 'tables' / 'survey_cases.tex').exists()
