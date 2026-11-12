#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Classifying a resource failure. Allocator messages and types are resource errors; anything
unrecognized is a pipeline failure.
"""

from __future__ import annotations

import cv2
import pytest

from stabilo_benchmark import resources
from stabilo_benchmark.resources import (
    describe_resource_error,
    is_resource_error,
    peak_memory,
    reset_peak_memory,
)


class TestResourceErrorsAreRecognized:
    def test_memory_error_is_unambiguous(self):
        assert is_resource_error(MemoryError())

    def test_a_subclass_is_recognized_too(self):
        # numpy's `_ArrayMemoryError` subclasses MemoryError; provoking it is not portable.
        error = type('_ArrayMemoryError', (MemoryError,), {})('Unable to allocate 8.00 TiB')
        assert is_resource_error(error)

    @pytest.mark.parametrize(
        'message',
        [
            'CUDA out of memory. Tried to allocate 20.00 GiB',
            'MPS backend out of memory (MPS allocated: 18.00 GB)',
            "Invalid buffer size: 24.00 GB",
            'OpenCV(4.10.0) error: Insufficient memory',
            'Failed to allocate 4194304000 bytes',
        ],
    )
    def test_the_allocators_own_words_are_matched(self, message):
        assert is_resource_error(RuntimeError(message))

    def test_a_torch_style_out_of_memory_class_is_matched_by_name(self):
        # torch.cuda.OutOfMemoryError is matched by type.
        error = type('OutOfMemoryError', (RuntimeError,), {})('')
        assert is_resource_error(error)


class TestPipelineFailuresAreNotExcused:
    """Pipeline failures are never classified as resource errors."""

    @pytest.mark.parametrize(
        'error',
        [
            ValueError('Invalid downsample_ratio'),
            RuntimeError('Not enough points to estimate the transformation matrix'),
            cv2.error('findHomography: DLT needs at least 4 points'),
            FileNotFoundError('labels_obb/b1_000000_d1.txt'),
            ZeroDivisionError(),
        ],
    )
    def test_ordinary_failures_are_not_resource_failures(self, error):
        assert not is_resource_error(error)


class TestDescription:
    def test_it_records_what_failed_and_where(self):
        description = describe_resource_error(MemoryError('out of memory'), where='b1_000000_d1.jpg')
        assert description['type'] == 'MemoryError'
        assert description['where'] == 'b1_000000_d1.jpg'
        assert 'out of memory' in description['message']
        assert isinstance(description['peak_memory'], dict)

    def test_where_is_omitted_rather_than_faked(self):
        assert 'where' not in describe_resource_error(MemoryError())

    def test_peak_memory_is_best_effort_and_never_raises(self):
        usage = peak_memory()
        assert isinstance(usage, dict)
        # Host memory is positive; a device counter can be 0.
        if 'host_peak_mb' in usage:
            assert usage['host_peak_mb'] > 0
        for key, value in usage.items():
            if key.endswith('_mb'):
                assert value >= 0


class TestPeakScope:
    """A peak is the run's only if the counter was reset; otherwise it must say so."""

    def test_where_the_host_counter_resets_the_peak_is_the_runs(self, tmp_path, monkeypatch):
        status = tmp_path / 'status'
        status.write_text('Name:\tpython\nVmHWM:\t  123456 kB\nVmRSS:\t  1000 kB\n', encoding='utf-8')
        monkeypatch.setattr(resources, '_CLEAR_REFS', tmp_path / 'clear_refs')
        monkeypatch.setattr(resources, '_STATUS', status)
        monkeypatch.setattr(resources, '_host_peak_is_per_run', False)
        reset_peak_memory()
        assert (tmp_path / 'clear_refs').read_text(encoding='utf-8') == '5'
        usage = peak_memory()
        assert usage['host_peak_scope'] == 'run'
        assert usage['host_peak_mb'] == pytest.approx(123.456)

    def test_where_it_cannot_reset_the_peak_is_labeled_the_processs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resources, '_CLEAR_REFS', tmp_path / 'missing' / 'clear_refs')
        monkeypatch.setattr(resources, '_host_peak_is_per_run', True)
        reset_peak_memory()
        assert peak_memory().get('host_peak_scope', 'process') == 'process'
