#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""CLI surface: every subcommand parses, and `--help` does not import heavy modules."""

from __future__ import annotations

import subprocess
import sys

import pytest

from stabilo_benchmark import __version__, environment
from stabilo_benchmark.cli import build_parser, main

# Every registered subcommand; keep exhaustive.
SUBCOMMANDS = [
    'fetch',
    'survey',
    'static',
    'prepare',
    'oracle',
    'annotate-gcp',
    'audit',
    'dynamic',
    'report',
]


class TestParser:
    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        actions = [a for a in parser._subparsers._group_actions if hasattr(a, 'choices')]
        registered = set(actions[0].choices)
        assert set(SUBCOMMANDS) <= registered

    @pytest.mark.parametrize('command', SUBCOMMANDS)
    def test_subcommand_help_exits_cleanly(self, command, capsys):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args([command, '--help'])
        assert exit_info.value.code == 0
        assert capsys.readouterr().out

    def test_top_level_help_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(['--help'])
        assert exit_info.value.code == 0
        assert 'stabilo-benchmark' in capsys.readouterr().out

    def test_version_reports_the_package_version(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--version'])
        assert __version__ in capsys.readouterr().out

    def test_no_command_prints_help_and_fails(self, capsys):
        assert main([]) == 1
        assert 'usage' in capsys.readouterr().out.lower()

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['teleport'])


class TestArgumentWiring:
    def test_static_accepts_the_shared_flags(self):
        args = build_parser().parse_args(
            ['static', 'c.yaml', '-o', '-v', '2', '--set', 'trials=5', '--name', 'x', '--scenes', 's']
        )
        assert args.overwrite and args.verbosity == 2
        assert args.set == ['trials=5'] and args.name == 'x'

    def test_set_is_repeatable(self):
        args = build_parser().parse_args(['static', 'c.yaml', '--set', 'a=1', '--set', 'b=2'])
        assert args.set == ['a=1', 'b=2']

    def test_overwrite_and_resume_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['static', 'c.yaml', '-o', '-r'])

    def test_dynamic_requires_a_config(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['dynamic', 'case'])

    def test_dynamic_parses_case_and_config(self):
        args = build_parser().parse_args(['dynamic', 'case', '-c', 'cfg.yaml'])
        assert str(args.case) == 'case' and str(args.config) == 'cfg.yaml'

    def test_prepare_defaults_are_sensible(self):
        args = build_parser().parse_args(['prepare', 'video.mp4'])
        assert args.stride == 50 and args.reference is None  # the config decides, else 'first'

    def test_prepare_reference_accepts_a_frame_index(self):
        # `survey --select` hands a segment over as `--reference <frame>`.
        args = build_parser().parse_args(['prepare', 'video.mp4', '--reference', '4500'])
        assert args.reference == 4500
        assert build_parser().parse_args(['prepare', 'video.mp4', '--reference', 'middle']).reference == 'middle'

    def test_prepare_frames_selects_adopted_frames_only(self):
        args = build_parser().parse_args(['prepare', '--from-frames', 'frames', '--frames', '300', '302'])
        assert args.frames == [300, 302]
        with pytest.raises(SystemExit, match='needs --from-frames'):
            main(['prepare', 'video.mp4', '--frames', '300', '302'])

    @pytest.mark.parametrize('command', ['annotate', 'obb-eval'])
    def test_annotation_and_conversion_are_not_subcommands(self, command):
        # Both are hbb2obb's own commands; see annotation.py.
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, 'images'])

    def test_oracle_masking_choices_are_constrained(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['oracle', 'case', '--masking', 'rotated'])

    def test_report_kind_choices_are_constrained(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['report', 'results', '--kind', 'sideways'])

    def test_annotate_accepts_a_frame_list(self):
        args = build_parser().parse_args(['annotate-gcp', 'case', '--frames', '10', '20', '30'])
        assert args.frames == [10, 20, 30]


class TestTheOpenCVGate:
    """Which subcommands refuse to start on an OpenCV without BRISK, KAZE and AKAZE."""

    STRICT = ('static', 'dynamic', 'oracle', 'survey')
    ADVISORY = ('fetch', 'report', 'prepare', 'annotate-gcp', 'audit')

    @pytest.mark.parametrize('command', STRICT)
    def test_a_command_that_runs_a_pipeline_is_gated(self, command):
        assert _needs_opencv(command) is True

    @pytest.mark.parametrize('command', ADVISORY)
    def test_a_command_that_runs_no_pipeline_is_not(self, command):
        assert _needs_opencv(command) is False

    def test_every_subcommand_declares_one_or_the_other(self):
        # Every subcommand states `needs_opencv`.
        subparsers = build_parser()._subparsers._group_actions[0].choices
        undeclared = [name for name in subparsers if 'needs_opencv' not in subparsers[name]._defaults]
        assert undeclared == []

    def test_a_refusal_is_reported_and_exits_non_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(environment, 'missing_detectors', lambda: ['brisk', 'kaze', 'akaze'])
        monkeypatch.delenv(environment.ALLOW_VARIABLE, raising=False)
        assert main(['static', 'nonexistent.yaml']) == 1
        assert 'BRISK' in capsys.readouterr().err


def _needs_opencv(command: str) -> bool:
    """What `main` would read off this subcommand's parsed namespace."""
    subparsers = build_parser()._subparsers._group_actions[0].choices
    return subparsers[command]._defaults['needs_opencv']


class TestStartupCost:
    def test_help_does_not_import_torch_or_ultralytics(self):
        code = (
            'import sys; from stabilo_benchmark.cli import build_parser; build_parser();'
            "print(','.join(m for m in ('torch', 'kornia', 'ultralytics', 'matplotlib') if m in sys.modules))"
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == ''

    def test_console_script_help_runs(self):
        result = subprocess.run(
            [sys.executable, '-m', 'stabilo_benchmark', '--help'], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0
        assert 'stabilo-benchmark' in result.stdout
