#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
cli.py - The `stabilo-benchmark` command.

Subcommand modules are imported inside their handlers, so `--help` does not import torch, kornia
or ultralytics.

    stabilo-benchmark fetch    <dataset>           Download a registered dataset from Zenodo
    stabilo-benchmark survey   <recordings> -c     Choose Part II's cases before any benchmark runs
    stabilo-benchmark static   <config>            Part I: known-homography benchmark
    stabilo-benchmark prepare  <video>             Build a dynamic case from footage
    stabilo-benchmark oracle   <case>              Cache the reference estimator's homographies
    stabilo-benchmark dynamic  <case> -c <config>   Part II: real-footage benchmark
    stabilo-benchmark annotate-gcp <case>          Click ground control points
    stabilo-benchmark audit    <case>              Bound the reference estimator's error
    stabilo-benchmark report   <results>           Tables and figures for the manuscript

Static campaign: annotate (hbb2obb) -> static -> report.
Dynamic campaign: static -> survey -> prepare -> annotate (hbb2obb) -> oracle -> annotate-gcp
-> audit -> dynamic -> report.

Commands that construct a `Stabilizer` (`needs_opencv` on the subparser) are gated on
`environment.check_opencv`; the others only warn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from stabilo_benchmark import __version__

EPILOG = """\
examples:
  # Part I on the committed sample (results -> data/sample/static/results/smoke)
  stabilo-benchmark static configs/sample/static.yaml -o -v 1

  # Part I on your own scenes: <scenes>/images annotated once with hbb2obb (see the README)
  stabilo-benchmark static configs/songdo/static/detectors.yaml --scenes <scenes> -v 1

  # A one-parameter sweep, one result directory per point
  for deg in 0 15 30 45 60 90; do
    stabilo-benchmark static configs/songdo/static/rotation_sweep.yaml --scenes <scenes> \\
      --set perturbation.geometric.rotation_deg="[$deg,$deg]" --name "rotation_${deg}deg" -o
  done

  # Part II on the committed sample case (GCPs included): no video, no detector weights, no GPU
  stabilo-benchmark oracle  data/sample/dynamic/2022-10-06_D9_AM3_S2 -c configs/sample/dynamic.yaml
  stabilo-benchmark audit   data/sample/dynamic/2022-10-06_D9_AM3_S2
  stabilo-benchmark dynamic data/sample/dynamic/2022-10-06_D9_AM3_S2 -c configs/sample/dynamic.yaml -o -v 1

  # Part II on your own footage: prepare a case, annotate its images with hbb2obb, audit, benchmark
  stabilo-benchmark prepare <video> -o <case> -c configs/songdo/dynamic/offline_masking.yaml
  stabilo-benchmark oracle <case> -c configs/songdo/dynamic/offline_masking.yaml
  stabilo-benchmark annotate-gcp <case>
  stabilo-benchmark audit <case>
  stabilo-benchmark dynamic <case> -c configs/songdo/dynamic/offline_masking.yaml -v 1

  # Rerun a campaign of the paper's data into a directory of your own, to compare with the published one
  stabilo-benchmark dynamic data/songdo/dynamic/<case> -c configs/songdo/dynamic/offline_masking.yaml \\
      --results-dir <mine>/<case>

  # Tables and figures
  stabilo-benchmark report <case>/results/offline_masking --kind dynamic --output <out>
"""


def _reference_arg(value: str) -> Any:
    """'first', 'middle', or a frame index, as `sampling.resolve_reference` expects it."""
    return int(value) if value.lstrip('-').isdigit() else value


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the benchmark runners."""
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing results')
    exclusive.add_argument('-r', '--resume', action='store_true', help='resume, skipping completed runs')
    exclusive.add_argument(
        '--regrid',
        action='store_true',
        help='resume onto a config whose pipeline grid changed: stored runs still in the grid are kept and '
        'renumbered, the others deleted, and only the new grid points run',
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='0=quiet, 1=per-run summary, 2=+warnings, 3=+per-measurement detail (default: 0)',
    )
    parser.add_argument(
        '--set',
        action='append',
        metavar='KEY.PATH=VALUE',
        default=None,
        help='override a config value (YAML-parsed); repeatable. Only existing keys may be set.',
    )
    parser.add_argument('--name', default=None, help="override the config's name, and so the result directory")


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog='stabilo-benchmark',
        description='Reproducible static and dynamic benchmarks for reference-based UAV video stabilization.',
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--version', action='version', version=f"stabilo-benchmark {__version__}")
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    # ---- fetch ------------------------------------------------------------------------
    fetch = subparsers.add_parser(
        'fetch',
        help='download a registered dataset from Zenodo into data/',
        description='Download, verify against the checksums pinned in data/registry.yaml, and '
        'extract the datasets the manuscript campaigns run on. The committed sample '
        'under data/sample/ needs none of this.',
    )
    fetch.add_argument(
        'datasets',
        nargs='*',
        default=None,
        help="dataset names to fetch (default: the one the registry marks 'default: true')",
    )
    fetch.add_argument(
        '-l', '--list', action='store_true', help='list the registered datasets and whether they are present, then exit'
    )
    fetch.add_argument(
        '--check', action='store_true', help='re-verify what is already on disk without downloading anything'
    )
    fetch.add_argument('--all', action='store_true', help='include files the registry marks optional')
    fetch.add_argument('--force', action='store_true', help='re-download even if a file already verifies')
    fetch.add_argument(
        '-y', '--yes', action='store_true', help='do not ask before a large download (the size is still printed)'
    )
    fetch.add_argument(
        '--registry',
        type=Path,
        default=None,
        help='registry file to use (default: data/registry.yaml, searched upwards)',
    )
    fetch.set_defaults(handler=_run_fetch, needs_opencv=False)

    # ---- survey -----------------------------------------------------------------------
    survey = subparsers.add_parser(
        'survey',
        help="choose Part II's cases from candidate recordings, before any benchmark runs",
        description='Read every candidate segment the way the benchmark will, as pairs of its reference '
        'frame with one later frame per survey bin of the dynamic config, and measure where the '
        "reference frame's objects are: found where they were, displaced, displaced together, or gone. "
        'Segments are placed in a 2x2 on moving foreground x coherence, plus stationary controls, and '
        'the design is filled from the pool. A cell that cannot be filled is reported, never padded.',
    )
    survey.add_argument('sources', type=Path, nargs='*', help='candidate videos, or directories of frames')
    survey.add_argument(
        '--pool',
        type=Path,
        nargs='+',
        default=None,
        metavar='SURVEY',
        help="earlier surveys' SURVEY.json, read with the same plan, whose candidates join this one's "
        '(shards of one pool read in parallel)',
    )
    survey.add_argument(
        '-c',
        '--config',
        type=Path,
        required=True,
        help='dynamic config whose sampling plan sets the horizon and the offsets read '
        '(e.g. configs/songdo/dynamic/offline_masking.yaml)',
    )
    survey.add_argument(
        '-w',
        '--work-dir',
        type=Path,
        default=Path('survey'),
        help='where decoded frames and their labels are staged (default: ./survey)',
    )
    survey.add_argument('--output', type=Path, default=None, help='SURVEY.json path (default: <work-dir>/SURVEY.json)')
    survey.add_argument(
        '--no-annotate',
        action='store_true',
        help='require labels_hbb/ and labels_obb/ beside the frames rather than running hbb2obb',
    )
    survey.add_argument('--device', default=None, help="hbb2obb inference device, e.g. '0', 'cpu', 'mps'")
    survey.add_argument(
        '--discard-frames',
        action='store_true',
        help="delete a recording's decoded frames once its segments are described (labels stay; a rerun "
        'decodes them again and checks them against the staging record)',
    )
    survey.add_argument(
        '-k',
        '--per-cell',
        type=int,
        default=2,
        metavar='K',
        help='segments wanted per moving x coherence cell (default: 2)',
    )
    survey.add_argument(
        '--controls',
        type=int,
        default=1,
        metavar='C',
        help='stationary controls wanted: much foreground, little of it moving (default: 1)',
    )
    survey.add_argument(
        '--moving-split',
        type=float,
        default=None,
        metavar='F',
        help='pin the high/low moving-foreground boundary, as a fraction of the frame (default: the pool median)',
    )
    survey.add_argument(
        '--coherence-split',
        type=float,
        default=None,
        metavar='F',
        help='pin the coherent/mixed boundary (default: the pool median)',
    )
    survey.add_argument(
        '--group-pattern',
        default=None,
        metavar='REGEX',
        help="regular expression whose first capture group, searched in a candidate's name, names its "
        "group (a site, say); with it, at most --max-per-group selections per cell share a group",
    )
    survey.add_argument(
        '--max-per-group',
        type=int,
        default=1,
        metavar='M',
        help='selections per cell that may share a --group-pattern group (default: 1)',
    )
    survey.add_argument(
        '--horizon',
        type=int,
        default=None,
        metavar='N',
        help="frames a segment needs after its reference, overriding the config's span",
    )
    long_recordings = survey.add_argument_group(
        'segments',
        "By default a recording is read from its first frame. --anchor scan reads a segment every "
        '--scan-stride frames inside it, clear of its first and last seconds, so a case can be referenced '
        'at any point of a hover.',
    )
    long_recordings.add_argument(
        '--anchor',
        choices=('start', 'scan'),
        default='start',
        help="where a recording's segments start: its first frame (default), or every --scan-stride frames",
    )
    long_recordings.add_argument(
        '--scan-stride',
        type=int,
        default=900,
        metavar='N',
        help='frames between segment starts with --anchor scan (default: 900, ~30 s)',
    )
    long_recordings.add_argument(
        '--max-per-video',
        type=int,
        default=2,
        metavar='M',
        help='most segments selected from one recording (default: 2)',
    )
    long_recordings.add_argument(
        '--min-separation',
        type=int,
        default=3000,
        metavar='N',
        help='frames two selections from one recording must be apart (default: 3000)',
    )
    survey.set_defaults(handler=_run_survey, needs_opencv=True)

    # ---- static -----------------------------------------------------------------------
    static = subparsers.add_parser(
        'static',
        help='Part I: benchmark under known synthetic homographies',
        description='Apply a known homography and photometric degradation to static aerial frames, '
        'then measure how accurately each pipeline recovers the transform.',
    )
    static.add_argument('config', type=Path, help='experiment config (.yaml or .json)')
    static.add_argument('--scenes', type=Path, default=None, help="override the config's scenes directory")
    static.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help='write the campaign to DIR/<name> (default: <scenes>/results/<name>)',
    )
    static.add_argument(
        '--scenes-list',
        type=Path,
        default=None,
        metavar='FILE',
        help='restrict to the scene names listed in FILE, one per line; this is how a '
        'tuning split is kept disjoint from the evaluation scenes',
    )
    static.add_argument(
        '--no-trial-cache',
        dest='trial_cache',
        action='store_false',
        help='regenerate the perturbed frames in every run instead of caching them under '
        '<results-dir>/.cache/<name>/ (about 25 MB per 4K scene and trial, deleted on completion)',
    )
    _add_common(static)
    static.set_defaults(handler=_run_static, needs_opencv=True)

    # ---- prepare ----------------------------------------------------------------------
    prepare = subparsers.add_parser(
        'prepare',
        help='build a dynamic benchmark case from a video or from extracted frames',
        description='Extract the sampled frames of a hovering-drone video into <case>/images, ready '
        'for hbb2obb. With --from-frames, adopt frames that have already been extracted '
        'instead of decoding a video.',
    )
    prepare.add_argument(
        'video', type=Path, nargs='?', default=None, help='source video (omit when using --from-frames)'
    )
    prepare.add_argument(
        '--from-frames',
        type=Path,
        default=None,
        metavar='DIR',
        help='directory of already-extracted frames, named by frame index, instead of a video',
    )
    prepare.add_argument(
        '--frames',
        type=int,
        nargs='+',
        default=None,
        metavar='INDEX',
        help='adopt only these frame indices (--from-frames only)',
    )
    prepare.add_argument(
        '--scale', type=float, default=1.0, help='resize adopted frames by this factor (--from-frames only)'
    )
    prepare.add_argument(
        '--fps',
        type=float,
        default=None,
        help='frame rate of the source recording, so offsets can be reported in seconds',
    )
    prepare.add_argument(
        '--video-name', default=None, help='name of the source recording, recorded as provenance (--from-frames only)'
    )
    prepare.add_argument('-o', '--output', type=Path, default=None, help='case directory to create')
    prepare.add_argument(
        '-c',
        '--config',
        type=Path,
        default=None,
        help='dynamic config to take the sampling plan from (recommended); with --from-frames, its reference '
        'and offset bins',
    )
    prepare.add_argument('--stride', type=int, default=50, help='frame stride when no config is given (default: 50)')
    prepare.add_argument('--max-frames', type=int, default=None, help='cap on the number of sampled frames')
    prepare.add_argument(
        '--reference',
        type=_reference_arg,
        default=None,
        help="'first', 'middle', or a frame index of the source recording; the starting frame of a "
        "surveyed segment goes here (default: the config's `reference`, else first)",
    )
    prepare.add_argument('--seed', type=int, default=0, help='sampling seed (default: 0)')
    prepare.add_argument('--jpeg-quality', type=int, default=95, help='JPEG quality for extracted frames (default: 95)')
    prepare.add_argument(
        '--overwrite',
        action='store_true',
        help='rebuild the frames of an existing case; refused while labels, points, caches or results made '
        'from its current frames remain',
    )
    prepare.add_argument(
        '--cell',
        default=None,
        help="the survey cell this case fills (e.g. 'high moving / coherent', 'stationary control'); "
        'recorded in CASE.json and with every measurement, so the scenario tables group on it',
    )
    prepare.set_defaults(handler=_run_prepare, needs_opencv=False)

    # ---- oracle -----------------------------------------------------------------------
    oracle = subparsers.add_parser(
        'oracle',
        help='cache the reference estimator (pseudo-ground truth) for a case',
        description='Run the best static pipeline at full resolution over every frame pair and cache its '
        'homographies, so all candidates are scored against identical geometry. '
        'This is a reference estimator, not ground truth: measure its error with `audit`.',
    )
    oracle.add_argument('case', type=Path, help='prepared case directory')
    oracle.add_argument(
        '-c', '--config', type=Path, default=None, help='dynamic config whose `oracle` block and sampling plan to use'
    )
    oracle.add_argument('--preset', default='oracle', help='pipeline preset when no config is given (default: oracle)')
    oracle.add_argument(
        '--masking',
        default='obb',
        choices=['none', 'hbb', 'obb'],
        help='masking strategy for the reference estimator when no config is given (default: obb)',
    )
    oracle.add_argument(
        '--set',
        action='append',
        metavar='KEY.PATH=VALUE',
        default=None,
        help='override a value of the config given with -c (YAML-parsed); repeatable, e.g. --set oracle.masking=none',
    )
    oracle.add_argument('--overwrite', action='store_true', help='recompute cached estimates')
    oracle.add_argument('-v', '--verbosity', type=int, default=0, choices=[0, 1, 2, 3], help='verbosity (default: 0)')
    oracle.set_defaults(handler=_run_oracle, needs_opencv=True)

    # ---- dynamic ----------------------------------------------------------------------
    dynamic = subparsers.add_parser(
        'dynamic',
        help='Part II: benchmark on real hovering-drone footage',
        description='Register sampled frames directly onto the reference frame and score them against the '
        'reference estimator and against masked background-alignment metrics.',
    )
    dynamic.add_argument('case', type=Path, help='prepared case directory')
    dynamic.add_argument('-c', '--config', type=Path, required=True, help='dynamic experiment config')
    dynamic.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help='write the campaign to DIR/<name> (default: <case>/results/<name>)',
    )
    _add_common(dynamic)
    dynamic.set_defaults(handler=_run_dynamic, needs_opencv=True)

    # ---- annotate-gcp -----------------------------------------------------------------
    annotate = subparsers.add_parser(
        'annotate-gcp',
        help='click ground control points for the audit',
        description='Annotate ground-plane control points in the reference frame and in selected frames. '
        'Needs an OpenCV build with GUI support; LabelMe JSON dropped into the case\'s gcp/ '
        'directory works just as well.',
    )
    annotate.add_argument('case', type=Path, help='prepared case directory')
    annotate.add_argument(
        '--frames',
        type=int,
        nargs='+',
        default=None,
        help='frame indices to annotate (default: two sampled frames per audit offset bin of the case)',
    )
    annotate.add_argument('--overwrite', action='store_true', help='re-annotate frames that already have points')
    annotate.add_argument(
        '--labels',
        nargs='+',
        default=None,
        metavar='LABEL',
        help='re-place only these points (e.g. p02) in the --frames given, keeping every other point; '
        'the reference frame may be listed, and s removes a point from a frame where it is not visible',
    )
    annotate.add_argument(
        '--zoom', type=float, default=4.0, choices=(2.0, 4.0, 8.0), help='magnification of the placement panel'
    )
    annotate.add_argument(
        '--propose',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='open each point of a frame after the reference on a proposal (a local ECC fit of its '
        'neighborhood, independent of the reference estimator), placed for Enter to accept; the repeat '
        'session never proposes (default: on)',
    )
    annotate.add_argument(
        '--repeat',
        action='store_true',
        help='a second, independent annotation session of the same frames, written to gcp_repeat/ so '
        'audit can report the click repeatability; do not look at the first session while placing points',
    )
    annotate.add_argument(
        '--manual',
        action='store_true',
        help="place the first session's frames again on its reference points, without proposals, written to "
        'gcp_manual/: unaided clicks to compare with the proposals; do not look at the first session meanwhile',
    )
    annotate.set_defaults(handler=_run_annotate, needs_opencv=False)

    # ---- audit ------------------------------------------------------------------------
    audit = subparsers.add_parser(
        'audit',
        help="measure the reference estimator's error against ground control points",
        description='Map hand-annotated ground-plane points through the cached homographies and report the '
        'residuals. The 95th percentile is the resolution floor of Part II: differences below '
        'it are not evidence.',
    )
    audit.add_argument('case', type=Path, help='prepared case directory')
    audit.add_argument('--gcp', type=Path, default=None, help="point directory (default: <case>/gcp)")
    audit.add_argument(
        '--results',
        type=Path,
        default=None,
        help='also audit the candidate pipelines in this results directory against the same points',
    )
    audit.set_defaults(handler=_run_audit, needs_opencv=False)

    # ---- report -----------------------------------------------------------------------
    report = subparsers.add_parser(
        'report',
        help='generate manuscript tables and figures from saved results',
        description='Aggregate a results directory into LaTeX tables and PDF figures. Nothing in the '
        'manuscript is typed by hand.',
    )
    report.add_argument(
        'results',
        type=Path,
        nargs='+',
        help="results directory, several dynamic ones to pool over cases (equal weight per case), or a "
        "survey's SURVEY.json for the table of selected cases, or an hbb2obb-optimize benchmark_results/ "
        "folder for the oriented-box appendix",
    )
    report.add_argument(
        '--kind',
        choices=['static', 'dynamic'],
        default=None,
        help="which protocol produced these results (default: read from the manifest)",
    )
    report.add_argument(
        '--output',
        type=Path,
        default=None,
        help='destination root; tables/ and figures/ are created inside (default: reports/)',
    )
    report.add_argument('--prefix', default=None, help="filename prefix (default: the results directory's name)")
    report.add_argument(
        '--audit-case',
        default=None,
        metavar='CASE',
        help="case name that illustrates the pooled GCP audit figure (default: the first audited case by name)",
    )
    report.set_defaults(handler=_run_report, needs_opencv=False)

    return parser


# --------------------------------------------------------------------------------------
# Handlers (imports are local so each command pays only for what it uses)
# --------------------------------------------------------------------------------------


def _run_survey(args: argparse.Namespace) -> int:
    from stabilo_benchmark.annotation import AnnotationError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.survey import SurveyError, run_survey  # noqa: PLC0415 - lazy dispatch

    try:
        run_survey(args.sources, args)
    except (SurveyError, AnnotationError) as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_static(args: argparse.Namespace) -> int:
    from stabilo_benchmark.config import ConfigError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.results import ResultError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.static import StaticBenchmarkError, run_static  # noqa: PLC0415 - lazy dispatch

    # Expected errors print their message and exit non-zero; anything else propagates.
    try:
        run_static(args.config, args)
    except (StaticBenchmarkError, ConfigError, ResultError) as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_fetch(args: argparse.Namespace) -> int:
    from stabilo_benchmark.datasets import RegistryError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.fetch import FetchError, run_fetch  # noqa: PLC0415 - lazy dispatch

    # Expected errors print their message and exit non-zero; anything else propagates.
    try:
        return run_fetch(args)
    except (FetchError, RegistryError) as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1


def _run_prepare(args: argparse.Namespace) -> int:
    from stabilo_benchmark.runners.prepare import (  # noqa: PLC0415 - lazy dispatch
        PrepareError,
        run_prepare,
        run_prepare_from_frames,
    )

    if bool(args.video) == bool(args.from_frames):
        raise SystemExit(
            'prepare needs exactly one source: a video, or --from-frames <dir> for frames that are already extracted.'
        )
    if args.frames and not args.from_frames:
        raise SystemExit('--frames selects frames to adopt and needs --from-frames; -c or --stride samples a video.')
    try:
        if args.from_frames:
            run_prepare_from_frames(args.from_frames, args)
        else:
            run_prepare(args.video, args)
    except PrepareError as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_oracle(args: argparse.Namespace) -> int:
    from stabilo_benchmark.config import apply_overrides, build_oracle, load_config  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.oracle import OracleError, compute_oracle  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.pipelines import make_spec  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.dynamic import plan_pairs  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.prepare import load_case  # noqa: PLC0415 - lazy dispatch

    case = load_case(args.case)
    if args.set and not args.config:
        print("\033[91m--set overrides a config value; give the config with -c.\033[0m", file=sys.stderr)
        return 1
    if args.config:
        config = apply_overrides(load_config(args.config), args.set)
        spec = build_oracle(config)
        pairs = plan_pairs(case, config)
    else:
        spec = make_spec({'preset': args.preset, 'masking': args.masking})
        # Without a config, use the sampling plan the case was prepared with.
        from stabilo_benchmark.sampling import FramePair  # noqa: PLC0415 - lazy dispatch

        reference_index = int(case['reference_index'])
        pairs = [
            FramePair(reference_index=reference_index, current_index=int(index))
            for index in case['frames']
            if int(index) != reference_index
        ]

    try:
        compute_oracle(args.case, pairs, spec, overwrite=args.overwrite, verbosity=args.verbosity)
    except OracleError as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_dynamic(args: argparse.Namespace) -> int:
    from stabilo_benchmark.config import ConfigError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.oracle import OracleError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.results import ResultError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.dynamic import DynamicBenchmarkError, run_dynamic  # noqa: PLC0415 - lazy dispatch

    # Expected errors print their message and exit non-zero; anything else propagates.
    try:
        run_dynamic(args.case, args.config, args)
    except (DynamicBenchmarkError, ConfigError, OracleError, ResultError) as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_annotate(args: argparse.Namespace) -> int:
    from stabilo_benchmark.report.annotate_gcp import run_annotate  # noqa: PLC0415 - lazy dispatch

    run_annotate(args.case, args)
    return 0


def _run_audit(args: argparse.Namespace) -> int:
    from stabilo_benchmark.gcp import GcpError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.oracle import OracleError  # noqa: PLC0415 - lazy dispatch
    from stabilo_benchmark.runners.audit import AuditError, run_audit  # noqa: PLC0415 - lazy dispatch

    # Expected errors print their message and exit non-zero; anything else propagates.
    try:
        run_audit(args.case, args)
    except (AuditError, GcpError, OracleError) as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from stabilo_benchmark.report.build import build_report  # noqa: PLC0415 - lazy dispatch

    build_report(list(args.results), kind=args.kind, output=args.output, prefix=args.prefix, audit_case=args.audit_case)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the `stabilo-benchmark` console script.

    The OpenCV check runs after parsing and before the handler is dispatched.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'handler', None):
        parser.print_help()
        return 1

    from stabilo_benchmark.environment import EnvironmentCheckError, check_opencv  # noqa: PLC0415 - lazy dispatch

    try:
        check_opencv(strict=getattr(args, 'needs_opencv', True))
    except EnvironmentCheckError as e:
        print(f"\033[91m{e}\033[0m", file=sys.stderr)
        return 1

    return args.handler(args)


if __name__ == '__main__':
    sys.exit(main())
