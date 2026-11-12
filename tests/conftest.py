"""
Session setup shared by the test modules.

The sample case's reference-estimator cache (`data/sample/dynamic/<case>/oracle/`) is an output, ignored by
git, so a fresh checkout lacks it. It is computed here, as the quick start does, before collection,
so the tests that read it run instead of being skipped.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CASE = REPO_ROOT / 'data' / 'sample' / 'dynamic' / '2022-10-06_D9_AM3_S2'
SAMPLE_CONFIG = REPO_ROOT / 'configs' / 'sample' / 'dynamic.yaml'


def pytest_configure(config):
    if not (SAMPLE_CASE / 'CASE.json').is_file() or (SAMPLE_CASE / 'oracle').is_dir():
        return
    result = subprocess.run(
        [sys.executable, '-m', 'stabilo_benchmark', 'oracle', str(SAMPLE_CASE), '-c', str(SAMPLE_CONFIG)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"`oracle` on the sample case failed:\n{result.stdout}\n{result.stderr}")
