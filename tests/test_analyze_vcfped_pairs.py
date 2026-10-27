"""
Tests for utilities/analyze_vcfped_pairs.py.

Smoke tests using the bundled test VCF's real .pair output (no synthetic
MZ-twin pairs needed — this script just needs a well-formed .pair file).
"""
import os
import pytest

from conftest import run_script, UTILITIES_DIR, requires_pandas, requires_numpy

ANALYZE_SCRIPT = os.path.join(UTILITIES_DIR, 'analyze_vcfped_pairs.py')


@requires_pandas
@requires_numpy
def test_analyze_runs_on_real_pair_output(vcfped_run):
    result = run_script(ANALYZE_SCRIPT, [vcfped_run])
    assert result.returncode == 0, result.stderr
    assert 'Total pairs in file' in result.stdout


@requires_pandas
@requires_numpy
def test_analyze_with_threshold(vcfped_run):
    result = run_script(ANALYZE_SCRIPT, [vcfped_run, '--threshold', '0.95'])
    assert result.returncode == 0, result.stderr


@requires_pandas
@requires_numpy
def test_analyze_missing_file_fails_cleanly(tmp_path):
    result = run_script(ANALYZE_SCRIPT, [str(tmp_path / "nonexistent")])
    assert result.returncode != 0
