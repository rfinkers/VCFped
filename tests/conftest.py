"""
Shared pytest fixtures and helpers for the VCFped test suite.

The `utilities/` scripts are not an importable package (no __init__.py,
designed to be run with `python utilities/foo.py`), so tests load them
dynamically via importlib — the same approach used throughout development
of these scripts. `import_module_from_path()` is the single helper that
makes this consistent across all test files.
"""
import importlib.util
import os
import subprocess
import sys
import gzip
import shutil
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILITIES_DIR = os.path.join(REPO_ROOT, 'utilities')
VCFPED_DIR = os.path.join(REPO_ROOT, 'vcfped')
TESTFILES_DIR = os.path.join(VCFPED_DIR, 'testfiles')

TEST_VCF = os.path.join(TESTFILES_DIR, 'trioHG002_22X.vcf')
TEST_PED = os.path.join(TESTFILES_DIR, 'trio.ped')


def import_module_from_path(name, path):
    """Dynamically import a script (not part of a package) as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(path, args, cwd=None, timeout=120, stdin_input=None):
    """
    Run a utility/vcfped script as a subprocess (mirrors real CLI usage).
    Returns CompletedProcess with .returncode, .stdout, .stderr.

    stdin_input: text to feed to the process's stdin, for scripts that
        prompt for interactive confirmation (e.g. --inplace).
    """
    cmd = [sys.executable, path] + [str(a) for a in args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, input=stdin_input)


def run_module(module_args, cwd=None, timeout=300):
    """
    Run a vcfped package module via `python -m vcfped.<module>` (mirrors
    documented usage), e.g. module_args=['vcfped.vcfped_optimized', 'file.vcf', '-o', 'out']
    """
    cmd = [sys.executable, '-m'] + [str(a) for a in module_args]
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True,
                          text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Optional-dependency skip markers
# ---------------------------------------------------------------------------

def _has_module(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

HAS_NUMPY = _has_module('numpy')
HAS_NUMBA = _has_module('numba')
HAS_PANDAS = _has_module('pandas')
HAS_PLOTLY = _has_module('plotly')

requires_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
requires_plotly = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def work_dir(tmp_path):
    """A scratch directory for a single test, as a string path."""
    return str(tmp_path)


@pytest.fixture
def test_vcf():
    assert os.path.exists(TEST_VCF), f"Bundled test VCF not found: {TEST_VCF}"
    return TEST_VCF


@pytest.fixture
def test_ped():
    assert os.path.exists(TEST_PED), f"Bundled test PED not found: {TEST_PED}"
    return TEST_PED


@pytest.fixture
def test_vcf_gz(work_dir, test_vcf):
    """Gzipped copy of the bundled test VCF, for .vcf.gz handling tests."""
    gz_path = os.path.join(work_dir, 'trioHG002_22X.vcf.gz')
    with open(test_vcf, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gz_path


@pytest.fixture
def vcfped_run(work_dir, test_vcf, test_ped):
    """
    Run vcfped.py (the pure-Python reference engine) once on the bundled
    trio test VCF and return the output prefix. Several test modules need
    this as a starting point (filter_trios, rename_samples,
    prepare_plink_beagle, fixed_pedigree_viz, etc.) — these utilities only
    need SOME valid vcfped output, not engine-specific behaviour (that is
    covered separately and thoroughly in test_vcfped_core.py, which tests
    both engines including their cross-engine agreement).

    Deliberately uses vcfped.py rather than vcfped_optimized.py: the latter
    has a hard numpy dependency (not optional, despite the genotype matrix
    machinery being framed as an "accelerator" — see CLAUDE.md), so using
    the original engine here keeps all downstream utility tests runnable
    in numpy-less environments.
    """
    prefix = os.path.join(work_dir, 'step3')
    result = run_module(
        ['vcfped.vcfped', test_vcf, '-o', prefix, '-j', '2', '-ped', test_ped],
    )
    assert result.returncode == 0, f"vcfped failed:\n{result.stdout}\n{result.stderr}"
    return prefix
