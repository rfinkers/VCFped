"""
Tests for vcfped/vcfped.py and vcfped/vcfped_optimized.py.

These run the bundled test VCF (HG002 child of HG003+HG004) through both
engines and verify:
  - Known pedigree confirmation (-ped trio.ped -> "OK")
  - Pairwise / trio output values match expected ground truth
  - Both engines produce byte-identical results (regression guard — earlier
    in development, vcfped_optimized.py had a missing-genotype bug and a
    trio-pivot-mislabelling bug that produced DIFFERENT results from
    vcfped.py for the same input)
  - gzip input support (vcfped_optimized.py only)
  - CLI flags (--no-gender, --no-pairwise, --no-trio) work
"""
import os
import filecmp
import pytest

from conftest import run_module, REPO_ROOT, HAS_NUMPY


ENGINES = ['vcfped.vcfped', 'vcfped.vcfped_optimized']


def _skip_optimized_without_numpy(engine):
    """
    vcfped_optimized.py has a hard `import numpy as np` (not wrapped in
    try/except, unlike numba/tqdm which DO degrade gracefully) — see
    CLAUDE.md. Skip cleanly rather than failing when numpy is absent.
    """
    if engine == 'vcfped.vcfped_optimized' and not HAS_NUMPY:
        pytest.skip("vcfped_optimized.py requires numpy (hard dependency)")


@pytest.mark.parametrize('engine', ENGINES)
def test_pedigree_confirmation(work_dir, test_vcf, test_ped, engine):
    """Both engines must confirm the known HG002/HG003/HG004 trio."""
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix, '-j', '2',
                         '-ped', test_ped])
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout
    assert 'FAIL' not in result.stdout


@pytest.mark.parametrize('engine', ENGINES)
def test_pairwise_output_values(work_dir, test_vcf, engine):
    """Known ground-truth pairwise values for the bundled test VCF."""
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr

    pair_file = f"{prefix}.pair"
    assert os.path.exists(pair_file)
    with open(pair_file) as f:
        content = f.read()

    # Both pairs (1,2) and (1,3) are confirmed Parent-child at Perc=10
    assert '1,2\t10\t1905\t1194\t906\t75.9\t595\t595\t100.0\tParent-child' in content
    assert '1,3\t10\t1905\t1127\t868\t77.0\t613\t613\t100.0\tParent-child' in content


@pytest.mark.parametrize('engine', ENGINES)
def test_trio_output_values(work_dir, test_vcf, engine):
    """Known ground-truth trio values for the bundled test VCF."""
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr

    trio_file = f"{prefix}.trio"
    assert os.path.exists(trio_file)
    with open(trio_file) as f:
        content = f.read()

    # Sample 1 (HG002) is the pivot/child; T1=100.0%, T2=99.3%
    assert '1,2,3\t1\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio' in content


def test_engines_produce_identical_output(work_dir, test_vcf):
    """
    Regression guard: vcfped.py and vcfped_optimized.py must produce
    byte-identical .pair/.trio/.gender files for the same input.
    Catches the missing-genotype-handling bug and the trio pivot
    mislabelling bug found during development (different engines silently
    disagreeing on results is the worst possible failure mode here).
    """
    if not HAS_NUMPY:
        pytest.skip("vcfped_optimized.py requires numpy (hard dependency)")
    prefix_a = os.path.join(work_dir, 'orig')
    prefix_b = os.path.join(work_dir, 'opt')

    result_a = run_module(['vcfped.vcfped', test_vcf, '-o', prefix_a, '-j', '2'])
    result_b = run_module(['vcfped.vcfped_optimized', test_vcf, '-o', prefix_b, '-j', '2'])
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr

    for ext in ('.pair', '.trio', '.gender'):
        file_a, file_b = f"{prefix_a}{ext}", f"{prefix_b}{ext}"
        assert os.path.exists(file_a), f"{file_a} missing"
        assert os.path.exists(file_b), f"{file_b} missing"
        assert filecmp.cmp(file_a, file_b, shallow=False), \
            f"{ext} differs between vcfped.py and vcfped_optimized.py"


def test_gzip_input_matches_plain(work_dir, test_vcf, test_vcf_gz):
    """
    vcfped_optimized.py must produce identical results for .vcf and .vcf.gz
    input. Regression guard for an off-by-one bug found during development:
    the gzip header-skip logic originally processed one extra variant
    compared to the plain-text path (skipped only the header line, not also
    the partially-consumed first data line that the plain-text seek+readline
    path skips).
    """
    if not HAS_NUMPY:
        pytest.skip("vcfped_optimized.py requires numpy (hard dependency)")
    prefix_plain = os.path.join(work_dir, 'plain')
    prefix_gz = os.path.join(work_dir, 'gz')

    result_plain = run_module(['vcfped.vcfped_optimized', test_vcf,
                               '-o', prefix_plain, '-j', '2'])
    result_gz = run_module(['vcfped.vcfped_optimized', test_vcf_gz,
                            '-o', prefix_gz, '-j', '2'])
    assert result_plain.returncode == 0, result_plain.stderr
    assert result_gz.returncode == 0, result_gz.stderr

    for ext in ('.pair', '.trio', '.gender'):
        assert filecmp.cmp(f"{prefix_plain}{ext}", f"{prefix_gz}{ext}", shallow=False), \
            f"{ext} differs between plain and gzip input"


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('flag,missing_file', [
    ('--no-gender', '.gender'),
    ('--no-pairwise', '.pair'),
    ('--no-trio', '.trio'),
])
def test_skip_flags(work_dir, test_vcf, engine, flag, missing_file):
    """--no-gender / --no-pairwise / --no-trio should skip that output file."""
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix, '-j', '2', flag])
    assert result.returncode == 0, result.stderr
    assert not os.path.exists(f"{prefix}{missing_file}"), \
        f"{flag} should have prevented {missing_file} from being written"


@pytest.mark.parametrize('engine', ENGINES)
def test_jobs_flag_default_is_not_hardcoded(work_dir, test_vcf, engine):
    """
    Regression guard: vcfped_optimized.py once had -j default hardcoded to 8
    instead of "all cores". Verify omitting -j still runs successfully
    (can't directly assert core count from output, but a crash here would
    indicate the default is broken).
    """
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{prefix}.trio")
