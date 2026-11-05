"""
Tests for vcfped/vcfped.py (maintained engine) and vcfped/archive/vcfped_original.py
(archived pure-Python reference engine, kept to check that both agree).

These run the bundled test VCF (HG002 child of HG003+HG004) through both
engines and verify:
  - Known pedigree confirmation (-ped trio.ped -> "OK")
  - Pairwise / trio output values match expected ground truth
  - Both engines produce byte-identical results (regression guard — earlier
    in development, the optimized engine had a missing-genotype bug and a
    trio-pivot-mislabelling bug that produced DIFFERENT results from
    the original for the same input)
  - gzip input support (vcfped.py only)
  - CLI flags (--no-gender, --no-pairwise, --no-trio) work
"""
import os
import filecmp
import pytest

from conftest import run_module, REPO_ROOT, HAS_NUMPY


ENGINES = ['vcfped.archive.vcfped_original', 'vcfped.vcfped']


def _skip_optimized_without_numpy(engine):
    """
    vcfped.py has a hard `import numpy as np` (not wrapped in
    try/except, unlike numba/tqdm which DO degrade gracefully) — see
    CLAUDE.md. Skip cleanly rather than failing when numpy is absent.
    """
    if engine == 'vcfped.vcfped' and not HAS_NUMPY:
        pytest.skip("vcfped.py requires numpy (hard dependency)")


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
    assert '1,2\t10\t1906\t1194\t906\t75.9\t595\t595\t100.0\tParent-child' in content
    assert '1,3\t10\t1906\t1127\t868\t77.0\t613\t613\t100.0\tParent-child' in content


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
    assert '1,2,3\t1\t10\t1906\t136\t136\t100.0\t269\t267\t99.3\tRegular trio' in content


def test_engines_produce_identical_output(work_dir, test_vcf):
    """
    Regression guard: vcfped.py and the archived original must produce
    byte-identical .pair/.trio/.gender files for the same input.
    Catches the missing-genotype-handling bug and the trio pivot
    mislabelling bug found during development (different engines silently
    disagreeing on results is the worst possible failure mode here).
    """
    if not HAS_NUMPY:
        pytest.skip("vcfped.py requires numpy (hard dependency)")
    prefix_a = os.path.join(work_dir, 'orig')
    prefix_b = os.path.join(work_dir, 'opt')

    result_a = run_module(['vcfped.archive.vcfped_original', test_vcf, '-o', prefix_a, '-j', '2'])
    result_b = run_module(['vcfped.vcfped', test_vcf, '-o', prefix_b, '-j', '2'])
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr

    for ext in ('.pair', '.trio', '.gender'):
        file_a, file_b = f"{prefix_a}{ext}", f"{prefix_b}{ext}"
        assert os.path.exists(file_a), f"{file_a} missing"
        assert os.path.exists(file_b), f"{file_b} missing"
        assert filecmp.cmp(file_a, file_b, shallow=False), \
            f"{ext} differs between vcfped.py and the archived original"


def test_gzip_input_matches_plain(work_dir, test_vcf, test_vcf_gz):
    """
    vcfped.py must produce identical results for .vcf and .vcf.gz
    input. Regression guard for an off-by-one bug found during development:
    the gzip header-skip logic originally processed one extra variant
    compared to the plain-text path (skipped only the header line, not also
    the partially-consumed first data line that the plain-text seek+readline
    path skips).
    """
    if not HAS_NUMPY:
        pytest.skip("vcfped.py requires numpy (hard dependency)")
    prefix_plain = os.path.join(work_dir, 'plain')
    prefix_gz = os.path.join(work_dir, 'gz')

    result_plain = run_module(['vcfped.vcfped', test_vcf,
                               '-o', prefix_plain, '-j', '2'])
    result_gz = run_module(['vcfped.vcfped', test_vcf_gz,
                            '-o', prefix_gz, '-j', '2'])
    assert result_plain.returncode == 0, result_plain.stderr
    assert result_gz.returncode == 0, result_gz.stderr

    for ext in ('.pair', '.trio', '.gender'):
        assert filecmp.cmp(f"{prefix_plain}{ext}", f"{prefix_gz}{ext}", shallow=False), \
            f"{ext} differs between plain and gzip input"


@pytest.mark.parametrize('engine', ENGINES)
def test_non_pass_records_are_excluded(work_dir, engine):
    """
    Regression guard: the old test `'\\t.\\t' in line` also matched an ID or
    INFO of '.', so LowQual records leaked in. Only FILTER == PASS or '.' may
    be analysed.
    """
    _skip_optimized_without_numpy(engine)
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n")
    gts = "GT:DP:GQ\t0/0:30:50\t0/1:30:50\t1/1:30:50"
    filters = ['PASS', 'PASS', '.', 'LowQual', 'LowQual', 'q10']
    rows = "".join(f"1\t{100 + i}\t.\tA\tG\t50\t{flt}\t.\t{gts}\n"
                   for i, flt in enumerate(filters))
    path = os.path.join(work_dir, 'filters.vcf')
    with open(path, 'w') as f:
        f.write(header + rows)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, path, '-o', prefix, '-j', '1', '--no-gender',
                         '--no-trio', '--all'])
    assert result.returncode == 0, result.stderr
    with open(f"{prefix}.pair") as f:
        autos = {line.split('\t')[2] for line in f.read().splitlines()[1:]}
    assert autos == {'3'}, f"analysed {autos} records; expected 3 (PASS, PASS, '.')"


@pytest.mark.parametrize('engine', ENGINES)
def test_log_states_best_of_percentiles_selection(work_dir, test_vcf, engine):
    """The log header must say that the best row over the tried percentiles is
    kept (more percentiles = more chances to cross a threshold)."""
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix, '-j', '1', '-p', '10', '50'])
    assert result.returncode == 0, result.stderr
    with open(f"{prefix}.log") as f:
        log = f.read()
    assert 'best of 2 percentile(s) kept per pair/trio' in log


@pytest.mark.parametrize('engine,gz', [
    ('vcfped.archive.vcfped_original', False),
    ('vcfped.vcfped', False),
    ('vcfped.vcfped', True),   # the archived original has no gzip support
])
def test_every_pass_record_is_analysed(work_dir, engine, gz):
    """
    Regression guard: Autos (sites left after the quality filter) must equal
    the number of PASS autosomal records in the file. Previously the first data
    record was silently dropped (plain files) and the first TWO were dropped
    (.vcf.gz), so every run analysed one or two variants too few. The bundled
    HG002 trio cannot catch this (its first records do not change any count),
    hence a synthetic file with a known number of records.
    """
    _skip_optimized_without_numpy(engine)
    n_records = 12
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n")
    rows = "".join(
        f"1\t{100 + i}\t.\tA\tG\t50\tPASS\t.\tGT:DP:GQ\t0/0:30:50\t0/1:30:50\t1/1:30:50\n"
        for i in range(n_records))
    path = os.path.join(work_dir, 'known.vcf' + ('.gz' if gz else ''))
    if gz:
        import gzip
        with gzip.open(path, 'wt') as f:
            f.write(header + rows)
    else:
        with open(path, 'w') as f:
            f.write(header + rows)

    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, path, '-o', prefix, '-j', '1', '--no-gender', '--no-trio',
                         '--all'])  # 12 sites are too few for any verdict; --all lists every pair
    assert result.returncode == 0, result.stderr
    with open(f"{prefix}.pair") as f:
        autos = {line.split('\t')[2] for line in f.read().splitlines()[1:]}
    assert autos == {str(n_records)}, f"analysed {autos} of {n_records} PASS records"


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
    Regression guard: the optimized engine once had -j default hardcoded to 8
    instead of "all cores". Verify omitting -j still runs successfully
    (can't directly assert core count from output, but a crash here would
    indicate the default is broken).
    """
    _skip_optimized_without_numpy(engine)
    prefix = os.path.join(work_dir, 'out')
    result = run_module([engine, test_vcf, '-o', prefix])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{prefix}.trio")


@pytest.mark.parametrize('engine', ENGINES)
def test_t2_threshold_controls_inverted_label(work_dir, test_vcf, engine):
    """
    'Inverted/generational trio' only means T2 (BB x BB -> BB) < -t2; it is not
    evidence of a reversed relationship. The bundled trio has T2 = 99.3%:
    -t2 100 relabels it, -t2 0 never does (documented in CLAUDE.md).
    """
    _skip_optimized_without_numpy(engine)

    def verdict(t2):
        prefix = os.path.join(work_dir, f'out_t2_{t2}')
        result = run_module([engine, test_vcf, '-o', prefix, '-j', '1', '-t2', t2,
                             '--no-pairwise', '--no-gender'])
        assert result.returncode == 0, result.stderr
        with open(f"{prefix}.trio") as f:
            return [l.split('\t')[-1].strip() for l in f.read().splitlines()[1:]]

    assert verdict('100') == ['Inverted/generational trio']
    assert verdict('0') == ['Regular trio']
