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


# ---------------------------------------------------------------------------
# --trio-method gemm (matrix products) must reproduce the Numba path exactly
# ---------------------------------------------------------------------------

def _pedigree_vcf(path, n_sites=1400, seed=7):
    """Small cohort with real trios, an unrelated sample, a sample with random
    genotyping errors and one with many missing calls, so that Regular,
    Inverted, Not trio and na verdicts all occur."""
    import random
    rnd = random.Random(seed)
    names = ['F1', 'F2', 'F3', 'F4', 'K1', 'K2', 'K3', 'U1', 'U2', 'ERR', 'MISS']
    rows = []
    for i in range(n_sites):
        g = {}
        for f in ('F1', 'F2', 'F3', 'F4', 'U1', 'U2'):
            g[f] = (rnd.random() < 0.5, rnd.random() < 0.5)       # two alleles (True = ALT)
        def child(a, b):
            return (g[a][rnd.randrange(2)], g[b][rnd.randrange(2)])
        g['K1'] = child('F1', 'F2')
        g['K2'] = child('F3', 'F4')
        g['K3'] = child('K1', 'F3')
        g['ERR'] = child('F1', 'F4') if rnd.random() > 0.2 else (rnd.random() < 0.5, rnd.random() < 0.5)
        g['MISS'] = child('F2', 'F3')
        cols = []
        for n in names:
            a, b = g[n]
            gt = '%d/%d' % tuple(sorted((a, b)))     # vcfped reads unphased 0/1 only, not 1/0
            if n == 'MISS' and rnd.random() < 0.4:
                gt = './.'
            cols.append(f"{gt}:30:50")
        rows.append(f"1\t{100 + i}\t.\tA\tG\t50\tPASS\t.\tGT:DP:GQ\t" + "\t".join(cols) + "\n")
    with open(path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(names) + "\n" + "".join(rows))
    return path


@pytest.mark.parametrize('extra', [[], ['--all'], ['-p', '0', '30', '60', '-t1', '75', '-t2', '40']])
def test_gemm_trio_method_matches_numba(work_dir, extra):
    _skip_optimized_without_numpy('vcfped.vcfped')
    vcf = _pedigree_vcf(os.path.join(work_dir, 'ped.vcf'))
    out = {}
    for method in ('numba', 'gemm'):
        prefix = os.path.join(work_dir, f'out_{method}')
        result = run_module(['vcfped.vcfped', vcf, '-o', prefix, '-j', '2', '--quiet',
                             '--no-gender', '--no-pairwise', '--trio-method', method] + extra)
        assert result.returncode == 0, result.stderr
        with open(f"{prefix}.trio", 'rb') as f:
            out[method] = f.read()
    assert out['gemm'] == out['numba']
    text = out['gemm'].decode()
    assert 'Regular trio' in text or 'Inverted/generational trio' in text   # the test is not vacuous
    if '--all' in extra:
        assert 'Not trio' in text and '\tna' in text


def test_gemm_trio_method_matches_numba_on_bundled_trio(work_dir, test_vcf):
    _skip_optimized_without_numpy('vcfped.vcfped')
    out = {}
    for method in ('numba', 'gemm'):
        prefix = os.path.join(work_dir, f'hg_{method}')
        result = run_module(['vcfped.vcfped', test_vcf, '-o', prefix, '-j', '2', '--quiet',
                             '--trio-method', method])
        assert result.returncode == 0, result.stderr
        with open(f"{prefix}.trio", 'rb') as f:
            out[method] = f.read()
    assert out['gemm'] == out['numba']
    assert b'1,2,3\t1\t10\t1906\t136\t136\t100.0\t269\t267\t99.3\tRegular trio' in out['gemm']


def test_full_log_is_opt_in(work_dir, test_vcf):
    _skip_optimized_without_numpy('vcfped.vcfped')
    logs = {}
    for flag, tag in (([], 'default'), (['--full-log'], 'full')):
        prefix = os.path.join(work_dir, f'log_{tag}')
        result = run_module(['vcfped.vcfped', test_vcf, '-o', prefix, '-j', '1', '--quiet'] + flag)
        assert result.returncode == 0, result.stderr
        with open(f"{prefix}.log") as f:
            logs[tag] = f.read()
    for section in ('====PAIRWISE RELATIONS====', '====TRIO RELATIONS===='):
        assert section not in logs['default']
        assert section in logs['full']
    # the pointer to the result files is kept either way
    assert 'Most confident trio calls written to' in logs['default']
    assert 'Most confident pairwise calls written to' in logs['default']


@pytest.mark.parametrize('engine', ENGINES)
def test_unphased_1_0_is_heterozygous(work_dir, engine):
    """
    A heterozygous call written 1/0 must give the same result as 0/1. Currently
    was silently treated as missing, so every pair/trio count that involves it
    shrank (the VCF spec allows either allele order).
    """
    _skip_optimized_without_numpy(engine)
    header = ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n")
    # s1 and s2 share every genotype (so MZ-like); s3 differs. Hets in s1/s2 are the point.
    pattern = [('0/1', '0/1', '0/0'), ('1/1', '1/1', '0/1'), ('0/1', '0/1', '1/1'),
               ('0/0', '0/0', '0/1'), ('0/1', '0/1', '0/1'), ('1/1', '1/1', '0/0')] * 4

    def write(path, het):
        rows = "".join(
            "1\t%d\t.\tA\tG\t50\tPASS\t.\tGT:DP:GQ\t%s\n" % (
                100 + i, "\t".join(f"{g.replace('0/1', het)}:30:50" for g in gts))
            for i, gts in enumerate(pattern))
        with open(path, 'w') as f:
            f.write(header + rows)

    result_text = {}
    for het in ('0/1', '1/0'):
        vcf = os.path.join(work_dir, f"het_{het.replace('/', '')}.vcf")
        write(vcf, het)
        prefix = os.path.join(work_dir, f"out_{het.replace('/', '')}")
        result = run_module([engine, vcf, '-o', prefix, '-j', '1', '--quiet', '--no-gender',
                             '--no-trio', '--all'])
        assert result.returncode == 0, result.stderr
        with open(f"{prefix}.pair") as f:
            result_text[het] = f.read()
    assert result_text['1/0'] == result_text['0/1']
