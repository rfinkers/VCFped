"""
Tests for utilities/purge_mz_twins.py.

Covers the pure functions (group_mz_twins transitive closure, parse_pair_file,
purge_vcf column removal) plus an end-to-end test: build a synthetic VCF with
a genuine MZ-twin duplicate (HG003 column copied verbatim as HG003_TWIN),
run vcfped's pairwise step to detect it, then purge it and verify the
duplicate is gone while the original trio survives.
"""
import os
import gzip
import pytest

from conftest import (import_module_from_path, run_module, run_script,
                      UTILITIES_DIR, TEST_VCF, TESTFILES_DIR)

pmz = import_module_from_path('purge_mz_twins',
                              os.path.join(UTILITIES_DIR, 'purge_mz_twins.py'))


# ---------------------------------------------------------------------------
# Unit tests: group_mz_twins (union-find transitive closure)
# ---------------------------------------------------------------------------

def test_group_mz_twins_simple_pair():
    groups = pmz.group_mz_twins([('A', 'B')])
    assert len(groups) == 1
    assert groups[0] == {'A', 'B'}


def test_group_mz_twins_transitive_chain():
    """A=B and B=C must merge into one group {A,B,C} (transitive closure)."""
    groups = pmz.group_mz_twins([('A', 'B'), ('B', 'C')])
    assert len(groups) == 1
    assert groups[0] == {'A', 'B', 'C'}


def test_group_mz_twins_disjoint_pairs():
    """Unrelated pairs must stay in separate groups."""
    groups = pmz.group_mz_twins([('A', 'B'), ('C', 'D')])
    assert len(groups) == 2
    assert {'A', 'B'} in groups
    assert {'C', 'D'} in groups


def test_group_mz_twins_empty():
    assert pmz.group_mz_twins([]) == []


# ---------------------------------------------------------------------------
# Unit tests: parse_pair_file
# ---------------------------------------------------------------------------

def test_parse_pair_file_extracts_mz_only(tmp_path):
    pair_file = tmp_path / "test.pair"
    pair_file.write_text(
        "Pair\tPerc\tAutos\tNeitherAA\tIBS2\tMZscore\tEitherBB\tIBS>0\tPOscore\tVerdict\n"
        "1,2\t10\t100\t50\t50\t100.0\t10\t10\t100.0\tMZ twins\n"
        "1,3\t10\t100\t50\t45\t90.0\t10\t9\t90.0\tParent-child\n"
    )
    vcf_samples = ['SampleA', 'SampleB', 'SampleC']
    pairs = pmz.parse_pair_file(str(pair_file), vcf_samples)
    assert pairs == [('SampleA', 'SampleB')]


def test_parse_pair_file_no_mz_twins(tmp_path):
    pair_file = tmp_path / "test.pair"
    pair_file.write_text(
        "Pair\tPerc\tAutos\tNeitherAA\tIBS2\tMZscore\tEitherBB\tIBS>0\tPOscore\tVerdict\n"
        "1,2\t10\t100\t50\t45\t90.0\t10\t9\t90.0\tParent-child\n"
    )
    pairs = pmz.parse_pair_file(str(pair_file), ['A', 'B'])
    assert pairs == []


# ---------------------------------------------------------------------------
# Unit tests: purge_vcf column removal
# ---------------------------------------------------------------------------

def test_purge_vcf_removes_specified_samples(tmp_path):
    src = tmp_path / "in.vcf"
    src.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
        "1\t100\t.\tA\tT\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"
    )
    dst = tmp_path / "out.vcf"
    pmz.purge_vcf(str(src), str(dst), {'B'}, format_col=8)

    out_lines = dst.read_text().splitlines()
    header = out_lines[1].split('\t')
    assert header == ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                      'INFO', 'FORMAT', 'A', 'C']
    data = out_lines[2].split('\t')
    assert data[-2:] == ['0/0', '1/1']  # B's 0/1 column removed


def test_purge_vcf_gzip_io(tmp_path):
    """purge_vcf must transparently handle .vcf.gz input and output."""
    src = tmp_path / "in.vcf.gz"
    with gzip.open(src, 'wt') as f:
        f.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
            "1\t100\t.\tA\tT\t.\t.\t.\tGT\t0/0\t0/1\n"
        )
    dst = tmp_path / "out.vcf.gz"
    pmz.purge_vcf(str(src), str(dst), {'B'}, format_col=8)

    with gzip.open(dst, 'rt') as f:
        content = f.read()
    assert 'A' in content
    assert '\tB\n' not in content.split('\n')[0] if content else True
    lines = content.splitlines()
    assert lines[1].split('\t')[-1] == 'A'


# ---------------------------------------------------------------------------
# End-to-end: real MZ-twin detection + purge on a synthetic duplicate
# ---------------------------------------------------------------------------

@pytest.fixture
def mz_twin_vcf(work_dir, test_vcf):
    """
    Synthetic VCF: bundled trio (HG002, HG003, HG004) plus an HG003_TWIN
    column that is a byte-identical copy of HG003's genotype data —
    a genuine MZ twin / technical replicate.
    """
    dst = os.path.join(work_dir, 'mz_twin.vcf')
    with open(test_vcf) as f, open(dst, 'w') as out:
        for line in f:
            if line.startswith('##'):
                out.write(line)
                continue
            fields = line.rstrip('\n').split('\t')
            if fields[0] == 'Chr':
                out.write('\t'.join(fields + ['HG003_TWIN']) + '\n')
            else:
                out.write(line.rstrip('\n') + '\t' + fields[-2] + '\n')  # dup HG003 col
    return dst


def test_end_to_end_mz_twin_detection_and_purge(work_dir, mz_twin_vcf):
    # Step 1: pairwise-only run should detect HG003 == HG003_TWIN as MZ twins
    step1_prefix = os.path.join(work_dir, 'step1')
    result = run_module(['vcfped.archive.vcfped_original', mz_twin_vcf,
                         '--no-trio', '--no-gender', '-o', step1_prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr
    pair_file = f"{step1_prefix}.pair"
    assert os.path.exists(pair_file)
    with open(pair_file) as f:
        pair_content = f.read()
    assert 'MZ twins' in pair_content

    # Step 2: purge — should remove the duplicate, leaving 3 of the 4 samples
    purged_vcf = os.path.join(work_dir, 'purged.vcf')
    purge_script = os.path.join(UTILITIES_DIR, 'purge_mz_twins.py')
    result = run_script(purge_script,
                        [mz_twin_vcf, pair_file, '-o', purged_vcf])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(purged_vcf)

    with open(purged_vcf) as f:
        for line in f:
            if not line.startswith('##'):
                header = line.rstrip('\n').split('\t')
                break
    samples = header[-4:] if len(header) >= 4 else header
    # Exactly one of HG003 / HG003_TWIN should remain (not both, not neither)
    n_hg003_variants = sum(1 for s in samples if s in ('HG003', 'HG003_TWIN'))
    assert n_hg003_variants == 1, f"Expected exactly 1 of HG003/HG003_TWIN, got: {samples}"
    assert 'HG002' in samples
    assert 'HG004' in samples


def test_dry_run_does_not_write_output(work_dir, mz_twin_vcf):
    step1_prefix = os.path.join(work_dir, 'step1')
    result = run_module(['vcfped.archive.vcfped_original', mz_twin_vcf,
                         '--no-trio', '--no-gender', '-o', step1_prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr

    purge_script = os.path.join(UTILITIES_DIR, 'purge_mz_twins.py')
    fake_output = os.path.join(work_dir, 'should_not_exist.vcf')
    result = run_script(purge_script,
                        [mz_twin_vcf, f"{step1_prefix}.pair",
                         '-o', fake_output, '--dry-run'])
    assert result.returncode == 0, result.stderr
    assert not os.path.exists(fake_output)


def test_keep_first_vs_last(work_dir, mz_twin_vcf):
    """--keep first vs --keep last should retain different members of the MZ group."""
    step1_prefix = os.path.join(work_dir, 'step1')
    result = run_module(['vcfped.archive.vcfped_original', mz_twin_vcf,
                         '--no-trio', '--no-gender', '-o', step1_prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr
    pair_file = f"{step1_prefix}.pair"
    purge_script = os.path.join(UTILITIES_DIR, 'purge_mz_twins.py')

    def get_remaining_hg003_sample(keep_mode):
        out_vcf = os.path.join(work_dir, f'purged_{keep_mode}.vcf')
        result = run_script(purge_script,
                            [mz_twin_vcf, pair_file, '-o', out_vcf, '--keep', keep_mode])
        assert result.returncode == 0, result.stderr
        with open(out_vcf) as f:
            for line in f:
                if not line.startswith('##'):
                    header = line.rstrip('\n').split('\t')
                    break
        return next(s for s in header if s in ('HG003', 'HG003_TWIN'))

    kept_first = get_remaining_hg003_sample('first')
    kept_last = get_remaining_hg003_sample('last')
    # HG003 appears before HG003_TWIN in the VCF column order
    assert kept_first == 'HG003'
    assert kept_last == 'HG003_TWIN'


# ---------------------------------------------------------------------------
# Heterozygosity guard: a selfed child / inbred line is not a duplicate
# ---------------------------------------------------------------------------

def _het_vcf(path, n_sites=40):
    """parent: heterozygous at every other site; clone: identical to parent;
    selfed: shares all alleles but is homozygous where the parent is het."""
    rows = []
    for i in range(n_sites):
        parent = '0/1' if i % 2 == 0 else ('0/0' if i % 4 == 1 else '1/1')
        selfed = ('0/0' if i % 8 < 4 else '1/1') if parent == '0/1' else parent
        rows.append(f"1\t{100 + i}\t.\tA\tG\t.\t.\t.\tGT\t{parent}\t{parent}\t{selfed}\n")
    path.write_text("##fileformat=VCFv4.2\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tparent\tclone\tselfed\n"
                    + "".join(rows))
    return str(path)


def test_sample_heterozygosity(tmp_path):
    vcf = _het_vcf(tmp_path / "h.vcf")
    het = pmz.sample_heterozygosity(vcf, 8, ['parent', 'clone', 'selfed'], {'parent', 'selfed'})
    assert set(het) == {'parent', 'selfed'}          # only the requested samples
    assert het['parent'] == pytest.approx(0.5)
    assert het['selfed'] == 0.0


def test_sample_heterozygosity_ignores_missing_and_phased(tmp_path):
    vcf = tmp_path / "m.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n"
                   "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\n"
                   "1\t1\t.\tA\tG\t.\t.\t.\tGT:DP\t0|1:9\n"
                   "1\t2\t.\tA\tG\t.\t.\t.\tGT\t./.\n"
                   "1\t3\t.\tA\tG\t.\t.\t.\tGT\t1/1\n")
    assert pmz.sample_heterozygosity(str(vcf), 8, ['A'], {'A'}) == {'A': 0.5}


def test_filter_pairs_by_heterozygosity():
    het = {'a': 0.20, 'b': 0.19, 'c': 0.10, 'z': 0.0}
    kept, rejected = pmz.filter_pairs_by_heterozygosity(
        [('a', 'b'), ('a', 'c'), ('z', 'z')], het, 0.9)
    assert kept == [('a', 'b'), ('z', 'z')]          # both-zero het counts as equal
    assert [(x, y) for x, y, _ in rejected] == [('a', 'c')]
    assert rejected[0][2] == pytest.approx(0.5)
    # exactly at the threshold is kept; 0 disables the check
    assert pmz.filter_pairs_by_heterozygosity([('a', 'c')], het, 0.5)[1] == []
    assert pmz.filter_pairs_by_heterozygosity([('a', 'c')], het, 0)[0] == [('a', 'c')]


def test_purge_keeps_selfed_child_but_removes_clone(tmp_path):
    vcf = _het_vcf(tmp_path / "h.vcf")
    pair = tmp_path / "x.pair"
    pair.write_text("Pair\tPerc\tAutos\tNeitherAA\tIBS2\tMZscore\tEitherBB\tIBS>0\tPOscore\tVerdict\n"
                    "1,2\t10\t40\t1\t1\t99.0\t1\t1\t99.0\tMZ twins\n"
                    "1,3\t10\t40\t1\t1\t95.0\t1\t1\t99.0\tMZ twins\n")
    script = os.path.join(UTILITIES_DIR, 'purge_mz_twins.py')

    out = tmp_path / "out.vcf"
    r = run_script(script, [vcf, str(pair), '-o', str(out)])
    assert r.returncode == 0, r.stderr
    assert "ratio=0.00" in r.stdout
    header = next(l for l in out.read_text().splitlines() if l.startswith('#CHROM'))
    assert header.split('\t')[9:] == ['parent', 'selfed']      # clone gone, selfed child kept

    out2 = tmp_path / "out2.vcf"
    r = run_script(script, [vcf, str(pair), '-o', str(out2), '--min-het-ratio', '0'])
    assert r.returncode == 0, r.stderr
    header = next(l for l in out2.read_text().splitlines() if l.startswith('#CHROM'))
    assert header.split('\t')[9:] == ['parent']                 # old behaviour: both purged
