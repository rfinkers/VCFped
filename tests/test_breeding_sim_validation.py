"""
Score vcfped against the ground truth of the simulated breeding cohort
(vcfped/testfiles/breeding_sim/, see its README.md).

Marked ``integration`` (excluded from the default run, see pytest.ini): it runs the
real engine on 500 samples. Run with:

    pytest -m integration tests/test_breeding_sim_validation.py -v

The full 500-sample trio stage takes too long without numba, so trios are scored on a
~100-sample subset of whole families.

The assertions are deliberately limited to what the engine does reliably on this
cohort. Known weaknesses on it -- raw "MZ twins" over-calls between deeply selfed
lines (the inbreeding-induced inflation described in CLAUDE.md) -- are only checked
to stay confined to inbred samples, so the tests do not lock in the over-calling itself.
"""
import collections
import csv
import gzip
import os

import pytest

from conftest import run_module, requires_numpy, TESTFILES_DIR

pytestmark = [pytest.mark.integration, requires_numpy]

SIM = os.path.join(TESTFILES_DIR, 'breeding_sim')
VCF = os.path.join(SIM, 'breeding_sim.vcf.gz')


def _tsv(suffix):
    with open(os.path.join(SIM, f'breeding_sim_truth_{suffix}'), newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


@pytest.fixture(scope='module')
def truth_relationships():
    return {tuple(sorted((r['ID1'], r['ID2']))): r for r in _tsv('relationships.tsv')}


@pytest.fixture(scope='module')
def sample_names():
    """vcfped numbers samples 1..N in VCF column order."""
    with gzip.open(VCF, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                return line.rstrip('\n').split('\t')[9:]


def _run_vcfped(tmp_path_factory, label, extra):
    pytest.importorskip('tabulate')
    prefix = str(tmp_path_factory.mktemp(label) / 'out')
    result = run_module(['vcfped.vcfped_optimized', VCF, '--no-gender', '--quiet',
                         '-o', prefix] + extra, timeout=1800)
    assert result.returncode == 0, result.stderr
    return prefix


@pytest.fixture(scope='module')
def pair_calls(tmp_path_factory, sample_names):
    """{sorted (name1, name2): verdict} from vcfped's pairwise stage."""
    prefix = _run_vcfped(tmp_path_factory, 'sim_pair', ['--no-trio', '-j', '4'])
    calls = {}
    with open(prefix + '.pair', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            i, j = (int(x) for x in row['Pair'].split(','))
            calls[tuple(sorted((sample_names[i - 1], sample_names[j - 1])))] = row['Verdict']
    return calls


def _by_class(truth_relationships, calls, rel, complex_flag='N'):
    return collections.Counter(
        calls.get(k, '(not reported)') for k, r in truth_relationships.items()
        if r['REL'] == rel and r['COMPLEX'] == complex_flag)


def test_every_true_duplicate_is_called_mz_twins(pair_calls, truth_relationships):
    dups = [k for k, r in truth_relationships.items() if r['REL'] == 'duplicate']
    assert len(dups) >= 30
    missed = [k for k in dups if pair_calls.get(k) != 'MZ twins']
    assert not missed, f"duplicates not called MZ twins: {missed}"


def test_simple_parent_offspring_pairs_are_found(pair_calls, truth_relationships):
    outcome = _by_class(truth_relationships, pair_calls, 'parent_offspring')
    assert outcome['Parent-child'] / sum(outcome.values()) >= 0.95, outcome


def test_simple_full_sibs_are_not_mistaken_for_parent_offspring(pair_calls, truth_relationships):
    outcome = _by_class(truth_relationships, pair_calls, 'full_sib')
    assert outcome['Parent-child'] / sum(outcome.values()) <= 0.03, outcome


def test_false_mz_calls_are_explained_by_inbreeding(pair_calls, truth_relationships, capsys):
    """
    Non-duplicate pairs called 'MZ twins' must be pairs where deep selfing makes two
    different samples near-identical (the mechanism described in CLAUDE.md), not
    noise elsewhere.  The count is printed so a change in over-calling is visible.
    """
    inbreeding = {r['ID']: float(r['F_PED']) for r in _tsv('samples.tsv') if r['GENOTYPED'] == 'Y'}
    false_mz = [k for k, v in pair_calls.items()
                if v == 'MZ twins' and truth_relationships.get(k, {}).get('REL') != 'duplicate']
    explained = [k for k in false_mz if any(inbreeding[s] >= 0.75 for s in k)]
    with capsys.disabled():
        print(f"\n[breeding_sim] {len(false_mz)} non-duplicate pairs called 'MZ twins'; "
              f"{len(explained)} involve a sample with pedigree F >= 0.75")
    assert false_mz and len(explained) / len(false_mz) >= 0.95


# ---------------------------------------------------------------------------
# Trio stage (on a subset of whole families)
# ---------------------------------------------------------------------------

SUBSET_TAGS = ('trio', 'cross_pop_trio', 'fullsib', 'halfsib', 'multigen', 'large_sibship',
               'one_parent_genotyped', 'fullsib_noparents')


@pytest.fixture(scope='module')
def trio_subset(tmp_path_factory):
    """Run the trio stage on whole families (clones included); return (samples, regular calls)."""
    truth = [r for r in _tsv('samples.tsv') if r['GENOTYPED'] == 'Y']
    by_family = collections.defaultdict(list)
    for r in truth:
        by_family[r['FID']].append(r)
    chosen, total = [], 0
    for tag in SUBSET_TAGS:
        for fid, members in by_family.items():
            if fid not in chosen and any(m['SCENARIO'] == tag for m in members) and total + len(members) <= 100:
                chosen.append(fid)
                total += len(members)
                break
    keep = {m['ID'] for fid in chosen for m in by_family[fid]}

    work = tmp_path_factory.mktemp('sim_trio')
    subset_vcf = str(work / 'subset.vcf')
    with open(os.path.join(SIM, 'breeding_sim.vcf')) as src, open(subset_vcf, 'w') as dst:
        cols = None
        for line in src:
            if line.startswith('##'):
                dst.write(line)
                continue
            parts = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                cols = [i for i, name in enumerate(parts) if i < 9 or name in keep]
                names = [parts[i] for i in cols[9:]]
            dst.write('\t'.join(parts[i] for i in cols) + '\n')

    pytest.importorskip('tabulate')
    prefix = str(work / 'out')
    result = run_module(['vcfped.vcfped_optimized', subset_vcf, '--no-pairwise', '--no-gender',
                         '--quiet', '-j', '4', '-o', prefix], timeout=1800)
    assert result.returncode == 0, result.stderr
    regular = collections.defaultdict(set)         # child -> {frozenset of parent names}
    with open(prefix + '.trio', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['Verdict'] != 'Regular trio':
                continue
            members = [names[int(i) - 1] for i in row['Triple'].split(',')]
            child = names[int(row['Pivot']) - 1]
            regular[child].add(frozenset(m for m in members if m != child))
    return set(names), regular


def test_simple_true_trios_are_found(trio_subset, capsys):
    """Outbred trios with both parents genotyped; a parent's clone is an equally correct parent."""
    names, regular = trio_subset
    truth = {r['ID']: r for r in _tsv('samples.tsv') if r['GENOTYPED'] == 'Y'}
    source = {s: (r['DUPLICATE_OF'] or s) for s, r in truth.items()}      # clone -> original
    fam = {}
    with open(os.path.join(SIM, 'breeding_sim_truth.fam')) as f:
        for line in f:
            _, iid, pat, mat, *_ = line.split()
            fam[iid] = (pat, mat)
    expected = {}
    for child in names:
        pat, mat = fam[child]
        flags = truth[child]['FLAGS']
        if pat in ('0', mat) or mat == '0' or 'inbred' in flags or 'parents_are_relatives' in flags:
            continue
        expected[child] = frozenset([source[pat], source[mat]])
    assert len(expected) >= 20
    found = [c for c, parents in expected.items()
             if any(frozenset(source[p] for p in called) == parents for called in regular.get(c, ()))]
    with capsys.disabled():
        print(f"\n[breeding_sim] {len(found)}/{len(expected)} simple true trios called 'Regular trio'")
    assert len(found) / len(expected) >= 0.90, f"{len(found)}/{len(expected)} simple trios found"
