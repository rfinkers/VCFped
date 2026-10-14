"""
Score vcfped against the ground truth of the simulated breeding cohort
(vcfped/testfiles/breeding_sim/, see its README.md).

Marked ``integration`` (excluded from the default run, see pytest.ini): it runs the
real engine on 500 samples. Run with:

    pytest -m integration tests/test_breeding_sim_validation.py -v

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
