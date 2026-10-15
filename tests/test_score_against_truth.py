"""
Tests for utilities/score_against_truth.py and utilities/run_validation_pipeline.sh.

Most tests use a tiny hand-built truth (14 duplicate classes, 15 samples) in which every
expected number can be worked out by hand; a few use the shipped breeding_sim truth with
inputs derived from the truth itself (a perfect reconstruction must score perfectly).

Mini truth (parents in brackets):
  founders S01 S02 S05 S06 S12      S03 (S01,S02)  S04 = clone of S03     S07 (S05,S06)  S14 (S05,S06)
  S08 (S07,S05) backcross           S09 (S06,S06) selfed, F=0.96          S10 (S01,-) one parent
  S11 (S02,S09) inbred child        S13 (S07,S14) parents are sibs        S15 (S03,S12) simple
"""
import csv
import gzip
import json
import os
import subprocess
import sys

import pytest

from conftest import import_module_from_path, run_script, TESTFILES_DIR, UTILITIES_DIR

SCORER = os.path.join(UTILITIES_DIR, 'score_against_truth.py')
RUNNER = os.path.join(UTILITIES_DIR, 'run_validation_pipeline.sh')
SHIPPED = os.path.join(TESTFILES_DIR, 'breeding_sim', 'breeding_sim')


@pytest.fixture(scope='module')
def sat():
    return import_module_from_path('score_against_truth', SCORER)


# ---------------------------------------------------------------------------
# Mini truth
# ---------------------------------------------------------------------------

# ID, PAT, MAT, DUPLICATE_OF, F_PED, FLAGS
MINI = [
    ('S01', '0', '0', '', 0.0, ''),
    ('S02', '0', '0', '', 0.0, ''),
    ('S03', 'S01', 'S02', '', 0.0, 'has_duplicate'),
    ('S04', 'S01', 'S02', 'S03', 0.0, 'is_duplicate;has_duplicate'),
    ('S05', '0', '0', '', 0.0, 'contaminated'),
    ('S06', '0', '0', '', 0.0, 'high_error'),
    ('S07', 'S05', 'S06', '', 0.0, ''),
    ('S08', 'S07', 'S05', '', 0.0, 'parent_x_own_offspring;BC1'),
    ('S09', 'S06', 'S06', '', 0.96, 'selfed;inbred'),
    ('S10', 'S01', '0', '', 0.0, 'parent_ungenotyped'),
    ('S11', 'S02', 'S09', '', 0.96, 'inbred'),
    ('S12', '0', '0', '', 0.0, 'low_callrate'),
    ('S13', 'S07', 'S14', '', 0.25, 'parents_are_relatives;inbred'),
    ('S14', 'S05', 'S06', '', 0.0, ''),
    ('S15', 'S03', 'S12', '', 0.0, ''),
]
# ID1, ID2, REL, COMPLEX
MINI_REL = [
    ('S01', 'S03', 'parent_offspring', 'N'), ('S01', 'S04', 'parent_offspring', 'N'),
    ('S02', 'S03', 'parent_offspring', 'N'), ('S02', 'S04', 'parent_offspring', 'N'),
    ('S03', 'S04', 'duplicate', 'N'),
    ('S05', 'S07', 'parent_offspring', 'N'), ('S06', 'S07', 'parent_offspring', 'N'),
    ('S05', 'S14', 'parent_offspring', 'N'), ('S06', 'S14', 'parent_offspring', 'N'),
    ('S07', 'S14', 'full_sib', 'N'), ('S07', 'S08', 'parent_offspring', 'N'),
    ('S05', 'S08', 'parent_offspring', 'Y'),
]
HEADER = ['ID', 'FID', 'PAT', 'MAT', 'GENERATION', 'SCENARIO', 'GENOTYPED', 'DUPLICATE_OF',
          'F_PED', 'N_MISSING', 'FLAGS']


def _write_tsv(path, header, rows):
    with open(path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')


def _write_gz(path, lines):
    with gzip.open(path, 'wt') as f:
        f.write(''.join(l + '\n' for l in lines))


@pytest.fixture
def mini(tmp_path):
    p = str(tmp_path / 'mini')
    _write_tsv(p + '_truth_samples.tsv', HEADER,
               [[i, 'F1', pat, mat, 0, 'mini', 'Y', dup, f, 0, fl] for i, pat, mat, dup, f, fl in MINI])
    with open(p + '_truth.fam', 'w') as f:
        for i, pat, mat, *_ in MINI:
            f.write(f"F1 {i} {pat} {mat} 0 -9\n")
    _write_tsv(p + '_truth_relationships.tsv',
               ['ID1', 'ID2', 'REL', 'PHI_PED', 'PHI_REALISED', 'NOMINAL_PHI', 'COMPLEX'],
               [[a, b, rel, 0.25, 0.25, 0.25, cx] for a, b, rel, cx in MINI_REL])
    _write_tsv(p + '_truth_duplicates.tsv', ['ID1', 'ID2'], [['S03', 'S04']])
    _write_tsv(p + '_truth_snps.tsv', ['CHROM', 'POS', 'CM', 'BAD_MARKER', 'CONSERVED_BLOCK'],
               [['ch01', 5000, 1.0, 'Y', 'N'], ['ch02', 100, 2.0, 'N', 'N']])
    _write_tsv(p + '_truth_conserved_regions.tsv', ['CHROM', 'START', 'END', 'START_CM', 'END_CM', 'NOTE'],
               [['ch04', 1000, 2000, 1, 2, 'x']])
    _write_gz(p + '_truth_ibd.ibd.gz', [
        'S01\t1\tS03\t1\tch01\t100\t1100\t5.0', 'S01\t1\tS03\t2\tch02\t200\t700\t4.0',
        'S02\t1\tS03\t1\tch01\t1\t2001\t9.0'])
    _write_gz(p + '_truth_hbd.hbd.gz', [
        'S09\t1\tS09\t2\tch01\t100\t1100\t10.0', 'S11\t1\tS11\t2\tch02\t5\t55\t3.0'])
    return p


def run_cli(args, cwd):
    """Run the scorer as a subprocess; return (CompletedProcess, summary dict or None)."""
    out = os.path.join(str(cwd), 'out')
    r = run_script(SCORER, list(args) + ['-o', out])
    summary = None
    if os.path.exists(out + '_score_summary.json'):
        with open(out + '_score_summary.json') as f:
            summary = json.load(f)
    return r, summary, out


def trio_file(path, rows):
    with open(path, 'w') as f:
        f.write('Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n')
        for triple, pivot, verdict in rows:
            f.write(f"{triple}\t{pivot}\t10\t0\t0\t0\t0\t0\t0\t0\t{verdict}\n")


# ---------------------------------------------------------------------------
# Truth classification
# ---------------------------------------------------------------------------

def test_child_kinds_follow_flags_and_parent_structure(sat, mini):
    items = sat.Truth(mini).child_items()
    kinds = {c: it['kind'] for c, it in items.items()}
    assert kinds['S03'] == kinds['S07'] == kinds['S14'] == kinds['S15'] == 'simple'
    assert kinds['S08'] == 'backcross'
    assert kinds['S13'] == 'related_parents'      # 'parents_are_relatives' wins over 'inbred'
    assert kinds['S11'] == 'inbred_child'
    assert kinds['S09'] == 'selfed' and kinds['S10'] == 'one_parent'
    assert {c for c, k in kinds.items() if k == 'no_parent'} == {'S01', 'S02', 'S05', 'S06', 'S12'}
    assert 'S04' not in items                      # a clone is not its own class
    assert items['S03']['parents'] == frozenset({'S01', 'S02'})


# ---------------------------------------------------------------------------
# Final pedigree (.fam)
# ---------------------------------------------------------------------------

def test_fam_outcomes_clone_equivalence_and_precision(mini, tmp_path):
    fam = {  # S03 partial, S07 wrong, S14 unassigned, S15 right via the clone S04
        'S03': ('S01', 'S12'), 'S07': ('S12', 'S02'), 'S15': ('S04', 'S12'),
        'S08': ('S07', 'S05'), 'S09': ('S06', 'S06'), 'S11': ('S02', 'S09'),
        'S13': ('S07', 'S14'), 'S12': ('S01', 'S02')}                    # S12: parents invented
    path = tmp_path / 'x.fam'
    path.write_text(''.join(f"F1 {i[0]} {fam.get(i[0], ('0', '0'))[0]} {fam.get(i[0], ('0', '0'))[1]} 0 -9\n"
                            for i in MINI))
    r, s, out = run_cli(['--truth', mini, '--fam', str(path)], tmp_path)
    assert r.returncode == 0, r.stderr
    f = s['fam']
    assert (f['simple']['correct'], f['simple']['partial'], f['simple']['wrong'],
            f['simple']['unassigned']) == (1, 1, 1, 1)
    for kind in ('backcross', 'inbred_child', 'related_parents'):
        assert f[kind]['correct'] == 1
    assert f['assigned'] == 8 and f['assigned_correct'] == 5      # 3 simple + 3 + selfed, + 1 spurious
    assert f['wrongly_assigned_parent_is'] == {'unrelated': 5}
    rows = {r['CLASS']: r for r in csv.DictReader(open(out + '_score_children.tsv'), delimiter='\t')}
    assert rows['S03']['FAM_OUTCOME'] == 'partial' and rows['S15']['FAM_OUTCOME'] == 'correct'
    assert rows['S10']['FAM_OUTCOME'] == 'not_representable'
    assert rows['S12']['FAM_OUTCOME'] == 'spurious' and rows['S01']['FAM_OUTCOME'] == 'correct_empty'


def test_fam_clone_may_carry_the_assignment(mini, tmp_path):
    path = tmp_path / 'x.fam'
    path.write_text(''.join(f"F1 {i[0]} {'S01 S02' if i[0] == 'S04' else '0 0'} 0 -9\n" for i in MINI))
    r, s, _ = run_cli(['--truth', mini, '--fam', str(path)], tmp_path)
    assert s['fam']['simple']['correct'] == 1       # S03's class is correct although only S04 was assigned


# ---------------------------------------------------------------------------
# Trio stage
# ---------------------------------------------------------------------------

def test_trio_outcomes(mini, tmp_path):
    R, I = 'Regular trio', 'Inverted/generational trio'
    path = tmp_path / 'x.trio'
    trio_file(path, [
        ('S01,S02,S03', 'S03', R), ('S01,S02,S04', 'S04', R),                  # clone pivot: same class pair
        ('S05,S06,S07', 'S07', R), ('S05,S07,S12', 'S07', R),                  # true pair + another -> ambiguous
        ('S05,S12,S14', 'S14', R),                                             # only a wrong pair
        ('S03,S12,S15', 'S15', I),                                             # true pair, but inverted
        ('S02,S09,S11', 'S11', R),                                             # inbred child found
        ('S01,S02,S12', 'S12', R)])                                            # a founder given parents
    r, s, out = run_cli(['--truth', mini, '--trio', str(path)], tmp_path)
    assert r.returncode == 0, r.stderr
    t = s['trios']
    assert (t['simple']['n'], t['simple']['found'], t['simple']['unique']) == (4, 2, 1)
    assert (t['simple']['wrong_regular'], t['simple']['inverted_only']) == (1, 1)
    assert t['backcross']['missed'] == 1 and t['related_parents']['missed'] == 1
    assert t['inbred_child']['found'] == 1
    assert t['no_pair_to_find']['no_parent']['spurious_regular'] == 1
    assert t['no_pair_to_find']['one_parent']['correctly_empty'] == 1
    assert (t['regular_calls'], t['regular_calls_correct']) == (6, 3)


def test_purge_losses_are_counted_as_misses(mini, tmp_path):
    log = tmp_path / 'purge.log'
    log.write_text("Found 2 MZ-twin pair(s)\n"
                   "  Cluster (2 samples): keep=S03, remove=S04\n"          # a real duplicate: fine
                   "  Cluster (2 samples): keep=S12, remove=S07\n")         # distinct individuals merged
    trio = tmp_path / 'x.trio'
    trio_file(trio, [('S01,S02,S03', 'S03', 'Regular trio')])
    r, s, _ = run_cli(['--truth', mini, '--purge-log', str(log), '--trio', str(trio)], tmp_path)
    assert r.returncode == 0, r.stderr
    p = s['purge']
    assert (p['clusters'], p['removed'], p['true_removals'], p['wrong_removals'], p['individuals_lost']) == (2, 2, 1, 1, 1)
    t = s['trios']
    # S07 is lost; so are its child S08 (backcross) and S13 (related parents)
    assert t['simple']['lost_by_purge'] == 1 and t['backcross']['lost_by_purge'] == 1
    assert t['related_parents']['lost_by_purge'] == 1
    assert t['simple']['unique'] == 1


# ---------------------------------------------------------------------------
# Pairwise verdicts
# ---------------------------------------------------------------------------

def test_pair_verdicts(mini, tmp_path):
    path = tmp_path / 'x.pair'
    rows = [('S03,S04', 'MZ twins'), ('S09,S11', 'MZ twins'), ('S05,S06', 'MZ twins'),
            ('S01,S03', 'Parent-child'), ('S05,S08', 'Parent-child'), ('S07,S14', 'Parent-child')]
    path.write_text('Pair\tPerc\tAutos\tNeitherAA\tIBS2\tMZscore\tEitherBB\tIBS>0\tPOscore\tVerdict\n'
                    + ''.join(f"{p}\t10\t0\t0\t0\t0\t0\t0\t0\t{v}\n" for p, v in rows))
    r, s, _ = run_cli(['--truth', mini, '--pair', str(path)], tmp_path)
    assert r.returncode == 0, r.stderr
    pr = s['pairs']
    assert pr['mz'] == dict(true_pairs=1, found=1, false_calls=2, false_both_inbred=1, false_one_inbred=0)
    assert pr['po_simple'] == dict(true_pairs=9, found=1)
    assert pr['po_inbred'] == dict(true_pairs=1, found=1)
    assert pr['po_false_calls'] == 1              # S07-S14 are full sibs


def test_numeric_pair_file_needs_and_uses_the_log(mini, tmp_path):
    pair = tmp_path / 'x.pair'
    pair.write_text('Pair\tVerdict\n3,4\tMZ twins\n')
    r, _, _ = run_cli(['--truth', mini, '--pair', str(pair)], tmp_path)
    assert r.returncode != 0 and '--log' in r.stderr
    log = tmp_path / 'x.log'
    log.write_text(''.join(f"  {n}: {m[0]}\n" for n, m in enumerate(MINI, 1)))
    r, s, _ = run_cli(['--truth', mini, '--pair', str(pair), '--log', str(log)], tmp_path)
    assert r.returncode == 0 and s['pairs']['mz']['found'] == 1


# ---------------------------------------------------------------------------
# KING
# ---------------------------------------------------------------------------

def test_king_bands_and_false_relatives(mini, tmp_path):
    path = tmp_path / 'x.kin0'
    rows = [('S01', 'S03', 0.25, 0.001), ('S02', 'S03', 0.30, 0.0), ('S05', 'S07', 0.40, 0.0),
            ('S05', 'S14', 0.19, 0.0),                       # 1st-degree band starts at 0.177
            ('S07', 'S14', 0.12, 0.02), ('S03', 'S04', 0.49, 0.0),
            ('S01', 'S12', 0.06, 0.05), ('S02', 'S12', 0.20, 0.01)]
    path.write_text('#FID1\tIID1\tFID2\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n'
                    + ''.join(f"F\t{a}\tF\t{b}\t900\t0.2\t{i}\t{k}\n" for a, b, k, i in rows))
    r, s, _ = run_cli(['--truth', mini, '--king', str(path)], tmp_path)
    assert r.returncode == 0, r.stderr
    k = s['king']
    assert k['parent_offspring_simple'] == dict(n=9, listed=4, exact_band=3)   # 0.40 is the dup band
    assert k['full_sib_simple'] == dict(n=1, listed=1, exact_band=0)           # 0.12 is the 2nd-degree band
    assert k['duplicate_simple'] == dict(n=1, listed=1, exact_band=1)
    assert k['parent_offspring_inbred']['listed'] == 0
    assert k['false_relatives'] == {'3rd': 1, '1st': 1}


@pytest.mark.parametrize('kinship, band', [
    (0.5, 'dup'), (0.354, 'dup'), (0.3539, '1st'), (0.177, '1st'), (0.1769, '2nd'),
    (0.0884, '2nd'), (0.0883, '3rd'), (0.0442, '3rd'), (0.0441, '<3rd'), (-0.02, '<3rd')])
def test_king_band_boundaries(sat, kinship, band):
    assert sat.kin_band(kinship) == band


# ---------------------------------------------------------------------------
# Quality plants
# ---------------------------------------------------------------------------

def test_quality_plants(mini, tmp_path):
    gq = tmp_path / 'gq.tsv'
    gq.write_text('ID\tF\tF_Percentile_Rank\tHet_Flagged\tHet_Reason\tF_MISS\tMiss_Percentile_Rank\tMiss_Flagged\tMiss_Reason\n'
                  'S12\t0\t50\tno\t\t0.2\t99\tYES\tHIGH\n'
                  'S05\t0.1\t99\tYES\tHIGH\t0.01\t10\tno\t\n'
                  'S01\t0\t50\tno\t\t0.01\t10\tno\t\n')
    mi = tmp_path / 'mi.tsv'
    mi.write_text('Individual\tN_Families\tN_CoParents\tTotal_N\tTotal_Child\tRate\tPercentile_Rank\tFlagged\tFlag_Reason\tIMendel_N\n'
                  'S06\t3\t3\t30\t9\t0.1\t99\tTrue\tboth\tNA\nS02\t2\t2\t4\t4\t0.0\t10\tFalse\t\tNA\n')
    art = tmp_path / 'art.txt'
    art.write_text('ch01:5000:A:T\nch03:777:C:G\n')
    r, s, _ = run_cli(['--truth', mini, '--genotype-quality', str(gq), '--mendel-individual', str(mi),
                       '--mendel-artifacts', str(art)], tmp_path)
    assert r.returncode == 0, r.stderr
    q = s['quality']
    assert q['genotype_quality'] == dict(low_callrate=[1, 1], contaminated=[1, 1])
    assert q['mendel_individual'] == dict(evaluated=1, flagged=1, planted=1)
    assert q['bad_markers'] == dict(found=1, planted=1, extra=1)


def test_empty_mendel_artifact_list_is_skipped_not_scored(mini, tmp_path):
    art = tmp_path / 'art.txt'
    art.write_text('')
    r, s, out = run_cli(['--truth', mini, '--mendel-artifacts', str(art)], tmp_path)
    assert r.returncode == 0 and 'bad_markers' not in s['quality']
    assert 'list is empty' in open(out + '_score_report.txt').read()


# ---------------------------------------------------------------------------
# IBD / HBD
# ---------------------------------------------------------------------------

def test_ibd_coverage_false_pairs_and_conserved_block(mini, tmp_path):
    called = tmp_path / 'c.ibd.gz'
    _write_gz(str(called), [
        'S03\t1\tS01\t1\tch01\t100\t600\t5.0',            # reversed pair order; half of the 1000 bp
        'S03\t2\tS01\t2\tch02\t200\t700\t5.0',            # all 500 bp
        'S02\t1\tS03\t1\tch01\t1\t2001\t1.0',             # below --min-cm: ignored
        'S05\t1\tS06\t1\tch03\t10\t20\t3.0',              # pair with no true IBD
        'S07\t1\tS14\t1\tch04\t1500\t1800\t3.0'])         # no true IBD, but inside the conserved block
    r, s, _ = run_cli(['--truth', mini, '--ibd', str(called)], tmp_path)
    assert r.returncode == 0, r.stderr
    ibd = s['ibd']['c.ibd.gz']
    assert ibd['recall']['parent_offspring'] == pytest.approx(1000 / 3500)   # 1000 of (1500 + 2000) bp
    assert ibd['false_pairs'] == 2
    assert ibd['false_mb_outside_block'] == pytest.approx(10 / 1e6)          # the block pair adds nothing
    assert ibd['pairs_in_conserved_block'] == 1


def test_hbd_recovery_and_spurious(mini, tmp_path):
    called = tmp_path / 'c.hbd.gz'
    _write_gz(str(called), ['S09\t1\tS09\t2\tch01\t100\t600\t5.0', 'S12\t1\tS12\t2\tch05\t1\t100\t3.0'])
    r, s, _ = run_cli(['--truth', mini, '--hbd', str(called)], tmp_path)
    assert r.returncode == 0, r.stderr
    h = s['hbd']['c.hbd.gz']
    assert h['median_recovered'] == pytest.approx(0.25)    # S09 half recovered, S11 not at all
    assert h['spurious_samples'] == 1


# ---------------------------------------------------------------------------
# Sample universe and CLI behaviour
# ---------------------------------------------------------------------------

def test_samples_restricts_scoring_and_missing_parents_are_not_counted(mini, tmp_path):
    names = tmp_path / 'names.txt'
    names.write_text('S01\nS03\nS04\nS05\nS06\nS07\nNOT_IN_TRUTH\n')        # S02 absent: S03's pair is unfindable
    fam = tmp_path / 'x.fam'
    fam.write_text(''.join(f"F1 {i[0]} {'S05 S06' if i[0] == 'S07' else '0 0'} 0 -9\n" for i in MINI))
    r, s, _ = run_cli(['--truth', mini, '--samples', str(names), '--fam', str(fam)], tmp_path)
    assert r.returncode == 0, r.stderr
    assert 'not in the truth' in r.stderr                                  # the unknown name is reported
    assert s['fam']['simple']['not_in_input'] == 1 and s['fam']['simple']['n'] == 1
    assert s['fam']['simple']['correct'] == 1                              # S07 only


def test_requires_an_input_and_writes_report_files(mini, tmp_path):
    r, _, _ = run_cli(['--truth', mini], tmp_path)
    assert r.returncode == 2
    fam = tmp_path / 'x.fam'
    fam.write_text(''.join(f"F1 {i[0]} {i[1]} {i[2]} 0 -9\n" for i in MINI))
    r, s, out = run_cli(['--truth', mini, '--fam', str(fam)], tmp_path)
    assert r.returncode == 0
    assert os.path.exists(out + '_score_report.txt') and os.path.exists(out + '_score_children.tsv')
    assert s['fam']['simple']['correct'] == 4        # the truth itself scores perfectly


# ---------------------------------------------------------------------------
# Shipped truth: a perfect reconstruction must score perfectly
# ---------------------------------------------------------------------------

def test_shipped_truth_fam_scores_perfectly(tmp_path):
    r, s, _ = run_cli(['--truth', SHIPPED, '--fam', SHIPPED + '_truth.fam'], tmp_path)
    assert r.returncode == 0, r.stderr
    for kind in ('simple', 'inbred_child', 'related_parents', 'backcross'):
        f = s['fam'][kind]
        assert f['n'] > 0 and f['correct'] == f['n'], (kind, f)
    assert s['fam']['assigned_correct'] == s['fam']['assigned']


def test_shipped_truth_ibd_and_hbd_score_perfectly(tmp_path):
    r, s, _ = run_cli(['--truth', SHIPPED, '--ibd', SHIPPED + '_truth_ibd.ibd.gz',
                       '--hbd', SHIPPED + '_truth_hbd.hbd.gz'], tmp_path)
    assert r.returncode == 0, r.stderr
    ibd = s['ibd']['breeding_sim_truth_ibd.ibd.gz']
    assert all(v == pytest.approx(1.0) for v in ibd['recall'].values()) and ibd['false_pairs'] == 0
    hbd = s['hbd']['breeding_sim_truth_hbd.hbd.gz']
    assert hbd['pearson'] == pytest.approx(1.0) and hbd['median_recovered'] == pytest.approx(1.0)


def test_shipped_duplicates_as_mz_calls_score_perfectly(tmp_path):
    pair = tmp_path / 'x.pair'
    dups = list(csv.DictReader(open(SHIPPED + '_truth_duplicates.tsv'), delimiter='\t'))
    pair.write_text('Pair\tVerdict\n' + ''.join(f"{d['ID1']},{d['ID2']}\tMZ twins\n" for d in dups))
    r, s, _ = run_cli(['--truth', SHIPPED, '--pair', str(pair)], tmp_path)
    assert r.returncode == 0, r.stderr
    mz = s['pairs']['mz']
    assert mz['found'] == mz['true_pairs'] == len(dups) and mz['false_calls'] == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_runner_syntax_and_help():
    assert subprocess.run(['bash', '-n', RUNNER]).returncode == 0
    r = subprocess.run(['bash', RUNNER, '--help'], capture_output=True, text=True)
    assert r.returncode == 0 and 'Stages' in r.stdout and '--true-map' in r.stdout


def test_runner_requires_output_dir():
    r = subprocess.run(['bash', RUNNER], capture_output=True, text=True)
    assert r.returncode == 2 and '-o OUTDIR' in r.stderr


def test_runner_score_stage_assembles_the_scorer_call(tmp_path):
    """--stages score picks up whatever outputs exist in OUTDIR (here: a perfect .fam and .pair)."""
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'step3_named_filtered_refined.fam').write_text(open(SHIPPED + '_truth.fam').read())
    dups = list(csv.DictReader(open(SHIPPED + '_truth_duplicates.tsv'), delimiter='\t'))
    (out / 'step1_named.pair').write_text('Pair\tVerdict\n' + ''.join(f"{d['ID1']},{d['ID2']}\tMZ twins\n" for d in dups))
    env = dict(os.environ, PYTHON=sys.executable)
    r = subprocess.run(['bash', RUNNER, '-o', str(out), '--stages', 'score'], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.load(open(out / 'validation_score_summary.json'))
    assert 'fam' in summary and 'pairs' in summary and 'trios' not in summary
    assert summary['pairs']['mz']['false_calls'] == 0
    assert (out / '.done_score').exists()
    # a second run is a no-op unless forced
    r2 = subprocess.run(['bash', RUNNER, '-o', str(out), '--stages', 'score'], capture_output=True,
                        text=True, env=env)
    assert 'already done' in r2.stdout
