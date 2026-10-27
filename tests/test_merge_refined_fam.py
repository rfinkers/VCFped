"""
Tests for utilities/merge_refined_fam.py.

Covers merging a candidate-subset refined FAM (from vcfped-rerun-on-candidates
verification) back into the full-cohort refined FAM:
  - founders in the base FAM receive newly-confirmed parents
  - individuals with DIFFERENT existing parents in both FAMs are treated as
    conflicts, not silently overwritten (both --on-conflict modes)
  - individuals only present in the new FAM get appended
  - already-matching entries are recognised as no-ops
  - base FAM row order is preserved; new individuals are appended at the end
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

MERGE_SCRIPT = os.path.join(UTILITIES_DIR, 'merge_refined_fam.py')


@pytest.fixture
def mrf():
    return import_module_from_path('merge_refined_fam', MERGE_SCRIPT)


# ---------------------------------------------------------------------------
# load_fam / write_fam round-trip
# ---------------------------------------------------------------------------

def test_load_fam_parses_rows(mrf, tmp_path):
    f = tmp_path / "test.fam"
    f.write_text("0\tCHILD\tFATHER\tMOTHER\t0\t-9\n0\tFOUNDER\t0\t0\t1\t-9\n")
    fam, order = mrf.load_fam(str(f))
    assert fam['CHILD'] == {'fid': '0', 'pat': 'FATHER', 'mat': 'MOTHER', 'sex': '0'}
    assert fam['FOUNDER'] == {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '1'}
    assert order == ['CHILD', 'FOUNDER']


def test_write_fam_preserves_order(mrf, tmp_path):
    fam = {'B': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'},
           'A': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}}
    out = tmp_path / "out.fam"
    mrf.write_fam(str(out), fam, ['B', 'A'])
    lines = out.read_text().splitlines()
    assert lines[0].split()[1] == 'B'
    assert lines[1].split()[1] == 'A'


# ---------------------------------------------------------------------------
# merge() logic
# ---------------------------------------------------------------------------

def test_founder_receives_new_parents(mrf):
    base_fam = {'CHILD': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}}
    base_order = ['CHILD']
    new_fam = {'CHILD': {'fid': '0', 'pat': 'FATHER', 'mat': 'MOTHER', 'sex': '0'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('base'))

    assert merged['CHILD']['pat'] == 'FATHER'
    assert merged['CHILD']['mat'] == 'MOTHER'
    assert stats['parents_added'] == ['CHILD']
    assert stats['conflicts'] == []


def test_conflict_keeps_base_by_default(mrf):
    base_fam = {'CHILD': {'fid': '0', 'pat': 'REAL_DAD', 'mat': 'REAL_MOM', 'sex': '0'}}
    base_order = ['CHILD']
    new_fam = {'CHILD': {'fid': '0', 'pat': 'OTHER_DAD', 'mat': 'OTHER_MOM', 'sex': '0'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('base'))

    assert merged['CHILD']['pat'] == 'REAL_DAD'
    assert merged['CHILD']['mat'] == 'REAL_MOM'
    assert len(stats['conflicts']) == 1
    assert stats['conflicts'][0][0] == 'CHILD'
    assert stats['conflicts'][0][-1] == 'base'
    assert stats['parents_added'] == []


def test_conflict_keep_new_overwrites(mrf):
    base_fam = {'CHILD': {'fid': '0', 'pat': 'REAL_DAD', 'mat': 'REAL_MOM', 'sex': '0'}}
    base_order = ['CHILD']
    new_fam = {'CHILD': {'fid': '0', 'pat': 'OTHER_DAD', 'mat': 'OTHER_MOM', 'sex': '0'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('new'))

    assert merged['CHILD']['pat'] == 'OTHER_DAD'
    assert merged['CHILD']['mat'] == 'OTHER_MOM'
    assert len(stats['conflicts']) == 1
    assert stats['conflicts'][0][-1] == 'new'
    # Regression guard: base_d = merged_fam[iid] is a reference, so mutating
    # merged_fam[iid]['pat']/['mat'] for the 'new' outcome must NOT corrupt
    # the recorded base_pat/base_mat in the conflicts tuple (indices 1, 2) --
    # they must still show the ORIGINAL base assignment, not the new one.
    iid, base_pat, base_mat, new_pat, new_mat, kept = stats['conflicts'][0]
    assert (base_pat, base_mat) == ('REAL_DAD', 'REAL_MOM')
    assert (new_pat, new_mat) == ('OTHER_DAD', 'OTHER_MOM')


def test_new_individual_appended(mrf):
    base_fam = {'A': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}}
    base_order = ['A']
    new_fam = {'B': {'fid': '0', 'pat': 'P1', 'mat': 'P2', 'sex': '0'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('base'))

    assert 'B' in merged
    assert merged['B'] == {'fid': '0', 'pat': 'P1', 'mat': 'P2', 'sex': '0'}
    assert stats['new_individuals'] == ['B']
    assert order == ['A', 'B']  # base order preserved, new appended at end


def test_matching_entry_is_unchanged_not_conflict(mrf):
    base_fam = {'CHILD': {'fid': '0', 'pat': 'DAD', 'mat': 'MOM', 'sex': '0'}}
    base_order = ['CHILD']
    new_fam = {'CHILD': {'fid': '0', 'pat': 'DAD', 'mat': 'MOM', 'sex': '0'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('base'))

    assert stats['unchanged'] == ['CHILD']
    assert stats['conflicts'] == []
    assert stats['parents_added'] == []


def test_sex_filled_in_from_new_fam_when_base_unknown(mrf):
    base_fam = {'CHILD': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}}
    base_order = ['CHILD']
    new_fam = {'CHILD': {'fid': '0', 'pat': 'DAD', 'mat': 'MOM', 'sex': '2'}}

    merged, order, stats = mrf.merge(base_fam, base_order, new_fam, mrf.fixed_resolver('base'))

    assert merged['CHILD']['sex'] == '2'


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_end_to_end_cli(tmp_path):
    base_path = tmp_path / "base.fam"
    base_path.write_text(
        "0\tFOUNDER1\t0\t0\t1\t-9\n"
        "0\tFOUNDER2\t0\t0\t2\t-9\n"
        "0\tCONFIRMED\tOLD_DAD\tOLD_MOM\t0\t-9\n"
    )
    new_path = tmp_path / "new.fam"
    new_path.write_text(
        "0\tFOUNDER1\tNEW_DAD\tNEW_MOM\t1\t-9\n"       # founder -> gets parents
        "0\tCONFIRMED\tRIVAL_DAD\tRIVAL_MOM\t0\t-9\n"  # conflict -> kept base
        "0\tBRAND_NEW\tP1\tP2\t0\t-9\n"                 # new individual
    )
    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [str(base_path), str(new_path), '-o', out_prefix])
    assert result.returncode == 0, result.stderr

    assert os.path.exists(f"{out_prefix}.fam")
    assert os.path.exists(f"{out_prefix}_merge_report.txt")

    fam_lines = open(f"{out_prefix}.fam").read().splitlines()
    fam = {l.split()[1]: l.split() for l in fam_lines}

    assert fam['FOUNDER1'][2:4] == ['NEW_DAD', 'NEW_MOM']
    assert fam['CONFIRMED'][2:4] == ['OLD_DAD', 'OLD_MOM']  # conflict kept base
    assert fam['FOUNDER2'][2:4] == ['0', '0']                 # untouched founder
    assert 'BRAND_NEW' in fam
    assert fam['BRAND_NEW'][2:4] == ['P1', 'P2']

    report = open(f"{out_prefix}_merge_report.txt").read()
    assert 'FOUNDER1' in report
    assert 'CONFIRMED' in report
    assert 'BRAND_NEW' in report


def test_default_output_prefix_derived_from_base_fam(tmp_path):
    base_path = tmp_path / "cohort_refined.fam"
    base_path.write_text("0\tA\t0\t0\t0\t-9\n")
    new_path = tmp_path / "new.fam"
    new_path.write_text("0\tA\tD\tM\t0\t-9\n")

    result = run_script(MERGE_SCRIPT, [str(base_path), str(new_path)], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert os.path.exists(str(tmp_path / "cohort_refined_merged.fam"))


def test_missing_input_files_error_cleanly(tmp_path):
    result = run_script(MERGE_SCRIPT, [str(tmp_path / "nope.fam"), str(tmp_path / "also_nope.fam")])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Conflict evidence: trio lookup, KING lookup, and the joined table
# ---------------------------------------------------------------------------

def test_default_trio_path_derivation(mrf):
    assert mrf.default_trio_path("foo_refined.fam") == "foo_refined.trio"
    assert mrf.default_trio_path("foo") == "foo.trio"


def test_load_trio_lookup_order_independent(mrf, tmp_path):
    f = tmp_path / "x.trio"
    f.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "DAD,MOM,CHILD\tCHILD\t50\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    lookup = mrf.load_trio_lookup(str(f))
    info_forward = mrf.lookup_trio_info(lookup, 'CHILD', 'DAD', 'MOM')
    info_reverse = mrf.lookup_trio_info(lookup, 'CHILD', 'MOM', 'DAD')
    assert info_forward['verdict'] == 'Regular trio'
    assert info_forward['perc'] == '50'
    assert info_forward['test1'] == '100.0'
    assert info_reverse == info_forward


def test_load_trio_lookup_missing_file_returns_empty(mrf, tmp_path):
    assert mrf.load_trio_lookup(str(tmp_path / "nope.trio")) == {}
    assert mrf.load_trio_lookup(None) == {}


def test_lookup_trio_info_defaults_to_na(mrf):
    info = mrf.lookup_trio_info({}, 'CHILD', 'DAD', 'MOM')
    assert info == {'perc': 'NA', 'test1': 'NA', 'test2': 'NA', 'verdict': 'NA'}


def test_load_king_lookup_parses_kin0(mrf, tmp_path):
    f = tmp_path / "test_king.kin0"
    f.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tDAD\t1000\t0.3\t0.005\t0.26\n"
    )
    king = mrf.load_king_lookup(str(tmp_path / "test_king"))
    assert mrf.get_king(king, 'CHILD', 'DAD') == {'kinship': 0.26, 'ibs0': 0.005}
    assert mrf.get_king(king, 'DAD', 'CHILD') == {'kinship': 0.26, 'ibs0': 0.005}  # order-independent


def test_load_king_lookup_missing_returns_empty(mrf, tmp_path):
    assert mrf.load_king_lookup(None) == {}
    assert mrf.load_king_lookup(str(tmp_path / "nope")) == {}


def test_build_conflict_table_joins_evidence_with_na_fallback(mrf):
    conflicts = [('CHILD', 'BASE_DAD', 'BASE_MOM', 'NEW_DAD', 'NEW_MOM', 'base')]
    base_trio = mrf.load_trio_lookup(None)  # empty -> NA fallback
    new_trio = {('CHILD', 'NEW_DAD', 'NEW_MOM'): {'perc': '10', 'test1': '95.8', 'test2': '8.3',
                                                    'verdict': 'Regular trio (reclassified from inverted)'},
                ('CHILD', 'NEW_MOM', 'NEW_DAD'): {'perc': '10', 'test1': '95.8', 'test2': '8.3',
                                                    'verdict': 'Regular trio (reclassified from inverted)'}}
    king = {('CHILD', 'NEW_DAD'): {'kinship': 0.24, 'ibs0': 0.004}}

    rows = mrf.build_conflict_table(conflicts, base_trio, new_trio, king)
    assert len(rows) == 1
    r = rows[0]
    assert r['child'] == 'CHILD'
    assert r['base_verdict'] == 'NA'  # no base trio data provided
    assert r['base_kinship_p1'] == 'NA'
    assert r['new_verdict'] == 'Regular trio (reclassified from inverted)'
    assert r['new_perc'] == '10'
    assert r['new_kinship_p1'] == 0.24
    assert r['new_kinship_p2'] == 'NA'  # NEW_MOM not in king
    assert r['kept'] == 'base'


def test_end_to_end_cli_writes_conflicts_tsv_with_evidence(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tOLD_DAD\tOLD_MOM\t0\t-9\n")
    base_trio = tmp_path / "base.trio"
    base_trio.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "OLD_DAD,OLD_MOM,CHILD\tCHILD\t50\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tNEW_DAD\tNEW_MOM\t0\t-9\n")
    new_trio = tmp_path / "new.trio"
    new_trio.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "NEW_DAD,NEW_MOM,CHILD\tCHILD\t10\t1905\t120\t115\t95.8\t60\t5\t8.3\tRegular trio (reclassified from inverted)\n"
    )
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tOLD_DAD\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD\tF1\tNEW_DAD\t1000\t0.3\t0.20\t0.10\n"  # IBS0 too high -> not PO
    )

    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '-o', out_prefix,
        '--base-trio', str(base_trio), '--new-trio', str(new_trio),
        '--king', str(tmp_path / "test_king"),
    ])
    assert result.returncode == 0, result.stderr

    conflicts_path = f"{out_prefix}_conflicts.tsv"
    assert os.path.exists(conflicts_path)
    lines = open(conflicts_path).read().splitlines()
    header = lines[0].split('\t')
    row = dict(zip(header, lines[1].split('\t')))

    assert row['Child'] == 'CHILD'
    assert row['Base_Verdict'] == 'Regular trio'
    assert row['Base_Perc'] == '50'
    assert row['Base_Kinship_P1'] == '0.26'
    assert row['New_Verdict'] == 'Regular trio (reclassified from inverted)'
    assert row['New_Perc'] == '10'
    assert row['New_Kinship_P1'] == '0.1'
    assert row['New_IBS0_P1'] == '0.2'
    assert row['Kept'] == 'base'

    # The narrative report should also surface the evidence, not just IDs.
    report = open(f"{out_prefix}_merge_report.txt").read()
    assert 'Regular trio' in report
    assert 'reclassified from inverted' in report


def test_no_conflicts_tsv_written_when_no_conflicts(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tA\t0\t0\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tA\tD\tM\t0\t-9\n")
    out_prefix = str(tmp_path / "merged")

    result = run_script(MERGE_SCRIPT, [str(base_fam), str(new_fam), '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert not os.path.exists(f"{out_prefix}_conflicts.tsv")


def test_auto_derived_trio_paths_used_when_present(tmp_path):
    """If --base-trio/--new-trio aren't given, the companion <fam>_refined.trio
    files (same prefix, .fam -> .trio) should be picked up automatically."""
    base_fam = tmp_path / "cohort_refined.fam"
    base_fam.write_text("0\tCHILD\tOLD_DAD\tOLD_MOM\t0\t-9\n")
    (tmp_path / "cohort_refined.trio").write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "OLD_DAD,OLD_MOM,CHILD\tCHILD\t50\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    new_fam = tmp_path / "subset_refined.fam"
    new_fam.write_text("0\tCHILD\tNEW_DAD\tNEW_MOM\t0\t-9\n")
    (tmp_path / "subset_refined.trio").write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "NEW_DAD,NEW_MOM,CHILD\tCHILD\t10\t1905\t120\t115\t95.8\t60\t5\t8.3\tRegular trio\n"
    )
    out_prefix = str(tmp_path / "merged")

    result = run_script(MERGE_SCRIPT, [str(base_fam), str(new_fam), '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert 'not found' not in result.stdout or 'cohort_refined.trio' not in result.stdout

    conflicts_path = f"{out_prefix}_conflicts.tsv"
    row = dict(zip(open(conflicts_path).read().splitlines()[0].split('\t'),
                    open(conflicts_path).read().splitlines()[1].split('\t')))
    assert row['Base_Verdict'] == 'Regular trio'
    assert row['New_Verdict'] == 'Regular trio'


# ---------------------------------------------------------------------------
# --score-conflicts
# ---------------------------------------------------------------------------

def test_ibs0_threshold_matches_refine_pedigree_formula(mrf):
    """Same formula/values as refine_pedigree.py's ibs0_threshold_for_error_rate()."""
    assert mrf.ibs0_threshold_for_error_rate(0.025) == pytest.approx(0.075)
    assert mrf.ibs0_threshold_for_error_rate(10.0) <= 0.10  # capped


def test_compute_kinship_percentiles(mrf):
    king = {(f'A{i}', f'B{i}'): {'kinship': k, 'ibs0': 0.001}
            for i, k in enumerate([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])}
    p25, p75, p95 = mrf.compute_kinship_percentiles(king)
    assert p25 == 0.15  # sorted[8//4] = sorted[2]
    assert p75 == 0.35  # sorted[3*8//4] = sorted[6]
    assert p95 == 0.40  # sorted[int(0.95*8)] = sorted[7]


def test_compute_kinship_percentiles_empty(mrf):
    assert mrf.compute_kinship_percentiles({}) == (None, None, None)


def test_classify_parent(mrf):
    ceiling, p25, p75 = 0.075, 0.122, 0.232
    assert mrf.classify_parent(None, None, ceiling, p25, p75) == 'UNTESTED'
    assert mrf.classify_parent(0.30, 0.10, ceiling, p25, p75) == 'REJECTED'  # IBS0 too high
    assert mrf.classify_parent(0.30, 0.001, ceiling, p25, p75) == 'STRONG'   # >= p75
    assert mrf.classify_parent(0.15, 0.001, ceiling, p25, p75) == 'MODERATE'  # p25-p75
    assert mrf.classify_parent(0.05, 0.001, ceiling, p25, p75) == 'WEAK'     # < p25


def test_score_conflict_row_prefer_new(mrf):
    """Base has an untested parent; new has both parents at MODERATE -> prefer_new."""
    row = {
        'base_p1': 'BASE_DAD', 'base_p2': 'BASE_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.30', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': 'NA', 'base_ibs0_p2': 'NA',
        'new_kinship_p1': '0.20', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.15', 'new_ibs0_p2': '0.001',
    }
    base_overall, new_overall, rec, base_mendel_rate, new_mendel_rate = \
        mrf.score_conflict_row(row, 0.075, 0.122, 0.232)
    assert base_overall == 'UNTESTED'
    assert new_overall == 'MODERATE'
    assert rec == 'prefer_new'
    assert base_mendel_rate is None
    assert new_mendel_rate is None


def test_score_conflict_row_prefer_base(mrf):
    """Base has both parents STRONG; new has one WEAK -> prefer_base."""
    row = {
        'base_p1': 'BASE_DAD', 'base_p2': 'BASE_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.30', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': '0.25', 'base_ibs0_p2': '0.001',
        'new_kinship_p1': '0.30', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.05', 'new_ibs0_p2': '0.001',
    }
    base_overall, new_overall, rec, _, _ = mrf.score_conflict_row(row, 0.075, 0.122, 0.232)
    assert base_overall == 'STRONG'
    assert new_overall == 'WEAK'
    assert rec == 'prefer_base'


def test_score_conflict_row_tie_new_higher(mrf):
    """Same tier (both WEAK) but new's weakest-link kinship is notably higher."""
    row = {
        'base_p1': 'BASE_DAD', 'base_p2': 'BASE_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.30', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': '0.01', 'base_ibs0_p2': '0.001',
        'new_kinship_p1': '0.30', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.10', 'new_ibs0_p2': '0.001',
    }
    base_overall, new_overall, rec, _, _ = mrf.score_conflict_row(row, 0.075, 0.122, 0.232)
    assert base_overall == new_overall == 'WEAK'
    assert rec == 'tie_new_higher'


def test_score_conflicts_requires_king(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tOTHER_DAD\tOTHER_MOM\t0\t-9\n")

    result = run_script(MERGE_SCRIPT, [str(base_fam), str(new_fam), '--score-conflicts'])
    assert result.returncode != 0
    assert 'requires --king' in (result.stderr + result.stdout)


def test_end_to_end_score_conflicts_populates_tsv_and_report(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tOLD_DAD\tOLD_MOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tNEW_DAD\tNEW_MOM\t0\t-9\n")
    king_file = tmp_path / "test_king.kin0"
    # Build a cohort where OLD_DAD/OLD_MOM look weak and NEW_DAD/NEW_MOM look
    # strong, plus filler pairs so percentiles are meaningful.
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tOLD_DAD\t1000\t0.3\t0.001\t0.05\n"
        "F1\tCHILD\tF1\tOLD_MOM\t1000\t0.3\t0.001\t0.06\n"
        "F1\tCHILD\tF1\tNEW_DAD\t1000\t0.3\t0.001\t0.35\n"
        "F1\tCHILD\tF1\tNEW_MOM\t1000\t0.3\t0.001\t0.30\n"
        "F1\tA\tF1\tB\t1000\t0.3\t0.001\t0.10\n"
        "F1\tC\tF1\tD\t1000\t0.3\t0.001\t0.15\n"
        "F1\tE\tF1\tF\t1000\t0.3\t0.001\t0.20\n"
        "F1\tG\tF1\tH\t1000\t0.3\t0.001\t0.25\n"
    )
    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '-o', out_prefix,
        '--king', str(tmp_path / "test_king"), '--score-conflicts',
    ])
    assert result.returncode == 0, result.stderr

    conflicts_path = f"{out_prefix}_conflicts.tsv"
    lines = open(conflicts_path).read().splitlines()
    header = lines[0].split('\t')
    row = dict(zip(header, lines[1].split('\t')))

    assert 'Base_Overall' in header
    assert 'New_Overall' in header
    assert 'Recommendation' in header
    assert row['Recommendation'] == 'prefer_new'

    report = open(f"{out_prefix}_merge_report.txt").read()
    assert 'Flagged for review' in report
    assert 'CHILD' in report
    assert 'prefer_new' in report


# ---------------------------------------------------------------------------
# --auto-resolve
# ---------------------------------------------------------------------------

def test_fixed_resolver_always_returns_same_choice(mrf):
    resolver = mrf.fixed_resolver('base')
    assert resolver('CHILD', 'P1', 'P2', 'P3', 'P4') == 'base'
    assert resolver('OTHER', 'X', 'Y', 'Z', 'W') == 'base'


def test_score_conflict_kinship_matches_score_conflict_row(mrf):
    """score_conflict_kinship() (raw KING lookups) must agree with
    score_conflict_row() (pre-built evidence row) for the same data."""
    king = {
        ('CHILD', 'BASE_DAD'): {'kinship': 0.05, 'ibs0': 0.001},
        ('CHILD', 'BASE_MOM'): {'kinship': 0.06, 'ibs0': 0.001},
        ('CHILD', 'NEW_DAD'): {'kinship': 0.30, 'ibs0': 0.001},
        ('CHILD', 'NEW_MOM'): {'kinship': 0.25, 'ibs0': 0.001},
    }
    ceiling, p25, p75 = 0.075, 0.122, 0.232

    rec_direct = mrf.score_conflict_kinship('CHILD', 'BASE_DAD', 'BASE_MOM', 'NEW_DAD', 'NEW_MOM',
                                             king, ceiling, p25, p75)

    row = {
        'base_p1': 'BASE_DAD', 'base_p2': 'BASE_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.05', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': '0.06', 'base_ibs0_p2': '0.001',
        'new_kinship_p1': '0.30', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.25', 'new_ibs0_p2': '0.001',
    }
    _, _, rec_from_row, _, _ = mrf.score_conflict_row(row, ceiling, p25, p75)

    assert rec_direct == rec_from_row == 'prefer_new'


def test_auto_resolve_requires_score_conflicts(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tOTHER_DAD\tOTHER_MOM\t0\t-9\n")

    result = run_script(MERGE_SCRIPT, [str(base_fam), str(new_fam), '--auto-resolve'])
    assert result.returncode != 0
    assert 'requires --score-conflicts' in (result.stderr + result.stdout)


def test_auto_resolve_applies_per_conflict_recommendation(tmp_path):
    """
    Two children with conflicts: STRONG_CHILD's new evidence clearly beats
    base (should auto-resolve to new); WEAK_CHILD's new evidence is clearly
    worse than base (should auto-resolve to base). A single global
    --on-conflict choice could not produce this mixed outcome.
    """
    base_fam = tmp_path / "base.fam"
    base_fam.write_text(
        "0\tSTRONG_CHILD\tOLD_DAD1\tOLD_MOM1\t0\t-9\n"
        "0\tWEAK_CHILD\tOLD_DAD2\tOLD_MOM2\t0\t-9\n"
    )
    new_fam = tmp_path / "new.fam"
    new_fam.write_text(
        "0\tSTRONG_CHILD\tNEW_DAD1\tNEW_MOM1\t0\t-9\n"
        "0\tWEAK_CHILD\tNEW_DAD2\tNEW_MOM2\t0\t-9\n"
    )
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        # STRONG_CHILD: old parents weak, new parents strong -> prefer_new
        "F1\tSTRONG_CHILD\tF1\tOLD_DAD1\t1000\t0.3\t0.001\t0.05\n"
        "F1\tSTRONG_CHILD\tF1\tOLD_MOM1\t1000\t0.3\t0.001\t0.06\n"
        "F1\tSTRONG_CHILD\tF1\tNEW_DAD1\t1000\t0.3\t0.001\t0.35\n"
        "F1\tSTRONG_CHILD\tF1\tNEW_MOM1\t1000\t0.3\t0.001\t0.30\n"
        # WEAK_CHILD: old parents strong, new parents weak -> prefer_base
        "F1\tWEAK_CHILD\tF1\tOLD_DAD2\t1000\t0.3\t0.001\t0.34\n"
        "F1\tWEAK_CHILD\tF1\tOLD_MOM2\t1000\t0.3\t0.001\t0.29\n"
        "F1\tWEAK_CHILD\tF1\tNEW_DAD2\t1000\t0.3\t0.001\t0.04\n"
        "F1\tWEAK_CHILD\tF1\tNEW_MOM2\t1000\t0.3\t0.001\t0.07\n"
        # filler pairs so percentiles are meaningful
        "F1\tA\tF1\tB\t1000\t0.3\t0.001\t0.10\n"
        "F1\tC\tF1\tD\t1000\t0.3\t0.001\t0.15\n"
        "F1\tE\tF1\tF\t1000\t0.3\t0.001\t0.20\n"
        "F1\tG\tF1\tH\t1000\t0.3\t0.001\t0.25\n"
    )
    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '-o', out_prefix,
        '--king', str(tmp_path / "test_king"), '--score-conflicts', '--auto-resolve',
        # --on-conflict must be ignored when --auto-resolve is set
        '--on-conflict', 'keep-base',
    ])
    assert result.returncode == 0, result.stderr

    fam = {}
    for line in open(f"{out_prefix}.fam"):
        parts = line.split()
        fam[parts[1]] = (parts[2], parts[3])

    assert fam['STRONG_CHILD'] == ('NEW_DAD1', 'NEW_MOM1'), "prefer_new conflict should resolve to new"
    assert fam['WEAK_CHILD'] == ('OLD_DAD2', 'OLD_MOM2'), "prefer_base conflict should resolve to base"

    assert 'auto-resolve' in result.stdout


# ---------------------------------------------------------------------------
# Mendelian veto (--base-mendel/--new-mendel)
# ---------------------------------------------------------------------------

def test_load_fmendel_lookup_parses_by_sorted_parent_pair(mrf, tmp_path):
    f = tmp_path / "test.fmendel"
    f.write_text(
        "FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tZDAD\tAMOM\t5\t3\n"
    )
    lookup = mrf.load_fmendel_lookup(str(f))
    assert lookup[('AMOM', 'ZDAD')] == {'n': 3, 'child': 5}
    # get_mendel_rate() looks it up order-independently via the same sort
    assert mrf.get_mendel_rate(lookup, 100, 'ZDAD', 'AMOM') == pytest.approx(3 / 500)
    assert mrf.get_mendel_rate(lookup, 100, 'AMOM', 'ZDAD') == pytest.approx(3 / 500)


def test_load_fmendel_lookup_missing_returns_empty(mrf, tmp_path):
    assert mrf.load_fmendel_lookup(None) == {}
    assert mrf.load_fmendel_lookup(str(tmp_path / "nope.fmendel")) == {}


def test_get_mendel_rate_untested_cases(mrf):
    lookup = {('AMOM', 'ZDAD'): {'n': 3, 'child': 5}}
    assert mrf.get_mendel_rate({}, 100, 'ZDAD', 'AMOM') is None
    assert mrf.get_mendel_rate(lookup, None, 'ZDAD', 'AMOM') is None
    assert mrf.get_mendel_rate(lookup, 100, 'OTHER1', 'OTHER2') is None


def test_expected_baseline_rate(mrf):
    assert mrf.expected_baseline_rate(0.025) == pytest.approx(0.05)


def test_is_mendel_elevated(mrf):
    assert mrf.is_mendel_elevated(None, 0.025, 2.0) is None  # untested
    assert mrf.is_mendel_elevated(0.03, 0.025, 2.0) is False  # below 2x0.05=0.10
    assert mrf.is_mendel_elevated(0.15, 0.025, 2.0) is True   # above 0.10


def test_apply_mendel_veto_only_downgrades_the_winning_side(mrf):
    # prefer_new but the NEW side is flagged -> vetoed
    assert mrf.apply_mendel_veto('prefer_new', True, False) == 'tie_mendel_veto'
    # prefer_new but the LOSING (base) side is flagged -> unchanged
    assert mrf.apply_mendel_veto('prefer_new', False, True) == 'prefer_new'
    # prefer_base but the BASE side is flagged -> vetoed
    assert mrf.apply_mendel_veto('prefer_base', False, True) == 'tie_mendel_veto'
    # prefer_base but the LOSING (new) side is flagged -> unchanged
    assert mrf.apply_mendel_veto('prefer_base', True, False) == 'prefer_base'
    # tie_new_higher favors new -> new side flagged vetoes it
    assert mrf.apply_mendel_veto('tie_new_higher', True, False) == 'tie_mendel_veto'
    # a plain tie has no "winning side" -- never vetoed, regardless of flags
    assert mrf.apply_mendel_veto('tie', True, True) == 'tie'
    # untested (None) never vetoes
    assert mrf.apply_mendel_veto('prefer_new', None, None) == 'prefer_new'


def test_score_conflict_row_mendel_veto_downgrades_prefer_new(mrf):
    """Kinship evidence alone clearly favors new (STRONG vs UNTESTED), but
    the new pair's own family shows an elevated Mendelian error rate --
    the recommendation should be downgraded to tie_mendel_veto."""
    row = {
        'base_p1': 'OLD_DAD', 'base_p2': 'OLD_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.05', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': 'NA', 'base_ibs0_p2': 'NA',
        'new_kinship_p1': '0.30', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.25', 'new_ibs0_p2': '0.001',
    }
    new_mendel_lookup = {('NEW_DAD', 'NEW_MOM'): {'n': 60, 'child': 5}}  # rate 0.12 > 0.10 threshold

    base_overall, new_overall, rec, base_rate, new_rate = mrf.score_conflict_row(
        row, 0.075, 0.122, 0.232, base_mendel_lookup=None, new_mendel_lookup=new_mendel_lookup,
        mendel_n_variants=100, error_rate=0.025, mendel_absolute_multiplier=2.0)

    assert new_overall == 'STRONG'  # kinship alone would clearly favor new
    assert rec == 'tie_mendel_veto'
    assert base_rate is None
    assert new_rate == pytest.approx(0.12)


def test_score_conflict_row_no_mendel_data_is_unaffected(mrf):
    """Without any --base-mendel/--new-mendel lookups, behavior is identical
    to the pre-veto scoring (regression guard for the default/no-mendel path)."""
    row = {
        'base_p1': 'OLD_DAD', 'base_p2': 'OLD_MOM',
        'new_p1': 'NEW_DAD', 'new_p2': 'NEW_MOM',
        'base_kinship_p1': '0.05', 'base_ibs0_p1': '0.001',
        'base_kinship_p2': 'NA', 'base_ibs0_p2': 'NA',
        'new_kinship_p1': '0.30', 'new_ibs0_p1': '0.001',
        'new_kinship_p2': '0.25', 'new_ibs0_p2': '0.001',
    }
    _, _, rec, base_rate, new_rate = mrf.score_conflict_row(row, 0.075, 0.122, 0.232)
    assert rec == 'prefer_new'
    assert base_rate is None
    assert new_rate is None


def test_score_conflict_kinship_applies_mendel_veto(mrf):
    king = {
        ('CHILD', 'OLD_DAD'): {'kinship': 0.05, 'ibs0': 0.001},
        ('CHILD', 'NEW_DAD'): {'kinship': 0.30, 'ibs0': 0.001},
        ('CHILD', 'NEW_MOM'): {'kinship': 0.25, 'ibs0': 0.001},
    }
    new_mendel_lookup = {('NEW_DAD', 'NEW_MOM'): {'n': 60, 'child': 5}}

    rec = mrf.score_conflict_kinship(
        'CHILD', 'OLD_DAD', 'OLD_MOM', 'NEW_DAD', 'NEW_MOM',
        king, 0.075, 0.122, 0.232,
        base_mendel_lookup=None, new_mendel_lookup=new_mendel_lookup,
        mendel_n_variants=100, error_rate=0.025, mendel_absolute_multiplier=2.0)

    assert rec == 'tie_mendel_veto'


def test_mendel_requires_score_conflicts(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tOTHER_DAD\tOTHER_MOM\t0\t-9\n")
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text("FID\tPAT\tMAT\tCHLD\tN\nF1\tDAD\tMOM\t5\t1\n")

    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '--base-mendel', str(fmendel), '--mendel-n-variants', '100',
    ])
    assert result.returncode != 0
    assert 'requires --score-conflicts' in (result.stderr + result.stdout)


def test_mendel_requires_n_variants_or_bim(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tOTHER_DAD\tOTHER_MOM\t0\t-9\n")
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tDAD\t1000\t0.3\t0.001\t0.05\n"
    )
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text("FID\tPAT\tMAT\tCHLD\tN\nF1\tDAD\tMOM\t5\t1\n")

    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '--king', str(tmp_path / "test_king"),
        '--score-conflicts', '--base-mendel', str(fmendel),
    ])
    assert result.returncode != 0
    assert 'requires --mendel-bim or --mendel-n-variants' in (result.stderr + result.stdout)


def test_mendel_bim_and_n_variants_mutually_exclusive(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tOTHER_DAD\tOTHER_MOM\t0\t-9\n")
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tDAD\t1000\t0.3\t0.001\t0.05\n"
    )
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text("FID\tPAT\tMAT\tCHLD\tN\nF1\tDAD\tMOM\t5\t1\n")
    bim = tmp_path / "test.bim"
    bim.write_text("1\tsnp1\t0\t100\tA\tG\n")

    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '--king', str(tmp_path / "test_king"),
        '--score-conflicts', '--base-mendel', str(fmendel),
        '--mendel-bim', str(bim), '--mendel-n-variants', '100',
    ])
    assert result.returncode != 0
    assert 'not both' in (result.stderr + result.stdout)


def test_end_to_end_mendel_veto_downgrades_recommendation(tmp_path):
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tOLD_DAD\tOLD_MOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tNEW_DAD\tNEW_MOM\t0\t-9\n")
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tOLD_DAD\t1000\t0.3\t0.001\t0.05\n"
        "F1\tCHILD\tF1\tOLD_MOM\t1000\t0.3\t0.001\t0.06\n"
        "F1\tCHILD\tF1\tNEW_DAD\t1000\t0.3\t0.001\t0.35\n"
        "F1\tCHILD\tF1\tNEW_MOM\t1000\t0.3\t0.001\t0.30\n"
        "F1\tA\tF1\tB\t1000\t0.3\t0.001\t0.10\n"
        "F1\tC\tF1\tD\t1000\t0.3\t0.001\t0.15\n"
        "F1\tE\tF1\tF\t1000\t0.3\t0.001\t0.20\n"
        "F1\tG\tF1\tH\t1000\t0.3\t0.001\t0.25\n"
    )
    # Without Mendel data this exact scenario resolves to plain 'prefer_new'
    # (see test_end_to_end_score_conflicts_populates_tsv_and_report). NEW_DAD
    # +NEW_MOM's family here has an elevated Mendelian error rate: with
    # n_variants=100 and 5 children, rate = 60/(100*5) = 0.12, above the
    # 2x0.05=0.10 default threshold -- this should veto it to tie_mendel_veto.
    new_mendel = tmp_path / "new_mendel.fmendel"
    new_mendel.write_text(
        "FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tNEW_DAD\tNEW_MOM\t5\t60\n"
    )

    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '-o', out_prefix,
        '--king', str(tmp_path / "test_king"), '--score-conflicts',
        '--new-mendel', str(new_mendel), '--mendel-n-variants', '100',
    ])
    assert result.returncode == 0, result.stderr

    conflicts_path = f"{out_prefix}_conflicts.tsv"
    lines = open(conflicts_path).read().splitlines()
    header = lines[0].split('\t')
    row = dict(zip(header, lines[1].split('\t')))
    assert 'Base_Mendel_Rate' in header
    assert 'New_Mendel_Rate' in header
    assert row['Recommendation'] == 'tie_mendel_veto'
    assert float(row['New_Mendel_Rate']) == pytest.approx(0.12)

    report = open(f"{out_prefix}_merge_report.txt").read()
    assert 'tie_mendel_veto' in report
    assert 'Mendelian error rate vetoed' in report


def test_end_to_end_mendel_veto_makes_auto_resolve_keep_base(tmp_path):
    """Same scenario as test_end_to_end_mendel_veto_downgrades_recommendation,
    but with --auto-resolve: without the veto this would auto-resolve to NEW
    (clear prefer_new by kinship alone); the Mendelian veto downgrades the
    recommendation to tie_mendel_veto, and --auto-resolve's conservative
    mapping (only 'prefer_new' keeps new) then keeps base instead."""
    base_fam = tmp_path / "base.fam"
    base_fam.write_text("0\tCHILD\tOLD_DAD\tOLD_MOM\t0\t-9\n")
    new_fam = tmp_path / "new.fam"
    new_fam.write_text("0\tCHILD\tNEW_DAD\tNEW_MOM\t0\t-9\n")
    king_file = tmp_path / "test_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tOLD_DAD\t1000\t0.3\t0.001\t0.05\n"
        "F1\tCHILD\tF1\tOLD_MOM\t1000\t0.3\t0.001\t0.06\n"
        "F1\tCHILD\tF1\tNEW_DAD\t1000\t0.3\t0.001\t0.35\n"
        "F1\tCHILD\tF1\tNEW_MOM\t1000\t0.3\t0.001\t0.30\n"
        "F1\tA\tF1\tB\t1000\t0.3\t0.001\t0.10\n"
        "F1\tC\tF1\tD\t1000\t0.3\t0.001\t0.15\n"
        "F1\tE\tF1\tF\t1000\t0.3\t0.001\t0.20\n"
        "F1\tG\tF1\tH\t1000\t0.3\t0.001\t0.25\n"
    )
    new_mendel = tmp_path / "new_mendel.fmendel"
    new_mendel.write_text(
        "FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tNEW_DAD\tNEW_MOM\t5\t60\n"
    )

    out_prefix = str(tmp_path / "merged")
    result = run_script(MERGE_SCRIPT, [
        str(base_fam), str(new_fam), '-o', out_prefix,
        '--king', str(tmp_path / "test_king"), '--score-conflicts', '--auto-resolve',
        '--new-mendel', str(new_mendel), '--mendel-n-variants', '100',
    ])
    assert result.returncode == 0, result.stderr

    fam = {}
    for line in open(f"{out_prefix}.fam"):
        parts = line.split()
        fam[parts[1]] = (parts[2], parts[3])
    assert fam['CHILD'] == ('OLD_DAD', 'OLD_MOM'), \
        "Mendelian veto should force --auto-resolve to keep base despite kinship favoring new"
