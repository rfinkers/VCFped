"""
Tests for utilities/fam_to_trio.py.

Covers converting a .fam file's confirmed (PAT and MAT both known)
parent-child links into a standard vcfped .trio file, so tools that only
read .trio (fixed_pedigree_viz.py, refine_pedigree.py's --ibd final
confirmation) can consume a pedigree that now only exists as a merged .fam --
carrying over REAL vcfped statistics when the current (p1,p2,child) matches
a row in one of the original *_refined.trio files, and falling back to a
KING kinship-based confidence proxy (never a fake flat placeholder) for rows
where refine_pedigree.py/merge_refined_fam.py substituted a different pair.
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR, requires_pandas, requires_plotly

FAM_TO_TRIO_SCRIPT = os.path.join(UTILITIES_DIR, 'fam_to_trio.py')
FPV_SCRIPT = os.path.join(UTILITIES_DIR, 'fixed_pedigree_viz.py')


@pytest.fixture
def f2t():
    return import_module_from_path('fam_to_trio', FAM_TO_TRIO_SCRIPT)


# ---------------------------------------------------------------------------
# fam_to_pairs
# ---------------------------------------------------------------------------

def test_fam_to_pairs_only_confirmed_rows(f2t, tmp_path):
    f = tmp_path / "test.fam"
    f.write_text(
        "0\tCHILD\tDAD\tMOM\t0\t-9\n"
        "0\tFOUNDER\t0\t0\t1\t-9\n"
        "0\tHALF\tDAD\t0\t0\t-9\n"
    )
    pairs = f2t.fam_to_pairs(str(f))
    assert pairs == [('DAD', 'MOM', 'CHILD')]


def test_fam_to_pairs_missing_file_exits(f2t, tmp_path):
    with pytest.raises(SystemExit):
        f2t.fam_to_pairs(str(tmp_path / "nope.fam"))


# ---------------------------------------------------------------------------
# load_trio_lookup
# ---------------------------------------------------------------------------

def test_load_trio_lookup_parses_regular_trio_rows(f2t, tmp_path):
    f = tmp_path / "source.trio"
    f.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "DAD,MOM,CHILD\tCHILD\t30\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
        "X,Y,Z\tZ\t10\t100\t50\t50\t95.0\t20\t18\t90.0\tNot trio\n"
    )
    lookup = f2t.load_trio_lookup([str(f)])
    key = ('CHILD', frozenset(['DAD', 'MOM']))
    assert key in lookup
    assert lookup[key]['perc'] == '30'
    assert lookup[key]['test1'] == '100.0'
    # 'Not trio' row must be excluded
    assert ('Z', frozenset(['X', 'Y'])) not in lookup


def test_load_trio_lookup_first_match_wins_across_sources(f2t, tmp_path):
    f1 = tmp_path / "a.trio"
    f1.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "DAD,MOM,CHILD\tCHILD\t30\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    f2 = tmp_path / "b.trio"
    f2.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "DAD,MOM,CHILD\tCHILD\t50\t1000\t50\t50\t80.0\t10\t8\t80.0\tRegular trio\n"
    )
    lookup = f2t.load_trio_lookup([str(f1), str(f2)])
    key = ('CHILD', frozenset(['DAD', 'MOM']))
    assert lookup[key]['perc'] == '30'  # from f1, not overwritten by f2


def test_load_trio_lookup_skips_missing_files(f2t, tmp_path):
    assert f2t.load_trio_lookup([str(tmp_path / "nope.trio")]) == {}
    assert f2t.load_trio_lookup([]) == {}


# ---------------------------------------------------------------------------
# load_king_lookup / get_king / kinship_confidence
# ---------------------------------------------------------------------------

def test_load_king_lookup_parses_kin0(f2t, tmp_path):
    f = tmp_path / "test_king.kin0"
    f.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tDAD\t1000\t0.3\t0.001\t0.26\n"
    )
    king = f2t.load_king_lookup(str(tmp_path / "test_king"))
    assert f2t.get_king(king, 'CHILD', 'DAD') == {'kinship': 0.26}
    assert f2t.get_king(king, 'DAD', 'CHILD') == {'kinship': 0.26}  # order-independent


def test_load_king_lookup_missing_returns_empty(f2t, tmp_path):
    assert f2t.load_king_lookup(None) == {}
    assert f2t.load_king_lookup(str(tmp_path / "nope")) == {}


def test_kinship_confidence_averages_both_parents(f2t):
    king = {
        ('CHILD', 'DAD'): {'kinship': 0.30},
        ('CHILD', 'MOM'): {'kinship': 0.20},
    }
    conf = f2t.kinship_confidence(king, 'DAD', 'MOM', 'CHILD')
    assert conf == pytest.approx(25.0)  # (0.30+0.20)/2 * 100


def test_kinship_confidence_no_data_returns_zero(f2t):
    assert f2t.kinship_confidence({}, 'DAD', 'MOM', 'CHILD') == 0.0


def test_kinship_confidence_partial_data(f2t):
    king = {('CHILD', 'DAD'): {'kinship': 0.30}}
    conf = f2t.kinship_confidence(king, 'DAD', 'MOM', 'CHILD')
    assert conf == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# build_trio_rows
# ---------------------------------------------------------------------------

def test_build_trio_rows_uses_real_stats_when_matched(f2t):
    trio_lookup = {
        ('CHILD', frozenset(['DAD', 'MOM'])): {
            'perc': '30', 'autos': '1905', 'aa_bb': '136', 'ab': '136',
            'test1': '100.0', 'bb_bb': '269', 'bb': '267', 'test2': '99.3',
        }
    }
    rows = f2t.build_trio_rows([('DAD', 'MOM', 'CHILD')], trio_lookup, {})
    assert rows[0]['perc'] == '30'
    assert rows[0]['test1'] == '100.0'


def test_build_trio_rows_falls_back_to_kinship_when_unmatched(f2t):
    king = {('CHILD', 'DAD'): {'kinship': 0.30}, ('CHILD', 'MOM'): {'kinship': 0.20}}
    rows = f2t.build_trio_rows([('DAD', 'MOM', 'CHILD')], {}, king)
    assert rows[0]['perc'] == '0'  # sentinel: no real vcfped test
    assert rows[0]['test1'] == '25.0'
    assert rows[0]['autos'] == '0'


def test_build_trio_rows_no_king_no_match_gives_zero_confidence(f2t):
    rows = f2t.build_trio_rows([('DAD', 'MOM', 'CHILD')], {}, {})
    assert rows[0]['perc'] == '0'
    assert rows[0]['test1'] == '0.0'


# ---------------------------------------------------------------------------
# write_trio_file
# ---------------------------------------------------------------------------

def test_write_trio_file_format(f2t, tmp_path):
    out = tmp_path / "out.trio"
    rows = f2t.build_trio_rows([('DAD', 'MOM', 'CHILD')], {}, {})
    f2t.write_trio_file(str(out), rows)
    lines = out.read_text().splitlines()
    assert lines[0].split('\t') == f2t.TRIO_HEADER
    row = dict(zip(f2t.TRIO_HEADER, lines[1].split('\t')))
    assert row['Triple'] == 'DAD,MOM,CHILD'
    assert row['Pivot'] == 'CHILD'
    assert row['Verdict'] == 'Regular trio'
    # Raw counts must be int-parseable (0), never 'NA' -- fixed_pedigree_viz.py
    # int()-converts these columns directly and pandas reads 'NA' as NaN.
    int(row['Autos'])
    int(row['AA+BB'])


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_cli_end_to_end_no_sources(tmp_path):
    fam = tmp_path / "merged.fam"
    fam.write_text(
        "0\tCHILD1\tDAD1\tMOM1\t0\t-9\n"
        "0\tCHILD2\tDAD2\tMOM2\t0\t-9\n"
        "0\tFOUNDER\t0\t0\t1\t-9\n"
    )
    out_prefix = tmp_path / "converted"
    result = run_script(FAM_TO_TRIO_SCRIPT, [str(fam), '-o', str(out_prefix)])
    assert result.returncode == 0, result.stderr

    trio_path = tmp_path / "converted.trio"
    assert trio_path.exists()
    content = trio_path.read_text()
    assert 'DAD1,MOM1,CHILD1' in content
    assert 'DAD2,MOM2,CHILD2' in content
    assert 'FOUNDER' not in content


def test_cli_end_to_end_with_trio_and_king_sources(tmp_path):
    fam = tmp_path / "merged.fam"
    fam.write_text(
        "0\tMATCHED\tDAD1\tMOM1\t0\t-9\n"
        "0\tSUBSTITUTED\tDAD2\tMOM2\t0\t-9\n"
    )
    source_trio = tmp_path / "source.trio"
    source_trio.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "DAD1,MOM1,MATCHED\tMATCHED\t30\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    king_file = tmp_path / "king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tSUBSTITUTED\tF1\tDAD2\t1000\t0.3\t0.001\t0.30\n"
        "F1\tSUBSTITUTED\tF1\tMOM2\t1000\t0.3\t0.001\t0.20\n"
    )
    out_prefix = tmp_path / "converted"
    result = run_script(FAM_TO_TRIO_SCRIPT, [
        str(fam), '--trio', str(source_trio), '--king', str(tmp_path / "king"),
        '-o', str(out_prefix),
    ])
    assert result.returncode == 0, result.stderr
    assert '1 real vcfped trio rows loaded' in result.stdout
    assert '1 matched real vcfped statistics' in result.stdout
    assert '1 KING-confirmed only' in result.stdout

    content = (tmp_path / "converted.trio").read_text()
    lines = {l.split('\t')[0]: l for l in content.splitlines()[1:]}
    matched_row = lines['DAD1,MOM1,MATCHED'].split('\t')
    assert matched_row[2] == '30'      # real Perc
    assert matched_row[6] == '100.0'   # real Test1
    substituted_row = lines['DAD2,MOM2,SUBSTITUTED'].split('\t')
    assert substituted_row[2] == '0'   # sentinel Perc
    assert substituted_row[6] == '25.0'  # kinship-derived Test1


def test_cli_default_output_prefix_strips_fam_suffix(tmp_path):
    fam = tmp_path / "merged.fam"
    fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    result = run_script(FAM_TO_TRIO_SCRIPT, [str(fam)], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "merged.trio").exists()


def test_cli_no_confirmed_trios_exits_nonzero(tmp_path):
    fam = tmp_path / "founders_only.fam"
    fam.write_text("0\tFOUNDER\t0\t0\t1\t-9\n")
    result = run_script(FAM_TO_TRIO_SCRIPT, [str(fam)])
    assert result.returncode != 0
    assert 'no confirmed' in (result.stdout + result.stderr)


def test_output_compatible_with_refine_pedigree_load_trio_file(tmp_path):
    """The written .trio file must satisfy refine_pedigree.py's
    REQUIRED_TRIO_COLS and load_trio_file()'s parsing logic."""
    import importlib.util
    rp_path = os.path.join(UTILITIES_DIR, 'refine_pedigree.py')
    spec = importlib.util.spec_from_file_location('refine_pedigree', rp_path)
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    fam = tmp_path / "merged.fam"
    fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    out_prefix = tmp_path / "converted"
    result = run_script(FAM_TO_TRIO_SCRIPT, [str(fam), '-o', str(out_prefix)])
    assert result.returncode == 0, result.stderr

    fieldnames, all_rows, filtered = rp.load_trio_file(str(tmp_path / "converted.trio"))
    assert len(filtered) == 1
    assert filtered[0]['_child'] == 'CHILD'
    assert {filtered[0]['_p1'], filtered[0]['_p2']} == {'DAD', 'MOM'}


@requires_pandas
@requires_plotly
def test_output_compatible_with_fixed_pedigree_viz(tmp_path):
    """
    Regression guard: fixed_pedigree_viz.py int()-converts the Autos/AA+BB/
    =AB/BB+BB/=BB columns directly (unlike refine_pedigree.py, which only
    reads Perc/Test1/Verdict) -- if fam_to_trio.py ever writes 'NA' there
    again, pandas reads it as NaN and int(NaN) crashes the whole report
    generation. Exercise the real consumer end-to-end, not just column
    presence, so this class of bug can't slip through silently.
    """
    fam = tmp_path / "merged.fam"
    fam.write_text(
        "0\tCHILD\tDAD\tMOM\t0\t-9\n"
        "0\tDAD\t0\t0\t1\t-9\n"
        "0\tMOM\t0\t0\t2\t-9\n"
    )
    trio_prefix = tmp_path / "converted"
    result = run_script(FAM_TO_TRIO_SCRIPT, [str(fam), '-o', str(trio_prefix)])
    assert result.returncode == 0, result.stderr

    out_html = tmp_path / "report.html"
    result = run_script(FPV_SCRIPT, [str(trio_prefix), '-o', str(out_html), '--no-browser'])
    assert result.returncode == 0, result.stderr
    assert out_html.exists()
