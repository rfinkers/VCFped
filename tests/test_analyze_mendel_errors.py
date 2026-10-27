"""
Tests for utilities/analyze_mendel_errors.py.

Covers turning PLINK2 --mendel's raw per-family error COUNTS (.fmendel) into
rates and flagging outliers two ways (cohort-relative percentile, and
absolute relative to the error-rate-derived theoretical baseline), since a
single fixed threshold is unreliable for cohorts with an elevated
genotyping error rate (same population-relative philosophy as
refine_pedigree.py's KING kinship percentiles).
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

AME_SCRIPT = os.path.join(UTILITIES_DIR, 'analyze_mendel_errors.py')


@pytest.fixture
def ame():
    return import_module_from_path('analyze_mendel_errors', AME_SCRIPT)


# ---------------------------------------------------------------------------
# load_fmendel
# ---------------------------------------------------------------------------

def test_load_fmendel_parses_rows_and_skips_header(ame, tmp_path):
    f = tmp_path / "test.fmendel"
    f.write_text(
        "#FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tDAD1\tMOM1\t1\t17\n"
        "F2\tDAD2\tMOM2\t2\t586\n"
    )
    rows = ame.load_fmendel(str(f))
    assert len(rows) == 2
    assert rows[0] == {'fid': 'F1', 'pat': 'DAD1', 'mat': 'MOM1', 'child': 1, 'n': 17}
    assert rows[1]['child'] == 2
    assert rows[1]['n'] == 586


def test_load_fmendel_missing_file_errors(ame, tmp_path):
    with pytest.raises(SystemExit):
        ame.load_fmendel(str(tmp_path / "nope.fmendel"))


def test_load_fmendel_empty_file_errors(ame, tmp_path):
    f = tmp_path / "empty.fmendel"
    f.write_text("#FID\tPAT\tMAT\tCHLD\tN\n")
    with pytest.raises(SystemExit):
        ame.load_fmendel(str(f))


# ---------------------------------------------------------------------------
# count_variants
# ---------------------------------------------------------------------------

def test_count_variants(ame, tmp_path):
    f = tmp_path / "test.bim"
    f.write_text("1\tvar1\t0\t100\tA\tT\n1\tvar2\t0\t200\tG\tC\n")
    assert ame.count_variants(str(f)) == 2


# ---------------------------------------------------------------------------
# compute_rates
# ---------------------------------------------------------------------------

def test_compute_rates(ame):
    rows = [
        {'fid': 'F1', 'pat': 'A', 'mat': 'B', 'child': 1, 'n': 100},
        {'fid': 'F2', 'pat': 'C', 'mat': 'D', 'child': 2, 'n': 200},
    ]
    ame.compute_rates(rows, n_variants=1000)
    assert rows[0]['rate'] == pytest.approx(0.1)   # 100 / (1000*1)
    assert rows[1]['rate'] == pytest.approx(0.1)   # 200 / (1000*2)


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------

def test_compute_rate_percentiles(ame):
    rows = [{'rate': r} for r in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]]
    pct = ame.compute_rate_percentiles(rows)
    sorted_rates = sorted(r['rate'] for r in rows)
    n = len(sorted_rates)
    assert pct[50] == sorted_rates[int(0.5 * n)]
    assert pct[95] == sorted_rates[int(0.95 * n)]


def test_compute_percentile_ranks_orders_correctly(ame):
    rows = [{'rate': 0.05}, {'rate': 0.01}, {'rate': 0.09}]
    ranks = ame.compute_percentile_ranks(rows)
    # lowest rate -> rank 0, highest -> rank 100
    assert ranks[1] == 0     # rate=0.01, the lowest
    assert ranks[2] == 100   # rate=0.09, the highest
    assert ranks[0] not in (0, 100)  # middle value


# ---------------------------------------------------------------------------
# expected_baseline_rate
# ---------------------------------------------------------------------------

def test_expected_baseline_rate_matches_documented_examples(ame):
    """Same '~2x' relationship documented in prepare_plink_beagle.py's
    generated _run_plink.sh: 0.1%->0.2%, 0.5%->1.0%, 1.0%->2.0%."""
    assert ame.expected_baseline_rate(0.001) == pytest.approx(0.002)
    assert ame.expected_baseline_rate(0.005) == pytest.approx(0.010)
    assert ame.expected_baseline_rate(0.010) == pytest.approx(0.020)
    assert ame.expected_baseline_rate(0.025) == pytest.approx(0.050)


# ---------------------------------------------------------------------------
# flag_outliers
# ---------------------------------------------------------------------------

def test_flag_outliers_cohort_only(ame):
    rows = [{'rate': 0.03}]
    ame.flag_outliers(rows, percentile_threshold=0.02, absolute_threshold=0.10)
    assert rows[0]['flagged'] is True
    assert rows[0]['flag_reason'] == 'cohort_outlier'


def test_flag_outliers_absolute_only(ame):
    rows = [{'rate': 0.15}]
    ame.flag_outliers(rows, percentile_threshold=0.20, absolute_threshold=0.10)
    assert rows[0]['flagged'] is True
    assert rows[0]['flag_reason'] == 'absolute_high'


def test_flag_outliers_both(ame):
    rows = [{'rate': 0.30}]
    ame.flag_outliers(rows, percentile_threshold=0.05, absolute_threshold=0.10)
    assert rows[0]['flagged'] is True
    assert rows[0]['flag_reason'] == 'cohort_outlier+absolute_high'


def test_flag_outliers_neither(ame):
    rows = [{'rate': 0.01}]
    ame.flag_outliers(rows, percentile_threshold=0.05, absolute_threshold=0.10)
    assert rows[0]['flagged'] is False
    assert rows[0]['flag_reason'] == ''


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_end_to_end_cli_with_bim(tmp_path):
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text(
        "#FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tDAD1\tMOM1\t1\t20\n"      # rate = 20/1000 = 2.0%  (near baseline)
        "F2\tDAD2\tMOM2\t1\t900\n"     # rate = 900/1000 = 90%  (extreme outlier)
    )
    bim = tmp_path / "test.bim"
    bim.write_text('\n'.join(f"1\tvar{i}\t0\t{i*100}\tA\tT" for i in range(1000)) + '\n')

    out_prefix = str(tmp_path / "out")
    result = run_script(AME_SCRIPT, [str(fmendel), '--bim', str(bim), '-o', out_prefix])
    assert result.returncode == 0, result.stderr

    assert os.path.exists(f"{out_prefix}_analysis.tsv")
    assert os.path.exists(f"{out_prefix}_report.txt")

    tsv_lines = open(f"{out_prefix}_analysis.tsv").read().splitlines()
    header = tsv_lines[0].split('\t')
    rows = [dict(zip(header, line.split('\t'))) for line in tsv_lines[1:]]
    by_fid = {r['FID']: r for r in rows}

    assert by_fid['F1']['Flagged'] == 'False'
    assert by_fid['F2']['Flagged'] == 'True'

    report = open(f"{out_prefix}_report.txt").read()
    assert 'F2' in report
    assert 'DAD2' in report


def test_bim_and_n_variants_mutually_exclusive(tmp_path):
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text("#FID\tPAT\tMAT\tCHLD\tN\nF1\tA\tB\t1\t10\n")
    bim = tmp_path / "test.bim"
    bim.write_text("1\tvar1\t0\t100\tA\tT\n")

    result = run_script(AME_SCRIPT, [str(fmendel), '--bim', str(bim), '--n-variants', '100'])
    assert result.returncode != 0


def test_neither_bim_nor_n_variants_errors(tmp_path):
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text("#FID\tPAT\tMAT\tCHLD\tN\nF1\tA\tB\t1\t10\n")
    result = run_script(AME_SCRIPT, [str(fmendel)])
    assert result.returncode != 0


def test_end_to_end_cli_with_n_variants(tmp_path):
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text(
        "#FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tDAD1\tMOM1\t1\t10\n"
    )
    out_prefix = str(tmp_path / "out")
    result = run_script(AME_SCRIPT, [str(fmendel), '--n-variants', '500', '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{out_prefix}_analysis.tsv")


def test_default_output_prefix_derived_from_fmendel_path(tmp_path):
    fmendel = tmp_path / "revalidate_mendel.fmendel"
    fmendel.write_text("#FID\tPAT\tMAT\tCHLD\tN\nF1\tA\tB\t1\t10\n")
    result = run_script(AME_SCRIPT, [str(fmendel), '--n-variants', '500'], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert os.path.exists(str(tmp_path / "revalidate_mendel_mendel_analysis.tsv"))


# ---------------------------------------------------------------------------
# load_imendel
# ---------------------------------------------------------------------------

def test_load_imendel_parses_rows(ame, tmp_path):
    f = tmp_path / "test.imendel"
    f.write_text(
        "#FID\tIID\tN\n"
        "F1\tCHILD1\t50\n"
        "F1\tCHILD2\t900\n"
    )
    errors = ame.load_imendel(str(f))
    assert errors == {'CHILD1': 50, 'CHILD2': 900}


def test_load_imendel_missing_or_none_returns_empty(ame, tmp_path):
    assert ame.load_imendel(None) == {}
    assert ame.load_imendel(str(tmp_path / "nope.imendel")) == {}


# ---------------------------------------------------------------------------
# aggregate_parent_errors: the repeat-offender detection this whole feature
# was built for
# ---------------------------------------------------------------------------

def test_aggregate_parent_errors_sums_across_all_pairings(ame):
    """
    Reproduces the real pattern that motivated this feature: one individual
    (REPEAT) appears as MAT across several DIFFERENT PAT parents, each family
    otherwise unrelated. aggregate_parent_errors() must sum N/CHILD across
    all of REPEAT's appearances and record each distinct co-parent.
    """
    rows = [
        {'fid': 'F1', 'pat': 'DAD_A', 'mat': 'REPEAT', 'child': 1, 'n': 800},
        {'fid': 'F1', 'pat': 'DAD_B', 'mat': 'REPEAT', 'child': 1, 'n': 750},
        {'fid': 'F1', 'pat': 'DAD_C', 'mat': 'REPEAT', 'child': 2, 'n': 1500},
        {'fid': 'F1', 'pat': 'DAD_D', 'mat': 'CLEAN_MOM', 'child': 1, 'n': 20},
    ]
    agg = ame.aggregate_parent_errors(rows)
    by_ind = {r['individual']: r for r in agg}

    repeat = by_ind['REPEAT']
    assert repeat['n_families'] == 3
    assert repeat['n_co_parents'] == 3
    assert set(repeat['co_parents']) == {'DAD_A', 'DAD_B', 'DAD_C'}
    assert repeat['total_n'] == 800 + 750 + 1500
    assert repeat['total_child'] == 1 + 1 + 2

    # DAD_D only appears once, paired with CLEAN_MOM, not REPEAT
    assert by_ind['DAD_D']['n_co_parents'] == 1
    assert by_ind['DAD_D']['co_parents'] == ['CLEAN_MOM']

    # Every PAT/MAT identity gets its own aggregate row
    assert set(by_ind) == {'DAD_A', 'DAD_B', 'DAD_C', 'DAD_D', 'REPEAT', 'CLEAN_MOM'}


def test_compute_individual_rates(ame):
    agg_rows = [{'individual': 'X', 'total_n': 100, 'total_child': 2}]
    ame.compute_individual_rates(agg_rows, n_variants=1000)
    assert agg_rows[0]['rate'] == pytest.approx(0.05)  # 100 / (1000*2)


# ---------------------------------------------------------------------------
# End-to-end: repeat-offender parent flagged, --min-co-parents respected
# ---------------------------------------------------------------------------

def test_end_to_end_flags_repeat_offender_parent(tmp_path):
    """
    REPEAT_PARENT pairs with 3 different, otherwise-unrelated co-parents,
    all showing a clearly elevated rate -- must appear in the
    individual_analysis.tsv as flagged with n_co_parents=3, and in the
    report's "Repeat-offender parents" section. A single-appearance
    individual with an equally high rate must NOT appear there (fewer than
    --min-co-parents), even though it IS flagged at the family level.
    """
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text(
        "#FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tDAD_A\tREPEAT_PARENT\t1\t900\n"
        "F1\tDAD_B\tREPEAT_PARENT\t1\t880\n"
        "F1\tDAD_C\tREPEAT_PARENT\t1\t910\n"
        "F1\tDAD_D\tSINGLE_HIGH\t1\t950\n"
        "F1\tDAD_E\tCLEAN_MOM1\t1\t20\n"
        "F1\tDAD_F\tCLEAN_MOM2\t1\t18\n"
        "F1\tDAD_G\tCLEAN_MOM3\t1\t22\n"
        "F1\tDAD_H\tCLEAN_MOM4\t1\t19\n"
    )
    out_prefix = str(tmp_path / "out")
    result = run_script(AME_SCRIPT, [str(fmendel), '--n-variants', '1000', '-o', out_prefix,
                                     '--min-co-parents', '2'])
    assert result.returncode == 0, result.stderr

    ind_tsv = f"{out_prefix}_individual_analysis.tsv"
    assert os.path.exists(ind_tsv)
    lines = open(ind_tsv).read().splitlines()
    header = lines[0].split('\t')
    by_ind = {row[0]: dict(zip(header, row)) for row in (l.split('\t') for l in lines[1:])}

    assert by_ind['REPEAT_PARENT']['N_CoParents'] == '3'
    assert by_ind['REPEAT_PARENT']['Flagged'] == 'True'
    assert by_ind['SINGLE_HIGH']['N_CoParents'] == '1'

    report = open(f"{out_prefix}_report.txt").read()
    assert 'Repeat-offender parents' in report
    assert 'REPEAT_PARENT' in report
    # SINGLE_HIGH has only one co-parent -- excluded from repeat-offender
    # section by --min-co-parents, even though individually flagged.
    repeat_section = report.split('Repeat-offender parents')[1]
    assert 'SINGLE_HIGH' not in repeat_section


def test_imendel_cross_reference_appears_in_individual_tsv(tmp_path):
    fmendel = tmp_path / "test.fmendel"
    fmendel.write_text(
        "#FID\tPAT\tMAT\tCHLD\tN\n"
        "F1\tDAD1\tMOM1\t1\t20\n"
    )
    imendel = tmp_path / "test.imendel"
    imendel.write_text("#FID\tIID\tN\nF1\tDAD1\t123\n")

    out_prefix = str(tmp_path / "out")
    result = run_script(AME_SCRIPT, [str(fmendel), '--n-variants', '1000', '-o', out_prefix,
                                     '--imendel', str(imendel)])
    assert result.returncode == 0, result.stderr

    lines = open(f"{out_prefix}_individual_analysis.tsv").read().splitlines()
    header = lines[0].split('\t')
    rows = [dict(zip(header, l.split('\t'))) for l in lines[1:]]
    by_ind = {r['Individual']: r for r in rows}

    assert by_ind['DAD1']['IMendel_N'] == '123'
    assert by_ind['MOM1']['IMendel_N'] == 'NA'  # not in the .imendel file
