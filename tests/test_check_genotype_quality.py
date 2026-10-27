"""
Tests for utilities/check_genotype_quality.py.

Covers parsing PLINK2 .het/.smiss output (name-based column lookup, robust
to minor PLINK2 version differences), population-relative outlier flagging
(rank-based, same approach as analyze_hbd.py), and cross-referencing
analyze_mendel_errors.py's repeat-offender individuals against genotyping
quality -- ruling out (or in) poor genotyping quality as a confound before
concluding a repeat offender reflects a real pedigree error.
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

CGQ_SCRIPT = os.path.join(UTILITIES_DIR, 'check_genotype_quality.py')


@pytest.fixture
def cgq():
    return import_module_from_path('check_genotype_quality', CGQ_SCRIPT)


# ---------------------------------------------------------------------------
# read_plink_table / load_het / load_missing
# ---------------------------------------------------------------------------

def test_read_plink_table_strips_leading_hash(cgq, tmp_path):
    f = tmp_path / "test.het"
    f.write_text("#IID\tO(HOM)\tE(HOM)\tOBS_CT\tF\nS1\t900\t800\t1000\t0.5\n")
    rows = cgq.read_plink_table(str(f))
    assert rows == [{'IID': 'S1', 'O(HOM)': '900', 'E(HOM)': '800', 'OBS_CT': '1000', 'F': '0.5'}]


def test_load_het_parses_f(cgq, tmp_path):
    f = tmp_path / "test.het"
    f.write_text("#IID\tO(HOM)\tE(HOM)\tOBS_CT\tF\nS1\t900\t800\t1000\t0.5\n")
    het = cgq.load_het(str(f))
    assert het['S1']['f'] == pytest.approx(0.5)
    assert het['S1']['o_hom'] == pytest.approx(900)
    assert het['S1']['e_hom'] == pytest.approx(800)


def test_load_het_missing_file_exits(cgq, tmp_path):
    with pytest.raises(SystemExit):
        cgq.load_het(str(tmp_path / "nope.het"))


def test_load_het_missing_columns_exits(cgq, tmp_path):
    f = tmp_path / "bad.het"
    f.write_text("#IID\tSOMETHING\nS1\t1\n")
    with pytest.raises(SystemExit):
        cgq.load_het(str(f))


def test_load_missing_parses_fmiss(cgq, tmp_path):
    f = tmp_path / "test.smiss"
    f.write_text("#IID\tMISSING_CT\tOBS_CT\tF_MISS\nS1\t10\t1000\t0.01\n")
    missing = cgq.load_missing(str(f))
    assert missing['S1'] == pytest.approx(0.01)


def test_load_missing_missing_file_exits(cgq, tmp_path):
    with pytest.raises(SystemExit):
        cgq.load_missing(str(tmp_path / "nope.smiss"))


# ---------------------------------------------------------------------------
# percentile rank / outlier flagging
# ---------------------------------------------------------------------------

def test_percentile_rank_orders_low_to_high(cgq):
    values = {'LOW': 0.1, 'MID': 0.5, 'HIGH': 0.9}
    ranks = cgq._percentile_rank(values)
    assert ranks['LOW'] < ranks['MID'] < ranks['HIGH']
    assert ranks['LOW'] == 0.0
    assert ranks['HIGH'] == 100.0


def test_flag_het_outliers_flags_both_directions(cgq):
    het = {f'M{i}': {'f': 0.5} for i in range(18)}
    het['LOW'] = {'f': -0.9}
    het['HIGH'] = {'f': 0.99}
    ranks = cgq._percentile_rank({iid: d['f'] for iid, d in het.items()})

    flags = cgq.flag_het_outliers(het, ranks, low_percentile=5, high_percentile=95)
    assert flags['LOW'][0] is True
    assert 'LOW_F' in flags['LOW'][1]
    assert flags['HIGH'][0] is True
    assert 'HIGH_F' in flags['HIGH'][1]
    assert flags['M0'][0] is False


def test_flag_missing_outliers_only_flags_high(cgq):
    missing = {f'M{i}': 0.01 for i in range(18)}
    missing['HIGH'] = 0.5
    ranks = cgq._percentile_rank(missing)

    flags = cgq.flag_missing_outliers(missing, ranks, outlier_percentile=95)
    assert flags['HIGH'][0] is True
    assert 'HIGH_MISSINGNESS' in flags['HIGH'][1]
    assert flags['M0'][0] is False
    # No such thing as a "too low" missingness flag
    assert all(not f for iid, (f, _) in flags.items() if iid != 'HIGH')


# ---------------------------------------------------------------------------
# Mendelian repeat-offender cross-reference
# ---------------------------------------------------------------------------

def test_load_mendel_repeat_offenders_filters_flagged(cgq, tmp_path):
    f = tmp_path / "individual_analysis.tsv"
    f.write_text(
        "Individual\tN_Families\tN_CoParents\tTotal_N\tTotal_Child\tRate\t"
        "Percentile_Rank\tFlagged\tFlag_Reason\tIMendel_N\n"
        "S1\t3\t3\t10\t20\t0.05\t99\tTrue\tcohort_outlier\tNA\n"
        "S2\t2\t2\t2\t20\t0.01\t20\tFalse\t\tNA\n"
    )
    offenders = cgq.load_mendel_repeat_offenders(str(f))
    assert offenders == {'S1'}


def test_load_mendel_repeat_offenders_missing_returns_empty(cgq, tmp_path):
    assert cgq.load_mendel_repeat_offenders(None) == set()
    assert cgq.load_mendel_repeat_offenders(str(tmp_path / "nope.tsv")) == set()


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def _write_het_smiss(tmp_path, extra_het_f=None, extra_miss=None):
    het_lines = ["#IID\tO(HOM)\tE(HOM)\tOBS_CT\tF"]
    smiss_lines = ["#IID\tMISSING_CT\tOBS_CT\tF_MISS"]
    for i in range(18):
        het_lines.append(f"S{i}\t900\t800\t1000\t0.5")
        smiss_lines.append(f"S{i}\t10\t1000\t0.01")
    if extra_het_f:
        for iid, f_val in extra_het_f.items():
            het_lines.append(f"{iid}\t900\t800\t1000\t{f_val}")
    if extra_miss:
        for iid, f_val in extra_miss.items():
            smiss_lines.append(f"{iid}\t500\t1000\t{f_val}")
    het_path = tmp_path / "test.het"
    smiss_path = tmp_path / "test.smiss"
    het_path.write_text("\n".join(het_lines) + "\n")
    smiss_path.write_text("\n".join(smiss_lines) + "\n")
    return het_path, smiss_path


def test_cli_end_to_end(tmp_path):
    het_path, smiss_path = _write_het_smiss(
        tmp_path, extra_het_f={'LOWF': -0.9, 'HIGHF': 0.99}, extra_miss={'HIGHMISS': 0.5})

    out_prefix = tmp_path / "qc_out"
    result = run_script(CGQ_SCRIPT, [str(het_path), str(smiss_path), '-o', str(out_prefix)])
    assert result.returncode == 0, result.stderr

    tsv = tmp_path / "qc_out_genotype_quality.tsv"
    report = tmp_path / "qc_out_genotype_quality_report.txt"
    assert tsv.exists()
    assert report.exists()

    content = tsv.read_text()
    assert 'LOWF' in content
    assert 'HIGHF' in content
    assert 'HIGHMISS' in content

    report_content = report.read_text()
    assert 'Heterozygosity-flagged individuals' in report_content
    assert 'Missingness-flagged individuals' in report_content


def test_cli_default_output_prefix_strips_het_suffix(tmp_path):
    het_path, smiss_path = _write_het_smiss(tmp_path)
    result = run_script(CGQ_SCRIPT, [str(het_path), str(smiss_path)], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_genotype_quality.tsv").exists()


def test_cli_mendel_cross_reference(tmp_path):
    het_path, smiss_path = _write_het_smiss(
        tmp_path, extra_het_f={'CONFOUNDED': -0.9}, extra_miss=None)

    mendel_tsv = tmp_path / "individual_analysis.tsv"
    mendel_tsv.write_text(
        "Individual\tN_Families\tN_CoParents\tTotal_N\tTotal_Child\tRate\t"
        "Percentile_Rank\tFlagged\tFlag_Reason\tIMendel_N\n"
        "CONFOUNDED\t3\t3\t10\t20\t0.05\t99\tTrue\tcohort_outlier\tNA\n"
        "S0\t2\t2\t2\t20\t0.01\t20\tTrue\tcohort_outlier\tNA\n"
    )

    out_prefix = tmp_path / "qc_out"
    result = run_script(CGQ_SCRIPT, [
        str(het_path), str(smiss_path), '-o', str(out_prefix),
        '--mendel-individual-tsv', str(mendel_tsv),
    ])
    assert result.returncode == 0, result.stderr

    report_content = (tmp_path / "qc_out_genotype_quality_report.txt").read_text()
    assert 'Mendelian repeat-offender cross-reference' in report_content
    assert 'With a genotyping-quality confound' in report_content
    assert 'CONFOUNDED' in report_content
    assert 'No genotyping-quality confound' in report_content
    assert 'S0' in report_content
