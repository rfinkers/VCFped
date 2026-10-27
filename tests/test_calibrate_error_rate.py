"""
Tests for utilities/calibrate_error_rate.py.

Covers empirically deriving the genotyping error rate from vcfped's own
confirmed "MZ twins" pairwise verdicts (an independent, non-KING-based
signal) cross-referenced against KING IBS0 for the same pairs — replacing
the assumed --error-rate 0.025 default used elsewhere in the pipeline
(refine_pedigree.py, analyze_mendel_errors.py) with a measured value.
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

CALIBRATE_SCRIPT = os.path.join(UTILITIES_DIR, 'calibrate_error_rate.py')


@pytest.fixture
def cer():
    return import_module_from_path('calibrate_error_rate', CALIBRATE_SCRIPT)


# ---------------------------------------------------------------------------
# load_confirmed_mz_pairs
# ---------------------------------------------------------------------------

def test_load_confirmed_mz_pairs_filters_verdict(cer, tmp_path):
    f = tmp_path / "test.pair"
    f.write_text(
        "Pair\tVerdict\tSomeOtherCol\n"
        "A,B\tMZ twins\tx\n"
        "C,D\tUnrelated\tx\n"
        "E,F\tMZ twins\tx\n"
    )
    pairs = cer.load_confirmed_mz_pairs(str(tmp_path / "test"))
    assert pairs == [('A', 'B'), ('E', 'F')]


def test_load_confirmed_mz_pairs_accepts_explicit_extension(cer, tmp_path):
    f = tmp_path / "test.pair"
    f.write_text("Pair\tVerdict\nA,B\tMZ twins\n")
    pairs = cer.load_confirmed_mz_pairs(str(f))
    assert pairs == [('A', 'B')]


def test_load_confirmed_mz_pairs_missing_file_exits(cer, tmp_path):
    with pytest.raises(SystemExit):
        cer.load_confirmed_mz_pairs(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# load_king_ibs0 / get_ibs0
# ---------------------------------------------------------------------------

def test_load_king_ibs0_parses_kin0(cer, tmp_path):
    f = tmp_path / "test_king.kin0"
    f.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tA\tF1\tB\t1000\t0.3\t0.004\t0.49\n"
    )
    king = cer.load_king_ibs0(str(tmp_path / "test_king"))
    assert cer.get_ibs0(king, 'A', 'B') == 0.004
    assert cer.get_ibs0(king, 'B', 'A') == 0.004  # order-independent
    assert cer.get_ibs0(king, 'A', 'C') is None


def test_load_king_ibs0_missing_file_exits(cer, tmp_path):
    with pytest.raises(SystemExit):
        cer.load_king_ibs0(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# collect_confirmed_duplicate_ibs0 / derive_error_rate
# ---------------------------------------------------------------------------

def test_collect_confirmed_duplicate_ibs0_skips_unmatched_pairs(cer):
    mz_pairs = [('A', 'B'), ('C', 'D')]
    king = {('A', 'B'): 0.006}
    values = cer.collect_confirmed_duplicate_ibs0(mz_pairs, king)
    assert values == [0.006]


def test_collect_confirmed_duplicate_ibs0_sorted(cer):
    mz_pairs = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    king = {('A', 'B'): 0.01, ('C', 'D'): 0.002, ('E', 'F'): 0.006}
    assert cer.collect_confirmed_duplicate_ibs0(mz_pairs, king) == [0.002, 0.006, 0.01]


def test_derive_error_rate_halves_median_ibs0(cer):
    assert cer.derive_error_rate(0.02) == pytest.approx(0.01)


def test_percentile_value(cer):
    values = [1, 2, 3, 4, 5]
    assert cer._percentile_value(values, 50) == 3
    assert cer._percentile_value(values, 0) == 1


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    pair_file = tmp_path / "step1.pair"
    pair_file.write_text(
        "Pair\tVerdict\n"
        "S1,S2\tMZ twins\n"
        "S3,S4\tMZ twins\n"
        "S5,S6\tUnrelated\n"
    )
    king_file = tmp_path / "step3_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tS1\tF1\tS2\t1000\t0.3\t0.004\t0.5\n"
        "F1\tS3\tF1\tS4\t1000\t0.3\t0.006\t0.5\n"
    )
    out_prefix = tmp_path / "calib"
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step1"), str(tmp_path / "step3_king"),
        '-o', str(out_prefix),
    ])
    assert result.returncode == 0, result.stderr
    assert "Derived error rate" in result.stdout

    report = out_prefix.parent / "calib_calibration_report.txt"
    assert report.exists()
    content = report.read_text()
    assert "Derived error rate" in content
    # median of [0.004, 0.006] = 0.006 (upper of the two, per _percentile_value's
    # int(p/100*n) indexing) -> error_rate = 0.003
    assert "0.003" in content


def test_cli_no_matching_king_pairs_exits_nonzero(tmp_path):
    pair_file = tmp_path / "step1.pair"
    pair_file.write_text("Pair\tVerdict\nS1,S2\tMZ twins\n")
    king_file = tmp_path / "step3_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tS9\tF1\tS10\t1000\t0.3\t0.004\t0.5\n"
    )
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step1"), str(tmp_path / "step3_king"),
    ])
    assert result.returncode != 0
    assert "no confirmed pairs found" in (result.stdout + result.stderr)


def test_cli_default_output_prefix(tmp_path):
    pair_file = tmp_path / "step1.pair"
    pair_file.write_text("Pair\tVerdict\nS1,S2\tMZ twins\n")
    king_file = tmp_path / "step3_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tS1\tF1\tS2\t1000\t0.3\t0.004\t0.5\n"
    )
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step1"), str(tmp_path / "step3_king"),
    ], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "step1_error_rate_calibration_report.txt").exists()


# ---------------------------------------------------------------------------
# --confirmed-duplicates (stricter, hap-IBD-coverage-validated ground truth)
# ---------------------------------------------------------------------------

def test_load_confirmed_duplicate_pairs_parses_id1_id2_header(cer, tmp_path):
    f = tmp_path / "confirmed.tsv"
    f.write_text("ID1\tID2\nS1\tS2\nS3\tS4\n")
    pairs = cer.load_confirmed_duplicate_pairs(str(f))
    assert pairs == [('S1', 'S2'), ('S3', 'S4')]


def test_load_confirmed_duplicate_pairs_missing_file_exits(cer, tmp_path):
    with pytest.raises(SystemExit):
        cer.load_confirmed_duplicate_pairs(str(tmp_path / "nope.tsv"))


def test_load_confirmed_duplicate_pairs_missing_columns_exits(cer, tmp_path):
    f = tmp_path / "bad.tsv"
    f.write_text("FOO\tBAR\nS1\tS2\n")
    with pytest.raises(SystemExit):
        cer.load_confirmed_duplicate_pairs(str(f))


def test_cli_requires_exactly_one_ground_truth_source(tmp_path):
    king_file = tmp_path / "king.kin0"
    king_file.write_text("#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n")

    # Neither given
    result = run_script(CALIBRATE_SCRIPT, [str(tmp_path / "king")])
    assert result.returncode != 0
    assert "specify exactly one" in (result.stdout + result.stderr)

    # Both given
    pair_file = tmp_path / "step1.pair"
    pair_file.write_text("Pair\tVerdict\n")
    confirmed_file = tmp_path / "confirmed.tsv"
    confirmed_file.write_text("ID1\tID2\n")
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step1"), str(tmp_path / "king"),
        '--confirmed-duplicates', str(confirmed_file),
    ])
    assert result.returncode != 0
    assert "specify exactly one" in (result.stdout + result.stderr)


def test_cli_confirmed_duplicates_end_to_end(tmp_path):
    """
    Same calibration math as the step1-pair path, but ground truth comes
    from a stricter hap-IBD-coverage-validated pair list (e.g.
    extend_pedigree.py's _confirmed_duplicates.tsv) instead of vcfped's raw
    pairwise "MZ twins" verdict, which inbreeding-induced allele-sharing
    inflation can contaminate with non-duplicate close relatives.
    """
    confirmed_file = tmp_path / "ext_out_confirmed_duplicates.tsv"
    confirmed_file.write_text(
        "ID1\tID2\n"
        "S1\tS2\n"
        "S3\tS4\n"
    )
    king_file = tmp_path / "step3_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tS1\tF1\tS2\t1000\t0.3\t0.004\t0.5\n"
        "F1\tS3\tF1\tS4\t1000\t0.3\t0.006\t0.5\n"
    )
    out_prefix = tmp_path / "calib_strict"
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step3_king"),
        '--confirmed-duplicates', str(confirmed_file),
        '-o', str(out_prefix),
    ])
    assert result.returncode == 0, result.stderr
    assert "Derived error rate" in result.stdout
    assert "hap-IBD-coverage validated" in result.stdout

    report = tmp_path / "calib_strict_calibration_report.txt"
    assert report.exists()
    content = report.read_text()
    assert "hap-IBD-coverage validated" in content
    # same median-halving math as the step1-pair path (median of
    # [0.004, 0.006] with n=2 -> index 1 -> 0.006 -> error_rate = 0.003)
    assert "0.003" in content


def test_cli_confirmed_duplicates_default_output_prefix(tmp_path):
    confirmed_file = tmp_path / "ext_out_confirmed_duplicates.tsv"
    confirmed_file.write_text("ID1\tID2\nS1\tS2\n")
    king_file = tmp_path / "step3_king.kin0"
    king_file.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tS1\tF1\tS2\t1000\t0.3\t0.004\t0.5\n"
    )
    result = run_script(CALIBRATE_SCRIPT, [
        str(tmp_path / "step3_king"), '--confirmed-duplicates', str(confirmed_file),
    ], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ext_out_confirmed_duplicates.tsv_error_rate_calibration_report.txt").exists()
