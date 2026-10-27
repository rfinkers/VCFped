"""
Tests for utilities/analyze_hbd.py.

Covers parsing hap-IBD .hbd.gz (self-IBD / autozygosity) output, per-individual
aggregation, population-relative percentile ranking and outlier flagging (low
autozygosity = inconsistent with a deep-selfed breeding history; high =
notably homozygous), and the --check-pairs side-by-side comparison used to
cross-check extend_pedigree.py's CLOSE_INBRED / CLOSE_INBRED_POSSIBLE_DUP
classification.
"""
import gzip
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

HBD_SCRIPT = os.path.join(UTILITIES_DIR, 'analyze_hbd.py')


@pytest.fixture
def ahbd():
    return import_module_from_path('analyze_hbd', HBD_SCRIPT)


def _write_hbd_gz(path, rows):
    """rows: list of (id1, id2, chrom, start, end, lod) tuples."""
    with gzip.open(path, 'wt') as f:
        for id1, id2, chrom, start, end, lod in rows:
            f.write(f"{id1}\t1\t{id2}\t2\t{chrom}\t{start}\t{end}\t{lod}\n")


# ---------------------------------------------------------------------------
# load_hbd
# ---------------------------------------------------------------------------

def test_load_hbd_groups_by_individual(ahbd, tmp_path):
    path = tmp_path / "test.hbd.gz"
    _write_hbd_gz(path, [
        ('A', 'A', '1', 0, 1_000_000, 5.0),
        ('A', 'A', '1', 2_000_000, 2_500_000, 4.0),
        ('B', 'B', '1', 0, 500_000, 6.0),
    ])
    by_indiv, mismatches = ahbd.load_hbd(str(path))
    assert by_indiv['A'] == [1_000_000, 500_000]
    assert by_indiv['B'] == [500_000]
    assert mismatches == 0


def test_load_hbd_filters_low_lod(ahbd, tmp_path):
    path = tmp_path / "test.hbd.gz"
    _write_hbd_gz(path, [
        ('A', 'A', '1', 0, 1_000_000, 2.0),  # below default min_lod=3.0
        ('A', 'A', '1', 2_000_000, 2_500_000, 4.0),
    ])
    by_indiv, _ = ahbd.load_hbd(str(path))
    assert by_indiv['A'] == [500_000]


def test_load_hbd_counts_id_mismatches(ahbd, tmp_path):
    path = tmp_path / "test.hbd.gz"
    _write_hbd_gz(path, [
        ('A', 'B', '1', 0, 1_000_000, 5.0),  # id1 != id2 -- unexpected for HBD
    ])
    by_indiv, mismatches = ahbd.load_hbd(str(path))
    assert mismatches == 1
    assert by_indiv['A'] == [1_000_000]


def test_load_hbd_missing_file_exits(ahbd, tmp_path):
    with pytest.raises(SystemExit):
        ahbd.load_hbd(str(tmp_path / "nope.hbd.gz"))


def test_load_hbd_accepts_plain_text(ahbd, tmp_path):
    path = tmp_path / "test.hbd"
    path.write_text("A\t1\tA\t2\t1\t0\t1000000\t5.0\n")
    by_indiv, _ = ahbd.load_hbd(str(path))
    assert by_indiv['A'] == [1_000_000]


# ---------------------------------------------------------------------------
# summarize_individual / build_individual_stats
# ---------------------------------------------------------------------------

def test_summarize_individual(ahbd):
    stats = ahbd.summarize_individual([1_000_000, 500_000, 1_500_000])
    assert stats['total_bp'] == 3_000_000
    assert stats['total_Mb'] == pytest.approx(3.0)
    assert stats['n_segments'] == 3
    assert stats['mean_segment_Mb'] == pytest.approx(1.0)


def test_summarize_individual_empty(ahbd):
    stats = ahbd.summarize_individual([])
    assert stats['total_bp'] == 0
    assert stats['n_segments'] == 0
    assert stats['mean_segment_Mb'] == 0.0


# ---------------------------------------------------------------------------
# percentile ranks / outlier flagging
# ---------------------------------------------------------------------------

def test_compute_percentile_ranks_orders_low_to_high(ahbd):
    stats = {
        'LOW': {'total_Mb': 1.0},
        'MID': {'total_Mb': 5.0},
        'HIGH': {'total_Mb': 10.0},
    }
    ranks = ahbd.compute_percentile_ranks(stats)
    assert ranks['LOW'] < ranks['MID'] < ranks['HIGH']
    assert ranks['LOW'] == 0.0
    assert ranks['HIGH'] == 100.0


def test_flag_outliers_flags_low_and_high(ahbd):
    # 20 individuals: one clear low outlier, one clear high outlier, rest mid-range
    stats = {f'M{i}': {'total_Mb': 50.0} for i in range(18)}
    stats['LOW'] = {'total_Mb': 0.1}
    stats['HIGH'] = {'total_Mb': 500.0}
    ranks = ahbd.compute_percentile_ranks(stats)

    flags = ahbd.flag_outliers(stats, ranks, low_percentile=5, outlier_percentile=95)
    assert flags['LOW'][0] is True
    assert 'LOW' in flags['LOW'][1]
    assert flags['HIGH'][0] is True
    assert 'HIGH' in flags['HIGH'][1]
    assert flags['M0'][0] is False


# ---------------------------------------------------------------------------
# load_pairs / compare_pair
# ---------------------------------------------------------------------------

def test_load_pairs_supports_comma_and_tab(ahbd, tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("A,B\nC\tD\n# comment\n\n")
    pairs = ahbd.load_pairs(str(f))
    assert pairs == [('A', 'B'), ('C', 'D')]


def test_compare_pair_reports_missing_data(ahbd):
    stats = {'A': {'total_Mb': 10.0}}
    ranks = {'A': 50.0}
    flags = {'A': (False, '')}
    note = ahbd.compare_pair('A', 'B', stats, ranks, flags)
    assert 'NA' in note
    assert 'B' in note


def test_compare_pair_flags_low_member(ahbd):
    stats = {'A': {'total_Mb': 1.0}, 'B': {'total_Mb': 50.0}}
    ranks = {'A': 2.0, 'B': 60.0}
    flags = {'A': (True, 'LOW (< p5=2.0 Mb) -- inconsistent with expected deep-selfing baseline'),
             'B': (False, '')}
    note = ahbd.compare_pair('A', 'B', stats, ranks, flags)
    assert 'atypically low autozygosity for A' in note


def test_compare_pair_both_typical(ahbd):
    stats = {'A': {'total_Mb': 50.0}, 'B': {'total_Mb': 55.0}}
    ranks = {'A': 40.0, 'B': 60.0}
    flags = {'A': (False, ''), 'B': (False, '')}
    note = ahbd.compare_pair('A', 'B', stats, ranks, flags)
    assert 'genuinely distinct inbred lines' in note


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    hbd_path = tmp_path / "step3_ibd.hbd.gz"
    rows = [(f'S{i}', f'S{i}', '1', 0, 50_000_000, 5.0) for i in range(18)]
    rows.append(('LOW', 'LOW', '1', 0, 100_000, 5.0))
    rows.append(('HIGH', 'HIGH', '1', 0, 500_000_000, 5.0))
    _write_hbd_gz(hbd_path, rows)

    out_prefix = tmp_path / "hbd_out"
    result = run_script(HBD_SCRIPT, [str(hbd_path), '-o', str(out_prefix)])
    assert result.returncode == 0, result.stderr

    analysis_tsv = tmp_path / "hbd_out_hbd_analysis.tsv"
    report = tmp_path / "hbd_out_hbd_report.txt"
    assert analysis_tsv.exists()
    assert report.exists()

    content = analysis_tsv.read_text()
    assert 'LOW' in content
    assert 'HIGH' in content

    report_content = report.read_text()
    assert 'Flagged individuals' in report_content


def test_cli_check_pairs(tmp_path):
    hbd_path = tmp_path / "step3_ibd.hbd.gz"
    rows = [(f'S{i}', f'S{i}', '1', 0, 50_000_000, 5.0) for i in range(18)]
    rows.append(('LOW', 'LOW', '1', 0, 100_000, 5.0))
    rows.append(('HIGH', 'HIGH', '1', 0, 500_000_000, 5.0))
    _write_hbd_gz(hbd_path, rows)

    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("LOW,S0\nS1,S2\n")

    out_prefix = tmp_path / "hbd_out"
    result = run_script(HBD_SCRIPT, [
        str(hbd_path), '-o', str(out_prefix), '--check-pairs', str(pairs_file),
    ])
    assert result.returncode == 0, result.stderr

    report_content = (tmp_path / "hbd_out_hbd_report.txt").read_text()
    assert 'Candidate pair comparison' in report_content
    assert 'LOW' in report_content


def test_cli_default_output_prefix_strips_hbd_suffix(tmp_path):
    hbd_path = tmp_path / "step3_ibd.hbd.gz"
    _write_hbd_gz(hbd_path, [('A', 'A', '1', 0, 1_000_000, 5.0)])
    result = run_script(HBD_SCRIPT, [str(hbd_path)], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "step3_ibd_hbd_analysis.tsv").exists()


def test_cli_no_segments_after_filter_exits_nonzero(tmp_path):
    hbd_path = tmp_path / "empty.hbd.gz"
    _write_hbd_gz(hbd_path, [('A', 'A', '1', 0, 1_000_000, 1.0)])  # below default min_lod
    result = run_script(HBD_SCRIPT, [str(hbd_path)])
    assert result.returncode != 0
    assert 'no HBD segments found' in (result.stdout + result.stderr)
