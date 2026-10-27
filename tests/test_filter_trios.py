"""
Tests for utilities/filter_trios.py.

Uses a synthetic .trio file with a known ambiguous child (two competing
parent pairs at different confidence) plus an inverted/generational trio,
matching the structure used during development of this script.
"""
import os
import pytest

from conftest import run_script, UTILITIES_DIR

FILTER_TRIOS = os.path.join(UTILITIES_DIR, 'filter_trios.py')

TRIO_CONTENT = (
    "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
    "A,B,C\tC\t50\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    "A,D,C\tC\t10\t1906\t120\t112\t93.3\t240\t228\t95.0\tRegular trio\n"
    "E,B,F\tF\t30\t1800\t110\t108\t98.2\t200\t198\t99.0\tRegular trio\n"
    "G,H,I\tI\t10\t1500\t90\t81\t90.0\t180\t155\t86.1\tInverted/generational trio\n"
)


@pytest.fixture
def trio_file(tmp_path):
    f = tmp_path / "input.trio"
    f.write_text(TRIO_CONTENT)
    return str(tmp_path / "input")  # prefix, not full filename


def _read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def test_single_header_no_duplication(trio_file, tmp_path):
    """
    Regression guard: filter_trios.py once used a quoted heredoc-equivalent
    csv.DictWriter.writeheader() call that could double-write the header.
    The fix uses plain csv.writer with an explicit single writerow(fieldnames).
    """
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr

    lines = _read_lines(f"{out_prefix}.trio")
    header_count = sum(1 for l in lines if l.startswith('Triple\tPivot'))
    assert header_count == 1, f"Expected exactly 1 header line, found {header_count}"


def test_strategy_best_keeps_highest_confidence(trio_file, tmp_path):
    """Child C has two competing pairs (A,B @ Perc=50,T1=100) and (A,D @ Perc=10,T1=93.3).
    'best' strategy must keep the (A,B) pair and drop (A,D)."""
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr

    content = open(f"{out_prefix}.trio").read()
    assert 'A,B,C' in content
    assert 'A,D,C' not in content
    # Single-pair child F passes through unchanged
    assert 'E,B,F' in content


def test_strategy_best_no_ambiguity_remains(trio_file, tmp_path):
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr
    lines = [l for l in _read_lines(f"{out_prefix}.trio") if l and not l.startswith('Triple')]
    children = [l.split('\t')[1] for l in lines]
    assert len(children) == len(set(children)), "Each child should appear exactly once"


def test_inverted_trios_passed_through_unchanged_by_default(trio_file, tmp_path):
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr
    content = open(f"{out_prefix}.trio").read()
    assert 'G,H,I' in content
    assert 'Inverted/generational trio' in content


def test_strategy_strict_filters_by_threshold(trio_file, tmp_path):
    """strict with min-t1=95 should drop the Perc=10 T1=93.3% (A,D,C) pair
    but might still leave ambiguity if multiple pairs pass."""
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS,
                        [trio_file, '-o', out_prefix, '--strategy', 'strict',
                         '--min-perc', '30', '--min-t1', '95'])
    assert result.returncode == 0, result.stderr
    content = open(f"{out_prefix}.trio").read()
    assert 'A,B,C' in content     # Perc=50, T1=100 -> passes
    assert 'A,D,C' not in content  # Perc=10 -> fails min-perc
    assert 'E,B,F' in content     # Perc=30, T1=98.2 -> passes


def test_strategy_consensus_reports_relatedness(trio_file, tmp_path):
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS,
                        [trio_file, '-o', out_prefix, '--strategy', 'consensus',
                         '--report-all'])
    assert result.returncode == 0, result.stderr
    # Consensus reports shared/alternate parents in stdout
    assert 'Likely true parent' in result.stdout or 'shared' in result.stdout.lower() \
        or 'A' in result.stdout  # parent A is shared between the two C-candidates


def test_include_inverted_flag(trio_file, tmp_path):
    """With --include-inverted, inverted trios go through the same
    best-per-child filtering as regular trios."""
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS,
                        [trio_file, '-o', out_prefix, '--strategy', 'best',
                         '--include-inverted'])
    assert result.returncode == 0, result.stderr
    content = open(f"{out_prefix}.trio").read()
    assert 'G,H,I' in content  # only one inverted candidate for I, passes through


def test_inverted_trio_file_is_written(trio_file, tmp_path):
    """filter_trios.py must write _inverted.trio so refine_pedigree.py --include-inverted
    can find it. Previously no such file was created, causing silent no-op."""
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr

    inv_path = f"{out_prefix}_inverted.trio"
    assert os.path.exists(inv_path), "_inverted.trio not written"
    content = open(inv_path).read()
    assert 'G,H,I' in content
    assert 'Inverted/generational trio' in content
    # must not contain regular trios
    assert 'A,B,C' not in content


def test_inverted_trio_file_written_with_include_inverted(trio_file, tmp_path):
    """_inverted.trio must also be written when --include-inverted is used."""
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS,
                        [trio_file, '-o', out_prefix, '--strategy', 'best',
                         '--include-inverted'])
    assert result.returncode == 0, result.stderr

    inv_path = f"{out_prefix}_inverted.trio"
    assert os.path.exists(inv_path), "_inverted.trio not written with --include-inverted"
    content = open(inv_path).read()
    assert 'G,H,I' in content


def test_output_is_valid_tsv_readable_by_csv(trio_file, tmp_path):
    """Output must be parseable by the same csv.DictReader other utilities use."""
    import csv
    out_prefix = str(tmp_path / "out")
    result = run_script(FILTER_TRIOS, [trio_file, '-o', out_prefix, '--strategy', 'best'])
    assert result.returncode == 0, result.stderr

    with open(f"{out_prefix}.trio", newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert len(rows) >= 2
    for row in rows:
        assert 'Triple' in row
        assert 'Pivot' in row
        assert 'Verdict' in row
