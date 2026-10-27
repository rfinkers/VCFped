"""
Tests for utilities/fixed_pedigree_viz.py.

Covers:
  - _validate_vcfped_file: good/missing/wrong-column files (regression guard
    for the original bug where a wrong-prefix run silently produced an empty
    pedigree instead of erroring)
  - PedigreeBuilder: trio/MZ-twin/generation/family-detection logic
  - End-to-end HTML report generation on the bundled test VCF's vcfped output
  - --pair file being optional (regression guard: required after a
    --no-pairwise run, which is the documented two-stage workflow)
"""
import os
import sys
import pytest

from conftest import (import_module_from_path, run_script, UTILITIES_DIR,
                      requires_pandas, requires_plotly)

pytestmark = [requires_pandas, requires_plotly]

VIZ_SCRIPT = os.path.join(UTILITIES_DIR, 'fixed_pedigree_viz.py')


@pytest.fixture
def viz():
    return import_module_from_path('fixed_pedigree_viz', VIZ_SCRIPT)


# ---------------------------------------------------------------------------
# _validate_vcfped_file
# ---------------------------------------------------------------------------

def test_validate_good_trio_file(viz, tmp_path):
    f = tmp_path / "good.trio"
    f.write_text("Triple\tPivot\tPerc\tVerdict\nA,B,C\tC\t10\tRegular trio\n")
    df = viz._validate_vcfped_file(str(f), 'trio')
    assert df is not None
    assert len(df) == 1


def test_validate_missing_required_file_exits(viz, tmp_path):
    """A missing .trio file (required) must hard-exit with a clear message,
    not silently produce an empty pedigree."""
    with pytest.raises(SystemExit):
        viz._validate_vcfped_file(str(tmp_path / "nonexistent.trio"), 'trio')


def test_validate_wrong_columns_exits(viz, tmp_path):
    """A file present but with the wrong columns (e.g. wrong prefix pointed
    at an unrelated file) must error, not silently proceed."""
    f = tmp_path / "wrong.trio"
    f.write_text("Col1\tCol2\nfoo\tbar\n")
    with pytest.raises(SystemExit):
        viz._validate_vcfped_file(str(f), 'trio')


def test_validate_missing_optional_gender_file_warns_not_fails(viz, tmp_path, capsys):
    result = viz._validate_vcfped_file(str(tmp_path / "nonexistent.gender"), 'gender')
    assert result is None  # optional -> None, not an exit


def test_validate_missing_optional_pair_file_warns_not_fails(viz, tmp_path):
    """Regression guard: .pair is absent after --no-pairwise runs (the
    documented two-stage MZ-purge workflow) — must not be treated as fatal."""
    result = viz._validate_vcfped_file(str(tmp_path / "nonexistent.pair"), 'pair')
    assert result is None


# ---------------------------------------------------------------------------
# PedigreeBuilder
# ---------------------------------------------------------------------------

def test_pedigree_builder_add_trio(viz):
    b = viz.PedigreeBuilder()
    b.add_trio('FATHER', 'MOTHER', 'CHILD')
    assert 'CHILD' in b.individuals
    assert 'FATHER' in b.individuals['CHILD']['parents']
    assert 'MOTHER' in b.individuals['CHILD']['parents']
    assert 'CHILD' in b.individuals['FATHER']['children']


def test_pedigree_builder_assign_generations(viz):
    b = viz.PedigreeBuilder()
    b.add_trio('GP1', 'GP2', 'PARENT')
    b.add_trio('PARENT', 'OTHER', 'CHILD')
    b.assign_generations()
    assert b.individuals['GP1']['generation'] == 0
    assert b.individuals['GP2']['generation'] == 0
    assert b.individuals['PARENT']['generation'] == 1
    assert b.individuals['OTHER']['generation'] == 0  # founder, no parents
    assert b.individuals['CHILD']['generation'] == 2


def test_pedigree_builder_detect_families_separates_unrelated_trios(viz):
    b = viz.PedigreeBuilder()
    b.add_trio('A1', 'A2', 'A3')   # family 1
    b.add_trio('B1', 'B2', 'B3')   # family 2, unrelated
    families = b.detect_families()
    assert len(families) == 2
    fam_sizes = sorted(len(f) for f in families)
    assert fam_sizes == [3, 3]


def test_pedigree_builder_mz_twin_collapse(viz):
    b = viz.PedigreeBuilder(collapse_mz_twins=True)
    b.add_mz_twins('TWIN1', 'TWIN2')
    b.add_individual('TWIN1')
    b.add_individual('TWIN2')
    b.collapse_mz_twin_groups()
    # Both twins should be merged into a single MZ group entity
    remaining = [k for k in b.individuals if k in ('TWIN1', 'TWIN2')]
    assert remaining == []  # originals removed
    mz_groups = [k for k, v in b.individuals.items() if v.get('is_mz_group')]
    assert len(mz_groups) == 1
    assert b.individuals[mz_groups[0]]['member_count'] == 2


def test_pedigree_builder_inverted_trio_excluded_from_relationship_graph(viz):
    b = viz.PedigreeBuilder()
    b.add_inverted_trio('X', 'Y', 'Z')
    assert len(b.inverted_trios) == 1
    assert b.parent_child == []  # not added to the regular graph


def test_compute_statistics_full_sibling_detection(viz):
    b = viz.PedigreeBuilder()
    b.add_trio('FATHER', 'MOTHER', 'CHILD1')
    b.add_trio('FATHER', 'MOTHER', 'CHILD2')  # full siblings
    stats = b.compute_statistics()
    assert stats['n_full_sibling_groups'] == 1


def test_compute_statistics_half_sibling_detection(viz):
    b = viz.PedigreeBuilder()
    b.add_trio('FATHER', 'MOTHER1', 'CHILD1')
    b.add_trio('FATHER', 'MOTHER2', 'CHILD2')  # half siblings via FATHER
    stats = b.compute_statistics()
    assert stats['n_half_sib_pairs'] == 1


# ---------------------------------------------------------------------------
# _family_plot: trace batching, width scaling, label thresholding
# ---------------------------------------------------------------------------

def _linear_family(viz, n):
    """Build a simple n-generation parent-child chain (P0 -> P1 -> ... -> Pn-1)
    plus per-generation individuals dict, for _family_plot() tests."""
    b = viz.PedigreeBuilder()
    members = [f"P{i}" for i in range(n)]
    for i in range(n - 1):
        b.add_individual(members[i])
        b.add_individual(members[i + 1])
        b.add_parent_child(members[i], members[i + 1])
    b.assign_generations()
    return members, b.individuals


def test_family_plot_batches_node_traces_regardless_of_size(viz):
    """
    Regression guard: the old implementation added ONE Plotly trace PER
    individual (and two per couple), so a large family produced thousands of
    traces -- the single biggest driver of slow, dense-looking renders.
    Node traces must now be batched by (gender, is_mz_group), bounded by a
    small constant regardless of family size.
    """
    members, individuals = _linear_family(viz, 60)
    couples = [(f"P{i}", f"P{i}", f"P{i+1}") for i in range(59)]  # not used, just needs a list
    fig = viz._family_plot(members, individuals, [], {}, title="test")
    # All members here have gender=None and are not MZ groups -> exactly one
    # node trace bucket, regardless of there being 60 individuals.
    node_traces = [t for t in fig.data if t.mode and 'markers' in t.mode]
    assert len(node_traces) == 1
    assert len(node_traces[0].x) == 60


def test_family_plot_batches_edge_traces(viz):
    """Edges must be batched into at most 2 traces (bars, drops) via
    None-separated segments, not 2 traces PER couple."""
    b = viz.PedigreeBuilder()
    couples = []
    for i in range(20):
        p1, p2, c = f"D{i}", f"M{i}", f"C{i}"
        b.add_trio(p1, p2, c)
        couples.append((p1, p2, c))
    b.assign_generations()
    fig = viz._family_plot(list(b.individuals.keys()), b.individuals, couples, {}, title="test")
    line_traces = [t for t in fig.data if t.mode == 'lines']
    assert len(line_traces) <= 2


def test_family_plot_width_scales_with_widest_generation(viz):
    small_members, small_individuals = _linear_family(viz, 3)
    # A family whose single generation has many members (wide, not deep)
    b = viz.PedigreeBuilder()
    for i in range(200):
        b.add_individual(f"F{i}")
    b.assign_generations()
    wide_members = list(b.individuals.keys())

    fig_small = viz._family_plot(small_members, small_individuals, [], {}, title="small")
    fig_wide = viz._family_plot(wide_members, b.individuals, [], {}, title="wide")
    assert fig_wide.layout.width > fig_small.layout.width


def test_family_plot_hides_labels_above_threshold(viz):
    members, individuals = _linear_family(viz, viz.LABEL_THRESHOLD + 5)
    fig = viz._family_plot(members, individuals, [], {}, title="large")
    node_trace = [t for t in fig.data if t.mode and 'markers' in t.mode][0]
    assert node_trace.mode == 'markers'  # no '+text'


def test_family_plot_shows_labels_below_threshold(viz):
    members, individuals = _linear_family(viz, min(5, viz.LABEL_THRESHOLD - 1))
    fig = viz._family_plot(members, individuals, [], {}, title="small")
    node_trace = [t for t in fig.data if t.mode and 'markers' in t.mode][0]
    assert node_trace.mode == 'markers+text'


# ---------------------------------------------------------------------------
# _family_summary_card
# ---------------------------------------------------------------------------

def test_family_summary_card_reports_stats_not_a_diagram(viz):
    members, individuals = _linear_family(viz, 10)
    html = viz._family_summary_card(members, individuals, "Family F1 — 10 individuals", 5)
    assert 'Too large to render' in html
    assert 'founders' in html
    assert 'leaves' in html
    assert '<b>10</b> generations' in html  # a 10-node linear chain spans 10 generations


# ---------------------------------------------------------------------------
# End-to-end report generation
# ---------------------------------------------------------------------------

def test_end_to_end_report_generation(vcfped_run, tmp_path):
    out_html = str(tmp_path / "pedigree.html")
    result = run_script(VIZ_SCRIPT, [vcfped_run, '-o', out_html, '--no-browser'])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out_html)
    content = open(out_html).read()
    # vcfped_run fixture does not rename samples, so the trio file (and
    # hence the report) uses numeric sample IDs (1,2,3), not HG002 etc.
    assert 'Summary Statistics' in content
    assert '<html' in content.lower()


def test_end_to_end_without_pair_file(work_dir, test_vcf, tmp_path):
    """Simulates the documented --no-pairwise workflow: no .pair file exists,
    visualization must still succeed using only .trio."""
    from conftest import run_module
    prefix = os.path.join(work_dir, 'nopw')
    result = run_module(['vcfped.vcfped_optimized', test_vcf,
                         '--no-pairwise', '-o', prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr
    assert not os.path.exists(f"{prefix}.pair")

    out_html = str(tmp_path / "pedigree.html")
    result = run_script(VIZ_SCRIPT, [prefix, '-o', out_html, '--no-browser'])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out_html)


def test_wrong_prefix_errors_clearly(tmp_path):
    out_html = str(tmp_path / "pedigree.html")
    result = run_script(VIZ_SCRIPT,
                        [str(tmp_path / "totally_wrong_prefix"), '-o', out_html,
                         '--no-browser'])
    assert result.returncode != 0
    assert 'not found' in result.stderr.lower() or 'not found' in result.stdout.lower()


def test_end_to_end_oversized_family_gets_summary_card(tmp_path):
    """A family bigger than --max-diagram-size must be rendered as a compact
    summary card, not a dense/slow full diagram."""
    lines = ["Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict"]
    # One big family: a shared couple with 30 children (32 individuals total)
    for i in range(30):
        lines.append(f"DAD,MOM,C{i}\tC{i}\t30\t100\t50\t50\t99.0\t20\t18\t95.0\tRegular trio")
    # One small, separate family of 3
    lines.append("A,B,C\tC\t30\t100\t50\t50\t99.0\t20\t18\t95.0\tRegular trio")
    trio_path = tmp_path / "test.trio"
    trio_path.write_text("\n".join(lines) + "\n")

    out_html = str(tmp_path / "pedigree.html")
    result = run_script(VIZ_SCRIPT, [
        str(tmp_path / "test"), '-o', out_html, '--no-browser', '--max-diagram-size', '10',
    ])
    assert result.returncode == 0, result.stderr
    content = open(out_html).read()
    assert 'Too large to render' in content
