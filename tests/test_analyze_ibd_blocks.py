"""
Tests for utilities/analyze_ibd_blocks.py.

Covers flagging pairs with unexpectedly large shared hap-IBD blocks given
their pedigree relationship: pedigree-graph connectivity (BFS over FAM
parent-child edges), hap-IBD pair aggregation (LOD filtering, self-segment
exclusion), and rank-based outlier flagging computed SEPARATELY within each
pedigree-status bucket (connected vs. unconnected) so a large block is only
flagged relative to what other pairs in the SAME bucket show.
"""
import gzip
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

IBD_BLOCKS_SCRIPT = os.path.join(UTILITIES_DIR, 'analyze_ibd_blocks.py')


@pytest.fixture
def aib():
    return import_module_from_path('analyze_ibd_blocks', IBD_BLOCKS_SCRIPT)


def _write_ibd_gz(path, rows):
    """rows: list of (id1, id2, chrom, start, end, lod) tuples."""
    with gzip.open(path, 'wt') as f:
        for id1, id2, chrom, start, end, lod in rows:
            f.write(f"{id1}\t1\t{id2}\t2\t{chrom}\t{start}\t{end}\t{lod}\n")


# ---------------------------------------------------------------------------
# load_fam_graph / bfs_connected_within / is_pedigree_connected
# ---------------------------------------------------------------------------

def test_load_fam_graph_builds_undirected_edges(aib, tmp_path):
    f = tmp_path / "test.fam"
    f.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    graph = aib.load_fam_graph(str(f))
    assert graph['CHILD'] == {'DAD', 'MOM'}
    assert 'CHILD' in graph['DAD']
    assert 'CHILD' in graph['MOM']


def test_load_fam_graph_missing_file_exits(aib, tmp_path):
    with pytest.raises(SystemExit):
        aib.load_fam_graph(str(tmp_path / "nope.fam"))


def test_is_pedigree_connected_direct_parent_child(aib):
    graph = {'CHILD': {'DAD'}, 'DAD': {'CHILD'}}
    cache = {}
    assert aib.is_pedigree_connected(graph, 'CHILD', 'DAD', max_hops=6, cache=cache)


def test_is_pedigree_connected_siblings_via_shared_parent(aib):
    graph = {'A': {'DAD'}, 'B': {'DAD'}, 'DAD': {'A', 'B'}}
    cache = {}
    assert aib.is_pedigree_connected(graph, 'A', 'B', max_hops=6, cache=cache)


def test_is_pedigree_connected_respects_max_hops(aib):
    # A chain: A-B-C-D-E (5 hops between A and E)
    graph = {'A': {'B'}, 'B': {'A', 'C'}, 'C': {'B', 'D'}, 'D': {'C', 'E'}, 'E': {'D'}}
    cache = {}
    assert aib.is_pedigree_connected(graph, 'A', 'E', max_hops=4, cache=cache)
    cache2 = {}
    assert not aib.is_pedigree_connected(graph, 'A', 'E', max_hops=3, cache=cache2)


def test_is_pedigree_connected_no_path_returns_false(aib):
    graph = {'A': {'B'}, 'B': {'A'}, 'X': {'Y'}, 'Y': {'X'}}
    cache = {}
    assert not aib.is_pedigree_connected(graph, 'A', 'X', max_hops=6, cache=cache)


def test_bfs_cache_reused_across_calls(aib):
    graph = {'A': {'B'}, 'B': {'A'}}
    cache = {}
    result1 = aib.bfs_connected_within(graph, 'A', 6, cache)
    assert 'A' in cache
    result2 = aib.bfs_connected_within(graph, 'A', 6, cache)
    assert result1 is result2  # served from cache, not recomputed


# ---------------------------------------------------------------------------
# load_ibd_pair_stats
# ---------------------------------------------------------------------------

def test_load_ibd_pair_stats_aggregates_and_sorts_key(aib, tmp_path):
    path = tmp_path / "test.ibd.gz"
    _write_ibd_gz(path, [
        ('B', 'A', '1', 0, 1_000_000, 5.0),
        ('B', 'A', '1', 2_000_000, 4_000_000, 5.0),
    ])
    stats = aib.load_ibd_pair_stats(str(path))
    assert stats[('A', 'B')]['total_bp'] == 3_000_000
    assert stats[('A', 'B')]['n_segments'] == 2
    assert stats[('A', 'B')]['max_segment_bp'] == 2_000_000


def test_load_ibd_pair_stats_skips_self_segments(aib, tmp_path):
    path = tmp_path / "test.ibd.gz"
    _write_ibd_gz(path, [('A', 'A', '1', 0, 1_000_000, 5.0)])
    stats = aib.load_ibd_pair_stats(str(path))
    assert stats == {}


def test_load_ibd_pair_stats_filters_low_lod(aib, tmp_path):
    path = tmp_path / "test.ibd.gz"
    _write_ibd_gz(path, [('A', 'B', '1', 0, 1_000_000, 1.0)])
    stats = aib.load_ibd_pair_stats(str(path), min_lod=3.0)
    assert stats == {}


def test_load_ibd_pair_stats_missing_file_exits(aib, tmp_path):
    with pytest.raises(SystemExit):
        aib.load_ibd_pair_stats(str(tmp_path / "nope.ibd.gz"))


# ---------------------------------------------------------------------------
# classify_pairs / flag_outliers
# ---------------------------------------------------------------------------

def test_classify_pairs(aib):
    graph = {'A': {'B'}, 'B': {'A'}}
    pair_stats = {('A', 'B'): {}, ('C', 'D'): {}}
    status = aib.classify_pairs(pair_stats, graph, max_hops=6)
    assert status[('A', 'B')] == 'connected'
    assert status[('C', 'D')] == 'unconnected'


def test_flag_outliers_separates_buckets(aib):
    """A pair's flag must be judged against its OWN bucket (connected vs.
    unconnected), not the global distribution -- a modest block that's
    unremarkable among connected (expected-related) pairs could still be
    a stand-out outlier among unconnected (expected-unrelated) pairs."""
    pair_stats = {}
    status = {}
    # 18 typical "connected" pairs with large blocks (expected, related)
    for i in range(18):
        pair_stats[(f'C{i}a', f'C{i}b')] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 50_000_000}
        status[(f'C{i}a', f'C{i}b')] = 'connected'
    # 18 typical "unconnected" pairs with small blocks (expected, unrelated)
    for i in range(18):
        pair_stats[(f'U{i}a', f'U{i}b')] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 1_000_000}
        status[(f'U{i}a', f'U{i}b')] = 'unconnected'
    # One unconnected pair with a surprisingly large block
    pair_stats[('SUSPECT_A', 'SUSPECT_B')] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 40_000_000}
    status[('SUSPECT_A', 'SUSPECT_B')] = 'unconnected'

    flags = aib.flag_outliers(pair_stats, status, hotspot_pairs=set(), outlier_percentile=95)
    assert flags[('SUSPECT_A', 'SUSPECT_B')][0] is True
    assert 'undocumented relationship' in flags[('SUSPECT_A', 'SUSPECT_B')][1]
    # A typical connected pair should NOT be flagged just for having a large
    # block -- that's normal/expected for connected pairs.
    assert flags[('C0a', 'C0b')][0] is False


# ---------------------------------------------------------------------------
# detect_hotspot_regions / hotspot-aware flag_outliers
# ---------------------------------------------------------------------------

def test_detect_hotspot_regions_finds_recurring_region(aib):
    """
    Regression guard for a real production finding: tens of thousands of
    pairs all reporting the SAME largest-segment length -- the signature of
    a population-wide conserved region, not independent private relatedness.
    A region that recurs as the max segment across many distinct pairs must
    be detected as a hotspot.
    """
    pair_stats = {}
    # 15 distinct pairs all sharing (roughly) the same genomic region as
    # their largest segment -- a population-wide hotspot.
    for i in range(15):
        pair_stats[(f'H{i}a', f'H{i}b')] = {
            'max_segment_bp': 45_000_000, 'max_segment_region': ('1', 0, 45_000_000)}
    # One pair with a large block in a genuinely DIFFERENT chromosome -- not a hotspot.
    pair_stats[('PRIVATE_A', 'PRIVATE_B')] = {
        'max_segment_bp': 40_000_000, 'max_segment_region': ('2', 10_000_000, 50_000_000)}

    hotspot_pairs, region_counts = aib.detect_hotspot_regions(pair_stats, min_pairs_for_hotspot=10)
    assert all((f'H{i}a', f'H{i}b') in hotspot_pairs for i in range(15))
    assert ('PRIVATE_A', 'PRIVATE_B') not in hotspot_pairs


def test_detect_hotspot_regions_unifies_jittered_boundaries(aib):
    """
    Regression guard for the real bug this per-chromosome design fixes: a
    single chromosome-scale conservation event was previously fragmented
    into 15 "different" detected regions (chr01:4000000-44000000,
    chr01:4000000-45000000, chr01:2000000-44000000, ...) purely because of a
    few Mb of segment-boundary jitter between different pairs -- first under
    exact-coordinate matching, then again under a coverage-fraction bucket
    (e.g. 95% vs. 99.6% coverage of the same chromosome landing in different
    buckets). Grouping by CHROMOSOME ALONE (once a segment clears
    min_coverage_frac) leaves no sub-bucket boundary to straddle, so jittered
    boundaries can no longer fragment a single real event.
    """
    pair_stats = {}
    # All ~96-100% of a ~45 Mb chromosome, but with jittered exact boundaries
    # -- exactly the real-world pattern (chrom_max_end ends up 45,000,000).
    jittered_regions = [
        ('1', 0, 45_000_000), ('1', 1_000_000, 45_000_000),
        ('1', 0, 44_000_000), ('1', 1_000_000, 44_500_000),
        ('1', 500_000, 44_800_000), ('1', 0, 44_200_000),
        ('1', 800_000, 45_000_000), ('1', 200_000, 44_600_000),
    ]
    for i, region in enumerate(jittered_regions):
        for j in range(3):  # 3 distinct pairs per jittered variant = 24 pairs total
            pair_stats[(f'H{i}_{j}a', f'H{i}_{j}b')] = {
                'max_segment_bp': region[2] - region[1], 'max_segment_region': region}
    # A pair covering well under min_coverage_frac of the chromosome -- never
    # a hotspot candidate at all, regardless of sharing the same chromosome.
    pair_stats[('PARTIAL_A', 'PARTIAL_B')] = {
        'max_segment_bp': 5_000_000, 'max_segment_region': ('1', 0, 5_000_000)}

    hotspot_pairs, region_counts = aib.detect_hotspot_regions(pair_stats, min_pairs_for_hotspot=20)
    # All 24 jittered pairs must land under the SAME single chromosome key.
    assert region_counts == {'1': 24}
    all_jittered_keys = [(f'H{i}_{j}a', f'H{i}_{j}b')
                         for i in range(len(jittered_regions)) for j in range(3)]
    assert all(key in hotspot_pairs for key in all_jittered_keys)
    assert ('PARTIAL_A', 'PARTIAL_B') not in hotspot_pairs


def test_detect_hotspot_regions_no_data_returns_empty(aib):
    hotspot_pairs, region_counts = aib.detect_hotspot_regions({})
    assert hotspot_pairs == set()
    assert region_counts == {}


def test_detect_hotspot_regions_self_calibrates_from_pair_fraction(aib):
    """
    Regression guard for the real bug the fraction-of-total-pairs design
    fixes: percentile-of-chromosome-counts is degenerate with only a
    handful of chromosomes -- the 99th percentile of ~15 values just picks
    the single largest chromosome's own count, silently excluding every
    OTHER chromosome that also shows strong widespread sharing. Threshold
    must instead scale with the TOTAL pair count, so a second chromosome
    with a comparably large (but not identical) count still clears it.
    """
    pair_stats = {}
    # chr01: 40 pairs at ~100% coverage.
    for i in range(40):
        pair_stats[(f'H1_{i}a', f'H1_{i}b')] = {'max_segment_region': ('1', 0, 45_000_000)}
    # chr02: 30 pairs at ~100% coverage -- fewer than chr01, but still a
    # real widespread-sharing signal that a percentile-of-2-chromosomes
    # calculation would exclude.
    for i in range(30):
        pair_stats[(f'H2_{i}a', f'H2_{i}b')] = {'max_segment_region': ('2', 0, 30_000_000)}
    # 1000 private, non-hotspot pairs on a third chromosome to establish a
    # realistic total-pair-count denominator.
    for i in range(1000):
        pair_stats[(f'P{i}a', f'P{i}b')] = {'max_segment_region': ('3', i * 1000, i * 1000 + 500)}

    hotspot_pairs, region_counts = aib.detect_hotspot_regions(pair_stats, hotspot_pair_fraction=0.01)
    assert region_counts == {'1': 40, '2': 30}
    # 1% of 1070 total pairs = ~11 -- both chr01 (40) and chr02 (30) clear it.
    assert all((f'H1_{i}a', f'H1_{i}b') in hotspot_pairs for i in range(40))
    assert all((f'H2_{i}a', f'H2_{i}b') in hotspot_pairs for i in range(30))


def test_detect_locus_hotspots_finds_recurring_subchromosomal_region(aib):
    """
    Regression guard for a real production finding: a ~39.6 Mb block at
    chr02:264222-39828158 was shared by 6,604 distinct pairs -- far too
    small a fraction of chr02's total span to ever be caught by
    detect_hotspot_regions()'s chromosome-coverage-floor mechanism. A
    tightly-bounded sub-chromosomal region recurring (with small boundary
    jitter) across many distinct pairs must be detected here instead.
    """
    pair_stats = {}
    jittered = [
        ('2', 264222, 39828158), ('2', 264277, 39828158),
        ('2', 264222, 39827961), ('2', 264222, 39827946),
        ('2', 264387, 39828158),
    ]
    for i, region in enumerate(jittered):
        for j in range(3):  # 3 pairs per variant = 15 pairs total
            pair_stats[(f'H{i}_{j}a', f'H{i}_{j}b')] = {'max_segment_region': region}
    # A genuinely different, non-overlapping region on the SAME chromosome.
    pair_stats[('PRIVATE_A', 'PRIVATE_B')] = {
        'max_segment_region': ('2', 60_000_000, 65_000_000)}
    # A different chromosome entirely.
    pair_stats[('OTHER_A', 'OTHER_B')] = {
        'max_segment_region': ('3', 0, 40_000_000)}

    hotspot_pairs, region_counts = aib.detect_locus_hotspots(pair_stats, min_pairs_for_hotspot=10)
    all_hotspot_keys = [(f'H{i}_{j}a', f'H{i}_{j}b')
                        for i in range(len(jittered)) for j in range(3)]
    assert all(key in hotspot_pairs for key in all_hotspot_keys)
    assert ('PRIVATE_A', 'PRIVATE_B') not in hotspot_pairs
    assert ('OTHER_A', 'OTHER_B') not in hotspot_pairs


def test_detect_locus_hotspots_respects_min_overlap(aib):
    """Two groups of segments overlapping each other by exactly 40% must be
    unified into one cluster only when min_overlap_frac allows it."""
    pair_stats = {}
    for i in range(15):
        pair_stats[(f'A{i}a', f'A{i}b')] = {'max_segment_region': ('1', 0, 1_000_000)}
    for i in range(15):
        pair_stats[(f'B{i}a', f'B{i}b')] = {'max_segment_region': ('1', 600_000, 1_600_000)}

    _, region_counts_strict = aib.detect_locus_hotspots(pair_stats, min_overlap_frac=0.5)
    assert len(region_counts_strict) == 2

    _, region_counts_loose = aib.detect_locus_hotspots(pair_stats, min_overlap_frac=0.3)
    assert len(region_counts_loose) == 1


def test_detect_locus_hotspots_no_data_returns_empty(aib):
    hotspot_pairs, region_counts = aib.detect_locus_hotspots({})
    assert hotspot_pairs == set()
    assert region_counts == {}


def test_detect_locus_hotspots_excludes_already_chromosome_hotspot_pairs(aib):
    """
    Regression guard for a real production finding: pairs already explained
    by detect_hotspot_regions()'s chromosome-scale mechanism were
    independently rediscovered by detect_locus_hotspots() too, fragmented
    into extra, misleading "loci" of their own (a single chr01 event became
    9 separate reported loci). exclude_pairs must keep already-explained
    pairs out of locus detection entirely, regardless of how they'd
    otherwise cluster.
    """
    pair_stats = {}
    for i in range(15):
        pair_stats[(f'A{i}a', f'A{i}b')] = {'max_segment_region': ('1', 0, 1_000_000)}
    for i in range(15):
        pair_stats[(f'B{i}a', f'B{i}b')] = {'max_segment_region': ('1', 600_000, 1_600_000)}
    already_explained = set(pair_stats.keys())

    # Without exclusion, these fragment into 2 separate locus clusters
    # (confirmed by test_detect_locus_hotspots_respects_min_overlap).
    _, region_counts_no_exclude = aib.detect_locus_hotspots(pair_stats, min_overlap_frac=0.5)
    assert len(region_counts_no_exclude) == 2

    # With exclusion, none of these pairs are even candidates.
    hotspot_pairs, region_counts = aib.detect_locus_hotspots(
        pair_stats, min_overlap_frac=0.5, exclude_pairs=already_explained)
    assert region_counts == {}
    assert hotspot_pairs == set()


def test_flag_outliers_excludes_hotspot_pairs_and_never_flags_them(aib):
    """Hotspot pairs must (a) never be flagged UNEXPECTED_LARGE_BLOCK even
    though their segment is large, (b) still appear in the result (never
    silently dropped), tagged distinctly, and (c) be excluded from the
    ranking pool so they don't distort percentiles for genuinely private
    large blocks that remain."""
    pair_stats = {}
    status = {}
    hotspot_pairs = set()
    # 15 hotspot pairs, all tied at the same large value
    for i in range(15):
        key = (f'H{i}a', f'H{i}b')
        pair_stats[key] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 45_000_000}
        status[key] = 'unconnected'
        hotspot_pairs.add(key)
    # A handful of typical unconnected pairs with modest, non-hotspot blocks
    for i in range(15):
        key = (f'U{i}a', f'U{i}b')
        pair_stats[key] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 1_000_000}
        status[key] = 'unconnected'
    # One genuinely suspicious, non-hotspot large block
    pair_stats[('SUSPECT_A', 'SUSPECT_B')] = {'total_bp': 0, 'n_segments': 1, 'max_segment_bp': 40_000_000}
    status[('SUSPECT_A', 'SUSPECT_B')] = 'unconnected'

    flags = aib.flag_outliers(pair_stats, status, hotspot_pairs, outlier_percentile=95)
    for i in range(15):
        flagged, reason, rank = flags[(f'H{i}a', f'H{i}b')]
        assert flagged is False
        assert 'COMMON_HOTSPOT_REGION' in reason
        assert rank is None
    assert flags[('SUSPECT_A', 'SUSPECT_B')][0] is True


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    fam = tmp_path / "test.fam"
    fam.write_text(
        "0\tCHILD\tDAD\tMOM\t0\t-9\n"
        "0\tDAD\t0\t0\t1\t-9\n"
        "0\tMOM\t0\t0\t2\t-9\n"
    )
    ibd_path = tmp_path / "test.ibd.gz"
    rows = [('CHILD', 'DAD', '1', 0, 50_000_000, 5.0)]  # pedigree-connected
    # Each filler pair gets a DIFFERENT genomic region -- same value in the
    # SAME region across many pairs would itself look like a hotspot (see
    # test_cli_end_to_end_hotspot_region below for that scenario).
    for i in range(18):
        start = i * 5_000_000
        rows.append((f'U{i}a', f'U{i}b', '1', start, start + 1_000_000, 5.0))
    rows.append(('SUSPECT_A', 'SUSPECT_B', '2', 0, 40_000_000, 5.0))
    _write_ibd_gz(ibd_path, rows)

    out_prefix = tmp_path / "blocks_out"
    result = run_script(IBD_BLOCKS_SCRIPT, [str(fam), str(ibd_path), '-o', str(out_prefix)])
    assert result.returncode == 0, result.stderr

    analysis_tsv = tmp_path / "blocks_out_ibd_block_analysis.tsv"
    report = tmp_path / "blocks_out_ibd_block_report.txt"
    assert analysis_tsv.exists()
    assert report.exists()

    content = analysis_tsv.read_text()
    assert 'SUSPECT_A' in content
    assert 'connected' in content
    assert 'unconnected' in content

    report_content = report.read_text()
    assert 'SUSPECT_A' in report_content
    assert 'Flagged pairs' in report_content


def test_cli_end_to_end_hotspot_region(tmp_path):
    """
    Reproduces the real production finding directly: many distinct pairs
    all sharing the identical large "largest segment" in the same region
    must be reported as a detected hotspot and NOT counted as unexpected,
    while a genuinely distinct large block in a different region still is.
    """
    fam = tmp_path / "test.fam"
    fam.write_text("0\tFOUNDER\t0\t0\t1\t-9\n")
    ibd_path = tmp_path / "test.ibd.gz"
    rows = []
    for i in range(20):
        rows.append((f'H{i}a', f'H{i}b', '1', 0, 45_000_000, 5.0))
    rows.append(('SUSPECT_A', 'SUSPECT_B', '2', 0, 40_000_000, 5.0))
    _write_ibd_gz(ibd_path, rows)

    out_prefix = tmp_path / "blocks_out"
    result = run_script(IBD_BLOCKS_SCRIPT, [
        str(fam), str(ibd_path), '-o', str(out_prefix), '--min-hotspot-pairs', '10',
    ])
    assert result.returncode == 0, result.stderr

    report_content = (tmp_path / "blocks_out_ibd_block_report.txt").read_text()
    assert 'Detected hotspot chromosomes' in report_content
    assert '1 — largest segment for 20 distinct pairs' in report_content

    content = (tmp_path / "blocks_out_ibd_block_analysis.tsv").read_text()
    rows_by_pair = {l.split('\t')[0]: l for l in content.splitlines()[1:]}
    assert 'YES' == rows_by_pair['H0a'].split('\t')[9]   # Hotspot column
    assert 'no' == rows_by_pair['H0a'].split('\t')[8]    # Flagged column -- not flagged
    assert 'no' == rows_by_pair['SUSPECT_A'].split('\t')[9]   # not a hotspot pair


def test_cli_end_to_end_locus_hotspot(tmp_path):
    """
    Reproduces the real production finding directly: many distinct pairs
    sharing a tightly-bounded sub-chromosomal region (too small a fraction
    of its chromosome to trip the chromosome-scale mechanism) must be
    reported as a detected hotspot LOCUS and NOT counted as unexpected.
    """
    fam = tmp_path / "test.fam"
    fam.write_text("0\tFOUNDER\t0\t0\t1\t-9\n")
    ibd_path = tmp_path / "test.ibd.gz"
    rows = []
    for i in range(20):
        rows.append((f'H{i}a', f'H{i}b', '2', 264222, 39828158, 5.0))
    # A big chromosome so the shared ~39.6 Mb block never crosses the 30%
    # chromosome-coverage floor used for chromosome-scale hotspot detection.
    rows.append(('ANCHOR_A', 'ANCHOR_B', '2', 0, 200_000_000, 5.0))
    rows.append(('SUSPECT_A', 'SUSPECT_B', '3', 0, 40_000_000, 5.0))
    _write_ibd_gz(ibd_path, rows)

    out_prefix = tmp_path / "blocks_out"
    result = run_script(IBD_BLOCKS_SCRIPT, [
        str(fam), str(ibd_path), '-o', str(out_prefix), '--min-hotspot-pairs', '10',
    ])
    assert result.returncode == 0, result.stderr

    report_content = (tmp_path / "blocks_out_ibd_block_report.txt").read_text()
    assert 'Detected hotspot loci' in report_content
    assert '2:264222-39828158 — shared by 20 distinct pairs' in report_content

    content = (tmp_path / "blocks_out_ibd_block_analysis.tsv").read_text()
    rows_by_pair = {l.split('\t')[0]: l for l in content.splitlines()[1:]}
    assert 'YES' == rows_by_pair['H0a'].split('\t')[9]   # Hotspot column
    assert 'no' == rows_by_pair['H0a'].split('\t')[8]    # Flagged column -- not flagged


def test_cli_default_output_prefix_strips_ibd_suffix(tmp_path):
    fam = tmp_path / "test.fam"
    fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    ibd_path = tmp_path / "test_stat_ibd.ibd.gz"
    _write_ibd_gz(ibd_path, [('CHILD', 'DAD', '1', 0, 1_000_000, 5.0)])
    result = run_script(IBD_BLOCKS_SCRIPT, [str(fam), str(ibd_path)], cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "test_stat_ibd_ibd_block_analysis.tsv").exists()


def test_cli_no_ibd_pairs_after_filter_exits_nonzero(tmp_path):
    fam = tmp_path / "test.fam"
    fam.write_text("0\tCHILD\tDAD\tMOM\t0\t-9\n")
    ibd_path = tmp_path / "test.ibd.gz"
    _write_ibd_gz(ibd_path, [('CHILD', 'DAD', '1', 0, 1_000_000, 1.0)])  # below default min_lod
    result = run_script(IBD_BLOCKS_SCRIPT, [str(fam), str(ibd_path)])
    assert result.returncode != 0
    assert 'no IBD pairs found' in (result.stdout + result.stderr)
