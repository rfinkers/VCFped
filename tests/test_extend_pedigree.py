"""
Tests for utilities/extend_pedigree.py.

This script went through the most extensive bug-fixing in development —
each test below documents which specific bug it guards against:

  1. build_king_index / indexed search:
     find_grandparent_candidates() and find_offspring_candidates() used to
     scan the ENTIRE king dict for every founder/leaf — O(N_individuals *
     N_pairs). For a cohort of ~3500 founders against ~1.6M KING pairs that
     is ~5.6 BILLION dict iterations for the grandparent search alone.
     Fixed by building a one-time adjacency index.

  2. Founder/leaf "Added 0" bug: current_fam already contains every
     individual in the cohort, so `child not in current_fam` was always
     False. Fixed to test "currently unassigned" (pat=='0' and mat=='0').

  3. Candidate-file truncation bug: args.max_candidates was reused both as
     a per-founder/leaf retention cap AND as a global slice when writing
     outputs, silently dropping >99% of candidates down to the first ~50
     records (covering only the first 1-2 founders). Fixed by writing the
     FULL retained pool to outputs, with a separate --report-limit for the
     human-readable report only.

  4. Upward feedback gap: grandparent candidates were never written back
     into current_fam during the loop, so --max-iter > 1 found 0 new
     founders in the upward direction. Fixed by feeding top-N grandparent
     assignments back, mirroring the offspring (downward) feedback.

  5. write_beagle_check_script phased-VCF path bug: guessed the phased VCF
     name from extend_pedigree.py's OWN output prefix, but it's actually
     named after prepare_plink_beagle.py's separate -o prefix. Fixed with
     an explicit --phased-vcf argument and a clearly-labeled guess fallback.

  6. Missing mandatory hap-IBD map= argument (same bug class as in
     prepare_plink_beagle.py, just a different generated script that wasn't
     updated when the first fix was made).
"""
import os
import re
import time
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

EP_SCRIPT = os.path.join(UTILITIES_DIR, 'extend_pedigree.py')


@pytest.fixture
def ep():
    return import_module_from_path('extend_pedigree', EP_SCRIPT)


# ---------------------------------------------------------------------------
# build_king_index + indexed search performance/correctness
# ---------------------------------------------------------------------------

def test_king_index_bidirectional(ep):
    king = {('A', 'B'): {'kinship': 0.25, 'ibs0': 0.005}}
    idx = ep.build_king_index(king)
    assert ('B', 0.25, 0.005) in idx['A']
    assert ('A', 0.25, 0.005) in idx['B']


def test_king_index_avoids_full_scan_per_founder(ep):
    """
    Regression guard for the O(N_individuals x N_pairs) bug: with the index,
    searching many founders against a moderately large KING table must
    complete near-instantly, not scale linearly with (founders x pairs).
    """
    import random
    random.seed(0)
    N = 150
    samples = [f"S{i:04d}" for i in range(N)]
    king = {}
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < 0.2:
                king[(samples[i], samples[j])] = {
                    'kinship': random.choice([0.13, 0.12, 0.26]),
                    'ibs0': 0.01,
                }
    king_index = ep.build_king_index(king)

    t0 = time.time()
    for s in samples:
        ep.find_grandparent_candidates(s, king, king_index, set(samples))
    elapsed = time.time() - t0
    # With the index this is milliseconds; the old O(N*P) approach at this
    # scale would already take a noticeably longer time. Generous bound to
    # avoid CI flakiness while still catching a regression to the old
    # unindexed behaviour.
    assert elapsed < 2.0, f"Indexed search took {elapsed:.2f}s — possible regression to full-scan"


# ---------------------------------------------------------------------------
# collapse_mz_twin_groups_in_king
# ---------------------------------------------------------------------------

def test_collapse_mz_twin_groups_basic(ep):
    king = {
        ('A', 'B'): {'kinship': 0.472, 'ibs0': 0.000},  # MZ twins
        ('A', 'C'): {'kinship': 0.13, 'ibs0': 0.01},
    }
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(king, set())
    assert remap['A'] == remap['B']
    assert ('A', 'B') not in collapsed and ('B', 'A') not in collapsed
    rep = remap['A']
    assert len(groups) == 1
    assert groups[rep] == ['A', 'B']
    assert close_pairs == {}


def test_collapse_mz_twin_groups_prefers_existing_fam_member(ep):
    """The representative should be whichever twin already survived the
    original purge_mz_twins.py step (i.e. is already in the working .fam),
    not an arbitrary group member."""
    king = {('TWIN_NOT_IN_FAM', 'TWIN_IN_FAM'): {'kinship': 0.472, 'ibs0': 0.000}}
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, fam_ids={'TWIN_IN_FAM'})
    assert remap['TWIN_NOT_IN_FAM'] == 'TWIN_IN_FAM'
    assert remap['TWIN_IN_FAM'] == 'TWIN_IN_FAM'


def test_collapse_mz_twin_groups_transitive(ep):
    """A=B and B=C must merge into one 3-member group."""
    king = {
        ('A', 'B'): {'kinship': 0.40, 'ibs0': 0.000},
        ('B', 'C'): {'kinship': 0.38, 'ibs0': 0.000},
    }
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(king, set())
    assert remap['A'] == remap['B'] == remap['C']
    rep = remap['A']
    assert sorted(groups[rep]) == ['A', 'B', 'C']


def test_collapse_mz_twin_groups_no_twins_unchanged(ep):
    king = {('A', 'B'): {'kinship': 0.13, 'ibs0': 0.01}}
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(king, set())
    assert collapsed == king
    assert groups == {}
    assert close_pairs == {}


def test_collapse_mz_twin_groups_eliminates_redundant_rediscovery(ep):
    """
    Regression guard: this exact scenario produced 2.3 million redundant
    "excluded MZ-twin pseudo-pair" log lines in a real run, because the old
    per-founder pairwise check rediscovered the same MZ-twin fact once per
    founder sharing both individuals as 2nd-degree relatives. After
    collapsing upstream, the twin-duplicate must never appear as a separate
    candidate at all.
    """
    king = {
        ('FOUNDER', 'C0000844_b2'): {'kinship': 0.175, 'ibs0': 0.000},
        ('FOUNDER', 'C0001317_b2'): {'kinship': 0.167, 'ibs0': 0.000},
        ('FOUNDER', 'OTHER'): {'kinship': 0.16, 'ibs0': 0.000},
        ('C0000844_b2', 'C0001317_b2'): {'kinship': 0.472, 'ibs0': 0.000},
        ('C0000844_b2', 'OTHER'): {'kinship': 0.26, 'ibs0': 0.005},
    }
    fam_ids = {'FOUNDER'}
    collapsed_king, remap, mz_groups, close_pairs = ep.collapse_mz_twin_groups_in_king(king, fam_ids)
    king_index = ep.build_king_index(collapsed_king)
    all_ids = {remap.get(s, s) for s in
               ['FOUNDER', 'C0000844_b2', 'C0001317_b2', 'OTHER']}

    candidates = ep.find_grandparent_candidates(
        'FOUNDER', collapsed_king, king_index, all_ids)
    all_members = {c['g1'] for c in candidates} | {c['g2'] for c in candidates}
    assert 'C0001317_b2' not in all_members, \
        "Collapsed twin ID must never appear as a separate candidate"


# ---------------------------------------------------------------------------
# hap-IBD coverage discrimination: true duplicate vs. close-but-distinct
# (inbred/backcrossed germplasm can elevate kinship into the nominal MZ
# range between genuinely DIFFERENT individuals — KING_MZ_MIN alone cannot
# tell the two apart, but hap-IBD genome coverage can)
# ---------------------------------------------------------------------------

def _write_ibd_gz(tmp_path, rows, name="test.ibd.gz"):
    """rows: list of (id1, id2, chrom, start, end, lod) -> hap-IBD .ibd.gz file."""
    import gzip
    path = str(tmp_path / name)
    with gzip.open(path, 'wt') as f:
        for id1, id2, chrom, start, end, lod in rows:
            f.write(f"{id1}\t1\t{id2}\t1\t{chrom}\t{start}\t{end}\t{lod}\n")
    return path


def test_load_ibd_segments_raw_basic(ep, tmp_path):
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 1000, 2000, 5.0),
        ('A', 'B', '2', 1, 500, 4.0),
        ('A', 'C', '1', 1, 100, 4.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    assert sorted(idx[('A', 'B')]) == [('1', 1000, 2000), ('2', 1, 500)]
    assert idx[('A', 'C')] == [('1', 1, 100)]


def test_load_ibd_segments_raw_filters_low_lod(ep, tmp_path):
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 1000, 2000, 2.0),   # below default min_lod=3.0
        ('A', 'B', '2', 1, 500, 4.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    assert idx[('A', 'B')] == [('2', 1, 500)]


def test_load_ibd_segments_raw_missing_file_returns_empty(ep):
    idx = ep.load_ibd_segments_raw('/nonexistent/path.ibd.gz')
    assert idx == {}


def test_merged_ibd_bp_merges_overlaps(ep):
    segments = [('1', 100, 200), ('1', 150, 300), ('2', 0, 50)]
    # chr1: [100,200) U [150,300) = [100,300) = 200bp ; chr2: 50bp
    assert ep._merged_ibd_bp(segments) == 250


def test_assess_duplicate_vs_relative_likely_duplicate(ep, tmp_path):
    """Pair shares ~all of A's testable genome with B -> true duplicate."""
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 0, 1_000_000, 10.0),
        ('A', 'B', '2', 0, 1_000_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    proxy = ep.build_ibd_genome_proxy(idx)
    classification, stats = ep.assess_duplicate_vs_relative('A', 'B', idx, proxy)
    assert classification == 'LIKELY_DUPLICATE'
    assert stats['coverage_frac'] == pytest.approx(1.0)


def test_assess_duplicate_vs_relative_close_but_distinct(ep, tmp_path):
    """A has substantial IBD elsewhere in the index, but only a small
    fraction of it is shared with B -> real recombination breaks, not a
    true duplicate, despite MZ-level KING kinship."""
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 0, 100_000, 10.0),       # small shared segment
        ('A', 'OTHER', '1', 0, 1_000_000, 10.0),  # A's broader testable genome
        ('A', 'OTHER', '2', 0, 1_000_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    proxy = ep.build_ibd_genome_proxy(idx)
    classification, stats = ep.assess_duplicate_vs_relative('A', 'B', idx, proxy)
    assert classification == 'CLOSE_BUT_DISTINCT'
    assert stats['coverage_frac'] < 0.95


def test_assess_duplicate_vs_relative_insufficient_data(ep, tmp_path):
    path = _write_ibd_gz(tmp_path, [('X', 'Y', '1', 0, 1000, 10.0)])
    idx = ep.load_ibd_segments_raw(path)
    proxy = ep.build_ibd_genome_proxy(idx)
    classification, stats = ep.assess_duplicate_vs_relative('A', 'B', idx, proxy)
    assert classification == 'INSUFFICIENT_DATA'
    assert stats is None


def test_build_ibd_genome_proxy_aggregates_per_individual(ep, tmp_path):
    """Proxy must sum/merge an individual's IBD across ALL its pairs, built
    once rather than rescanned per pair (the original O(M*P) hang)."""
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 0, 100_000, 10.0),
        ('A', 'C', '2', 0, 200_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    proxy = ep.build_ibd_genome_proxy(idx)
    assert proxy['A'] == 100_000 + 200_000
    assert proxy['B'] == 100_000
    assert proxy['C'] == 200_000


def test_collapse_mz_twin_groups_ibd_scales_with_many_pairs(ep, tmp_path):
    """
    Regression guard: assess_duplicate_vs_relative() used to rescan the
    ENTIRE ibd_segments_index per MZ-level pair to compute the genome-proxy
    denominator (O(P) per call), so a cohort with many MZ-level pairs AND a
    large IBD segment table degenerated to O(M*P) — this hung silently for
    hours on a real ~3577-sample run. build_ibd_genome_proxy() now
    precomputes the per-individual proxy ONCE. With 200 unrelated background
    pairs plus 50 MZ-level pairs to assess, this must complete near-
    instantly, not scale with pairs-assessed x index-size.
    """
    rows = []
    king = {}
    # 200 background pairs inflate the IBD index without being MZ-level.
    for i in range(200):
        rows.append((f"BG{i}A", f"BG{i}B", '1', 0, 50_000, 10.0))
    # 50 MZ-level kinship pairs, each needing assess_duplicate_vs_relative().
    for i in range(50):
        a, b = f"P{i}A", f"P{i}B"
        king[(a, b)] = {'kinship': 0.40, 'ibs0': 0.000}
        rows.append((a, b, '1', 0, 100_000, 10.0))
        rows.append((a, f"OTHER{i}", '1', 0, 1_000_000, 10.0))
    path = _write_ibd_gz(tmp_path, rows)
    idx = ep.load_ibd_segments_raw(path)

    t_start = time.time()
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)
    elapsed = time.time() - t_start
    assert elapsed < 2.0, f"Collapse with IBD evidence took {elapsed:.2f}s — possible O(M*P) regression"
    assert len(close_pairs) == 50


def test_collapse_mz_twin_groups_ibd_confirms_duplicate(ep, tmp_path):
    """With IBD evidence of near-complete coverage, an MZ-level kinship pair
    IS collapsed, same as the no-IBD-evidence default."""
    king = {('A', 'B'): {'kinship': 0.472, 'ibs0': 0.000}}
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 0, 1_000_000, 10.0),
        ('A', 'B', '2', 0, 1_000_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)
    assert remap['A'] == remap['B']
    assert close_pairs == {}


def test_collapse_mz_twin_groups_ibd_rejects_close_but_distinct(ep, tmp_path):
    """With IBD evidence showing incomplete coverage, an MZ-level kinship
    pair must NOT be collapsed — it's reported as close-but-distinct
    instead, so a real candidate from inbred germplasm isn't erased."""
    king = {('A', 'B'): {'kinship': 0.40, 'ibs0': 0.000}}
    path = _write_ibd_gz(tmp_path, [
        ('A', 'B', '1', 0, 100_000, 10.0),
        ('A', 'OTHER', '1', 0, 1_000_000, 10.0),
        ('A', 'OTHER', '2', 0, 1_000_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)
    assert remap.get('A', 'A') == 'A'
    assert remap.get('B', 'B') == 'B'
    assert groups == {}
    assert ('A', 'B') in close_pairs
    # The pair must survive unchanged in the collapsed king (not erased)
    assert ('A', 'B') in collapsed


# ---------------------------------------------------------------------------
# write_confirmed_duplicates
# ---------------------------------------------------------------------------

def test_write_confirmed_duplicates_writes_all_pairwise_combinations(ep, tmp_path):
    """A 3-member confirmed MZ group must expand to all 3 pairwise combos --
    calibrate_error_rate.py treats every row as an independently confirmed
    duplicate pair."""
    mz_groups = {'A': ['A', 'B', 'C']}
    path = tmp_path / "confirmed.tsv"
    ep.write_confirmed_duplicates(str(path), mz_groups)
    lines = path.read_text().splitlines()
    assert lines[0] == 'ID1\tID2'
    pairs = {tuple(l.split('\t')) for l in lines[1:]}
    assert pairs == {('A', 'B'), ('A', 'C'), ('B', 'C')}


def test_write_confirmed_duplicates_empty_groups(ep, tmp_path):
    path = tmp_path / "confirmed.tsv"
    ep.write_confirmed_duplicates(str(path), {})
    lines = path.read_text().splitlines()
    assert lines == ['ID1\tID2']


def test_find_grandparent_candidates_reports_close_inbred_not_skipped(ep, tmp_path):
    """The user's core ask: don't silently drop a candidate just because
    kinship crossed the approximate MZ threshold — if hap-IBD shows the
    pair is genuinely distinct, report it (flagged) rather than hiding it."""
    king = {
        ('FOUNDER', 'G1'): {'kinship': 0.175, 'ibs0': 0.000},
        ('FOUNDER', 'G2'): {'kinship': 0.167, 'ibs0': 0.000},
        ('G1', 'G2'): {'kinship': 0.40, 'ibs0': 0.000},  # MZ-level kinship
    }
    king_index = ep.build_king_index(king)
    path = _write_ibd_gz(tmp_path, [
        ('G1', 'G2', '1', 0, 100_000, 10.0),
        ('G1', 'OTHER', '1', 0, 1_000_000, 10.0),
        ('G1', 'OTHER', '2', 0, 1_000_000, 10.0),
    ])
    idx = ep.load_ibd_segments_raw(path)
    _, _, _, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'G1', 'G2'},
        close_but_distinct_pairs=close_pairs)
    assert len(candidates) == 1
    assert candidates[0]['g1g2_relation'] == 'CLOSE_INBRED'
    assert candidates[0]['g1g2_ibd_coverage'] is not None


# ---------------------------------------------------------------------------
# Fragmented-duplicate detection: statistical phasing can fragment a TRUE
# duplicate's whole-genome IBD into many short segments, deflating measured
# coverage below the MZ-confirmation threshold. A self-calibrated reference
# (segment-length pattern of CONFIRMED duplicates in the same run) should
# catch this and flag it distinctly rather than calling it a genuinely
# distinct relative.
# ---------------------------------------------------------------------------

def _build_fragmentation_scenario(tmp_path, n_confirmed_dups=5):
    """
    Shared scenario: N confirmed duplicate pairs (each with 2 short, 50kb
    segments -> mean_segment_Mb=0.05, coverage=1.0 since each individual
    appears in only that one pair), plus:
      - (X, Y): kinship 0.40, segments matching the SAME 0.05 Mb fragmentation
        pattern but with low coverage (X has a much bigger genome-proxy via
        X-OTHER) -> should be reclassified LIKELY_DUPLICATE_FRAGMENTED.
      - (P, Q): kinship 0.40, ONE large 5 Mb segment (real recombination
        block size, not phasing fragmentation), also low coverage via
        P-OTHER2 -> should remain CLOSE_BUT_DISTINCT.
    Returns (king, ibd_segments_index).
    """
    king = {}
    rows = []
    for i in range(n_confirmed_dups):
        a, b = f"D{i}a", f"D{i}b"
        king[(a, b)] = {'kinship': 0.472, 'ibs0': 0.000}
        rows.append((a, b, '1', 0, 50_000, 10.0))
        rows.append((a, b, '1', 100_000, 150_000, 10.0))

    king[('X', 'Y')] = {'kinship': 0.40, 'ibs0': 0.000}
    rows.append(('X', 'Y', '1', 0, 50_000, 10.0))
    rows.append(('X', 'Y', '1', 100_000, 150_000, 10.0))
    rows.append(('X', 'OTHER', '2', 0, 2_000_000, 10.0))

    king[('P', 'Q')] = {'kinship': 0.40, 'ibs0': 0.000}
    rows.append(('P', 'Q', '1', 0, 5_000_000, 10.0))
    rows.append(('P', 'OTHER2', '2', 0, 50_000_000, 10.0))

    path = _write_ibd_gz(tmp_path, rows)
    return king, path


def test_collapse_mz_twin_groups_detects_fragmented_duplicate(ep, tmp_path):
    king, ibd_path = _build_fragmentation_scenario(tmp_path)
    idx = ep.load_ibd_segments_raw(ibd_path)
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)

    # X/Y: fragmentation pattern matches the confirmed-duplicate noise floor
    # -> flagged distinctly, NOT collapsed.
    assert ('X', 'Y') in close_pairs
    assert close_pairs[('X', 'Y')]['classification'] == 'LIKELY_DUPLICATE_FRAGMENTED'
    assert remap.get('X', 'X') == 'X' and remap.get('Y', 'Y') == 'Y'
    # ibs0 (phasing-independent KING signal) and the calibrated reference
    # must be attached for downstream sanity-checking, not left implicit.
    assert close_pairs[('X', 'Y')]['ibs0'] == pytest.approx(0.000)
    assert close_pairs[('X', 'Y')]['fragmentation_reference_mb'] == pytest.approx(0.05)

    # P/Q: large real-recombination-sized segment -> genuinely distinct,
    # ordinary CLOSE_BUT_DISTINCT, NOT collapsed.
    assert ('P', 'Q') in close_pairs
    assert close_pairs[('P', 'Q')]['classification'] == 'CLOSE_BUT_DISTINCT'
    assert remap.get('P', 'P') == 'P' and remap.get('Q', 'Q') == 'Q'
    assert close_pairs[('P', 'Q')]['ibs0'] == pytest.approx(0.000)
    # Reference is attached even on the CLOSE_BUT_DISTINCT bucket (same run).
    assert close_pairs[('P', 'Q')]['fragmentation_reference_mb'] == pytest.approx(0.05)

    # The 5 confirmed duplicate pairs ARE collapsed.
    for i in range(5):
        assert remap[f"D{i}a"] == remap[f"D{i}b"]


def test_collapse_mz_twin_groups_skips_fragmentation_check_below_minimum(ep, tmp_path):
    """With fewer than MIN_DUPLICATES_FOR_FRAGMENTATION_REFERENCE confirmed
    duplicates, there's no reliable reference to compare against — the
    fragmented-looking pair must NOT be reclassified, staying a plain
    CLOSE_BUT_DISTINCT (conservative: don't guess without enough data)."""
    king, ibd_path = _build_fragmentation_scenario(tmp_path, n_confirmed_dups=2)
    idx = ep.load_ibd_segments_raw(ibd_path)
    collapsed, remap, groups, close_pairs = ep.collapse_mz_twin_groups_in_king(
        king, set(), ibd_segments_index=idx)
    assert close_pairs[('X', 'Y')]['classification'] == 'CLOSE_BUT_DISTINCT'


def test_find_grandparent_candidates_reports_close_inbred_possible_dup(ep, tmp_path):
    """A CLOSE_INBRED_POSSIBLE_DUP candidate must be surfaced (not hidden)
    with a stronger confidence penalty than a confirmed-distinct CLOSE_INBRED
    candidate, since it may actually be one individual counted twice."""
    king, ibd_path = _build_fragmentation_scenario(tmp_path)
    king[('FOUNDER', 'X')] = {'kinship': 0.175, 'ibs0': 0.000}
    king[('FOUNDER', 'Y')] = {'kinship': 0.167, 'ibs0': 0.000}
    idx = ep.load_ibd_segments_raw(ibd_path)
    _, _, _, close_pairs = ep.collapse_mz_twin_groups_in_king(king, set(), ibd_segments_index=idx)

    king_index = ep.build_king_index(king)
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'X', 'Y'},
        close_but_distinct_pairs=close_pairs)
    assert len(candidates) == 1
    c = candidates[0]
    assert c['g1g2_relation'] == 'CLOSE_INBRED_POSSIBLE_DUP'
    expected_base = (0.175 + 0.167) / 2 - abs(0.175 - 0.167) * 0.5
    assert c['confidence'] == pytest.approx(expected_base - 0.15)


# ---------------------------------------------------------------------------
# MZ-twin alternatives reported, not silently hidden
# ---------------------------------------------------------------------------

def test_grandparent_candidate_records_mz_alternatives(ep):
    """
    True MZ twins are genetically indistinguishable, so no genetic test can
    validate which twin is the "real" grandparent — the representative
    chosen by collapse_mz_twin_groups_in_king() is a bookkeeping choice, not
    a verified identity. The alternative twin ID(s) must be recorded on the
    candidate so they can be surfaced to the user rather than silently lost.
    """
    king = {
        ('FOUNDER', 'C0000844_b2'): {'kinship': 0.175, 'ibs0': 0.000},
        ('FOUNDER', 'OTHER'): {'kinship': 0.16, 'ibs0': 0.000},
        ('C0000844_b2', 'OTHER'): {'kinship': 0.26, 'ibs0': 0.005},
    }
    king_index = ep.build_king_index(king)
    mz_groups = {'C0000844_b2': ['C0000844_b2', 'C0001317_b2']}
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'C0000844_b2', 'OTHER'},
        mz_groups=mz_groups)
    assert len(candidates) == 1
    c = candidates[0]
    # C0000844_b2 could be g1 or g2 depending on iteration order
    alts = c['g1_alternatives'] if c['g1'] == 'C0000844_b2' else c['g2_alternatives']
    assert alts == ['C0001317_b2']


def test_grandparent_candidate_no_alternatives_when_no_mz_group(ep):
    king = {
        ('FOUNDER', 'G1'): {'kinship': 0.175, 'ibs0': 0.000},
        ('FOUNDER', 'G2'): {'kinship': 0.16, 'ibs0': 0.000},
        ('G1', 'G2'): {'kinship': 0.26, 'ibs0': 0.005},
    }
    king_index = ep.build_king_index(king)
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'G1', 'G2'}, mz_groups={})
    assert candidates[0]['g1_alternatives'] == []
    assert candidates[0]['g2_alternatives'] == []


def test_offspring_candidate_records_mz_alternatives(ep):
    king = {('LEAF', 'CANDIDATE'): {'kinship': 0.26, 'ibs0': 0.005}}
    king_index = ep.build_king_index(king)
    mz_groups = {'CANDIDATE': ['CANDIDATE', 'CANDIDATE_TWIN']}
    results = ep.find_offspring_candidates(
        'LEAF', set(), king, king_index, {'LEAF', 'CANDIDATE'},
        mz_groups=mz_groups)
    assert len(results) == 1
    assert results[0]['candidate_child_alternatives'] == ['CANDIDATE_TWIN']


def test_write_vcfped_candidates_documents_relation_and_coverage(ep, tmp_path):
    """
    Regression guard: the candidate triple note previously carried only
    confidence=X, so a CLOSE_INBRED / CLOSE_INBRED_POSSIBLE_DUP candidate
    (elevated kinship, hap-IBD-evidenced) was indistinguishable from an
    ordinary PO candidate in this file — the fragmentation/coverage evidence
    visible in the extension report had no counterpart here.
    """
    up_candidates = [{
        'founder': 'FOUNDER', 'g1': 'G1', 'g2': 'G2', 'confidence': 0.117,
        'g1g2_relation': 'CLOSE_INBRED_POSSIBLE_DUP',
        'g1g2_ibd_coverage': {'coverage_frac': 0.048, 'n_segments': 2,
                               'mean_segment_Mb': 0.1, 'total_ibd_Mb': 0.2,
                               'classification': 'LIKELY_DUPLICATE_FRAGMENTED',
                               'ibs0': 0.0021, 'fragmentation_reference_mb': 0.08},
        'g1_alternatives': [], 'g2_alternatives': [],
    }]
    out_path = str(tmp_path / "candidates.txt")
    ep.write_vcfped_candidates(out_path, up_candidates, [])
    content = open(out_path).read()
    assert 'relation=CLOSE_INBRED_POSSIBLE_DUP' in content
    assert 'ibd_coverage=4.8%' in content
    assert 'ibs0=0.0021' in content
    assert 'mean_seg_Mb=0.1' in content


def test_write_vcfped_candidates_documents_mz_alternatives(ep, tmp_path):
    """Alternatives must be documented in the candidate file's comments, NOT
    added to the sample subset (testing a genetic duplicate adds no
    information — only non-genetic records can resolve twin identity)."""
    up_candidates = [{
        'founder': 'FOUNDER', 'g1': 'G1', 'g2': 'G2', 'confidence': 0.5,
        'g1g2_relation': 'PO', 'g1g2_ibd_coverage': None,
        'g1_alternatives': ['G1_TWIN'], 'g2_alternatives': [],
    }]
    out_path = str(tmp_path / "candidates.txt")
    ep.write_vcfped_candidates(out_path, up_candidates, [])
    content = open(out_path).read()
    assert 'G1_TWIN' in content
    assert 'MZ-TWIN ALTERNATIVES' in content
    # Alternative must NOT be in the actual sample subset list
    sample_lines = {l for l in content.splitlines()
                    if l and not l.startswith('#')}
    assert 'G1_TWIN' not in sample_lines


def test_grandparent_candidates_excludes_mz_twin_pseudo_pairs(ep, capsys):
    """
    Regression guard: found via a real run. Founder had a "candidate
    grandparent pair" (G1, G2) where G1<->G2 kinship was 0.472 — above even
    the MZ-twin threshold (0.354). Confirmed via step1.pair: G1 and G2 ARE
    MZ twins of each other. This is not a valid two-person grandparent pair
    (it's one individual counted twice) but was previously mislabelled
    'PO' because MZ-level kinship also clears the KING_1ST_MIN threshold
    used for the PO classification. Must be excluded, not returned.

    This test calls find_grandparent_candidates() directly with RAW
    (uncollapsed) KING data, exercising the function's internal safety net
    (the primary fix is collapse_mz_twin_groups_in_king(), applied upstream
    in main() — see test_collapse_mz_twin_groups_eliminates_redundant_rediscovery
    for that path). The safety net is intentionally silent (no per-occurrence
    print): with collapsing applied, millions of founders sharing the same
    twin pair would otherwise each reprint the same fact redundantly (a real
    run logged 2.3 million such lines before this was fixed).
    """
    king = {
        ('FOUNDER', 'G1'): {'kinship': 0.175, 'ibs0': 0.000},
        ('FOUNDER', 'G2'): {'kinship': 0.167, 'ibs0': 0.000},
        ('G1', 'G2'): {'kinship': 0.472, 'ibs0': 0.000},  # MZ-twin-level kinship
    }
    king_index = ep.build_king_index(king)
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'G1', 'G2'})
    assert candidates == [], \
        "MZ-twin pseudo-pair must be excluded, not returned as a grandparent candidate"


def test_grandparent_candidates_normal_po_pair_still_works(ep):
    """Sanity check: a genuine (non-MZ) PO-related grandparent pair must
    still be returned — the MZ exclusion must not be overly broad."""
    king = {
        ('FOUNDER', 'G1'): {'kinship': 0.13, 'ibs0': 0.01},
        ('FOUNDER', 'G2'): {'kinship': 0.12, 'ibs0': 0.01},
        ('G1', 'G2'): {'kinship': 0.26, 'ibs0': 0.005},  # normal PO, not MZ
    }
    king_index = ep.build_king_index(king)
    candidates = ep.find_grandparent_candidates(
        'FOUNDER', king, king_index, {'FOUNDER', 'G1', 'G2'})
    assert len(candidates) == 1
    assert candidates[0]['g1g2_relation'] == 'PO'


def test_grandparent_pairing_capped_for_dense_cohorts(ep):
    """
    Regression guard: in a closely related cohort, a single founder can have
    hundreds of 2nd-degree relatives, making the O(M^2) pairing step blow up.
    Must not hang/take excessively long even with 300 2nd-degree relatives.
    """
    king = {}
    king_index = {}
    founder = 'FOUNDER'
    relatives = [f"REL{i}" for i in range(300)]
    adj = []
    for r in relatives:
        adj.append((r, 0.13, 0.01))  # all 2nd-degree to founder
    king_index[founder] = adj
    for r in relatives:
        king_index[r] = [(founder, 0.13, 0.01)]
    # A few of the relatives are 1st-degree to each other (candidate GP pairs)
    for i in range(0, 20, 2):
        king[(relatives[i], relatives[i + 1])] = {'kinship': 0.26, 'ibs0': 0.005}

    t0 = time.time()
    result = ep.find_grandparent_candidates(founder, king, king_index, set(relatives) | {founder})
    elapsed = time.time() - t0
    assert elapsed < 5.0, "Uncapped O(M^2) pairing took too long for a dense cohort"


# ---------------------------------------------------------------------------
# analyse_structure
# ---------------------------------------------------------------------------

def test_analyse_structure_founders_and_leaves(ep):
    fam = {
        'GP1': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'},
        'GP2': {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'},
        'CHILD': {'fid': '0', 'pat': 'GP1', 'mat': 'GP2', 'sex': '0'},
    }
    founders, leaves, children_of = ep.analyse_structure(fam)
    assert founders == {'GP1', 'GP2'}
    assert leaves == {'CHILD'}  # CHILD has no children of its own
    assert children_of['GP1'] == {'CHILD'}


# ---------------------------------------------------------------------------
# find_offspring_candidates: founder/leaf bookkeeping
# ---------------------------------------------------------------------------

def test_offspring_candidates_excludes_already_assigned_children(ep):
    king = {('LEAF', 'CANDIDATE'): {'kinship': 0.26, 'ibs0': 0.005}}
    king_index = ep.build_king_index(king)
    all_children_already_assigned = {'CANDIDATE'}  # CANDIDATE already has parents
    results = ep.find_offspring_candidates(
        'LEAF', all_children_already_assigned, king, king_index, {'LEAF', 'CANDIDATE'})
    assert results == []


def test_offspring_candidates_finds_unassigned_po_signal(ep):
    king = {('LEAF', 'CANDIDATE'): {'kinship': 0.26, 'ibs0': 0.005}}
    king_index = ep.build_king_index(king)
    results = ep.find_offspring_candidates(
        'LEAF', set(), king, king_index, {'LEAF', 'CANDIDATE'})
    assert len(results) == 1
    assert results[0]['candidate_child'] == 'CANDIDATE'


def test_offspring_candidates_caps_per_leaf(ep):
    """Regression guard: in a densely related cohort, a leaf can have
    hundreds of PO-signal candidates; each triggers an O(degree) inner
    lookup. Must be capped to bound worst-case runtime."""
    king = {}
    king_index = {'LEAF': []}
    candidates = [f"C{i}" for i in range(200)]
    for c in candidates:
        king_index['LEAF'].append((c, 0.20 + (hash(c) % 100) / 10000, 0.01))
        king_index[c] = []
    results = ep.find_offspring_candidates(
        'LEAF', set(), king, king_index, set(candidates) | {'LEAF'})
    assert len(results) <= 100, "Per-leaf candidate count should be capped"


# ---------------------------------------------------------------------------
# write_vcfped_candidates: full-pool regression (the truncation bug)
# ---------------------------------------------------------------------------

def test_write_vcfped_candidates_includes_full_pool_not_truncated(ep, tmp_path):
    """
    Regression guard: this function must write the FULL candidate pool it
    is given. The bug was upstream (callers slicing the list before passing
    it in), but this test locks in that the function itself performs no
    hidden truncation.
    """
    up_candidates = [
        {'founder': f'F{i}', 'g1': f'G1_{i}', 'g2': f'G2_{i}', 'confidence': 0.5}
        for i in range(100)
    ]
    out_path = str(tmp_path / "candidates.txt")
    ep.write_vcfped_candidates(out_path, up_candidates, [])
    content = open(out_path).read()
    samples_in_file = {l for l in content.splitlines() if l and not l.startswith('#')}
    # All 100 founders + their 200 grandparent candidates should be present
    assert len(samples_in_file) == 300, \
        f"Expected 300 unique samples from 100 founders, got {len(samples_in_file)}"


# ---------------------------------------------------------------------------
# write_beagle_check_script: phased-VCF-path and map= regressions
# ---------------------------------------------------------------------------

def test_beagle_check_script_explicit_phased_vcf_used(ep, tmp_path):
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'my_out_prefix', 'fam.fam',
                                 phased_vcf='actual_phased.vcf.gz')
    content = open(out_path).read()
    assert 'PHASED_VCF="actual_phased.vcf.gz"' in content


def test_beagle_check_script_guess_is_clearly_labeled(ep, tmp_path):
    """
    Regression guard: when no explicit --phased-vcf is given, the script
    used to silently guess <out_prefix>_phased.vcf.gz, which is WRONG
    whenever extend_pedigree.py's own output prefix differs from
    prepare_plink_beagle.py's -o prefix (the common case). The guess must
    now be clearly labelled so users know to check it.
    """
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'my_out_prefix', 'fam.fam')
    content = open(out_path).read()
    assert 'GUESSED' in content


def test_beagle_check_script_includes_mandatory_map_argument(ep, tmp_path):
    """Regression guard: hap-IBD requires map= (mandatory), distinct from
    the prepare_plink_beagle.py fix — this is a different generated script."""
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'my_out_prefix', 'fam.fam',
                                 phased_vcf='phased.vcf.gz',
                                 genetic_map='my.map')
    content = open(out_path).read()
    assert 'MAP="my.map"' in content
    assert re.search(r'map="\$\{MAP\}"', content)


def test_beagle_check_script_map_guessed_from_phased_vcf(ep, tmp_path):
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'my_out_prefix', 'fam.fam',
                                 phased_vcf='step3_phased.vcf.gz')
    content = open(out_path).read()
    assert 'step3_synthetic.map' in content


def test_beagle_check_script_uses_configurable_hapibd(ep, tmp_path):
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'prefix', 'fam.fam',
                                 hapibd_exe='hap-ibd')
    content = open(out_path).read()
    assert 'HAPIBD="hap-ibd"' in content
    assert 'java' not in content.split('HAPIBD=')[0]  # no leftover java -jar style


def test_beagle_check_script_valid_bash_syntax(ep, tmp_path):
    import subprocess
    out_path = str(tmp_path / "check.sh")
    ep.write_beagle_check_script(out_path, [], [], 'prefix', 'fam.fam',
                                 phased_vcf='x.vcf.gz', genetic_map='x.map')
    result = subprocess.run(['bash', '-n', out_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# load_ibd_summary + ibd_relationship_hint
# ---------------------------------------------------------------------------

def test_load_ibd_summary_filters_low_lod(ep, tmp_path):
    f = tmp_path / "test.ibd"
    f.write_text(
        "ID1\th1\tID2\th2\tchrom\t100\t200000\t5.0\n"
        "ID1\th1\tID2\th2\tchrom\t300000\t400000\t1.0\n"  # below default min_lod=3.0
    )
    # Rewrite with tab-separated proper format
    f.write_text(
        "ID1\t1\tID2\t1\t1\t100\t200000\t5.0\n"
        "ID1\t1\tID2\t1\t1\t300000\t400000\t1.0\n"
    )
    summary = ep.load_ibd_summary(str(f), min_lod=3.0)
    key = tuple(sorted(['ID1', 'ID2']))
    assert summary[key]['n_segments'] == 1  # only the LOD=5.0 segment counted


def test_ibd_relationship_hint_classification(ep):
    long_seg = {'mean_segment_Mb': 50, 'total_Mb': 800, 'n_segments': 20}
    short_seg = {'mean_segment_Mb': 10, 'total_Mb': 800, 'n_segments': 80}
    long_hint = ep.ibd_relationship_hint(long_seg)
    short_hint = ep.ibd_relationship_hint(short_seg)
    assert 'HALF-SIBLING' in long_hint
    assert 'GRANDPARENT' in short_hint


def test_ibd_relationship_hint_no_data(ep):
    assert 'UNKNOWN' in ep.ibd_relationship_hint(None)


# ---------------------------------------------------------------------------
# End-to-end: full CLI run with the offspring-feedback bookkeeping
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_extend_inputs(tmp_path):
    """3 founders, all unrelated to each other except via KING signal to a
    single shared candidate offspring."""
    fam_path = tmp_path / "test.fam"
    fam_path.write_text(
        "F1\tFOUNDER1\t0\t0\t0\t-9\n"
        "F2\tFOUNDER2\t0\t0\t0\t-9\n"
    )
    king_path = tmp_path / "king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tFOUNDER1\tF2\tCANDIDATE\t1000\t0.3\t0.005\t0.26\n"
        "F1\tFOUNDER2\tF2\tCANDIDATE\t1000\t0.3\t0.005\t0.24\n"
    )
    return str(fam_path), str(tmp_path / "king")


def test_end_to_end_added_offspring_count_nonzero(synthetic_extend_inputs, tmp_path):
    """
    Regression guard: this exact scenario (with --max-iter > 1) used to
    print 'Added 0 offspring' despite finding valid candidates, because of
    the `child not in current_fam` bug. Now must correctly assign the
    unclaimed candidate.
    """
    fam_path, king_prefix = synthetic_extend_inputs
    out_prefix = str(tmp_path / "ext_out")
    result = run_script(EP_SCRIPT, [fam_path, f"{king_prefix}.kin0",
                                    '-o', out_prefix, '--max-iter', '2',
                                    '--direction', 'down'])
    assert result.returncode == 0, result.stderr
    assert 'Added' in result.stdout
    # Should NOT report "Added 0" for this clearly-resolvable scenario
    assert 'Added 0 offspring' not in result.stdout


@pytest.mark.parametrize('direction', ['up', 'down', 'both'])
def test_end_to_end_direction_with_max_iter_no_crash(synthetic_extend_inputs, tmp_path, direction):
    """
    Regression guard: discovered while building this test suite.
    `--direction up` or `--direction down` (restricting to one direction)
    combined with `--max-iter > 1` crashed with UnboundLocalError, because
    iter_up/iter_down were only initialized inside their respective
    `if args.direction in (...)` blocks, but the iteration-feedback logic
    unconditionally referenced both.
    """
    fam_path, king_prefix = synthetic_extend_inputs
    out_prefix = str(tmp_path / f"ext_{direction}")
    result = run_script(EP_SCRIPT, [fam_path, f"{king_prefix}.kin0",
                                    '-o', out_prefix, '--max-iter', '2',
                                    '--direction', direction])
    assert result.returncode == 0, result.stderr
    assert 'UnboundLocalError' not in result.stderr


def test_end_to_end_no_crash_when_one_fragmentation_bucket_is_empty(tmp_path):
    """
    Regression guard: the console summary computes a median IBS0 separately
    for the CLOSE_INBRED and CLOSE_INBRED_POSSIBLE_DUP buckets. When ALL
    flagged pairs land in one bucket (the other is empty — exactly what
    happened on a real run), the median of an empty list is None, and the
    summary print crashed with TypeError formatting None as a float.
    """
    fam_path = str(tmp_path / "test.fam")
    with open(fam_path, 'w') as f:
        f.write("FAM1\tFOUNDER\t0\t0\t0\t-9\n")

    king_path = str(tmp_path / "king.kin0")
    with open(king_path, 'w') as f:
        f.write("ID1\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n")
        for i in range(5):
            f.write(f"D{i}a\tD{i}b\t1000\t100\t0.000\t0.472\n")
        f.write("FOUNDER\tX\t1000\t100\t0.000\t0.175\n")
        f.write("FOUNDER\tY\t1000\t100\t0.000\t0.167\n")
        f.write("X\tY\t1000\t100\t0.001\t0.40\n")

    ibd_path = str(tmp_path / "test.ibd.gz")
    rows = []
    for i in range(5):
        a, b = f"D{i}a", f"D{i}b"
        rows.append((a, 1, b, 1, '1', 0, 50_000, 10.0))
        rows.append((a, 1, b, 1, '1', 100_000, 150_000, 10.0))
    rows.append(('X', 1, 'Y', 1, '1', 0, 50_000, 10.0))
    rows.append(('X', 1, 'Y', 1, '1', 100_000, 150_000, 10.0))
    rows.append(('X', 1, 'OTHER', 1, '2', 0, 2_000_000, 10.0))
    import gzip
    with gzip.open(ibd_path, 'wt') as f:
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')

    out_prefix = str(tmp_path / "out")
    result = run_script(EP_SCRIPT, [fam_path, king_path, '--ibd', ibd_path,
                                    '--direction', 'up', '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert 'TypeError' not in result.stderr
    assert 'Median IBS0' in result.stdout
    assert 'CLOSE_INBRED=n/a' in result.stdout


def test_end_to_end_header_logs_mz_tuning_args(synthetic_extend_inputs, tmp_path):
    """
    Regression guard: --min-mz-coverage / --fragmentation-factor were not
    printed in the run header, so a log file gave no way to tell which
    settings produced it — a real problem when comparing a parameter sweep
    across multiple log files after the fact.
    """
    fam_path, king_prefix = synthetic_extend_inputs
    out_prefix = str(tmp_path / "ext_out")
    result = run_script(EP_SCRIPT, [fam_path, f"{king_prefix}.kin0",
                                    '--min-mz-coverage', '0.90',
                                    '--fragmentation-factor', '1.2',
                                    '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert 'Min MZ coverage     : 0.9' in result.stdout
    assert 'Fragmentation factor: 1.2' in result.stdout


def test_end_to_end_outputs_created(synthetic_extend_inputs, tmp_path):
    fam_path, king_prefix = synthetic_extend_inputs
    out_prefix = str(tmp_path / "ext_out")
    result = run_script(EP_SCRIPT, [fam_path, f"{king_prefix}.kin0",
                                    '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{out_prefix}_extended.fam")
    assert os.path.exists(f"{out_prefix}_extension_report.txt")
    assert os.path.exists(f"{out_prefix}_vcfped_candidates.txt")
    assert os.path.exists(f"{out_prefix}_grandparent_check.sh")
    # No --ibd given -> collapse was kinship-only, no stricter than vcfped's
    # own verdict, so no confirmed_duplicates.tsv should be written.
    assert not os.path.exists(f"{out_prefix}_confirmed_duplicates.tsv")


def test_end_to_end_confirmed_duplicates_written_only_with_ibd(tmp_path):
    """
    _confirmed_duplicates.tsv should contain only the pairs whose hap-IBD
    coverage actually confirmed identity (D{i}a/D{i}b), not X/Y which has
    MZ-level kinship but incomplete coverage (CLOSE_BUT_DISTINCT) -- this
    is the stricter ground truth calibrate_error_rate.py's
    --confirmed-duplicates is meant to consume.
    """
    fam_path = str(tmp_path / "test.fam")
    with open(fam_path, 'w') as f:
        f.write("FAM1\tFOUNDER\t0\t0\t0\t-9\n")

    king_path = str(tmp_path / "king.kin0")
    with open(king_path, 'w') as f:
        f.write("ID1\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n")
        for i in range(5):
            f.write(f"D{i}a\tD{i}b\t1000\t100\t0.000\t0.472\n")
        f.write("FOUNDER\tX\t1000\t100\t0.000\t0.175\n")
        f.write("FOUNDER\tY\t1000\t100\t0.000\t0.167\n")
        f.write("X\tY\t1000\t100\t0.001\t0.40\n")

    ibd_path = str(tmp_path / "test.ibd.gz")
    rows = []
    for i in range(5):
        a, b = f"D{i}a", f"D{i}b"
        rows.append((a, 1, b, 1, '1', 0, 50_000, 10.0))
        rows.append((a, 1, b, 1, '1', 100_000, 150_000, 10.0))
    rows.append(('X', 1, 'Y', 1, '1', 0, 50_000, 10.0))
    rows.append(('X', 1, 'Y', 1, '1', 100_000, 150_000, 10.0))
    rows.append(('X', 1, 'OTHER', 1, '2', 0, 2_000_000, 10.0))
    import gzip
    with gzip.open(ibd_path, 'wt') as f:
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')

    out_prefix = str(tmp_path / "out")
    result = run_script(EP_SCRIPT, [fam_path, king_path, '--ibd', ibd_path,
                                    '--direction', 'up', '-o', out_prefix])
    assert result.returncode == 0, result.stderr

    confirmed_path = f"{out_prefix}_confirmed_duplicates.tsv"
    assert os.path.exists(confirmed_path)
    content = open(confirmed_path).read()
    for i in range(5):
        assert f"D{i}a\tD{i}b" in content
    # X/Y is CLOSE_BUT_DISTINCT (incomplete coverage) -- must NOT appear
    assert "X\tY" not in content
