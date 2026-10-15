#!/usr/bin/env python3
"""
Iteratively extend an identified pedigree upward (grandparents) and downward
(grandchildren) using PLINK2 KING kinship and optional hap-IBD segment data.

BACKGROUND
----------
After refine_pedigree.py produces a high-confidence parent assignment (.fam),
two extensions are biologically meaningful in a breeding pool:

  UPWARD   — finding grandparents of current founders
  DOWNWARD — finding offspring of current leaves (individuals without children)

Both use the same core KING evidence but differ in how candidates are identified
and how confidence is assessed.

UPWARD EXTENSION (grandparent search)
  Grandparents are 2nd-degree relatives of the focal individual
  (KINSHIP 0.0884–0.177).  To identify a grandparent PAIR:
    1. Find all 2nd-degree relatives of a founder F.
    2. Among those, find pairs (G1, G2) that are themselves 1st-degree relatives
       (i.e., G1 and G2 are parent-offspring: KINSHIP > 0.177, IBS0 < 0.05).
    3. Such (G1, G2) pairs are candidate grandparent pairs.
    If hap-IBD is available: grandparent-grandchild segments are shorter than
    half-sibling segments on average (extra recombination generation), further
    distinguishing grandparent from other 2nd-degree relatives.

DOWNWARD EXTENSION (grandchild / offspring search)
  Individuals not currently assigned as children of any known "leaf" individual
  (no identified children in current pedigree) with PO signal (KINSHIP > 0.177,
  IBS0 < 0.05) are candidate grandchildren.  To complete the trio:
    1. Identify the leaf as one parent.
    2. Search for a compatible second parent from KING pairs of the candidate child.

STRATEGY ASSESSMENT
-------------------
This utility assesses and recommends which downstream strategy to use for each
identified candidate extension:

  KING-only     Suitable when KINSHIP is unambiguous and you need fast results.
                Misses some cases; can't confirm without additional evidence.

  vcfped rerun  Best when candidates are numerous or KING is ambiguous.
                Run vcfped --no-pairwise on a subset VCF containing only the
                focal founder + candidate grandparents. Gets exact T1/T2 scores.

  BEAGLE/hap-IBD Best for distinguishing grandparent vs. half-sibling and
                identifying which member of a 2nd-degree pair is the true parent.
                Requires phased VCF (BEAGLE output from run_beagle.sh).

  Combined      KING screening → candidates → vcfped or BEAGLE confirmation.
                Recommended default for large cohorts.

INPUT
-----
  <fam_file>          — current pedigree (.fam format)
  <king_prefix>.kin0  — PLINK2 --make-king-table output
  [--ibd <prefix>_ibd.ibd.gz]  — hap-IBD output for segment analysis

OUTPUT
------
  <out_prefix>_extended.fam                — updated pedigree file
  <out_prefix>_extension_report.txt        — candidate list + strategy recommendations
  <out_prefix>_vcfped_candidates.txt       — candidate triples for vcfped rerun
  <out_prefix>_grandparent_check.sh        — shell script for BEAGLE IBD confirmation
  <out_prefix>_confirmed_duplicates.tsv    — every pair collapsed as a hap-IBD-coverage-
                                              confirmed true duplicate (only written when
                                              --ibd is given, i.e. when the collapse decision
                                              was coverage-validated, not kinship-only). A
                                              stricter ground truth for calibrate_error_rate.py's
                                              --confirmed-duplicates than vcfped's own pairwise
                                              "MZ twins" verdict, which inbreeding-induced
                                              allele-sharing inflation can contaminate with
                                              non-duplicate close relatives in this germplasm.

Usage:
  python extend_pedigree.py refined.fam king_results.kin0
  python extend_pedigree.py refined.fam king_results.kin0 --ibd prefix_ibd.ibd.gz
  python extend_pedigree.py refined.fam king_results.kin0 --direction up --max-iter 2
  python extend_pedigree.py refined.fam king_results.kin0 --strategy beagle
"""

import sys
import os
import csv
import gzip
import time
import argparse
import itertools
from collections import defaultdict

KING_2ND_MIN  = 0.0884   # lower bound of 2nd-degree
KING_1ST_MIN  = 0.177    # lower bound of 1st-degree
KING_MZ_MIN   = 0.354    # MZ threshold
KING_IBS0_PO  = 0.05     # IBS0 PO threshold

# Minimum number of CONFIRMED duplicate pairs (coverage-based) needed in a
# single run before their segment-length pattern is trusted as a reference
# "fragmentation noise floor" for flagging LIKELY_DUPLICATE_FRAGMENTED pairs.
# See assess_duplicate_vs_relative() / collapse_mz_twin_groups_in_king().
MIN_DUPLICATES_FOR_FRAGMENTATION_REFERENCE = 5


# ---------------------------------------------------------------------------
# File I/O helpers (shared with refine_pedigree.py)
# ---------------------------------------------------------------------------

def load_king(king_path, required_min_kinship=KING_2ND_MIN):
    """
    Parse PLINK2 --make-king-table output.

    PLINK2 silently drops pairs below its --king-table-filter threshold
    (default 0.0442).  extend_pedigree.py needs 2nd-degree pairs (0.0884)
    for grandparent search.  Warn if the file appears too heavily filtered.

    Use the _king_ext.kin0 output (--king-table-filter 0.044) from
    _run_plink.sh, NOT the _king_1st.kin0 (--king-table-filter 0.177).
    """
    king = {}
    all_ids = set()
    min_kinship_seen = float('inf')
    try:
        with open(king_path, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fieldnames = reader.fieldnames or []
            id1_col = next((c for c in fieldnames if 'ID1' in c and 'FID' not in c), None)
            id2_col = next((c for c in fieldnames if 'ID2' in c and 'FID' not in c), None)
            if not id1_col or not id2_col:
                sys.exit(f"Error: cannot find ID1/ID2 in {king_path}.\n"
                         f"  Expected PLINK2 --make-king-table output.\n"
                         f"  Use the _king_ext.kin0 file from _run_plink.sh "
                         f"(--king-table-filter 0.044), not the _king_1st.kin0.")
            for row in reader:
                id1 = row[id1_col].strip()
                id2 = row[id2_col].strip()
                try:
                    kinship = float(row.get('KINSHIP', 0))
                    ibs0    = float(row.get('IBS0', 0))
                    hethet  = float(row.get('HETHET', 0))
                except ValueError:
                    continue
                king[(id1, id2)] = {'kinship': kinship, 'ibs0': ibs0, 'hethet': hethet}
                all_ids.update([id1, id2])
                if kinship < min_kinship_seen:
                    min_kinship_seen = kinship
    except FileNotFoundError:
        sys.exit(f"Error: {king_path} not found.")

    if not king:
        sys.exit(f"Error: {king_path} is empty. "
                 f"Was it generated with --make-king-table --king-table-filter 0.044?")

    if min_kinship_seen > required_min_kinship * 1.5:
        print(f"  WARNING: lowest kinship in {king_path} is {min_kinship_seen:.4f}, "
              f"but grandparent search needs pairs down to {required_min_kinship:.4f}.")
        print(f"  Re-run PLINK2 with: --king-table-filter {required_min_kinship:.4f}")
        print(f"  Use the _king_ext.kin0 output from _run_plink.sh, not _king_1st.kin0.")

    return king, all_ids


def get_king(king, a, b):
    return king.get((a, b)) or king.get((b, a))


def load_fam(fam_path):
    fam = {}
    try:
        with open(fam_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6:
                    fam[p[1]] = {'fid': p[0], 'pat': p[2], 'mat': p[3], 'sex': p[4]}
    except FileNotFoundError:
        sys.exit(f"Error: {fam_path} not found.")
    return fam


def write_fam(path, fam):
    with open(path, 'w') as f:
        for iid, d in sorted(fam.items()):
            f.write(f"{d['fid']}\t{iid}\t{d['pat']}\t{d['mat']}\t{d['sex']}\t-9\n")
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Pedigree structure analysis
# ---------------------------------------------------------------------------

def analyse_structure(fam):
    """
    Return:
      founders: individuals with no known parents (pat='0' AND mat='0')
      leaves:   individuals with no known children in fam
      children_of: parent → set of children
    """
    children_of = defaultdict(set)
    for iid, d in fam.items():
        if d['pat'] != '0':
            children_of[d['pat']].add(iid)
        if d['mat'] != '0':
            children_of[d['mat']].add(iid)

    founders = {iid for iid, d in fam.items()
                if d['pat'] == '0' and d['mat'] == '0'}
    leaves   = {iid for iid in fam if iid not in children_of}

    return founders, leaves, dict(children_of)


# ---------------------------------------------------------------------------
# hap-IBD segment analysis
# ---------------------------------------------------------------------------

def load_ibd_summary(ibd_gz_path, min_lod=3.0):
    """
    Summarise hap-IBD output: return dict (id1, id2) → {total_bp, n_segments,
    mean_segment_bp, max_segment_bp}.
    """
    if not ibd_gz_path or not os.path.exists(ibd_gz_path):
        return {}

    pair_data = defaultdict(lambda: {'total_bp': 0, 'n_segs': 0,
                                      'segments': []})
    opener = gzip.open if ibd_gz_path.endswith('.gz') else open
    mode   = 'rt'
    try:
        with opener(ibd_gz_path, mode) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                id1, _, id2, _, _, start, end, lod = parts[:8]
                try:
                    start = int(start); end = int(end); lod = float(lod)
                except ValueError:
                    continue
                if lod < min_lod:
                    continue
                key = tuple(sorted([id1, id2]))
                seg_len = end - start
                pair_data[key]['total_bp']  += seg_len
                pair_data[key]['n_segs']    += 1
                pair_data[key]['segments'].append(seg_len)
    except Exception as e:
        print(f"  Warning: could not read {ibd_gz_path}: {e}")
        return {}

    result = {}
    for key, d in pair_data.items():
        segs = d['segments']
        result[key] = {
            'total_bp':       d['total_bp'],
            'total_Mb':       d['total_bp'] / 1e6,
            'n_segments':     d['n_segs'],
            'mean_segment_Mb': (sum(segs) / len(segs)) / 1e6 if segs else 0,
            'max_segment_Mb':  max(segs) / 1e6 if segs else 0,
        }
    return result


def get_ibd_summary(ibd, id1, id2):
    key = tuple(sorted([id1, id2]))
    return ibd.get(key)


def load_ibd_segments_raw(ibd_gz_path, min_lod=3.0):
    """
    Build a per-pair RAW segment index from hap-IBD output: dict
    (id1, id2) sorted-tuple -> list of (chrom, start, end).

    Unlike load_ibd_summary() (which only keeps aggregate segment LENGTHS),
    this retains chromosome + position so genome COVERAGE can be computed —
    needed to distinguish a true duplicate/MZ-twin (near-complete coverage)
    from a genuinely distinct but highly inbred close relative (elevated
    KING kinship from a threshold-based call, but real recombination breaks
    visible in the IBD structure). See assess_duplicate_vs_relative().
    """
    index = defaultdict(list)
    if not ibd_gz_path or not os.path.exists(ibd_gz_path):
        return index
    opener = gzip.open if ibd_gz_path.endswith('.gz') else open
    try:
        with opener(ibd_gz_path, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                id1, _, id2, _, chrom, start, end, lod = parts[:8]
                try:
                    if float(lod) < min_lod:
                        continue
                    key = tuple(sorted([id1, id2]))
                    index[key].append((chrom, int(start), int(end)))
                except ValueError:
                    continue
    except Exception as e:
        print(f"  Warning: could not read {ibd_gz_path}: {e}")
    return index


def _merged_ibd_bp(segments):
    """Merge overlapping (chrom, start, end) intervals; return total covered bp."""
    by_chrom = defaultdict(list)
    for chrom, start, end in segments:
        by_chrom[chrom].append((start, end))
    total = 0
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append([s, e])
        total += sum(e - s for s, e in merged)
    return total


def build_ibd_genome_proxy(ibd_segments_index):
    """
    Precompute, ONCE, each individual's total merged IBD bp across ALL its
    pairs in ibd_segments_index: dict id -> merged_bp. Used by
    assess_duplicate_vs_relative() as a self-calibrating proxy for
    "testable genome size".

    Must be built once and reused for every MZ-level pair assessed.
    assess_duplicate_vs_relative() previously rescanned the FULL
    ibd_segments_index per call to compute this (an O(P) scan per pair,
    O(M*P) total across M MZ-level pairs) — in a cohort with thousands of
    MZ-level pairs and a large hap-IBD segment table, that silently hung
    for hours with no output (same antipattern class as the 2.3M-redundant-
    log-line bug fixed elsewhere in this file, except this one doesn't even
    print before it hangs).
    """
    by_individual = defaultdict(list)
    for (id1, id2), segs in ibd_segments_index.items():
        by_individual[id1].extend(segs)
        by_individual[id2].extend(segs)
    return {iid: _merged_ibd_bp(segs) for iid, segs in by_individual.items()}


def assess_duplicate_vs_relative(id1, id2, ibd_segments_index, genome_proxy_bp,
                                  min_coverage_for_duplicate=0.95,
                                  duplicate_segment_reference_mb=None,
                                  fragmentation_factor=1.5):
    """
    For a pair with KING kinship >= KING_MZ_MIN, use hap-IBD segment
    structure to distinguish:

      LIKELY_DUPLICATE   — near-complete genome-wide IBD coverage between
                            id1 and id2 directly. Consistent with a true
                            MZ twin / technical replicate / clonal duplicate.

      LIKELY_DUPLICATE_FRAGMENTED — coverage is incomplete (looks like
                            CLOSE_BUT_DISTINCT by coverage alone), BUT the
                            segment-length PATTERN matches the fragmentation
                            noise floor seen among CONFIRMED duplicates in
                            this same dataset/phasing run (mean_segment_Mb
                            <= duplicate_segment_reference_mb *
                            fragmentation_factor). Statistical phasing (no
                            pedigree information — the case for candidate
                            pairs, which aren't in the .fam yet) introduces
                            phase-switch errors that fragment a true
                            duplicate's whole-genome IBD into many short
                            segments, artificially deflating measured
                            coverage below min_coverage_for_duplicate. This
                            is "probably actually a duplicate, but the
                            coverage check under-measured it due to phasing
                            noise" — NOT auto-collapsed (the heuristic is not
                            proof of identity), but reported distinctly from
                            CLOSE_BUT_DISTINCT so it can be prioritised for
                            closer inspection. Only evaluated when
                            duplicate_segment_reference_mb is provided (the
                            caller needs >= MIN_DUPLICATES_FOR_FRAGMENTATION_
                            REFERENCE CONFIRMED duplicate pairs in the same
                            run for the reference to be meaningful — see
                            collapse_mz_twin_groups_in_king()).

      CLOSE_BUT_DISTINCT — elevated KING kinship (crosses the approximate
                            MZ threshold), incomplete coverage, AND a
                            segment-length pattern unlike the fragmented-
                            duplicate noise floor (or no reference available
                            to compare against). Consistent with a genuinely
                            distinct individual from a highly inbred
                            relationship (e.g. repeated backcrossing,
                            extensive selfing) — real recombination still
                            occurred, it just elevated kinship enough to
                            cross the threshold-based MZ cutoff.

      INSUFFICIENT_DATA  — no IBD segments for this specific pair; cannot
                            distinguish without IBD evidence.

    Coverage is computed against a proxy for "testable genome size": the
    total IBD bp id1 has with ANY individual in the index (same approach as
    refine_pedigree.py's compute_trio_genome_coverage). This avoids needing
    an external genome-size constant and self-calibrates to whatever
    fraction of the genome hap-IBD successfully phased/tested.

    genome_proxy_bp: dict id -> merged_bp, from build_ibd_genome_proxy()
    (built ONCE by the caller and reused across all pairs — see that
    function's docstring for why this matters).
    """
    key = tuple(sorted([id1, id2]))
    segments = ibd_segments_index.get(key)
    if not segments:
        return 'INSUFFICIENT_DATA', None

    pair_ibd_bp = _merged_ibd_bp(segments)

    proxy_bp = genome_proxy_bp.get(id1, 0)
    coverage = pair_ibd_bp / proxy_bp if proxy_bp > 0 else None

    n_segments = len(segments)
    mean_seg_mb = (pair_ibd_bp / n_segments) / 1e6 if n_segments else 0

    if coverage is not None and coverage >= min_coverage_for_duplicate:
        classification = 'LIKELY_DUPLICATE'
    elif (duplicate_segment_reference_mb is not None and mean_seg_mb > 0
          and mean_seg_mb <= duplicate_segment_reference_mb * fragmentation_factor):
        classification = 'LIKELY_DUPLICATE_FRAGMENTED'
    else:
        classification = 'CLOSE_BUT_DISTINCT'

    return classification, {
        'coverage_frac':   coverage,
        'total_ibd_Mb':    pair_ibd_bp / 1e6,
        'n_segments':      n_segments,
        'mean_segment_Mb': mean_seg_mb,
        'classification':  classification,
    }


def ibd_relationship_hint(ibd_summary):
    """
    Use IBD segment statistics to hint at the relationship type for
    2nd-degree relatives (half-sib vs. grandparent vs. avuncular).

    Half-sibling:            total_Mb ~700–900, mean_segment_Mb ~30–50
    Grandparent–grandchild:  total_Mb ~700–900, mean_segment_Mb ~15–25 (shorter)
    Avuncular:               total_Mb ~700–900, mean_segment_Mb ~25–35

    These are approximate for a species with a medium-size genome.
    Scale mean_segment thresholds proportionally with genome size.
    """
    if not ibd_summary:
        return 'UNKNOWN (no IBD data)'
    mean_Mb = ibd_summary['mean_segment_Mb']
    total   = ibd_summary['total_Mb']
    if mean_Mb > 40:
        hint = 'likely HALF-SIBLING (long mean segment)'
    elif mean_Mb > 20:
        hint = 'likely AVUNCULAR or HALF-SIBLING (medium segment)'
    elif mean_Mb > 0:
        hint = 'likely GRANDPARENT-GRANDCHILD (short mean segment)'
    else:
        hint = 'UNKNOWN'
    return f"{hint} | total={total:.0f} Mb, mean_seg={mean_Mb:.1f} Mb, n={ibd_summary['n_segments']}"


# ---------------------------------------------------------------------------
# Upward extension: grandparent search
# ---------------------------------------------------------------------------

def collapse_mz_twin_groups_in_king(king, fam_ids, ibd_segments_index=None,
                                     min_coverage_for_duplicate=0.95,
                                     fragmentation_factor=1.5):
    """
    Detect MZ-twin / duplicate-sample groups in the KING data (any pair with
    KINSHIP >= KING_MZ_MIN) and collapse the whole KING graph to one
    representative per group, BEFORE any grandparent/offspring searching.

    Why this is needed (found via a real run): prepare_plink_beagle.py is
    deliberately run on the pre-purge VCF so refine_pedigree.py can do
    MZ-substitution checks — meaning extend_pedigree.py's KING input
    legitimately contains many duplicate individuals. Without collapsing:
      - find_grandparent_candidates()'s O(M^2) pairing loop independently
        rediscovers the same "G1 and G2 are twins" fact once per founder
        that has both in its 2nd-degree-relative list — in one real run,
        2.3 MILLION such (redundant, harmless but extremely noisy)
        exclusions were logged.
      - Worse: an MZ pair only gets excluded when BOTH members land in the
        SAME candidate pair. If "G1 of candidate A" and "G1 of candidate B"
        are twins of EACH OTHER (not paired together), both candidates
        survive and look like two distinct grandparents when they may be
        the same biological individual counted twice.
    Collapsing once up front fixes both: redundant computation disappears,
    and cross-candidate duplication is eliminated because every reference
    to any group member is rewritten to the same representative ID.

    KINSHIP >= KING_MZ_MIN is an APPROXIMATE threshold, not proof of literal
    genetic identity. In closely related / repeatedly inbred germplasm
    (backcrossed or extensively selfed lines), two genuinely DISTINCT
    individuals can have kinship elevated into the nominal "MZ" range
    without being the same genotype — real meiotic recombination still
    occurred. Collapsing such a pair would be wrong: it would erase a
    genuine candidate, not deduplicate a technical artifact.

    If ibd_segments_index is provided (from load_ibd_segments_raw), each
    KING-MZ-level pair is checked with assess_duplicate_vs_relative():
    only pairs with near-complete IBD coverage of each other's genome
    (true duplicates) are collapsed. Pairs with elevated kinship but
    incomplete IBD coverage (close-but-genuinely-distinct) are explicitly
    NOT collapsed and are instead returned in close_but_distinct_pairs so
    callers can still report them — flagged, not erased.

    Statistical phasing (the only option for candidates not yet in the
    .fam) introduces phase-switch errors that can fragment a TRUE
    duplicate's whole-genome IBD into many short segments, artificially
    deflating its measured coverage below min_coverage_for_duplicate. To
    catch this, a second pass self-calibrates a "fragmentation noise floor"
    from the segment-length pattern of CONFIRMED duplicates in this same
    run (median mean_segment_Mb, when >= MIN_DUPLICATES_FOR_FRAGMENTATION_
    REFERENCE such pairs exist) and reclassifies any CLOSE_BUT_DISTINCT pair
    whose own segmentation matches that noise floor as
    LIKELY_DUPLICATE_FRAGMENTED instead — still NOT collapsed (the
    heuristic isn't proof), but reported distinctly so it can be
    prioritised for closer inspection rather than dismissed as a confirmed-
    distinct relative.

    Without ibd_segments_index, the conservative default (collapse on
    kinship alone) is used, since there is no way to distinguish the two
    cases from KING statistics alone.

    fam_ids: individuals already present in the working .fam. When picking
        a representative, PREFER one of these — it's the individual that
        survived the original purge_mz_twins.py step, so results stay
        meaningful against the existing pedigree rather than referencing an
        arbitrary twin ID that isn't even in the trio-calling VCF.

    Returns:
      collapsed_king:  dict (id1, id2) -> record, with all keys remapped to
                        representatives (the MZ pair itself is dropped)
      remap:            dict original_id -> representative_id
      mz_groups:        dict representative_id -> sorted list of all members
                         (only for groups with >1 member; confirmed duplicates only)
      close_but_distinct_pairs: dict (id1,id2) sorted-tuple -> stats dict,
                        for pairs with MZ-level kinship that IBD evidence
                        shows are NOT literal duplicates. Each stats dict
                        additionally carries 'ibs0' (the pair's raw KING
                        IBS0 — phasing-independent, ~0 for true duplicates,
                        included so the phasing-dependent classification can
                        be sanity-checked against an unphased signal) and
                        'fragmentation_reference_mb' (the calibrated
                        duplicate-segment reference used for this run's
                        LIKELY_DUPLICATE_FRAGMENTED calls, or None if fewer
                        than MIN_DUPLICATES_FOR_FRAGMENTATION_REFERENCE
                        confirmed duplicates were available to calibrate it
                        — same value on every entry in a given run).
    """
    # Union-find over MZ-level kinship pairs CONFIRMED as true duplicates
    parent = {}
    close_but_distinct_pairs = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Built ONCE up front (not per-pair) — see build_ibd_genome_proxy()
    # docstring for why a per-pair full-index scan here previously caused
    # multi-hour hangs in densely related cohorts.
    genome_proxy_bp = (build_ibd_genome_proxy(ibd_segments_index)
                        if ibd_segments_index is not None else None)

    mz_level_pairs = [(a, b) for (a, b), rec in king.items() if rec['kinship'] >= KING_MZ_MIN]

    if ibd_segments_index is None:
        for a, b in mz_level_pairs:
            union(a, b)
    else:
        # Pass 1: classify every MZ-level pair WITHOUT a fragmentation
        # reference yet, to first identify which pairs are CONFIRMED
        # duplicates (by coverage alone) in this run.
        prelim = {}
        duplicate_mean_segment_mbs = []
        for a, b in mz_level_pairs:
            classification, stats = assess_duplicate_vs_relative(
                a, b, ibd_segments_index, genome_proxy_bp, min_coverage_for_duplicate)
            prelim[(a, b)] = (classification, stats)
            if classification == 'LIKELY_DUPLICATE' and stats['mean_segment_Mb'] > 0:
                duplicate_mean_segment_mbs.append(stats['mean_segment_Mb'])

        duplicate_mean_segment_mbs.sort()
        fragmentation_reference_mb = (
            duplicate_mean_segment_mbs[len(duplicate_mean_segment_mbs) // 2]
            if len(duplicate_mean_segment_mbs) >= MIN_DUPLICATES_FOR_FRAGMENTATION_REFERENCE
            else None)

        # Pass 2: only pairs that were CLOSE_BUT_DISTINCT in pass 1 need
        # reclassifying against the now-known reference.
        for a, b in mz_level_pairs:
            classification, stats = prelim[(a, b)]
            if classification == 'CLOSE_BUT_DISTINCT' and fragmentation_reference_mb is not None:
                classification, stats = assess_duplicate_vs_relative(
                    a, b, ibd_segments_index, genome_proxy_bp, min_coverage_for_duplicate,
                    duplicate_segment_reference_mb=fragmentation_reference_mb,
                    fragmentation_factor=fragmentation_factor)
            if classification in ('CLOSE_BUT_DISTINCT', 'LIKELY_DUPLICATE_FRAGMENTED'):
                # IBS0 (opposite-homozygote rate) is independent of phasing —
                # KING computes it directly from unphased genotypes. True
                # duplicates should have IBS0 ~ 0; attaching it here lets
                # downstream review sanity-check the (phasing-dependent)
                # fragmentation call against a phasing-free signal. Also
                # attach the calibrated reference itself so it's never a
                # black box — same value on every entry in this run.
                stats['ibs0'] = king[(a, b)]['ibs0']
                stats['fragmentation_reference_mb'] = fragmentation_reference_mb
                close_but_distinct_pairs[tuple(sorted([a, b]))] = stats
                continue   # do NOT collapse either bucket — heuristic, not proof
            # LIKELY_DUPLICATE or INSUFFICIENT_DATA -> fall through to collapse
            union(a, b)

    groups = defaultdict(set)
    for x in parent:
        groups[find(x)].add(x)

    remap = {}
    mz_groups = {}
    for members in groups.values():
        if len(members) == 1:
            sample = next(iter(members))
            remap[sample] = sample
            continue
        # Prefer a representative already in the working .fam
        in_fam = sorted(m for m in members if m in fam_ids)
        rep = in_fam[0] if in_fam else sorted(members)[0]
        for m in members:
            remap[m] = rep
        mz_groups[rep] = sorted(members)

    # Rebuild king with all keys remapped; drop self-pairs (the MZ pair
    # itself); when multiple original pairs collapse onto the same
    # (rep_a, rep_b), keep the highest-kinship record (most conservative).
    collapsed_king = {}
    for (a, b), rec in king.items():
        ra, rb = remap.get(a, a), remap.get(b, b)
        if ra == rb:
            continue   # was an MZ pair (or self) — collapsed away
        key = tuple(sorted([ra, rb]))
        if key not in collapsed_king or rec['kinship'] > collapsed_king[key]['kinship']:
            collapsed_king[key] = rec

    return collapsed_king, remap, mz_groups, close_but_distinct_pairs


def build_king_index(king):
    """
    Build an adjacency index: individual -> list of (other, kinship, ibs0).

    This is the critical performance fix for extend_pedigree.py. Without it,
    find_grandparent_candidates() and find_offspring_candidates() each scan
    the ENTIRE king dict (every pair) for every single founder/leaf, giving
    O(N_individuals * N_pairs) total cost. For a cohort of ~3500 founders
    against ~1.6M KING pairs that is ~5.6 billion dict iterations for the
    grandparent search alone — effectively does not finish.

    Building this index is a single O(N_pairs) pass; every subsequent lookup
    for an individual is O(that individual's degree) instead of O(N_pairs).
    """
    index = defaultdict(list)
    for (a, b), rec in king.items():
        index[a].append((b, rec['kinship'], rec['ibs0']))
        index[b].append((a, rec['kinship'], rec['ibs0']))
    return index


def find_grandparent_candidates(founder, king, king_index, all_ids, ibd=None,
                                 mz_groups=None, close_but_distinct_pairs=None):
    """
    For a founder (no known parents), find candidate grandparent PAIRS.

    mz_groups: dict representative_id -> [all MZ-twin-group members], from
        collapse_mz_twin_groups_in_king(). When g1 or g2 is the
        representative of a collapsed MZ-twin group, the candidate dict
        records the OTHER group members as 'g1_alternatives'/'g2_alternatives'.
        IMPORTANT: true MZ twins are genetically indistinguishable, so no
        genetic test (this one, BEAGLE, hap-IBD, vcfped) can determine which
        twin is the "real" grandparent — that requires external evidence
        (breeding records, sample tracking). The representative shown is a
        bookkeeping choice (preferring whichever twin is already in the
        working .fam), NOT a validated identity. Report the alternatives
        rather than silently hiding them.

    close_but_distinct_pairs: dict (id1,id2) sorted-tuple -> IBD coverage
        stats, from collapse_mz_twin_groups_in_king(). These are pairs with
        KING kinship >= KING_MZ_MIN that hap-IBD evidence shows are NOT
        literal duplicates (incomplete genome coverage — consistent with a
        genuinely distinct individual from a highly inbred relationship,
        e.g. repeated backcrossing/selfing). Such pairs are NOT skipped —
        they are reported with g1g2_relation='CLOSE_INBRED' so the user can
        evaluate them rather than losing a real candidate to an
        approximate-threshold false positive. A pair whose stats
        classification is 'LIKELY_DUPLICATE_FRAGMENTED' (segment-length
        pattern matches the fragmentation noise floor of confirmed
        duplicates in this run — likely a true duplicate under-measured due
        to statistical-phasing switch errors, not a genuinely distinct
        individual) is instead reported with g1g2_relation=
        'CLOSE_INBRED_POSSIBLE_DUP' and a larger confidence penalty.

    Returns list of dicts: {g1, g2, kinship_g1, kinship_g2, g1g2_kinship,
                             ibd_hint_g1, ibd_hint_g2, confidence,
                             g1_alternatives, g2_alternatives}
    """
    mz_groups = mz_groups or {}
    close_but_distinct_pairs = close_but_distinct_pairs or {}
    # Step 1: all 2nd-degree relatives of founder (O(degree), not O(N_pairs))
    second_degree = [
        (other, kinship, ibs0)
        for other, kinship, ibs0 in king_index.get(founder, [])
        if KING_2ND_MIN <= kinship < KING_1ST_MIN
    ]

    if len(second_degree) < 2:
        return []

    # Safety cap: in a closely related cohort a single founder can have
    # hundreds of 2nd-degree relatives, making the O(M^2) pairing below
    # blow up (M=500 -> 250k pairs per founder x thousands of founders).
    # Keep only the highest-kinship candidates for pairing.
    _MAX_SECOND_DEGREE_FOR_PAIRING = 200
    if len(second_degree) > _MAX_SECOND_DEGREE_FOR_PAIRING:
        second_degree = sorted(second_degree, key=lambda x: -x[1])[:_MAX_SECOND_DEGREE_FOR_PAIRING]

    # Step 2: among 2nd-degree relatives, find pairs that are 1st-degree to each other
    # → candidate grandparent pairs
    candidates = []
    checked = set()
    for i, (g1, k1, i1) in enumerate(second_degree):
        for j, (g2, k2, i2) in enumerate(second_degree):
            if j <= i:
                continue
            pair_key = tuple(sorted([g1, g2]))
            if pair_key in checked:
                continue
            checked.add(pair_key)
            g1g2_rec = get_king(king, g1, g2)
            if g1g2_rec is None:
                continue
            g1g2_kin = g1g2_rec['kinship']
            g1g2_ibs = g1g2_rec['ibs0']

            # MZ-twin / duplicate safety net: by the time this function runs,
            # main() has already collapsed CONFIRMED MZ-twin/duplicate groups
            # in the KING graph to single representatives
            # (collapse_mz_twin_groups_in_king), so a g1g2 pair at MZ-level
            # kinship reaching this point is one of two cases:
            #   1. It's in close_but_distinct_pairs: hap-IBD coverage showed
            #      these are genuinely DISTINCT individuals despite elevated
            #      kinship (plausible in highly inbred/backcrossed germplasm).
            #      Do NOT skip — report as a flagged 'CLOSE_INBRED' candidate.
            #   2. No IBD evidence was available to distinguish duplicate vs.
            #      distinct, so collapse_mz_twin_groups_in_king() already
            #      collapsed it conservatively — this case should not occur
            #      here. Kept as a silent guard (counted once in main(),
            #      rather than a per-occurrence print) against the same
            #      rediscovery-storm bug fixed previously.
            close_inbred_stats = close_but_distinct_pairs.get(tuple(sorted([g1, g2])))
            if g1g2_kin >= KING_MZ_MIN and close_inbred_stats is None:
                continue

            # Grandparent pair: they should be 1st-degree relatives of each other
            # (parent and the other grandparent are unrelated, but two grandparents
            # from the SAME side could be PO to each other).
            # Actually, in a non-inbred case, the two grandparents are NOT related.
            # If they ARE 1st-degree to each other, they might be parent-offspring
            # (one grandparent is a child of the other → 3-generation chain).
            # Let's report both cases.

            po_pair = (g1g2_kin >= KING_1ST_MIN and g1g2_ibs <= KING_IBS0_PO)
            sib_pair = (g1g2_kin >= KING_1ST_MIN and g1g2_ibs > KING_IBS0_PO)
            unrelated_pair = g1g2_kin < KING_2ND_MIN

            if not (po_pair or sib_pair):
                continue   # pairs must be at least somewhat related for grandparent structure

            ibd_g1 = ibd_relationship_hint(get_ibd_summary(ibd, founder, g1)) if ibd else None
            ibd_g2 = ibd_relationship_hint(get_ibd_summary(ibd, founder, g2)) if ibd else None

            is_fragmented_dup = (close_inbred_stats is not None and
                                  close_inbred_stats.get('classification') == 'LIKELY_DUPLICATE_FRAGMENTED')

            conf = (k1 + k2) / 2 - abs(k1 - k2) * 0.5   # prefer balanced kinship
            if is_fragmented_dup:
                # Segment-length pattern matches the fragmentation noise
                # floor of CONFIRMED duplicates in this run — this pairing
                # may really be ONE individual counted twice (just under-
                # measured by the coverage check due to phasing noise).
                # Penalize harder than a confirmed-distinct CLOSE_INBRED.
                conf -= 0.15
            elif close_inbred_stats is not None:
                # Elevated kinship crossed the approximate MZ threshold but
                # hap-IBD coverage rules out literal duplication — still a
                # real candidate, just flagged for extra scrutiny.
                conf -= 0.05

            g1_alts = [m for m in mz_groups.get(g1, []) if m != g1]
            g2_alts = [m for m in mz_groups.get(g2, []) if m != g2]

            if is_fragmented_dup:
                relation = 'CLOSE_INBRED_POSSIBLE_DUP'
            elif close_inbred_stats is not None:
                relation = 'CLOSE_INBRED'
            elif po_pair:
                relation = 'PO'
            elif sib_pair:
                relation = 'SIBLING'
            else:
                relation = 'RELATED'

            candidates.append({
                'founder': founder,
                'g1': g1, 'kinship_g1': k1, 'ibs0_g1': i1,
                'g2': g2, 'kinship_g2': k2, 'ibs0_g2': i2,
                'g1g2_kinship': g1g2_kin, 'g1g2_ibs0': g1g2_ibs,
                'g1g2_relation': relation,
                'g1g2_ibd_coverage': close_inbred_stats,   # set only for CLOSE_INBRED
                'ibd_hint_g1': ibd_g1,
                'ibd_hint_g2': ibd_g2,
                'confidence': conf,
                'g1_alternatives': g1_alts,   # other MZ-twin IDs equally valid for this slot
                'g2_alternatives': g2_alts,
            })

    return sorted(candidates, key=lambda x: -x['confidence'])


# ---------------------------------------------------------------------------
# Downward extension: offspring search
# ---------------------------------------------------------------------------

def find_offspring_candidates(leaf, all_children, king, king_index, all_ids, ibd=None,
                               mz_groups=None):
    """
    For a leaf (no known children), find individuals in all_ids not yet
    assigned as someone's child who have PO signal to this leaf.
    Also find a compatible second parent for each candidate.

    all_children: precomputed set of all individuals with a known parent
                  (pass in once from main() — recomputing this O(len(fam))
                  set inside this function for every leaf was itself a
                  needless O(leaves x len(fam)) cost).

    mz_groups: see find_grandparent_candidates() — when candidate_child or
        second_parent is the representative of a collapsed MZ-twin group,
        the result records the other group members as alternatives. Cannot
        be genetically validated which twin is correct; report, don't hide.

    Returns list of dicts: {candidate_child, kinship_to_leaf, second_parent,
                             kinship_sp_to_child, confidence,
                             candidate_child_alternatives, second_parent_alternatives}
    """
    mz_groups = mz_groups or {}
    # PO-signal candidates for this leaf: O(leaf's degree), not O(N_pairs)
    po_candidates = [
        (other, kinship, ibs0)
        for other, kinship, ibs0 in king_index.get(leaf, [])
        if kinship >= KING_1ST_MIN and ibs0 <= KING_IBS0_PO and other not in all_children
    ]

    # Safety cap: in a densely related cohort a single leaf can have hundreds
    # of PO-signal candidates, each triggering an O(degree) inner lookup for
    # a compatible second parent. Cap to the highest-kinship candidates to
    # bound worst-case runtime per leaf.
    _MAX_PO_CANDIDATES_PER_LEAF = 100
    if len(po_candidates) > _MAX_PO_CANDIDATES_PER_LEAF:
        po_candidates = sorted(po_candidates, key=lambda x: -x[1])[:_MAX_PO_CANDIDATES_PER_LEAF]

    results = []
    for candidate, kinship, ibs0 in po_candidates:
        rec = {'kinship': kinship, 'ibs0': ibs0}

        # Find a compatible second parent: O(candidate's degree), not O(N_pairs)
        second_parent_candidates = []
        for other, k2, i2 in king_index.get(candidate, []):
            if other == leaf or other == candidate:
                continue
            if k2 < KING_1ST_MIN or i2 > KING_IBS0_PO:
                continue
            # Check that leaf and second parent are not themselves PO
            # (would imply co-parents are parent-offspring, unusual but possible in inbreeding)
            leaf_sp_rec = get_king(king, leaf, other)
            leaf_sp_compatible = (leaf_sp_rec is None or
                                   leaf_sp_rec['kinship'] < KING_1ST_MIN or
                                   leaf_sp_rec['ibs0'] > KING_IBS0_PO)
            if leaf_sp_compatible:
                second_parent_candidates.append((other, k2, i2))

        second_parent_candidates.sort(key=lambda x: (-x[1], x[2]))
        sp = second_parent_candidates[0] if second_parent_candidates else (None, None, None)

        ibd_hint = ibd_relationship_hint(get_ibd_summary(ibd, leaf, candidate)) if ibd else None

        child_alts = [m for m in mz_groups.get(candidate, []) if m != candidate]
        sp_alts = ([m for m in mz_groups.get(sp[0], []) if m != sp[0]]
                  if sp[0] else [])

        results.append({
            'leaf': leaf,
            'candidate_child': candidate,
            'kinship_leaf_child': rec['kinship'],
            'ibs0_leaf_child': rec['ibs0'],
            'second_parent': sp[0],
            'kinship_sp_child': sp[1],
            'ibs0_sp_child': sp[2],
            'ibd_hint': ibd_hint,
            'n_second_parent_options': len(second_parent_candidates),
            'confidence': rec['kinship'] - rec['ibs0'],
            'candidate_child_alternatives': child_alts,
            'second_parent_alternatives': sp_alts,
        })

    return sorted(results, key=lambda x: -x['confidence'])


# ---------------------------------------------------------------------------
# Strategy assessment
# ---------------------------------------------------------------------------

def assess_strategy(up_candidates, down_candidates, n_founders, n_leaves,
                     has_ibd, n_samples, strategy_arg):
    """
    Recommend a strategy based on cohort size and candidate statistics.
    Returns a list of recommendation strings.
    """
    recs = []
    n_up   = len(up_candidates)
    n_down = len(down_candidates)

    if strategy_arg != 'auto':
        recs.append(f"Strategy forced to: {strategy_arg}")
        return recs

    # ── KING-only is sufficient when ─────────────────────────────────────────
    if n_up + n_down == 0:
        recs.append("KING-only: No extension candidates found. "
                    "Pedigree appears complete within this cohort.")
        return recs

    if n_up + n_down <= 20 and not has_ibd:
        recs.append(
            "KING-only: Few candidates found. KING evidence alone is sufficient "
            "for a first-pass extension. Proceed to confirm with:")
        recs.append("  → vcfped rerun on candidate subsets (for T1/T2 scores)")

    # ── vcfped rerun recommended when ────────────────────────────────────────
    if n_up > 50 or n_down > 50:
        recs.append(
            "vcfped RERUN recommended: many candidates (>50). "
            "Extract candidate samples into a subset VCF and run:\n"
            "  python -m vcfped.vcfped subset.vcf.gz "
            "--no-pairwise --no-gender --all -o extension_out\n"
            "This gives exact T1/T2 Mendelian scores for all candidate triples. "
            "Use vcfped.py (not vcfped_optimized.py) and --all here: --all needs the "
            "low-confidence rows (Not trio/Other) that vcfped_optimized.py's --all "
            "path has a known, partially-confirmed scoring discrepancy for (see "
            "vcfped_optimized.py's _is_better_pair_result()/_is_better_trio_result() "
            "docstrings) -- the subset is small enough that the reference engine's "
            "lack of multiprocessing/JIT costs nothing here.")

    # ── BEAGLE/hap-IBD recommended when ─────────────────────────────────────
    if has_ibd:
        recs.append(
            "BEAGLE/hap-IBD AVAILABLE: Use IBD segment lengths to distinguish "
            "grandparent-grandchild (shorter segments) from half-sibling "
            "(longer segments). This resolves 2nd-degree ambiguity in KING.")

    # ── Combined strategy for large cohorts ──────────────────────────────────
    if n_samples > 200 and (n_up + n_down) > 20:
        recs.append(
            "COMBINED strategy recommended for large cohort:\n"
            "  Phase 1: KING screening identifies candidates (done).\n"
            "  Phase 2: vcfped rerun on candidate subset (fast; ~O(k³) not O(N³)).\n"
            "  Phase 3: BEAGLE hap-IBD for ambiguous 2nd-degree cases.")

    return recs


# ---------------------------------------------------------------------------
# VCF subset candidate file (for vcfped rerun)
# ---------------------------------------------------------------------------

def write_vcfped_candidates(path, up_candidates, down_candidates):
    """
    Write a sample list for subsetting the VCF before a vcfped rerun.
    Also writes the candidate triples to try.

    MZ-twin alternatives (if any candidate involved a collapsed MZ-twin
    group) are documented in a comment block, NOT added to the sample list:
    since true MZ twins are genetically identical, vcfped/KING/BEAGLE would
    produce IDENTICAL statistics for either twin — testing the alternative
    adds no genetic information. The alternatives are listed so the user
    knows to consult non-genetic records (breeding logs, sample tracking)
    if it matters WHICH specific twin was the real individual.
    """
    samples_needed = set()
    triples = []
    mz_notes = []

    for c in up_candidates:
        samples_needed.update([c['founder'], c['g1'], c['g2']])
        note = (f"grandparent_candidate confidence={c['confidence']:.3f} "
                f"relation={c.get('g1g2_relation', 'UNKNOWN')}")
        cov = c.get('g1g2_ibd_coverage')
        if cov:
            cov_str = f"{cov['coverage_frac']:.1%}" if cov['coverage_frac'] is not None else 'n/a'
            note += (f" ibd_coverage={cov_str} n_segments={cov['n_segments']} "
                     f"mean_seg_Mb={cov['mean_segment_Mb']:.1f}")
            if cov.get('ibs0') is not None:
                # Phasing-independent (KING-derived) sanity check: ~0 supports
                # a true duplicate regardless of what the (phasing-dependent)
                # segment pattern suggested.
                note += f" ibs0={cov['ibs0']:.4f}"
        triples.append((c['g1'], c['g2'], c['founder'], note))
        if c.get('g1_alternatives'):
            mz_notes.append(f"{c['g1']} (used as G1 for founder {c['founder']}) "
                            f"<-> indistinguishable from: {', '.join(c['g1_alternatives'])}")
        if c.get('g2_alternatives'):
            mz_notes.append(f"{c['g2']} (used as G2 for founder {c['founder']}) "
                            f"<-> indistinguishable from: {', '.join(c['g2_alternatives'])}")

    for c in down_candidates:
        samples_needed.add(c['leaf'])
        samples_needed.add(c['candidate_child'])
        if c['second_parent']:
            samples_needed.add(c['second_parent'])
            triples.append((c['leaf'], c['second_parent'], c['candidate_child'],
                             f"offspring_candidate confidence={c['confidence']:.3f}"))
        if c.get('candidate_child_alternatives'):
            mz_notes.append(f"{c['candidate_child']} (candidate child of leaf {c['leaf']}) "
                            f"<-> indistinguishable from: "
                            f"{', '.join(c['candidate_child_alternatives'])}")
        if c.get('second_parent_alternatives'):
            mz_notes.append(f"{c['second_parent']} (second parent for leaf {c['leaf']}) "
                            f"<-> indistinguishable from: "
                            f"{', '.join(c['second_parent_alternatives'])}")

    with open(path, 'w') as f:
        f.write("# Sample list for VCF subsetting (bcftools view -S)\n")
        f.write("# Use: bcftools view -S this_file input.vcf.gz | bgzip > subset.vcf.gz\n")
        f.write("# Then: python -m vcfped.vcfped subset.vcf.gz "
                "--no-pairwise --no-gender --all -o extension_out\n")
        f.write("# Use vcfped.py here, NOT vcfped_optimized.py: --all needs the low-\n"
                "# confidence rows (Not trio/Other) that vcfped_optimized.py's --all path\n"
                "# has a known, partially-confirmed scoring discrepancy for (see\n"
                "# vcfped_optimized.py's _is_better_pair_result()/_is_better_trio_result()\n"
                "# docstrings). This subset is small enough that vcfped.py's lack of\n"
                "# multiprocessing/JIT costs nothing here.\n\n")
        f.write("# Samples needed:\n")
        for s in sorted(samples_needed):
            f.write(s + "\n")
        f.write("\n# Candidate triples (parent1, parent2, expected_child, note):\n")
        for p1, p2, c, note in triples:
            f.write(f"#{p1}\t{p2}\t{c}\t{note}\n")
        if mz_notes:
            f.write("\n# MZ-TWIN ALTERNATIVES — genetically indistinguishable from the\n"
                    "# individual used above. A vcfped/KING/BEAGLE rerun on the alternative\n"
                    "# would give IDENTICAL results (same genotype) — this is NOT something\n"
                    "# a genetic test can resolve. Check breeding/sample records if it\n"
                    "# matters which specific individual was the true parent/grandparent:\n")
            for note in sorted(set(mz_notes)):
                f.write(f"#  {note}\n")

    print(f"  Written: {path}")


def write_confirmed_duplicates(path, mz_groups):
    """
    Write every pairwise combination within each hap-IBD-coverage-confirmed
    MZ-twin/duplicate group (from collapse_mz_twin_groups_in_king(), when
    called with ibd_segments_index -- i.e. the collapse decision was
    coverage-validated, not kinship-only).

    This is a stricter ground truth than vcfped's own pairwise "MZ twins"
    verdict for calibrate_error_rate.py's --confirmed-duplicates: in densely
    inbred germplasm, vcfped's allele-sharing test can call two genuinely
    distinct-but-closely-related samples "MZ twins" (inbreeding-induced
    inflation), whereas this file only contains pairs whose hap-IBD genome
    coverage actually confirmed literal identity.
    """
    with open(path, 'w') as f:
        f.write('ID1\tID2\n')
        for members in mz_groups.values():
            for a, b in itertools.combinations(sorted(members), 2):
                f.write(f"{a}\t{b}\n")
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# BEAGLE confirmation script
# ---------------------------------------------------------------------------

def write_beagle_check_script(path, up_candidates, down_candidates,
                               out_prefix, fam_path, phased_vcf=None,
                               genetic_map=None, hapibd_exe='hap-ibd'):
    """
    Generate a shell script that runs hap-IBD on specific pairs to confirm
    grandparent and offspring candidates.

    phased_vcf: explicit path to the pedigree-aware phased VCF. This is
        produced by prepare_plink_beagle.py's run_beagle.sh under THAT
        script's own -o prefix, which is commonly DIFFERENT from
        extend_pedigree.py's out_prefix (derived from the .fam filename) —
        guessing it from out_prefix is wrong unless the two happen to match.
    genetic_map: explicit path to the genetic map. hap-IBD REQUIRES this
        (unlike BEAGLE phasing, where it's optional). Reuse the synthetic
        map already generated alongside the phased VCF by run_beagle.sh.
    """
    phased_vcf_value = phased_vcf or f"{out_prefix}_phased.vcf.gz"
    guessed_note = ("" if phased_vcf else
                    "  # GUESSED — likely wrong; pass --phased-vcf explicitly "
                    "to extend_pedigree.py if this differs from your "
                    "prepare_plink_beagle.py -o prefix")

    # Guess the map from the phased VCF's prefix if not given explicitly:
    # prepare_plink_beagle.py's run_beagle.sh writes <its_own_prefix>_synthetic.map
    # alongside <its_own_prefix>_phased.vcf.gz, so stripping "_phased.vcf.gz"
    # from phased_vcf_value and appending "_synthetic.map" recovers it.
    if genetic_map:
        map_value = genetic_map
        map_guessed_note = ""
    else:
        base = phased_vcf_value
        for suffix in ('_phased.vcf.gz', '.vcf.gz'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        map_value = f"{base}_synthetic.map"
        map_guessed_note = ("  # GUESSED from --phased-vcf prefix — pass "
                            "--genetic-map explicitly if this is wrong")

    script = f"""#!/usr/bin/env bash
# BEAGLE hap-IBD confirmation for pedigree extension candidates
# Generated by extend_pedigree.py
# Requires: phased VCF + genetic map from run_beagle.sh, hap-ibd

set -euo pipefail

PHASED_VCF="{phased_vcf_value}"{guessed_note}
MAP="{map_value}"{map_guessed_note}
HAPIBD="{hapibd_exe}"    # conda: hap-ibd | jar: java -Xmx8g -jar hap-ibd.jar
OUT="{out_prefix}_extension"

if [ ! -f "${{PHASED_VCF}}" ]; then
  echo "Error: ${{PHASED_VCF}} not found."
  echo "This is the prefix used by prepare_plink_beagle.py's -o argument, NOT"
  echo "necessarily the same as this script's own output prefix ({out_prefix})."
  echo "Find your actual phased VCF with: ls *_phased.vcf.gz"
  echo "Then re-run extend_pedigree.py with: --phased-vcf <correct_path>"
  exit 1
fi

if [ ! -f "${{MAP}}" ]; then
  echo "Error: ${{MAP}} not found."
  echo "hap-IBD requires a genetic map (unlike BEAGLE phasing, this is mandatory)."
  echo "Find your synthetic map with: ls *_synthetic.map"
  echo "Then re-run extend_pedigree.py with: --genetic-map <correct_path>"
  exit 1
fi

# ── Run hap-IBD on full phased VCF ─────────────────────────────────────────
"${{HAPIBD}}" \\
    gt="${{PHASED_VCF}}" \\
    map="${{MAP}}" \\
    out="${{OUT}}_ibd" \\
    min-seed=1.0 \\
    min-extend=0.5 \\
    min-output=1.0 \\
    nthreads=4

echo "IBD segments written to: ${{OUT}}_ibd.ibd.gz"

# ── R script: segment-length analysis for grandparent vs. half-sibling ──────
cat << 'RSCRIPT' > "{out_prefix}_extension_ibd_analysis.R"
library(data.table)
ibd <- fread("{out_prefix}_extension_ibd.ibd.gz",
    col.names=c("id1","h1","id2","h2","chrom","start","end","lod"))
ibd[, len_Mb := (end - start) / 1e6]

# Mean segment length per pair
pair_summary <- ibd[, .(
    total_Mb = sum(len_Mb),
    n_segs   = .N,
    mean_seg = mean(len_Mb),
    max_seg  = max(len_Mb)
), by=.(id1, id2)]

# Relationship classification by mean segment length
# (scale these thresholds proportionally with your genome size)
pair_summary[, rel_hint := fcase(
    mean_seg > 40, "likely HALF-SIBLING",
    mean_seg > 20, "likely AVUNCULAR or HALF-SIBLING",
    mean_seg > 5,  "likely GRANDPARENT-GRANDCHILD",
    default = "distant or unrelated"
)]

# Filter to candidate pairs
candidates <- c(
"""

    pair_lines = []
    for c in up_candidates[:30]:
        pair_lines.append(f'  "{c["founder"]}:{c["g1"]}", "{c["founder"]}:{c["g2"]}"')
    for c in down_candidates[:30]:
        pair_lines.append(f'  "{c["leaf"]}:{c["candidate_child"]}"')

    script += "\n".join(pair_lines) + "\n"
    script += f"""# ... (truncated to 30 per direction; edit as needed)
)
cat("\\nIBD summary for extension candidates:\\n")
print(pair_summary[order(-total_Mb)][1:50])
fwrite(pair_summary, "{out_prefix}_extension_ibd_summary.tsv", sep="\\t")
cat("\\nWritten: {out_prefix}_extension_ibd_summary.tsv\\n")
RSCRIPT

echo "Run: Rscript {out_prefix}_extension_ibd_analysis.R"
echo "Mean segment length > 40 Mb → half-sibling"
echo "Mean segment length 15-25 Mb → grandparent-grandchild"
echo ""
echo "Done. Cross-reference with {out_prefix}_extension_report.txt"
"""

    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Iteratively extend a pedigree up (grandparents) and down (grandchildren)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s refined.fam king_results.kin0
  %(prog)s refined.fam king_results.kin0 --ibd prefix_ibd.ibd.gz
  %(prog)s refined.fam king_results.kin0 --direction up --max-iter 2
  %(prog)s refined.fam king_results.kin0 --strategy combined
        """,
    )
    parser.add_argument('fam',
                        help='.fam file with current pedigree assignments')
    parser.add_argument('king_kin0',
                        help='PLINK2 .kin0 file from --make-king-table')
    parser.add_argument('--ibd',
                        help='hap-IBD .ibd.gz file for segment-based analysis',
                        default=None)
    parser.add_argument('--phased-vcf',
                        help='Pedigree-aware phased VCF for the generated '
                             '_grandparent_check.sh script (e.g. '
                             '<prefix>_phased.vcf.gz from prepare_plink_beagle.py / '
                             'run_beagle.sh). NOTE: this is usually a DIFFERENT '
                             'prefix than this script\'s own --output, since '
                             'phasing was run with prepare_plink_beagle.py\'s own '
                             '-o. Default: guessed as <fam-without-.fam>_phased.vcf.gz, '
                             'which is WRONG unless your prepare_plink_beagle.py '
                             '-o matched your refine_pedigree.py output exactly — '
                             'set this explicitly if the generated script reports '
                             'the phased VCF as not found.',
                        default=None)
    parser.add_argument('--genetic-map',
                        help='Genetic map for hap-IBD (REQUIRED by hap-IBD itself, '
                             'not optional). Reuse the synthetic map already '
                             'generated by prepare_plink_beagle.py\'s run_beagle.sh '
                             '(<prepare_plink_beagle_prefix>_synthetic.map), or your '
                             'own species-specific map. Default: guessed from '
                             '--phased-vcf\'s prefix, which is correct only if you '
                             'did not move/rename that file.',
                        default=None)
    parser.add_argument('--hapibd', default='hap-ibd',
                        help='Command for hap-IBD executable in the generated '
                             '_grandparent_check.sh (default: hap-ibd). Use '
                             '"java -Xmx8g -jar /path/to/hap-ibd.jar" for jar installs.')
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: derived from fam file)',
                        default=None)
    parser.add_argument('--direction', choices=['up', 'down', 'both'],
                        default='both',
                        help='Extension direction (default: both)')
    parser.add_argument('--max-iter', type=int, default=1,
                        help='Maximum extension iterations (default: 1; use 2–3 '
                             'for multi-generation search)')
    parser.add_argument('--strategy',
                        choices=['auto', 'king', 'vcfped', 'beagle', 'combined'],
                        default='auto',
                        help='Confirmation strategy (default: auto)')
    parser.add_argument('--kinship-2nd-min', type=float, default=KING_2ND_MIN,
                        help=f'Min KING kinship for 2nd degree (default: {KING_2ND_MIN})')
    parser.add_argument('--min-mz-coverage', type=float, default=0.95,
                        help='Min hap-IBD genome coverage fraction (vs. the pair\'s '
                             'own testable-genome proxy) required to collapse a '
                             'KING_MZ_MIN-level pair as a confirmed duplicate '
                             '(default: 0.95). Pairs below this are treated as '
                             'CLOSE_BUT_DISTINCT — genuinely separate individuals '
                             'whose kinship was elevated by inbreeding/backcrossing '
                             'rather than true identity — and are reported as '
                             'CLOSE_INBRED candidates instead of being collapsed. '
                             'Only used when --ibd is provided; without IBD evidence '
                             'all KING_MZ_MIN-level pairs are collapsed conservatively.')
    parser.add_argument('--fragmentation-factor', type=float, default=1.5,
                        help='A CLOSE_BUT_DISTINCT pair (see --min-mz-coverage) whose '
                             'mean hap-IBD segment length is <= this factor times the '
                             'median segment length among CONFIRMED duplicates in the '
                             'same run (default: 1.5) is reclassified as '
                             'LIKELY_DUPLICATE_FRAGMENTED instead: statistical phasing '
                             '(no pedigree information) can fragment a TRUE duplicate\'s '
                             'whole-genome IBD into many short segments, deflating its '
                             'measured coverage below --min-mz-coverage even though it '
                             'is probably the same individual. Still NOT auto-collapsed '
                             '(the heuristic is not proof), but reported distinctly so '
                             'it can be prioritised for closer inspection. Needs >= 5 '
                             'confirmed-duplicate pairs in the run to self-calibrate a '
                             'reference; otherwise this check is skipped.')
    parser.add_argument('--max-candidates', type=int, default=3,
                        help='Max candidates to RETAIN per individual founder/leaf '
                             'during the search (default: 3). This determines the '
                             'size of the candidate pool written to the .fam extension '
                             'and the vcfped_candidates.txt file. In a densely related '
                             'cohort most candidates beyond the top few are noise — '
                             'keep this small (3-5) rather than the previous default '
                             'of 50, or the candidate pool can balloon to nearly the '
                             'full cohort size.')
    parser.add_argument('--report-limit', type=int, default=50,
                        help='Max candidates to print in the human-readable text '
                             'report per direction (default: 50). Does NOT affect '
                             'the .fam extension or vcfped_candidates.txt, which '
                             'always include the full retained candidate pool.')
    parser.add_argument('--feedback-top-n', type=int, default=10,
                        help='Number of highest-confidence candidates (per '
                             'direction) to tentatively write into the working '
                             'pedigree between iterations (default: 10). This is '
                             'what enables genuine multi-generation reach with '
                             '--max-iter > 1: a newly-assigned grandparent with no '
                             'recorded parents of its own becomes a founder, '
                             'eligible for search in the next iteration (i.e. '
                             'finding the ORIGINAL founder\'s great-grandparents). '
                             'These are provisional, unconfirmed assignments used '
                             'only to unlock further search — the final '
                             '_extended.fam still incorporates the FULL candidate '
                             'pool, not just these top-N.')
    args = parser.parse_args()

    out_prefix = args.output or args.fam.replace('.fam', '')

    print("=" * 70)
    print("VCFped Pedigree Extension")
    print("=" * 70)
    print(f"  Input FAM  : {args.fam}")
    print(f"  KING file  : {args.king_kin0}")
    print(f"  IBD file   : {args.ibd or 'not provided'}")
    print(f"  Direction  : {args.direction}")
    print(f"  Max iters  : {args.max_iter}")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Min MZ coverage     : {args.min_mz_coverage}")
    print(f"  Fragmentation factor: {args.fragmentation_factor}")
    print()

    print("Loading pedigree and KING data...")
    fam  = load_fam(args.fam)
    king_raw, all_ids_raw = load_king(args.king_kin0)
    print(f"  {len(fam)} individuals in pedigree")
    print(f"  {len(king_raw):,} KING pairs, {len(all_ids_raw):,} individuals")

    ibd = {}
    ibd_segments_index = None
    if args.ibd:
        print(f"  Loading IBD segments from {args.ibd}...")
        ibd = load_ibd_summary(args.ibd)
        ibd_segments_index = load_ibd_segments_raw(args.ibd)
        print(f"  {len(ibd):,} IBD pairs loaded")

    print("  Collapsing MZ-twin/duplicate groups in KING data "
          "(prepare_plink_beagle.py deliberately runs on the pre-purge VCF, "
          "so duplicates are expected here)...")
    king, mz_remap, mz_groups, close_but_distinct_pairs = collapse_mz_twin_groups_in_king(
        king_raw, set(fam), ibd_segments_index=ibd_segments_index,
        min_coverage_for_duplicate=args.min_mz_coverage,
        fragmentation_factor=args.fragmentation_factor)
    all_ids = {mz_remap.get(s, s) for s in all_ids_raw}
    if mz_groups:
        n_collapsed = sum(len(members) - 1 for members in mz_groups.values())
        print(f"  {len(mz_groups)} MZ-twin group(s) found, "
              f"collapsing {n_collapsed} duplicate sample IDs to a single "
              f"representative each ({len(king_raw):,} -> {len(king):,} KING pairs)")
    else:
        print("  No MZ-twin groups detected in KING data")
    if close_but_distinct_pairs:
        distinct_stats    = [s for s in close_but_distinct_pairs.values()
                              if s.get('classification') == 'CLOSE_BUT_DISTINCT']
        fragmented_stats  = [s for s in close_but_distinct_pairs.values()
                              if s.get('classification') == 'LIKELY_DUPLICATE_FRAGMENTED']
        n_distinct, n_fragmented = len(distinct_stats), len(fragmented_stats)
        print(f"  {len(close_but_distinct_pairs)} pair(s) had MZ-level kinship "
              f"(>= {KING_MZ_MIN}) but were NOT collapsed:")
        print(f"    {n_distinct} confirmed genuinely DISTINCT via hap-IBD coverage "
              f"(reported as CLOSE_INBRED candidates)")
        if n_fragmented:
            ref_mb = next((s['fragmentation_reference_mb'] for s in fragmented_stats
                           if s.get('fragmentation_reference_mb') is not None), None)
            print(f"    {n_fragmented} show duplicate-like segment fragmentation, "
                  f"possibly caused by statistical-phasing switch errors rather than "
                  f"being genuinely distinct (reported as CLOSE_INBRED_POSSIBLE_DUP — "
                  f"investigate before assuming distinct)")
            if ref_mb is not None:
                print(f"      Calibrated fragmentation reference: {ref_mb:.2f} Mb "
                      f"(median mean-segment-length among confirmed duplicates; "
                      f"flagged pairs have mean_segment_Mb <= {ref_mb * args.fragmentation_factor:.2f} Mb "
                      f"= reference x --fragmentation-factor {args.fragmentation_factor})")
            # IBS0 is computed from unphased genotypes (KING), independent of
            # hap-IBD phasing quality — a useful sanity check on the
            # phasing-dependent fragmentation call. True duplicates should
            # show IBS0 ~ 0; if CLOSE_INBRED_POSSIBLE_DUP's IBS0 distribution
            # looks like CLOSE_BUT_DISTINCT's rather than like confirmed
            # duplicates', the fragmentation heuristic is likely over-firing
            # for this dataset and --fragmentation-factor should be lowered.
            def _median(vals):
                vals = sorted(vals)
                n = len(vals)
                return None if n == 0 else (vals[n // 2] if n % 2 else
                                             (vals[n // 2 - 1] + vals[n // 2]) / 2)
            ibs0_distinct   = _median([s['ibs0'] for s in distinct_stats])
            ibs0_fragmented = _median([s['ibs0'] for s in fragmented_stats])
            ibs0_distinct_str   = f"{ibs0_distinct:.4f}" if ibs0_distinct is not None else 'n/a'
            ibs0_fragmented_str = f"{ibs0_fragmented:.4f}" if ibs0_fragmented is not None else 'n/a'
            print(f"      Median IBS0 (phasing-independent, KING-derived): "
                  f"CLOSE_INBRED={ibs0_distinct_str}, "
                  f"CLOSE_INBRED_POSSIBLE_DUP={ibs0_fragmented_str} "
                  f"(close to 0 supports a true duplicate; well above 0 contradicts it)")

    print("  Building per-individual KING index "
          "(avoids O(N×pairs) full-table scans per founder/leaf)...")
    king_index = build_king_index(king)
    print()

    founders, leaves, children_of = analyse_structure(fam)
    print(f"Pedigree structure:")
    print(f"  Founders (no known parents):  {len(founders)}")
    print(f"  Leaves   (no known children): {len(leaves)}")
    print()

    # ── Iterative extension ───────────────────────────────────────────────────
    all_up_candidates   = []
    all_down_candidates = []
    current_fam = dict(fam)

    # Grandparent candidates depend ONLY on static KING data (never on
    # current_fam state), so a founder's result is identical in every
    # iteration. Cache by founder and only search NEWLY-encountered founders
    # each iteration — across max_iter>1 runs, the founder set only shrinks
    # by the handful of individuals resolved by the offspring search, so
    # without this cache, iteration 2+ re-derives ~99% identical results for
    # founders that were already fully searched, wasting time and duplicating
    # entries in the output (counted once per iteration instead of once total).
    grandparent_cache = {}
    # Offspring search's "compatible second parent" lookup is also pure KING
    # data, but the primary candidate list additionally depends on the
    # (iteration-growing) set of already-assigned children, so a leaf's
    # result could in principle change. We still cache by leaf and only
    # search new leaves each iteration: a leaf only stops being searched once
    # it leaves the `leaves` set entirely (because it gained a child itself),
    # so no leaf is ever re-evaluated against a changed all_children set.
    offspring_cache = {}

    for iteration in range(1, args.max_iter + 1):
        print(f"── Iteration {iteration} ──────────────────────────────────────")
        founders, leaves, _ = analyse_structure(current_fam)

        # Defined unconditionally: the feedback blocks below reference these
        # regardless of --direction (e.g. --direction down never populates
        # iter_up, but still needs it defined as empty to skip cleanly).
        iter_up = []
        iter_down = []

        if args.direction in ('up', 'both'):
            founders_sorted = sorted(founders)
            new_founders = [f for f in founders_sorted if f not in grandparent_cache]
            n_cached = len(founders_sorted) - len(new_founders)
            print(f"  Searching for grandparents of {len(new_founders)} founders "
                  f"({n_cached} already searched in a previous iteration)...")
            t_start = time.time()
            for n, founder in enumerate(new_founders, 1):
                cands = find_grandparent_candidates(
                    founder, king, king_index, all_ids, ibd=ibd,
                    mz_groups=mz_groups,
                    close_but_distinct_pairs=close_but_distinct_pairs)[:args.max_candidates]
                grandparent_cache[founder] = cands
                if n % 500 == 0 or n == len(new_founders):
                    elapsed = time.time() - t_start
                    rate = n / elapsed if elapsed > 0 else 0
                    remaining = (len(new_founders) - n) / rate if rate > 0 else 0
                    found_so_far = sum(len(grandparent_cache[f]) for f in new_founders[:n])
                    print(f"    {n}/{len(new_founders)} founders "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                          f"{found_so_far} candidates so far)")
            iter_up = [c for f in new_founders for c in grandparent_cache[f]]
            print(f"  Found {len(iter_up)} NEW grandparent candidate pairs "
                  f"({sum(len(grandparent_cache[f]) for f in founders_sorted)} total across all founders)")
            all_up_candidates.extend(iter_up)

        if args.direction in ('down', 'both'):
            leaves_sorted = sorted(leaves)
            new_leaves = [l for l in leaves_sorted if l not in offspring_cache]
            n_cached = len(leaves_sorted) - len(new_leaves)
            print(f"  Searching for offspring of {len(new_leaves)} leaves "
                  f"({n_cached} already searched in a previous iteration)...")
            # Compute once — was previously recomputed inside the per-leaf
            # function call, an unnecessary O(leaves x len(fam)) cost.
            all_children = {iid for iid, d in current_fam.items()
                            if d['pat'] != '0' or d['mat'] != '0'}
            t_start = time.time()
            for n, leaf in enumerate(new_leaves, 1):
                cands = find_offspring_candidates(
                    leaf, all_children, king, king_index, all_ids, ibd=ibd,
                    mz_groups=mz_groups)[:args.max_candidates]
                offspring_cache[leaf] = cands
                if n % 500 == 0 or n == len(new_leaves):
                    elapsed = time.time() - t_start
                    rate = n / elapsed if elapsed > 0 else 0
                    remaining = (len(new_leaves) - n) / rate if rate > 0 else 0
                    found_so_far = sum(len(offspring_cache[l]) for l in new_leaves[:n])
                    print(f"    {n}/{len(new_leaves)} leaves "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                          f"{found_so_far} candidates so far)")
            iter_down = [c for l in new_leaves for c in offspring_cache[l]]
            print(f"  Found {len(iter_down)} NEW offspring candidate trios "
                  f"({sum(len(offspring_cache[l]) for l in leaves_sorted)} total across all leaves)")
            all_down_candidates.extend(iter_down)

        # ── Feed UP candidates back into the pedigree (enables true multi-
        # generation reach) ────────────────────────────────────────────────
        # Without this, grandparent candidates were only merged into the
        # output .fam at the very end, AFTER all iterations — meaning a
        # newly-identified grandparent never became a searchable "founder"
        # during the loop, so --max-iter > 1 found 0 new founders in the
        # upward direction (iteration 2 only ever changed because of
        # offspring resolutions, never grandparent ones).
        #
        # Tentatively assign the top-N (by confidence) grandparent pairs as
        # the founder's parents. If a grandparent has no recorded parents of
        # its own, it becomes a founder in the NEXT iteration's
        # analyse_structure() call — i.e. we now search for the ORIGINAL
        # founder's great-grandparents. These assignments are provisional
        # (used only to unlock deeper search); the final _extended.fam still
        # incorporates the FULL candidate pool below, not just these top-N.
        if args.max_iter > 1 and iter_up:
            added_up = 0
            iter_up_sorted = sorted(iter_up, key=lambda x: -x['confidence'])
            for c in iter_up_sorted[:args.feedback_top_n]:
                founder = c['founder']
                entry = current_fam.get(founder)
                if entry is not None and entry['pat'] == '0' and entry['mat'] == '0':
                    entry['pat'] = c['g1']
                    entry['mat'] = c['g2']
                    added_up += 1
                    # Ensure the grandparents themselves are tracked so they
                    # are picked up as founders (if they have no recorded
                    # parents) when analyse_structure() runs next iteration.
                    for gp in (c['g1'], c['g2']):
                        if gp not in current_fam:
                            current_fam[gp] = {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}
            print(f"  Added {added_up} grandparent assignments to pedigree for next iteration")

        # ── Feed DOWN candidates back into the pedigree ─────────────────────
        #
        # BUG FIX: `current_fam` already contains every individual in the
        # cohort (founders/leaves are subsets of the SAME population, not an
        # external candidate pool) — `child not in current_fam` was therefore
        # always False, since candidate_child is virtually always an existing
        # entry (typically a founder with pat='0', mat='0'). The correct test
        # is whether the candidate is currently UNASSIGNED, matching the same
        # pattern used for grandparent incorporation above. Also sort by
        # confidence first: iter_down is built leaf-by-leaf in alphabetical
        # order, NOT globally sorted, so a positional slice previously took
        # the first N records (mostly from one leaf) rather than the top N
        # candidates overall.
        if args.max_iter > 1 and iter_down:
            added = 0
            iter_down_sorted = sorted(iter_down, key=lambda x: -x['confidence'])
            for c in iter_down_sorted[:args.feedback_top_n]:
                child = c['candidate_child']
                entry = current_fam.get(child)
                if entry is None:
                    current_fam[child] = {'fid': '0', 'pat': c['leaf'],
                                           'mat': c['second_parent'] or '0', 'sex': '0'}
                    added += 1
                elif entry['pat'] == '0' and entry['mat'] == '0':
                    entry['pat'] = c['leaf']
                    entry['mat'] = c['second_parent'] or '0'
                    added += 1
            print(f"  Added {added} offspring to pedigree for next iteration")
        print()

    # ── Strategy assessment ───────────────────────────────────────────────────
    recs = assess_strategy(all_up_candidates, all_down_candidates,
                            len(founders), len(leaves),
                            bool(ibd), len(fam), args.strategy)

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("Writing output files...")

    # Incorporate best candidates into extended fam.
    # NOTE: all_up_candidates / all_down_candidates are already capped per
    # founder/leaf (via args.max_candidates inside the search loop) and
    # sorted best-first within each founder/leaf's own block. The `if ...
    # == '0'` guard below means only the FIRST (highest-confidence) entry
    # per founder/leaf actually gets written — so iterating the FULL list
    # here (not a global slice) is both correct and necessary: a global
    # slice would only ever reach the first 1-2 founders processed.
    ext_fam = dict(fam)
    for c in all_up_candidates:
        founder = c['founder']
        if ext_fam.get(founder, {}).get('pat', '0') == '0':
            ext_fam[founder]['pat'] = c['g1']
            ext_fam[founder]['mat'] = c['g2']
            # Ensure grandparents are in fam
            for gp in [c['g1'], c['g2']]:
                if gp not in ext_fam:
                    ext_fam[gp] = {'fid': '0', 'pat': '0', 'mat': '0', 'sex': '0'}

    for c in all_down_candidates:
        child = c['candidate_child']
        if child not in ext_fam:
            ext_fam[child] = {'fid': '0', 'pat': c['leaf'],
                               'mat': c['second_parent'] or '0', 'sex': '0'}

    write_fam(f"{out_prefix}_extended.fam", ext_fam)

    # Confirmed-duplicates ground truth for calibrate_error_rate.py --
    # only written when the collapse was coverage-validated (--ibd given),
    # since without IBD the collapse is kinship-only and no stricter than
    # vcfped's own pairwise verdict (see collapse_mz_twin_groups_in_king()).
    if mz_groups and ibd_segments_index is not None:
        confirmed_dup_path = f"{out_prefix}_confirmed_duplicates.tsv"
        write_confirmed_duplicates(confirmed_dup_path, mz_groups)

    # vcfped candidate sample list — full retained pool, NOT re-sliced.
    # (Previously sliced to args.max_candidates total here, which silently
    # dropped >99.9% of candidates down to just the first 1-2 founders/leaves.)
    vcfped_candidates_path = f"{out_prefix}_vcfped_candidates.txt"
    write_vcfped_candidates(vcfped_candidates_path,
                             all_up_candidates,
                             all_down_candidates)

    # BEAGLE confirmation script
    beagle_script_path = f"{out_prefix}_grandparent_check.sh"
    write_beagle_check_script(beagle_script_path,
                               all_up_candidates[:30],
                               all_down_candidates[:30],
                               out_prefix, args.fam,
                               phased_vcf=args.phased_vcf,
                               genetic_map=args.genetic_map,
                               hapibd_exe=args.hapibd)

    # Full report
    report_path = f"{out_prefix}_extension_report.txt"
    with open(report_path, 'w') as rpt:
        rpt.write("VCFped Pedigree Extension Report\n" + "=" * 70 + "\n\n")
        rpt.write(f"Input FAM  : {args.fam}\n")
        rpt.write(f"KING file  : {args.king_kin0}\n")
        rpt.write(f"IBD file   : {args.ibd or 'not provided'}\n\n")

        rpt.write("Pedigree structure\n" + "-" * 40 + "\n")
        rpt.write(f"  Founders : {len(founders)}\n")
        rpt.write(f"  Leaves   : {len(leaves)}\n")
        rpt.write(f"  Total    : {len(fam)}\n\n")

        rpt.write("Strategy Recommendations\n" + "-" * 40 + "\n")
        for r in recs:
            rpt.write(f"  {r}\n")
        rpt.write("\n")

        rpt.write("Upward extension — grandparent candidates\n" + "-" * 40 + "\n")
        if len(all_up_candidates) > args.report_limit:
            rpt.write(f"(Showing {args.report_limit} of {len(all_up_candidates)} total; "
                      f"full set in {vcfped_candidates_path})\n")
        for c in all_up_candidates[:args.report_limit]:
            rpt.write(f"\nFounder: {c['founder']}\n")
            rpt.write(f"  Candidate GP pair: {c['g1']} + {c['g2']}\n")
            if c.get('g1_alternatives'):
                rpt.write(f"    ⚠ G1 ({c['g1']}) is genetically indistinguishable from "
                          f"MZ twin(s): {', '.join(c['g1_alternatives'])}. "
                          f"Cannot determine which is the true grandparent from genetic "
                          f"data alone — check breeding/sample records if identity matters.\n")
            if c.get('g2_alternatives'):
                rpt.write(f"    ⚠ G2 ({c['g2']}) is genetically indistinguishable from "
                          f"MZ twin(s): {', '.join(c['g2_alternatives'])}. "
                          f"Cannot determine which is the true grandparent from genetic "
                          f"data alone — check breeding/sample records if identity matters.\n")
            rpt.write(f"  founder↔G1: kinship={c['kinship_g1']:.3f}, "
                      f"IBS0={c['ibs0_g1']:.3f}\n")
            rpt.write(f"  founder↔G2: kinship={c['kinship_g2']:.3f}, "
                      f"IBS0={c['ibs0_g2']:.3f}\n")
            rpt.write(f"  G1↔G2: kinship={c['g1g2_kinship']:.3f}, "
                      f"IBS0={c['g1g2_ibs0']:.3f}, "
                      f"relation={c['g1g2_relation']}\n")
            if c['g1g2_relation'] == 'CLOSE_INBRED' and c.get('g1g2_ibd_coverage'):
                cov = c['g1g2_ibd_coverage']
                cov_str = f"{cov['coverage_frac']:.1%}" if cov['coverage_frac'] is not None else 'n/a'
                rpt.write(f"    ⚠ G1↔G2 kinship is at MZ-twin level (>= {KING_MZ_MIN}), but "
                          f"hap-IBD coverage is only {cov_str} of the testable genome "
                          f"({cov['n_segments']} segment(s), mean {cov['mean_segment_Mb']:.1f} Mb) "
                          f"— NOT a true duplicate, consistent with two genuinely distinct "
                          f"individuals from a highly inbred/backcrossed relationship. "
                          f"Treat with extra scrutiny.\n")
            elif c['g1g2_relation'] == 'CLOSE_INBRED_POSSIBLE_DUP' and c.get('g1g2_ibd_coverage'):
                cov = c['g1g2_ibd_coverage']
                cov_str = f"{cov['coverage_frac']:.1%}" if cov['coverage_frac'] is not None else 'n/a'
                rpt.write(f"    ⚠⚠ G1↔G2 kinship is at MZ-twin level (>= {KING_MZ_MIN}), "
                          f"measured hap-IBD coverage is only {cov_str}, BUT the segment "
                          f"pattern ({cov['n_segments']} segment(s), mean "
                          f"{cov['mean_segment_Mb']:.1f} Mb) matches the fragmentation seen "
                          f"among CONFIRMED duplicates in this run — likely caused by "
                          f"statistical-phasing switch errors splitting a true duplicate's "
                          f"IBD into short pieces, not real recombination breaks. This pair "
                          f"may actually be ONE individual counted twice. Confidence is "
                          f"penalized accordingly — verify before trusting this as two "
                          f"distinct grandparents. Cross-check: the G1↔G2 IBS0 above "
                          f"(opposite-homozygote rate) is phasing-INDEPENDENT — close to 0 "
                          f"corroborates a true duplicate, well above 0 contradicts it.\n")
            if c['ibd_hint_g1']:
                rpt.write(f"  IBD hint founder↔G1: {c['ibd_hint_g1']}\n")
            if c['ibd_hint_g2']:
                rpt.write(f"  IBD hint founder↔G2: {c['ibd_hint_g2']}\n")
            rpt.write(f"  Confidence: {c['confidence']:.3f}\n")
            rpt.write(f"  Strategy hint: ")
            if c['confidence'] > 0.15 and not args.ibd:
                rpt.write("KING evidence strong — confirm with vcfped rerun or hap-IBD\n")
            elif args.ibd:
                rpt.write("Use IBD segment length to confirm grandparent vs. half-sib\n")
            else:
                rpt.write("KING evidence weak — rerun vcfped on subset VCF\n")

        rpt.write("\nDownward extension — offspring candidates\n" + "-" * 40 + "\n")
        if len(all_down_candidates) > args.report_limit:
            rpt.write(f"(Showing {args.report_limit} of {len(all_down_candidates)} total; "
                      f"full set in {vcfped_candidates_path})\n")
        for c in all_down_candidates[:args.report_limit]:
            rpt.write(f"\nLeaf: {c['leaf']}\n")
            rpt.write(f"  Candidate child: {c['candidate_child']}\n")
            if c.get('candidate_child_alternatives'):
                rpt.write(f"    ⚠ Candidate child ({c['candidate_child']}) is genetically "
                          f"indistinguishable from MZ twin(s): "
                          f"{', '.join(c['candidate_child_alternatives'])}. "
                          f"Check breeding/sample records if identity matters.\n")
            rpt.write(f"  leaf↔child: kinship={c['kinship_leaf_child']:.3f}, "
                      f"IBS0={c['ibs0_leaf_child']:.3f}\n")
            if c['second_parent']:
                rpt.write(f"  Second parent: {c['second_parent']} "
                          f"(kinship↔child={c['kinship_sp_child']:.3f}, "
                          f"IBS0={c['ibs0_sp_child']:.3f})\n")
                if c.get('second_parent_alternatives'):
                    rpt.write(f"    ⚠ Second parent ({c['second_parent']}) is genetically "
                              f"indistinguishable from MZ twin(s): "
                              f"{', '.join(c['second_parent_alternatives'])}.\n")
                rpt.write(f"  ({c['n_second_parent_options']} second-parent options)\n")
            else:
                rpt.write(f"  Second parent: NOT FOUND in KING — may be outside cohort\n")
            if c['ibd_hint']:
                rpt.write(f"  IBD hint: {c['ibd_hint']}\n")
            rpt.write(f"  Confidence: {c['confidence']:.3f}\n")

    print(f"  Written: {report_path}")

    print()
    print("Summary:")
    print(f"  Grandparent candidate pairs: {len(all_up_candidates)}")
    print(f"  Offspring candidate trios  : {len(all_down_candidates)}")
    print()
    print("Strategy recommendations:")
    for r in recs:
        print(f"  {r}")
    print()
    print("Output files:")
    print(f"  {out_prefix}_extended.fam          — updated pedigree")
    print(f"  {vcfped_candidates_path}   — sample list for vcfped rerun")
    print(f"  {beagle_script_path}  — IBD confirmation script")
    print(f"  {report_path}  — full report")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
