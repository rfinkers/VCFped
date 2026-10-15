#!/usr/bin/env python3
"""
Flag pairs of individuals with unexpectedly large shared hap-IBD haplotype
blocks given their CURRENT pedigree relationship (or lack of one).

BACKGROUND
----------
A large shared IBD block between two individuals indicates a RECENT common
ancestor -- the more recent the ancestor, the longer/less-recombined the
shared block. If the current pedigree records NO relationship between a
pair (no path through parent-child edges within a handful of generations)
but their hap-IBD data shows a large shared block anyway, that's a signal
worth investigating: an undocumented relationship (a missing pedigree link
-- an unrecorded shared parent, or a sample mix-up), or two samples more
closely related than the pedigree currently captures. This is a first pass
at understanding germplasm structure beyond what the confirmed pedigree
records.

As elsewhere in this pipeline, "large" is population-relative, not a fixed
Mb cutoff: computed against the distribution of block sizes among OTHER
pairs the pedigree ALSO currently records as unconnected, in this same
cohort. Fixed cutoffs are unreliable here -- general background relatedness
in this germplasm is unusually high (see project history), so what counts
as a "large" shared block can only be judged relative to this cohort's own
"pedigree says unrelated" baseline, not an absolute genetics textbook value.

Pedigree "connectedness" is a coarse, undirected hop-count through the FAM's
parent-child edges (not a full relationship-type classifier distinguishing
siblings from grandparent-grandchild, etc.) -- sufficient for this specific
question ("does the current pedigree record ANY relationship here at all"),
not a replacement for refine_pedigree.py/extend_pedigree.py's KING-based
relationship inference.

POPULATION-WIDE HOTSPOTS (confirmed on real data)
--------------------------------------------------
In a densely inbred cohort, a specific genomic region can be broadly shared
(near-fixed) across a large fraction of the population -- not because any
two carriers are privately related, but because the whole population
descends from it. If that region happens to be the LARGEST segment for many
different pairs, naive rank-based flagging (by segment length alone) can't
tell "this pair shares a private recent ancestor" apart from "this pair
both happen to carry the population's common haplotype at this locus" --
and a huge plateau of pairs tied at the identical segment length can
distort the percentile ranking itself. This was directly observed on a real
cohort: tens of thousands of pairs flagged, most reporting the exact same
"largest segment" length, which is the signature of a shared regional
haplotype, not tens of thousands of coincidentally-large individual
findings.

CHROMOSOME-SCALE hotspots specifically (also confirmed on real data) need
per-CHROMOSOME detection, not binning by exact coordinates or by coverage
fraction -- both were tried and both still fragmented a single real event.
Exact-coordinate matching (even 1 Mb binned) split one near-whole-chromosome
conservation event into 15 "different" detected regions
(chr01:4000000-44000000, chr01:4000000-45000000, chr01:2000000-44000000,
...). Grouping by coverage FRACTION instead (rounded to a bucket) still
split real values like 99.6%/95.0%/96.0% coverage of the same chromosome
across a bucket boundary -- a continuous real-world distribution always
straddles SOME bucket edge. The question that actually matters is "does an
unusually large number of distinct pairs share a huge chunk of THIS
chromosome," not "do they share precisely the same huge chunk," so once a
pair's largest-segment REGION covers at least --hotspot-min-coverage of its
chromosome's observed span (self-calibrated chromosome length from the
largest observed segment end on that chromosome -- same "genome proxy"
convention as extend_pedigree.py's build_ibd_genome_proxy()), it's grouped
by CHROMOSOME ALONE. Any chromosome recurring across an unusually large
number of DISTINCT pairs (--hotspot-percentile of this cohort's own
per-chromosome candidate-pair-count distribution, with a floor of
--min-hotspot-pairs) is treated as a population-wide hotspot: pairs whose
largest segment falls there are reported (never hidden) but NOT counted as
"unexpected" -- and are excluded from the percentile computation for the
pairs that remain, so a small number of genuinely surprising pairs doesn't
just contaminate the ranking of everyone else.

LOCUS-SCALE hotspots are a SEPARATE, complementary detection pass (also
confirmed on real data): a specific sub-chromosomal region -- far too small
to ever cover --hotspot-min-coverage of its chromosome, so invisible to the
chromosome-scale mechanism above -- can still be shared as the largest
segment by thousands of distinct pairs (a real cohort showed a ~39.6 Mb
block shared, to within a few hundred bp, by 6,604 pairs). Because this
jitter is orders of magnitude smaller than the chromosome-scale case's
megabase-scale jitter, these are grouped by RECIPROCAL OVERLAP
(--locus-min-overlap, default 50%) rather than by chromosome -- see
detect_locus_hotspots() for why a single grouping strategy can't serve both
scales at once. Detected loci are reported separately from detected
hotspot chromosomes, and their pairs are folded into the same excluded-
from-ranking / COMMON_HOTSPOT_REGION treatment as chromosome-scale hotspot
pairs.

INPUT
-----
  <fam_path>  — current pedigree (.fam format)
  <ibd_path>  — hap-IBD .ibd.gz/.ibd output (pairwise segments, id1 != id2)

OUTPUT
------
  <output>_ibd_block_analysis.tsv — every pair with IBD data: pedigree status
                                     (connected/unconnected), total shared Mb,
                                     largest segment Mb + region, segment
                                     count, percentile rank (within its own
                                     bucket, hotspot pairs excluded), flagged,
                                     hotspot flag, reason
  <output>_ibd_block_report.txt   — distribution summary + detected hotspot
                                     regions + flagged pairs

Usage:
  python analyze_ibd_blocks.py step3_named_filtered_refined_merged2.fam step3_named_filtered_stat_ibd.ibd.gz
  python analyze_ibd_blocks.py merged2.fam stat_ibd.ibd.gz -o blocks --max-hops 6 --outlier-percentile 95
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict, deque, Counter


def load_fam_graph(fam_path):
    """
    Parse a .fam file into an undirected adjacency dict of parent-child
    edges: {iid: set(parents) | set(children)}. Used only to answer "is
    there ANY pedigree path between these two individuals within a few
    generations" -- not a full relationship-type classifier.
    """
    if not os.path.exists(fam_path):
        sys.exit(f"Error: {fam_path} not found.")
    graph = defaultdict(set)
    with open(fam_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iid, pat, mat = parts[1], parts[2], parts[3]
            for parent in (pat, mat):
                if parent != '0':
                    graph[iid].add(parent)
                    graph[parent].add(iid)
    return graph


def bfs_connected_within(graph, start, max_hops, cache):
    """
    BFS from `start` through the undirected pedigree graph, up to max_hops
    edges. Memoized per starting node (cache), since the same individual
    appears as one side of many IBD pairs.
    """
    if start in cache:
        return cache[start]
    visited = {start}
    frontier = deque([(start, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for neighbor in graph.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
    visited.discard(start)
    cache[start] = visited
    return visited


def is_pedigree_connected(graph, a, b, max_hops, cache):
    return b in bfs_connected_within(graph, a, max_hops, cache)


def load_ibd_pair_stats(ibd_path, min_lod=3.0):
    """
    Parse hap-IBD output into per-pair stats: {(id1,id2) sorted: {total_bp,
    n_segments, max_segment_bp, max_segment_region}}. max_segment_region is
    (chrom, start, end) of whichever segment set max_segment_bp -- needed to
    detect population-wide hotspot regions (see module docstring). Same
    min_lod filtering convention as extend_pedigree.py's load_ibd_summary().
    Rows with id1==id2 (HBD/self segments -- see analyze_hbd.py) are
    skipped, since this tool is about sharing BETWEEN individuals.
    """
    if not os.path.exists(ibd_path):
        sys.exit(f"Error: {ibd_path} not found.")

    pair_data = defaultdict(lambda: {'total_bp': 0, 'n_segments': 0,
                                      'max_segment_bp': 0, 'max_segment_region': None})
    opener = gzip.open if ibd_path.endswith('.gz') else open
    with opener(ibd_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            id1, _, id2, _, chrom, start, end, lod = parts[:8]
            if id1 == id2:
                continue
            try:
                start = int(start)
                end = int(end)
                lod = float(lod)
            except ValueError:
                continue
            if lod < min_lod:
                continue
            seg_len = end - start
            key = tuple(sorted([id1, id2]))
            d = pair_data[key]
            d['total_bp'] += seg_len
            d['n_segments'] += 1
            if seg_len > d['max_segment_bp']:
                d['max_segment_bp'] = seg_len
                d['max_segment_region'] = (chrom, start, end)

    return dict(pair_data)


def classify_pairs(pair_stats, graph, max_hops):
    """Return dict (id1,id2) -> 'connected' | 'unconnected'."""
    cache = {}
    status = {}
    for (a, b) in pair_stats:
        status[(a, b)] = ('connected' if is_pedigree_connected(graph, a, b, max_hops, cache)
                          else 'unconnected')
    return status


def _percentile_value(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return None
    idx = min(n - 1, int(p / 100 * n))
    return sorted_values[idx]


def _percentile_rank(values_by_key):
    """Cohort-relative percentile RANK (0-100), same rank-based approach as
    analyze_hbd.py -- avoids the small-cohort boundary case a value-threshold
    approach misses."""
    keys_sorted = sorted(values_by_key.keys(), key=lambda k: values_by_key[k])
    n = len(keys_sorted)
    ranks = {}
    for i, k in enumerate(keys_sorted):
        ranks[k] = (i / (n - 1) * 100) if n > 1 else 50.0
    return ranks


def detect_hotspot_regions(pair_stats, hotspot_pair_fraction=0.01,
                           min_pairs_for_hotspot=None, min_coverage_frac=0.3):
    """
    Detect CHROMOSOME-SCALE hotspots: chromosomes where an unusually large
    number of DISTINCT pairs share a segment covering at least
    min_coverage_frac of that chromosome's observed span -- e.g. a
    near-whole-chromosome conserved haplotype (plausible in clonally-
    propagated or narrow-founder-base germplasm).

    Grouped by CHROMOSOME ALONE once a segment clears min_coverage_frac --
    NOT by exact (chrom, start, end), and NOT by a further coverage-fraction
    bucket either. Both of those were tried and both still fragmented a
    single real chromosome-scale event across multiple near-duplicate
    buckets, confirmed on real data: exact-coordinate matching (even 1 Mb
    binned) split one event into 15 "different" regions
    (chr01:4000000-44000000, chr01:4000000-45000000, ...); a 10%-wide
    coverage-fraction bucket STILL split values like 99.6%/95.0%/96.0%
    coverage of the same chromosome across a 0.9/1.0 bucket boundary. Real
    segment lengths (from statistical phasing) never land on exactly the
    same fraction even when they're all "basically the whole chromosome" --
    there's always some bucket boundary a continuous distribution can
    straddle. The question that matters is "does an unusually large number
    of distinct pairs share a huge chunk of THIS chromosome," not "do they
    share precisely the same huge chunk" -- so once a segment is big enough
    to count at all (min_coverage_frac), only the chromosome identity is
    used for hotspot grouping.

    Chromosome length is self-calibrated from the data itself -- the largest
    observed segment end on that chromosome across all pairs' max segments
    -- same "genome proxy" convention as extend_pedigree.py's
    build_ibd_genome_proxy(), since no external reference is assumed.

    min_coverage_frac: segments covering LESS than this fraction of their
    chromosome are never hotspot candidates, regardless of how often they
    recur. Small segments recurring across many pairs is the NORMAL
    background case (most pairs' max segment is small relative to a whole
    chromosome) -- not a population-wide near-whole-chromosome conservation
    event. Without this floor, a spread of many small segments from
    different pairs inflates the "genome proxy" chromosome-length estimate,
    making each individual small segment look like a tiny fraction of that
    inflated length, and huge numbers of ordinary small segments cluster
    into a spurious near-0%-coverage "hotspot" (confirmed while testing
    this exact function).

    min_pairs_for_hotspot: absolute count threshold. If not given, self-
    calibrated as hotspot_pair_fraction of the TOTAL number of pairs with IBD
    data (with a floor of 10) -- NOT a percentile of the per-chromosome
    candidate-count distribution. A real genome only has a handful of
    chromosomes (~10-20), and a percentile computed over that few values is
    statistically degenerate: confirmed on real data, the 99th percentile of
    ~15 chromosome counts just picked out the single largest chromosome's own
    count as the threshold, silently excluding every OTHER chromosome that
    also showed strong widespread sharing (visible instead, misleadingly
    fragmented, in detect_locus_hotspots()'s output) from ever being
    recognized as a chromosome-scale hotspot at all. Calibrating against the
    total pair population instead -- large enough for a fraction-based
    threshold to be meaningful -- avoids that degeneracy.

    Returns (hotspot_pairs: set of pair keys, region_counts: dict
    chrom -> distinct candidate pair count).
    """
    chrom_max_end = defaultdict(int)
    for s in pair_stats.values():
        if s['max_segment_region'] is None:
            continue
        chrom, _, end = s['max_segment_region']
        chrom_max_end[chrom] = max(chrom_max_end[chrom], end)

    region_counts = Counter()
    pair_region = {}
    for key, s in pair_stats.items():
        if s['max_segment_region'] is None:
            continue
        chrom, start, end = s['max_segment_region']
        chrom_len = chrom_max_end[chrom]
        coverage_frac = (end - start) / chrom_len if chrom_len > 0 else 0.0
        if coverage_frac < min_coverage_frac:
            continue
        region_counts[chrom] += 1
        pair_region[key] = chrom

    if not region_counts:
        return set(), {}

    if min_pairs_for_hotspot is None:
        min_pairs_for_hotspot = max(round(hotspot_pair_fraction * len(pair_stats)), 10)

    hotspot_chroms = {c for c, cnt in region_counts.items() if cnt >= min_pairs_for_hotspot}
    hotspot_pairs = {key for key, c in pair_region.items() if c in hotspot_chroms}
    return hotspot_pairs, dict(region_counts)


def detect_locus_hotspots(pair_stats, min_overlap_frac=0.5, hotspot_percentile=99,
                          min_pairs_for_hotspot=None, exclude_pairs=None):
    """
    Detect LOCUS-SCALE hotspots: a specific, tightly-bounded sub-chromosomal
    region shared as the largest segment by an unusually large number of
    distinct pairs -- e.g. a conserved haplotype block under selection, or a
    recombination cold-spot -- as opposed to detect_hotspot_regions()'s
    near-whole-chromosome case.

    Confirmed on real data: a ~39.6 Mb block at chr02:264222-39828158 was
    shared (to within a few hundred bp) by 6,604 distinct pairs, but never
    reached detect_hotspot_regions() at all because it covered well under
    30% of chr02's much longer observed span -- a real recurring conserved
    region, invisible to the chromosome-coverage-floor mechanism because
    that mechanism is deliberately restricted to (near-)whole-chromosome
    events.

    The two failure modes need different grouping strategies because their
    boundary jitter is a different order of magnitude: a chromosome-scale
    event's segment boundaries can jitter by MEGABASES between pairs (see
    detect_hotspot_regions()'s docstring), while this real locus-scale event
    jittered by only a few hundred bp. Exact/fixed-bin coordinate matching
    would misfire again on some other locus with wider jitter than that, so
    segments are grouped by RECIPROCAL OVERLAP instead (both segments must
    cover >= min_overlap_frac of each other) -- this adapts to whatever
    jitter magnitude a given real locus happens to have, rather than
    assuming one bin width fits every locus.

    Segments are sorted by (start, end) per chromosome and swept once,
    comparing each segment only to the single currently-open cluster's SEED
    (its first member's own interval, kept fixed rather than growing with
    each merge, to avoid a slow drift across a long chain of only-barely-
    overlapping segments) -- O(n log n) instead of the O(n^2) an all-pairs
    reciprocal-overlap comparison would cost, which matters at millions of
    pairs.

    exclude_pairs: pair keys to skip entirely -- e.g. pairs already explained
    by detect_hotspot_regions()'s chromosome-scale mechanism. Without this,
    a pair whose segment covers most of a chromosome (already counted as a
    chromosome-scale hotspot pair) also gets swept into whatever locus
    cluster its own (start, end) happens to fall into, double-reporting the
    same underlying pairs as extra, spurious "loci". Confirmed on real data:
    a single near-whole-chromosome event, already 100% explained by
    detect_hotspot_regions(), was independently rediscovered here as 9
    different chr01 "loci" (each sharing the SAME ~40.9 Mb right boundary
    within ~200bp, but with widely varying left boundaries spanning several
    Mb) purely because the fixed-seed sweep's reciprocal-overlap test breaks
    the chain once a segment's start has drifted far enough from the seed --
    the exact same megabase-scale-jitter failure mode
    detect_hotspot_regions() exists to handle, just one level down. Passing
    those pairs' keys here so they're skipped keeps this function scoped to
    genuinely sub-chromosomal recurring regions.

    Returns (hotspot_pairs: set of pair keys, region_counts: dict
    "chrom:seed_start-seed_end" -> distinct pair count).
    """
    exclude_pairs = exclude_pairs or set()
    by_chrom = defaultdict(list)
    for key, s in pair_stats.items():
        if key in exclude_pairs:
            continue
        region = s.get('max_segment_region')
        if region is None:
            continue
        chrom, start, end = region
        by_chrom[chrom].append((start, end, key))

    clusters = []
    for chrom, segs in by_chrom.items():
        segs.sort()
        current = None
        for start, end, key in segs:
            if current is not None:
                c_start, c_end = current['seed']
                overlap = min(end, c_end) - max(start, c_start)
                seg_len = end - start
                c_len = c_end - c_start
                if (overlap > 0 and seg_len > 0 and c_len > 0
                        and overlap / seg_len >= min_overlap_frac
                        and overlap / c_len >= min_overlap_frac):
                    current['keys'].add(key)
                    continue
                clusters.append(current)
            current = {'chrom': chrom, 'seed': (start, end), 'keys': {key}}
        if current is not None:
            clusters.append(current)

    region_counts = {f"{c['chrom']}:{c['seed'][0]}-{c['seed'][1]}": len(c['keys']) for c in clusters}
    if not region_counts:
        return set(), {}

    if min_pairs_for_hotspot is None:
        counts_sorted = sorted(region_counts.values())
        threshold = _percentile_value(counts_sorted, hotspot_percentile)
        min_pairs_for_hotspot = max(threshold, 10)

    hotspot_pairs = set()
    for c in clusters:
        if len(c['keys']) >= min_pairs_for_hotspot:
            hotspot_pairs |= c['keys']
    return hotspot_pairs, region_counts


def flag_outliers(pair_stats, status, hotspot_pairs, outlier_percentile=95):
    """
    Rank-based outlier flagging, computed SEPARATELY within each pedigree-
    status bucket (connected/unconnected) -- a large block is only
    surprising relative to what OTHER pairs in the SAME bucket show.
    Pairs in a detected population-wide hotspot region are excluded from
    the ranking pool entirely (so they don't distort percentiles for the
    pairs that remain) and are never flagged as "unexpected" -- they're
    still reported, tagged distinctly, never hidden.
    Returns dict (id1,id2) -> (flagged: bool, reason: str, percentile_rank:
    float or None for hotspot pairs).
    """
    by_bucket = defaultdict(dict)
    for key, s in status.items():
        if key in hotspot_pairs:
            continue
        by_bucket[s][key] = pair_stats[key]['max_segment_bp']

    ranks_by_bucket = {bucket: _percentile_rank(vals) for bucket, vals in by_bucket.items()}

    result = {}
    for key, s in status.items():
        max_mb = pair_stats[key]['max_segment_bp'] / 1e6
        if key in hotspot_pairs:
            reason = (f"COMMON_HOTSPOT_REGION ({max_mb:.1f} Mb largest segment falls in a "
                     f"population-wide conserved region shared with many other pairs -- "
                     f"not evidence of private relatedness)")
            result[key] = (False, reason, None)
            continue
        rank = ranks_by_bucket[s][key]
        if rank >= outlier_percentile:
            reason = (f"UNEXPECTED_LARGE_BLOCK (percentile rank {rank:.0f} >= "
                     f"p{outlier_percentile} within '{s}' pairs, {max_mb:.1f} Mb largest segment)")
            if s == 'unconnected':
                reason += " -- no pedigree path recorded; possible undocumented relationship"
            result[key] = (True, reason, rank)
        else:
            result[key] = (False, '', rank)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Flag pairs with unexpectedly large shared hap-IBD blocks given their pedigree relationship',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_named_filtered_refined_merged2.fam step3_named_filtered_stat_ibd.ibd.gz
  %(prog)s merged2.fam stat_ibd.ibd.gz -o blocks --max-hops 6 --outlier-percentile 95
        """,
    )
    parser.add_argument('fam_path', help='Current pedigree .fam file')
    parser.add_argument('ibd_path', help='hap-IBD .ibd.gz/.ibd file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output prefix (default: derived from ibd_path)')
    parser.add_argument('--max-hops', type=int, default=6,
                        help='Max pedigree-graph hops to search for a connection (default: 6)')
    parser.add_argument('--min-lod', type=float, default=3.0,
                        help='Minimum LOD score to keep a segment (default: 3.0)')
    parser.add_argument('--outlier-percentile', type=float, default=95,
                        help='Percentile above which a block is flagged, within its own '
                             'pedigree-status bucket (default: 95)')
    parser.add_argument('--hotspot-pair-fraction', type=float, default=0.01,
                        help='Fraction of ALL pairs with IBD data above which a chromosome\'s '
                             '>=min-coverage candidate-pair count makes it a chromosome-scale '
                             'hotspot (default: 0.01 = 1%%). Calibrated against the total pair '
                             'population, not a percentile of the handful of chromosomes in a '
                             'genome -- see detect_hotspot_regions() docstring for why the '
                             'latter is statistically degenerate')
    parser.add_argument('--hotspot-percentile', type=float, default=99,
                        help='Percentile of this cohort\'s own per-locus candidate-pair-count '
                             'distribution above which a locus-scale region is a hotspot '
                             '(default: 99)')
    parser.add_argument('--min-hotspot-pairs', type=int, default=None,
                        help='Absolute minimum distinct-pair count for a chromosome or locus to '
                             'be a hotspot, overriding --hotspot-pair-fraction/--hotspot-percentile '
                             '(default: self-calibrated, floor of 10)')
    parser.add_argument('--hotspot-min-coverage', type=float, default=0.3,
                        help='Minimum fraction of a chromosome a segment must cover to be a '
                             'chromosome-scale hotspot candidate at all -- small segments '
                             'recurring across many pairs is normal background, not a '
                             'chromosome-scale conservation event (default: 0.3 = 30%%)')
    parser.add_argument('--locus-min-overlap', type=float, default=0.5,
                        help='Minimum reciprocal overlap fraction for two pairs\' largest '
                             'segments to be grouped as the same locus-scale hotspot '
                             'candidate region (default: 0.5 = 50%%)')
    parser.add_argument('--report-limit', type=int, default=200,
                        help='Max pairs/regions to list in the report (default: 200)')
    args = parser.parse_args()

    base = args.ibd_path
    for suffix in ('.ibd.gz', '.ibd'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    out_prefix = args.output or base

    print("=" * 70)
    print("VCFped Haplotype Block Analysis")
    print("=" * 70)
    print(f"  FAM file : {args.fam_path}")
    print(f"  IBD file : {args.ibd_path}")
    print()

    graph = load_fam_graph(args.fam_path)
    print(f"  {len(graph)} individuals with pedigree edges")

    pair_stats = load_ibd_pair_stats(args.ibd_path, min_lod=args.min_lod)
    if not pair_stats:
        sys.exit(f"Error: no IBD pairs found in {args.ibd_path} (after --min-lod {args.min_lod} filter).")
    print(f"  {len(pair_stats)} pairs with IBD data")

    status = classify_pairs(pair_stats, graph, args.max_hops)
    n_connected = sum(1 for s in status.values() if s == 'connected')
    n_unconnected = len(status) - n_connected
    print(f"  {n_connected} pairs pedigree-connected (within {args.max_hops} hops), "
          f"{n_unconnected} unconnected")

    chrom_hotspot_pairs, region_counts = detect_hotspot_regions(
        pair_stats, hotspot_pair_fraction=args.hotspot_pair_fraction,
        min_pairs_for_hotspot=args.min_hotspot_pairs, min_coverage_frac=args.hotspot_min_coverage)
    locus_hotspot_pairs, locus_region_counts = detect_locus_hotspots(
        pair_stats, min_overlap_frac=args.locus_min_overlap,
        hotspot_percentile=args.hotspot_percentile, min_pairs_for_hotspot=args.min_hotspot_pairs,
        exclude_pairs=chrom_hotspot_pairs)
    hotspot_pairs = chrom_hotspot_pairs | locus_hotspot_pairs
    print(f"  {len(chrom_hotspot_pairs)} pairs fall in a detected chromosome-scale hotspot")
    print(f"  {len(locus_hotspot_pairs)} pairs fall in a detected locus-scale hotspot")
    print(f"  {len(hotspot_pairs)} pairs fall in a detected population-wide hotspot region (either kind)")
    print()

    flags = flag_outliers(pair_stats, status, hotspot_pairs, args.outlier_percentile)
    flagged_keys = [k for k, (f, _, _) in flags.items() if f]
    n_flagged_unconnected = sum(1 for k in flagged_keys if status[k] == 'unconnected')
    print(f"  {len(flagged_keys)} pairs flagged ({n_flagged_unconnected} pedigree-unconnected)")

    analysis_path = f"{out_prefix}_ibd_block_analysis.tsv"
    with open(analysis_path, 'w') as f:
        f.write('\t'.join(['ID1', 'ID2', 'Pedigree_Status', 'Total_Mb', 'Max_Segment_Mb',
                           'Max_Segment_Region', 'N_Segments', 'Percentile_Rank',
                           'Flagged', 'Hotspot', 'Reason']) + '\n')
        for key in sorted(pair_stats.keys(), key=lambda k: -pair_stats[k]['max_segment_bp']):
            s = pair_stats[key]
            flagged, reason, rank = flags[key]
            region = s['max_segment_region']
            region_str = f"{region[0]}:{region[1]}-{region[2]}" if region else 'NA'
            f.write('\t'.join([
                key[0], key[1], status[key],
                f"{s['total_bp']/1e6:.3f}", f"{s['max_segment_bp']/1e6:.3f}", region_str,
                str(s['n_segments']), f"{rank:.1f}" if rank is not None else 'NA',
                'YES' if flagged else 'no', 'YES' if key in hotspot_pairs else 'no', reason,
            ]) + '\n')
    print(f"  Written: {analysis_path}")

    report_path = f"{out_prefix}_ibd_block_report.txt"
    with open(report_path, 'w') as rpt:
        rpt.write("VCFped Haplotype Block Analysis Report\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"FAM file : {args.fam_path}\n")
        rpt.write(f"IBD file : {args.ibd_path}\n\n")
        rpt.write(f"Pairs with IBD data      : {len(pair_stats)}\n")
        rpt.write(f"Pedigree-connected       : {n_connected}\n")
        rpt.write(f"Pedigree-unconnected     : {n_unconnected}\n")
        rpt.write(f"Pairs in a hotspot region (either kind): {len(hotspot_pairs)}\n")
        rpt.write(f"  chromosome-scale : {len(chrom_hotspot_pairs)}\n")
        rpt.write(f"  locus-scale      : {len(locus_hotspot_pairs)}\n\n")

        top_regions = sorted(region_counts.items(), key=lambda kv: -kv[1])[:args.report_limit]
        min_thresh = args.min_hotspot_pairs or max(
            round(args.hotspot_pair_fraction * len(pair_stats)), 10)
        rpt.write(f"Detected hotspot chromosomes (>= {min_thresh:.0f} distinct pairs "
                  f"[{args.hotspot_pair_fraction:.1%} of all pairs], largest segment covering "
                  f">= {args.hotspot_min_coverage:.0%} of the chromosome)\n" + "-" * 40 + "\n")
        any_hotspot = False
        for chrom, count in top_regions:
            if count < min_thresh:
                continue
            any_hotspot = True
            rpt.write(f"  {chrom} — largest segment for {count} distinct pairs\n")
        if not any_hotspot:
            rpt.write("  (none)\n")
        rpt.write("\n")

        locus_top_regions = sorted(locus_region_counts.items(), key=lambda kv: -kv[1])[:args.report_limit]
        locus_min_thresh = args.min_hotspot_pairs or max(
            _percentile_value(sorted(locus_region_counts.values()), args.hotspot_percentile) or 0, 10)
        rpt.write(f"Detected hotspot loci (>= {locus_min_thresh:.0f} distinct pairs, "
                  f"reciprocal overlap >= {args.locus_min_overlap:.0%})\n" + "-" * 40 + "\n")
        any_locus_hotspot = False
        for region, count in locus_top_regions:
            if count < locus_min_thresh:
                continue
            any_locus_hotspot = True
            rpt.write(f"  {region} — shared by {count} distinct pairs\n")
        if not any_locus_hotspot:
            rpt.write("  (none)\n")
        rpt.write("\n")

        rpt.write(f"Flagged pairs ({len(flagged_keys)})\n" + "-" * 40 + "\n")
        flagged_sorted = sorted(flagged_keys, key=lambda k: -pair_stats[k]['max_segment_bp'])
        for key in flagged_sorted[:args.report_limit]:
            _, reason, _ = flags[key]
            rpt.write(f"  {key[0]} <-> {key[1]}: {reason}\n")
        if not flagged_keys:
            rpt.write("  (none)\n")
        if len(flagged_sorted) > args.report_limit:
            rpt.write(f"  ... and {len(flagged_sorted) - args.report_limit} more "
                      f"(full list in {analysis_path})\n")

    print(f"  Written: {report_path}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
