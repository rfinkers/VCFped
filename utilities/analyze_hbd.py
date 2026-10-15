#!/usr/bin/env python3
"""
Analyze per-individual homozygosity-by-descent (HBD) segments from hap-IBD output.

BACKGROUND
----------
prepare_plink_beagle.py's generated _run_beagle.sh already produces
<prefix>_ibd.hbd.gz (self-IBD / autozygosity segments: same 8-column hap-IBD
format as regular IBD, but id1==id2 — pairs of the two haplotypes of the SAME
individual that are identical by descent) plus an R summary script. Until
now nothing in the Python pipeline consumed this file.

In a cohort of deep-selfed lines (e.g. F6), high genome-wide autozygosity is
the EXPECTED baseline, not a curiosity — so this data is diagnostic in two
ways:

  (a) An individual with UNEXPECTEDLY LOW autozygosity relative to the
      cohort's own distribution is inconsistent with the claimed breeding
      history (deep selfing) and is a contamination/mislabeling signal —
      complementary to, but independent of, PLINK2's frequency-based --het
      inbreeding estimate (see check_genotype_quality.py).
  (b) For a candidate pair extend_pedigree.py flagged as CLOSE_INBRED vs
      CLOSE_INBRED_POSSIBLE_DUP (using IBD coverage + segment-length
      fragmentation alone), each member's own autozygosity level is an
      extra, independent data point: two genuinely distinct deep-selfed
      lines should each show cohort-typical (usually high) autozygosity;
      a member with atypically low autozygosity is harder to reconcile
      with being a genuine deep-selfed founder.

As with every other threshold in this pipeline, outlier flagging is
population-relative (percentiles computed from this cohort's own
distribution), not a fixed absolute cutoff — a cohort of largely inbred
lines has a very different "normal" than a cohort of outbred individuals.

INPUT
-----
  <hbd_path>  — hap-IBD .hbd.gz (or plain .hbd) output, 8 tab-separated
                columns with no header: id1 h1 id2 h2 chrom start end lod

OUTPUT
------
  <output>_hbd_analysis.tsv  — every individual: total HBD Mb, segment count,
                                mean segment Mb, percentile rank, flagged, reason
  <output>_hbd_report.txt    — distribution summary + flagged individuals
  If --check-pairs is given, the report also includes a side-by-side
  comparison for each listed pair.

Usage:
  python analyze_hbd.py step3_named_filtered_ibd.hbd.gz
  python analyze_hbd.py step3_named_filtered_ibd.hbd.gz -o hbd --low-percentile 5 --outlier-percentile 95
  python analyze_hbd.py step3_named_filtered_ibd.hbd.gz --check-pairs candidate_pairs.txt
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict


def load_hbd(path, min_lod=3.0):
    """
    Parse a hap-IBD .hbd.gz/.hbd file into per-individual segment lists.
    Returns dict: id -> list of segment lengths (bp). Rows below min_lod
    are dropped (same convention as extend_pedigree.py's load_ibd_summary).
    A row's id2 is expected to equal id1 (that's what makes it HBD rather
    than regular IBD) but this isn't hard-enforced -- id1 is used as the
    grouping key either way.
    """
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")

    by_individual = defaultdict(list)
    id_mismatches = 0
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            id1, _, id2, _, _, start, end, lod = parts[:8]
            try:
                start = int(start)
                end = int(end)
                lod = float(lod)
            except ValueError:
                continue
            if lod < min_lod:
                continue
            if id1 != id2:
                id_mismatches += 1
            by_individual[id1].append(end - start)

    return dict(by_individual), id_mismatches


def summarize_individual(segments):
    total_bp = sum(segments)
    n_segments = len(segments)
    return {
        'total_bp': total_bp,
        'total_Mb': total_bp / 1e6,
        'n_segments': n_segments,
        'mean_segment_Mb': (total_bp / n_segments) / 1e6 if n_segments else 0.0,
    }


def build_individual_stats(by_individual):
    return {iid: summarize_individual(segs) for iid, segs in by_individual.items()}


def _percentile_value(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return None
    idx = min(n - 1, int(p / 100 * n))
    return sorted_values[idx]


def compute_percentile_ranks(stats):
    """Return dict id -> percentile rank (0-100) of its total_Mb within the cohort."""
    sorted_ids = sorted(stats.keys(), key=lambda i: stats[i]['total_Mb'])
    n = len(sorted_ids)
    ranks = {}
    for rank, iid in enumerate(sorted_ids):
        ranks[iid] = (rank / (n - 1) * 100) if n > 1 else 50.0
    return ranks


def flag_outliers(stats, ranks, low_percentile=5, outlier_percentile=95):
    """
    Flag individuals whose percentile RANK (not raw value) falls at or
    below low_percentile or at or above outlier_percentile. Rank-based
    (rather than value-threshold-based) so the single most extreme
    individual in a small cohort is reliably flagged -- a value-threshold
    approach can have the top item's own value equal its own percentile
    threshold, which a strict inequality then misses.
    Returns dict id -> (flagged: bool, reason: str).
    """
    flags = {}
    for iid, s in stats.items():
        rank = ranks[iid]
        mb = s['total_Mb']
        if rank <= low_percentile:
            flags[iid] = (True, f'LOW (percentile rank {rank:.0f} <= p{low_percentile}, {mb:.1f} Mb) '
                                 f'-- inconsistent with expected deep-selfing baseline')
        elif rank >= outlier_percentile:
            flags[iid] = (True, f'HIGH (percentile rank {rank:.0f} >= p{outlier_percentile}, {mb:.1f} Mb)')
        else:
            flags[iid] = (False, '')
    return flags


def load_pairs(path):
    """Parse a simple pair list file: one pair per line, comma- or tab-separated."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', '\t').split('\t')
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def compare_pair(a, b, stats, ranks, flags):
    """Build a side-by-side note for a candidate pair (e.g. CLOSE_INBRED vs
    CLOSE_INBRED_POSSIBLE_DUP from extend_pedigree.py's classification)."""
    a_stats = stats.get(a)
    b_stats = stats.get(b)
    if a_stats is None or b_stats is None:
        missing = a if a_stats is None else b
        return f"{a} <-> {b}: NA -- no HBD data for {missing}"

    a_mb, b_mb = a_stats['total_Mb'], b_stats['total_Mb']
    a_rank, b_rank = ranks[a], ranks[b]
    a_flagged, a_reason = flags[a]
    b_flagged, b_reason = flags[b]
    low_flag_a = a_flagged and a_reason.startswith('LOW')
    low_flag_b = b_flagged and b_reason.startswith('LOW')

    if low_flag_a or low_flag_b:
        low_one = a if low_flag_a else b
        note = (f"atypically low autozygosity for {low_one} -- inconsistent with "
                f"expected deep-selfing baseline, worth investigating (contamination/mislabeling)")
    else:
        note = "both cohort-typical -- consistent with genuinely distinct inbred lines"

    return (f"{a} ({a_mb:.1f} Mb, p{a_rank:.0f}) <-> {b} ({b_mb:.1f} Mb, p{b_rank:.0f}): {note}")


ANALYSIS_TSV_HEADER = ['ID', 'HBD_Mb', 'N_Segments', 'Mean_Segment_Mb',
                       'Percentile_Rank', 'Flagged', 'Reason']


def write_analysis_tsv(path, stats, ranks, flags):
    with open(path, 'w') as f:
        f.write('\t'.join(ANALYSIS_TSV_HEADER) + '\n')
        for iid in sorted(stats.keys(), key=lambda i: -stats[i]['total_Mb']):
            s = stats[iid]
            flagged, reason = flags[iid]
            f.write('\t'.join([
                iid,
                f"{s['total_Mb']:.3f}",
                str(s['n_segments']),
                f"{s['mean_segment_Mb']:.3f}",
                f"{ranks[iid]:.1f}",
                'YES' if flagged else 'no',
                reason,
            ]) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze per-individual homozygosity-by-descent (HBD) segments from hap-IBD output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_named_filtered_ibd.hbd.gz
  %(prog)s step3_named_filtered_ibd.hbd.gz -o hbd --low-percentile 5 --outlier-percentile 95
  %(prog)s step3_named_filtered_ibd.hbd.gz --check-pairs candidate_pairs.txt
        """,
    )
    parser.add_argument('hbd_path', help='hap-IBD .hbd.gz/.hbd file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output prefix (default: derived from hbd_path)')
    parser.add_argument('--min-lod', type=float, default=3.0,
                        help='Minimum LOD score to keep a segment (default: 3.0)')
    parser.add_argument('--low-percentile', type=float, default=5,
                        help='Percentile below which HBD is flagged LOW (default: 5)')
    parser.add_argument('--outlier-percentile', type=float, default=95,
                        help='Percentile above which HBD is flagged HIGH (default: 95)')
    parser.add_argument('--check-pairs', default=None,
                        help='File of candidate pairs (one per line, "id1,id2" or "id1<TAB>id2") '
                             'to report side-by-side, e.g. extend_pedigree.py CLOSE_INBRED candidates')
    args = parser.parse_args()

    base = args.hbd_path
    for suffix in ('.hbd.gz', '.hbd'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    out_prefix = args.output or base

    by_individual, id_mismatches = load_hbd(args.hbd_path, min_lod=args.min_lod)
    if not by_individual:
        sys.exit(f"Error: no HBD segments found in {args.hbd_path} (after --min-lod {args.min_lod} filter).")

    stats = build_individual_stats(by_individual)
    ranks = compute_percentile_ranks(stats)
    flags = flag_outliers(stats, ranks, args.low_percentile, args.outlier_percentile)

    values = sorted(s['total_Mb'] for s in stats.values())
    n = len(values)
    p25 = _percentile_value(values, 25)
    p50 = _percentile_value(values, 50)
    p75 = _percentile_value(values, 75)
    flagged_ids = [iid for iid, (f, _) in flags.items() if f]

    print("=" * 70)
    print("HBD (Homozygosity-by-Descent) Analysis")
    print("=" * 70)
    print(f"  Individuals with HBD data: {n}")
    if id_mismatches:
        print(f"  Warning: {id_mismatches} rows had id1 != id2 (unexpected for an HBD file)")
    print(f"  HBD Mb distribution: p25={p25:.1f} median={p50:.1f} p75={p75:.1f}")
    print(f"  Flagged individuals: {len(flagged_ids)}")
    print()

    analysis_tsv_path = f"{out_prefix}_hbd_analysis.tsv"
    write_analysis_tsv(analysis_tsv_path, stats, ranks, flags)
    print(f"  Written: {analysis_tsv_path}")

    report_path = f"{out_prefix}_hbd_report.txt"
    with open(report_path, 'w') as rpt:
        rpt.write("HBD (Homozygosity-by-Descent) Analysis Report\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"Input: {args.hbd_path}\n")
        rpt.write(f"Individuals with HBD data: {n}\n")
        if id_mismatches:
            rpt.write(f"Warning: {id_mismatches} rows had id1 != id2 (unexpected for an HBD file)\n")
        rpt.write(f"HBD Mb distribution: p25={p25:.1f} median={p50:.1f} p75={p75:.1f}\n\n")

        rpt.write(f"Flagged individuals ({len(flagged_ids)})\n" + "-" * 40 + "\n")
        for iid in sorted(flagged_ids, key=lambda i: stats[i]['total_Mb']):
            _, reason = flags[iid]
            rpt.write(f"  {iid}: {stats[iid]['total_Mb']:.1f} Mb (p{ranks[iid]:.0f}) -- {reason}\n")
        if not flagged_ids:
            rpt.write("  (none)\n")

        if args.check_pairs:
            pairs = load_pairs(args.check_pairs)
            rpt.write(f"\nCandidate pair comparison ({len(pairs)} pairs from {args.check_pairs})\n"
                      + "-" * 40 + "\n")
            for a, b in pairs:
                rpt.write("  " + compare_pair(a, b, stats, ranks, flags) + "\n")

    print(f"  Written: {report_path}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
