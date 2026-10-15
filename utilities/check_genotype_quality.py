#!/usr/bin/env python3
"""
Check per-sample genotyping quality (heterozygosity, missingness) from PLINK2
--het/--missing output, and rule it out (or in) as a confound for Mendelian
"repeat offender" parents.

BACKGROUND
----------
analyze_mendel_errors.py can flag an individual as a "repeat offender" parent:
elevated Mendelian error rate across every co-parent they appear with,
regardless of who that co-parent is. The natural read of that pattern is a
pedigree problem (wrong parent assignment, or an MZ-twin representative that's
actually the wrong twin) -- but the IDENTICAL symptom also arises from poor
genotyping quality alone (elevated missingness, or elevated heterozygosity
from sample contamination), which needs a completely different remedy
(re-genotyping/QC exclusion, not pedigree reassignment). Until now nothing in
the pipeline checked for this alternative explanation.

prepare_plink_beagle.py's generated _run_plink.sh now runs PLINK2 --het (on
the LD-pruned set, same reason KING itself is LD-pruned) and --missing (on
the full quality-filtered set) as Step 4. This script consumes those outputs.

As with every other threshold in this pipeline, outlier flagging is
population-relative (percentile rank within this cohort), not a fixed cutoff:
a cohort of deep-selfed lines has high homozygosity (high F) as its normal
baseline, so a fixed "F > 0.5 is suspicious" rule would misfire constantly.

INPUT
-----
  <het_path>    — PLINK2 .het (per-individual O(HOM)/E(HOM)/F)
  <smiss_path>  — PLINK2 .smiss or .imiss (per-individual F_MISS)

OUTPUT
------
  <output>_genotype_quality.tsv    — every individual: F, percentile rank,
                                      F_MISS, percentile rank, flagged, reason
  <output>_genotype_quality_report.txt  — distribution summary + flagged
                                      individuals + (if --mendel-individual-tsv
                                      given) repeat-offender cross-reference

Usage:
  python check_genotype_quality.py step3_plink.het step3_plink.smiss
  python check_genotype_quality.py step3_plink.het step3_plink.smiss -o qc \\
      --het-low-percentile 5 --het-high-percentile 95 --miss-outlier-percentile 95
  python check_genotype_quality.py step3_plink.het step3_plink.smiss \\
      --mendel-individual-tsv step3_mendel_individual_analysis.tsv
"""

import argparse
import csv
import os
import sys


def read_plink_table(path):
    """
    Generic reader for whitespace-delimited PLINK2 tabular output (.het,
    .smiss, .imiss, ...): first non-empty line is the header (PLINK2
    prefixes the first column name with '#'), remaining lines are rows.
    Returns list of dicts keyed by header name (leading '#' stripped) --
    column-NAME based (not fixed positions), since exact column sets can
    vary slightly across PLINK2 versions.
    """
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if header is None:
                header = [h.lstrip('#') for h in parts]
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def load_het(path):
    """Parse a PLINK2 .het file into {iid: {'o_hom', 'e_hom', 'f'}}."""
    rows = read_plink_table(path)
    if not rows:
        sys.exit(f"Error: {path} contains no data rows.")
    header = list(rows[0].keys())
    iid_col = next((c for c in header if c.upper() == 'IID'), None)
    ohom_col = next((c for c in header if 'O(HOM)' in c.upper() or c.upper() == 'O_HOM'), None)
    ehom_col = next((c for c in header if 'E(HOM)' in c.upper() or c.upper() == 'E_HOM'), None)
    f_col = next((c for c in header if c.upper() == 'F'), None)
    if not iid_col or not f_col:
        sys.exit(f"Error: could not find IID/F columns in {path}. Found columns: {header}")

    het = {}
    for r in rows:
        try:
            het[r[iid_col]] = {
                'o_hom': float(r[ohom_col]) if ohom_col and r.get(ohom_col) not in (None, 'NA') else None,
                'e_hom': float(r[ehom_col]) if ehom_col and r.get(ehom_col) not in (None, 'NA') else None,
                'f': float(r[f_col]),
            }
        except ValueError:
            continue
    return het


def load_missing(path):
    """Parse a PLINK2 .smiss/.imiss file into {iid: f_miss}."""
    rows = read_plink_table(path)
    if not rows:
        sys.exit(f"Error: {path} contains no data rows.")
    header = list(rows[0].keys())
    iid_col = next((c for c in header if c.upper() == 'IID'), None)
    fmiss_col = next((c for c in header if c.upper() == 'F_MISS'), None)
    if not iid_col or not fmiss_col:
        sys.exit(f"Error: could not find IID/F_MISS columns in {path}. Found columns: {header}")

    missing = {}
    for r in rows:
        try:
            missing[r[iid_col]] = float(r[fmiss_col])
        except ValueError:
            continue
    return missing


def _percentile_rank(values_by_id):
    """
    Cohort-relative percentile RANK (0-100) per id, same rank-based approach
    as analyze_hbd.py (avoids the boundary case where a value-threshold
    approach can put the single most extreme individual in a small cohort
    exactly at its own threshold).
    """
    ids_sorted = sorted(values_by_id.keys(), key=lambda i: values_by_id[i])
    n = len(ids_sorted)
    ranks = {}
    for i, iid in enumerate(ids_sorted):
        ranks[iid] = (i / (n - 1) * 100) if n > 1 else 50.0
    return ranks


def flag_het_outliers(het, ranks, low_percentile=5, high_percentile=95):
    """
    Flag individuals whose F percentile rank is at or below low_percentile
    (unexpectedly heterozygous for this cohort -- possible contamination) or
    at or above high_percentile (unexpectedly homozygous even relative to
    this cohort's own norm). Returns dict id -> (flagged: bool, reason: str).
    """
    flags = {}
    for iid, d in het.items():
        rank = ranks[iid]
        if rank <= low_percentile:
            flags[iid] = (True, f'LOW_F (percentile rank {rank:.0f} <= p{low_percentile}, F={d["f"]:.3f}) '
                                 f'-- unexpectedly heterozygous for this cohort, possible contamination')
        elif rank >= high_percentile:
            flags[iid] = (True, f'HIGH_F (percentile rank {rank:.0f} >= p{high_percentile}, F={d["f"]:.3f})')
        else:
            flags[iid] = (False, '')
    return flags


def flag_missing_outliers(missing, ranks, outlier_percentile=95):
    """
    Flag individuals whose F_MISS percentile rank is at or above
    outlier_percentile -- elevated missingness only has one concerning
    direction (unlike heterozygosity). Returns dict id -> (flagged, reason).
    """
    flags = {}
    for iid, f_miss in missing.items():
        rank = ranks[iid]
        if rank >= outlier_percentile:
            flags[iid] = (True, f'HIGH_MISSINGNESS (percentile rank {rank:.0f} >= p{outlier_percentile}, '
                                 f'F_MISS={f_miss:.3f})')
        else:
            flags[iid] = (False, '')
    return flags


def load_mendel_repeat_offenders(path):
    """
    Parse analyze_mendel_errors.py's <output>_individual_analysis.tsv,
    returning the set of IIDs flagged there as repeat-offender parents
    (elevated Mendelian error rate across multiple co-parents). Optional --
    returns an empty set if no path given or the file doesn't exist.
    """
    offenders = set()
    if not path or not os.path.exists(path):
        return offenders
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('Flagged', '').strip() == 'True':
                offenders.add(row['Individual'].strip())
    return offenders


def main():
    parser = argparse.ArgumentParser(
        description='Check per-sample genotyping quality and cross-reference against '
                    'Mendelian repeat offenders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_plink.het step3_plink.smiss
  %(prog)s step3_plink.het step3_plink.smiss -o qc --het-low-percentile 5 \\
      --het-high-percentile 95 --miss-outlier-percentile 95
  %(prog)s step3_plink.het step3_plink.smiss \\
      --mendel-individual-tsv step3_mendel_individual_analysis.tsv
        """,
    )
    parser.add_argument('het_path', help='PLINK2 .het file (per-individual O(HOM)/E(HOM)/F)')
    parser.add_argument('smiss_path', help='PLINK2 .smiss/.imiss file (per-individual F_MISS)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output prefix (default: derived from het_path)')
    parser.add_argument('--het-low-percentile', type=float, default=5,
                        help='Percentile at/below which F is flagged LOW (default: 5)')
    parser.add_argument('--het-high-percentile', type=float, default=95,
                        help='Percentile at/above which F is flagged HIGH (default: 95)')
    parser.add_argument('--miss-outlier-percentile', type=float, default=95,
                        help='Percentile at/above which F_MISS is flagged (default: 95)')
    parser.add_argument('--mendel-individual-tsv', default=None,
                        help="analyze_mendel_errors.py's <output>_individual_analysis.tsv, "
                             'to cross-reference repeat-offender parents against genotyping '
                             'quality (optional)')
    args = parser.parse_args()

    base = args.het_path[:-len('.het')] if args.het_path.endswith('.het') else args.het_path
    out_prefix = args.output or base

    het = load_het(args.het_path)
    missing = load_missing(args.smiss_path)
    print("=" * 70)
    print("Per-Sample Genotyping Quality Check")
    print("=" * 70)
    print(f"  {len(het)} individuals in {args.het_path}")
    print(f"  {len(missing)} individuals in {args.smiss_path}")

    het_ranks = _percentile_rank({iid: d['f'] for iid, d in het.items()})
    miss_ranks = _percentile_rank(missing)
    het_flags = flag_het_outliers(het, het_ranks, args.het_low_percentile, args.het_high_percentile)
    miss_flags = flag_missing_outliers(missing, miss_ranks, args.miss_outlier_percentile)

    n_het_flagged = sum(1 for f, _ in het_flags.values() if f)
    n_miss_flagged = sum(1 for f, _ in miss_flags.values() if f)
    print(f"  {n_het_flagged} individuals flagged for heterozygosity (F)")
    print(f"  {n_miss_flagged} individuals flagged for missingness")
    print()

    all_ids = sorted(set(het.keys()) | set(missing.keys()))
    tsv_path = f"{out_prefix}_genotype_quality.tsv"
    with open(tsv_path, 'w') as f:
        f.write('\t'.join(['ID', 'F', 'F_Percentile_Rank', 'Het_Flagged', 'Het_Reason',
                           'F_MISS', 'Miss_Percentile_Rank', 'Miss_Flagged', 'Miss_Reason']) + '\n')
        for iid in all_ids:
            f_val = het.get(iid, {}).get('f')
            f_rank = het_ranks.get(iid)
            hf, hr = het_flags.get(iid, (False, ''))
            m_val = missing.get(iid)
            m_rank = miss_ranks.get(iid)
            mf, mr = miss_flags.get(iid, (False, ''))
            f.write('\t'.join([
                iid,
                f"{f_val:.4f}" if f_val is not None else 'NA',
                f"{f_rank:.1f}" if f_rank is not None else 'NA',
                'YES' if hf else 'no', hr,
                f"{m_val:.4f}" if m_val is not None else 'NA',
                f"{m_rank:.1f}" if m_rank is not None else 'NA',
                'YES' if mf else 'no', mr,
            ]) + '\n')
    print(f"  Written: {tsv_path}")

    report_path = f"{out_prefix}_genotype_quality_report.txt"
    with open(report_path, 'w') as rpt:
        rpt.write("Per-Sample Genotyping Quality Report\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"Het file : {args.het_path} ({len(het)} individuals)\n")
        rpt.write(f"Miss file: {args.smiss_path} ({len(missing)} individuals)\n\n")

        rpt.write(f"Heterozygosity-flagged individuals ({n_het_flagged})\n" + "-" * 40 + "\n")
        for iid in sorted((i for i, (fl, _) in het_flags.items() if fl), key=lambda i: het[i]['f']):
            rpt.write(f"  {iid}: {het_flags[iid][1]}\n")
        if not n_het_flagged:
            rpt.write("  (none)\n")
        rpt.write("\n")

        rpt.write(f"Missingness-flagged individuals ({n_miss_flagged})\n" + "-" * 40 + "\n")
        for iid in sorted((i for i, (fl, _) in miss_flags.items() if fl), key=lambda i: -missing[i]):
            rpt.write(f"  {iid}: {miss_flags[iid][1]}\n")
        if not n_miss_flagged:
            rpt.write("  (none)\n")
        rpt.write("\n")

        if args.mendel_individual_tsv:
            offenders = load_mendel_repeat_offenders(args.mendel_individual_tsv)
            rpt.write(f"Mendelian repeat-offender cross-reference ({len(offenders)} from "
                      f"{args.mendel_individual_tsv})\n" + "-" * 40 + "\n")
            confounded, clean = [], []
            for iid in sorted(offenders):
                het_flagged = het_flags.get(iid, (False, ''))[0]
                miss_flagged = miss_flags.get(iid, (False, ''))[0]
                if het_flagged or miss_flagged:
                    confounded.append(iid)
                else:
                    clean.append(iid)

            rpt.write(f"\nWith a genotyping-quality confound ({len(confounded)}) -- elevated "
                      f"Mendelian error could be explained by poor genotyping quality alone, "
                      f"not necessarily a pedigree error:\n")
            for iid in confounded:
                reasons = [r for r in (het_flags.get(iid, (False, ''))[1],
                                       miss_flags.get(iid, (False, ''))[1]) if r]
                rpt.write(f"  {iid}: {'; '.join(reasons)}\n")
            if not confounded:
                rpt.write("  (none)\n")

            rpt.write(f"\nNo genotyping-quality confound ({len(clean)}) -- the pedigree-error "
                      f"explanation is not undermined by this check:\n")
            for iid in clean:
                rpt.write(f"  {iid}\n")
            if not clean:
                rpt.write("  (none)\n")

    print(f"  Written: {report_path}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
