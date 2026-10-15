#!/usr/bin/env python3
"""
Empirically calibrate the per-genotype error rate from confirmed MZ-twin/duplicate pairs.

BACKGROUND
----------
refine_pedigree.py's IBS0 noise ceiling and analyze_mendel_errors.py's expected
Mendelian-error baseline both depend on a --error-rate value that, until now,
has only ever been an ASSUMED constant (default 0.025) — never measured
against this cohort's actual data, even though the data needed to measure it
directly already exists.

vcfped's pairwise "MZ twins" verdict is an independent statistical call
(allele-sharing based, not KING-derived) that two samples are the same
individual (clonal duplicate / technical replicate). For a genuinely
identical pair, KING's IBS0 should be exactly 0 in the absence of genotyping
error — any IBS0 observed among CONFIRMED duplicates is therefore pure
genotyping noise, not biology. That gives a direct empirical estimate:

    error_rate ≈ median(IBS0 among confirmed MZ pairs) / 2

(same "IBS0 ≈ 2ε(1-ε) ≈ 2ε for small ε" relationship already documented in
refine_pedigree.py's ibs0_threshold_for_error_rate()).

GROUND TRUTH CAVEAT (densely inbred cohorts)
---------------------------------------------
vcfped's pairwise "MZ twins" call is only a clean ground truth if it isn't
itself contaminated by false positives. In densely inbred/repeatedly-selfed
germplasm, extreme homozygosity can inflate allele-sharing statistics enough
that genuinely DISTINCT-but-closely-related samples also get called "MZ
twins" (the same inbreeding-induced inflation documented for
extend_pedigree.py's CLOSE_INBRED/CLOSE_INBRED_POSSIBLE_DUP distinction) --
if a large fraction of a cohort's pairs (e.g. double digits of a percent of
all possible pairs) get flagged "MZ twins", that's a sign of contamination,
not that many literal duplicates exist. --confirmed-duplicates lets you swap
in a stricter, hap-IBD-coverage-validated ground truth instead (see below).

This replaces an assumed number with a measured one — every downstream
threshold that depends on --error-rate (IBS0 ceiling, Mendelian baseline)
inherits whatever accuracy this calibration has.

INPUT
-----
  <step1_pair_prefix>       — vcfped step1 pairwise output (reads <prefix>.pair),
                              the SAME file already used by refine_pedigree.py's
                              --step1-pair for MZ-group loading. Mutually
                              exclusive with --confirmed-duplicates.
  --confirmed-duplicates    — alternative, stricter ground truth: a two-column
                              pair list (header ID1/ID2) of hap-IBD-coverage-
                              CONFIRMED duplicates, e.g. extend_pedigree.py's
                              <prefix>_confirmed_duplicates.tsv (only written
                              when that script was run with --ibd). Use this
                              instead of <step1_pair_prefix> when the cohort's
                              own "MZ twins" verdict count looks implausibly
                              high (see GROUND TRUTH CAVEAT above).
  <king_prefix>             — PLINK2 --make-king-table output (reads <prefix>.kin0)

OUTPUT
------
  Prints the calibrated error rate and IBS0 distribution among confirmed
  duplicates; writes <output>_calibration_report.txt with the same detail
  for the record.

Usage:
  python calibrate_error_rate.py step1_named step3_named_filtered_king_ext
  python calibrate_error_rate.py step1_named step3_named_filtered_king_ext -o calib
  python calibrate_error_rate.py --confirmed-duplicates ext_out_confirmed_duplicates.tsv \\
      step3_named_filtered_king_ext
"""

import argparse
import csv
import os
import sys


def load_confirmed_mz_pairs(pair_path):
    """
    Parse a vcfped .pair file, returning [(id1, id2), ...] for every pair
    vcfped itself called 'MZ twins' — an independent statistical signal
    (allele sharing), not KING-derived, making it a clean ground truth for
    calibrating KING-based thresholds against.
    """
    path = pair_path if pair_path.endswith('.pair') else f"{pair_path}.pair"
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    pairs = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('Verdict', '').strip() == 'MZ twins':
                parts = row.get('Pair', '').split(',')
                if len(parts) == 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def load_confirmed_duplicate_pairs(path):
    """
    Parse a stricter ground-truth duplicate-pairs file -- e.g.
    extend_pedigree.py's <prefix>_confirmed_duplicates.tsv, whose pairs are
    hap-IBD-coverage-validated true duplicates rather than vcfped's raw
    pairwise allele-sharing call. Expects a header row with ID1/ID2 columns
    (name-matched, same convention as load_king_ibs0() below).
    """
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    pairs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        id1_col = next((c for c in fieldnames if c.upper() == 'ID1'), None)
        id2_col = next((c for c in fieldnames if c.upper() == 'ID2'), None)
        if not id1_col or not id2_col:
            sys.exit(f"Error: expected ID1/ID2 columns in {path}. Found: {fieldnames}")
        for row in reader:
            pairs.append((row[id1_col].strip(), row[id2_col].strip()))
    return pairs


def load_king_ibs0(king_prefix):
    """Parse a PLINK2 --make-king-table .kin0 file into {(id1,id2): ibs0}."""
    path = king_prefix if king_prefix.endswith('.kin0') else f"{king_prefix}.kin0"
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    king = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        id1_col = next((c for c in fieldnames if 'ID1' in c and 'FID' not in c), None)
        id2_col = next((c for c in fieldnames if 'ID2' in c and 'FID' not in c), None)
        if not id1_col or not id2_col:
            sys.exit(f"Error: cannot find ID1/ID2 columns in {path}.")
        for row in reader:
            id1 = row[id1_col].strip()
            id2 = row[id2_col].strip()
            try:
                ibs0 = float(row.get('IBS0', 0))
            except ValueError:
                continue
            king[(id1, id2)] = ibs0
    return king


def get_ibs0(king, a, b):
    return king.get((a, b), king.get((b, a)))


def collect_confirmed_duplicate_ibs0(mz_pairs, king):
    """Return sorted list of IBS0 values for confirmed MZ pairs found in KING."""
    values = []
    for a, b in mz_pairs:
        ibs0 = get_ibs0(king, a, b)
        if ibs0 is not None:
            values.append(ibs0)
    return sorted(values)


def _percentile_value(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return None
    idx = min(n - 1, int(p / 100 * n))
    return sorted_values[idx]


def derive_error_rate(median_ibs0):
    """
    error_rate ~= median_ibs0 / 2, inverting the "IBS0 ~= 2*error_rate"
    relationship documented in refine_pedigree.py's
    ibs0_threshold_for_error_rate().
    """
    return median_ibs0 / 2


def main():
    parser = argparse.ArgumentParser(
        description='Empirically calibrate the genotyping error rate from confirmed MZ-twin pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step1_named step3_named_filtered_king_ext
  %(prog)s step1_named step3_named_filtered_king_ext -o calib
  %(prog)s --confirmed-duplicates ext_out_confirmed_duplicates.tsv step3_named_filtered_king_ext

Feed the result back in:
  python refine_pedigree.py ... --error-rate <derived value>
  python analyze_mendel_errors.py ... --error-rate <derived value>
        """,
    )
    parser.add_argument('step1_pair_prefix', nargs='?', default=None,
                        help='vcfped step1 pairwise output prefix (reads <prefix>.pair). '
                             'Mutually exclusive with --confirmed-duplicates.')
    parser.add_argument('king_prefix',
                        help='PLINK2 --make-king-table prefix (reads <prefix>.kin0)')
    parser.add_argument('--confirmed-duplicates', default=None,
                        help='Alternative, stricter ground truth: a two-column pair list '
                             '(header ID1/ID2) of hap-IBD-coverage-CONFIRMED duplicates, e.g. '
                             "extend_pedigree.py's <prefix>_confirmed_duplicates.tsv. Use this "
                             "instead of step1_pair_prefix when vcfped's own \"MZ twins\" verdict "
                             'count looks implausibly high for this cohort (inbreeding-induced '
                             'allele-sharing inflation -- see module docstring). Mutually '
                             'exclusive with step1_pair_prefix.')
    parser.add_argument('-o', '--output',
                        help='Output prefix for the report (default: derived from whichever '
                             'ground-truth source is used)',
                        default=None)
    args = parser.parse_args()

    if bool(args.step1_pair_prefix) == bool(args.confirmed_duplicates):
        sys.exit("Error: specify exactly one of step1_pair_prefix or --confirmed-duplicates.")

    default_prefix = (f"{args.step1_pair_prefix}_error_rate" if args.step1_pair_prefix
                      else f"{args.confirmed_duplicates}_error_rate")
    out_prefix = args.output or default_prefix
    out_report_path = f"{out_prefix}_calibration_report.txt"

    print("=" * 70)
    print("VCFped Error Rate Calibration")
    print("=" * 70)
    if args.step1_pair_prefix:
        print(f"  Step1 pair file : {args.step1_pair_prefix}.pair")
    else:
        print(f"  Confirmed duplicates file : {args.confirmed_duplicates} "
              f"(hap-IBD-coverage validated)")
    print(f"  KING file       : {args.king_prefix}.kin0")
    print()

    if args.step1_pair_prefix:
        mz_pairs = load_confirmed_mz_pairs(args.step1_pair_prefix)
        ground_truth_label = "confirmed MZ-twin pairs (vcfped's own allele-sharing call)"
    else:
        mz_pairs = load_confirmed_duplicate_pairs(args.confirmed_duplicates)
        ground_truth_label = "confirmed duplicate pairs (hap-IBD-coverage validated)"
    print(f"  {len(mz_pairs)} {ground_truth_label}")

    king = load_king_ibs0(args.king_prefix)
    print(f"  {len(king)} KING pairs loaded")

    ibs0_values = collect_confirmed_duplicate_ibs0(mz_pairs, king)
    n_matched = len(ibs0_values)
    n_missing = len(mz_pairs) - n_matched
    print(f"  {n_matched} confirmed pairs found in KING data"
          f"{f' ({n_missing} not found — likely filtered by --king-table-filter)' if n_missing else ''}")
    print()

    if n_matched == 0:
        sys.exit("Error: no confirmed pairs found in the KING file — cannot calibrate. "
                 "Check that both files reference the same cohort/sample naming.")

    p25 = _percentile_value(ibs0_values, 25)
    p50 = _percentile_value(ibs0_values, 50)
    p75 = _percentile_value(ibs0_values, 75)
    mean_ibs0 = sum(ibs0_values) / n_matched

    derived_error_rate = derive_error_rate(p50)

    print(f"  IBS0 among confirmed duplicates: p25={p25:.4f} median={p50:.4f} "
          f"p75={p75:.4f} mean={mean_ibs0:.4f}")
    print()
    print(f"  Derived error rate (median IBS0 / 2): {derived_error_rate:.4f} ({derived_error_rate:.2%})")
    print()
    print("  Use this value in place of the assumed default:")
    print(f"    python refine_pedigree.py ... --error-rate {derived_error_rate:.4f}")
    print(f"    python analyze_mendel_errors.py ... --error-rate {derived_error_rate:.4f}")

    with open(out_report_path, 'w') as rpt:
        rpt.write("VCFped Error Rate Calibration Report\n")
        rpt.write("=" * 70 + "\n\n")
        if args.step1_pair_prefix:
            rpt.write(f"Step1 pair file : {args.step1_pair_prefix}.pair\n")
        else:
            rpt.write(f"Confirmed duplicates file : {args.confirmed_duplicates} "
                      f"(hap-IBD-coverage validated)\n")
        rpt.write(f"KING file       : {args.king_prefix}.kin0\n\n")
        rpt.write(f"{ground_truth_label.capitalize()}           : {len(mz_pairs)}\n")
        rpt.write(f"Found in KING data                : {n_matched}\n")
        rpt.write(f"Not found (filtered / missing)     : {n_missing}\n\n")
        rpt.write("IBS0 distribution among confirmed duplicates\n" + "-" * 40 + "\n")
        rpt.write(f"  p25    : {p25:.4f}\n")
        rpt.write(f"  median : {p50:.4f}\n")
        rpt.write(f"  p75    : {p75:.4f}\n")
        rpt.write(f"  mean   : {mean_ibs0:.4f}\n\n")
        rpt.write(f"Derived error rate (median IBS0 / 2): {derived_error_rate:.4f} "
                  f"({derived_error_rate:.2%})\n\n")
        rpt.write("Use this value in place of the assumed default:\n")
        rpt.write(f"  python refine_pedigree.py ... --error-rate {derived_error_rate:.4f}\n")
        rpt.write(f"  python analyze_mendel_errors.py ... --error-rate {derived_error_rate:.4f}\n")

    print(f"\n  Written: {out_report_path}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
