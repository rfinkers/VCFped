#!/usr/bin/env python3
"""
Flag outlier families from a PLINK2 --mendel .fmendel report.

BACKGROUND
----------
PLINK2's --mendel writes .fmendel with columns FID PAT MAT CHILD N, where N
is a raw COUNT of Mendelian errors summed across all quality-filtered
variants for that family — not yet a rate. To judge whether a family's error
count is "too high" (suggesting a wrong parent assignment) rather than just
ordinary genotyping noise, it must be normalized by the number of variants
tested and the number of children in that family:

    rate = N / (n_variants * CHILD)

A single fixed threshold on this rate is unreliable across cohorts, for the
same reason fixed KING kinship thresholds are unreliable in this project's
closely related germplasm (see refine_pedigree.py): the EXPECTED baseline
rate for a genuinely correct trio scales with the cohort's own genotyping
error rate (documented in prepare_plink_beagle.py's generated
_run_plink.sh: expected family error ≈ 2 × genotype error rate). A cohort
with an elevated error rate (e.g. 2.5%) has an expected baseline of ~5% —
right at the generic "family error > 5% -> likely wrong parent" guideline,
which was calibrated for cleaner data. Applying that guideline unmodified
here would flag many genuinely correct trios.

This script instead flags outliers TWO ways, and reports both so nothing is
hidden behind a single number:
  - cohort-relative: rate exceeds a percentile of this cohort's own rate
    distribution (default: 95th percentile)
  - absolute: rate exceeds a multiple of the theoretical expected baseline
    for the given --error-rate (default: 2x, i.e. an already-doubled
    "safety margin" on top of the already-doubled expected rate)
A family flagged by either is reported; which one(s) triggered is recorded
so you can judge for yourself rather than trust either signal blindly.

REPEAT-OFFENDER PARENTS (per-individual aggregation)
-----------------------------------------------------
A flagged family only says "this specific PAT+MAT pairing has an elevated
error rate" — it doesn't distinguish a genuinely wrong assignment from an
individual-level problem (sample mislabeling, an MZ-twin representative
that's actually the wrong twin, or a sample-specific genotyping quality
issue). The signature of the latter is an individual appearing as a parent
across MANY DIFFERENT co-parents, all showing elevated error — if the
problem tracked with the specific pairing, different co-parents would
behave differently.

This script aggregates every .fmendel row by individual (across every
family where they appear as EITHER PAT or MAT) to surface this pattern
directly, rather than relying on spotting repeated IDs by eye in the
family-level table. Individuals appearing with fewer than --min-co-parents
distinct co-parents are excluded from this specific flagging (with only one
pairing, "repeat offender" isn't yet distinguishable from ordinary
per-family noise) but still appear in the full individual TSV.

Optionally cross-references PLINK2's own --mendel .imendel file (per-CHILD
error counts — a DIFFERENT, complementary count: only children accumulate
.imendel errors, whereas the per-individual aggregation above covers PARENT
roles specifically). Both are reported side by side, clearly labelled,
rather than conflated into one number.

INPUT
-----
  <fmendel>            — PLINK2 --mendel output (<prefix>.fmendel)
  --bim <path>          — corresponding .bim file, to count variants (n_variants)
  --n-variants <int>    — alternative to --bim: give the variant count directly
  --imendel <path>      — optional PLINK2 .imendel file, for the per-child
                          error count cross-reference described above

OUTPUT
------
  <output>_mendel_analysis.tsv     — every family: rate, cohort percentile
                                      rank, flagged, reason
  <output>_mendel_report.txt       — narrative: rate distribution, flagged
                                      families sorted by rate descending,
                                      repeat-offender parents section
  <output>_individual_analysis.tsv — every individual who appears as a
                                      parent: aggregate rate across all
                                      their families, number of distinct
                                      co-parents, flagged, reason,
                                      .imendel child-error count if given

Usage:
  python analyze_mendel_errors.py revalidate_mendel.fmendel --bim revalidate_plink.bim
  python analyze_mendel_errors.py revalidate_mendel.fmendel --n-variants 8789 \\
      --error-rate 0.025 --outlier-percentile 95 --absolute-multiplier 2.0 \\
      --imendel revalidate_mendel.imendel --min-co-parents 2
"""

import argparse
import os
import sys


def load_fmendel(path):
    """
    Parse a PLINK2 .fmendel file (per-family Mendelian error counts).
    Returns list of dicts: {fid, pat, mat, child, n} — n is a raw error
    COUNT, not yet a rate (see compute_rates()).
    """
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].lstrip('#').upper() == 'FID':
                continue  # header line
            if len(parts) < 5:
                continue
            fid, pat, mat, child, n = parts[:5]
            try:
                rows.append({'fid': fid, 'pat': pat, 'mat': mat,
                             'child': int(child), 'n': int(n)})
            except ValueError:
                continue
    if not rows:
        sys.exit(f"Error: {path} contains no parseable family rows. "
                 f"Expected PLINK2 --mendel output (FID PAT MAT CHILD N).")
    return rows


def count_variants(bim_path):
    """Count lines in a PLINK .bim file (one line per variant)."""
    if not os.path.exists(bim_path):
        sys.exit(f"Error: {bim_path} not found.")
    with open(bim_path) as f:
        return sum(1 for _ in f)


def load_imendel(path):
    """
    Parse a PLINK2 .imendel file (per-individual Mendelian error counts).
    This is PLINK's own count, attributed to each CHILD's genotype
    specifically — a different, complementary number from
    aggregate_parent_errors()'s PARENT-role aggregation computed from
    .fmendel below (only children accumulate .imendel errors).
    Returns dict: {iid: n_errors}. Empty dict if no path given or not found
    (optional cross-reference, not required for the rest of the analysis).
    """
    errors = {}
    if not path or not os.path.exists(path):
        return errors
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].lstrip('#').upper() == 'FID':
                continue  # header line
            if len(parts) < 3:
                continue
            fid, iid, n = parts[:3]
            try:
                errors[iid] = int(n)
            except ValueError:
                continue
    return errors


def aggregate_parent_errors(rows):
    """
    Aggregate .fmendel rows by individual, across every family where they
    appear as either PAT or MAT, to detect "repeat offender" parents whose
    elevated error rate shows up consistently regardless of co-parent — see
    the module docstring for why this is the signature of an
    individual-level problem rather than independent per-trio noise.

    Returns list of dicts: {individual, n_families, total_n, total_child,
    co_parents (sorted list), n_co_parents}. total_child is the sum of
    CHILD across every family this individual appears in as a parent — the
    correct denominator for a per-individual rate (see compute_rates()).
    """
    agg = {}
    for r in rows:
        for role, other in (('pat', r['mat']), ('mat', r['pat'])):
            ind = r[role]
            if ind not in agg:
                agg[ind] = {'individual': ind, 'n_families': 0, 'total_n': 0,
                            'total_child': 0, 'co_parents': set()}
            agg[ind]['n_families'] += 1
            agg[ind]['total_n'] += r['n']
            agg[ind]['total_child'] += r['child']
            agg[ind]['co_parents'].add(other)

    result = list(agg.values())
    for r in result:
        r['co_parents'] = sorted(r['co_parents'])
        r['n_co_parents'] = len(r['co_parents'])
    return result


def compute_rates(rows, n_variants):
    """Attach rate = n / (n_variants * child) to each row, in place."""
    for r in rows:
        denom = n_variants * r['child']
        r['rate'] = r['n'] / denom if denom > 0 else 0.0
    return rows


def compute_individual_rates(agg_rows, n_variants):
    """
    Attach rate = total_n / (n_variants * total_child) to each aggregated
    individual row, in place — the per-individual analog of compute_rates(),
    for rows produced by aggregate_parent_errors().
    """
    for r in agg_rows:
        denom = n_variants * r['total_child']
        r['rate'] = r['total_n'] / denom if denom > 0 else 0.0
    return agg_rows


def _percentile_value(sorted_values, p):
    """Same simple sorted-index percentile formula used elsewhere in this
    project (merge_refined_fam.py's compute_kinship_percentiles(),
    refine_pedigree.py's printed kinship distribution)."""
    n = len(sorted_values)
    if n == 0:
        return None
    idx = min(n - 1, int(p / 100 * n))
    return sorted_values[idx]


def compute_rate_percentiles(rows, percentiles=(50, 75, 90, 95, 99)):
    """Cohort-relative percentile VALUES of the rate distribution."""
    rates = sorted(r['rate'] for r in rows)
    return {p: _percentile_value(rates, p) for p in percentiles}


def compute_percentile_ranks(rows):
    """
    Cohort-relative percentile RANK (0-100) per row, same style as
    refine_pedigree.py's build_king_percentiles() — lets each family be
    judged relative to this cohort's own distribution rather than an
    absolute number.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]['rate'])
    n = len(order)
    ranks = [0] * n
    for i, idx in enumerate(order):
        ranks[idx] = round(100 * i / max(n - 1, 1))
    return ranks


def expected_baseline_rate(error_rate):
    """
    Expected family-level Mendelian error rate for a genuinely correct
    parent-offspring trio, given a per-genotype error rate — same "≈ 2x"
    relationship documented in prepare_plink_beagle.py's generated
    _run_plink.sh (0.1%->0.2%, 0.5%->1.0%, 1.0%->2.0%).
    """
    return 2 * error_rate


def flag_outliers(rows, percentile_threshold, absolute_threshold):
    """
    Flag each row as an outlier if its rate exceeds EITHER the cohort
    percentile threshold OR the absolute (error-rate-derived) threshold.
    Adds 'flagged' (bool) and 'flag_reason' (str) to each row, in place.
    """
    for r in rows:
        exceeds_pct = percentile_threshold is not None and r['rate'] > percentile_threshold
        exceeds_abs = r['rate'] > absolute_threshold
        if exceeds_pct and exceeds_abs:
            reason = 'cohort_outlier+absolute_high'
        elif exceeds_pct:
            reason = 'cohort_outlier'
        elif exceeds_abs:
            reason = 'absolute_high'
        else:
            reason = ''
        r['flagged'] = bool(reason)
        r['flag_reason'] = reason
    return rows


ANALYSIS_TSV_HEADER = ['FID', 'PAT', 'MAT', 'CHILD', 'N', 'Rate', 'Percentile_Rank',
                       'Flagged', 'Flag_Reason']


def write_analysis_tsv(path, rows, percentile_ranks):
    with open(path, 'w') as f:
        f.write('\t'.join(ANALYSIS_TSV_HEADER) + '\n')
        for r, rank in zip(rows, percentile_ranks):
            f.write('\t'.join(str(x) for x in [
                r['fid'], r['pat'], r['mat'], r['child'], r['n'],
                f"{r['rate']:.6f}", rank, r['flagged'], r['flag_reason']
            ]) + '\n')


INDIVIDUAL_TSV_HEADER = ['Individual', 'N_Families', 'N_CoParents', 'Total_N', 'Total_Child',
                         'Rate', 'Percentile_Rank', 'Flagged', 'Flag_Reason', 'IMendel_N']


def write_individual_tsv(path, agg_rows, percentile_ranks, imendel):
    with open(path, 'w') as f:
        f.write('\t'.join(INDIVIDUAL_TSV_HEADER) + '\n')
        for r, rank in zip(agg_rows, percentile_ranks):
            imendel_n = imendel.get(r['individual'], 'NA')
            f.write('\t'.join(str(x) for x in [
                r['individual'], r['n_families'], r['n_co_parents'], r['total_n'],
                r['total_child'], f"{r['rate']:.6f}", rank, r['flagged'],
                r['flag_reason'], imendel_n
            ]) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Flag outlier families from a PLINK2 --mendel .fmendel report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s revalidate_mendel.fmendel --bim revalidate_plink.bim
  %(prog)s revalidate_mendel.fmendel --n-variants 8789 --outlier-percentile 99
        """,
    )
    parser.add_argument('fmendel', help='PLINK2 .fmendel file (per-family Mendelian error counts)')
    parser.add_argument('--bim', default=None,
                        help='Corresponding .bim file, to count variants (n_variants). '
                             'Alternative to --n-variants.')
    parser.add_argument('--n-variants', type=int, default=None,
                        help='Number of quality-filtered variants used for --mendel. '
                             'Alternative to --bim.')
    parser.add_argument('--error-rate', type=float, default=0.025,
                        help='Expected per-genotype error rate for this cohort, used to '
                             'derive the absolute outlier threshold (same value you used '
                             'with refine_pedigree.py; default: 0.025)')
    parser.add_argument('--outlier-percentile', type=float, default=95,
                        help='Cohort-relative percentile above which a family rate is '
                             'flagged as an outlier (default: 95)')
    parser.add_argument('--absolute-multiplier', type=float, default=2.0,
                        help='Multiple of the theoretical expected baseline rate '
                             '(2 x --error-rate) above which a family is flagged '
                             'regardless of cohort percentile (default: 2.0)')
    parser.add_argument('--imendel', default=None,
                        help="Optional PLINK2 .imendel file (per-child error counts) — "
                             "cross-referenced against the per-individual PARENT-role "
                             "aggregation this script computes from .fmendel; these are "
                             "different, complementary counts (see module docstring).")
    parser.add_argument('--min-co-parents', type=int, default=2,
                        help='Minimum number of distinct co-parents an individual must '
                             'have before being eligible for repeat-offender flagging '
                             '(default: 2 — with only one pairing, "repeat offender" is '
                             'not yet distinguishable from ordinary per-family noise)')
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: <fmendel without .fmendel>_mendel)',
                        default=None)
    parser.add_argument('--report-limit', type=int, default=200,
                        help='Max flagged families/individuals to list in the report (default: 200)')
    args = parser.parse_args()

    if not args.bim and args.n_variants is None:
        sys.exit("Error: specify either --bim or --n-variants.")
    if args.bim and args.n_variants is not None:
        sys.exit("Error: specify either --bim or --n-variants, not both.")

    out_prefix = args.output or (
        args.fmendel[:-len('.fmendel')] if args.fmendel.endswith('.fmendel') else args.fmendel
    ) + '_mendel'
    out_tsv_path = f"{out_prefix}_analysis.tsv"
    out_report_path = f"{out_prefix}_report.txt"

    print("=" * 70)
    print("VCFped Mendelian Error Outlier Analysis")
    print("=" * 70)
    print(f"  Input .fmendel : {args.fmendel}")

    n_variants = args.n_variants if args.n_variants is not None else count_variants(args.bim)
    print(f"  Variant count  : {n_variants} ({'from --n-variants' if args.n_variants is not None else args.bim})")
    print(f"  Genotype error : {args.error_rate:.1%}")
    print()

    rows = load_fmendel(args.fmendel)
    print(f"  {len(rows)} families loaded")
    compute_rates(rows, n_variants)

    pct_values = compute_rate_percentiles(rows)
    percentile_threshold = pct_values.get(args.outlier_percentile)
    if percentile_threshold is None:
        # Requested percentile isn't one of the defaults computed above; add it.
        rates_sorted = sorted(r['rate'] for r in rows)
        percentile_threshold = _percentile_value(rates_sorted, args.outlier_percentile)

    expected = expected_baseline_rate(args.error_rate)
    absolute_threshold = args.absolute_multiplier * expected

    print(f"  Expected baseline rate (2 x error rate): {expected:.2%}")
    print(f"  Absolute outlier threshold ({args.absolute_multiplier}x baseline): {absolute_threshold:.2%}")
    print(f"  Cohort rate distribution: "
          f"p50={pct_values[50]:.2%} p75={pct_values[75]:.2%} p90={pct_values[90]:.2%} "
          f"p95={pct_values[95]:.2%} p99={pct_values[99]:.2%}")
    print(f"  Cohort-relative outlier threshold (p{args.outlier_percentile:g}): "
          f"{percentile_threshold:.2%}")
    print()

    flag_outliers(rows, percentile_threshold, absolute_threshold)
    percentile_ranks = compute_percentile_ranks(rows)

    n_flagged = sum(1 for r in rows if r['flagged'])
    n_cohort_only = sum(1 for r in rows if r['flag_reason'] == 'cohort_outlier')
    n_absolute_only = sum(1 for r in rows if r['flag_reason'] == 'absolute_high')
    n_both = sum(1 for r in rows if r['flag_reason'] == 'cohort_outlier+absolute_high')

    print("Summary:")
    print(f"  Families flagged        : {n_flagged} / {len(rows)}")
    print(f"    cohort-outlier only    : {n_cohort_only}")
    print(f"    absolute-high only     : {n_absolute_only}")
    print(f"    both                   : {n_both}")
    print()

    write_analysis_tsv(out_tsv_path, rows, percentile_ranks)
    print(f"  Written: {out_tsv_path}")

    flagged_sorted = sorted((r for r in rows if r['flagged']), key=lambda r: -r['rate'])

    # --- Per-individual (repeat-offender) analysis ---------------------------
    out_individual_tsv_path = f"{out_prefix}_individual_analysis.tsv"
    imendel = load_imendel(args.imendel)
    if args.imendel:
        print(f"  {len(imendel)} individuals loaded from --imendel {args.imendel}")

    agg_rows = aggregate_parent_errors(rows)
    compute_individual_rates(agg_rows, n_variants)
    ind_pct_values = compute_rate_percentiles(agg_rows)
    ind_percentile_threshold = ind_pct_values.get(args.outlier_percentile)
    if ind_percentile_threshold is None:
        ind_rates_sorted = sorted(r['rate'] for r in agg_rows)
        ind_percentile_threshold = _percentile_value(ind_rates_sorted, args.outlier_percentile)
    flag_outliers(agg_rows, ind_percentile_threshold, absolute_threshold)
    ind_percentile_ranks = compute_percentile_ranks(agg_rows)

    # Repeat-offender flagging additionally requires >= --min-co-parents
    # distinct co-parents -- with only one pairing this collapses to the
    # same signal the family-level analysis already gives.
    repeat_offenders = sorted(
        (r for r in agg_rows if r['flagged'] and r['n_co_parents'] >= args.min_co_parents),
        key=lambda r: -r['rate'])

    print(f"  {len(agg_rows)} individuals appear as a parent in >=1 family")
    print(f"  {len(repeat_offenders)} repeat-offender parent(s) "
          f"(flagged AND >= {args.min_co_parents} distinct co-parents)")
    print()

    write_individual_tsv(out_individual_tsv_path, agg_rows, ind_percentile_ranks, imendel)
    print(f"  Written: {out_individual_tsv_path}")

    with open(out_report_path, 'w') as rpt:
        rpt.write("VCFped Mendelian Error Outlier Analysis\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"Input .fmendel : {args.fmendel}\n")
        rpt.write(f"Variant count  : {n_variants}\n")
        rpt.write(f"Genotype error : {args.error_rate:.1%}\n\n")

        rpt.write("Rate distribution\n" + "-" * 40 + "\n")
        for p in (50, 75, 90, 95, 99):
            rpt.write(f"  p{p:<3g}: {pct_values[p]:.2%}\n")
        rpt.write(f"\n  Expected baseline (2 x error rate)         : {expected:.2%}\n")
        rpt.write(f"  Absolute outlier threshold ({args.absolute_multiplier}x baseline)   : "
                  f"{absolute_threshold:.2%}\n")
        rpt.write(f"  Cohort-relative outlier threshold (p{args.outlier_percentile:g}) : "
                  f"{percentile_threshold:.2%}\n\n")

        rpt.write("Summary\n" + "-" * 40 + "\n")
        rpt.write(f"  Families flagged     : {n_flagged} / {len(rows)}\n")
        rpt.write(f"    cohort-outlier only: {n_cohort_only}\n")
        rpt.write(f"    absolute-high only : {n_absolute_only}\n")
        rpt.write(f"    both               : {n_both}\n\n")

        rpt.write("Flagged families (sorted by rate, descending)\n" + "-" * 40 + "\n")
        if not flagged_sorted:
            rpt.write("  None.\n")
        for r in flagged_sorted[:args.report_limit]:
            rpt.write(f"  {r['fid']}  PAT={r['pat']} MAT={r['mat']} CHILD={r['child']} "
                      f"N={r['n']}  rate={r['rate']:.2%}  [{r['flag_reason']}]\n")
        if len(flagged_sorted) > args.report_limit:
            rpt.write(f"  ... and {len(flagged_sorted) - args.report_limit} more "
                      f"(full list in {out_tsv_path})\n")
        rpt.write("\n")

        rpt.write(f"Repeat-offender parents (flagged AND >= {args.min_co_parents} "
                  f"distinct co-parents)\n" + "-" * 40 + "\n")
        rpt.write("An individual appearing here shows an elevated Mendelian error rate\n"
                  "across MULTIPLE different co-parents -- consistent with an\n"
                  "individual-level problem (sample mislabeling, an MZ-twin\n"
                  "representative that's actually the wrong twin, or a sample-specific\n"
                  "genotyping quality issue) rather than one specific wrong pairing.\n"
                  "See module docstring for interpretation guidance.\n\n")
        if not repeat_offenders:
            rpt.write("  None.\n")
        for r in repeat_offenders[:args.report_limit]:
            imendel_note = f", imendel_N={imendel[r['individual']]}" if r['individual'] in imendel else ""
            co_parents_preview = ', '.join(r['co_parents'][:5])
            if r['n_co_parents'] > 5:
                co_parents_preview += f", ... and {r['n_co_parents'] - 5} more"
            rpt.write(f"  {r['individual']}  rate={r['rate']:.2%}  "
                      f"n_co_parents={r['n_co_parents']}  [{r['flag_reason']}]{imendel_note}\n")
            rpt.write(f"    co-parents: {co_parents_preview}\n")
        if len(repeat_offenders) > args.report_limit:
            rpt.write(f"  ... and {len(repeat_offenders) - args.report_limit} more "
                      f"(full list in {out_individual_tsv_path})\n")

    print(f"  Written: {out_report_path}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
