#!/usr/bin/env python3
"""
Post-process vcfped .trio output to resolve ambiguous parent assignments.

BACKGROUND
----------
VCFped tests every combination of three samples and reports any triple that
passes the allele-sharing thresholds.  When the cohort contains related
individuals, the same child can appear with several parent-pair assignments:

  Child C  ← parents (Father F,  Mother M)   T1=100%  Perc=50  <- true
  Child C  ← parents (Brother B, Mother M)   T1= 92%  Perc=10  <- artefact
  Child C  ← parents (Father F,  Aunt A)     T1= 91%  Perc=10  <- artefact

Brother B and Father F share ~50% of alleles, so (B, M, C) also passes at
low stringency.  The job of this script is to identify and remove artefactual
assignments, leaving one reliable parent pair per child.

STRATEGIES
----------
best       Keep the single call with the highest (Perc, T1%) per child.
           Fast and safe; use when you trust the scores.
           Default.

strict     Keep all calls where BOTH Perc >= --min-perc AND T1 >= --min-t1.
           May still leave some children with multiple pairs if two calls
           pass the threshold; inspect the report for those children.

consensus  Extend 'best' with a relatedness report: when two competing pairs
           share one parent (e.g. both contain Father F), that shared parent
           is flagged as 'likely true parent'; the differing parents are
           flagged as 'likely relatives of true parent'.

OUTPUT
------
  <prefix>_filtered.trio  — cleaned file in the same format as the input
  stdout                  — summary report

Usage:
    python filter_trios.py step3_named
    python filter_trios.py step3_named --strategy strict --min-perc 30 --min-t1 95
    python filter_trios.py step3_named --strategy consensus -o step3_clean
"""

import sys
import argparse
import csv
from collections import defaultdict

# Expected .trio columns
_REQUIRED = {'Triple', 'Pivot', 'Perc', 'Test1', 'Verdict'}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_trio_file(path):
    """Read a vcfped .trio file; return (fieldnames, list-of-dicts)."""
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except FileNotFoundError:
        sys.exit(f"Error: {path} not found.")
    except Exception as e:
        sys.exit(f"Error reading {path}: {e}")

    missing = _REQUIRED - set(fieldnames)
    if missing:
        sys.exit(
            f"Error: {path} is missing required columns: {sorted(missing)}\n"
            f"  Found: {fieldnames}\n"
            f"  This does not look like a vcfped .trio file.")
    return fieldnames, rows


def _parse_row(row):
    """Extract typed fields from a raw dict row."""
    triple = [s.strip() for s in str(row['Triple']).split(',')]
    pivot  = str(row['Pivot']).strip()
    if len(triple) != 3 or pivot not in triple:
        return None
    child   = pivot
    parents = [s for s in triple if s != child]
    if len(parents) != 2:
        return None
    return {
        'p1':      parents[0],
        'p2':      parents[1],
        'child':   child,
        'perc':    int(row.get('Perc', 0)),
        't1':      float(row.get('Test1', 0.0)),
        't2':      float(row.get('Test2', 0.0)),
        'verdict': str(row.get('Verdict', '')).strip(),
        '_raw':    row,   # keep original row for output
    }


def _write_trio_file(path, fieldnames, rows):
    """Write filtered rows to a .trio file in the same tab-separated format."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)                        # header — exactly once
        for r in rows:
            writer.writerow([r['_raw'].get(k, '') for k in fieldnames])
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_best(parsed):
    """
    For each child, keep the single call with the highest (Perc, T1%).
    Ties broken by Perc first, then T1.
    Returns (kept, removed).
    """
    best = {}
    for r in parsed:
        child = r['child']
        if child not in best or (r['perc'], r['t1']) > (best[child]['perc'], best[child]['t1']):
            best[child] = r

    kept    = list(best.values())
    removed = [r for r in parsed if r is not best[r['child']]]
    return kept, removed


def strategy_strict(parsed, min_perc, min_t1):
    """
    Keep all calls with Perc >= min_perc AND T1 >= min_t1.
    Returns (kept, removed).
    """
    kept    = [r for r in parsed if r['perc'] >= min_perc and r['t1'] >= min_t1]
    removed = [r for r in parsed if r['perc'] <  min_perc or  r['t1'] <  min_t1]
    return kept, removed


def strategy_consensus(parsed):
    """
    Like 'best', but also analyses shared parents across competing assignments
    to flag likely true parents and likely relatives.

    Returns (kept, removed, relatedness_report).
    relatedness_report: list of dicts per ambiguous child.
    """
    kept, removed = strategy_best(parsed)

    # Group all calls (including removed ones) by child
    by_child = defaultdict(list)
    for r in parsed:
        by_child[r['child']].append(r)

    report = []
    for child, calls in by_child.items():
        if len(calls) < 2:
            continue   # no ambiguity

        # Count how often each potential parent appears across all calls for this child
        parent_counts = defaultdict(int)
        for c in calls:
            parent_counts[c['p1']] += 1
            parent_counts[c['p2']] += 1

        n_calls = len(calls)
        best_call = next(r for r in kept if r['child'] == child)

        # A parent appearing in ALL calls is the most-confident true parent.
        # A parent appearing in only one call is likely a relative of the true parent.
        always_present  = [p for p, cnt in parent_counts.items() if cnt == n_calls]
        sometimes       = [p for p, cnt in parent_counts.items() if 0 < cnt < n_calls]
        competing_pairs = [(c['p1'], c['p2'], c['perc'], c['t1'])
                           for c in sorted(calls,
                                           key=lambda x: (-x['perc'], -x['t1']))
                           if c is not best_call]

        report.append({
            'child':           child,
            'n_calls':         n_calls,
            'best_p1':         best_call['p1'],
            'best_p2':         best_call['p2'],
            'best_perc':       best_call['perc'],
            'best_t1':         best_call['t1'],
            'shared_parents':  always_present,
            'alt_parents':     sometimes,
            'competing_pairs': competing_pairs,
        })

    return kept, removed, report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_header(prefix, strategy, n_total, n_regular, n_inverted):
    print()
    print("=" * 70)
    print("VCFped Trio Post-processing")
    print("=" * 70)
    print(f"  Input prefix : {prefix}")
    print(f"  Strategy     : {strategy}")
    print(f"  Total rows   : {n_total}  "
          f"(Regular: {n_regular}, Inverted/other: {n_inverted})")
    print()


def _print_ambiguity_summary(parsed, title="Before filtering"):
    by_child = defaultdict(list)
    for r in parsed:
        by_child[r['child']].append(r)
    ambiguous = {c: calls for c, calls in by_child.items() if len(calls) > 1}
    print(f"  {title}:")
    print(f"    Unique children   : {len(by_child)}")
    print(f"    Ambiguous children: {len(ambiguous)}  "
          f"({100*len(ambiguous)/max(len(by_child),1):.0f}%)")
    if ambiguous:
        sizes = sorted(len(v) for v in ambiguous.values())
        print(f"    Parent-pair counts per ambiguous child: "
              f"min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}")
    print()
    return ambiguous


def _print_removed_summary(removed):
    if not removed:
        print("  No calls removed.")
        return
    by_child = defaultdict(list)
    for r in removed:
        by_child[r['child']].append(r)
    print(f"  Removed {len(removed)} calls from {len(by_child)} children.")
    print(f"  Perc distribution of removed calls: "
          + ", ".join(f"p{p}:{sum(1 for r in removed if r['perc']==p)}"
                      for p in sorted({r['perc'] for r in removed})))
    print()


def _print_consensus_report(report, max_show=20):
    if not report:
        return
    print(f"  Consensus relatedness report ({len(report)} ambiguous children):")
    print()
    for entry in report[:max_show]:
        child = entry['child']
        print(f"  Child: {child}  "
              f"[best: ({entry['best_p1']}, {entry['best_p2']})  "
              f"Perc={entry['best_perc']}  T1={entry['best_t1']:.1f}%]")
        if entry['shared_parents']:
            print(f"    Likely true parent(s) — appear in ALL {entry['n_calls']} calls:")
            for p in entry['shared_parents']:
                print(f"      {p}")
        if entry['alt_parents']:
            print(f"    Likely relatives of true parent — appear in some calls only:")
            for p in entry['alt_parents']:
                print(f"      {p}")
        if entry['competing_pairs']:
            print(f"    Competing pairs (not selected):")
            for p1, p2, perc, t1 in entry['competing_pairs']:
                print(f"      ({p1}, {p2})  Perc={perc}  T1={t1:.1f}%")
        print()
    if len(report) > max_show:
        print(f"  ... and {len(report) - max_show} more (use --report-all to show all).")
    print()


def _print_kept_summary(kept):
    by_child = defaultdict(list)
    for r in kept:
        by_child[r['child']].append(r)
    still_ambiguous = {c: v for c, v in by_child.items() if len(v) > 1}
    print(f"  After filtering:")
    print(f"    Unique children        : {len(by_child)}")
    print(f"    Still ambiguous        : {len(still_ambiguous)}")
    perc_counts = defaultdict(int)
    for r in kept:
        perc_counts[r['perc']] += 1
    print(f"    Calls per stringency   : "
          + ", ".join(f"Perc={p}: {n}" for p, n in sorted(perc_counts.items())))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Resolve ambiguous parent assignments in vcfped .trio output',
        epilog='''
Strategies:
  best      Keep the single highest-confidence call per child (default).
  strict    Keep all calls passing --min-perc and --min-t1 thresholds.
  consensus Like best, but also reports which competing parents are
            likely relatives of the true parents.

Examples:
  %(prog)s step3_named
  %(prog)s step3_named --strategy strict --min-perc 30 --min-t1 95
  %(prog)s step3_named --strategy consensus --report-all
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('prefix',
                        help='VCFped output prefix (reads <prefix>.trio)')
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: <prefix>_filtered)',
                        default=None)
    parser.add_argument('--strategy', choices=['best', 'strict', 'consensus'],
                        default='best',
                        help='Filtering strategy (default: best)')
    parser.add_argument('--min-perc', type=int, default=30,
                        help='Minimum Perc for --strategy strict (default: 30)')
    parser.add_argument('--min-t1', type=float, default=95.0,
                        help='Minimum T1%% for --strategy strict (default: 95.0)')
    parser.add_argument('--include-inverted', action='store_true',
                        help='Also process Inverted/generational trios '
                             '(default: pass them through unchanged)')
    parser.add_argument('--report-all', action='store_true',
                        help='Show consensus report for all ambiguous children '
                             '(default: first 20)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.prefix}_filtered"

    trio_path = f"{args.prefix}.trio"
    out_path  = f"{args.output}.trio"

    fieldnames, raw_rows = _read_trio_file(trio_path)

    # Split into regular vs inverted vs other passthrough
    parsed_regular  = []
    inverted_all    = []   # all inverted/generational rows — also written to _inverted.trio
    passthrough     = []   # other non-regular verdict rows, passed through unchanged
    for row in raw_rows:
        p = _parse_row(row)
        verdict = str(row.get('Verdict', '')).strip()
        if p is None:
            passthrough.append(row)
            continue
        is_inverted = 'Inverted' in verdict or 'generational' in verdict.lower()
        if verdict == 'Regular trio':
            parsed_regular.append(p)
        elif is_inverted:
            inverted_all.append(row)  # always collect for _inverted.trio
            if args.include_inverted:
                parsed_regular.append(p)
            else:
                passthrough.append(row)  # also pass through in main output
        else:
            passthrough.append(row)

    n_inverted = len(inverted_all)

    _print_header(args.prefix, args.strategy,
                  len(raw_rows), len(parsed_regular), n_inverted)

    ambiguous_before = _print_ambiguity_summary(parsed_regular, "Before filtering")

    # Apply strategy
    consensus_report = []
    if args.strategy == 'best':
        kept, removed = strategy_best(parsed_regular)
    elif args.strategy == 'strict':
        kept, removed = strategy_strict(parsed_regular, args.min_perc, args.min_t1)
        if any(len(v) > 1
               for v in defaultdict(list,
                   {r['child']: [x for x in kept if x['child']==r['child']] for r in kept}).values()):
            print("  Note: --strategy strict may leave some children with multiple pairs.")
            print(f"        Try raising --min-perc (current: {args.min_perc}) "
                  f"or --min-t1 (current: {args.min_t1}) to resolve remaining ambiguity,")
            print(f"        or run again with --strategy best on the output.\n")
    else:  # consensus
        kept, removed, consensus_report = strategy_consensus(parsed_regular)

    _print_removed_summary(removed)

    if consensus_report:
        max_show = None if args.report_all else 20
        _print_consensus_report(consensus_report,
                                max_show=len(consensus_report) if args.report_all else 20)

    _print_kept_summary(kept)

    # Warn if strict still leaves ambiguity
    if args.strategy == 'strict':
        by_child = defaultdict(list)
        for r in kept:
            by_child[r['child']].append(r)
        still = {c: v for c, v in by_child.items() if len(v) > 1}
        if still:
            print(f"  ⚠  {len(still)} children still have multiple parent pairs "
                  f"after strict filtering.")
            print(f"     Increase thresholds or switch to --strategy best.\n")

    # Write output
    print(f"Writing filtered output...")
    out_rows = [r['_raw'] for r in kept] + passthrough
    # Re-sort to match original order: (Triple, Pivot)
    orig_order = {(r['Triple'], r['Pivot']): i for i, r in enumerate(raw_rows)}
    out_rows.sort(key=lambda r: orig_order.get((r.get('Triple',''), r.get('Pivot','')), 99999))
    _write_trio_file(out_path, fieldnames, [{'_raw': r} for r in out_rows])

    # Write _inverted.trio so refine_pedigree.py --include-inverted can find them
    inv_out_path = f"{args.output}_inverted.trio"
    if inverted_all:
        _write_trio_file(inv_out_path, fieldnames, [{'_raw': r} for r in inverted_all])

    print()
    print("Summary:")
    print(f"  Input  : {len(parsed_regular)} Regular trio calls, "
          f"{len(ambiguous_before)} ambiguous children")
    print(f"  Removed: {len(removed)} calls")
    print(f"  Output : {len(kept)} Regular trio calls → {out_path}")
    if inverted_all:
        print(f"  Inverted/generational: {len(inverted_all)} rows → {inv_out_path}")
    n_other_pass = sum(1 for r in passthrough
                       if 'Inverted' not in str(r.get('Verdict', ''))
                       and 'generational' not in str(r.get('Verdict', '')).lower())
    if n_other_pass:
        print(f"  Passed through unchanged: {n_other_pass} other non-Regular rows")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
