#!/usr/bin/env python3
"""
Convert a PLINK .fam file's confirmed parent-child links into a vcfped .trio
file, so tools that only read .trio (fixed_pedigree_viz.py, refine_pedigree.py's
--ibd final-confirmation pass) can consume a pedigree that now only exists as
a .fam -- e.g. merge_refined_fam.py's output after folding several rounds of
candidate-subset extension back into the full-cohort FAM.

BACKGROUND
----------
A working pedigree built via the iterative extend/vcfped-rerun/merge cycle
accumulates confirmed trios from MULTIPLE originating .trio files (the
original vcfped run, plus one per candidate-subset round) into a single
merged .fam. There is no single .trio file representing that combined state.

Most trios in the merged FAM DO have real evidence behind them, though --
they just live in one of those original per-round .trio files, keyed by the
CURRENT (p1, p2, child) triple. This script looks each confirmed FAM row up
across one or more --trio sources and carries over the REAL vcfped
Perc/Autos/AA+BB/=AB/Test1/BB+BB/=BB/Test2 when the current pair matches a
row there exactly (i.e. refine_pedigree.py confirmed the original assignment
as-is, or merge_refined_fam.py kept the base side of a conflict).

Only a MINORITY of rows have no such match: ones where refine_pedigree.py
substituted a different parent (mz_swap/king_new) or merge_refined_fam.py
picked the "new" side of a conflict -- vcfped never scored that exact final
combination. For these, if --king is given, a KING kinship-based confidence
proxy is used for Test1 instead of a meaningless flat placeholder (the raw
vcfped counts don't apply and stay 0). Perc is set to 0 for every such row --
a value real vcfped output never produces (its own Perc is always one of the
--no-pairwise/--no-trio quality-filter percentiles, e.g. 10/30/50), so it
doubles as a visible, self-documenting marker distinguishing "genuinely
vcfped-tested" rows from "KING-confirmed only" ones in fixed_pedigree_viz.py's
table -- without needing a different Verdict string, which would break
refine_pedigree.py's default Verdict == 'Regular trio' filter downstream.

INPUT
-----
  <fam_path>     — a .fam file (any prefix; the trailing '.fam' is optional)
  --trio <path>  — one or more original *_refined.trio files to look up real
                   vcfped statistics from (repeatable)
  --king <prefix> — PLINK2 --make-king-table prefix (reads <prefix>.kin0),
                   for a kinship-based Test1 proxy on rows with no --trio match

OUTPUT
------
  <output>.trio — standard vcfped .trio format (Triple/Pivot/Perc/Autos/
                  AA+BB/=AB/Test1/BB+BB/=BB/Test2/Verdict), Verdict='Regular
                  trio' for every confirmed PAT+MAT+child row

Usage:
  python fam_to_trio.py step3_named_filtered_refined_merged2.fam
  python fam_to_trio.py step3_named_filtered_refined_merged2.fam \\
      --trio step3_named_filtered_refined.trio \\
      --trio extension_out_named_filtered_refined.trio \\
      --trio extension_out2_named_filtered_refined.trio \\
      --king revalidate_king_ext -o merged2_trio
"""

import argparse
import csv
import os
import sys

TRIO_HEADER = ['Triple', 'Pivot', 'Perc', 'Autos', 'AA+BB', '=AB',
               'Test1', 'BB+BB', '=BB', 'Test2', 'Verdict']


def fam_to_pairs(fam_path):
    """Parse a .fam file's confirmed (PAT and MAT both known) parent-child
    links into a list of (p1, p2, child) tuples."""
    if not os.path.exists(fam_path):
        sys.exit(f"Error: {fam_path} not found.")
    trios = []
    with open(fam_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iid, pat, mat = parts[1], parts[2], parts[3]
            if pat != '0' and mat != '0':
                trios.append((pat, mat, iid))
    return trios


def load_trio_lookup(trio_paths):
    """
    Build a lookup (child, frozenset({p1,p2})) -> real vcfped stats dict,
    from one or more .trio files (only Verdict == 'Regular trio' rows).
    First match wins if the same key appears in more than one source file.
    Skips paths that don't exist rather than failing -- --trio sources are
    optional context, not a hard requirement.
    """
    lookup = {}
    for path in trio_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row.get('Verdict', '').strip() != 'Regular trio':
                    continue
                triple = [s.strip() for s in str(row.get('Triple', '')).split(',')]
                pivot = str(row.get('Pivot', '')).strip()
                if len(triple) != 3 or pivot not in triple:
                    continue
                parents = [s for s in triple if s != pivot]
                if len(parents) != 2:
                    continue
                key = (pivot, frozenset(parents))
                if key not in lookup:
                    lookup[key] = {
                        'perc': row.get('Perc', '0'),
                        'autos': row.get('Autos', '0'),
                        'aa_bb': row.get('AA+BB', '0'),
                        'ab': row.get('=AB', '0'),
                        'test1': row.get('Test1', '0.0'),
                        'bb_bb': row.get('BB+BB', '0'),
                        'bb': row.get('=BB', '0'),
                        'test2': row.get('Test2', '0.0'),
                    }
    return lookup


def load_king_lookup(king_prefix):
    """Parse a PLINK2 --make-king-table .kin0 file into {(id1,id2): {kinship, ibs0}}.
    Optional -- returns {} if no prefix is given or the file doesn't exist."""
    king = {}
    if not king_prefix:
        return king
    path = king_prefix if king_prefix.endswith('.kin0') else f"{king_prefix}.kin0"
    if not os.path.exists(path):
        return king
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        id1_col = next((c for c in fieldnames if 'ID1' in c and 'FID' not in c), None)
        id2_col = next((c for c in fieldnames if 'ID2' in c and 'FID' not in c), None)
        if not id1_col or not id2_col:
            return king
        for row in reader:
            id1 = row[id1_col].strip()
            id2 = row[id2_col].strip()
            try:
                kinship = float(row.get('KINSHIP', row.get('Kinship', 0)))
            except ValueError:
                continue
            king[(id1, id2)] = {'kinship': kinship}
    return king


def get_king(king, a, b):
    return king.get((a, b)) or king.get((b, a))


def kinship_confidence(king, p1, p2, child):
    """
    Kinship-based Test1 proxy (0-100) for a row with no real vcfped
    statistical test -- the average of p1<->child and p2<->child KING
    kinship, scaled to a percentage. Not vcfped's T1 statistic (a
    Mendelian-consistency rate); a different, independent signal, used only
    because it's genuinely meaningful evidence rather than a fake placeholder.
    Returns 0.0 if no KING data is available for either pair.
    """
    kinships = []
    for p in (p1, p2):
        rec = get_king(king, p, child)
        if rec is not None:
            kinships.append(rec['kinship'])
    if not kinships:
        return 0.0
    return max(0.0, min(100.0, sum(kinships) / len(kinships) * 100))


def build_trio_rows(pairs, trio_lookup, king):
    rows = []
    for p1, p2, child in pairs:
        key = (child, frozenset([p1, p2]))
        real = trio_lookup.get(key)
        if real is not None:
            rows.append({
                'p1': p1, 'p2': p2, 'child': child,
                'perc': real['perc'], 'autos': real['autos'], 'aa_bb': real['aa_bb'],
                'ab': real['ab'], 'test1': real['test1'], 'bb_bb': real['bb_bb'],
                'bb': real['bb'], 'test2': real['test2'],
            })
        else:
            rows.append({
                'p1': p1, 'p2': p2, 'child': child,
                # Perc=0 is never produced by real vcfped output (its Perc is
                # always one of the quality-filter percentiles, e.g. 10/30/50)
                # -- a self-documenting marker that this row has no real
                # vcfped test behind it.
                'perc': '0', 'autos': '0', 'aa_bb': '0', 'ab': '0',
                'test1': f"{kinship_confidence(king, p1, p2, child):.1f}",
                'bb_bb': '0', 'bb': '0', 'test2': '0.0',
            })
    return rows


def write_trio_file(path, rows):
    with open(path, 'w') as f:
        f.write('\t'.join(TRIO_HEADER) + '\n')
        for r in rows:
            triple = ','.join([r['p1'], r['p2'], r['child']])
            f.write('\t'.join([
                triple, r['child'], str(r['perc']), str(r['autos']), str(r['aa_bb']),
                str(r['ab']), str(r['test1']), str(r['bb_bb']), str(r['bb']),
                str(r['test2']), 'Regular trio',
            ]) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Convert a .fam file's confirmed parent-child links into a vcfped .trio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_named_filtered_refined_merged2.fam
  %(prog)s step3_named_filtered_refined_merged2.fam \\
      --trio step3_named_filtered_refined.trio \\
      --trio extension_out_named_filtered_refined.trio \\
      --trio extension_out2_named_filtered_refined.trio \\
      --king revalidate_king_ext -o merged2_trio
        """,
    )
    parser.add_argument('fam_path', help='.fam file to convert')
    parser.add_argument('--trio', action='append', default=[],
                        help='Original *_refined.trio file to look up real vcfped statistics '
                             'from (repeatable). Rows are matched by (child, {p1,p2}) against '
                             'the CURRENT pair in the FAM.')
    parser.add_argument('--king', default=None,
                        help='PLINK2 --make-king-table prefix (reads <prefix>.kin0), for a '
                             'kinship-based Test1 proxy on rows with no --trio match')
    parser.add_argument('-o', '--output', default=None,
                        help='Output prefix (default: derived from fam_path)')
    args = parser.parse_args()

    base = args.fam_path[:-len('.fam')] if args.fam_path.endswith('.fam') else args.fam_path
    out_prefix = args.output or base
    out_path = f"{out_prefix}.trio"

    pairs = fam_to_pairs(args.fam_path)
    if not pairs:
        sys.exit(f"Error: no confirmed (PAT and MAT both known) trios found in {args.fam_path}.")

    trio_lookup = load_trio_lookup(args.trio)
    king = load_king_lookup(args.king)
    if args.trio:
        print(f"  {len(trio_lookup)} real vcfped trio rows loaded from {len(args.trio)} source file(s)")
    if args.king:
        print(f"  {len(king)} KING pairs loaded from {args.king}")

    rows = build_trio_rows(pairs, trio_lookup, king)
    n_real = sum(1 for r in rows if r['perc'] != '0')
    n_king_only = len(rows) - n_real
    print(f"  {len(rows)} confirmed trios converted from {args.fam_path}")
    print(f"    {n_real} matched real vcfped statistics")
    print(f"    {n_king_only} KING-confirmed only (no matching vcfped test; Perc=0)")

    write_trio_file(out_path, rows)
    print(f"  Written: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
