#!/usr/bin/env python3
"""
Purge MZ-twin duplicates from a VCF file based on VCFped .pair output.

Two-stage workflow:
  Stage 1: Run vcfped.py --no-trio --no-gender to identify MZ twins (fast, O(n^2)).
  Stage 2: Use this script to remove duplicate MZ-twin samples from the VCF,
           keeping one representative per MZ group. The reduced VCF is then
           suitable for trio analysis with far fewer combinations (O(n^3)).

Usage:
  python purge_mz_twins.py <input.vcf[.gz]> <prefix.pair> -o <output.vcf[.gz]>

The .pair file is the best-calls output from vcfped.py. This script:
  1. Parses MZ-twin pairs from the .pair file
  2. Groups them transitively (if A=B and B=C, keeps only A)
  3. Reads sample names from the VCF header
  4. Writes a new VCF with duplicate columns removed
  5. Reports which samples were kept and removed
"""

import argparse
import gzip
import sys
import os


def smart_open(filename, mode='r'):
    """Open plain or gzip-compressed files transparently."""
    if filename.endswith('.gz'):
        if 'b' not in mode:
            mode = mode + 't' if 't' not in mode else mode
        return gzip.open(filename, mode)
    return open(filename, mode)


def smart_open_write(filename):
    """Open for writing, gzip if filename ends with .gz."""
    if filename.endswith('.gz'):
        return gzip.open(filename, 'wt')
    return open(filename, 'w')


def parse_pair_file(pair_file, vcf_samples):
    """Parse .pair file and return list of (sample_a, sample_b) MZ-twin pairs.

    The .pair file uses 1-based sample indices in the Pair column (e.g. '3,17').
    vcf_samples maps those indices back to sample names.
    """
    mz_pairs = []
    with open(pair_file) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 10:
                continue
            verdict = fields[-1]
            if verdict != 'MZ twins':
                continue
            pair_str = fields[0]
            try:
                idx_a, idx_b = [int(x) for x in pair_str.split(',')]
            except ValueError:
                continue
            # Convert 1-based indices to sample names
            name_a = vcf_samples[idx_a - 1]
            name_b = vcf_samples[idx_b - 1]
            mz_pairs.append((name_a, name_b))
    return mz_pairs


def group_mz_twins(mz_pairs):
    """Group MZ-twin pairs into connected components using union-find.

    Returns a list of sets, each containing sample names that are MZ twins
    of each other (transitive closure).
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in mz_pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    groups = {}
    for sample in parent:
        root = find(sample)
        groups.setdefault(root, set()).add(sample)

    return list(groups.values())


def get_vcf_sample_info(vcf_file):
    """Read VCF header and return (header_lines, column_header_line, sample_names, format_col_idx).

    Handles both standard VCF (#CHROM) and non-standard (annotated) VCF files,
    matching the logic in vcfped.py inspectFile().
    """
    header_lines = []
    with smart_open(vcf_file) as f:
        for line in f:
            if line.startswith('##'):
                header_lines.append(line)
                continue
            # This is the column header line
            col_header = line
            fields = line.rstrip('\n').split('\t')
            # Find FORMAT column (same heuristic as vcfped)
            # Read the first data line to locate FORMAT
            first_data = f.readline()
            first_fields = first_data.rstrip('\n').split('\t')
            break

    # Standard VCF: FORMAT is at index 8
    if fields[0] == '#CHROM' and len(fields) > 8 and fields[8] == 'FORMAT':
        format_col = 8
    else:
        # Non-standard: find the column where data starts with "GT:" or is "GT"
        format_col = None
        for i, val in enumerate(first_fields):
            if val.startswith('GT:') or val == 'GT':
                format_col = i
                break
        if format_col is None:
            raise RuntimeError("Could not identify FORMAT column in VCF")

    sample_names = [s.rstrip('\n') for s in fields[format_col + 1:]]
    return header_lines, col_header, sample_names, format_col


def sample_heterozygosity(vcf_file, format_col, sample_names, wanted):
    """Fraction of heterozygous calls (among non-missing) for each sample in `wanted`.

    Only the wanted sample columns are parsed. Returns {name: het_fraction}; a
    sample without any non-missing call gets 0.0.
    """
    idx = {format_col + 1 + i: n for i, n in enumerate(sample_names) if n in wanted}
    het = dict.fromkeys(idx.values(), 0)
    called = dict.fromkeys(idx.values(), 0)
    with smart_open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            for col, name in idx.items():
                if col >= len(fields):
                    continue
                gt = fields[col].split(':', 1)[0].replace('|', '/')
                a = gt.split('/')
                if len(a) != 2 or '.' in a:
                    continue
                called[name] += 1
                if a[0] != a[1]:
                    het[name] += 1
    return {n: (het[n] / called[n] if called[n] else 0.0) for n in idx.values()}


def filter_pairs_by_heterozygosity(mz_pairs, het, min_ratio):
    """Split MZ-twin pairs into (kept, rejected) by heterozygosity similarity.

    vcfped calls a pair "MZ twins" on allele sharing alone. A selfed child of a
    heterozygous parent shares almost everything with it but has about half its
    heterozygosity, and two deeply inbred lines can look alike the same way;
    a true duplicate has the same heterozygosity as its twin. A pair is
    rejected when min(het)/max(het) < min_ratio. Returns (kept, rejected) with
    rejected as (a, b, ratio). min_ratio <= 0 keeps everything.
    """
    if min_ratio <= 0:
        return list(mz_pairs), []
    kept, rejected = [], []
    for a, b in mz_pairs:
        hi, lo = max(het[a], het[b]), min(het[a], het[b])
        ratio = lo / hi if hi > 0 else 1.0
        if ratio < min_ratio:
            rejected.append((a, b, ratio))
        else:
            kept.append((a, b))
    return kept, rejected


def purge_vcf(vcf_file, output_file, samples_to_remove, format_col):
    """Write a new VCF with specified sample columns removed."""
    # Read header to find which column indices to drop
    with smart_open(vcf_file) as f:
        with smart_open_write(output_file) as out:
            for line in f:
                if line.startswith('##'):
                    out.write(line)
                    continue

                # Column header line
                fields = line.rstrip('\n').split('\t')
                sample_cols = fields[format_col + 1:]
                # Build index of columns to keep (fixed cols + kept samples)
                remove_indices = set()
                for i, name in enumerate(sample_cols):
                    if name in samples_to_remove:
                        remove_indices.add(format_col + 1 + i)

                # Write filtered header
                kept = [f for i, f in enumerate(fields) if i not in remove_indices]
                out.write('\t'.join(kept) + '\n')

                # Process data lines
                for line in f:
                    fields = line.rstrip('\n').split('\t')
                    kept = [f for i, f in enumerate(fields) if i not in remove_indices]
                    out.write('\t'.join(kept) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Purge MZ-twin duplicates from a VCF file.',
        epilog="""
Two-stage workflow:
  Step 1: python -m vcfped.vcfped input.vcf.gz --no-trio --no-gender -o step1
  Step 2: python purge_mz_twins.py input.vcf.gz step1.pair -o input_no_mz.vcf.gz
  Step 3: python -m vcfped.vcfped input_no_mz.vcf.gz --no-pairwise -o step3
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('vcf', help='Input VCF file (.vcf or .vcf.gz)')
    parser.add_argument('pair_file', help='VCFped .pair output file from step 1')
    parser.add_argument('-o', '--output', required=True,
                        help='Output VCF file (.vcf or .vcf.gz)')
    parser.add_argument('--keep', choices=['first', 'last'], default='first',
                        help='Which sample to keep per MZ group: first or last in VCF order (default: first)')
    parser.add_argument('--min-het-ratio', type=float, default=0.9,
                        help='Only treat a vcfped MZ-twin pair as a duplicate if the two samples have similar '
                             'heterozygosity: min(het)/max(het) >= this value (default: 0.9; 0 disables). '
                             'Protects selfed children and inbred lines that vcfped calls "MZ twins".')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report what would be removed, do not write output')

    args = parser.parse_args()

    if not os.path.exists(args.vcf):
        sys.exit(f"Error: VCF file not found: {args.vcf}")
    if not os.path.exists(args.pair_file):
        sys.exit(f"Error: Pair file not found: {args.pair_file}")

    # Read sample names from VCF
    header_lines, col_header, vcf_samples, format_col = get_vcf_sample_info(args.vcf)
    print(f"VCF contains {len(vcf_samples)} samples")

    # Parse MZ twins from .pair file
    mz_pairs = parse_pair_file(args.pair_file, vcf_samples)
    if not mz_pairs:
        print("No MZ twins found in .pair file. No filtering needed.")
        sys.exit(0)

    print(f"Found {len(mz_pairs)} MZ-twin pair(s)")

    if args.min_het_ratio > 0:
        wanted = {s for pair in mz_pairs for s in pair}
        het = sample_heterozygosity(args.vcf, format_col, vcf_samples, wanted)
        mz_pairs, rejected = filter_pairs_by_heterozygosity(mz_pairs, het, args.min_het_ratio)
        if rejected:
            print(f"Kept {len(rejected)} pair(s) apart: heterozygosity ratio < {args.min_het_ratio} "
                  f"(selfed child, inbred line or other near-identical but distinct sample):")
            for a, b, ratio in sorted(rejected, key=lambda r: r[2])[:20]:
                print(f"  {a} ({het[a]:.3f}) vs {b} ({het[b]:.3f})  ratio={ratio:.2f}")
            if len(rejected) > 20:
                print(f"  ... and {len(rejected) - 20} more")
        if not mz_pairs:
            print("No MZ twins left after the heterozygosity check. No filtering needed.")
            sys.exit(0)
        print(f"{len(mz_pairs)} pair(s) remain")

    # Group transitively
    groups = group_mz_twins(mz_pairs)
    print(f"Grouped into {len(groups)} MZ cluster(s):")

    # Determine which to keep and which to remove
    samples_to_remove = set()
    for group in groups:
        # Sort by VCF column order
        ordered = sorted(group, key=lambda s: vcf_samples.index(s))
        if args.keep == 'first':
            keep = ordered[0]
            remove = ordered[1:]
        else:
            keep = ordered[-1]
            remove = ordered[:-1]
        samples_to_remove.update(remove)
        print(f"  Cluster ({len(ordered)} samples): keep={keep}, remove={', '.join(remove)}")

    remaining = len(vcf_samples) - len(samples_to_remove)
    print(f"\nRemoving {len(samples_to_remove)} sample(s), {remaining} remaining")

    n_old = len(vcf_samples)
    n_new = remaining
    pairs_old = n_old * (n_old - 1) // 2
    triples_old = n_old * (n_old - 1) * (n_old - 2) // 6
    triples_new = n_new * (n_new - 1) * (n_new - 2) // 6
    if triples_old > 0:
        print(f"Trio combinations: {triples_old:,} -> {triples_new:,} ({100*(1 - triples_new/triples_old):.1f}% reduction)")

    if args.dry_run:
        print("\nDry run - no output written.")
        return

    # Write filtered VCF
    print(f"\nWriting filtered VCF to {args.output}...")
    purge_vcf(args.vcf, args.output, samples_to_remove, format_col)
    print("Done.")


if __name__ == '__main__':
    main()
