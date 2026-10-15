#!/usr/bin/env python
"""
Analyze VCFped pair file to diagnose MZ twin relationship issues.

This tool helps identify problems with excessive or incorrect MZ twin calls.

Usage:
    python analyze_vcfped_pairs.py results_named
    python analyze_vcfped_pairs.py results_named --threshold 0.95
"""

import sys
import argparse
import pandas as pd
from collections import defaultdict, Counter
import numpy as np


def analyze_pair_file(prefix, mz_threshold=None):
    """
    Analyze VCFped pair file to understand MZ twin calls.
    
    Args:
        prefix: Path prefix to VCFped output files
        mz_threshold: Optional IBD0/relatedness threshold to filter MZ calls
    """
    
    print("="*80)
    print("VCFped Pair File Analysis")
    print("="*80)
    print()
    
    # Read pair file
    try:
        pair_df = pd.read_csv(f"{prefix}.pair", sep='\t')
    except FileNotFoundError:
        print(f"Error: Could not find {prefix}.pair")
        return 1
    
    print(f"Total pairs in file: {len(pair_df)}")
    print()
    
    # Count verdicts
    print("Verdict Distribution:")
    print("-" * 80)
    verdict_counts = pair_df['Verdict'].value_counts()
    for verdict, count in verdict_counts.items():
        pct = 100 * count / len(pair_df)
        print(f"  {verdict:25s}: {count:8d} ({pct:5.1f}%)")
    print()
    
    # Focus on MZ twins
    mz_df = pair_df[pair_df['Verdict'] == 'MZ twins'].copy()
    
    if len(mz_df) == 0:
        print("No MZ twin pairs found!")
        return 0
    
    print(f"MZ Twin Pairs: {len(mz_df)}")
    print("="*80)
    print()
    
    # Check if there are relatedness/IBD columns
    numeric_cols = mz_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("Available numeric columns for filtering:")
        for col in numeric_cols:
            print(f"  - {col}")
        print()
    
    # Parse pairs and count connections per individual
    individual_connections = defaultdict(list)
    all_pairs = []
    
    for _, row in mz_df.iterrows():
        pair = str(row['Pair']).split(',')
        if len(pair) != 2:
            continue
        
        sample1, sample2 = pair[0].strip(), pair[1].strip()
        all_pairs.append((sample1, sample2))
        
        # Store IBD0 or relatedness if available
        metadata = {}
        if 'IBD0' in row:
            metadata['IBD0'] = row['IBD0']
        if 'Relatedness' in row:
            metadata['Relatedness'] = row['Relatedness']
        if 'IBS0' in row:
            metadata['IBS0'] = row['IBS0']
        
        individual_connections[sample1].append((sample2, metadata))
        individual_connections[sample2].append((sample1, metadata))
    
    print(f"Unique individuals in MZ twin pairs: {len(individual_connections)}")
    print()
    
    # Count how many MZ twin pairs each individual has
    connection_counts = {ind: len(partners) for ind, partners in individual_connections.items()}
    count_distribution = Counter(connection_counts.values())
    
    print("Distribution of MZ twin partners per individual:")
    print("-" * 80)
    for n_partners in sorted(count_distribution.keys())[:20]:  # Show first 20
        count = count_distribution[n_partners]
        print(f"  {n_partners:4d} partner(s): {count:5d} individuals")
    
    if len(count_distribution) > 20:
        print(f"  ... ({len(count_distribution) - 20} more values)")
    print()
    
    # Identify individuals with many MZ twin connections (likely problematic)
    print("Top individuals with most MZ twin partners:")
    print("-" * 80)
    top_individuals = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    for ind, n_partners in top_individuals:
        print(f"  {ind:20s}: {n_partners:4d} partners")
        
        # Show some example partners
        partners = individual_connections[ind][:5]
        for partner, metadata in partners:
            meta_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in metadata.items()])
            if meta_str:
                print(f"      → {partner:20s} ({meta_str})")
            else:
                print(f"      → {partner}")
        
        if len(individual_connections[ind]) > 5:
            print(f"      ... and {len(individual_connections[ind]) - 5} more")
        print()
    
    # Analyze IBD0/Relatedness distributions if available
    if 'IBD0' in mz_df.columns:
        print("\nIBD0 Distribution for MZ Twin Calls:")
        print("-" * 80)
        ibd0_values = mz_df['IBD0'].dropna()
        
        print(f"  Count: {len(ibd0_values)}")
        print(f"  Mean:  {ibd0_values.mean():.4f}")
        print(f"  Std:   {ibd0_values.std():.4f}")
        print(f"  Min:   {ibd0_values.min():.4f}")
        print(f"  Max:   {ibd0_values.max():.4f}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(ibd0_values, p)
            print(f"    {p:2d}%: {val:.4f}")
        
        # Count how many have IBD0 > threshold values
        print("\n  IBD0 threshold analysis:")
        for thresh in [0.0, 0.01, 0.05, 0.10]:
            count = (ibd0_values > thresh).sum()
            pct = 100 * count / len(ibd0_values)
            print(f"    IBD0 > {thresh:.2f}: {count:6d} pairs ({pct:5.1f}%)")
        print()
    
    if 'Relatedness' in mz_df.columns:
        print("\nRelatedness Distribution for MZ Twin Calls:")
        print("-" * 80)
        rel_values = mz_df['Relatedness'].dropna()
        
        print(f"  Count: {len(rel_values)}")
        print(f"  Mean:  {rel_values.mean():.4f}")
        print(f"  Std:   {rel_values.std():.4f}")
        print(f"  Min:   {rel_values.min():.4f}")
        print(f"  Max:   {rel_values.max():.4f}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(rel_values, p)
            print(f"    {p:2d}%: {val:.4f}")
        
        # Count how many have Relatedness < threshold values
        print("\n  Relatedness threshold analysis:")
        for thresh in [1.0, 0.95, 0.90, 0.85]:
            count = (rel_values < thresh).sum()
            pct = 100 * count / len(rel_values)
            print(f"    Relatedness < {thresh:.2f}: {count:6d} pairs ({pct:5.1f}%)")
        print()
    
    # Build transitive groups
    print("\nTransitive MZ Twin Groups:")
    print("-" * 80)
    print("(Building connected components...)")
    
    # Union-Find to build connected components
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for sample1, sample2 in all_pairs:
        union(sample1, sample2)
    
    # Group individuals by root
    groups = defaultdict(list)
    for ind in individual_connections.keys():
        root = find(ind)
        groups[root].append(ind)
    
    # Sort groups by size
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    
    print(f"  Number of connected components: {len(sorted_groups)}")
    print(f"  Group sizes: {[len(g) for g in sorted_groups[:10]]}")
    
    if len(sorted_groups) > 10:
        print(f"  ... and {len(sorted_groups) - 10} more groups")
    print()
    
    # Show details of largest groups
    print("Largest MZ Twin Groups:")
    print("-" * 80)
    for i, group in enumerate(sorted_groups[:5], 1):
        print(f"\n  Group {i}: {len(group)} individuals")
        print(f"    Members: {', '.join(sorted(group)[:10])}")
        if len(group) > 10:
            print(f"    ... and {len(group) - 10} more")
    
    print()
    print("="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    # Diagnose the problem
    avg_partners = np.mean(list(connection_counts.values()))
    max_partners = max(connection_counts.values())
    
    if len(mz_df) > len(individual_connections) * 2:
        print("⚠️  WARNING: You have an excessive number of MZ twin calls!")
        print(f"    {len(mz_df)} pairs for {len(individual_connections)} individuals")
        print(f"    Average: {avg_partners:.1f} partners per individual")
        print()
        print("   This suggests one of these issues:")
        print("   1. VCFped threshold for MZ twins is too lenient")
        print("   2. Duplicate or cloned samples in your data")
        print("   3. Data quality issues (contamination, batch effects)")
        print()
    
    if max_partners > 10:
        print(f"⚠️  WARNING: Some individuals have {max_partners} MZ twin partners!")
        print("   True MZ twins should only have 1 partner (their twin).")
        print()
    
    if len(sorted_groups) < 10 and len(individual_connections) > 100:
        print(f"⚠️  WARNING: Only {len(sorted_groups)} connected components for {len(individual_connections)} individuals!")
        print("   This means most individuals are transitively connected as 'twins'.")
        print()
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 80)
    
    if 'IBD0' in mz_df.columns:
        ibd0_median = mz_df['IBD0'].median()
        if ibd0_median > 0.01:
            print(f"• Consider filtering MZ twin calls by IBD0 < 0.01")
            print(f"  (Current median IBD0: {ibd0_median:.4f})")
    
    if 'Relatedness' in mz_df.columns:
        rel_median = mz_df['Relatedness'].median()
        if rel_median < 0.95:
            print(f"• Consider filtering MZ twin calls by Relatedness > 0.95")
            print(f"  (Current median Relatedness: {rel_median:.4f})")
    
    print("• Review VCFped parameters and increase stringency for MZ twin calls")
    print("• Check for duplicate samples in your VCF file")
    print("• Use --keep-mz-separate flag to visualize all individuals separately")
    print()
    
    # Offer to create filtered pair file
    if mz_threshold and 'Relatedness' in mz_df.columns:
        print(f"\nCreating filtered pair file with Relatedness > {mz_threshold}...")
        filtered_df = pair_df.copy()
        mz_mask = (pair_df['Verdict'] == 'MZ twins') & (pair_df['Relatedness'] < mz_threshold)
        filtered_df.loc[mz_mask, 'Verdict'] = 'Unrelated'
        
        output_file = f"{prefix}_filtered.pair"
        filtered_df.to_csv(output_file, sep='\t', index=False)
        
        n_filtered = mz_mask.sum()
        print(f"✓ Filtered {n_filtered} MZ twin calls")
        print(f"✓ Saved to: {output_file}")
        print()
        print("You can now use this filtered file by replacing the original .pair file")
    
    print("="*80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VCFped pair file to diagnose MZ twin issues',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'prefix',
        help='VCFped output prefix (e.g., "results_named" for results_named.pair)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Relatedness threshold for filtering MZ calls (e.g., 0.95)'
    )
    
    args = parser.parse_args()
    
    return analyze_pair_file(args.prefix, args.threshold)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
