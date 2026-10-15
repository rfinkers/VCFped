#!/usr/bin/env python
"""
Script to replace sample numbers with actual sample names in VCFped output files.

Reads the sample name mapping from the .log file and updates .gender, .pair, 
and .trio files with actual sample names instead of numbers.

Usage:
    python rename_vcfped_samples.py output_prefix
    
Example:
    python rename_vcfped_samples.py results
    
This will process:
    - results.log (read sample names)
    - results.gender → results_named.gender
    - results.pair → results_named.pair
    - results.trio → results_named.trio
"""

import sys
import os
import re
import argparse


def parse_sample_names(log_file):
    """
    Parse sample names from VCFped log file.
    
    Looks for section like:
        Sample names:
          1: sample_name_1
          2: sample_name_2
          3: sample_name_3
    
    Returns:
        dict: {1: 'sample_name_1', 2: 'sample_name_2', ...}
    """
    sample_map = {}
    
    with open(log_file, 'r') as f:
        in_sample_section = False
        
        for line in f:
            line = line.rstrip()
            
            # Check if we're entering the sample names section
            if 'Sample names:' in line:
                in_sample_section = True
                continue
            
            # If we're in the section, parse sample lines
            if in_sample_section:
                # Sample lines look like: "  1: sample_name"
                match = re.match(r'\s+(\d+):\s+(.+)', line)
                if match:
                    sample_num = int(match.group(1))
                    sample_name = match.group(2).strip()
                    sample_map[sample_num] = sample_name
                elif line.strip() and not line.startswith(' '):
                    # We've left the sample section (non-indented line)
                    break
    
    if not sample_map:
        raise ValueError("Could not find sample names in log file")
    
    return sample_map


def rename_gender_file(input_file, output_file, sample_map):
    """
    Rename sample numbers in gender file.
    
    Input format:
        Sample  Perc  Xvars  AA  AB  BB  Xhet  Gender
        1       50    1234   ...
    
    Output format:
        Sample          Perc  Xvars  AA  AB  BB  Xhet  Gender
        sample_name_1   50    1234   ...
    """
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            if line_num == 0:
                # Header line - keep as is
                f_out.write(line)
            else:
                # Data line
                fields = line.strip().split('\t')
                if fields:
                    try:
                        sample_num = int(fields[0])
                        if sample_num in sample_map:
                            fields[0] = sample_map[sample_num]
                        else:
                            print(f"  Warning: Sample number {sample_num} not in mapping")
                    except ValueError:
                        # Not a number, keep as is
                        pass
                    
                    f_out.write('\t'.join(fields) + '\n')
    
    print(f"  ✓ Created {output_file}")


def rename_pair_file(input_file, output_file, sample_map):
    """
    Rename sample numbers in pairwise file.
    
    Input format:
        Pair   Perc  Autos  ...
        1,2    50    45234  ...
    
    Output format:
        Pair                    Perc  Autos  ...
        sample_name_1,sample_name_2  50    45234  ...
    """
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            if line_num == 0:
                # Header line - keep as is
                f_out.write(line)
            else:
                # Data line
                fields = line.strip().split('\t')
                if fields:
                    # Parse pair like "1,2"
                    pair_str = fields[0]
                    match = re.match(r'(\d+),(\d+)', pair_str)
                    if match:
                        num1 = int(match.group(1))
                        num2 = int(match.group(2))
                        
                        if num1 in sample_map and num2 in sample_map:
                            fields[0] = f"{sample_map[num1]},{sample_map[num2]}"
                        else:
                            print(f"  Warning: Pair {pair_str} has unmapped sample(s)")
                    
                    f_out.write('\t'.join(fields) + '\n')
    
    print(f"  ✓ Created {output_file}")


def rename_trio_file(input_file, output_file, sample_map):
    """
    Rename sample numbers in trio file.
    
    Input format:
        Triple  Pivot  Perc  Autos  ...
        1,2,3   3      50    45234  ...
    
    Output format:
        Triple                           Pivot         Perc  Autos  ...
        sample_name_1,sample_name_2,sample_name_3  sample_name_3  50    45234  ...
    """
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            if line_num == 0:
                # Header line - keep as is
                f_out.write(line)
            else:
                # Data line
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    # Parse triple like "1,2,3"
                    triple_str = fields[0]
                    match = re.match(r'(\d+),(\d+),(\d+)', triple_str)
                    if match:
                        num1 = int(match.group(1))
                        num2 = int(match.group(2))
                        num3 = int(match.group(3))
                        
                        if num1 in sample_map and num2 in sample_map and num3 in sample_map:
                            fields[0] = f"{sample_map[num1]},{sample_map[num2]},{sample_map[num3]}"
                        else:
                            print(f"  Warning: Triple {triple_str} has unmapped sample(s)")
                    
                    # Parse pivot (single number)
                    pivot_str = fields[1]
                    try:
                        pivot_num = int(pivot_str)
                        if pivot_num in sample_map:
                            fields[1] = sample_map[pivot_num]
                        else:
                            print(f"  Warning: Pivot {pivot_num} not in mapping")
                    except ValueError:
                        # Not a number, keep as is
                        pass
                    
                    f_out.write('\t'.join(fields) + '\n')
    
    print(f"  ✓ Created {output_file}")


def process_vcfped_output(prefix, output_suffix='_named'):
    """
    Process all VCFped output files for a given prefix.
    
    Args:
        prefix: Output prefix used with VCFped (e.g., 'results')
        output_suffix: Suffix to add to output files (e.g., '_named')
    """
    log_file = f"{prefix}.log"
    
    # Check if log file exists
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    print("="*70)
    print("VCFped Sample Name Renamer")
    print("="*70)
    print(f"Reading sample names from: {log_file}")
    
    # Parse sample names
    sample_map = parse_sample_names(log_file)
    
    print(f"\nFound {len(sample_map)} samples:")
    for num, name in sorted(sample_map.items()):
        print(f"  {num} → {name}")
    
    print("\n" + "-"*70)
    
    # Process each output file
    files_to_process = [
        ('gender', rename_gender_file),
        ('pair', rename_pair_file),
        ('trio', rename_trio_file)
    ]
    
    processed_count = 0
    
    for file_type, rename_func in files_to_process:
        input_file = f"{prefix}.{file_type}"
        output_file = f"{prefix}{output_suffix}.{file_type}"
        
        if os.path.exists(input_file):
            try:
                rename_func(input_file, output_file, sample_map)
                processed_count += 1
            except Exception as e:
                print(f"  ✗ Error processing {input_file}: {e}")
        else:
            print(f"Skipping {input_file} (file not found)")
    
    print("-"*70)
    print(f"\n✓ Successfully processed {processed_count} file(s)")
    print(f"\nOutput files created with suffix '{output_suffix}':")
    for file_type, _ in files_to_process:
        output_file = f"{prefix}{output_suffix}.{file_type}"
        if os.path.exists(output_file):
            print(f"  • {output_file}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Replace sample numbers with names in VCFped output files',
        epilog='''
Examples:
  %(prog)s results
  %(prog)s results --suffix _with_names
  %(prog)s /path/to/analysis/output
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'prefix',
        help='VCFped output prefix (without file extension)'
    )
    
    parser.add_argument(
        '-s', '--suffix',
        default='_named',
        help='Suffix to add to output files (default: _named)'
    )
    
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Overwrite original files instead of creating new ones (use with caution!)'
    )
    
    args = parser.parse_args()
    
    # If inplace, use empty suffix and backup originals
    if args.inplace:
        print("WARNING: --inplace mode will overwrite original files!")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        
        # Backup originals
        for ext in ['gender', 'pair', 'trio']:
            original = f"{args.prefix}.{ext}"
            backup = f"{args.prefix}.{ext}.backup"
            if os.path.exists(original):
                import shutil
                shutil.copy2(original, backup)
                print(f"Backed up {original} → {backup}")
        
        # Process with empty suffix (will overwrite)
        output_suffix = '.tmp'
        process_vcfped_output(args.prefix, output_suffix)
        
        # Move temp files to original names
        for ext in ['gender', 'pair', 'trio']:
            temp_file = f"{args.prefix}{output_suffix}.{ext}"
            original = f"{args.prefix}.{ext}"
            if os.path.exists(temp_file):
                os.replace(temp_file, original)
                print(f"Updated {original}")
        
        print("\nOriginal files backed up with .backup extension")
    else:
        process_vcfped_output(args.prefix, args.suffix)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
