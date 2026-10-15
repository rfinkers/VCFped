#!/usr/bin/env python
"""
Debug version to identify KeyError location.

Usage:
    python debug_vcfped_pedigree.py results_named
"""

import sys
import traceback
import os

def debug_pedigree(prefix):
    """Run pedigree visualization with detailed debugging."""
    
    print("="*70)
    print("DEBUG MODE: VCFped Pedigree Visualization")
    print("="*70)
    print(f"Input prefix: {prefix}\n")
    
    # Check if main script exists
    if not os.path.exists('plot_vcfped_pedigree.py'):
        print("✗ ERROR: plot_vcfped_pedigree.py not found in current directory!")
        print("\nPlease ensure plot_vcfped_pedigree.py is in the same directory as this debug script.")
        return 1
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    try:
        print("Importing plot_vcfped_pedigree module...")
        import plot_vcfped_pedigree
        print("✓ Module imported successfully\n")
        
        print("Step 1: Parsing VCFped files...")
        try:
            builder = plot_vcfped_pedigree.parse_vcfped_files(prefix, collapse_mz_twins=True)
            print(f"✓ Parsed successfully")
            print(f"  Individuals after parsing: {len(builder.individuals)}")
            print(f"  Individual names: {list(builder.individuals.keys())[:10]}")  # First 10
            if len(builder.individuals) > 10:
                print(f"  ... and {len(builder.individuals) - 10} more")
            print(f"  MZ twins found: {len(builder.mz_twins)}")
            if builder.mz_twins:
                print(f"  MZ twin pairs: {builder.mz_twins[:3]}")  # First 3
            print()
        except Exception as e:
            print(f"✗ Error in parse_vcfped_files(): {e}")
            traceback.print_exc()
            return 1
        
        print("Step 2: Checking MZ groups after parsing...")
        print(f"  MZ groups mapping: {builder.mz_groups}")
        print(f"  Collapse enabled: {builder.collapse_mz_twins}")
        print()
        
        print("Step 3: Calculating positions (includes collapse)...")
        try:
            builder.calculate_positions()
            print(f"✓ Positions calculated")
            print(f"  Individuals after collapse: {len(builder.individuals)}")
            print(f"  Individual names: {list(builder.individuals.keys())[:10]}")
            if len(builder.individuals) > 10:
                print(f"  ... and {len(builder.individuals) - 10} more")
            print()
            
            # Check generations
            print("  Generations assigned:")
            gen_counts = {}
            for name, info in builder.individuals.items():
                gen = info.get('generation', 'None')
                gen_counts[gen] = gen_counts.get(gen, 0) + 1
            for gen, count in sorted(gen_counts.items()):
                print(f"    Generation {gen}: {count} individuals")
            print()
            
        except Exception as e:
            print(f"✗ Error in calculate_positions(): {e}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            
            # Try to identify the exact problem
            print("\nDiagnostic information:")
            print(f"  Current individuals: {list(builder.individuals.keys())[:10]}")
            print(f"  Parent-child relationships (first 5):")
            for parent, child in builder.parent_child[:5]:
                p_exists = parent in builder.individuals
                c_exists = child in builder.individuals
                print(f"    {parent} → {child} (parent exists: {p_exists}, child exists: {c_exists})")
            return 1
        
        print("Step 4: Checking relationships...")
        print("  4a. Parent-child (first 5):")
        for parent, child in builder.parent_child[:5]:
            parent_exists = parent in builder.individuals
            child_exists = child in builder.individuals
            status = "✓" if (parent_exists and child_exists) else "✗"
            print(f"    {status} {parent} → {child}")
            if not parent_exists:
                print(f"      WARNING: Parent '{parent}' not in individuals!")
            if not child_exists:
                print(f"      WARNING: Child '{child}' not in individuals!")
        if len(builder.parent_child) > 5:
            print(f"    ... and {len(builder.parent_child) - 5} more")
        print()
        
        print("  4b. Couples (first 5):")
        for p1, p2, child in builder.couples[:5]:
            p1_exists = p1 in builder.individuals
            p2_exists = p2 in builder.individuals
            child_exists = child in builder.individuals
            status = "✓" if (p1_exists and p2_exists and child_exists) else "✗"
            print(f"    {status} {p1}, {p2} → {child}")
            if not p1_exists:
                print(f"      WARNING: Parent '{p1}' not in individuals!")
            if not p2_exists:
                print(f"      WARNING: Parent '{p2}' not in individuals!")
            if not child_exists:
                print(f"      WARNING: Child '{child}' not in individuals!")
        if len(builder.couples) > 5:
            print(f"    ... and {len(builder.couples) - 5} more")
        print()
        
        print("Step 5: Creating plot...")
        try:
            fig = plot_vcfped_pedigree.create_pedigree_plot(builder, "Debug Test")
            print(f"✓ Plot created successfully!")
            print()
        except KeyError as e:
            print(f"✗ KeyError in create_pedigree_plot(): {e}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            print(f"\nThe KeyError is trying to access: {e}")
            print(f"Available individuals: {list(builder.individuals.keys())[:20]}")
            return 1
        except Exception as e:
            print(f"✗ Error in create_pedigree_plot(): {e}")
            traceback.print_exc()
            return 1
        
        print("Step 6: Saving HTML...")
        try:
            output = f"{prefix}_debug.html"
            html = fig.to_html(include_plotlyjs='cdn')
            with open(output, 'w') as f:
                f.write(html)
            print(f"✓ Saved to {output}")
            print()
        except Exception as e:
            print(f"✗ Error saving HTML: {e}")
            traceback.print_exc()
            return 1
        
        print("="*70)
        print("✓ SUCCESS! No errors detected.")
        print("="*70)
        print(f"\nVisualization saved to: {prefix}_debug.html")
        return 0
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure plot_vcfped_pedigree.py is in the current directory.")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_vcfped_pedigree.py <prefix>")
        print("\nExample:")
        print("  python debug_vcfped_pedigree.py results_named")
        print("\nThis will help identify where the KeyError occurs.")
        sys.exit(1)
    
    sys.exit(debug_pedigree(sys.argv[1]))
