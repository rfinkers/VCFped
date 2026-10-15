#!/usr/bin/env python
"""
Generate interactive pedigree visualizations from VCFped output.

Creates beautiful, interactive pedigree plots using Plotly from VCFped results.
Supports multiple families, MZ twins, and complex relationships.

Usage:
    python plot_vcfped_pedigree.py results_named
    python plot_vcfped_pedigree.py results_named --output family_tree.html
    
Requirements:
    pip install plotly pandas
"""

import sys
import argparse
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class PedigreeBuilder:
    """Build pedigree structure from VCFped output."""
    
    def __init__(self, collapse_mz_twins=True):
        self.individuals = {}  # name -> {gender, generation, parents, children}
        self.parent_child = []  # (parent, child) tuples
        self.mz_twins = []  # (twin1, twin2) tuples
        self.couples = []  # (parent1, parent2, child) tuples
        self.collapse_mz_twins = collapse_mz_twins
        self.mz_groups = {}  # twin_name -> group_name mapping
        self.mz_group_members = {}  # group_name -> [member1, member2, ...]
        
    def add_individual(self, name, gender=None):
        """Add an individual to the pedigree."""
        if name not in self.individuals:
            self.individuals[name] = {
                'gender': gender,
                'generation': None,
                'parents': [],
                'children': [],
                'position': None,
                'is_mz_group': False,
                'members': [name],
                'member_count': 1
            }
        elif gender and not self.individuals[name]['gender']:
            self.individuals[name]['gender'] = gender
    
    def add_parent_child(self, parent, child):
        """Add a parent-child relationship."""
        self.add_individual(parent)
        self.add_individual(child)
        
        if child not in self.individuals[parent]['children']:
            self.individuals[parent]['children'].append(child)
        if parent not in self.individuals[child]['parents']:
            self.individuals[child]['parents'].append(parent)
        
        if (parent, child) not in self.parent_child:
            self.parent_child.append((parent, child))
    
    def add_mz_twins(self, twin1, twin2):
        """Add MZ twin relationship."""
        self.add_individual(twin1)
        self.add_individual(twin2)
        if (twin1, twin2) not in self.mz_twins and (twin2, twin1) not in self.mz_twins:
            self.mz_twins.append((twin1, twin2))
    
    def collapse_mz_twin_groups(self):
        """
        Collapse MZ twin pairs into single nodes with _mz suffix.
        This reduces pedigree complexity by representing twins as one entity.
        """
        if not self.collapse_mz_twins or not self.mz_twins:
            return
        
        # Build twin groups (handle transitive relationships)
        twin_sets = []
        for twin1, twin2 in self.mz_twins:
            # Find existing group containing either twin
            found_group = None
            for group in twin_sets:
                if twin1 in group or twin2 in group:
                    found_group = group
                    break
            
            if found_group:
                found_group.add(twin1)
                found_group.add(twin2)
            else:
                twin_sets.append({twin1, twin2})
        
        # Create merged entities for each group
        for group_idx, twin_set in enumerate(twin_sets, 1):
            twin_list = sorted(twin_set)
            
            # Create group name
            group_name = f"MZ_Group_{group_idx}_mz"
            
            # Store mappings
            for twin in twin_list:
                self.mz_groups[twin] = group_name
            self.mz_group_members[group_name] = twin_list
            
            # Get info from first twin (they should all be the same)
            first_twin = twin_list[0]
            
            # Safety check: make sure first twin exists
            if first_twin not in self.individuals:
                continue
            
            # Create merged individual
            self.individuals[group_name] = {
                'gender': self.individuals[first_twin]['gender'],
                'generation': self.individuals[first_twin]['generation'],
                'parents': self.individuals[first_twin]['parents'].copy(),
                'children': [],
                'position': None,
                'is_mz_group': True,
                'members': twin_list,
                'member_count': len(twin_list)
            }
            
            # Merge children from all twins
            all_children = set()
            for twin in twin_list:
                if twin in self.individuals:
                    all_children.update(self.individuals[twin]['children'])
            self.individuals[group_name]['children'] = list(all_children)
        
        # Update parent-child relationships to use group names
        new_parent_child = []
        for parent, child in self.parent_child:
            parent_name = self.mz_groups.get(parent, parent)
            child_name = self.mz_groups.get(child, child)
            
            relationship = (parent_name, child_name)
            if relationship not in new_parent_child:
                new_parent_child.append(relationship)
        
        self.parent_child = new_parent_child
        
        # Update couples to use group names
        new_couples = []
        for p1, p2, child in self.couples:
            p1_name = self.mz_groups.get(p1, p1)
            p2_name = self.mz_groups.get(p2, p2)
            child_name = self.mz_groups.get(child, child)
            
            trio = (p1_name, p2_name, child_name)
            if trio not in new_couples:
                new_couples.append(trio)
        
        self.couples = new_couples
        
        # Update parents lists in all individuals to use group names
        for name, info in list(self.individuals.items()):
            if 'parents' in info:
                new_parents = [self.mz_groups.get(p, p) for p in info['parents']]
                info['parents'] = new_parents
        
        # Update children lists in parent individuals
        for name, info in list(self.individuals.items()):
            if name.endswith('_mz'):
                continue
            new_children = [self.mz_groups.get(c, c) for c in info['children']]
            info['children'] = list(set(new_children))
        
        # Remove original twin individuals (keep only group)
        for twin_name in self.mz_groups.keys():
            if twin_name in self.individuals:
                del self.individuals[twin_name]
    
    def add_trio(self, parent1, parent2, child):
        """Add a complete trio."""
        self.add_parent_child(parent1, child)
        self.add_parent_child(parent2, child)
        
        trio = tuple(sorted([parent1, parent2]) + [child])
        if trio not in self.couples:
            self.couples.append((parent1, parent2, child))
    
    def assign_generations(self):
        """Assign generation numbers to all individuals."""
        # Find individuals with no parents (generation 0)
        generation = 0
        assigned = set()
        
        # Start with founders (no parents)
        founders = [name for name, info in self.individuals.items() 
                   if not info['parents']]
        
        for founder in founders:
            self.individuals[founder]['generation'] = 0
            assigned.add(founder)
        
        # Assign generations iteratively
        changed = True
        while changed and generation < 10:  # Safety limit
            changed = False
            generation += 1
            
            for name, info in list(self.individuals.items()):
                if name in assigned:
                    continue
                
                # If all parents are assigned, assign this individual
                if info['parents']:
                    parent_gens = []
                    for p in info['parents']:
                        # Safety check: parent might have been collapsed
                        if p in self.individuals and self.individuals[p]['generation'] is not None:
                            parent_gens.append(self.individuals[p]['generation'])
                    
                    if len(parent_gens) == len(info['parents']):
                        info['generation'] = max(parent_gens) + 1
                        assigned.add(name)
                        changed = True
        
        # Assign remaining individuals (isolated nodes)
        for name, info in self.individuals.items():
            if info['generation'] is None:
                info['generation'] = 0
    
    def calculate_positions(self):
        """Calculate x,y positions for pedigree layout."""
        # First collapse MZ twins if enabled
        self.collapse_mz_twin_groups()
        
        # Then assign generations
        self.assign_generations()
        
        # Group by generation
        by_generation = defaultdict(list)
        for name, info in self.individuals.items():
            by_generation[info['generation']].append(name)
        
        # Calculate positions
        for gen, members in by_generation.items():
            n = len(members)
            for i, name in enumerate(sorted(members)):
                # Y position based on generation (invert so parents are on top)
                y = -gen
                
                # X position: spread individuals across the width
                if n == 1:
                    x = 0
                else:
                    x = (i - (n-1)/2) * 2  # Space individuals out
                
                self.individuals[name]['position'] = (x, y)
        
        # Adjust positions to keep families together
        self._adjust_family_positions()
    
    def _adjust_family_positions(self):
        """Adjust positions to keep parents centered above their children."""
        # For each couple, center them above their children
        for parent1, parent2, child in self.couples:
            # Safety check: ensure all individuals exist
            if (parent1 not in self.individuals or 
                parent2 not in self.individuals or 
                child not in self.individuals):
                continue
            
            # Safety check: ensure positions have been calculated
            if (self.individuals[child]['position'] is None or
                self.individuals[parent1]['position'] is None or
                self.individuals[parent2]['position'] is None):
                continue
            
            child_pos = self.individuals[child]['position']
            child_x, child_y = child_pos
            
            # Get all children of this couple
            p1_children = set(self.individuals[parent1]['children'])
            p2_children = set(self.individuals[parent2]['children'])
            shared_children = p1_children & p2_children
            
            if shared_children:
                # Calculate center of children (only those that exist)
                child_positions = []
                for c in shared_children:
                    if c in self.individuals and self.individuals[c]['position'] is not None:
                        child_positions.append(self.individuals[c]['position'][0])
                
                if not child_positions:
                    continue
                
                center_x = sum(child_positions) / len(child_positions)
                
                # Position parents centered above children
                parent_y = child_y + 1
                self.individuals[parent1]['position'] = (center_x - 0.5, parent_y)
                self.individuals[parent2]['position'] = (center_x + 0.5, parent_y)


def parse_vcfped_files(prefix, collapse_mz_twins=True):
    """
    Parse VCFped output files to build pedigree.
    
    Args:
        prefix: Path prefix to VCFped output files (without extension)
        collapse_mz_twins: If True, merge MZ twins into single nodes (default: True)
    
    Returns:
        PedigreeBuilder object with parsed relationships
    """
    builder = PedigreeBuilder(collapse_mz_twins=collapse_mz_twins)
    
    # Parse gender file
    try:
        gender_df = pd.read_csv(f"{prefix}.gender", sep='\t')
        for _, row in gender_df.iterrows():
            sample = str(row['Sample'])
            gender = row['Gender']
            if gender in ['Male', 'Female']:
                builder.add_individual(sample, gender)
    except FileNotFoundError:
        print("Warning: Gender file not found, genders will be inferred from relationships")
    except Exception as e:
        print(f"Warning: Could not parse gender file: {e}")
    
    # Parse pair file
    try:
        pair_df = pd.read_csv(f"{prefix}.pair", sep='\t')
        for _, row in pair_df.iterrows():
            pair = str(row['Pair']).split(',')
            if len(pair) != 2:
                continue
            
            sample1, sample2 = pair[0].strip(), pair[1].strip()
            verdict = row['Verdict']
            
            if verdict == 'MZ twins':
                builder.add_mz_twins(sample1, sample2)
            elif verdict == 'Parent-child':
                # We don't know who is parent and who is child yet
                # Will be resolved from trio file or generation analysis
                pass
    except FileNotFoundError:
        print("Warning: Pair file not found")
    except Exception as e:
        print(f"Warning: Could not parse pair file: {e}")
    
    # Parse trio file (most informative)
    try:
        trio_df = pd.read_csv(f"{prefix}.trio", sep='\t')
        for _, row in trio_df.iterrows():
            verdict = row['Verdict']
            if verdict != 'Regular trio':
                continue
            
            triple = str(row['Triple']).split(',')
            pivot = str(row['Pivot']).strip()
            
            if len(triple) != 3:
                continue
            
            triple = [s.strip() for s in triple]
            
            # Pivot is the child
            child = pivot
            parents = [s for s in triple if s != child]
            
            if len(parents) == 2:
                builder.add_trio(parents[0], parents[1], child)
    except FileNotFoundError:
        print("Warning: Trio file not found")
    except Exception as e:
        print(f"Warning: Could not parse trio file: {e}")
    
    # If no trios found, try to infer from pairs
    if not builder.couples:
        try:
            pair_df = pd.read_csv(f"{prefix}.pair", sep='\t')
            for _, row in pair_df.iterrows():
                if row['Verdict'] == 'Parent-child':
                    pair = str(row['Pair']).split(',')
                    if len(pair) == 2:
                        sample1, sample2 = pair[0].strip(), pair[1].strip()
                        # Try to guess who is parent based on gender and other relationships
                        # This is a simple heuristic
                        builder.add_parent_child(sample1, sample2)
        except:
            pass
    
    return builder


def create_pedigree_plot(builder, title="Family Pedigree"):
    """
    Create an interactive Plotly pedigree visualization.
    
    Args:
        builder: PedigreeBuilder object with pedigree data
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure
    """
    builder.calculate_positions()
    
    # Create figure
    fig = go.Figure()
    
    # Define colors and symbols
    gender_colors = {
        'Male': '#4A90E2',    # Blue
        'Female': '#E24A90',  # Pink
        None: '#95A5A6'       # Gray for unknown
    }
    
    gender_symbols = {
        'Male': 'square',
        'Female': 'circle',
        None: 'diamond'
    }
    
    # IMPORTANT: MZ twin connection lines are ONLY drawn if twins are NOT collapsed
    # When collapsed, twins no longer exist as separate individuals in builder.individuals
    # So we must check this BEFORE trying to access them anywhere in this function
    draw_twin_lines = not builder.collapse_mz_twins and builder.mz_twins
    
    # Draw MZ twin connection lines FIRST (only if not collapsed)
    if draw_twin_lines:
        for twin1, twin2 in builder.mz_twins:
            # Safety check: both twins must exist in individuals
            if twin1 in builder.individuals and twin2 in builder.individuals:
                if (builder.individuals[twin1]['position'] is not None and 
                    builder.individuals[twin2]['position'] is not None):
                    t1x, t1y = builder.individuals[twin1]['position']
                    t2x, t2y = builder.individuals[twin2]['position']
                    
                    fig.add_trace(go.Scatter(
                        x=[t1x, t2x],
                        y=[t1y, t2y],
                        mode='lines',
                        line=dict(color='#9B59B6', width=2, dash='dot'),
                        hoverinfo='none',
                        showlegend=False,
                        name='MZ twins'
                    ))
    
    # Parent-child edges
    for parent, child in builder.parent_child:
        # Check if both parent and child exist (they should after collapse)
        if parent not in builder.individuals or child not in builder.individuals:
            continue
        
        # Check positions exist
        if (builder.individuals[parent]['position'] is None or 
            builder.individuals[child]['position'] is None):
            continue
            
        px, py = builder.individuals[parent]['position']
        cx, cy = builder.individuals[child]['position']
        
        # Draw line from parent to child
        fig.add_trace(go.Scatter(
            x=[px, cx],
            y=[py, cy],
            mode='lines',
            line=dict(color='#34495E', width=2),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Couple connections (horizontal lines between parents)
    couples_drawn = set()
    for p1, p2, child in builder.couples:
        couple = tuple(sorted([p1, p2]))
        if couple in couples_drawn:
            continue
        couples_drawn.add(couple)
        
        # Check that both parents exist
        if p1 not in builder.individuals or p2 not in builder.individuals:
            continue
        
        # Check positions exist
        if (builder.individuals[p1]['position'] is None or 
            builder.individuals[p2]['position'] is None):
            continue
        
        p1x, p1y = builder.individuals[p1]['position']
        p2x, p2y = builder.individuals[p2]['position']
        
        # Horizontal line connecting parents
        fig.add_trace(go.Scatter(
            x=[p1x, p2x],
            y=[p1y, p2y],
            mode='lines',
            line=dict(color='#34495E', width=3),
            hoverinfo='none',
            showlegend=False
        ))
    
    # MZ twin connections (special dashed line)
    for twin1, twin2 in builder.mz_twins:
        t1x, t1y = builder.individuals[twin1]['position']
        t2x, t2y = builder.individuals[twin2]['position']
        
        fig.add_trace(go.Scatter(
            x=[t1x, t2x],
            y=[t1y, t2y],
            mode='lines',
            line=dict(color='#9B59B6', width=2, dash='dot'),
            hoverinfo='none',
            showlegend=False,
            name='MZ twins'
        ))
    
    # Add nodes (individuals) grouped by gender for legend
    for gender in ['Male', 'Female', None]:
        # Separate regular individuals and MZ groups
        regular_individuals = [
            (name, info) for name, info in builder.individuals.items()
            if info['gender'] == gender and not info['is_mz_group']
        ]
        
        mz_groups = [
            (name, info) for name, info in builder.individuals.items()
            if info['gender'] == gender and info['is_mz_group']
        ]
        
        # Plot regular individuals
        if regular_individuals:
            x_coords = []
            y_coords = []
            hover_texts = []
            
            for name, info in regular_individuals:
                x, y = info['position']
                x_coords.append(x)
                y_coords.append(y)
                
                # Create hover text
                hover_text = f"<b>{name}</b><br>"
                hover_text += f"Gender: {gender or 'Unknown'}<br>"
                hover_text += f"Generation: {info['generation']}<br>"
                if info['parents']:
                    hover_text += f"Parents: {', '.join(info['parents'])}<br>"
                if info['children']:
                    hover_text += f"Children: {', '.join(info['children'])}"
                
                hover_texts.append(hover_text)
            
            # Add trace for this gender
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=40,
                    color=gender_colors[gender],
                    symbol=gender_symbols[gender],
                    line=dict(color='#2C3E50', width=2)
                ),
                text=[name.split('_')[-1] if '_' in name and not name.endswith('_mz') else name 
                      for name, _ in regular_individuals],
                textposition='bottom center',
                textfont=dict(size=10, color='#2C3E50'),
                hovertext=hover_texts,
                hoverinfo='text',
                name=gender or 'Unknown',
                showlegend=True
            ))
        
        # Plot MZ groups with special styling
        if mz_groups:
            x_coords = []
            y_coords = []
            hover_texts = []
            
            for name, info in mz_groups:
                x, y = info['position']
                x_coords.append(x)
                y_coords.append(y)
                
                # Create hover text for MZ group
                members_str = ', '.join(info['members'])
                hover_text = f"<b>MZ Twins ({info['member_count']})</b><br>"
                hover_text += f"Members: {members_str}<br>"
                hover_text += f"Gender: {gender or 'Unknown'}<br>"
                hover_text += f"Generation: {info['generation']}<br>"
                if info['parents']:
                    hover_text += f"Parents: {', '.join(info['parents'])}<br>"
                if info['children']:
                    hover_text += f"Children: {', '.join(info['children'])}"
                
                hover_texts.append(hover_text)
            
            # Add trace for MZ groups with double outline
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=45,  # Slightly larger
                    color=gender_colors[gender],
                    symbol=gender_symbols[gender],
                    line=dict(color='#9B59B6', width=4)  # Purple double border
                ),
                text=[f"MZ×{info['member_count']}" for name, info in mz_groups],
                textposition='bottom center',
                textfont=dict(size=10, color='#2C3E50', family='Arial Black'),
                hovertext=hover_texts,
                hoverinfo='text',
                name=f"{gender or 'Unknown'} (MZ Twins)",
                showlegend=True,
                legendgroup=f'{gender}_mz'
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2C3E50'}
        },
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#2C3E50',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='#ECF0F1',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title='',
            autorange='reversed'  # Parents on top
        ),
        height=600,
        width=1000,
        margin=dict(l=50, r=200, t=80, b=50)
    )
    
    return fig


def create_summary_statistics(builder):
    """Create a summary statistics text."""
    n_individuals = len(builder.individuals)
    n_males = sum(1 for info in builder.individuals.values() if info['gender'] == 'Male')
    n_females = sum(1 for info in builder.individuals.values() if info['gender'] == 'Female')
    n_unknown = n_individuals - n_males - n_females
    n_trios = len(builder.couples)
    n_mz_groups = sum(1 for info in builder.individuals.values() if info.get('is_mz_group', False))
    n_mz_individuals = sum(info.get('member_count', 1) for info in builder.individuals.values() 
                           if info.get('is_mz_group', False))
    n_generations = len(set(info['generation'] for info in builder.individuals.values()))
    
    # Calculate actual number of biological individuals (including collapsed twins)
    actual_individuals = n_individuals - n_mz_groups + n_mz_individuals
    
    summary = f"""
    <div style='font-family: Arial, sans-serif; margin: 20px; padding: 20px; 
                background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #4A90E2;'>
        <h3 style='color: #2C3E50; margin-top: 0;'>Pedigree Summary</h3>
        <ul style='list-style: none; padding-left: 0;'>
            <li>👥 <b>Nodes displayed:</b> {n_individuals}</li>
            {f"<li>🧬 <b>Total biological individuals:</b> {actual_individuals} (includes {n_mz_individuals} in MZ groups)</li>" if n_mz_groups > 0 else ""}
            <li>👨 <b>Males:</b> {n_males}</li>
            <li>👩 <b>Females:</b> {n_females}</li>
            {f"<li>❓ <b>Unknown gender:</b> {n_unknown}</li>" if n_unknown > 0 else ""}
            <li>👨‍👩‍👦 <b>Trios identified:</b> {n_trios}</li>
            {f"<li>👯 <b>MZ twin groups:</b> {n_mz_groups} (collapsed for clarity)</li>" if n_mz_groups > 0 else ""}
            <li>🔢 <b>Generations:</b> {n_generations}</li>
        </ul>
        {f"<p style='color: #7f8c8d; font-size: 0.9em; margin-top: 15px;'><i>Note: MZ twins are shown as single nodes with purple borders and labeled 'MZ×N' where N is the number of twins.</i></p>" if n_mz_groups > 0 else ""}
    </div>
    """
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive pedigree visualization from VCFped output',
        epilog='''
Examples:
  %(prog)s results_named
  %(prog)s results_named --output family_tree.html
  %(prog)s results_named --title "Smith Family Pedigree"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'prefix',
        help='VCFped output prefix (e.g., "results_named" for results_named.trio)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output HTML file (default: <prefix>_pedigree.html)'
    )
    
    parser.add_argument(
        '-t', '--title',
        default='Family Pedigree',
        help='Plot title (default: "Family Pedigree")'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    parser.add_argument(
        '--keep-mz-separate',
        action='store_true',
        help='Show MZ twins as separate individuals instead of collapsing them into one node'
    )
    
    args = parser.parse_args()
    
    # Set output file
    if args.output is None:
        args.output = f"{args.prefix}_pedigree.html"
    
    print("="*70)
    print("VCFped Pedigree Visualization")
    print("="*70)
    print(f"Input prefix: {args.prefix}")
    print(f"Output file: {args.output}")
    print()
    
    # Parse VCFped files
    print("Parsing VCFped output files...")
    collapse_mz = not args.keep_mz_separate
    builder = parse_vcfped_files(args.prefix, collapse_mz_twins=collapse_mz)
    
    if not builder.individuals:
        print("Error: No individuals found in VCFped output files!")
        print("Make sure the following files exist:")
        print(f"  - {args.prefix}.gender (optional)")
        print(f"  - {args.prefix}.pair (optional)")
        print(f"  - {args.prefix}.trio (recommended)")
        return 1
    
    print(f"✓ Found {len(builder.individuals)} individual(s)")
    print(f"✓ Found {len(builder.couples)} trio(s)")
    if builder.mz_twins:
        n_mz_groups = sum(1 for info in builder.individuals.values() if info.get('is_mz_group', False))
        if collapse_mz and n_mz_groups > 0:
            print(f"✓ Found {len(builder.mz_twins)} MZ twin pair(s) (collapsed into {n_mz_groups} group(s))")
        else:
            print(f"✓ Found {len(builder.mz_twins)} MZ twin pair(s)")
    print()
    
    # Create visualization
    print("Creating pedigree visualization...")
    fig = create_pedigree_plot(builder, title=args.title)
    
    # Get summary statistics
    summary = create_summary_statistics(builder)
    
    # Save to HTML
    html_content = fig.to_html(include_plotlyjs='cdn')
    
    # Add summary statistics to HTML
    html_with_summary = html_content.replace(
        '</body>',
        f'{summary}</body>'
    )
    
    with open(args.output, 'w') as f:
        f.write(html_with_summary)
    
    print(f"✓ Saved visualization to: {args.output}")
    print()
    
    # Print summary
    print("Summary:")
    print(f"  • {len(builder.individuals)} individuals")
    print(f"  • {len(builder.couples)} complete trios")
    print(f"  • {len(builder.parent_child)} parent-child relationships")
    if builder.mz_twins:
        print(f"  • {len(builder.mz_twins)} MZ twin pairs")
    
    print()
    print("="*70)
    print("✓ Done!")
    
    if not args.no_browser:
        import webbrowser
        import os
        webbrowser.open('file://' + os.path.abspath(args.output))
        print(f"Opening {args.output} in browser...")
    
    print("="*70)
    
    return 0


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
