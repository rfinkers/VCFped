#!/usr/bin/env python
"""
Generate interactive pedigree visualizations from VCFped output.

Creates an HTML report with:
  - Summary statistics (families, founders, generations, sibling groups)
  - Sortable parent1/parent2/child table
  - Per-family pedigree diagrams (top families by size)

Usage:
    python fixed_pedigree_viz.py results_named
    python fixed_pedigree_viz.py results_named -o family_tree.html
    python fixed_pedigree_viz.py results_named --keep-mz-separate --debug

Requirements:
    pip install plotly pandas
"""

import sys
import argparse
import json
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# Required columns per file type
_REQUIRED_COLS = {
    'gender': {'Sample', 'Gender'},
    'pair':   {'Pair', 'Verdict'},
    'trio':   {'Triple', 'Pivot', 'Verdict'},
}


def _validate_vcfped_file(path, filetype):
    """
    Read a vcfped output file and validate its header.
    Returns the DataFrame on success.
    Raises SystemExit with a clear message on missing file or wrong columns.
    The .gender file is optional — returns None with a warning when absent.
    """
    required = _REQUIRED_COLS[filetype]
    optional = filetype in ('gender', 'pair')
    try:
        df = pd.read_csv(path, sep='\t')
    except FileNotFoundError:
        if optional:
            msg = {'gender': 'gender information will be skipped',
                   'pair': 'MZ twin and pairwise info will be skipped (normal for --no-pairwise runs)'}
            print(f"Warning: {path} not found — {msg.get(filetype, 'file will be skipped')}.")
            return None
        sys.exit(f"Error: {path} not found.\n"
                 f"Make sure you are using the correct output prefix and that vcfped has been run.")
    except Exception as e:
        sys.exit(f"Error: Could not read {path}: {e}")

    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"Error: {path} is missing required columns: {sorted(missing)}\n"
            f"  Found columns: {list(df.columns)}\n"
            f"  This file does not look like a vcfped .{filetype} output.\n"
            f"  Check that you passed the correct prefix and that the file was produced by vcfped."
        )
    return df


# ---------------------------------------------------------------------------
# Pedigree data model
# ---------------------------------------------------------------------------

class PedigreeBuilder:
    """Build pedigree structure from VCFped output."""

    def __init__(self, collapse_mz_twins=True, debug=False):
        self.individuals = {}        # name -> dict(gender, generation, parents, children, ...)
        self.parent_child = []       # (parent, child) tuples
        self.mz_twins = []           # (twin1, twin2) tuples
        self.couples = []            # (parent1, parent2, child) tuples
        self.collapse_mz_twins = collapse_mz_twins
        self.mz_groups = {}          # twin_name -> group_name
        self.mz_group_members = {}   # group_name -> [members]
        self.debug = debug
        # Raw trio records including inverted trios
        self.inverted_trios = []     # (parent1, parent2, child) for inverted/generational trios

    def add_individual(self, name, gender=None):
        if name not in self.individuals:
            self.individuals[name] = {
                'gender': gender,
                'generation': None,
                'parents': [],
                'children': [],
                'position': None,
                'is_mz_group': False,
                'members': [name],
                'member_count': 1,
            }
        elif gender and not self.individuals[name]['gender']:
            self.individuals[name]['gender'] = gender

    def add_parent_child(self, parent, child):
        self.add_individual(parent)
        self.add_individual(child)
        if child not in self.individuals[parent]['children']:
            self.individuals[parent]['children'].append(child)
        if parent not in self.individuals[child]['parents']:
            self.individuals[child]['parents'].append(parent)
        if (parent, child) not in self.parent_child:
            self.parent_child.append((parent, child))

    def add_mz_twins(self, twin1, twin2):
        self.add_individual(twin1)
        self.add_individual(twin2)
        if (twin1, twin2) not in self.mz_twins and (twin2, twin1) not in self.mz_twins:
            self.mz_twins.append((twin1, twin2))

    def add_trio(self, parent1, parent2, child):
        self.add_parent_child(parent1, child)
        self.add_parent_child(parent2, child)
        trio = (parent1, parent2, child)
        if trio not in self.couples:
            self.couples.append(trio)

    def add_inverted_trio(self, parent1, parent2, child):
        """Record an inverted/generational trio (not added to the relationship graph)."""
        self.inverted_trios.append((parent1, parent2, child))

    def collapse_mz_twin_groups(self):
        if not self.collapse_mz_twins or not self.mz_twins:
            return
        # Build transitive twin groups
        twin_sets = []
        for twin1, twin2 in self.mz_twins:
            groups_to_merge, remaining = [], []
            for group in twin_sets:
                (groups_to_merge if twin1 in group or twin2 in group else remaining).append(group)
            merged = set()
            for g in groups_to_merge:
                merged.update(g)
            merged.update([twin1, twin2])
            remaining.append(merged)
            twin_sets = remaining

        all_twins = set()
        for ts in twin_sets:
            all_twins.update(ts)

        for idx, twin_set in enumerate(twin_sets, 1):
            twin_list = sorted(twin_set)
            group_name = f"MZ_Group_{idx}_mz"
            for twin in twin_list:
                self.mz_groups[twin] = group_name
            self.mz_group_members[group_name] = twin_list
            first = twin_list[0]
            if first not in self.individuals:
                continue
            all_children = set()
            for twin in twin_list:
                if twin in self.individuals:
                    all_children.update(self.individuals[twin]['children'])
            self.individuals[group_name] = {
                'gender': self.individuals[first]['gender'],
                'generation': self.individuals[first]['generation'],
                'parents': self.individuals[first]['parents'].copy(),
                'children': list(all_children),
                'position': None,
                'is_mz_group': True,
                'members': twin_list,
                'member_count': len(twin_list),
            }

        def remap(name):
            return self.mz_groups.get(name, name)

        self.parent_child = list({(remap(p), remap(c)) for p, c in self.parent_child})
        self.couples = list({(remap(p1), remap(p2), remap(c)) for p1, p2, c in self.couples})
        for info in self.individuals.values():
            info['parents'] = [remap(p) for p in info['parents']]
            info['children'] = list(set(remap(c) for c in info['children']))
        for twin in all_twins:
            self.individuals.pop(twin, None)

    def assign_generations(self):
        assigned = {}
        founders = [n for n, info in self.individuals.items() if not info['parents']]
        for f in founders:
            assigned[f] = 0
        changed = True
        while changed:
            changed = False
            for name, info in self.individuals.items():
                if name in assigned:
                    continue
                parent_gens = [assigned[p] for p in info['parents'] if p in assigned]
                if len(parent_gens) == len(info['parents']) and parent_gens:
                    assigned[name] = max(parent_gens) + 1
                    changed = True
        for name in self.individuals:
            self.individuals[name]['generation'] = assigned.get(name, 0)

    def detect_families(self):
        """Return list of frozensets, each a connected component in the pedigree graph."""
        parent_map = defaultdict(set)
        for p, c in self.parent_child:
            parent_map[p].add(c)
            parent_map[c].add(p)
        for p1, p2, c in self.couples:
            parent_map[p1].update([p2, c])
            parent_map[p2].update([p1, c])
            parent_map[c].update([p1, p2])

        visited = set()
        families = []
        for start in self.individuals:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(parent_map[node] - visited)
            families.append(frozenset(component))
        return sorted(families, key=len, reverse=True)

    def compute_statistics(self, trio_df=None):
        """Return a dict of summary statistics."""
        self.collapse_mz_twin_groups()
        self.assign_generations()
        families = self.detect_families()

        n = len(self.individuals)
        founders = [nm for nm, info in self.individuals.items() if not info['parents']]
        leaves = [nm for nm, info in self.individuals.items() if not info['children']]

        # Sibling groups
        sibling_map = defaultdict(set)
        for p1, p2, c in self.couples:
            key = tuple(sorted([p1, p2]))
            sibling_map[key].add(c)
        full_sibling_groups = [sorted(s) for s in sibling_map.values() if len(s) > 1]

        # Half-siblings: share exactly one parent
        child_parents = defaultdict(set)
        for p, c in self.parent_child:
            child_parents[c].add(p)
        shared_parent = defaultdict(set)
        for c, parents in child_parents.items():
            for p in parents:
                shared_parent[p].add(c)
        half_sib_pairs = set()
        for p, children in shared_parent.items():
            children = list(children)
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    a, b = children[i], children[j]
                    shared = child_parents[a] & child_parents[b]
                    if len(shared) == 1:
                        half_sib_pairs.add(tuple(sorted([a, b])))

        gen_counts = defaultdict(int)
        for info in self.individuals.values():
            gen_counts[info['generation']] += 1
        max_gen = max(gen_counts.keys()) if gen_counts else 0

        mz_groups = sum(1 for info in self.individuals.values() if info.get('is_mz_group'))
        n_males = sum(1 for info in self.individuals.values() if info['gender'] == 'Male')
        n_females = sum(1 for info in self.individuals.values() if info['gender'] == 'Female')

        inverted_count = len(self.inverted_trios)

        family_sizes = sorted([len(f) for f in families], reverse=True)

        return {
            'n_individuals': n,
            'n_families': len(families),
            'n_founders': len(founders),
            'n_leaves': len(leaves),
            'n_trios': len(self.couples),
            'n_parent_child': len(self.parent_child),
            'n_full_sibling_groups': len(full_sibling_groups),
            'n_half_sib_pairs': len(half_sib_pairs),
            'max_generation': max_gen,
            'generation_counts': dict(gen_counts),
            'n_mz_groups': mz_groups,
            'n_males': n_males,
            'n_females': n_females,
            'n_inverted_trios': inverted_count,
            'family_sizes': family_sizes,
            'families': families,
            'full_sibling_groups': full_sibling_groups,
        }


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_vcfped_files(prefix, collapse_mz_twins=True, debug=False):
    builder = PedigreeBuilder(collapse_mz_twins=collapse_mz_twins, debug=debug)

    # Gender (optional)
    gender_df = _validate_vcfped_file(f"{prefix}.gender", 'gender')
    if gender_df is not None:
        for _, row in gender_df.iterrows():
            gender = row['Gender']
            if gender in ('Male', 'Female'):
                builder.add_individual(str(row['Sample']), gender)

    # Pair file (optional — absent when vcfped was run with --no-pairwise)
    # MZ twins were already removed before step 3, so this file is not needed for the pedigree
    pair_df = _validate_vcfped_file(f"{prefix}.pair", 'pair')
    if pair_df is not None:
        for _, row in pair_df.iterrows():
            parts = str(row['Pair']).split(',')
            if len(parts) != 2:
                continue
            s1, s2 = parts[0].strip(), parts[1].strip()
            if row['Verdict'] == 'MZ twins':
                builder.add_mz_twins(s1, s2)

    # Trio file (required)
    trio_df = _validate_vcfped_file(f"{prefix}.trio", 'trio')
    for _, row in trio_df.iterrows():
        verdict = row['Verdict']
        triple = str(row['Triple']).split(',')
        pivot = str(row['Pivot']).strip()
        if len(triple) != 3:
            continue
        triple = [s.strip() for s in triple]
        child = pivot
        parents = [s for s in triple if s != child]
        if len(parents) != 2:
            continue
        if verdict == 'Regular trio':
            builder.add_trio(parents[0], parents[1], child)
        elif 'Inverted' in verdict or 'generational' in verdict.lower():
            builder.add_inverted_trio(parents[0], parents[1], child)

    # If no complete trios found and pair file is available, fall back to parent-child pairs
    if not builder.couples and pair_df is not None:
        for _, row in pair_df.iterrows():
            if row['Verdict'] == 'Parent-child':
                parts = str(row['Pair']).split(',')
                if len(parts) == 2:
                    builder.add_parent_child(parts[0].strip(), parts[1].strip())

    return builder, trio_df


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _layout_family(members, individuals, couples):
    """
    Compute (x, y) positions for one family using a simple hierarchical layout.
    Returns dict: name -> (x, y)
    """
    # Assign generation y-coordinates
    gen = {m: individuals[m]['generation'] for m in members}
    by_gen = defaultdict(list)
    for m in members:
        by_gen[gen[m]].append(m)

    # Initial x: equally spaced within each generation
    pos = {}
    for g, nodes in by_gen.items():
        nodes_sorted = sorted(nodes)
        for i, n in enumerate(nodes_sorted):
            pos[n] = [i * 2.0, -g]

    # Barycenter iterations: pull each node toward mean position of neighbours.
    # 15 (not 5) iterations -- larger families need more passes to settle into
    # a reasonably untangled arrangement before the fixed +2.0 re-spread below
    # locks generation order back in.
    child_to_parents = defaultdict(list)
    parent_to_children = defaultdict(list)
    for p1, p2, c in couples:
        if p1 in members and p2 in members and c in members:
            child_to_parents[c].extend([p1, p2])
            parent_to_children[p1].append(c)
            parent_to_children[p2].append(c)

    for _ in range(15):
        # Pull children toward mean of parents
        for node in members:
            neighbours = child_to_parents.get(node, []) + parent_to_children.get(node, [])
            neighbours = [n for n in neighbours if n in pos]
            if neighbours:
                mean_x = sum(pos[n][0] for n in neighbours) / len(neighbours)
                pos[node][0] = 0.5 * pos[node][0] + 0.5 * mean_x
        # Re-spread within each generation to avoid overlap
        for g, nodes in by_gen.items():
            nodes_by_x = sorted(nodes, key=lambda n: pos[n][0])
            for i, n in enumerate(nodes_by_x):
                if i > 0:
                    prev = nodes_by_x[i - 1]
                    if pos[n][0] < pos[prev][0] + 2.0:
                        pos[n][0] = pos[prev][0] + 2.0

    # Center the family around x=0
    all_x = [pos[n][0] for n in members]
    mid = (min(all_x) + max(all_x)) / 2
    for n in members:
        pos[n][0] -= mid

    return {n: tuple(pos[n]) for n in members}


# Families at or below this size still get on-plot text labels; above it,
# labels are hover-only -- crammed small-font text on a dense diagram is
# noise, not information, past this scale.
LABEL_THRESHOLD = 40


def _family_summary_card(members, individuals, label, max_diagram_size):
    """
    Compact stats-only card for a family too large to render as a readable
    node-link diagram (no layout algorithm makes hundreds/thousands of nodes
    legible as one picture). Full per-individual detail is still available in
    the searchable table above -- this just skips wasting a render on
    something no one could read anyway.
    """
    known = [m for m in members if m in individuals]
    gen_counts = Counter(individuals[m]['generation'] for m in known)
    founders = sum(1 for m in known if not individuals[m]['parents'])
    leaves = sum(1 for m in known if not individuals[m]['children'])
    n_gen = (max(gen_counts) - min(gen_counts) + 1) if gen_counts else 0
    gen_bar = "".join(f"<span>G{g}: {c}</span>" for g, c in sorted(gen_counts.items()))
    return (
        f"<h3>{label}</h3>"
        f"<div class='info'>Too large to render as a diagram ({len(members)} individuals "
        f"exceeds the {max_diagram_size}-individual limit) — shown as a summary instead. "
        f"Full detail is in the searchable table above.</div>"
        f"<p><b>{founders}</b> founders, <b>{leaves}</b> leaves, <b>{n_gen}</b> generations</p>"
        f"<div class='gen-bar'>{gen_bar}</div>"
    )


# ---------------------------------------------------------------------------
# Plot one family
# ---------------------------------------------------------------------------

def _family_plot(members, individuals, couples, mz_info, title=""):
    pos = _layout_family(members, individuals, couples)

    gender_color = {'Male': '#4A90E2', 'Female': '#E24A90', None: '#95A5A6'}
    gender_symbol = {'Male': 'square', 'Female': 'circle', None: 'diamond'}
    show_labels = len(members) <= LABEL_THRESHOLD

    fig = go.Figure()

    # Edges: parent-child. Batched into TWO traces total (not two PER couple)
    # using None-separated segments -- Plotly's standard technique for many
    # line segments in one trace. At hundreds/thousands of couples, one trace
    # per edge was the single biggest driver of render slowness and DOM bloat.
    bar_x, bar_y = [], []
    drop_x, drop_y = [], []
    drawn_couples = set()
    for p1, p2, c in couples:
        if p1 not in pos or p2 not in pos or c not in pos:
            continue
        couple_key = tuple(sorted([p1, p2]))
        if couple_key in drawn_couples:
            continue
        drawn_couples.add(couple_key)
        mx = (pos[p1][0] + pos[p2][0]) / 2
        my = pos[p1][1]
        bar_x.extend([pos[p1][0], pos[p2][0], None])
        bar_y.extend([my, my, None])
        drop_x.extend([mx, pos[c][0], None])
        drop_y.extend([my, pos[c][1], None])

    if bar_x:
        fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='lines',
                                 line=dict(color='#555', width=2),
                                 hoverinfo='none', showlegend=False))
    if drop_x:
        fig.add_trace(go.Scatter(x=drop_x, y=drop_y, mode='lines',
                                 line=dict(color='#555', width=1.5),
                                 hoverinfo='none', showlegend=False))

    # Nodes: batched by (gender, is_mz_group) into at most 6 traces total
    # (not one PER individual) -- same marker style within a bucket, so a
    # single trace with array x/y/text/hovertext renders identically.
    buckets = defaultdict(lambda: {'x': [], 'y': [], 'text': [], 'hover': []})
    for name in members:
        info = individuals[name]
        g = info['gender']
        is_mz = bool(info.get('is_mz_group'))
        x, y = pos[name]
        hover = (f"<b>{name}</b><br>"
                 f"Gender: {g or 'Unknown'}<br>"
                 f"Generation: {info['generation']}<br>"
                 f"Parents: {', '.join(info['parents']) or '—'}<br>"
                 f"Children: {', '.join(info['children']) or '—'}")
        if is_mz:
            hover = f"<b>MZ Group ({info['member_count']})</b><br>Members: {', '.join(info['members'])}<br>" + hover
        label = f"MZ×{info['member_count']}" if is_mz else name
        b = buckets[(g, is_mz)]
        b['x'].append(x)
        b['y'].append(y)
        b['text'].append(label)
        b['hover'].append(hover)

    for (g, is_mz), b in buckets.items():
        fig.add_trace(go.Scatter(
            x=b['x'], y=b['y'],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=22, color=gender_color[g], symbol=gender_symbol[g],
                        line=dict(color='#9B59B6' if is_mz else '#2C3E50',
                                  width=3 if is_mz else 1.5)),
            text=b['text'] if show_labels else None,
            textposition='bottom center',
            textfont=dict(size=9),
            hovertext=b['hover'], hoverinfo='text',
            showlegend=False))

    max_gen = max((individuals[m]['generation'] for m in members if m in individuals), default=0)
    widest_gen = max(Counter(individuals[m]['generation'] for m in members
                             if m in individuals).values(), default=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(250, 120 * (max_gen + 2)),
        # Width scales with the WIDEST generation, not just generation depth --
        # without this, a generation with hundreds of members gets squeezed
        # into a fixed-width box regardless of how many nodes it holds.
        width=max(600, 70 * widest_gen),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML report assembly
# ---------------------------------------------------------------------------

_CSS = """
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
       background: #f4f6f9; color: #2C3E50; }
h1 { color: #2C3E50; border-bottom: 3px solid #4A90E2; padding-bottom: 8px; }
h2 { color: #34495E; margin-top: 32px; }
h3 { color: #34495E; margin-top: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin: 16px 0 24px; }
.card { background: white; border-radius: 8px; padding: 16px 22px;
        box-shadow: 0 2px 6px rgba(0,0,0,.10); min-width: 160px; }
.card .val { font-size: 2em; font-weight: bold; color: #4A90E2; }
.card .lbl { font-size: .85em; color: #7f8c8d; margin-top: 2px; }
.alert  { background: #fff3cd; border-left: 4px solid #ffc107;
          padding: 10px 16px; border-radius: 4px; margin: 12px 0; }
.info   { background: #d1ecf1; border-left: 4px solid #17a2b8;
          padding: 10px 16px; border-radius: 4px; margin: 12px 0; }
.danger { background: #f8d7da; border-left: 4px solid #dc3545;
          padding: 10px 16px; border-radius: 4px; margin: 12px 0; }
table { width: 100%; border-collapse: collapse; background: white;
        border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,.08); font-size: .9em; }
thead { background: #4A90E2; color: white; }
th, td { padding: 7px 11px; text-align: left; border-bottom: 1px solid #ecf0f1; }
tbody tr:hover { background: #EBF5FB; }
tr.conf-high  { border-left: 4px solid #27ae60; }
tr.conf-med   { border-left: 4px solid #f39c12; }
tr.conf-low   { border-left: 4px solid #e74c3c; }
tr.duplicate  { background: #fef9e7; }
.controls { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px;
            align-items: center; }
.controls label { font-size: .9em; color: #555; }
.controls input[type=text] { padding: 6px 10px; width: 220px; border: 1px solid #ccc;
                              border-radius: 4px; font-size: .9em; }
.controls input[type=range] { width: 120px; }
.controls span.range-val { font-weight: bold; color: #4A90E2; min-width: 32px; }
.legend { display: flex; gap: 16px; margin: 8px 0 14px; flex-wrap: wrap; }
.legend span { display: flex; align-items: center; gap: 6px; font-size: .85em; }
.legend .dot { width: 14px; height: 14px; border-radius: 2px; }
.dot-high { background: #27ae60; }
.dot-med  { background: #f39c12; }
.dot-low  { background: #e74c3c; }
.dot-dup  { background: #fef9e7; border: 1px solid #f39c12; }
.family-grid { display: grid;
               grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
               gap: 20px; margin-top: 16px; }
.family-card { background: white; border-radius: 8px; padding: 12px;
               box-shadow: 0 2px 6px rgba(0,0,0,.09); }
.gen-bar { display: flex; gap: 6px; align-items: flex-end; margin-top: 8px; }
.gen-bar span { background: #4A90E2; color: white; font-size: .75em;
                padding: 2px 5px; border-radius: 3px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: .8em; font-weight: bold; }
.badge-blue   { background: #D6EAF8; color: #1A5276; }
.badge-orange { background: #FDEBD0; color: #784212; }
.badge-red    { background: #FADBD8; color: #922B21; }
</style>
"""

_TABLE_JS = """
<script>
function applyFilters(tableId) {
  var search  = (document.getElementById('search-' + tableId) || {value:''}).value.toLowerCase();
  var minPerc = parseInt((document.getElementById('perc-' + tableId) || {value:0}).value) || 0;
  var minT1   = parseFloat((document.getElementById('t1-' + tableId) || {value:0}).value) || 0;
  var rows = document.getElementById(tableId).getElementsByTagName('tr');
  for (var i = 1; i < rows.length; i++) {
    var r = rows[i];
    var txt   = r.innerText.toLowerCase();
    var perc  = parseFloat(r.getAttribute('data-perc') || '0');
    var t1    = parseFloat(r.getAttribute('data-t1')   || '0');
    var show  = txt.includes(search) && perc >= minPerc && t1 >= minT1;
    r.style.display = show ? '' : 'none';
  }
}
function updateRange(id, displayId) {
  document.getElementById(displayId).innerText = document.getElementById(id).value;
}
</script>
"""


def _stat_card(val, label):
    return f'<div class="card"><div class="val">{val}</div><div class="lbl">{label}</div></div>'


def _conf_class(perc, t1):
    """CSS class based on stringency at which the trio was confirmed."""
    if perc >= 50 and t1 >= 99:
        return 'conf-high'
    if perc >= 30 and t1 >= 95:
        return 'conf-med'
    return 'conf-low'


def _trio_table_html(rows, table_id, title, explanation=None):
    """Render a filterable trio table with full statistics."""
    parts = [f"<h3>{title}</h3>"]
    if explanation:
        parts.append(f"<div class='info'>{explanation}</div>")

    parts.append(
        "<div class='legend'>"
        "<span><span class='dot dot-high'></span> High confidence (Perc≥50, T1≥99%)</span>"
        "<span><span class='dot dot-med'></span> Medium (Perc≥30, T1≥95%)</span>"
        "<span><span class='dot dot-low'></span> Low (Perc=10 or T1&lt;95%)</span>"
        "<span><span class='dot dot-dup'></span> Child has multiple parent assignments</span>"
        "</div>")

    parts.append(
        f"<div class='controls'>"
        f"<label>Search: <input type='text' id='search-{table_id}' "
        f"  placeholder='sample name…' oninput='applyFilters(\"{table_id}\")'></label>"
        f"<label>Min Perc: <input type='range' id='perc-{table_id}' "
        f"  min='0' max='50' step='10' value='0' "
        f"  oninput='updateRange(this.id,\"perc-val-{table_id}\"); applyFilters(\"{table_id}\")'>"
        f"  <span class='range-val' id='perc-val-{table_id}'>0</span></label>"
        f"<label>Min T1%: <input type='range' id='t1-{table_id}' "
        f"  min='90' max='100' step='1' value='90' "
        f"  oninput='updateRange(this.id,\"t1-val-{table_id}\"); applyFilters(\"{table_id}\")'>"
        f"  <span class='range-val' id='t1-val-{table_id}'>90</span></label>"
        f"</div>")

    parts.append(
        f"<table id='{table_id}'><thead><tr>"
        "<th>Parent 1</th><th>Parent 2</th><th>Child</th><th>Family</th>"
        "<th title='Quality percentile at which this trio was detected'>Perc</th>"
        "<th title='Autosomal variants used'>Variants</th>"
        "<th title='Variants where one parent is AA and the other BB'>AA+BB</th>"
        "<th title='Of those, fraction where child is AB (should be ~100%)'>→AB (T1%)</th>"
        "<th title='Variants where both parents are BB'>BB+BB</th>"
        "<th title='Of those, fraction where child is also BB (should be ~100%)'>→BB (T2%)</th>"
        "<th>Note</th>"
        "</tr></thead><tbody>")

    # Identify children with multiple parent assignments
    child_counts = {}
    for r in rows:
        child_counts[r['child']] = child_counts.get(r['child'], 0) + 1

    for row in rows:
        is_dup = child_counts[row['child']] > 1
        css = _conf_class(row['perc'], row['t1'])
        if is_dup:
            css += ' duplicate'
        note = ''
        if is_dup:
            n = child_counts[row['child']]
            note = f"<span class='badge badge-orange'>{n} parent pairs</span>"
        fam_badge = f"<span class='badge badge-blue'>{row['family']}</span>"
        parts.append(
            f"<tr class='{css}' data-perc='{row['perc']}' data-t1='{row['t1']}'>"
            f"<td>{row['p1']}</td><td>{row['p2']}</td><td>{row['child']}</td>"
            f"<td>{fam_badge}</td>"
            f"<td>{row['perc']}</td>"
            f"<td>{row['autos']}</td>"
            f"<td>{row['aa_bb']}</td>"
            f"<td><b>{row['t1']:.1f}%</b></td>"
            f"<td>{row['bb_bb']}</td>"
            f"<td>{row['t2']:.1f}%</td>"
            f"<td>{note}</td>"
            "</tr>")

    parts.append("</tbody></table>")
    n_dup_children = sum(1 for c, n in child_counts.items() if n > 1)
    if n_dup_children:
        parts.append(
            f"<p style='margin-top:8px; color:#7f8c8d; font-size:.88em;'>"
            f"⚠ <b>{n_dup_children} children</b> appear with more than one parent combination "
            f"(highlighted in yellow). This typically means related individuals in the cohort "
            f"(e.g. siblings of the true parent) also pass the trio test. "
            f"Use the <b>Min Perc</b> and <b>Min T1%</b> sliders to keep only the most "
            f"confident assignments, or inspect each child's rows to identify the best-supported pair."
            f"</p>")
    return "\n".join(parts)


def _build_html_report(builder, stats, trio_table_rows, inverted_table_rows,
                        best_per_child, top_n_families, title, prefix):
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
             f"<title>{title}</title>", _CSS, _TABLE_JS, "</head><body>"]

    parts.append(f"<h1>{title}</h1>")
    parts.append(f"<p style='color:#7f8c8d;'>Source: <code>{prefix}</code></p>")

    # ---- Statistics cards ----
    parts.append("<h2>Summary Statistics</h2><div class='cards'>")
    parts.append(_stat_card(stats['n_individuals'], 'Individuals'))
    parts.append(_stat_card(stats['n_families'], 'Families (clusters)'))
    parts.append(_stat_card(stats['n_founders'], 'Founders (no known parents)'))
    parts.append(_stat_card(stats['n_trios'], 'Raw trio calls'))
    parts.append(_stat_card(stats['n_unique_children'], 'Children with ≥1 trio call'))
    parts.append(_stat_card(stats['n_ambiguous_children'], 'Children with multiple parent pairs'))
    parts.append(_stat_card(stats['n_full_sibling_groups'], 'Full-sibling groups'))
    parts.append(_stat_card(stats['n_half_sib_pairs'], 'Half-sibling pairs'))
    parts.append(_stat_card(stats['max_generation'], 'Max generation depth'))
    if stats['n_mz_groups']:
        parts.append(_stat_card(stats['n_mz_groups'], 'MZ twin groups'))
    if stats['n_males'] or stats['n_females']:
        parts.append(_stat_card(f"{stats['n_males']}M / {stats['n_females']}F", 'Sex (M/F)'))
    if stats['n_inverted_trios']:
        parts.append(_stat_card(stats['n_inverted_trios'], 'Inverted / generational trios'))
    parts.append("</div>")

    # Generation distribution
    gc = stats['generation_counts']
    if gc:
        parts.append("<p><b>Individuals per generation:</b> <span class='gen-bar'>")
        for g in sorted(gc):
            parts.append(f"<span>G{g}: {gc[g]}</span>")
        parts.append("</span></p>")

    # Family size distribution
    fsizes = stats['family_sizes']
    if fsizes:
        parts.append(f"<p><b>Family sizes:</b> largest={fsizes[0]}, "
                     f"median={sorted(fsizes)[len(fsizes)//2]}, "
                     f"singletons={sum(1 for s in fsizes if s == 1)}</p>")

    # Ambiguous children warning
    if stats['n_ambiguous_children'] > 0:
        pct = 100 * stats['n_ambiguous_children'] / max(stats['n_unique_children'], 1)
        parts.append(
            f"<div class='{'danger' if pct > 20 else 'alert'}'>"
            f"<b>{stats['n_ambiguous_children']} of {stats['n_unique_children']} children "
            f"({pct:.0f}%) have more than one parent-pair assignment.</b> "
            "This is expected when the cohort contains related individuals: a sibling, "
            "half-sibling, or cousin of the true parent also passes the allele-sharing test "
            "(especially at low stringency / Perc=10). "
            "The <em>Best assignment per child</em> table below shows only the highest-confidence "
            "call per child. Use the T1% and Perc sliders in the full table to filter out "
            "weaker calls."
            "</div>")

    # Inverted trio note
    if stats['n_inverted_trios']:
        parts.append(
            "<div class='alert'>"
            f"<b>{stats['n_inverted_trios']} inverted/generational trios</b> detected. "
            "These pass T1 (AA+BB→AB ≥ 90%) but fail T2 (BB+BB→BB &lt; 95%), "
            "suggesting a grandparent–parent–grandchild or reversed-direction relationship. "
            "They appear in a separate table below and are excluded from the pedigree diagrams."
            "</div>")

    # ---- Best-per-child table ----
    if best_per_child:
        parts.append("<h2>Best Parent Assignment per Child</h2>")
        expl = (
            "For each child, the single parent pair with the <b>highest T1% at the highest "
            "stringency percentile</b> is shown. This is the most trustworthy pedigree assignment. "
            "Children where only one parent pair was detected have no ambiguity. "
            "Children marked <span class='badge badge-orange'>N parent pairs</span> in the full "
            "table had competing assignments — inspect those rows to understand why "
            "(the competing parents are likely relatives of the best-scoring parent)."
        )
        parts.append(_trio_table_html(best_per_child, 'best-table', '', explanation=expl))

    # ---- Full trio table ----
    parts.append("<h2>Full Trio Table (all calls)</h2>")
    expl_full = (
        "All Regular trio calls from vcfped, including competing parent-pair assignments for "
        "the same child. <b>Perc</b> is the quality-filter percentile at which the trio was "
        "detected (10=loose, 50=strict). <b>T1%</b> = fraction of AA+BB variants where the "
        "child is AB (Mendelian expectation: 100%). <b>T2%</b> = fraction of BB+BB variants "
        "where the child is BB (expectation: 100%). Higher Perc + higher T1/T2 = more reliable."
    )
    parts.append(_trio_table_html(trio_table_rows, 'trio-table', '', explanation=expl_full))

    # ---- Inverted trio table ----
    if inverted_table_rows:
        parts.append("<h2>Inverted / Generational Trio Table</h2>")
        parts.append(
            "<div class='info'>These trios pass T1 but fail T2. The pivot is the most likely "
            "child but the relationship may span more than one generation.</div>")
        parts.append(
            "<table id='inv-table'><thead><tr>"
            "<th>Parent 1</th><th>Parent 2</th><th>Pivot</th>"
            "<th>Perc</th><th>Variants</th><th>AA+BB</th>"
            "<th>→AB (T1%)</th><th>BB+BB</th><th>→BB (T2%)</th>"
            "</tr></thead><tbody>")
        for row in inverted_table_rows:
            parts.append(
                f"<tr><td>{row['p1']}</td><td>{row['p2']}</td><td>{row['child']}</td>"
                f"<td>{row['perc']}</td><td>{row['autos']}</td><td>{row['aa_bb']}</td>"
                f"<td>{row['t1']:.1f}%</td><td>{row['bb_bb']}</td><td>{row['t2']:.1f}%</td>"
                "</tr>")
        parts.append("</tbody></table>")

    # ---- Family pedigree diagrams ----
    if top_n_families:
        n_shown = len(top_n_families)
        n_total = stats['n_families']
        parts.append(f"<h2>Pedigree Diagrams — Top {n_shown} Families</h2>")
        parts.append(
            "<div class='info'>"
            "Each diagram uses only the <b>best parent assignment per child</b> "
            "(highest T1% at highest Perc). Children with ambiguous parents appear with their "
            "most confident pair only."
            "</div>")
        if n_shown < n_total:
            parts.append(f"<p style='color:#7f8c8d;'>Showing the {n_shown} largest families "
                         f"out of {n_total} total. Smaller families appear in the tables above.</p>")
        parts.append("<div class='family-grid'>")
        for fig_html, fam_label in top_n_families:
            parts.append(f"<div class='family-card'>{fig_html}</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive pedigree report from VCFped output',
        epilog='''
Examples:
  %(prog)s results_named
  %(prog)s results_named -o family_tree.html
  %(prog)s results_named --keep-mz-separate --debug
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('prefix',
                        help='VCFped output prefix (e.g., "step3_named")')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML file (default: <prefix>_pedigree.html)')
    parser.add_argument('-t', '--title', default='VCFped Pedigree Report',
                        help='Report title')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')
    parser.add_argument('--keep-mz-separate', action='store_true',
                        help='Show MZ twins as separate nodes instead of collapsing')
    parser.add_argument('--max-families', type=int, default=20,
                        help='Maximum number of family diagrams to draw (default: 20)')
    parser.add_argument('--min-family-size', type=int, default=3,
                        help='Minimum family size to draw a diagram for (default: 3)')
    parser.add_argument('--max-diagram-size', type=int, default=150,
                        help='Families larger than this get a compact summary card instead '
                             'of a full diagram -- no layout algorithm makes a diagram with '
                             'hundreds/thousands of nodes readable as one picture (default: 150)')
    parser.add_argument('--debug', action='store_true',
                        help='Print detailed debugging information')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.prefix}_pedigree.html"

    print("=" * 70)
    print("VCFped Pedigree Visualization")
    print("=" * 70)
    print(f"Input prefix: {args.prefix}")
    print(f"Output file:  {args.output}")
    print()

    print("Parsing VCFped output files...")
    builder, trio_df = parse_vcfped_files(
        args.prefix,
        collapse_mz_twins=not args.keep_mz_separate,
        debug=args.debug,
    )
    print(f"  {len(builder.individuals)} individuals, "
          f"{len(builder.couples)} regular trios, "
          f"{len(builder.inverted_trios)} inverted trios")
    print()

    print("Computing statistics...")
    stats = builder.compute_statistics()
    families = stats['families']
    print(f"  {stats['n_families']} family clusters detected")
    print(f"  Largest family: {stats['family_sizes'][0] if stats['family_sizes'] else 0} individuals")
    print()

    # Build family-ID lookup
    fam_label = {}
    for fam_idx, fam in enumerate(families, 1):
        for m in fam:
            fam_label[m] = f"F{fam_idx}"

    # Build a lookup from (p1,p2,child) -> full statistics row in trio_df
    # trio_df columns: Triple, Pivot, Perc, Autos, AA+BB, =AB, Test1, BB+BB, =BB, Test2, Verdict
    trio_stats = {}
    col_autos = 'Autos'
    col_aa_bb = 'AA+BB'
    col_ab    = '=AB'
    col_t1    = 'Test1'
    col_bb_bb = 'BB+BB'
    col_bb    = '=BB'
    col_t2    = 'Test2'
    for _, row in trio_df.iterrows():
        triple = [s.strip() for s in str(row['Triple']).split(',')]
        pivot  = str(row['Pivot']).strip()
        if len(triple) != 3:
            continue
        child   = pivot
        parents = [s for s in triple if s != child]
        if len(parents) != 2:
            continue
        key = (parents[0], parents[1], child)
        trio_stats[key] = {
            'perc':  int(row.get('Perc', 0)),
            'autos': int(row.get(col_autos, 0)),
            'aa_bb': int(row.get(col_aa_bb, 0)),
            'ab':    int(row.get(col_ab, 0)),
            't1':    float(row.get(col_t1, 0.0)),
            'bb_bb': int(row.get(col_bb_bb, 0)),
            'bb':    int(row.get(col_bb, 0)),
            't2':    float(row.get(col_t2, 0.0)),
        }

    def _make_row(p1, p2, child, verdict='Regular trio'):
        key = (p1, p2, child)
        s = trio_stats.get(key, trio_stats.get((p2, p1, child), {}))
        return {
            'p1': p1, 'p2': p2, 'child': child,
            'family': fam_label.get(child, '?'),
            'perc':  s.get('perc', 0),
            'autos': s.get('autos', 0),
            'aa_bb': s.get('aa_bb', 0),
            't1':    s.get('t1', 0.0),
            'bb_bb': s.get('bb_bb', 0),
            't2':    s.get('t2', 0.0),
        }

    # Full trio table rows (all calls)
    trio_table_rows = [_make_row(p1, p2, c) for p1, p2, c in builder.couples]
    trio_table_rows.sort(key=lambda r: (r['family'], r['child'], -r['perc'], -r['t1']))

    # Best-per-child: highest Perc then T1 per child
    best_per_child_map = {}
    for r in trio_table_rows:
        child = r['child']
        if child not in best_per_child_map or \
           (r['perc'], r['t1']) > (best_per_child_map[child]['perc'],
                                   best_per_child_map[child]['t1']):
            best_per_child_map[child] = r
    best_per_child = sorted(best_per_child_map.values(),
                            key=lambda r: (r['family'], r['child']))

    # Inverted trio table rows
    inverted_table_rows = [_make_row(p1, p2, c, 'Inverted')
                           for p1, p2, c in builder.inverted_trios]

    # Add ambiguous-child counts to stats
    child_counts = {}
    for r in trio_table_rows:
        child_counts[r['child']] = child_counts.get(r['child'], 0) + 1
    stats['n_unique_children']    = len(child_counts)
    stats['n_ambiguous_children'] = sum(1 for n in child_counts.values() if n > 1)

    # Draw top-N family pedigrees
    print(f"Drawing pedigree diagrams (top {args.max_families} families ≥ {args.min_family_size} members, "
          f"max {args.max_diagram_size} individuals per diagram)...")
    top_n_families = []
    drawn = 0
    n_summarized = 0
    plotlyjs_included = False
    for fam_idx, fam in enumerate(families, 1):
        if len(fam) < args.min_family_size:
            break
        if drawn >= args.max_families:
            break
        members = [m for m in fam if m in builder.individuals]
        n_gen = max((builder.individuals[m]['generation'] for m in members
                     if m in builder.individuals), default=0)
        label = f"Family F{fam_idx} — {len(members)} individuals, {n_gen} generations"
        if len(members) > args.max_diagram_size:
            card_html = _family_summary_card(members, builder.individuals, label,
                                             args.max_diagram_size)
            top_n_families.append((card_html, label))
            drawn += 1
            n_summarized += 1
            continue
        # Use only best-per-child assignments for diagrams to avoid cluttered multi-parent nodes
        fam_couples = [(r['p1'], r['p2'], r['child'])
                       for r in best_per_child
                       if r['child'] in fam and r['p1'] in fam and r['p2'] in fam]
        try:
            fig = _family_plot(members, builder.individuals, fam_couples,
                               builder.mz_group_members, title=label)
            fig_html = fig.to_html(include_plotlyjs='cdn' if not plotlyjs_included else False,
                                   full_html=False, div_id=f'fam{fam_idx}')
            plotlyjs_included = True
            top_n_families.append((fig_html, label))
            drawn += 1
        except Exception as e:
            print(f"  Warning: could not draw family F{fam_idx}: {e}")

    print(f"  Drew {drawn - n_summarized} full diagram(s), {n_summarized} summarized "
          f"(too large to render), {drawn} total")
    print()

    print("Assembling HTML report...")
    html = _build_html_report(
        builder, stats, trio_table_rows, inverted_table_rows,
        best_per_child, top_n_families, args.title, args.prefix,
    )
    with open(args.output, 'w') as f:
        f.write(html)
    print(f"✓ Saved to {args.output}")

    print()
    print("Summary:")
    print(f"  {stats['n_individuals']} individuals across {stats['n_families']} families")
    print(f"  {stats['n_trios']} regular trios / {stats['n_inverted_trios']} inverted trios")
    print(f"  {stats['n_founders']} founders, {stats['max_generation']} generation depth")
    print(f"  {stats['n_full_sibling_groups']} full-sibling groups, "
          f"{stats['n_half_sib_pairs']} half-sibling pairs")
    print("=" * 70)

    if not args.no_browser:
        import webbrowser, os
        webbrowser.open('file://' + os.path.abspath(args.output))
        print(f"Opening {args.output} in browser...")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
