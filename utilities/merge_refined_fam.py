#!/usr/bin/env python3
"""
Merge a candidate-verification refined FAM back into the full-cohort pedigree.

BACKGROUND
----------
The standard large-cohort extension workflow is:

  1. refine_pedigree.py on the full cohort -> <cohort>_refined.fam
     (founders = individuals with no confirmed parents yet)
  2. extend_pedigree.py on that FAM -> candidate grandparent/offspring lists
  3. Extract candidate samples into a subset VCF and re-run vcfped.py (NOT
     vcfped_optimized.py -- see that script's --all discrepancy note) --all
     on just that subset (fast: O(k^3) not O(N^3)) to get real T1/T2 verdicts
  4. filter_trios.py + refine_pedigree.py on the SUBSET's vcfped output
     -> <subset>_refined.fam

Step 4's output FAM only ever contains individuals who got a confirmed
parent-pair during that subset run — refine_pedigree.py's FAM-writing loop
has no path to carry over an unassigned founder from a base FAM unless one
was explicitly loaded. Passed a bare subset trio_prefix with no --fam, that
FAM starts empty, so EVERY row in <subset>_refined.fam already has parents
by construction: running extend_pedigree.py directly on it will always
report zero founders and zero grandparent candidates, regardless of the
actual biology. That's an artifact of scope, not a finding.

This script merges the subset's newly-confirmed parent-child assignments
back into the full-cohort FAM from step 1, so the founders that just got
confirmed parents are updated, and a subsequent extend_pedigree.py run has
the full cohort's founders and leaves to search from again.

CONFLICT HANDLING
------------------
An individual already having DIFFERENT (non-'0') parents in the base FAM
than what the new FAM proposes is a conflict, not silently resolved:
  --on-conflict keep-base (default)  Keep the base FAM's existing assignment.
  --on-conflict keep-new             Overwrite with the new FAM's assignment.
Either way, every conflict is listed in the report so nothing is hidden.

To make an informed per-conflict decision rather than guessing from bare
parent IDs, this script optionally joins in the evidence behind each side:
  --base-trio / --new-trio   the companion <prefix>_refined.trio files
                              (auto-derived from base_fam/new_fam by default)
                              -> Verdict, Perc, Test1 for each proposed pair
  --king                      a PLINK2 --make-king-table prefix
                              -> KINSHIP/IBS0 for child<->each proposed parent,
                                 an independent genetic check alongside the
                                 trio verdict (useful since fixed T1/Perc
                                 confidence can be inflated in closely related
                                 germplasm — see refine_pedigree.py)
All three are optional and degrade gracefully to "NA" if not found — this
script still works with just two bare FAM files.

MENDELIAN VETO
--------------
A fourth, independent signal can be layered on top of --score-conflicts:
  --base-mendel / --new-mendel   PLINK2 .fmendel per-family error counts for
                                   each side's proposed pair (not auto-derived
                                   — there's no fixed naming convention tying
                                   an .fmendel report to a specific FAM prefix)
  --mendel-bim / --mendel-n-variants   variant count, to turn the raw error
                                   count into a rate (same formula as
                                   analyze_mendel_errors.py)
If the side --score-conflicts currently favors (prefer_new, prefer_base, or
tie_new_higher) has a Mendelian error rate above --mendel-absolute-multiplier
(default 2.0) x the expected baseline (2 x --error-rate), the recommendation
is downgraded to 'tie_mendel_veto'. This can only VETO an automatic decision,
never independently promote one — a plain 'tie' is left as-is, and
--auto-resolve's mapping doesn't change (only 'prefer_new' keeps new; every
other value, including 'tie_mendel_veto', keeps base) — so the veto's only
effect is to make --auto-resolve conservative when kinship and Mendelian
evidence disagree, without adding a new automated code path.

INPUT
-----
  <base_fam>  — the full-cohort refined FAM to merge into
  <new_fam>   — the candidate-subset run's refined FAM with new confirmations

OUTPUT
------
  <output>.fam               — merged pedigree (base FAM order preserved;
                                individuals only in new_fam appended at the end)
  <output>_merge_report.txt  — counts + narrative conflict/founder/new-individual list
  <output>_conflicts.tsv     — full per-conflict evidence table (only written
                                if there are any conflicts), for sorting/
                                filtering in a spreadsheet or pandas

Usage:
  python merge_refined_fam.py step3_named_filtered_refined.fam \\
      extension_out_named_relevant_filtered_refined.fam --king step3_named_filtered_king_ext
  python merge_refined_fam.py base.fam new.fam -o merged --on-conflict keep-new
"""

import argparse
import csv
import os
import sys
from collections import Counter


def load_fam(path):
    """Load a .fam file (whitespace-delimited, PLINK convention).

    Returns (fam, order): fam is {iid: {fid, pat, mat, sex}}, order is the
    list of IIDs in the order they appeared, for stable output ordering.
    """
    fam = {}
    order = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            fid, iid, pat, mat = parts[0], parts[1], parts[2], parts[3]
            sex = parts[4] if len(parts) > 4 else '0'
            fam[iid] = {'fid': fid, 'pat': pat, 'mat': mat, 'sex': sex}
            order.append(iid)
    return fam, order


def write_fam(path, fam, order):
    with open(path, 'w') as f:
        for iid in order:
            d = fam[iid]
            f.write(f"{d['fid']}\t{iid}\t{d['pat']}\t{d['mat']}\t{d['sex']}\t-9\n")


def is_founder(d):
    return d['pat'] in ('0', '') and d['mat'] in ('0', '')


def default_trio_path(fam_path):
    """Derive the companion <prefix>_refined.trio path from a <prefix>_refined.fam
    path — refine_pedigree.py always writes both from the same prefix."""
    if fam_path.endswith('.fam'):
        return fam_path[:-len('.fam')] + '.trio'
    return fam_path + '.trio'


def load_trio_lookup(path):
    """
    Parse a .trio file into a lookup: (child, p1, p2) and (child, p2, p1) ->
    {perc, test1, test2, verdict}. Used to attach confidence evidence to each
    side of a conflict so it can be judged on more than bare parent IDs.
    Optional enhancement — returns {} if the file doesn't exist, same as a
    missing --king file.
    """
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            triple = [s.strip() for s in str(row.get('Triple', '')).split(',')]
            pivot = str(row.get('Pivot', '')).strip()
            if len(triple) != 3 or pivot not in triple:
                continue
            child = pivot
            parents = [s for s in triple if s != child]
            if len(parents) != 2:
                continue
            p1, p2 = parents
            info = {
                'perc': row.get('Perc', 'NA'),
                'test1': row.get('Test1', 'NA'),
                'test2': row.get('Test2', 'NA'),
                'verdict': row.get('Verdict', 'NA'),
            }
            lookup[(child, p1, p2)] = info
            lookup[(child, p2, p1)] = info
    return lookup


def lookup_trio_info(trio_lookup, child, p1, p2):
    return trio_lookup.get((child, p1, p2)) or {'perc': 'NA', 'test1': 'NA', 'test2': 'NA', 'verdict': 'NA'}


def load_king_lookup(king_prefix):
    """
    Parse a PLINK2 --make-king-table .kin0 file into {(id1,id2): {kinship, ibs0}}.
    Optional — returns {} if no prefix is given or the file doesn't exist.
    """
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
                ibs0 = float(row.get('IBS0', 0))
            except ValueError:
                continue
            king[(id1, id2)] = {'kinship': kinship, 'ibs0': ibs0}
    return king


def get_king(king, a, b):
    return king.get((a, b)) or king.get((b, a))


# ---------------------------------------------------------------------------
# Mendelian veto: a third independent signal (analyze_mendel_errors.py's
# family-level error rate) that can DOWNGRADE an automatic kinship-based
# recommendation to 'tie_mendel_veto', but never independently promote one
# -- same conservative philosophy --auto-resolve already applies to ties.
# ---------------------------------------------------------------------------

def count_variants(bim_path):
    """Count lines in a PLINK .bim file (one line per variant). Same as
    analyze_mendel_errors.py's count_variants() -- duplicated per this
    project's standalone-script convention."""
    if not os.path.exists(bim_path):
        sys.exit(f"Error: {bim_path} not found.")
    with open(bim_path) as f:
        return sum(1 for _ in f)


def load_fmendel_lookup(path):
    """
    Parse a PLINK2 .fmendel file into {sorted (pat,mat) tuple: {n, child}}.
    n is a raw error COUNT and child is the family's offspring count (see
    analyze_mendel_errors.py's load_fmendel()) -- not yet a rate.
    Optional: returns {} if no path given or the file doesn't exist, same
    convention as load_king_lookup()/load_trio_lookup().
    """
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].lstrip('#').upper() == 'FID':
                continue  # header line
            if len(parts) < 5:
                continue
            _, pat, mat, child, n = parts[:5]
            try:
                child = int(child)
                n = int(n)
            except ValueError:
                continue
            lookup[tuple(sorted([pat, mat]))] = {'n': n, 'child': child}
    return lookup


def get_mendel_rate(mendel_lookup, n_variants, pat, mat):
    """
    Family-level Mendelian error rate for a (pat, mat) pair: n / (n_variants
    x child), same formula as analyze_mendel_errors.py's compute_rates().
    Returns None if untested (no lookup, no n_variants, or pair not found).
    """
    if not mendel_lookup or not n_variants:
        return None
    rec = mendel_lookup.get(tuple(sorted([pat, mat])))
    if not rec:
        return None
    denom = n_variants * rec['child']
    return rec['n'] / denom if denom > 0 else None


def expected_baseline_rate(error_rate):
    """Same '~2x error rate' relationship as analyze_mendel_errors.py /
    prepare_plink_beagle.py's generated _run_plink.sh."""
    return 2 * error_rate


def is_mendel_elevated(rate, error_rate, absolute_multiplier):
    """True/False if elevated relative to the error-rate-derived baseline,
    None if untested (rate is None)."""
    if rate is None:
        return None
    return rate > absolute_multiplier * expected_baseline_rate(error_rate)


def apply_mendel_veto(recommendation, new_mendel_flag, base_mendel_flag):
    """
    Downgrade a kinship-based recommendation to 'tie_mendel_veto' if the
    side it currently favors shows an elevated Mendelian error rate.
    Mendelian evidence only vetoes an automatic decision -- it never
    promotes one: a plain 'tie' is left as-is (already conservative), and
    only the WINNING side's own flag can trigger a downgrade (a losing
    side's elevated rate changes nothing -- it was already not being acted on).
    None (untested) never vetoes.
    """
    winning_side = None
    if recommendation in ('prefer_new', 'tie_new_higher'):
        winning_side = 'new'
    elif recommendation == 'prefer_base':
        winning_side = 'base'

    if winning_side == 'new' and new_mendel_flag:
        return 'tie_mendel_veto'
    if winning_side == 'base' and base_mendel_flag:
        return 'tie_mendel_veto'
    return recommendation


# ---------------------------------------------------------------------------
# --score-conflicts: systematic classification of each conflict using the
# same population-relative criteria refine_pedigree.py already established
# for this kind of closely related cohort (fixed kinship thresholds are
# unreliable here — see that script's own NOTE about high background
# kinship — so thresholds are derived from the --king file itself, not
# hardcoded).
# ---------------------------------------------------------------------------

def ibs0_threshold_for_error_rate(error_rate):
    """Same formula as refine_pedigree.py's ibs0_threshold_for_error_rate()."""
    return min(0.10, 2 * error_rate * 1.5)


def compute_kinship_percentiles(king):
    """
    Cohort-relative kinship percentiles (p25/p75/p95), same computation
    refine_pedigree.py prints at startup. Returns (None, None, None) if king
    is empty — scoring degrades to UNTESTED/REJECTED-only classification.
    """
    kinships = sorted(v['kinship'] for v in king.values())
    if not kinships:
        return None, None, None
    n = len(kinships)
    p25 = kinships[n // 4]
    p75 = kinships[3 * n // 4]
    p95 = kinships[int(0.95 * n)]
    return p25, p75, p95


CLASSIFICATION_RANK = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1, 'UNTESTED': 0, 'REJECTED': -1}


def classify_parent(kinship, ibs0, ibs0_ceiling, p25, p75):
    """
    Classify one proposed parent's evidence:
      REJECTED — IBS0 above the noise ceiling: incompatible with true PO
      UNTESTED — no KING data for this pair
      STRONG   — kinship >= cohort p75 (clearly closer than typical background)
      MODERATE — kinship in the p25-p75 band (typical background relatedness
                 in this closely related cohort -- consistent with, but not
                 distinguishing, a true PO pair)
      WEAK     — kinship below cohort p25
    """
    if kinship is None or ibs0 is None:
        return 'UNTESTED'
    if ibs0 > ibs0_ceiling:
        return 'REJECTED'
    if p75 is not None and kinship >= p75:
        return 'STRONG'
    if p25 is not None and kinship >= p25:
        return 'MODERATE'
    return 'WEAK'


def _recommend(base_p1, base_p2, new_p1, new_p2, base_weak_kinship, new_weak_kinship):
    """
    Shared recommendation core: given each side's two parent classifications
    and each side's weakest-parent kinship, decide prefer_new / prefer_base /
    tie / tie_new_higher. Used both for already-built evidence rows
    (score_conflict_row(), for the report/TSV) and for resolving
    --auto-resolve conflicts inline during merge() (score_conflict_kinship(),
    before that evidence table exists).
    """
    base_overall = min(base_p1, base_p2, key=lambda s: CLASSIFICATION_RANK[s])
    new_overall = min(new_p1, new_p2, key=lambda s: CLASSIFICATION_RANK[s])
    base_rank, new_rank = CLASSIFICATION_RANK[base_overall], CLASSIFICATION_RANK[new_overall]

    if new_rank > base_rank:
        recommendation = 'prefer_new'
    elif base_rank > new_rank:
        recommendation = 'prefer_base'
    else:
        recommendation = 'tie_new_higher' if (new_weak_kinship - base_weak_kinship) > 0.05 else 'tie'

    return base_overall, new_overall, recommendation


def score_conflict_row(row, ibs0_ceiling, p25, p75, base_mendel_lookup=None,
                       new_mendel_lookup=None, mendel_n_variants=None,
                       error_rate=0.025, mendel_absolute_multiplier=2.0):
    """
    Given one row from build_conflict_table() (already has base/new
    kinship+ibs0 fields, possibly 'NA'), classify each side by its WORSE
    parent (a trio is only as strong as its weakest confirmed link) and
    recommend prefer_new / prefer_base / tie based on which side scores
    higher overall. If Mendelian evidence is available (base_mendel_lookup/
    new_mendel_lookup), it can downgrade that recommendation to
    'tie_mendel_veto' -- see apply_mendel_veto().

    Returns (base_overall, new_overall, recommendation, base_mendel_rate,
    new_mendel_rate) -- the last two are None if untested.
    """
    def _val(x):
        return None if x == 'NA' else float(x)

    base_p1 = classify_parent(_val(row['base_kinship_p1']), _val(row['base_ibs0_p1']), ibs0_ceiling, p25, p75)
    base_p2 = classify_parent(_val(row['base_kinship_p2']), _val(row['base_ibs0_p2']), ibs0_ceiling, p25, p75)
    new_p1 = classify_parent(_val(row['new_kinship_p1']), _val(row['new_ibs0_p1']), ibs0_ceiling, p25, p75)
    new_p2 = classify_parent(_val(row['new_kinship_p2']), _val(row['new_ibs0_p2']), ibs0_ceiling, p25, p75)

    base_weak_k = min(_val(row['base_kinship_p1']) or 0, _val(row['base_kinship_p2']) or 0)
    new_weak_k = min(_val(row['new_kinship_p1']) or 0, _val(row['new_kinship_p2']) or 0)

    base_overall, new_overall, recommendation = _recommend(
        base_p1, base_p2, new_p1, new_p2, base_weak_k, new_weak_k)

    base_mendel_rate = get_mendel_rate(base_mendel_lookup, mendel_n_variants, row['base_p1'], row['base_p2'])
    new_mendel_rate = get_mendel_rate(new_mendel_lookup, mendel_n_variants, row['new_p1'], row['new_p2'])
    base_mendel_flag = is_mendel_elevated(base_mendel_rate, error_rate, mendel_absolute_multiplier)
    new_mendel_flag = is_mendel_elevated(new_mendel_rate, error_rate, mendel_absolute_multiplier)
    recommendation = apply_mendel_veto(recommendation, new_mendel_flag, base_mendel_flag)

    return base_overall, new_overall, recommendation, base_mendel_rate, new_mendel_rate


def score_conflict_kinship(iid, bp, bm, np_, nm, king, ibs0_ceiling, p25, p75,
                          base_mendel_lookup=None, new_mendel_lookup=None,
                          mendel_n_variants=None, error_rate=0.025,
                          mendel_absolute_multiplier=2.0):
    """
    Lightweight recommendation computed directly from raw KING lookups,
    for --auto-resolve's per-conflict decision during merge() — this runs
    BEFORE the full per-conflict evidence table (build_conflict_table())
    exists, so it can't reuse a pre-built row dict like score_conflict_row()
    does. Returns just the recommendation string (post Mendelian veto, if
    Mendelian lookups are given — see apply_mendel_veto()).
    """
    def _lookup(a, b):
        rec = get_king(king, a, b)
        return (rec['kinship'], rec['ibs0']) if rec else (None, None)

    base_k1, base_i1 = _lookup(iid, bp)
    base_k2, base_i2 = _lookup(iid, bm)
    new_k1, new_i1 = _lookup(iid, np_)
    new_k2, new_i2 = _lookup(iid, nm)

    base_p1 = classify_parent(base_k1, base_i1, ibs0_ceiling, p25, p75)
    base_p2 = classify_parent(base_k2, base_i2, ibs0_ceiling, p25, p75)
    new_p1 = classify_parent(new_k1, new_i1, ibs0_ceiling, p25, p75)
    new_p2 = classify_parent(new_k2, new_i2, ibs0_ceiling, p25, p75)

    base_weak_k = min(base_k1 or 0, base_k2 or 0)
    new_weak_k = min(new_k1 or 0, new_k2 or 0)

    _, _, recommendation = _recommend(base_p1, base_p2, new_p1, new_p2, base_weak_k, new_weak_k)

    base_mendel_rate = get_mendel_rate(base_mendel_lookup, mendel_n_variants, bp, bm)
    new_mendel_rate = get_mendel_rate(new_mendel_lookup, mendel_n_variants, np_, nm)
    base_mendel_flag = is_mendel_elevated(base_mendel_rate, error_rate, mendel_absolute_multiplier)
    new_mendel_flag = is_mendel_elevated(new_mendel_rate, error_rate, mendel_absolute_multiplier)
    return apply_mendel_veto(recommendation, new_mendel_flag, base_mendel_flag)


def fixed_resolver(choice):
    """
    resolve_conflict callback that always returns `choice` ('base' or 'new'),
    regardless of the individual conflict — used for the --on-conflict
    (single global choice) mode, as opposed to --auto-resolve's
    per-conflict score_conflict_kinship()-based callback.
    """
    def _resolve(iid, bp, bm, np_, nm):
        return choice
    return _resolve


def merge(base_fam, base_order, new_fam, resolve_conflict):
    """
    resolve_conflict(iid, base_pat, base_mat, new_pat, new_mat) -> 'base' | 'new'
    Called once per conflicting individual to decide which side's parents to
    keep. This lets the caller plug in either a single global choice
    (--on-conflict, same answer for every conflict) or a per-conflict
    decision (--auto-resolve, using score_conflict_kinship()'s recommendation
    for THIS specific child) without merge() needing to know which.

    Returns (merged_fam, merged_order, stats) where stats has:
      parents_added   — base founders that received parents from new_fam
      conflicts       — [(iid, base_pat, base_mat, new_pat, new_mat, kept)]
      new_individuals — IIDs present only in new_fam
      unchanged       — new_fam entries that already matched base_fam exactly
    """
    merged_fam = {iid: dict(d) for iid, d in base_fam.items()}
    merged_order = list(base_order)
    conflicts = []
    parents_added = []
    new_individuals = []
    unchanged = []

    for iid, new_d in new_fam.items():
        if iid not in merged_fam:
            merged_fam[iid] = dict(new_d)
            merged_order.append(iid)
            new_individuals.append(iid)
            continue

        base_d = merged_fam[iid]
        # Snapshot as plain strings BEFORE any mutation below -- base_d is a
        # reference into merged_fam, so if we kept reading base_d['pat']/
        # ['mat'] after writing merged_fam[iid]['pat']/['mat'] in the 'new'
        # branch, we'd see the just-written NEW values instead of the
        # original base ones (silently corrupting the conflicts list and
        # the evidence table's Base_P1/Base_P2 columns whenever a conflict
        # resolves to 'new').
        base_pat, base_mat = base_d['pat'], base_d['mat']
        same = (base_pat == new_d['pat'] and base_mat == new_d['mat'])
        if same:
            unchanged.append(iid)
            continue

        if is_founder(base_d):
            merged_fam[iid]['pat'] = new_d['pat']
            merged_fam[iid]['mat'] = new_d['mat']
            parents_added.append(iid)
        else:
            kept = resolve_conflict(iid, base_pat, base_mat, new_d['pat'], new_d['mat'])
            if kept == 'new':
                merged_fam[iid]['pat'] = new_d['pat']
                merged_fam[iid]['mat'] = new_d['mat']
            conflicts.append((iid, base_pat, base_mat,
                               new_d['pat'], new_d['mat'], kept))

        # Fill in sex if the base FAM didn't have it and the new one does.
        if merged_fam[iid]['sex'] in ('0', '') and new_d['sex'] not in ('0', ''):
            merged_fam[iid]['sex'] = new_d['sex']

    stats = dict(parents_added=parents_added, conflicts=conflicts,
                 new_individuals=new_individuals, unchanged=unchanged)
    return merged_fam, merged_order, stats


CONFLICT_TSV_FIELDS = [
    'child', 'base_p1', 'base_p2', 'base_verdict', 'base_perc', 'base_test1',
    'base_kinship_p1', 'base_ibs0_p1', 'base_kinship_p2', 'base_ibs0_p2',
    'new_p1', 'new_p2', 'new_verdict', 'new_perc', 'new_test1',
    'new_kinship_p1', 'new_ibs0_p1', 'new_kinship_p2', 'new_ibs0_p2', 'kept',
    'base_overall', 'new_overall', 'recommendation',
    'base_mendel_rate', 'new_mendel_rate',
]
CONFLICT_TSV_HEADER = [
    'Child', 'Base_P1', 'Base_P2', 'Base_Verdict', 'Base_Perc', 'Base_Test1',
    'Base_Kinship_P1', 'Base_IBS0_P1', 'Base_Kinship_P2', 'Base_IBS0_P2',
    'New_P1', 'New_P2', 'New_Verdict', 'New_Perc', 'New_Test1',
    'New_Kinship_P1', 'New_IBS0_P1', 'New_Kinship_P2', 'New_IBS0_P2', 'Kept',
    'Base_Overall', 'New_Overall', 'Recommendation',
    'Base_Mendel_Rate', 'New_Mendel_Rate',
]


def build_conflict_table(conflicts, base_trio, new_trio, king,
                          score=False, ibs0_ceiling=None, p25=None, p75=None,
                          base_mendel_lookup=None, new_mendel_lookup=None,
                          mendel_n_variants=None, error_rate=0.025,
                          mendel_absolute_multiplier=2.0):
    """
    Assemble the full evidence table for each conflict: trio verdict/
    confidence and KING kinship/IBS0 for both the base and new proposed
    parent pairs, so a human can judge which side is more trustworthy
    instead of just seeing two bare ID pairs. Any source not provided (or
    missing the relevant pair) fills in 'NA' rather than failing.

    When score=True (--score-conflicts), also fills base_overall/new_overall/
    recommendation via score_conflict_row() — see that function's docstring,
    including the Mendelian veto if base_mendel_lookup/new_mendel_lookup are
    given. Otherwise these fields are 'NA' (not scored).
    """
    rows = []
    for iid, bp, bm, np_, nm, kept in conflicts:
        base_info = lookup_trio_info(base_trio, iid, bp, bm)
        new_info = lookup_trio_info(new_trio, iid, np_, nm)

        base_k_p1 = get_king(king, iid, bp)
        base_k_p2 = get_king(king, iid, bm)
        new_k_p1 = get_king(king, iid, np_)
        new_k_p2 = get_king(king, iid, nm)

        row = {
            'child': iid,
            'base_p1': bp, 'base_p2': bm,
            'base_verdict': base_info['verdict'], 'base_perc': base_info['perc'],
            'base_test1': base_info['test1'],
            'base_kinship_p1': base_k_p1['kinship'] if base_k_p1 else 'NA',
            'base_ibs0_p1': base_k_p1['ibs0'] if base_k_p1 else 'NA',
            'base_kinship_p2': base_k_p2['kinship'] if base_k_p2 else 'NA',
            'base_ibs0_p2': base_k_p2['ibs0'] if base_k_p2 else 'NA',
            'new_p1': np_, 'new_p2': nm,
            'new_verdict': new_info['verdict'], 'new_perc': new_info['perc'],
            'new_test1': new_info['test1'],
            'new_kinship_p1': new_k_p1['kinship'] if new_k_p1 else 'NA',
            'new_ibs0_p1': new_k_p1['ibs0'] if new_k_p1 else 'NA',
            'new_kinship_p2': new_k_p2['kinship'] if new_k_p2 else 'NA',
            'new_ibs0_p2': new_k_p2['ibs0'] if new_k_p2 else 'NA',
            'kept': kept,
            'base_overall': 'NA', 'new_overall': 'NA', 'recommendation': 'NA',
            'base_mendel_rate': 'NA', 'new_mendel_rate': 'NA',
        }
        if score:
            (row['base_overall'], row['new_overall'], row['recommendation'],
             base_mendel_rate, new_mendel_rate) = score_conflict_row(
                row, ibs0_ceiling, p25, p75, base_mendel_lookup, new_mendel_lookup,
                mendel_n_variants, error_rate, mendel_absolute_multiplier)
            row['base_mendel_rate'] = base_mendel_rate if base_mendel_rate is not None else 'NA'
            row['new_mendel_rate'] = new_mendel_rate if new_mendel_rate is not None else 'NA'
        rows.append(row)
    return rows


def write_conflict_tsv(path, rows):
    with open(path, 'w') as f:
        f.write('\t'.join(CONFLICT_TSV_HEADER) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[k]) for k in CONFLICT_TSV_FIELDS) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Merge a candidate-verification refined FAM back into the full-cohort pedigree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_named_filtered_refined.fam extension_out_named_relevant_filtered_refined.fam \\
      --king step3_named_filtered_king_ext
  %(prog)s base.fam new.fam -o merged --on-conflict keep-new

Typical workflow position:
  refine_pedigree.py (full cohort) -> extend_pedigree.py -> vcfped rerun on
  candidates -> filter_trios.py + refine_pedigree.py (subset) -> THIS SCRIPT
  -> extend_pedigree.py again, now with real founders to search from.
        """,
    )
    parser.add_argument('base_fam', help='Full-cohort refined FAM to merge into')
    parser.add_argument('new_fam', help='Candidate-subset refined FAM with new confirmations')
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: <base_fam>_merged)',
                        default=None)
    parser.add_argument('--on-conflict', choices=['keep-base', 'keep-new'], default='keep-base',
                        help='When an individual already has DIFFERENT parents in both FAMs, '
                             'which to keep (default: keep-base). Applied uniformly to every '
                             'conflict; ignored if --auto-resolve is set. Every conflict is '
                             'reported either way.')
    parser.add_argument('--auto-resolve', action='store_true',
                        help='Resolve each conflict individually using its own '
                             '--score-conflicts recommendation instead of one global '
                             '--on-conflict choice: prefer_new -> keep new; prefer_base, '
                             'tie, and tie_new_higher -> keep base (conservative — only an '
                             'unambiguous rank difference is auto-applied; tie_new_higher is '
                             'still flagged in the report for manual review, but not acted on '
                             'automatically). Requires --score-conflicts (and therefore --king).')
    parser.add_argument('--base-trio',
                        help='Companion <prefix>_refined.trio for base_fam, for conflict '
                             'evidence (default: derived from base_fam by replacing .fam '
                             'with .trio)',
                        default=None)
    parser.add_argument('--new-trio',
                        help='Companion <prefix>_refined.trio for new_fam, for conflict '
                             'evidence (default: derived from new_fam by replacing .fam '
                             'with .trio)',
                        default=None)
    parser.add_argument('--king',
                        help='PLINK2 --make-king-table prefix (reads <prefix>.kin0), for '
                             'independent KINSHIP/IBS0 evidence on each conflict (optional)',
                        default=None)
    parser.add_argument('--score-conflicts', action='store_true',
                        help='Classify each conflict (STRONG/MODERATE/WEAK/UNTESTED per '
                             'proposed parent, worst-of-two per side) and recommend '
                             'prefer_new/prefer_base/tie, using cohort-relative kinship '
                             'percentiles computed from --king (same population-agnostic '
                             'approach as refine_pedigree.py — fixed thresholds are '
                             'unreliable in closely related germplasm). Requires --king.')
    parser.add_argument('--error-rate', type=float, default=0.025,
                        help='Expected per-genotype error rate, used with --score-conflicts '
                             'to set the IBS0 noise ceiling (same formula as '
                             'refine_pedigree.py; default: 0.025), and (with --base-mendel/'
                             '--new-mendel) the Mendelian baseline rate')
    parser.add_argument('--base-mendel',
                        help='PLINK2 .fmendel for the base FAM\'s proposed pairs (per-family '
                             'Mendelian error counts) — an independent third signal that can '
                             'VETO (downgrade to tie_mendel_veto) a --score-conflicts '
                             'recommendation that favors a Mendelian-elevated pair. Requires '
                             '--score-conflicts and --mendel-bim/--mendel-n-variants.',
                        default=None)
    parser.add_argument('--new-mendel',
                        help='PLINK2 .fmendel for the new FAM\'s proposed pairs (optional, '
                             'same veto role as --base-mendel)',
                        default=None)
    parser.add_argument('--mendel-bim',
                        help='Corresponding .bim file, to count variants for Mendelian rate '
                             'normalization. Alternative to --mendel-n-variants.',
                        default=None)
    parser.add_argument('--mendel-n-variants', type=int, default=None,
                        help='Number of variants used for --mendel. Alternative to --mendel-bim.')
    parser.add_argument('--mendel-absolute-multiplier', type=float, default=2.0,
                        help='Multiple of the expected baseline Mendelian rate (2 x '
                             '--error-rate) above which a proposed pair is considered '
                             'Mendelian-elevated (default: 2.0, same as '
                             'analyze_mendel_errors.py)')
    parser.add_argument('--report-limit', type=int, default=200,
                        help='Max individuals to list per section in the report (default: 200)')
    args = parser.parse_args()

    if not os.path.exists(args.base_fam):
        sys.exit(f"Error: {args.base_fam} not found.")
    if not os.path.exists(args.new_fam):
        sys.exit(f"Error: {args.new_fam} not found.")
    if args.score_conflicts and not args.king:
        sys.exit("Error: --score-conflicts requires --king (scoring needs kinship/IBS0 data).")
    if args.auto_resolve and not args.score_conflicts:
        sys.exit("Error: --auto-resolve requires --score-conflicts (and therefore --king).")
    if (args.base_mendel or args.new_mendel) and not args.score_conflicts:
        sys.exit("Error: --base-mendel/--new-mendel requires --score-conflicts (the Mendelian "
                 "veto only applies to a kinship-based recommendation).")
    if (args.base_mendel or args.new_mendel) and not (args.mendel_bim or args.mendel_n_variants is not None):
        sys.exit("Error: --base-mendel/--new-mendel requires --mendel-bim or --mendel-n-variants.")
    if args.mendel_bim and args.mendel_n_variants is not None:
        sys.exit("Error: specify either --mendel-bim or --mendel-n-variants, not both.")

    out_prefix = args.output or f"{args.base_fam.rsplit('.fam', 1)[0]}_merged"
    out_fam_path = f"{out_prefix}.fam"
    out_report_path = f"{out_prefix}_merge_report.txt"
    out_conflicts_path = f"{out_prefix}_conflicts.tsv"

    base_trio_path = args.base_trio or default_trio_path(args.base_fam)
    new_trio_path = args.new_trio or default_trio_path(args.new_fam)

    print("=" * 70)
    print("VCFped FAM Merge")
    print("=" * 70)
    print(f"  Base FAM   : {args.base_fam}")
    print(f"  New FAM    : {args.new_fam}")
    print(f"  On conflict: {'auto-resolve (per-conflict score)' if args.auto_resolve else args.on_conflict}")
    print(f"  Base trio  : {base_trio_path} ({'found' if os.path.exists(base_trio_path) else 'not found — conflict evidence will show NA'})")
    print(f"  New trio   : {new_trio_path} ({'found' if os.path.exists(new_trio_path) else 'not found — conflict evidence will show NA'})")
    if args.king:
        king_path = args.king if args.king.endswith('.kin0') else f"{args.king}.kin0"
        print(f"  KING file  : {king_path} ({'found' if os.path.exists(king_path) else 'not found — kinship/IBS0 will show NA'})")
    else:
        print("  KING file  : not provided — conflict table will have no kinship/IBS0 evidence")
    if args.score_conflicts:
        print(f"  Scoring    : enabled (error rate {args.error_rate:.4%})")
    print()

    base_fam, base_order = load_fam(args.base_fam)
    new_fam, _ = load_fam(args.new_fam)
    print(f"  {len(base_fam)} individuals in base FAM")
    print(f"  {len(new_fam)} individuals in new FAM")
    print()

    # --auto-resolve needs KING + scoring thresholds ready DURING merge()
    # itself (the resolve_conflict callback below), before we even know
    # whether there will be any conflicts — so load eagerly whenever
    # --score-conflicts is requested, rather than only if conflicts exist.
    king_lookup = {}
    ibs0_ceiling = p25 = p75 = p95 = None
    base_mendel_lookup = new_mendel_lookup = {}
    mendel_n_variants = None
    if args.score_conflicts:
        king_lookup = load_king_lookup(args.king)
        ibs0_ceiling = ibs0_threshold_for_error_rate(args.error_rate)
        p25, p75, p95 = compute_kinship_percentiles(king_lookup)
        print(f"  Cohort kinship percentiles: p25={p25}, p75={p75}, p95={p95}")
        print(f"  IBS0 noise ceiling: {ibs0_ceiling:.4f}")
        if args.base_mendel or args.new_mendel:
            mendel_n_variants = (args.mendel_n_variants if args.mendel_n_variants is not None
                                 else count_variants(args.mendel_bim))
            base_mendel_lookup = load_fmendel_lookup(args.base_mendel)
            new_mendel_lookup = load_fmendel_lookup(args.new_mendel)
            print(f"  Mendel veto: base={len(base_mendel_lookup)} families, "
                  f"new={len(new_mendel_lookup)} families, n_variants={mendel_n_variants}, "
                  f"threshold={args.mendel_absolute_multiplier}x baseline")
        if args.auto_resolve:
            print("  --auto-resolve active: --on-conflict is ignored; each conflict is "
                  "resolved individually (prefer_new -> new, otherwise -> base).")
        print()

    if args.auto_resolve:
        def resolve_conflict(iid, bp, bm, np_, nm):
            rec = score_conflict_kinship(iid, bp, bm, np_, nm, king_lookup, ibs0_ceiling, p25, p75,
                                         base_mendel_lookup, new_mendel_lookup, mendel_n_variants,
                                         args.error_rate, args.mendel_absolute_multiplier)
            return 'new' if rec == 'prefer_new' else 'base'
    else:
        resolve_conflict = fixed_resolver('new' if args.on_conflict == 'keep-new' else 'base')

    merged_fam, merged_order, stats = merge(base_fam, base_order, new_fam, resolve_conflict)
    write_fam(out_fam_path, merged_fam, merged_order)

    kept_counts = Counter(c[5] for c in stats['conflicts'])
    print("Summary:")
    print(f"  Founders given new parents : {len(stats['parents_added'])}")
    if args.auto_resolve:
        print(f"  Conflicts (auto-resolved)  : {len(stats['conflicts'])} "
              f"({kept_counts.get('new', 0)} kept new, {kept_counts.get('base', 0)} kept base)")
    else:
        print(f"  Conflicts ({args.on_conflict} kept)  : {len(stats['conflicts'])}")
    print(f"  New individuals added      : {len(stats['new_individuals'])}")
    print(f"  Already matched (no-op)    : {len(stats['unchanged'])}")
    print(f"  Total individuals in merged FAM: {len(merged_fam)}")
    print()
    print(f"  Written: {out_fam_path}")

    conflict_rows = []
    if stats['conflicts']:
        base_trio_lookup = load_trio_lookup(base_trio_path)
        new_trio_lookup = load_trio_lookup(new_trio_path)
        if not args.score_conflicts:
            king_lookup = load_king_lookup(args.king)  # not loaded yet on this path

        conflict_rows = build_conflict_table(
            stats['conflicts'], base_trio_lookup, new_trio_lookup, king_lookup,
            score=args.score_conflicts, ibs0_ceiling=ibs0_ceiling, p25=p25, p75=p75,
            base_mendel_lookup=base_mendel_lookup, new_mendel_lookup=new_mendel_lookup,
            mendel_n_variants=mendel_n_variants, error_rate=args.error_rate,
            mendel_absolute_multiplier=args.mendel_absolute_multiplier)
        write_conflict_tsv(out_conflicts_path, conflict_rows)
        print(f"  Written: {out_conflicts_path}")

        if args.score_conflicts:
            rec_counts = Counter(r['recommendation'] for r in conflict_rows)
            print("Scoring summary:")
            print(f"  prefer_new       : {rec_counts.get('prefer_new', 0)}")
            print(f"  prefer_base      : {rec_counts.get('prefer_base', 0)}")
            print(f"  tie_new_higher   : {rec_counts.get('tie_new_higher', 0)}")
            print(f"  tie              : {rec_counts.get('tie', 0)}")
            print(f"  tie_mendel_veto  : {rec_counts.get('tie_mendel_veto', 0)}")

    conflict_by_child = {r['child']: r for r in conflict_rows}

    with open(out_report_path, 'w') as rpt:
        rpt.write("VCFped FAM Merge Report\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"Base FAM: {args.base_fam}\n")
        rpt.write(f"New FAM : {args.new_fam}\n")
        rpt.write(f"On conflict: {'auto-resolve (per-conflict score)' if args.auto_resolve else args.on_conflict}\n\n")

        rpt.write("Summary\n" + "-" * 40 + "\n")
        rpt.write(f"  Founders given new parents : {len(stats['parents_added'])}\n")
        rpt.write(f"  Conflicts                  : {len(stats['conflicts'])}\n")
        rpt.write(f"  New individuals added      : {len(stats['new_individuals'])}\n")
        rpt.write(f"  Already matched (no-op)    : {len(stats['unchanged'])}\n")
        if args.score_conflicts and conflict_rows:
            rec_counts = Counter(r['recommendation'] for r in conflict_rows)
            rpt.write(f"  Conflict scoring (error rate {args.error_rate:.4%}, "
                      f"IBS0 ceiling {ibs0_ceiling:.4f}, cohort p25={p25}/p75={p75}):\n")
            rpt.write(f"    prefer_new       : {rec_counts.get('prefer_new', 0)}\n")
            rpt.write(f"    prefer_base      : {rec_counts.get('prefer_base', 0)}\n")
            rpt.write(f"    tie_new_higher   : {rec_counts.get('tie_new_higher', 0)}\n")
            rpt.write(f"    tie              : {rec_counts.get('tie', 0)}\n")
            if base_mendel_lookup or new_mendel_lookup:
                rpt.write(f"    tie_mendel_veto  : {rec_counts.get('tie_mendel_veto', 0)} "
                          f"(Mendelian error rate vetoed the kinship-favored side, "
                          f"{args.mendel_absolute_multiplier}x baseline threshold)\n")
        rpt.write("\n")

        limit = args.report_limit

        rpt.write("Founders given new parents\n" + "-" * 40 + "\n")
        for iid in stats['parents_added'][:limit]:
            d = merged_fam[iid]
            rpt.write(f"  {iid}: pat={d['pat']}, mat={d['mat']}\n")
        if len(stats['parents_added']) > limit:
            rpt.write(f"  ... and {len(stats['parents_added']) - limit} more\n")
        rpt.write("\n")

        rpt.write("Conflicts (individual already had different parents)\n" + "-" * 40 + "\n")
        if stats['conflicts']:
            rpt.write(f"Full evidence (KING kinship/IBS0 for both options) in {out_conflicts_path}\n\n")
        for iid, bp, bm, np_, nm, kept in stats['conflicts'][:limit]:
            r = conflict_by_child.get(iid, {})
            rpt.write(f"  {iid}:\n")
            rpt.write(f"    base=({bp},{bm}) verdict={r.get('base_verdict','NA')} "
                      f"Perc={r.get('base_perc','NA')} T1={r.get('base_test1','NA')} "
                      f"kinship=({r.get('base_kinship_p1','NA')},{r.get('base_kinship_p2','NA')}) "
                      f"ibs0=({r.get('base_ibs0_p1','NA')},{r.get('base_ibs0_p2','NA')})\n")
            rpt.write(f"    new =({np_},{nm}) verdict={r.get('new_verdict','NA')} "
                      f"Perc={r.get('new_perc','NA')} T1={r.get('new_test1','NA')} "
                      f"kinship=({r.get('new_kinship_p1','NA')},{r.get('new_kinship_p2','NA')}) "
                      f"ibs0=({r.get('new_ibs0_p1','NA')},{r.get('new_ibs0_p2','NA')})\n")
            if args.score_conflicts:
                rpt.write(f"    score: base={r.get('base_overall','NA')} new={r.get('new_overall','NA')} "
                          f"-> {r.get('recommendation','NA')}\n")
                if r.get('base_mendel_rate', 'NA') != 'NA' or r.get('new_mendel_rate', 'NA') != 'NA':
                    def _fmt_rate(x):
                        return 'NA' if x == 'NA' else f"{x:.2%}"
                    rpt.write(f"    mendel: base_rate={_fmt_rate(r.get('base_mendel_rate','NA'))} "
                              f"new_rate={_fmt_rate(r.get('new_mendel_rate','NA'))}\n")
            rpt.write(f"    -> kept {kept}\n")
        if len(stats['conflicts']) > limit:
            rpt.write(f"  ... and {len(stats['conflicts']) - limit} more (full list in {out_conflicts_path})\n")
        rpt.write("\n")

        if args.score_conflicts:
            flagged = [r for r in conflict_rows
                      if r['recommendation'] in ('prefer_new', 'tie_new_higher', 'tie_mendel_veto')]
            rpt.write("Flagged for review (--score-conflicts: new evidence >= base, "
                      "or a Mendelian veto was triggered)\n" + "-" * 40 + "\n")
            if not flagged:
                rpt.write("  None — base evidence was at least as strong as new for every conflict.\n")
            for r in flagged[:limit]:
                if r['recommendation'] == 'tie_mendel_veto':
                    rpt.write(f"  {r['child']}  [{r['recommendation']}]  base={r['base_overall']} "
                              f"new={r['new_overall']} — Mendelian error rate vetoed the "
                              f"kinship-favored side; treat as a tie, review manually\n")
                else:
                    rpt.write(f"  {r['child']}  [{r['recommendation']}]  base={r['base_overall']} "
                              f"new={r['new_overall']}\n")
                rpt.write(f"    base: {r['base_p1']} + {r['base_p2']} (kept)\n")
                rpt.write(f"    new : {r['new_p1']} + {r['new_p2']} — consider --on-conflict keep-new for this child\n")
            if len(flagged) > limit:
                rpt.write(f"  ... and {len(flagged) - limit} more (full list in {out_conflicts_path})\n")
            rpt.write("\n")

        rpt.write("New individuals (not previously in base FAM)\n" + "-" * 40 + "\n")
        for iid in stats['new_individuals'][:limit]:
            d = merged_fam[iid]
            rpt.write(f"  {iid}: pat={d['pat']}, mat={d['mat']}\n")
        if len(stats['new_individuals']) > limit:
            rpt.write(f"  ... and {len(stats['new_individuals']) - limit} more\n")

    print(f"  Written: {out_report_path}")
    print()
    print("Next step: python utilities/extend_pedigree.py "
          f"{out_fam_path} <king_prefix>.kin0 --ibd <prefix>_stat_ibd.ibd.gz")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
