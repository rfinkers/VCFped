#!/usr/bin/env python3
"""
Refine vcfped parent assignments using PLINK2 KING kinship evidence.

BACKGROUND
----------
The vcfped two-stage workflow (MZ purge → trio analysis) can produce suboptimal
parent assignments in closely related breeding pools:

  1. MZ-twin substitution problem:
     purge_mz_twins.py keeps one representative per MZ group (e.g. A1 from
     {A1, A2, A3}).  If A2 is the actual biological parent of a child C, the
     trio analysis assigns A1 as parent (correct genotype, wrong individual).
     For breeding records this distinction matters.

     CAVEAT: this only works if the MZ group is accurate. In densely
     inbred/repeatedly-selfed germplasm, vcfped's own pairwise "MZ twins"
     verdict (--step1-pair) can be inflated by inbreeding-induced allele-
     sharing -- genuinely distinct-but-closely-related samples get called
     "MZ twins" too, producing implausibly large groups (confirmed on a real
     cohort: ~15x more raw "MZ twins" pairs than hap-IBD-coverage validated
     true duplicates). The mz_swap substitution trusts every group member as
     interchangeable, so a contaminated group can silently substitute in a
     genuinely DIFFERENT individual, not the same one under another label.
     Prefer --confirmed-duplicates (extend_pedigree.py's stricter,
     hap-IBD-coverage-validated <prefix>_confirmed_duplicates.tsv) when
     available.

  2. Unchecked candidates:
     vcfped tests ALL N³/6 triples, but filter_trios.py keeps only the
     highest-confidence call per child.  Relatives that scored slightly lower
     may still be biologically plausible parents (e.g. half-sibs of the chosen
     parent who share the relevant alleles).

  3. Inverted/generational trios as evidence:
     A trio flagged as "Inverted/generational" (T1 high, T2 low) means the
     "child" may itself be a parent of one of the other two — or all three
     span two generations.  These complement the regular trios.

STRATEGY
--------
This utility uses the PLINK2 KING kinship matrix as the primary evidence
source.  KING requires no additional VCF processing and produces two metrics
that directly address the above problems:

  KINSHIP > 0.177 AND IBS0 < 0.05  →  parent-offspring signal
  KINSHIP > 0.354                   →  MZ twin / duplicate

Workflow:
  1. Load KING kinship matrix, current trio assignments, MZ group info.
  2. For every child in current .trio:
       a. Confirm both assigned parents have PO signal in KING.
       b. Check if any MZ-group member of either parent also has PO signal.
       c. Search ALL KING pairs for unassigned PO candidates to this child.
  3. For inverted trios (T1 high, T2 low):
       a. Check KING to determine generational direction.
       b. Reclassify if clear evidence supports it.
  4. Report and write refined .trio + .fam.

When to use additional strategies
  • If KING screening finds many unresolved ambiguities → re-run vcfped.py
    (NOT vcfped_optimized.py) with --all on the candidate subset (most
    accurate T1/T2 scores). vcfped_optimized.py's --all path has a known,
    partially-confirmed scoring discrepancy for low-confidence rows (Not
    trio/Other) -- see its _is_better_pair_result()/_is_better_trio_result()
    docstrings. The subset is small enough that vcfped.py's lack of
    multiprocessing/JIT costs nothing here.
  • If hap-IBD output is available → use IBS segment patterns to distinguish
    PO (no Z2 segments) from full sibling (has Z2 segments).
  • For MZ substitution decisions in breeding programs → external records
    (birth dates, tags, breeding logs) take priority over genotype evidence.

INPUT
-----
  <trio_prefix>.trio          — filtered trio file (filter_trios.py output)
  <king_prefix>.kin0          — PLINK2 --make-king-table output
  <step1_prefix>.pair         — step 1 pairwise file for MZ group info (optional).
                                 Mutually exclusive with --confirmed-duplicates.
  --confirmed-duplicates      — alternative, stricter MZ group source: a two-column
                                 ID1/ID2 pair list, e.g. extend_pedigree.py's
                                 <prefix>_confirmed_duplicates.tsv (optional; see
                                 CAVEAT above). Mutually exclusive with --step1-pair.

OUTPUT
------
  <out_prefix>_refined.trio   — updated assignments (same format as input)
  <out_prefix>_refined.fam    — updated PLINK/BEAGLE family file
  <out_prefix>_refinement_report.txt  — full audit trail

Usage:
  python refine_pedigree.py step3_named_filtered king_results --step1-pair step1_named
  python refine_pedigree.py step3_named_filtered king_results \\
      --confirmed-duplicates ext_out_confirmed_duplicates.tsv
  python refine_pedigree.py step3_named_filtered king_results --include-inverted
  python refine_pedigree.py step3_named_filtered king_results --kinship-min 0.20
"""

import sys
import os
import csv
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# KING kinship constants — read carefully before using as hard thresholds
# ---------------------------------------------------------------------------
#
# IMPORTANT FOR RELATED / PLANT BREEDING GERMPLASM
# -------------------------------------------------
# The values below are the theoretical expectation for OUTBRED populations:
#   PO expected kinship ≈ 0.25  (range 0.177–0.354 = 1st degree)
#   GP expected kinship ≈ 0.125 (range 0.0884–0.177 = 2nd degree)
#
# In closely related breeding pools (F2, RILs, backcrosses, elite panels),
# background kinship can be 0.10–0.30 for all pairs. This makes fixed
# thresholds unreliable:
#   - Many non-PO pairs will have KINSHIP > 0.177 due to background relatedness
#   - True PO pairs still cluster at the TOP of the kinship distribution
#
# WHICH METRICS REMAIN VALID IN RELATED GERMPLASM:
#   IBS0 ≈ 0     — most robust: a parent must share ≥1 allele with their child at
#                  EVERY variant. IBS0 (0 alleles IBS) is physically impossible for
#                  true PO regardless of background relatedness or inbreeding.
#   Relative kinship ranking — true PO pairs should still rank highest within the
#                  cohort, even if the absolute value is inflated.
#   Phased IBD block coverage — gold standard: the child's full genome should tile
#                  onto exactly two parental haplotypes. False parents leave gaps.
#
# USE FIXED THRESHOLDS ONLY AS INITIAL SCREENING, not as definitive confirmation.
# The highest-certainty confirmation is pedigree-aware phasing + genome coverage.

KING_PO_KINSHIP_OUTBRED = 0.177   # 1st-degree lower bound in OUTBRED populations
KING_MZ_KINSHIP         = 0.354   # MZ twin / duplicate (valid in any population)
KING_PO_IBS0_MAX        = 0.05    # PO IBS0 threshold — see ibs0_threshold_for_error_rate()
KING_PO_IBS0_STRICT     = 0.02    # Stricter IBS0 for high confidence


def ibs0_threshold_for_error_rate(error_rate):
    """
    Compute the maximum expected IBS0 for a TRUE parent-offspring pair
    given a per-genotype error rate.

    DERIVATION:
    IBS0 occurs when sample A is 0/0 (AA) and sample B is 1/1 (BB) (or vice versa).
    For a true PO pair this is impossible with correct genotypes.
    With error rate ε per allele call:
      - A true heterozygote (AB) can be miscalled as BB with probability ε
      - If parent is miscalled as BB and child is correctly called as AA → false IBS0
      - Expected false IBS0 ≈ 2ε(1-ε) ≈ 2ε for small ε

    TYPICAL VALUES:
      WGS 30×  (Illumina): ε ≈ 0.001 → IBS0 threshold ≈ 0.002
      WGS 10×              ε ≈ 0.005 → IBS0 threshold ≈ 0.010
      GBS / RADseq:        ε ≈ 0.010 → IBS0 threshold ≈ 0.020
      SNP array:           ε ≈ 0.002 → IBS0 threshold ≈ 0.004
      Low-quality data:    ε ≈ 0.025 → IBS0 threshold ≈ 0.050 (our default)

    The default KING_PO_IBS0_MAX = 0.05 is conservative — it tolerates up to
    2.5% genotype error, which is generous for most WGS data.
    Tighten it (e.g. 0.01) for high-quality data to reduce false negatives.
    """
    return min(0.10, 2 * error_rate * 1.5)  # 1.5× safety margin


class _KingDict(dict):
    """Dict subclass carrying a per-individual neighbor index for O(1) lookups.

    neighbors: {id → list of (other_id, record_dict)} — built once at load time
    so find_po_candidates() can skip the O(N_pairs) full-dict scan.
    """
    def __init__(self, pair_data: dict, neighbors: dict):
        super().__init__(pair_data)
        self.neighbors = neighbors


def load_king(king_kin0_path, required_min_kinship=None):
    """
    Parse PLINK2 --make-king-table output (.kin0 file).

    IMPORTANT — default filter warning:
    PLINK2 silently drops pairs with KINSHIP < 0.0442 by default.
    Pass required_min_kinship to warn if the file appears to have been
    filtered too aggressively for the intended use.

    Returns:
        king: _KingDict (id1, id2) → {kinship, ibs0, hethet}
              also carries king.neighbors for O(1) per-individual lookup
        all_ids: set of all sample IDs seen
    """
    king = {}
    all_ids = set()
    min_kinship_seen = float('inf')
    try:
        with open(king_kin0_path, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fieldnames = reader.fieldnames or []
            id1_col = next((c for c in fieldnames if 'ID1' in c and 'FID' not in c), None)
            id2_col = next((c for c in fieldnames if 'ID2' in c and 'FID' not in c), None)
            if not id1_col or not id2_col:
                sys.exit(f"Error: cannot find ID1/ID2 columns in {king_kin0_path}.\n"
                         f"  Found columns: {fieldnames}\n"
                         f"  Expected PLINK2 --make-king-table output.\n"
                         f"  Generate with: plink2 --make-king-table "
                         f"--king-table-filter 0.044 ...")
            for row in reader:
                id1 = row[id1_col].strip()
                id2 = row[id2_col].strip()
                try:
                    kinship = float(row.get('KINSHIP', row.get('Kinship', 0)))
                    ibs0    = float(row.get('IBS0', 0))
                    hethet  = float(row.get('HETHET', 0))
                except ValueError:
                    continue
                key = (id1, id2)
                king[key] = {'kinship': kinship, 'ibs0': ibs0, 'hethet': hethet}
                all_ids.update([id1, id2])
                if kinship < min_kinship_seen:
                    min_kinship_seen = kinship
    except FileNotFoundError:
        sys.exit(f"Error: {king_kin0_path} not found.")

    if not king:
        sys.exit(f"Error: {king_kin0_path} contains no pairs.\n"
                 f"  Was it generated with --make-king-table? "
                 f"Check with: head {king_kin0_path}")

    # Warn if the effective filter appears too aggressive for the intended use
    if required_min_kinship is not None and min_kinship_seen > required_min_kinship * 1.5:
        print(f"  WARNING: lowest kinship in {king_kin0_path} is "
              f"{min_kinship_seen:.4f}, but this utility needs pairs down to "
              f"{required_min_kinship:.4f}.")
        print(f"  Re-run PLINK2 with: --king-table-filter {required_min_kinship:.4f}")
        print(f"  Pairs below the effective filter in this file will be silently "
              f"treated as 'not in KING output', which may cause missed candidates.")

    # Build per-individual neighbor index so find_po_candidates() is O(degree)
    # instead of O(N_pairs).  Stored as king.neighbors on the returned dict.
    neighbors = defaultdict(list)
    for (id1, id2), rec in king.items():
        neighbors[id1].append((id2, rec))
        neighbors[id2].append((id1, rec))

    return _KingDict(king, dict(neighbors)), all_ids


def get_king(king, id1, id2):
    """Look up KING metrics for a pair in either order; return None if absent."""
    return king.get((id1, id2)) or king.get((id2, id1))


def is_po_signal(king, id1, id2, kinship_min=None, ibs0_max=KING_PO_IBS0_MAX,
                  king_percentiles=None):
    """
    Test for parent-offspring KING signature.

    Priority:
      1. IBS0 ≈ 0  —  necessary and sufficient in ANY population.
         IBS0 > ibs0_max rules out PO definitively.
      2. Kinship rank  —  in related germplasm, use percentile rank rather
         than an absolute threshold. A true PO pair should be near the top
         of the kinship distribution.
      3. Absolute kinship  —  only checked if kinship_min is given AND
         king_percentiles is None (outbred-only mode).

    kinship_min:    optional absolute lower bound; unreliable in related pools
    king_percentiles: dict from build_king_percentiles(); if provided, absolute
                    kinship thresholds are replaced by percentile rank check
    """
    rec = get_king(king, id1, id2)
    if rec is None:
        return False

    # IBS0 check: the only threshold that holds in ALL populations
    if rec['ibs0'] > ibs0_max:
        return False

    # Kinship must be positive (no kinship data at all → not PO)
    if rec['kinship'] <= 0:
        return False

    # In related germplasm: use rank rather than absolute value
    if king_percentiles is not None:
        pct = king_percentiles.get((id1, id2)) or king_percentiles.get((id2, id1))
        if pct is not None and pct < 85:
            # Below 85th percentile in a related pool → unlikely PO
            return False
        return True

    # Outbred mode: absolute kinship threshold
    if kinship_min is not None:
        return rec['kinship'] >= kinship_min

    # Default: IBS0 filter passed and kinship > 0 is sufficient as a flag
    return True


def build_king_percentiles(king):
    """
    Compute the cohort-relative kinship percentile for every pair in KING.
    Returns dict (id1, id2) → percentile (0–100).

    In a related breeding pool, a true PO pair should be near the top of
    the kinship distribution even if the absolute value is inflated.
    Using percentile rank makes the analysis population-agnostic.
    """
    if not king:
        return {}
    pairs = sorted(king.items(), key=lambda x: x[1]['kinship'])
    n = len(pairs)
    pct = {}
    for i, (key, _) in enumerate(pairs):
        p = round(100 * i / max(n - 1, 1))
        pct[key] = p
    return pct


def is_mz_signal(king, id1, id2):
    """Return True if the pair looks like MZ twins / duplicates.
    KINSHIP > 0.354 holds in any population (identical genotypes).
    """
    rec = get_king(king, id1, id2)
    return rec is not None and rec['kinship'] >= KING_MZ_KINSHIP


def kinship_of(king, id1, id2):
    rec = get_king(king, id1, id2)
    return rec['kinship'] if rec else None


def ibs0_of(king, id1, id2):
    rec = get_king(king, id1, id2)
    return rec['ibs0'] if rec else None


def king_evidence_summary(rec, king_pct=None):
    """
    Return a human-readable evidence string for a KING pair record.
    Describes strength relative to the cohort, not absolute thresholds.
    """
    if rec is None:
        return "not in KING output"
    k, i = rec['kinship'], rec['ibs0']
    pct_str = f", cohort rank={king_pct:.0f}th pct" if king_pct is not None else ""
    ibs0_verdict = ("IBS0≈0 ✓ (PO consistent)" if i <= KING_PO_IBS0_MAX
                    else f"IBS0={i:.3f} ✗ (NOT PO — impossible for true parent-offspring)")
    return f"kinship={k:.3f}{pct_str}, {ibs0_verdict}"


# ---------------------------------------------------------------------------
# MZ group parsing from step1 .pair file
# ---------------------------------------------------------------------------

def _group_pairs_transitively(pairs):
    """
    Union-find transitive closure over a list of (a, b) pairs.
    Returns: mz_of[sample] = frozenset of all group members (including self).
    Shared by load_mz_groups() (raw vcfped .pair verdict) and
    load_mz_groups_from_confirmed_duplicates() (stricter, hap-IBD-coverage-
    validated ground truth) -- same grouping logic, different input sources.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    groups = defaultdict(set)
    for x in parent:
        groups[find(x)].add(x)

    mz_of = {}
    for members in groups.values():
        fs = frozenset(members)
        for m in members:
            mz_of[m] = fs

    return mz_of


def load_mz_groups(pair_path):
    """
    Parse MZ twin pairs from a vcfped .pair file.
    Returns: mz_of[sample] = frozenset of all MZ group members (including self).

    CAVEAT: vcfped's own pairwise "MZ twins" verdict is a raw statistical call
    (allele-sharing based) -- in densely inbred/repeatedly-selfed germplasm,
    inbreeding-induced allele-sharing inflation can call genuinely distinct-
    but-closely-related samples "MZ twins" too, producing implausibly large
    groups (confirmed on a real cohort: ~15x more raw "MZ twins" pairs than
    hap-IBD-coverage validated true duplicates). check_mz_swap() below trusts
    every member of a group as an interchangeable substitute parent, so a
    contaminated group can silently substitute in a genuinely DIFFERENT
    individual, not the same one under another label. Prefer
    load_mz_groups_from_confirmed_duplicates() when a stricter file is
    available (see --confirmed-duplicates).
    """
    if not pair_path or not os.path.exists(pair_path + '.pair'):
        return {}

    pairs = []
    try:
        with open(pair_path + '.pair', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row.get('Verdict', '').strip() == 'MZ twins':
                    parts = row.get('Pair', '').split(',')
                    if len(parts) == 2:
                        pairs.append((parts[0].strip(), parts[1].strip()))
    except FileNotFoundError:
        return {}

    return _group_pairs_transitively(pairs)


def load_mz_groups_from_confirmed_duplicates(path):
    """
    Parse a stricter ground-truth duplicate-pairs file -- e.g.
    extend_pedigree.py's <prefix>_confirmed_duplicates.tsv (hap-IBD-coverage
    validated true duplicates, only written when that script was run with
    --ibd) -- instead of vcfped's raw pairwise "MZ twins" verdict. Expects a
    header row with ID1/ID2 columns (name-matched, same convention as
    calibrate_error_rate.py's load_confirmed_duplicate_pairs()).
    Returns: mz_of[sample] = frozenset of all group members (including self).
    """
    if not path or not os.path.exists(path):
        return {}

    pairs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        id1_col = next((c for c in fieldnames if c.upper() == 'ID1'), None)
        id2_col = next((c for c in fieldnames if c.upper() == 'ID2'), None)
        if not id1_col or not id2_col:
            sys.exit(f"Error: expected ID1/ID2 columns in {path}. Found: {fieldnames}")
        for row in reader:
            pairs.append((row[id1_col].strip(), row[id2_col].strip()))

    return _group_pairs_transitively(pairs)


# ---------------------------------------------------------------------------
# Genome coverage from hap-IBD (gold-standard PO confirmation)
# ---------------------------------------------------------------------------

import gzip as _gzip

class _IbdIndex(dict):
    """Dict subclass carrying per-child total IBD bp for O(1) genome-proxy lookups.

    child_total_bp: {id → total bp across all IBD partners} — built once at load
    time so compute_trio_genome_coverage() can avoid the O(N_pairs) full-dict scan
    that it previously used to estimate the child's segmented-genome size.
    """
    def __init__(self, pair_segs: dict, child_total_bp: dict):
        super().__init__(pair_segs)
        self.child_total_bp = child_total_bp


def load_ibd_index(ibd_gz_path, min_lod=3.0):
    """
    Build a per-pair segment index from hap-IBD output.
    Returns _IbdIndex: (id1, id2) → list of (chrom, start, end).
    Also carries .child_total_bp for fast genome-size proxy computation.
    Segments below min_lod are excluded.
    """
    pair_segs = defaultdict(list)
    child_total_bp = defaultdict(int)
    if not ibd_gz_path or not os.path.exists(ibd_gz_path):
        return _IbdIndex({}, {})
    opener = _gzip.open if ibd_gz_path.endswith('.gz') else open
    try:
        with opener(ibd_gz_path, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                id1, _, id2, _, chrom, start, end, lod = parts[:8]
                try:
                    if float(lod) < min_lod:
                        continue
                    s, e = int(start), int(end)
                    key = tuple(sorted([id1, id2]))
                    pair_segs[key].append((chrom, s, e))
                    seg_len = e - s
                    child_total_bp[id1] += seg_len
                    child_total_bp[id2] += seg_len
                except ValueError:
                    continue
    except Exception as e:
        print(f"  Warning: could not read {ibd_gz_path}: {e}")
    return _IbdIndex(dict(pair_segs), dict(child_total_bp))


def compute_trio_genome_coverage(child, p1, p2, ibd_index):
    """
    Compute what fraction of the child's genome is covered by IBD segments
    from BOTH proposed parents combined.

    This is the most reliable PO confirmation for related germplasm:
      True parents   → ~100% coverage: every genomic position in the child
                       is IBD with one of the two parents (Mendelian partition)
      False parents  → < 100%: gaps exist where neither parent contributed
                       the observed allele

    Returns a dict with:
      coverage_frac   : fraction of IBD-segmented genome covered (0–1)
      p1_bp, p2_bp    : total IBD bp from P1, P2
      combined_bp     : union of P1 + P2 coverage (bp)
      total_segmented : all IBD bp in child's segments across all pairs
      balance         : how evenly split the coverage is (close to 0.5 = good)
      interpretation  : text summary
    """
    key_p1 = tuple(sorted([child, p1]))
    key_p2 = tuple(sorted([child, p2]))

    segs_p1 = ibd_index.get(key_p1, [])
    segs_p2 = ibd_index.get(key_p2, [])

    if not segs_p1 and not segs_p2:
        return {
            'coverage_frac': None,
            'p1_bp': 0, 'p2_bp': 0, 'combined_bp': 0,
            'total_segmented': 0, 'balance': None,
            'interpretation': 'No IBD segments found — VCF not yet phased or '
                              'hap-IBD not run. Run run_beagle.sh first.',
        }

    # Per-chromosome union
    all_chroms = set(c for c, _, _ in segs_p1 + segs_p2)
    combined_bp = 0
    p1_bp = sum(e - s for _, s, e in segs_p1)
    p2_bp = sum(e - s for _, s, e in segs_p2)

    for chrom in all_chroms:
        # Merge overlapping intervals from both parents on this chromosome
        intervals = [(s, e) for c, s, e in segs_p1 + segs_p2 if c == chrom]
        intervals.sort()
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append([s, e])
        combined_bp += sum(e - s for s, e in merged)

    # Total IBD bp in child with ANYONE (denominator proxy for genome size).
    # Use precomputed index when available (O(1)); fall back to O(N) scan for
    # plain dicts passed from external code.
    child_total_bp = getattr(ibd_index, 'child_total_bp', None)
    if child_total_bp is not None:
        total_segmented = child_total_bp.get(child, 0) or combined_bp
    else:
        all_child_segs = [(s, e) for key, segs in ibd_index.items()
                          for c, s, e in segs if child in key]
        total_segmented = sum(e - s for s, e in all_child_segs) if all_child_segs else combined_bp

    coverage_frac = combined_bp / total_segmented if total_segmented > 0 else 0
    balance = p1_bp / (p1_bp + p2_bp) if (p1_bp + p2_bp) > 0 else None

    if coverage_frac is None:
        interp = 'Cannot compute — no segments'
    elif coverage_frac >= 0.90:
        bal_note = (f", P1/P2 split={balance:.2f}/{ 1-balance:.2f} "
                    f"({'balanced ✓' if 0.35 < balance < 0.65 else 'imbalanced ⚠'})"
                    if balance is not None else "")
        interp = f"HIGH CONFIDENCE — {coverage_frac:.1%} genome covered by both parents{bal_note}"
    elif coverage_frac >= 0.70:
        interp = (f"MODERATE — {coverage_frac:.1%} covered. May be a correct trio with "
                  f"phasing gaps, or a false assignment with partial allele sharing.")
    else:
        interp = (f"LOW — only {coverage_frac:.1%} covered. Unlikely to be the true "
                  f"parent pair; check alternative candidates.")

    return {
        'coverage_frac': coverage_frac,
        'p1_bp': p1_bp,
        'p2_bp': p2_bp,
        'combined_bp': combined_bp,
        'total_segmented': total_segmented,
        'balance': balance,
        'interpretation': interp,
    }


# ---------------------------------------------------------------------------
# Trio file I/O
# ---------------------------------------------------------------------------

REQUIRED_TRIO_COLS = {'Triple', 'Pivot', 'Perc', 'Test1', 'Verdict'}


def load_trio_file(path, verdicts=('Regular trio',)):
    """Load a .trio file; return (fieldnames, all_rows, filtered_rows)."""
    if not os.path.exists(path):
        return None, [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)

    missing = REQUIRED_TRIO_COLS - set(fieldnames)
    if missing:
        sys.exit(f"Error: {path} missing columns: {sorted(missing)}")

    filtered = []
    for row in all_rows:
        if row.get('Verdict', '').strip() in verdicts:
            triple = [s.strip() for s in str(row.get('Triple', '')).split(',')]
            pivot  = str(row.get('Pivot', '')).strip()
            if len(triple) == 3 and pivot in triple:
                child = pivot
                parents = [s for s in triple if s != child]
                if len(parents) == 2:
                    row['_child']  = child
                    row['_p1']     = parents[0]
                    row['_p2']     = parents[1]
                    row['_perc']   = int(row.get('Perc', 0))
                    row['_t1']     = float(row.get('Test1', 0.0))
                    filtered.append(row)
    return fieldnames, all_rows, filtered


def write_trio_file(path, fieldnames, rows):
    """Write a .trio file (rows are raw dicts from csv.DictReader)."""
    # Strip internal _xxx keys before writing
    clean_fields = [f for f in fieldnames if not f.startswith('_')]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(clean_fields)
        for row in rows:
            writer.writerow([row.get(k, '') for k in clean_fields])
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# FAM file I/O
# ---------------------------------------------------------------------------

def load_fam(fam_path):
    """Return dict: iid → {fid, pat, mat, sex}."""
    fam = {}
    if not fam_path or not os.path.exists(fam_path):
        return fam
    with open(fam_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                fam[parts[1]] = {'fid': parts[0], 'pat': parts[2],
                                  'mat': parts[3], 'sex': parts[4]}
    return fam


def write_fam(path, fam):
    """Write updated .fam file."""
    with open(path, 'w') as f:
        for iid, d in sorted(fam.items()):
            f.write(f"{d['fid']}\t{iid}\t{d['pat']}\t{d['mat']}\t{d['sex']}\t-9\n")
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Core refinement logic
# ---------------------------------------------------------------------------

def find_po_candidates(target, king, all_ids, kinship_min=KING_PO_KINSHIP_OUTBRED,
                        ibs0_max=KING_PO_IBS0_MAX, exclude=None):
    """
    Find all individuals in KING with a PO signature relative to target.
    Returns sorted list of (individual, kinship, ibs0).
    Uses king.neighbors for O(degree) lookup when available; falls back to
    the O(N_pairs) full-dict scan for plain dicts.
    """
    exclude = exclude or set()
    results = []
    neighbors = getattr(king, 'neighbors', None)
    if neighbors is not None:
        for other, rec in neighbors.get(target, []):
            if other not in exclude and rec['kinship'] >= kinship_min and rec['ibs0'] <= ibs0_max:
                results.append((other, rec['kinship'], rec['ibs0']))
    else:
        for id1, id2 in king:
            other = None
            if id1 == target and id2 not in exclude:
                other = id2
            elif id2 == target and id1 not in exclude:
                other = id1
            if other is None:
                continue
            rec = king[(id1, id2)]
            if rec['kinship'] >= kinship_min and rec['ibs0'] <= ibs0_max:
                results.append((other, rec['kinship'], rec['ibs0']))
    return sorted(results, key=lambda x: (-x[1], x[2]))


def parents_are_compatible(p1, p2, king, ibs0_max=KING_PO_IBS0_MAX):
    """
    Co-parents of the same child should not themselves be parent-offspring.
    In related germplasm this check can produce false positives (two closely
    related individuals that look like PO due to background kinship).
    IBS0 is the decisive metric: truly PO pairs have IBS0 ≈ 0.

    ibs0_max should be the SAME calibrated ceiling used everywhere else (the
    question -- "is this pair statistically consistent with true PO?" -- is
    identical to what is_po_signal() answers, which check_inverted_direction()
    already threads the calibrated ibs0_max through for). Left hardcoded at
    the KING_PO_IBS0_MAX=0.05 default, at a much lower calibrated ceiling
    (e.g. 0.0008) this bar becomes far HARDER to clear than the real PO noise
    floor, so almost any pair -- unrelated or not -- can fail to exceed it in
    a cohort with elevated background relatedness, incorrectly rejecting
    candidates as "looks like PO to each other" for reasons unrelated to a
    genuine grandparent/parent conflict.
    """
    rec = get_king(king, p1, p2)
    if rec is None:
        return True
    return rec['ibs0'] > ibs0_max   # compatible if IBS0 is above the PO noise floor


def score_assignment(child, p1, p2, king, king_pct=None, ibd_index=None):
    """
    Return a numeric confidence score for the (p1, p2) → child assignment.

    Score components (all between 0 and 1, higher = better):
      IBS0 component (most reliable): 1 - mean(IBS0_p1, IBS0_p2) / 0.05
        — penalises any IBS0 > 0 heavily
      Kinship percentile: relative rank within cohort (population-agnostic)
      Genome coverage (if ibd_index provided): fraction of child's genome
        covered by both parents combined — gold standard in any population
    """
    i1 = ibs0_of(king, child, p1) or 0
    i2 = ibs0_of(king, child, p2) or 0
    ibs0_score = 1.0 - min(1.0, (i1 + i2) / (2 * KING_PO_IBS0_MAX))

    if king_pct is not None:
        pct1 = (king_pct.get((child, p1)) or king_pct.get((p1, child)) or 0) / 100
        pct2 = (king_pct.get((child, p2)) or king_pct.get((p2, child)) or 0) / 100
        kinship_score = (pct1 + pct2) / 2
    else:
        k1 = kinship_of(king, child, p1) or 0
        k2 = kinship_of(king, child, p2) or 0
        kinship_score = (k1 + k2) / 2

    coverage_score = 0.0
    if ibd_index is not None:
        cov = compute_trio_genome_coverage(child, p1, p2, ibd_index)
        if cov['coverage_frac'] is not None:
            coverage_score = cov['coverage_frac']

    if ibd_index is not None:
        # Weight: IBS0 40%, coverage 50%, kinship rank 10%
        return 0.40 * ibs0_score + 0.50 * coverage_score + 0.10 * kinship_score
    else:
        # Without IBD: IBS0 60%, kinship rank 40%
        return 0.60 * ibs0_score + 0.40 * kinship_score


# ---------------------------------------------------------------------------
# Refinement of a single child
# ---------------------------------------------------------------------------

class Assignment:
    def __init__(self, p1, p2, child, source, perc=0, t1=0.0,
                 k_p1=None, k_p2=None, ibs_p1=None, ibs_p2=None,
                 pct_p1=None, pct_p2=None, coverage=None, notes=None):
        self.p1 = p1
        self.p2 = p2
        self.child = child
        self.source = source
        self.perc = perc
        self.t1 = t1
        self.k_p1 = k_p1
        self.k_p2 = k_p2
        self.ibs_p1 = ibs_p1
        self.ibs_p2 = ibs_p2
        self.pct_p1 = pct_p1       # cohort kinship percentile child-p1
        self.pct_p2 = pct_p2
        self.coverage = coverage    # genome coverage dict from compute_trio_genome_coverage
        self.notes = notes or []

    def ibs0_passes(self):
        """IBS0 ≈ 0 for BOTH parents — the most reliable PO check."""
        p1_ok = (self.ibs_p1 is not None) and (self.ibs_p1 <= KING_PO_IBS0_MAX)
        p2_ok = (self.ibs_p2 is not None) and (self.ibs_p2 <= KING_PO_IBS0_MAX)
        return p1_ok, p2_ok

    def coverage_frac(self):
        if self.coverage:
            return self.coverage.get('coverage_frac')
        return None

    def confidence(self, ibs0_max=KING_PO_IBS0_MAX):
        """
        ibs0_max should be the SAME calibrated ceiling used for the pass/fail
        check (args.ibs0_max), not the hardcoded KING_PO_IBS0_MAX default.
        At a much lower calibrated ceiling (e.g. 0.0008 vs. the 0.05
        default), scaling ibs0_score against the wrong constant flattens it
        to near-1.0 for every near-zero-IBS0 candidate -- silently
        neutralizing "IBS0: most reliable, weight highest" and leaving
        kinship percentile as the real (much lower-weighted) discriminator.
        """
        # IBS0: most reliable, weight highest
        i1 = self.ibs_p1 if self.ibs_p1 is not None else ibs0_max
        i2 = self.ibs_p2 if self.ibs_p2 is not None else ibs0_max
        ibs0_score = 1.0 - min(1.0, (i1 + i2) / (2 * ibs0_max))

        # Genome coverage: gold standard
        cov = self.coverage_frac()
        cov_score = cov if cov is not None else 0.0

        # T1 from vcfped
        t1_score = self.t1 / 100.0

        # Kinship percentile (population-agnostic)
        pct1 = (self.pct_p1 or 0) / 100
        pct2 = (self.pct_p2 or 0) / 100
        pct_score = (pct1 + pct2) / 2

        if cov is not None:
            # IBD available: IBS0 30%, coverage 50%, T1 10%, pct 10%
            return 0.30 * ibs0_score + 0.50 * cov_score + 0.10 * t1_score + 0.10 * pct_score
        else:
            # No IBD: IBS0 50%, T1 30%, pct 20%
            return 0.50 * ibs0_score + 0.30 * t1_score + 0.20 * pct_score

    def po_confirmed(self, kinship_min=None, ibs0_max=KING_PO_IBS0_MAX):
        """
        Confirm PO status.  IBS0 is the primary metric.
        kinship_min is used only as a secondary indicator — do not treat
        failure to meet it as a definitive rejection in related germplasm.
        """
        p1_ibs_ok = (self.ibs_p1 is not None) and (self.ibs_p1 <= ibs0_max)
        p2_ibs_ok = (self.ibs_p2 is not None) and (self.ibs_p2 <= ibs0_max)
        return p1_ibs_ok, p2_ibs_ok

    def __repr__(self):
        cov = f", cov={self.coverage_frac():.0%}" if self.coverage_frac() is not None else ""
        return (f"Assignment({self.p1}+{self.p2}→{self.child}, "
                f"src={self.source}, T1={self.t1:.1f}%"
                f", IBS0={self.ibs_p1}/{self.ibs_p2}{cov})")


def refine_child(child, orig_p1, orig_p2, vcfped_row, king, mz_of, all_ids,
                  kinship_min, ibs0_max, king_pct=None, ibd_index=None):
    """
    Return (best_assignment, alternatives, issues) for one child.

    kinship_min is treated as a SOFT indicator, not a hard cutoff.
    IBS0 ≤ ibs0_max is the hard requirement for PO.
    king_pct enables population-agnostic ranking (recommended in related pools).
    ibd_index enables genome coverage computation (gold standard).
    """
    issues = []
    alternatives = []
    t1   = float(vcfped_row.get('Test1', 0.0))
    perc = int(vcfped_row.get('Perc', 0))

    def make_assign(p1, p2, source, notes=None):
        pct1 = (king_pct.get((child, p1)) or king_pct.get((p1, child))
                if king_pct else None)
        pct2 = (king_pct.get((child, p2)) or king_pct.get((p2, child))
                if king_pct else None)
        cov = compute_trio_genome_coverage(child, p1, p2, ibd_index) if ibd_index else None
        return Assignment(
            p1=p1, p2=p2, child=child, source=source, perc=perc, t1=t1,
            k_p1=kinship_of(king, child, p1),
            k_p2=kinship_of(king, child, p2),
            ibs_p1=ibs0_of(king, child, p1),
            ibs_p2=ibs0_of(king, child, p2),
            pct_p1=pct1, pct_p2=pct2,
            coverage=cov,
            notes=notes or [],
        )

    current = make_assign(orig_p1, orig_p2, 'original')
    p1_ibs_ok, p2_ibs_ok = current.po_confirmed(ibs0_max=ibs0_max)

    # ── IBS0 check — hard criterion valid in any population ──────────────────
    # IBS0 > 0 for a proposed parent is a definitive rejection: it proves
    # both individuals are homozygous for OPPOSITE alleles at that variant,
    # which is physically impossible between a true parent and their offspring.
    def _ibs0_note(name, ibs0_val, ibs0_max, pct):
        if ibs0_val is None:
            return (f"{name}: not in KING output — cannot confirm IBS0. "
                    f"Pair may have been dropped by --king-table-filter "
                    f"(kinship below threshold) or parent absent from KING VCF.")
        pct_note = f" (cohort rank: {pct:.0f}th pct)" if pct is not None else ""
        if ibs0_val > ibs0_max:
            # Distinguish between "clearly wrong" and "borderline, may be error noise"
            if ibs0_val > 3 * ibs0_max:
                level = "STRONG REJECTION"
                detail = (f"IBS0={ibs0_val:.3f} is far above the error-noise "
                          f"tolerance ({ibs0_max:.3f}). Cannot be a true parent.")
            else:
                level = "SOFT REJECTION"
                detail = (f"IBS0={ibs0_val:.3f} is slightly above {ibs0_max:.3f}. "
                          f"At this level, genotyping errors alone could NOT explain "
                          f"the IBS0 (that would require >3% error rate). "
                          f"However, verify with Mendelian error check and IBD coverage.")
            return (f"{name}: {level} — {detail}\n"
                    f"    IBS0 > 0 between parent and child is Mendelian-impossible "
                    f"with correct genotypes — but genotyping errors create false IBS0. "
                    f"The error-noise ceiling with quality-filtered variants is {ibs0_max:.3f}.")
        else:
            return (f"{name}: IBS0={ibs0_val:.3f} ✓ (within noise ceiling {ibs0_max:.3f})"
                    f", kinship={current.k_p1 if 'P1' in name else current.k_p2:.3f}"
                    f"{pct_note}")

    issues.append(_ibs0_note('P1', current.ibs_p1, ibs0_max, current.pct_p1))
    issues.append(_ibs0_note('P2', current.ibs_p2, ibs0_max, current.pct_p2))

    # ── Genome coverage — gold standard (only if phased IBD is available) ────
    if current.coverage:
        cov_frac = current.coverage_frac()
        issues.append(f"Genome coverage: {current.coverage['interpretation']}")
        if cov_frac is not None:
            if cov_frac < 0.70:
                issues.append(
                    f"  LOW COVERAGE ({cov_frac:.0%}) — this parent pair does not "
                    f"tile the child's genome. Note: phasing switch errors "
                    f"(from statistical phasing or genotyping errors) can reduce "
                    f"apparent coverage. If this is statistically-phased output "
                    f"(*_stat_phased.vcf.gz), re-run with pedigree-aware phasing "
                    f"(*_phased.vcf.gz) for a more accurate coverage estimate.")
            elif cov_frac < 0.85:
                issues.append(
                    f"  MODERATE COVERAGE ({cov_frac:.0%}) — consistent with a true "
                    f"parent pair in low-quality phasing, OR a false pair with high "
                    f"background IBD. Repeat with pedigree-aware phasing to "
                    f"distinguish: coverage should rise to > 90% for true parents.")

    # ── MZ twin substitution check ────────────────────────────────────────────
    def check_mz_swap(parent_to_swap, other_parent, label):
        group = mz_of.get(parent_to_swap)
        if not group or len(group) <= 1:
            return
        for twin in sorted(group - {parent_to_swap}):
            twin_rec = get_king(king, child, twin)
            if twin_rec is None:
                issues.append(
                    f"MZ group member {twin} of {label} ({parent_to_swap}) "
                    f"not found in KING output — re-run KING on the original "
                    f"(pre-purge) VCF to include all MZ group members.")
                continue
            twin_is_po = twin_rec['ibs0'] <= ibs0_max  # IBS0-first
            note = (f"MZ twin of {label}: {twin} "
                    f"(IBS0={twin_rec['ibs0']:.3f}, "
                    f"kinship={twin_rec['kinship']:.3f})")
            if twin_is_po:
                alt = make_assign(twin, other_parent, 'mz_swap',
                                  notes=[f"Substituted {label} ({parent_to_swap}) "
                                         f"with MZ twin {twin}"])
                alternatives.append(alt)
                issues.append(
                    f"CANDIDATE MZ SWAP: {note} — IBS0≈0 with child. "
                    f"MZ twins are genotypically identical so KING cannot distinguish "
                    f"which is the actual biological parent. Use breeding records, "
                    f"chronological information, or genome-coverage check if phased.")
            else:
                issues.append(f"MZ group check: {note} — IBS0={twin_rec['ibs0']:.3f} "
                               f"> {ibs0_max}: this twin is NOT a PO candidate.")

    check_mz_swap(orig_p1, orig_p2, 'P1')
    check_mz_swap(orig_p2, orig_p1, 'P2')

    # ── Search for unchecked PO candidates ───────────────────────────────────
    # Use IBS0-only filter for finding candidates — kinship_min is only
    # informational in related germplasm
    known_parents = {orig_p1, orig_p2}
    new_po = find_po_candidates(child, king, all_ids,
                                 kinship_min=0.0,  # IBS0 is the real filter
                                 ibs0_max=ibs0_max,
                                 exclude=known_parents | {child})

    if new_po:
        mz_of_p1 = mz_of.get(orig_p1, frozenset())
        mz_of_p2 = mz_of.get(orig_p2, frozenset())
        truly_new = [(ind, k, i) for ind, k, i in new_po
                     if ind not in mz_of_p1 and ind not in mz_of_p2]

        if truly_new:
            for ind, k, i in truly_new:
                pct = (king_pct.get((child, ind)) or king_pct.get((ind, child))
                       if king_pct else None)
                pct_note = f", cohort rank={pct:.0f}th pct" if pct is not None else ""
                issues.append(
                    f"UNCHECKED CANDIDATE: {ind} — IBS0={i:.3f}≈0 "
                    f"(kinship={k:.3f}{pct_note}). "
                    f"Possible interpretations (in order of likelihood in related pools): "
                    f"(a) sibling/half-sib of the true parent sharing relevant alleles; "
                    f"(b) a better parent than the current assignment; "
                    f"(c) second parent. "
                    f"IBS0≈0 is necessary but NOT sufficient in related germplasm — "
                    f"use genome-coverage check (run_beagle.sh + refine with --ibd) "
                    f"to distinguish these cases definitively.")

                if not p1_ibs_ok and parents_are_compatible(ind, orig_p2, king, ibs0_max):
                    alt = make_assign(ind, orig_p2, 'king_new',
                                      notes=[f"Replaced IBS0-failing P1 ({orig_p1}) "
                                             f"with IBS0-passing candidate {ind}"])
                    alternatives.append(alt)
                elif not p2_ibs_ok and parents_are_compatible(orig_p1, ind, king, ibs0_max):
                    alt = make_assign(orig_p1, ind, 'king_new',
                                      notes=[f"Replaced IBS0-failing P2 ({orig_p2}) "
                                             f"with IBS0-passing candidate {ind}"])
                    alternatives.append(alt)

    # ── Parent pair compatibility check ──────────────────────────────────────
    if not parents_are_compatible(orig_p1, orig_p2, king, ibs0_max):
        issues.append(
            f"WARNING: Parents {orig_p1} and {orig_p2} themselves appear to be "
            f"parent-offspring in KING (kinship={kinship_of(king,orig_p1,orig_p2):.3f}, "
            f"IBS0={ibs0_of(king,orig_p1,orig_p2):.3f}). "
            f"This suggests a multi-generation structure: the assigned 'parents' "
            f"are actually grandparent + parent, not two co-parents. "
            f"Consider checking the inverted trio analysis.")

    # Sort alternatives by confidence score (scaled against the SAME
    # calibrated ibs0_max used for the pass/fail check above, not the
    # hardcoded KING_PO_IBS0_MAX default -- see confidence()'s docstring).
    alternatives.sort(key=lambda a: -a.confidence(ibs0_max))

    # Best assignment: prefer original if both parents pass the IBS0 check,
    # else fall back to the best-scoring alternative.
    # Note: `issues` is an audit-trail list (always has informational entries
    # for confirmed pairs too), so it is NOT used here to gate confirmation.
    if p1_ibs_ok and p2_ibs_ok:
        best = current
        best.source = 'original_confirmed'
    elif alternatives and alternatives[0].confidence(ibs0_max) > current.confidence(ibs0_max):
        best = alternatives[0]
        best.notes.insert(0, f"Replaced original ({orig_p1} + {orig_p2}); "
                            f"original T1={t1:.1f}%")
    else:
        best = current

    return best, alternatives, issues


# ---------------------------------------------------------------------------
# Inverted trio direction check
# ---------------------------------------------------------------------------

def check_inverted_direction(p1, p2, child, king, kinship_min, ibs0_max):
    """
    For an inverted trio (p1, p2, child), use KING to determine whether:
      A. The trio is actually a regular parent-offspring (false T2 failure)
      B. The "child" is actually a parent of p1 or p2 (inverted direction)
      C. The structure is grandparent-parent-grandchild

    Returns a string classification and notes.
    """
    po_c_p1 = is_po_signal(king, child, p1, kinship_min, ibs0_max)
    po_c_p2 = is_po_signal(king, child, p2, kinship_min, ibs0_max)
    po_p1_p2 = is_po_signal(king, p1, p2, kinship_min, ibs0_max)

    k_c_p1 = kinship_of(king, child, p1)
    k_c_p2 = kinship_of(king, child, p2)
    k_p1_p2 = kinship_of(king, p1, p2)

    notes = []

    if po_c_p1 and po_c_p2 and not po_p1_p2:
        classification = 'LIKELY_REGULAR_TRIO'
        notes.append("KING confirms PO signal child↔P1 and child↔P2; P1↔P2 not PO.")
        notes.append("Inverted T2 failure may be noise or insufficient BB+BB variants.")
        notes.append("Recommend: include as Regular trio if T1 > 95%.")

    elif po_p1_p2 and (po_c_p1 or po_c_p2):
        # P1 and P2 are themselves PO → grandparent chain
        classification = 'GRANDPARENT_CHAIN'
        notes.append(f"P1 ({p1}) and P2 ({p2}) appear PO to each other "
                     f"(KING kinship={k_p1_p2:.3f}). Likely structure: "
                     f"grandparent→parent→grandchild.")
        if po_c_p1 and not po_c_p2:
            notes.append(f"Child↔P1 shows PO signal → P1 is likely the parent; "
                         f"P2 is likely grandparent.")
        elif po_c_p2 and not po_c_p1:
            notes.append(f"Child↔P2 shows PO signal → P2 is likely the parent; "
                         f"P1 is likely grandparent.")

    elif po_c_p1 and not po_c_p2:
        classification = 'PARTIAL_PO'
        notes.append(f"Only child↔P1 has PO signal (kinship={k_c_p1:.3f}). "
                     f"P2 ({p2}) may be more distantly related or unrelated.")

    elif po_c_p2 and not po_c_p1:
        classification = 'PARTIAL_PO'
        notes.append(f"Only child↔P2 has PO signal (kinship={k_c_p2:.3f}). "
                     f"P1 ({p1}) may be more distantly related or unrelated.")

    else:
        classification = 'UNCLEAR'
        notes.append("KING does not clearly confirm any PO relationship in this trio.")
        notes.append(f"KING kinships: child-P1={k_c_p1}, child-P2={k_c_p2}, "
                     f"P1-P2={k_p1_p2}")

    return classification, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Refine vcfped parent assignments using PLINK2 KING kinship',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s step3_named_filtered king_results
  %(prog)s step3_named_filtered king_results --step1-pair step1_named
  %(prog)s step3_named_filtered king_results --include-inverted --kinship-min 0.20

Typical workflow:
  1. python filter_trios.py step3_named --strategy best
  2. python prepare_plink_beagle.py step3_named_filtered input.vcf.gz
  3. bash step3_named_filtered_run_plink.sh          # produces king.kin0
  4. python refine_pedigree.py step3_named_filtered king_results
        """,
    )
    parser.add_argument('trio_prefix',
                        help='Filtered trio prefix (reads <prefix>.trio)')
    parser.add_argument('king_prefix',
                        help='PLINK2 KING prefix (reads <prefix>.kin0)')
    parser.add_argument('--step1-pair',
                        help='Step 1 pairwise prefix for MZ group info (reads <prefix>.pair). '
                             "CAVEAT: vcfped's raw pairwise \"MZ twins\" verdict can be inflated "
                             'by inbreeding-induced allele-sharing in densely inbred cohorts -- '
                             'prefer --confirmed-duplicates when available. Mutually exclusive '
                             'with --confirmed-duplicates.',
                        default=None)
    parser.add_argument('--confirmed-duplicates', default=None,
                        help='Alternative, stricter MZ-group source: a two-column pair list '
                             '(header ID1/ID2) of hap-IBD-coverage-CONFIRMED duplicates, e.g. '
                             "extend_pedigree.py's <prefix>_confirmed_duplicates.tsv. The "
                             'mz_swap check only trusts group members as interchangeable '
                             'substitute parents if they are genuinely identical -- this avoids '
                             "silently substituting in a merely closely-related, NOT identical, "
                             'individual from an inflated raw --step1-pair group. Mutually '
                             'exclusive with --step1-pair.')
    parser.add_argument('--fam',
                        help='Current .fam file (default: <trio_prefix>.fam)',
                        default=None)
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: <trio_prefix>)',
                        default=None)
    parser.add_argument('--include-inverted', action='store_true',
                        help='Also analyse inverted/generational trios for direction')
    parser.add_argument('--ibd',
                        help='hap-IBD .ibd.gz file for genome-coverage check (gold standard). '
                             'Use *_stat_ibd.ibd.gz for a first pass (candidates aren\'t in any '
                             '.fam yet, so only statistical phasing applies). For trios ALREADY '
                             'confirmed/merged into a .fam, re-run run_beagle.sh with that FAM '
                             '(prepare_plink_beagle.py --from-fam <merged.fam> ...) and pass the '
                             'pedigree-aware *_ibd.ibd.gz instead -- higher precision, since '
                             'Mendelian constraints propagate through those specific trios.',
                        default=None)
    parser.add_argument('--kinship-min', type=float, default=0.0,
                        help='Soft minimum KING kinship for candidate search '
                             '(default: 0 = IBS0 is the only hard filter; '
                             'set to 0.177 for outbred populations)')
    parser.add_argument('--ibs0-max', type=float, default=None,
                        help='Maximum IBS0 for PO signal. Overrides --error-rate. '
                             'Default: computed from --error-rate.')
    parser.add_argument('--error-rate', type=float, default=0.025,
                        help='Expected per-genotype error rate in your data, used to '
                             'set the IBS0 noise ceiling. '
                             'WGS 30x≈0.001, WGS 10x≈0.005, GBS≈0.010, default=0.025 '
                             '(conservative: tolerates up to 2.5%% error).')
    parser.add_argument('--min-t1', type=float, default=90.0,
                        help='Minimum T1%% to include a trio in output (default: 90)')
    args = parser.parse_args()

    if args.step1_pair and args.confirmed_duplicates:
        sys.exit("Error: specify at most one of --step1-pair or --confirmed-duplicates.")

    out_prefix    = args.output or args.trio_prefix
    fam_path      = args.fam or f"{args.trio_prefix}.fam"
    king_kin0     = f"{args.king_prefix}.kin0"
    trio_path     = f"{args.trio_prefix}.trio"
    inv_trio_path = f"{args.trio_prefix}_inverted.trio"

    # Resolve IBS0 threshold from error rate if not explicitly set
    if args.ibs0_max is None:
        args.ibs0_max = ibs0_threshold_for_error_rate(args.error_rate)

    print("=" * 70)
    print("VCFped Pedigree Refinement")
    print("=" * 70)
    print(f"  Trio file      : {trio_path}")
    print(f"  KING file      : {king_kin0}")
    print(f"  Output         : {out_prefix}")
    # .4f/.4% (not the old .1%/.3f) -- at very low calibrated error rates
    # (e.g. 0.0002), coarser precision made two materially different
    # --ibs0-max values print identically, hiding whether an override
    # actually took effect.
    print(f"  Genotype error : {args.error_rate:.4%} (IBS0 noise ceiling: {args.ibs0_max:.4f})")
    if args.error_rate > 0.01:
        print(f"  NOTE: error rate {args.error_rate:.4%} is elevated. "
              f"Consider filtering variants with --geno 0.05 --maf 0.01 --hwe 1e-10 "
              f"in PLINK2 before running KING (see _run_plink.sh).")
    print()

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("Loading KING kinship matrix...")
    required_min = 0.0884 if args.include_inverted else 0.044
    king, all_ids = load_king(king_kin0, required_min_kinship=required_min)
    print(f"  {len(king):,} pairs, {len(all_ids):,} unique individuals")

    # Build cohort-relative percentile ranks (population-agnostic)
    print("  Computing cohort-relative kinship percentiles...")
    king_pct = build_king_percentiles(king)
    all_kinships = sorted(r['kinship'] for r in king.values())
    if all_kinships:
        p25 = all_kinships[len(all_kinships)//4]
        p75 = all_kinships[3*len(all_kinships)//4]
        p95 = all_kinships[int(0.95*len(all_kinships))]
        print(f"  Kinship distribution: p25={p25:.3f}, p75={p75:.3f}, p95={p95:.3f}")
        if p75 > 0.10:
            print(f"  NOTE: median background kinship is high (p75={p75:.3f}). "
                  f"This cohort is closely related — fixed kinship thresholds "
                  f"are unreliable. IBS0 and genome coverage will be used as "
                  f"primary evidence.")

    print("Loading trio assignments...")
    fieldnames, all_trio_rows, regular_trios = load_trio_file(trio_path)
    if not regular_trios:
        sys.exit(f"No Regular trio rows found in {trio_path}.")
    print(f"  {len(regular_trios)} Regular trio calls")

    inv_trios = []
    if args.include_inverted and os.path.exists(inv_trio_path):
        _, _, inv_trios = load_trio_file(
            inv_trio_path, verdicts=('Inverted/generational trio',))
        print(f"  {len(inv_trios)} Inverted/generational trios loaded")

    mz_of = {}
    if args.confirmed_duplicates:
        print(f"Loading MZ groups from {args.confirmed_duplicates} "
              f"(hap-IBD-coverage validated)...")
        mz_of = load_mz_groups_from_confirmed_duplicates(args.confirmed_duplicates)
    elif args.step1_pair:
        print(f"Loading MZ groups from {args.step1_pair}.pair "
              f"(vcfped's raw pairwise verdict)...")
        mz_of = load_mz_groups(args.step1_pair)

    if mz_of:
        group_sizes = sorted({len(v) for v in mz_of.values()})
        n_groups = len({v for v in mz_of.values()})
        max_size = max(group_sizes)
        print(f"  {len(mz_of)} samples in {n_groups} MZ groups "
              f"(group size: min={min(group_sizes)}, max={max_size}, "
              f"mean={len(mz_of)/n_groups:.1f})")
        if not args.confirmed_duplicates and max_size > 10:
            print(f"  WARNING: largest group has {max_size} members — implausible for a "
                  f"literal clonal-duplicate cluster. In densely inbred cohorts, vcfped's raw "
                  f"pairwise \"MZ twins\" verdict can be inflated by inbreeding-induced "
                  f"allele-sharing (a real cohort measured ~15x inflation this way). "
                  f"check_mz_swap() trusts every group member as an interchangeable "
                  f"substitute parent — a contaminated group can silently substitute in a "
                  f"genuinely DIFFERENT individual, not the same one under another label. "
                  f"Prefer --confirmed-duplicates (extend_pedigree.py's "
                  f"<prefix>_confirmed_duplicates.tsv, run with --ibd) if available.")

    ibd_index = {}
    if args.ibd:
        print(f"Loading IBD segments from {args.ibd} (gold standard)...")
        ibd_index = load_ibd_index(args.ibd)
        print(f"  {len(ibd_index):,} IBD pairs loaded")
        print(f"  Genome coverage check ENABLED — this is the highest-certainty "
              f"confirmation and is reliable even in closely related germplasm.")
    else:
        print("  No IBD file provided — genome coverage check DISABLED.")
        print("  Run run_beagle.sh and re-run with --ibd *_stat_ibd.ibd.gz for "
              "gold-standard confirmation.")

    fam = load_fam(fam_path)
    print()

    # ── Refine regular trios ──────────────────────────────────────────────────
    print("Refining parent assignments...")
    results = []       # (best_assignment, alternatives, issues, orig_row)
    status_counts = defaultdict(int)

    for row in regular_trios:
        if float(row.get('Test1', 0)) < args.min_t1:
            continue
        child = row['_child']
        p1    = row['_p1']
        p2    = row['_p2']

        best, alts, issues = refine_child(
            child, p1, p2, row, king, mz_of, all_ids,
            args.kinship_min, args.ibs0_max,
            king_pct=king_pct, ibd_index=ibd_index or None)

        results.append((best, alts, issues, row))

        p1_ibs, p2_ibs = best.po_confirmed(ibs0_max=args.ibs0_max)
        cov = best.coverage_frac()
        if p1_ibs and p2_ibs and (cov is None or cov >= 0.90):
            status_counts['confirmed'] += 1
        elif p1_ibs and p2_ibs:
            status_counts['ibs0_pass_low_coverage'] += 1
        elif best.source in ('mz_swap', 'king_new'):
            status_counts['improved'] += 1
        else:
            status_counts['ibs0_fail_needs_review'] += 1

    # ── Analyse inverted trios ────────────────────────────────────────────────
    inv_results = []
    if inv_trios:
        print("Checking inverted trio directions with KING...")
        for row in inv_trios:
            cls, notes = check_inverted_direction(
                row['_p1'], row['_p2'], row['_child'],
                king, args.kinship_min, args.ibs0_max)
            inv_results.append((row, cls, notes))
            if cls == 'LIKELY_REGULAR_TRIO':
                status_counts['inv_reclassify_regular'] += 1
            elif cls == 'GRANDPARENT_CHAIN':
                status_counts['inv_grandparent_chain'] += 1
            else:
                status_counts['inv_unclear'] += 1

    # ── Write outputs ─────────────────────────────────────────────────────────
    print()
    print("Writing output files...")

    # Refined .trio file
    out_trio_path = f"{out_prefix}_refined.trio"
    out_rows = []
    for best, alts, issues, orig_row in results:
        row = dict(orig_row)
        # Update Triple and Pivot to reflect refined assignment
        triple_ids = sorted([best.p1, best.p2])
        row['Triple'] = f"{triple_ids[0]},{triple_ids[1]},{best.child}"
        row['Pivot']  = best.child
        # Remove internal keys
        for k in list(row):
            if k.startswith('_'):
                del row[k]
        out_rows.append(row)

    # Optionally append reclassified inverted trios as regular
    for row, cls, notes in inv_results:
        if cls == 'LIKELY_REGULAR_TRIO':
            r = dict(row)
            r['Verdict'] = 'Regular trio (reclassified from inverted)'
            for k in list(r):
                if k.startswith('_'):
                    del r[k]
            out_rows.append(r)

    write_trio_file(out_trio_path, fieldnames, out_rows)

    # Refined .fam file
    out_fam_path = f"{out_prefix}_refined.fam"
    for best, alts, issues, orig_row in results:
        child = best.child
        if child in fam:
            fam[child]['pat'] = best.p1
            fam[child]['mat'] = best.p2
        else:
            fam[child] = {'fid': '0', 'pat': best.p1, 'mat': best.p2, 'sex': '0'}

    # Also fold in reclassified-from-inverted trios (cls == LIKELY_REGULAR_TRIO).
    # Without this, only `results` children ever reach the FAM, even though
    # inv_reclassify_regular can be orders of magnitude larger than `results`
    # itself — the inverted pool is never deduplicated per child upstream
    # (filter_trios.py only dedupes the Regular pool unless --include-inverted
    # is used there too, which has its own downstream risks — see
    # _is_better_pair_result()'s TODO in vcfped_optimized.py for the analogous
    # issue on the vcfped side). Children already assigned above are left
    # untouched: `results`-based regular-trio evidence outranks a
    # reclassified-inverted call. For children with multiple competing
    # LIKELY_REGULAR_TRIO rows, pick the highest-Test1 candidate, matching
    # filter_trios.py's own best-per-child tie-break so the whole pipeline
    # uses one consistent confidence metric rather than an arbitrary
    # first-seen/last-seen pick.
    reclassified_best = {}   # child -> (t1, p1, p2)
    for row, cls, notes in inv_results:
        if cls != 'LIKELY_REGULAR_TRIO':
            continue
        child, p1, p2 = row['_child'], row['_p1'], row['_p2']
        if child in fam:
            continue
        t1 = row.get('_t1', 0.0)
        current = reclassified_best.get(child)
        if current is None or t1 > current[0]:
            reclassified_best[child] = (t1, p1, p2)

    for child, (t1, p1, p2) in reclassified_best.items():
        fam[child] = {'fid': '0', 'pat': p1, 'mat': p2, 'sex': '0'}
    status_counts['fam_new_from_reclassified_inverted'] = len(reclassified_best)

    write_fam(out_fam_path, fam)

    # Refinement report
    out_report = f"{out_prefix}_refinement_report.txt"
    with open(out_report, 'w') as rpt:
        rpt.write("VCFped Pedigree Refinement Report\n")
        rpt.write("=" * 70 + "\n\n")
        rpt.write(f"Input trio file : {trio_path}\n")
        rpt.write(f"KING file       : {king_kin0}\n")
        mz_source = (f"{args.confirmed_duplicates} (hap-IBD-coverage validated)" if args.confirmed_duplicates
                    else args.step1_pair or 'not provided')
        rpt.write(f"MZ group source : {mz_source}\n\n")

        rpt.write("Status summary\n" + "-" * 40 + "\n")
        for k, v in sorted(status_counts.items()):
            rpt.write(f"  {k:40s}: {v}\n")
        rpt.write("\n")

        rpt.write("Regular trio refinement details\n" + "-" * 40 + "\n")
        for best, alts, issues, orig_row in results:
            rpt.write(f"\nChild: {best.child}\n")
            rpt.write(f"  Original : {orig_row['_p1']} + {orig_row['_p2']} "
                      f"(T1={orig_row.get('Test1','?')}%, "
                      f"Perc={orig_row.get('Perc','?')})\n")
            rpt.write(f"  Refined  : {best.p1} + {best.p2}  "
                      f"[source: {best.source}]\n")
            if best.k_p1 is not None:
                rpt.write(f"  KING P1  : kinship={best.k_p1:.3f}, "
                           f"IBS0={best.ibs_p1:.3f}\n")
            if best.k_p2 is not None:
                rpt.write(f"  KING P2  : kinship={best.k_p2:.3f}, "
                           f"IBS0={best.ibs_p2:.3f}\n")
            if best.notes:
                for n in best.notes:
                    rpt.write(f"  Note: {n}\n")
            if issues:
                for iss in issues:
                    rpt.write(f"  ⚠ {iss}\n")
            if alts:
                rpt.write(f"  Alternatives considered:\n")
                for alt in alts:
                    rpt.write(f"    {alt.p1} + {alt.p2} "
                               f"[{alt.source}] conf={alt.confidence(args.ibs0_max):.3f}\n")

        if inv_results:
            rpt.write("\nInverted/generational trio analysis\n" + "-" * 40 + "\n")
            for row, cls, notes in inv_results:
                rpt.write(f"\nTriple: {row['_p1']}, {row['_p2']}, {row['_child']} "
                           f"[pivot={row['_child']}]\n")
                rpt.write(f"  Classification: {cls}\n")
                for n in notes:
                    rpt.write(f"  {n}\n")

        # Strategy recommendations
        rpt.write("\n" + "=" * 70 + "\n")
        rpt.write("STRATEGY RECOMMENDATIONS\n")
        rpt.write("=" * 70 + "\n\n")

        n_unconfirmed = status_counts.get('original_unconfirmed', 0)
        n_improved    = status_counts.get('improved', 0)
        n_inv_reg     = status_counts.get('inv_reclassify_regular', 0)

        if n_unconfirmed == 0:
            rpt.write("✓ All parent assignments are confirmed by KING kinship.\n"
                      "  Proceed to extend_pedigree.py for grandparent/grandchild search.\n")
        else:
            rpt.write(f"⚠ {n_unconfirmed} assignments not confirmed by KING.\n\n")
            rpt.write("Recommended next steps:\n\n")
            rpt.write("  Option A — Re-run vcfped with full candidate set (most accurate):\n"
                      "    Extract the unconfirmed children and all their KING 1st-degree\n"
                      "    relatives into a subset VCF, then re-run vcfped --no-gender on that.\n"
                      "    This gives exact T1/T2 Mendelian scores for all candidate triples.\n\n")
            rpt.write("  Option B — Use hap-IBD segments (most accurate for direction):\n"
                      "    Run the BEAGLE script, then extend_pedigree.py --ibd <prefix>.ibd.gz.\n"
                      "    IBD segment analysis distinguishes PO (no Z2 segments) from\n"
                      "    full-sibling (has Z2) and grandparent chains (shorter segments).\n\n")
            rpt.write("  Option C — KING is sufficient for most cases:\n"
                      "    IBS0 < 0.02 + KINSHIP > 0.20 is strong PO evidence.\n"
                      "    Proceed to extend_pedigree.py and flag unconfirmed cases for\n"
                      "    manual review using external breeding records.\n\n")

        if not mz_of:
            rpt.write("NOTE: No MZ group source was provided. To check whether\n"
                      "  non-representative MZ twins are better parent candidates,\n"
                      "  re-run with --confirmed-duplicates <prefix>_confirmed_duplicates.tsv\n"
                      "  (extend_pedigree.py run with --ibd -- preferred, hap-IBD-coverage\n"
                      "  validated) or --step1-pair <step1_prefix> (vcfped's raw pairwise\n"
                      "  verdict, which inbreeding-induced allele-sharing can inflate in\n"
                      "  densely inbred cohorts), AND run KING on the original (pre-purge)\n"
                      "  VCF so all MZ group members appear in the kinship matrix.\n")

    print(f"  Written: {out_report}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("Summary:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k:40s}: {v}")
    print()
    print("Next step: python utilities/extend_pedigree.py "
          f"{out_fam_path} {args.king_prefix}.kin0")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
