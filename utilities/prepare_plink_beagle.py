#!/usr/bin/env python3
"""
Generate PLINK and BEAGLE input files from a VCF + vcfped trio output.

Reads:
  <trio_prefix>.trio   — filtered trio file (use filter_trios.py first)
  <vcf>                — original VCF or VCF.gz (for chromosome names + sample order)
  <trio_prefix>.gender — optional, for sex assignment in PLINK .fam

Writes:
  <out_prefix>.fam                — PLINK/BEAGLE pedigree (family file)
  <out_prefix>_chr_rename.txt     — chromosome rename map  (old → new)
  <out_prefix>_run_plink.sh       — ready-to-run PLINK verification script
  <out_prefix>_run_beagle.sh      — ready-to-run BEAGLE phasing + IBD script

Usage:
    python prepare_plink_beagle.py step3_named_filtered input.vcf.gz
    python prepare_plink_beagle.py step3_named_filtered input.vcf.gz -o plink_beagle
    python prepare_plink_beagle.py step3_named_filtered input.vcf.gz --chr-prefix ch
"""

import sys
import os
import re
import gzip
import csv
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# VCF helpers
# ---------------------------------------------------------------------------

def _open_vcf(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def scan_vcf_header(vcf_path):
    """Return (chromosome_list_in_order, sample_names)."""
    contigs = []
    samples = []
    data_chroms_seen = []

    with _open_vcf(vcf_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('##contig'):
                m = re.search(r'ID=([^,>]+)', line)
                if m:
                    contigs.append(m.group(1))
            elif line.startswith('#CHROM'):
                parts = line.split('\t')
                samples = parts[9:]
                break   # stop at header line; data scanning below

        # If no ##contig lines, collect unique chromosomes from the first 5000 variants
        if not contigs:
            seen = set()
            n = 0
            for line in f:
                if line.startswith('#'):
                    continue
                chrom = line.split('\t', 1)[0]
                if chrom not in seen:
                    seen.add(chrom)
                    data_chroms_seen.append(chrom)
                n += 1
                if n >= 5000:
                    break

    return contigs if contigs else data_chroms_seen, samples


def build_chr_rename(chroms, chr_prefix):
    """
    Map chromosome names to plain integers for PLINK.

    Handles:
      ch01..ch25  →  1..25
      chr1..chr25 →  1..25
      1..25       →  no rename needed
    Returns list of (old, new) pairs (only for names that need renaming).
    """
    rename = []
    numeric_re = re.compile(r'^(?:' + re.escape(chr_prefix) + r')0*(\d+)$', re.IGNORECASE)
    already_numeric = re.compile(r'^\d+$')

    # Sort so that ch01 < ch02 < … < ch09 < ch10 etc.
    def sort_key(c):
        m = numeric_re.match(c)
        return int(m.group(1)) if m else 9999

    ordered = sorted(set(chroms), key=sort_key)
    counter = 1
    for c in ordered:
        if already_numeric.match(c):
            continue   # already numeric; PLINK can handle it
        m = numeric_re.match(c)
        if m:
            new = m.group(1).lstrip('0') or '0'
        else:
            new = str(counter)
            counter += 1
        rename.append((c, new))
    return rename


# ---------------------------------------------------------------------------
# Trio / gender file readers
# ---------------------------------------------------------------------------

def read_trio_file(path):
    """Return list of dicts with keys: triple, pivot, p1, p2, child, perc, t1, verdict."""
    try:
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
    except FileNotFoundError:
        sys.exit(f"Error: {path} not found.")

    result = []
    for row in rows:
        verdict = row.get('Verdict', '').strip()
        if verdict != 'Regular trio':
            continue
        triple = [s.strip() for s in str(row.get('Triple', '')).split(',')]
        pivot = str(row.get('Pivot', '')).strip()
        if len(triple) != 3 or pivot not in triple:
            continue
        child = pivot
        parents = [s for s in triple if s != child]
        if len(parents) != 2:
            continue
        result.append({
            'p1': parents[0], 'p2': parents[1], 'child': child,
            'perc': int(row.get('Perc', 0)),
            't1':   float(row.get('Test1', 0.0)),
        })
    return result


def read_gender_file(path):
    """Return dict: sample_name → 'Male'|'Female'|None."""
    if not path or not os.path.exists(path):
        return {}
    gender = {}
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                s = str(row.get('Sample', '')).strip()
                g = str(row.get('Gender', '')).strip()
                gender[s] = g if g in ('Male', 'Female') else None
    except Exception:
        pass
    return gender


def fam_to_trios(fam_path):
    """
    Convert an existing PLINK .fam file's confirmed parent-child links into
    the same trio-list structure read_trio_file() produces, so build_pedigree()
    runs unchanged on a pedigree that's already fully decided elsewhere (e.g.
    merge_refined_fam.py's output) instead of re-deriving it from a raw
    vcfped .trio file.

    Only rows where BOTH pat and mat are known (non-'0') become a trio — a
    synthetic 'full confidence' entry (perc=100, t1=100.0): there's no vcfped
    Perc/Test1 score to carry over here, since the assignment itself is the
    input, already decided upstream (possibly via KING/IBD evidence rather
    than vcfped at all, e.g. reclassified-from-inverted trios).
    """
    trios = []
    with open(fam_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iid, pat, mat = parts[1], parts[2], parts[3]
            if pat != '0' and mat != '0':
                trios.append({'p1': pat, 'p2': mat, 'child': iid, 'perc': 100, 't1': 100.0})
    return trios


def read_fam_sex(fam_path):
    """
    Read an existing .fam file's SEX column into the same
    {sample: 'Male'|'Female'|None} shape read_gender_file() produces, so
    --from-fam mode feeds already-known sex into build_pedigree()'s existing
    "biological sex takes priority over role-derived sex" logic instead of
    discarding it and re-deriving everything from PAT/MAT role alone.
    """
    sex = {}
    with open(fam_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            iid, sex_code = parts[1], parts[4]
            sex[iid] = 'Male' if sex_code == '1' else 'Female' if sex_code == '2' else None
    return sex


# ---------------------------------------------------------------------------
# Pedigree construction
# ---------------------------------------------------------------------------

def _sex_code(name, gender):
    g = gender.get(name)
    if g == 'Male':
        return '1'
    if g == 'Female':
        return '2'
    return '0'


def build_pedigree(trios, gender, samples):
    """
    Return:
      fam_of[sample]    → family label (F1, F2, …)
      pat_of[child]     → father ID (or '0')
      mat_of[child]     → mother ID (or '0')
      all_individuals   → ordered list of all sample names in dataset
      warnings          → list of warning strings
    """
    warnings = []

    # Connected components → family labels
    adj = defaultdict(set)
    for t in trios:
        for a in (t['p1'], t['p2'], t['child']):
            for b in (t['p1'], t['p2'], t['child']):
                if a != b:
                    adj[a].add(b)

    visited = {}
    fam_counter = [0]

    def bfs(start):
        fam_counter[0] += 1
        label = f'F{fam_counter[0]}'
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited[node] = label
            queue.extend(adj[node] - set(visited))

    for t in trios:
        for s in (t['p1'], t['p2'], t['child']):
            if s not in visited:
                bfs(s)

    # Samples not in any trio get their own singleton family
    for s in samples:
        if s not in visited:
            fam_counter[0] += 1
            visited[s] = f'F{fam_counter[0]}'

    fam_of = visited

    # Assign paternal / maternal from gender
    pat_of = {}
    mat_of = {}
    sex = {s: gender.get(s) for s in samples}

    for t in trios:
        child, p1, p2 = t['child'], t['p1'], t['p2']

        g1, g2 = sex.get(p1), sex.get(p2)

        if g1 == 'Male' and g2 == 'Female':
            father, mother = p1, p2
        elif g1 == 'Female' and g2 == 'Male':
            father, mother = p2, p1
        elif g1 == 'Male' and g2 is None:
            father, mother = p1, p2   # p2 sex unknown → assume female
        elif g1 is None and g2 == 'Male':
            father, mother = p2, p1
        elif g1 == 'Female' and g2 is None:
            father, mother = p2, p1   # p1 sex unknown → assume male
        elif g1 is None and g2 == 'Female':
            father, mother = p1, p2
        else:
            # Both unknown or both same sex → arbitrary; flag it
            father, mother = p1, p2
            if g1 == g2 and g1 is not None:
                warnings.append(
                    f"WARNING: both parents of {child} have same gender ({g1}): "
                    f"{p1}, {p2}.  Assigned arbitrarily — check manually.")
            else:
                warnings.append(
                    f"NOTE: sex unknown for both parents of {child} ({p1}, {p2}). "
                    f"Assigned {p1}=father, {p2}=mother arbitrarily.")

        # A child can appear in multiple trios (post filter_trios this is rare,
        # but handle gracefully: keep highest-confidence assignment)
        if child not in pat_of:
            pat_of[child] = father
            mat_of[child] = mother
        else:
            if (t['perc'], t['t1']) > (pat_of.get('_perc_' + child, 0),
                                        mat_of.get('_t1_' + child, 0)):
                pat_of[child] = father
                mat_of[child] = mother

    # ── Derive sex from parent role when biological sex is unknown ───────────
    # PLINK2 --mendel requires SEX=1 for any individual listed as PAT (father)
    # and SEX=2 for any individual listed as MAT (mother).
    # ── Derive sex from parent role (required by PLINK2 --mendel) ────────────
    # For plant species (no sex chromosomes), this assignment is structural only
    # and does NOT imply biological sex.
    #
    # In plant breeding, the SAME accession can be used as:
    #   - pollen donor (paternal, PAT) in one cross
    #   - seed parent (maternal, MAT) in another cross
    # This creates a role conflict. We use majority-vote: if an individual
    # appears more often as father than as mother (across all trios), it gets
    # SEX=1; otherwise SEX=2. The minority-role trios will have a structurally
    # "wrong" sex in the .fam, which PLINK2 --mendel handles as follows:
    #   PLINK2 a.7+: emits a warning but continues (non-fatal for most versions)
    #   PLINK2 older: may error on "inconsistent sex" for that individual
    # In the latter case, add --allow-no-sex to the mendel command, or exclude
    # the conflicted individual from Mendelian checking.

    pat_count = {}   # individual → number of trios where it is PAT
    mat_count = {}   # individual → number of trios where it is MAT

    for child, father in pat_of.items():
        if father != '0':
            pat_count[father] = pat_count.get(father, 0) + 1

    for child, mother in mat_of.items():
        if mother != '0':
            mat_count[mother] = mat_count.get(mother, 0) + 1

    derived_sex = {}
    all_parents = set(pat_count) | set(mat_count)
    for ind in all_parents:
        p = pat_count.get(ind, 0)
        m = mat_count.get(ind, 0)
        if p > 0 and m > 0:
            # Appears in both roles — common in hermaphrodite plant breeding
            majority = 'Male' if p >= m else 'Female'
            derived_sex[ind] = majority
            warnings.append(
                f"NOTE (plant crossing): {ind} used as father in {p} trio(s) "
                f"and mother in {m} trio(s). Assigned SEX={'1 (PAT majority)' if majority=='Male' else '2 (MAT majority)'}. "
                f"PLINK2 --mendel may still warn for the minority-role trios — "
                f"this is expected for hermaphrodite plant accessions.")
        elif p > 0:
            derived_sex[ind] = 'Male'
        else:
            derived_sex[ind] = 'Female'

    # Merge: biological sex (from .gender file) takes priority over role-derived sex
    merged_sex = {}
    for s in samples:
        bio = gender.get(s)
        role = derived_sex.get(s)
        if bio in ('Male', 'Female'):
            merged_sex[s] = bio       # biological sex known → use it
        elif role in ('Male', 'Female'):
            merged_sex[s] = role      # role-derived (structural, not biological)
        else:
            merged_sex[s] = None      # not a parent → sex unknown → 0

    return fam_of, pat_of, mat_of, list(samples), warnings, merged_sex


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_fam(path, samples, fam_of, pat_of, mat_of, merged_sex):
    """
    Write PLINK .fam file.  Same format accepted by BEAGLE's ped parameter.

    Columns: FID  IID  PAT  MAT  SEX  PHENO

    SEX assignment:
      1 = Male   (biological sex from .gender file, or role-derived: listed as PAT)
      2 = Female (biological sex from .gender file, or role-derived: listed as MAT)
      0 = Unknown (not a parent, or sex conflict)

    PLINK2 --mendel requires SEX=1 for fathers and SEX=2 for mothers.
    For species without sex chromosomes, the role-derived assignment is
    structural only and has no biological meaning.
    """
    # PLINK2 --mendel rule: if individual X has SEX=1 it may ONLY appear in PAT
    # columns, never in MAT columns (and vice versa for SEX=2).
    # For plant hermaphrodites that appear in both roles, we assigned the
    # majority-role sex.  Any child row where the minority-role parent would
    # violate this rule must have that parent link set to '0'.
    # Those trios lose their PLINK Mendelian check but --mendel runs without error.
    # KING kinship (--make-king-table) is unaffected by this clearing.
    removed_links = []
    lines = []
    for s in samples:
        fid  = fam_of.get(s, '0')
        iid  = s
        pat  = pat_of.get(s, '0')
        mat  = mat_of.get(s, '0')
        g    = merged_sex.get(s)
        sex  = '1' if g == 'Male' else '2' if g == 'Female' else '0'

        # Remove parent links that contradict the parent's assigned sex
        if pat != '0' and merged_sex.get(pat) == 'Female':
            removed_links.append((s, 'PAT', pat, 'SEX=2 (female) cannot be a father'))
            pat = '0'
        if mat != '0' and merged_sex.get(mat) == 'Male':
            removed_links.append((s, 'MAT', mat, 'SEX=1 (male) cannot be a mother'))
            mat = '0'

        pheno = '-9'
        lines.append('\t'.join([fid, iid, pat, mat, sex, pheno]))

    n_males   = sum(1 for s in samples if merged_sex.get(s) == 'Male')
    n_females = sum(1 for s in samples if merged_sex.get(s) == 'Female')
    n_unknown = len(samples) - n_males - n_females
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    if removed_links:
        affected = {parent for _, _, parent, _ in removed_links}
        print(f"  NOTE: {len(removed_links)} parent links cleared for "
              f"{len(affected)} hermaphrodite individual(s) used in both crossing roles.")
        for child, role, parent, reason in removed_links[:5]:
            print(f"    {child} {role}={parent} removed ({reason})")
        if len(removed_links) > 5:
            print(f"    ... and {len(removed_links) - 5} more")
        print(f"  These trios are excluded from PLINK --mendel but still "
              f"verified by KING kinship and BEAGLE/hap-IBD.")

    print(f"  Written: {path}  ({len(lines)} individuals: "
          f"{n_males} SEX=1, {n_females} SEX=2, {n_unknown} SEX=0)")


def write_chr_rename(path, rename_pairs):
    """
    Write two-column TSV: old_chrom_name  new_chrom_name
    Used by bcftools annotate --rename-chrs and referenced in run scripts.
    """
    if not rename_pairs:
        print("  Chromosomes already numeric — no rename file needed.")
        return False
    with open(path, 'w') as f:
        for old, new in rename_pairs:
            f.write(f'{old}\t{new}\n')
    print(f"  Written: {path}  ({len(rename_pairs)} chromosomes)")
    return True


def write_plink_script(path, vcf_path, fam_path, chr_rename_path,
                        out_prefix, rename_pairs, n_chr, plink2_exe='./plink2'):
    """Generate a PLINK 2.0 shell script for pedigree verification."""
    # chr_flags used only for the comment; the CHR variable is now self-documented
    # in the script body so users can easily adjust --allow-extra-chr.

    script = f"""#!/usr/bin/env bash
# PLINK 2.0 pedigree verification script
# Generated by prepare_plink_beagle.py
#
# Requirements:
#   plink2  — https://www.cog-genomics.org/plink/2.0/
#
# Non-numeric chromosome names (e.g. ch01–ch{n_chr:02d}) are handled by
# --chr-set {n_chr} no-xy --allow-extra-chr  (no VCF renaming needed).
#
# Usage: bash {os.path.basename(path)}

set -euo pipefail

PLINK2="{plink2_exe}"   # adjust if plink2 is on PATH: PLINK2=plink2
VCF="{vcf_path}"
FAM="{fam_path}"
OUT="{out_prefix}"

# Chromosome flags for all plink2 calls.
# --chr-set {n_chr} no-xy  tells PLINK2 this organism has {n_chr} autosomes and no sex chr.
# --allow-extra-chr        needed when chromosome names in the BIM are non-numeric
#                          (e.g. "ch01"…"ch{n_chr:02d}"). If --chr-set alone is sufficient
#                          for your PLINK2 version and chromosome naming, remove it.
CHR="--chr-set {n_chr} no-xy --allow-extra-chr"


# ── Step 1: Convert VCF → PLINK binary format ───────────────────────────────
# --set-all-var-ids assigns unique variant IDs from chromosome:position:ref:alt.
# This is required because VCF files often use '.' for all variant IDs, causing
# PLINK to fail on --indep-pairwise and other operations that need unique IDs.
# Format: @ = chromosome, # = position, $r = ref allele, $a = alt allele.

${{PLINK2}} \\
    --vcf "${{VCF}}" \\
    ${{CHR}} \\
    --double-id \\
    --set-all-var-ids '@:#:$r:$a' \\
    --new-id-max-allele-len 1000 \\
    --make-bed \\
    --out "${{OUT}}_plink_raw"


# ── Step 1b: Quality filter before KING and Mendelian checks ─────────────────
# Genotyping errors corrupt all downstream statistics.
# These filters remove the most error-prone variants BEFORE any relationship
# inference.  Adjust thresholds to your sequencing technology:
#
#   WGS (Illumina, 30×):   --geno 0.05  --maf 0.01  --hwe 1e-10
#   Low-coverage WGS:      --geno 0.10  --maf 0.05  --hwe 1e-6
#   GBS / SNP arrays:      --geno 0.10  --maf 0.05  --hwe 1e-6
#
# --geno: exclude variants missing in > X% of samples
#         Missingness clusters at error-prone sites (repetitive regions,
#         low-complexity sequence, paralogous mappings).
# --maf:  exclude very rare variants (MAF < 1%).
#         Rare variants have higher absolute error rates even if relative
#         error rate is similar; and they contribute little statistical power.
# --hwe:  exclude Hardy-Weinberg violators.
#         Large deviations (p < 1e-10) indicate paralogous loci, systematic
#         miscalling of one genotype class, or contamination.
#         NOTE: in highly inbred lines HWE may be violated legitimately —
#         disable with --nonfounders or raise p-threshold to 1e-4.

${{PLINK2}} \\
    --bfile "${{OUT}}_plink_raw" \\
    ${{CHR}} \\
    --geno 0.05 \\
    --maf 0.01 \\
    --hwe 1e-10 \\
    --make-bed \\
    --out "${{OUT}}_plink"

cp "${{FAM}}" "${{OUT}}_plink.fam"

echo "Variants after quality filtering:"
wc -l "${{OUT}}_plink.bim"
echo "(raw: $(wc -l < ${{OUT}}_plink_raw.bim) variants)"


# ── Step 2: Mendelian error check ────────────────────────────────────────────
# Mendelian errors have two distinct causes:
#   (a) Wrong parent assignment  — errors cluster in a specific family across
#       many variants and chromosomes.
#   (b) Genotyping errors at a locus — errors cluster at a specific variant
#       across many families.
#
# Separating (a) from (b) is critical:
#   .fmendel: per-family rate > 2% with quality-filtered variants → suspect
#             assignment (at 0.5% genotype error, expected rate ≈ 1%)
#   .mendel:  per-variant rate in > 10% of families → filter that variant
#             before re-checking families
#
# Rule: first remove high-error VARIANTS, then re-check FAMILIES.
# The script runs both and generates a cleaned file for follow-up.

${{PLINK2}} \\
    --bfile "${{OUT}}_plink" \\
    ${{CHR}} \\
    --mendel \\
    --out "${{OUT}}_mendel"

echo ""
echo "Mendelian error files:"
echo "  ${{OUT}}_mendel.mendel    — per-variant errors (high rate → genotyping artifact)"
echo "  ${{OUT}}_mendel.imendel   — per-individual errors"
echo "  ${{OUT}}_mendel.fmendel   — per-family errors (high rate → wrong parent)"

# Filter out variants with high Mendelian error rate across families.
# These are genotyping artifacts, not pedigree errors.
# Threshold: variants present in Mendelian-error families > 10% of the time
# are excluded before re-running KING (avoids corrupting kinship estimates).
echo ""
echo "Extracting high-Mendelian-error variants for exclusion..."
# PLINK2 .mendel format: FID IID PAT MAT CHROM SNP BP GENOTYPES
# Column 6 is the variant ID; count occurrences across families
awk 'NR>1 {{print $6}}' "${{OUT}}_mendel.mendel" | sort | uniq -c | sort -rn | \\
    awk -v ntrios=$(wc -l < "${{FAM}}") '{{if ($1/ntrios > 0.10) print $2}}' \\
    > "${{OUT}}_mendel_artifact_snps.txt"
NARTIFACT=$(wc -l < "${{OUT}}_mendel_artifact_snps.txt")
echo "  $NARTIFACT variants flagged as likely genotyping artifacts (Mendelian error in > 10% of families)"
echo "  These are excluded before KING kinship computation."
echo ""
echo "Expected Mendelian error rates for a TRUE parent-offspring pair:"
echo "  Genotype error rate 0.1%  →  expected family Mendelian error ≈ 0.2%"
echo "  Genotype error rate 0.5%  →  expected family Mendelian error ≈ 1.0%"
echo "  Genotype error rate 1.0%  →  expected family Mendelian error ≈ 2.0%"
echo "  Family error rate > 5%    →  likely wrong parent assignment"
echo "  Variant error rate > 10%  →  genotyping artifact, filter before re-checking"


# ── Step 3: KING kinship (replaces --genome in PLINK 2.0) ────────────────────
# --make-king-table outputs the KING-robust kinship coefficient per pair.
#
# !! DEFAULT FILTER WARNING !!
# Without --king-table-filter, PLINK2 silently drops pairs with KINSHIP < 0.0442.
# This default excludes 4th-degree relatives and beyond.
# We run TWO passes with different thresholds:
#   Pass A (0.177): 1st-degree only   →  refine_pedigree.py (fast, focused)
#   Pass B (0.044): up to 3rd-degree  →  extend_pedigree.py (grandparent search)
#
# If you need great-grandparents (4th degree, KINSHIP ~0.031), lower Pass B
# to --king-table-filter 0.022.  Note: at very low thresholds the output can
# be very large for related cohorts (all pairs in a breeding pool are included).
#
# Output columns: #FID1  ID1  FID2  ID2  NSNP  HETHET  IBS0  KINSHIP
#
# KINSHIP thresholds:
#   > 0.354          : duplicate / MZ twin
#   0.177 – 0.354    : 1st degree (parent–offspring OR full sibling)
#   0.0884 – 0.177   : 2nd degree (half sib, grandparent, avuncular)
#   0.0442 – 0.0884  : 3rd degree (1st cousin)
#   0.022  – 0.0442  : 4th degree (great-grandparent, 2nd cousin)
#   < 0.022          : effectively unrelated for pedigree purposes
#
# DISTINGUISHING parent–offspring from full sibling (both KINSHIP ≈ 0.25):
#   IBS0 ≈ 0.00  →  parent–offspring
#   IBS0 > 0.05  →  full sibling
#
# LD-prune first for cleaner estimates:

${{PLINK2}} \\
    --bfile "${{OUT}}_plink" \\
    ${{CHR}} \\
    --indep-pairwise 50 10 0.2 \\
    --out "${{OUT}}_pruned_snps"

# Exclude both LD-correlated SNPs AND Mendelian-artifact SNPs before KING.
# Genotyping errors at artifact SNPs directly inflate IBS0 and corrupt KING.
${{PLINK2}} \\
    --bfile "${{OUT}}_plink" \\
    ${{CHR}} \\
    --extract "${{OUT}}_pruned_snps.prune.in" \\
    --exclude "${{OUT}}_mendel_artifact_snps.txt" \\
    --make-bed \\
    --out "${{OUT}}_pruned"

echo "Variants in cleaned set for KING: $(wc -l < ${{OUT}}_pruned.bim)"

# Pass A: 1st-degree pairs only — used by refine_pedigree.py for PO confirmation
${{PLINK2}} \\
    --bfile "${{OUT}}_pruned" \\
    ${{CHR}} \\
    --make-king-table \\
    --king-table-filter 0.177 \\
    --out "${{OUT}}_king_1st"

echo "KING 1st-degree: ${{OUT}}_king_1st.kin0"
echo "  → Input for: python utilities/refine_pedigree.py prefix ${{OUT}}_king_1st"

# Pass B: up to 3rd-degree — used by extend_pedigree.py for grandparent/offspring search
# For great-grandparent search: lower to --king-table-filter 0.022
${{PLINK2}} \\
    --bfile "${{OUT}}_pruned" \\
    ${{CHR}} \\
    --make-king-table \\
    --king-table-filter 0.044 \\
    --out "${{OUT}}_king_ext"

echo "KING up to 3rd-degree: ${{OUT}}_king_ext.kin0"
echo "  → Input for: python utilities/extend_pedigree.py refined.fam ${{OUT}}_king_ext.kin0"
echo ""
echo "Quick filters on Pass B output:"
echo "  1st-degree PO :  awk '\\$8 > 0.177 && \\$7 < 0.02' ${{OUT}}_king_ext.kin0"
echo "  1st-degree sib:  awk '\\$8 > 0.177 && \\$7 > 0.05' ${{OUT}}_king_ext.kin0"
echo "  2nd-degree    :  awk '\\$8 > 0.0884 && \\$8 <= 0.177' ${{OUT}}_king_ext.kin0"
echo "  3rd-degree    :  awk '\\$8 > 0.044  && \\$8 <= 0.0884' ${{OUT}}_king_ext.kin0"


# ── Step 4: Per-sample genotyping quality (--het / --missing) ───────────────
# Rules out an alternative explanation for a "repeat offender" parent flagged
# by analyze_mendel_errors.py: elevated Mendelian error across every trio an
# individual appears in COULD mean a wrong-parent/pedigree problem, but the
# identical symptom also arises from poor genotyping quality alone (elevated
# missingness, or elevated heterozygosity from sample contamination) -- a
# completely different remedy (re-genotyping/QC exclusion, not pedigree
# reassignment).
#
# --het runs on the SAME LD-pruned set used for KING (correlated SNPs bias
# the inbreeding coefficient estimate, same reason KING is LD-pruned).
# --missing runs on the full quality-filtered set (understating missingness
# by restricting to independent SNPs would hide genuinely poorly-sequenced
# samples).

${{PLINK2}} \\
    --bfile "${{OUT}}_pruned" \\
    ${{CHR}} \\
    --het \\
    --out "${{OUT}}_plink"

${{PLINK2}} \\
    --bfile "${{OUT}}_plink" \\
    ${{CHR}} \\
    --missing \\
    --out "${{OUT}}_plink"

echo ""
echo "Per-sample genotyping quality files:"
echo "  ${{OUT}}_plink.het    — per-individual O(HOM)/E(HOM)/F (inbreeding coefficient estimate)"
echo "  ${{OUT}}_plink.smiss  — per-individual missingness (F_MISS)"
echo "  → Cross-reference with: python utilities/check_genotype_quality.py ${{OUT}}_plink.het ${{OUT}}_plink.smiss"
echo ""
echo "Done. See PEDIGREE_VERIFICATION.md for interpretation guidance."



"""

    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f"  Written: {path}")


def write_beagle_script(path, vcf_path, fam_path, chr_rename_path,
                         out_prefix, rename_pairs, n_chr,
                         beagle_exe='beagle', hapibd_exe='hap-ibd', cm_per_mb=1.0):
    """Generate a shell script for BEAGLE phasing and IBD segment detection."""
    # BEAGLE 5.x and hap-IBD accept any chromosome names natively.
    # No renaming step needed regardless of chromosome naming convention.

    cat_cmd = 'zcat' if vcf_path.endswith('.gz') else 'cat'

    script = f"""#!/usr/bin/env bash
# BEAGLE phasing + hap-IBD segment detection script
# Generated by prepare_plink_beagle.py
#
# BEAGLE and hap-IBD accept non-numeric chromosome names (ch01..ch{n_chr:02d}) natively.
# No chromosome renaming is required.
#
# Requirements:
#   beagle   — https://faculty.washington.edu/browning/beagle/beagle.html
#   hap-ibd  — https://github.com/browning-lab/hap-ibd
#   tabix    — part of htslib
#
# Usage: bash {os.path.basename(path)}

set -euo pipefail

VCF="{vcf_path}"
FAM="{fam_path}"
OUT="{out_prefix}"
BEAGLE="{beagle_exe}"    # conda: beagle  |  jar: java -Xmx16g -jar beagle.jar
HAPIBD="{hapibd_exe}"   # conda: hap-ibd |  jar: java -Xmx16g -jar hap-ibd.jar

INPUT_VCF="${{VCF}}"


# ── Step 1c: Generate genetic map (REQUIRED by hap-IBD) ──────────────────────
# hap-IBD's min-seed/min-extend/min-output thresholds are specified in
# centimorgans (cM), so it requires a genetic map — unlike BEAGLE phasing,
# where a map is optional (defaults to a uniform 1 cM/Mb assumption).
#
# If you have a species-specific genetic map (PLINK format: chrom, marker,
# cM, bp), set MAP to its path below and skip the generation step.
# Otherwise this generates a SYNTHETIC map assuming a UNIFORM recombination
# rate of {cm_per_mb} cM/Mb. This is a rough approximation:
#   - bp-based interpretation (segment start/end, total bp shared) is unaffected
#   - cM-based thresholds (min-seed etc.) and reported segment cM lengths
#     depend on this assumed rate — adjust --cm-per-mb in prepare_plink_beagle.py
#     if you know your species' approximate recombination rate
#   - hap-IBD interpolates linearly between map points; two points per
#     chromosome (start, end) is sufficient for a uniform-rate approximation

MAP="${{OUT}}_synthetic.map"

if [ ! -f "${{MAP}}" ]; then
  echo "Generating synthetic genetic map (uniform {cm_per_mb} cM/Mb)..."
  echo "Replace MAP= above with a real genetic map if available for accurate segment lengths."
  {cat_cmd} "${{VCF}}" | awk -v rate="{cm_per_mb}" 'BEGIN{{OFS="\\t"}}
      !/^#/ {{
          chrom=$1; pos=$2
          if (!(chrom in minpos) || pos<minpos[chrom]) minpos[chrom]=pos
          if (!(chrom in maxpos) || pos>maxpos[chrom]) maxpos[chrom]=pos
      }}
      END {{
          for (c in minpos) {{
              cm_start = minpos[c] / 1e6 * rate
              cm_end   = maxpos[c] / 1e6 * rate
              print c, ".", cm_start, minpos[c]
              print c, ".", cm_end, maxpos[c]
          }}
      }}' > "${{MAP}}"
  echo "Synthetic map written: ${{MAP}} ($(wc -l < ${{MAP}}) lines, $(cut -f1 ${{MAP}} | sort -u | wc -l) chromosomes)"
fi


# ── PHASING OVERVIEW ─────────────────────────────────────────────────────────
# Two phasing strategies are used in sequence:
#
# Step 2a: STATISTICAL phasing (no pedigree) — screens ALL individuals,
#          including those not yet assigned to trios/families.
#          Use this output with extend_pedigree.py and refine_pedigree.py
#          to find grandparent/offspring candidates.
#          Long IBD segments (>10 Mb) are reliably detected even with switch
#          errors; short segments (<5 Mb) may be fragmented.
#
# Step 2b: PEDIGREE-AWARE phasing (using .fam) — re-phases with Mendelian
#          constraints for known trios.  Trio members get near-zero switch
#          error; unrelated individuals get the same quality as Step 2a.
#          Use this output for confirming known relationships and for
#          grandparent/half-sibling segment-length distinction.
#
# Run both steps.  Use Step 2a output for candidate screening (extend/refine),
# use Step 2b output for relationship confirmation.
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 2a: Statistical phasing (no pedigree — screens all individuals) ─────
# This phase allows IBD detection for ALL individuals, including those without
# pedigree assignments.  Grandparent / offspring candidates identified by
# extend_pedigree.py can be validated here before updating the pedigree.

"${{BEAGLE}}" \\
    gt="${{INPUT_VCF}}" \\
    out="${{OUT}}_stat_phased" \\
    nthreads=4 \\
    iterations=10 \\
    burnin=3

tabix -p vcf "${{OUT}}_stat_phased.vcf.gz"
echo "Statistical phased VCF: ${{OUT}}_stat_phased.vcf.gz"

# hap-IBD on statistically phased VCF (screens all individuals)
# Lower min-seed/min-output to catch long segments across all sample pairs.
"${{HAPIBD}}" \\
    gt="${{OUT}}_stat_phased.vcf.gz" \\
    map="${{MAP}}" \\
    out="${{OUT}}_stat_ibd" \\
    min-seed=5.0 \\
    min-extend=2.0 \\
    min-output=5.0 \\
    nthreads=4

echo "Screening IBD (all individuals): ${{OUT}}_stat_ibd.ibd.gz"
echo "  → Use with: python utilities/extend_pedigree.py refined.fam king.kin0 --ibd ${{OUT}}_stat_ibd.ibd.gz"
echo ""


# ── Step 2b: Pedigree-aware phasing (known trios — high accuracy) ────────────
# BEAGLE uses the .fam pedigree to phase parent-offspring trios with
# Mendelian constraints.  Results:
#   Trio members:  near-zero switch errors (Mendelian constraints propagate phase)
#   Duo members:   good accuracy (partial Mendelian constraints)
#   Unrelated:     same statistical accuracy as Step 2a
#
# IMPORTANT: Only individuals currently in the .fam file benefit from
# pedigree-aware phasing.  After extend_pedigree.py adds new grandparents /
# offspring, update the .fam and re-run this step for those new members.

"${{BEAGLE}}" \\
    gt="${{INPUT_VCF}}" \\
    ped="${{FAM}}" \\
    out="${{OUT}}_phased" \\
    nthreads=4 \\
    iterations=10 \\
    burnin=3

tabix -p vcf "${{OUT}}_phased.vcf.gz"
echo "Pedigree-aware phased VCF: ${{OUT}}_phased.vcf.gz"


# ── Step 3: IBD segment detection on pedigree-phased VCF ─────────────────────
# For relationship confirmation.  Lower thresholds than Step 2a because
# pedigree-phased members have accurate phases — shorter segments are reliable.
# Uses the same MAP generated/set in Step 1c above.

"${{HAPIBD}}" \\
    gt="${{OUT}}_phased.vcf.gz" \\
    map="${{MAP}}" \\
    out="${{OUT}}_ibd" \\
    min-seed=2.0 \\
    min-extend=1.0 \\
    min-output=2.0 \\
    nthreads=4

echo ""
echo "Confirmation IBD (pedigree-phased): ${{OUT}}_ibd.ibd.gz"
echo "HBD inbreeding segments:            ${{OUT}}_ibd.hbd.gz"
echo ""
echo "IBD file format (tab-separated):"
echo "  sample1 haplotype1 sample2 haplotype2 chromosome start(bp) end(bp) LOD"


# ── Step 4: Summarise IBD sharing per pair ───────────────────────────────────
# Total IBD sharing between a parent–offspring pair should be ~50% of genome.
# Run this quick summary in R or Python after step 3.

cat << RSCRIPT > "${{OUT}}_ibd_summary.R"
library(data.table)
ibd <- fread("${{OUT}}_ibd.ibd.gz",
             col.names=c("id1","h1","id2","h2","chrom","start","end","lod"))
ibd[, len := end - start]

# Total IBD length per pair (merge both haplotype combos)
pair_ibd <- ibd[, .(total_ibd_bp = sum(len)), by=.(id1, id2)]
pair_ibd[, total_ibd_Mb := total_ibd_bp / 1e6]
pair_ibd <- pair_ibd[order(-total_ibd_Mb)]

cat("Top pairs by total IBD sharing (Mb):\\n")
print(head(pair_ibd, 20))
fwrite(pair_ibd, "${{OUT}}_ibd_per_pair.tsv", sep="\\t")
cat("\\nWritten: ${{OUT}}_ibd_per_pair.tsv\\n")
RSCRIPT

echo "R summary script: ${{OUT}}_ibd_summary.R"
echo "Run with: Rscript ${{OUT}}_ibd_summary.R"


# ── Step 5: HBD (inbreeding) summary ─────────────────────────────────────────
cat << RSCRIPT > "${{OUT}}_hbd_summary.R"
library(data.table)
hbd <- fread("${{OUT}}_ibd.hbd.gz",
             col.names=c("id1","h1","id2","h2","chrom","start","end","lod"))
hbd[, len := end - start]
indiv_hbd <- hbd[, .(hbd_Mb = sum(len)/1e6, n_segments = .N), by=id1]
indiv_hbd <- indiv_hbd[order(-hbd_Mb)]
cat("Inbreeding (HBD) per individual:\\n")
print(indiv_hbd)
fwrite(indiv_hbd, "${{OUT}}_hbd_per_indiv.tsv", sep="\\t")
RSCRIPT

echo "HBD summary: ${{OUT}}_hbd_summary.R"
echo ""
echo "Done. See PEDIGREE_VERIFICATION.md for interpretation guidance."
"""

    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate PLINK and BEAGLE input files from vcfped output',
        epilog='''
Examples:
  %(prog)s step3_named_filtered input.vcf.gz
  %(prog)s step3_named_filtered input.vcf.gz -o pedigree_check
  %(prog)s step3_named_filtered input.vcf.gz --chr-prefix ch
  %(prog)s --from-fam step3_named_filtered_refined_merged.fam input.vcf.gz -o revalidate
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('trio_prefix', nargs='?', default=None,
                        help='vcfped output prefix (reads <prefix>.trio and optionally '
                             '.gender). Omit if using --from-fam instead.')
    parser.add_argument('vcf',
                        help='VCF or VCF.gz file (used for chromosome names and sample order)')
    parser.add_argument('-o', '--output',
                        help='Output prefix (default: same as trio_prefix, or '
                             '<from-fam path without .fam> if using --from-fam)',
                        default=None)
    parser.add_argument('--from-fam',
                        help='Use an existing PLINK .fam file directly instead of deriving '
                             'one from a vcfped .trio file — e.g. merge_refined_fam.py\'s '
                             'output, which has already-decided PAT/MAT assignments '
                             '(including any reclassified-from-inverted or KING-confirmed '
                             'links vcfped itself never scored) with no vcfped Perc/Test1 '
                             'confidence to read. Confirmed parent-child links (both PAT '
                             'and MAT known) become synthetic full-confidence trios; '
                             'existing SEX values are carried over. Everything else (family '
                             'grouping, role-derived sex for unknowns, Mendelian link '
                             'repair) runs unchanged. Mutually exclusive with trio_prefix.',
                        default=None)
    parser.add_argument('--chr-prefix', default='ch',
                        help='Prefix of chromosome names to strip when renaming '
                             '(e.g. "ch" for ch01→1, "Chr" for Chr1→1; default: ch)')
    parser.add_argument('--plink2', default='./plink2',
                        help='Path or command for plink2 executable '
                             '(default: ./plink2). Use "plink2" if it is on PATH.')
    parser.add_argument('--beagle', default='beagle',
                        help='Command for BEAGLE executable (default: beagle). '
                             'Use "java -Xmx16g -jar /path/to/beagle.jar" for jar installs.')
    parser.add_argument('--hapibd', default='hap-ibd',
                        help='Command for hap-IBD executable (default: hap-ibd). '
                             'Use "java -Xmx16g -jar /path/to/hap-ibd.jar" for jar installs.')
    parser.add_argument('--cm-per-mb', type=float, default=1.0,
                        help='Assumed uniform recombination rate (cM/Mb) for the '
                             'synthetic genetic map generated for hap-IBD, used only '
                             'if you do not have a species-specific genetic map '
                             '(default: 1.0, the standard fallback assumption).')
    args = parser.parse_args()

    if args.trio_prefix and args.from_fam:
        sys.exit("Error: specify either trio_prefix or --from-fam, not both.")
    if not args.trio_prefix and not args.from_fam:
        sys.exit("Error: specify either trio_prefix or --from-fam.")
    if args.from_fam and not os.path.exists(args.from_fam):
        sys.exit(f"Error: {args.from_fam} not found.")
    if args.from_fam and not args.output:
        # Stripping ".fam" from --from-fam and re-adding it for the output
        # .fam would silently overwrite the input file itself (unlike
        # trio_prefix, whose <prefix>.trio input never collides with a
        # <prefix>.fam output). Require an explicit, different prefix.
        sys.exit("Error: --from-fam requires -o/--output (its natural prefix "
                 "would otherwise collide with — and overwrite — the input FAM).")

    out_prefix = args.output or args.trio_prefix

    print("=" * 70)
    print("VCFped → PLINK / BEAGLE file generator")
    print("=" * 70)
    if args.from_fam:
        print(f"  From FAM    : {args.from_fam}")
    else:
        print(f"  Trio prefix : {args.trio_prefix}")
    print(f"  VCF         : {args.vcf}")
    print(f"  Output      : {out_prefix}")
    print()

    # --- Read inputs ---
    print("Reading VCF header...")
    chroms, samples = scan_vcf_header(args.vcf)
    print(f"  {len(chroms)} chromosomes, {len(samples)} samples")
    if not samples:
        sys.exit("Error: could not find sample names in VCF #CHROM line.")

    if args.from_fam:
        print(f"Reading existing FAM file: {args.from_fam}...")
        trios = fam_to_trios(args.from_fam)
        print(f"  {len(trios)} confirmed parent-child links (both parents known) "
              f"→ synthetic full-confidence trios")
        gender = read_fam_sex(args.from_fam)
        n_known_sex = sum(1 for v in gender.values() if v is not None)
        print(f"  {n_known_sex} individuals with known sex carried over from the FAM's SEX column")
    else:
        print("Reading trio file...")
        trios = read_trio_file(f"{args.trio_prefix}.trio")
        print(f"  {len(trios)} Regular trio calls")

        gender_path = f"{args.trio_prefix}.gender"
        gender = read_gender_file(gender_path)
        if gender:
            n_m = sum(1 for v in gender.values() if v == 'Male')
            n_f = sum(1 for v in gender.values() if v == 'Female')
            print(f"  Gender file: {n_m} male, {n_f} female, "
                  f"{len(gender)-n_m-n_f} unknown")
        else:
            print(f"  No gender file found at {gender_path} — sex coded as 0 (unknown)")
    print()

    # --- Build chromosome rename map ---
    rename_pairs = build_chr_rename(chroms, args.chr_prefix)

    # --- Build pedigree ---
    print("Building pedigree structure...")
    fam_of, pat_of, mat_of, ordered_samples, warnings, merged_sex = build_pedigree(
        trios, gender, samples)

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")

    n_with_parents = sum(1 for s in ordered_samples
                         if pat_of.get(s, '0') != '0' or mat_of.get(s, '0') != '0')
    n_founders = len(ordered_samples) - n_with_parents
    n_sex_derived = sum(1 for s in ordered_samples
                        if merged_sex.get(s) is not None and gender.get(s) is None)
    print(f"  {len(ordered_samples)} individuals: "
          f"{n_founders} founders, {n_with_parents} with ≥1 known parent")
    if n_sex_derived:
        print(f"  {n_sex_derived} individuals assigned structural sex from parent role "
              f"(SEX=1 for PAT, SEX=2 for MAT) — required by PLINK2 --mendel. "
              f"No biological sex implied.")
    print()

    # --- Write output files ---
    print("Writing output files...")

    fam_path        = f"{out_prefix}.fam"
    chr_rename_path = f"{out_prefix}_chr_rename.txt"
    plink_script    = f"{out_prefix}_run_plink.sh"
    beagle_script   = f"{out_prefix}_run_beagle.sh"

    write_fam(fam_path, ordered_samples, fam_of, pat_of, mat_of, merged_sex)

    has_rename = write_chr_rename(chr_rename_path, rename_pairs)

    write_plink_script(plink_script, args.vcf, fam_path,
                        chr_rename_path if has_rename else None,
                        out_prefix, rename_pairs, len(chroms),
                        plink2_exe=args.plink2)

    write_beagle_script(beagle_script, args.vcf, fam_path,
                         chr_rename_path if has_rename else None,
                         out_prefix, rename_pairs, len(chroms),
                         beagle_exe=args.beagle, hapibd_exe=args.hapibd,
                         cm_per_mb=args.cm_per_mb)

    print()
    print("Summary:")
    print(f"  {fam_path}           — PLINK/BEAGLE pedigree file")
    if has_rename:
        rename_example = f"{rename_pairs[0][0]} → {rename_pairs[0][1]}"
        print(f"  {chr_rename_path}  — chr rename ({rename_example}, …)")
    print(f"  {plink_script}  — PLINK verification workflow")
    print(f"  {beagle_script}  — BEAGLE phasing + IBD workflow")
    print()
    print("Next steps:")
    print("  1. Run PLINK:  bash " + plink_script)
    print("  2. Run BEAGLE: bash " + beagle_script)
    print("  3. See PEDIGREE_VERIFICATION.md for interpretation guidance.")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
