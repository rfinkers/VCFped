#!/usr/bin/env python3
"""
Generate synthetic VCF + ancillary files for VCFped integration testing.

Scenarios covered
-----------------
clean_trio   TRI_P1 + TRI_P2 → TRI_C1        (outbred, T1/T2 both pass)
three_gen    GP1 + GP2 → P1; P1 + P2 → C1   (grandparent chain, inverted trios expected)
inbred       F6 line1 × F6 line2 → F1 hybrid  (low T2 heterozygous loci)
mz_twin      MZ_A; MZ_B = clone of MZ_A       (sequencing-error duplicates)
half_sib     HS_DAD × HS_MOM1 → HS_C1         (shared father)
             HS_DAD × HS_MOM2 → HS_C2
selfing      SELF_P1 × SELF_P1 → SELF_C1      (plant self-fertilisation)
control      CTRL1                              (unrelated negative control)

Outputs (written to the same directory as this script)
------------------------------------------------------
integration.vcf             — multi-sample VCF (2000 biallelic SNPs, 21 samples)
integration_true.ped        — ground-truth PLINK/BEAGLE FAM-format pedigree
integration_king.kin0       — KING kinship table computed from VCF genotypes

Usage
-----
  python generate_integration_vcf.py           # default settings
  python generate_integration_vcf.py --seed 7 --n-variants 5000
"""

import argparse
import os
import sys
from itertools import combinations

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Pedigree definition
# (id, pat_id | "0", mat_id | "0", sex 1=M/2=F/0=U, scenario_tag)
# ---------------------------------------------------------------------------

PEDIGREE = [
    # Clean regular trio (outbred founders)
    ("TRI_P1",   "0",        "0",        1, "clean_trio"),
    ("TRI_P2",   "0",        "0",        2, "clean_trio"),
    ("TRI_C1",   "TRI_P1",   "TRI_P2",   0, "clean_trio"),

    # Three-generation chain — produces inverted/generational trio calls
    ("GEN3_GP1", "0",        "0",        1, "three_gen"),
    ("GEN3_GP2", "0",        "0",        2, "three_gen"),
    ("GEN3_P1",  "GEN3_GP1", "GEN3_GP2", 1, "three_gen"),
    ("GEN3_P2",  "0",        "0",        2, "three_gen"),   # unrelated second parent
    ("GEN3_C1",  "GEN3_P1",  "GEN3_P2",  0, "three_gen"),

    # F6 inbred lines + F1 hybrid
    ("INBRD_L1", "0",        "0",        0, "inbred"),
    ("INBRD_L2", "0",        "0",        0, "inbred"),
    ("INBRD_F1", "INBRD_L1", "INBRD_L2", 0, "inbred"),

    # MZ twins / clonal duplicates
    ("MZ_A",     "0",        "0",        0, "mz_twin"),
    ("MZ_B",     "0",        "0",        0, "mz_twin"),    # clone of MZ_A

    # Half-siblings sharing HS_DAD
    ("HS_DAD",   "0",        "0",        1, "half_sib"),
    ("HS_MOM1",  "0",        "0",        2, "half_sib"),
    ("HS_MOM2",  "0",        "0",        2, "half_sib"),
    ("HS_C1",    "HS_DAD",   "HS_MOM1",  0, "half_sib"),
    ("HS_C2",    "HS_DAD",   "HS_MOM2",  0, "half_sib"),

    # Self-fertilisation (plant-specific; tests prepare_plink_beagle sex handling)
    ("SELF_P1",  "0",        "0",        0, "selfing"),
    ("SELF_C1",  "SELF_P1",  "SELF_P1",  0, "selfing"),

    # Unrelated negative control
    ("CTRL1",    "0",        "0",        0, "control"),
]

N_INBRED_SELF_GENS = 6   # F6 → ~97% homozygous
MZ_ERROR_RATE     = 0.001 # sequencing error rate for MZ_B vs MZ_A
MISSING_RATE      = 0.01  # per-genotype missingness probability

CHROM    = "chr1"
START_BP = 1_000_000
STEP_BP  = 45_000   # ~45 kb between SNPs (avoids LD while keeping all on one chrom)


# ---------------------------------------------------------------------------
# Genotype simulation
# ---------------------------------------------------------------------------

def _founder_genotypes(n_variants, allele_freqs, rng):
    """Sample n_variants diploid genotypes from Hardy–Weinberg equilibrium."""
    p = allele_freqs
    a1 = (rng.random(n_variants) < p).astype(np.int8)
    a2 = (rng.random(n_variants) < p).astype(np.int8)
    return np.stack([a1, a2], axis=1)  # (n_variants, 2)


def _mendelian_child(pat_g, mat_g, rng):
    """Mendelian segregation: draw one allele from each parent."""
    n = len(pat_g)
    a1 = pat_g[np.arange(n), rng.integers(2, size=n)]
    a2 = mat_g[np.arange(n), rng.integers(2, size=n)]
    return np.stack([a1, a2], axis=1)


def _self_fertilize(g, n_gens, rng):
    """Apply n_gens rounds of self-fertilisation to reach inbred state."""
    for _ in range(n_gens):
        g = _mendelian_child(g, g, rng)
    return g


def simulate_all_genotypes(n_variants, rng):
    """
    Returns dict: sample_id → genotype array (n_variants, 2), int8 {0,1}.
    Missing genotypes applied afterward (set to -1).
    """
    allele_freqs = rng.uniform(0.15, 0.45, size=n_variants)

    # Topological order: founders before offspring
    pat_of = {s[0]: s[1] for s in PEDIGREE}
    mat_of = {s[0]: s[2] for s in PEDIGREE}
    G = {}

    def get_g(sid):
        if sid in G:
            return G[sid]
        pat, mat = pat_of[sid], mat_of[sid]
        if pat == "0" and mat == "0":
            g = _founder_genotypes(n_variants, allele_freqs, rng)
        elif pat == mat:
            # Self-fertilisation
            g = _mendelian_child(get_g(pat), get_g(pat), rng)
        else:
            g = _mendelian_child(get_g(pat), get_g(mat), rng)
        G[sid] = g
        return g

    # Process all samples except special cases
    for sid, *_ in PEDIGREE:
        if sid not in ("MZ_B", "INBRD_L1", "INBRD_L2", "INBRD_F1"):
            get_g(sid)

    # Inbred founders: start from HWE then self-fertilise to F6
    for sid in ("INBRD_L1", "INBRD_L2"):
        base = _founder_genotypes(n_variants, allele_freqs, rng)
        G[sid] = _self_fertilize(base, N_INBRED_SELF_GENS, rng)

    G["INBRD_F1"] = _mendelian_child(G["INBRD_L1"], G["INBRD_L2"], rng)

    # MZ_B: copy of MZ_A with sequencing error
    mz_b = G["MZ_A"].copy()
    err = rng.random((n_variants, 2)) < MZ_ERROR_RATE
    mz_b[err] = 1 - mz_b[err]
    G["MZ_B"] = mz_b

    # Apply missing genotypes
    for sid in G:
        mask = rng.random(n_variants) < MISSING_RATE
        G[sid][mask] = -1

    return G, allele_freqs


# ---------------------------------------------------------------------------
# KING kinship computation
# ---------------------------------------------------------------------------

def compute_king(g1, g2):
    """
    PLINK2 KING kinship and IBS0 from two genotype arrays (n_variants, 2).
    Missing genotypes (-1 in either sample) are excluded.
    Returns (kinship, ibs0, hethet, n_snp) or (None, None, 0, 0) if insufficient data.
    """
    valid = (g1[:, 0] >= 0) & (g1[:, 1] >= 0) & (g2[:, 0] >= 0) & (g2[:, 1] >= 0)
    g1v, g2v = g1[valid], g2[valid]
    n = len(g1v)
    if n < 100:
        return None, None, 0, n

    s1 = g1v.sum(axis=1)  # 0, 1, 2
    s2 = g2v.sum(axis=1)

    is_het1 = s1 == 1
    is_het2 = s2 == 1
    N_HetHet = int((is_het1 & is_het2).sum())
    N_IBS0   = int(((s1 == 0) & (s2 == 2) | (s1 == 2) & (s2 == 0)).sum())
    N_Het1   = int(is_het1.sum())
    N_Het2   = int(is_het2.sum())

    denom = 2 * min(N_Het1, N_Het2)
    kinship = (N_HetHet - 2 * N_IBS0) / denom if denom > 0 else None

    n_hom = int(((s1 != 1) & (s2 != 1)).sum())
    ibs0 = N_IBS0 / n_hom if n_hom > 0 else None

    hethet_frac = N_HetHet / n if n > 0 else 0.0

    return kinship, ibs0, hethet_frac, n


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_vcf(path, sample_ids, genotypes, allele_freqs, n_variants):
    bases = [("A", "T"), ("A", "C"), ("G", "T"), ("G", "C"), ("A", "G"), ("C", "T")]
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write(f"##contig=<ID={CHROM},length={START_BP + n_variants * STEP_BP}>\n")
        f.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for sid in sample_ids:
            f.write(f"\t{sid}")
        f.write("\n")

        for i in range(n_variants):
            pos = START_BP + i * STEP_BP
            ref, alt = bases[i % len(bases)]
            # Swap so REF is the major allele (allele_freqs[i] = P(alt))
            row = [CHROM, str(pos), ".", ref, alt, "100", "PASS", ".", "GT"]
            for sid in sample_ids:
                g = genotypes[sid][i]
                if g[0] < 0:
                    row.append("./.")
                else:
                    row.append(f"{g[0]}/{g[1]}")
            f.write("\t".join(row) + "\n")
    print(f"  Written: {path}")


def write_ped(path, pedigree):
    """Write ground-truth pedigree in PLINK FAM format (6 columns, space-delimited)."""
    with open(path, "w") as f:
        for sid, pat, mat, sex, _ in pedigree:
            fid = "FAM"
            f.write(f"{fid} {sid} {pat} {mat} {sex} -9\n")
    print(f"  Written: {path}")


def write_king(path, sample_ids, genotypes, min_kinship=0.044):
    """Write synthetic KING --make-king-table .kin0 file for all pairs >= min_kinship."""
    rows = []
    for id1, id2 in combinations(sample_ids, 2):
        kinship, ibs0, hethet, nsnp = compute_king(genotypes[id1], genotypes[id2])
        if kinship is None or kinship < min_kinship:
            continue
        rows.append({
            "#FID1": "FAM", "ID1": id1,
            "#FID2": "FAM", "ID2": id2,
            "NSNP":  nsnp,
            "HETHET": f"{hethet:.4f}",
            "IBS0":   f"{ibs0:.4f}" if ibs0 is not None else "0.0000",
            "KINSHIP": f"{kinship:.4f}",
        })

    fieldnames = ["#FID1", "ID1", "#FID2", "ID2", "NSNP", "HETHET", "IBS0", "KINSHIP"]

    # load_king() looks for a column containing 'ID1' but not 'FID'.
    # The header uses '#FID1' and 'ID1' — the latter matches.
    # Re-map to match PLINK2 actual output header exactly.
    header_line = "\t".join(["#FID1", "ID1", "FID2", "ID2",
                              "NSNP", "HETHET", "IBS0", "KINSHIP"])
    with open(path, "w") as f:
        f.write(header_line + "\n")
        for r in rows:
            f.write("\t".join([
                r["#FID1"], r["ID1"], r["#FID2"], r["ID2"],
                str(r["NSNP"]), r["HETHET"], r["IBS0"], r["KINSHIP"],
            ]) + "\n")
    print(f"  Written: {path}  ({len(rows)} pairs >= {min_kinship})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate VCFped integration test files")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-variants", type=int, default=2000,
                        help="Number of SNPs (default 2000)")
    parser.add_argument("--out-dir",    default=SCRIPT_DIR,
                        help="Output directory (default: same as this script)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sample_ids = [s[0] for s in PEDIGREE]

    print(f"Simulating {args.n_variants} SNPs for {len(sample_ids)} samples "
          f"(seed={args.seed}) ...")
    genotypes, allele_freqs = simulate_all_genotypes(args.n_variants, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    vcf_path  = os.path.join(args.out_dir, "integration.vcf")
    ped_path  = os.path.join(args.out_dir, "integration_true.ped")
    king_path = os.path.join(args.out_dir, "integration_king.kin0")

    write_vcf(vcf_path, sample_ids, genotypes, allele_freqs, args.n_variants)
    write_ped(ped_path, PEDIGREE)
    write_king(king_path, sample_ids, genotypes)

    print("\nScenarios in integration.vcf:")
    for tag in dict.fromkeys(s[4] for s in PEDIGREE):
        members = [s[0] for s in PEDIGREE if s[4] == tag]
        print(f"  {tag:12s}: {', '.join(members)}")
    print("\nDone. Commit these files or re-generate with --seed / --n-variants.")


if __name__ == "__main__":
    main()
