# Pedigree Verification Guide

Using PLINK 2.0 and BEAGLE to confirm and refine vcfped trio calls in a diploid outcrossing species.

---

## Background: what vcfped found and why it needs verification

VCFped detects trios using IBS allele-sharing statistics across the whole genome.
It is fast but has two known limitations:

1. **Ambiguous parent assignments** — in a cohort with related individuals, a sibling of the true father can pass the same allele-sharing test. `filter_trios.py` removes lower-confidence duplicates, but the best call is still statistical, not phased.
2. **No Mendelian checks** — vcfped works on genotype frequencies, not variant-by-variant inheritance. A single Mendelian error at one variant does not fail a trio call.

PLINK 2.0 and BEAGLE address both limitations independently and are complementary.

---

## Genotyping errors and how they affect each statistic

Genotyping errors are present in all sequencing technologies. Understanding how they corrupt each statistic — and what the tolerances are — determines how much you can trust any single line of evidence.

### How errors propagate per statistic

| Statistic | Effect of a single genotyping error | Threshold implication |
|---|---|---|
| **IBS0** | A miscalled parent (true `AB` → called `BB`) paired with a correctly called child (`AA`) creates a false IBS0 hit | Expected IBS0 for true PO ≈ 2ε where ε is the per-genotype error rate |
| **Mendelian errors** | Each miscalled genotype can create a Mendelian violation | Expected rate for true PO ≈ 2ε; use 5× this as the rejection threshold |
| **vcfped T1/T2** | An AA parent miscalled as AB removes one valid AA+BB site | T1 decreases slightly; use T1 as a relative ranking, not absolute |
| **KING KINSHIP** | Inflated by errors at rare variants (homozygotes miscalled) | Quality filter before KING; errors inflate KINSHIP slightly |
| **Phased IBD segment length** | A single error mid-segment causes a "switch error" that splits one long segment into two short ones | `min-seed` in hap-IBD filters the short artifactual segments |
| **Genome coverage** | Switch errors create apparent gaps in coverage | Statistical phasing coverage < pedigree-aware phasing; coverage rises with better phasing |

### IBS0 noise ceiling by technology

The `--error-rate` parameter in `refine_pedigree.py` computes the maximum expected IBS0 for a *true* parent-offspring pair given your data quality. Default = 0.025 (conservative).

| Technology | Typical error rate (ε) | IBS0 noise ceiling (≈ 2ε) |
|---|---|---|
| WGS 30× Illumina | 0.001 (0.1%) | 0.002 |
| WGS 10× Illumina | 0.005 (0.5%) | 0.010 |
| GBS / RAD-seq | 0.010 (1.0%) | 0.020 |
| SNP array (dense) | 0.002 (0.2%) | 0.004 |
| Low-coverage / GBS (plant breeding) | 0.010–0.025 | 0.020–0.050 |

**IBS0 above this ceiling is strong evidence against PO** — the probability that all observed IBS0 hits are genotyping artifacts falls exponentially as IBS0 increases beyond 2ε.

```bash
# Specify your expected error rate:
python utilities/refine_pedigree.py step3_named_filtered king_ext \
    --error-rate 0.010          # GBS / low-quality data
    --error-rate 0.001          # high-quality WGS 30×
```

### Separating genotyping artifacts from pedigree errors in Mendelian output

The PLINK2 `--mendel` output has two distinct patterns:

**Pattern 1: genotyping artifact**
```
Variant V appears as Mendelian error in 40 of 100 families
→ V is a genotyping artifact (repetitive region, paralog, miscalled allele)
→ Action: exclude V from KING and re-check families without it
```

**Pattern 2: wrong parent assignment**
```
Family F shows 8% Mendelian error rate spread across 500 variants on all chromosomes
→ The assigned parent is wrong (8% >> expected 1% from 0.5% error rate)
→ Action: flag family for re-assignment in filter_trios.py + refine_pedigree.py
```

The generated `_run_plink.sh` automatically:
1. Identifies variants with Mendelian error in > 10% of families → flags as artifacts
2. Excludes those artifacts before computing KING kinship
3. Reports the expected error rate range so you can contextualize family-level rates

### Genome coverage and phasing quality

The genome coverage check in `refine_pedigree.py --ibd` is sensitive to phasing quality:

| Phasing input | Expected coverage for true PO | Notes |
|---|---|---|
| Statistical (`*_stat_phased.vcf.gz`) | 80–95% | Switch errors fragment segments; gaps are phasing artifacts |
| Pedigree-aware (`*_phased.vcf.gz`) | 95–100% | Mendelian constraints near-eliminate switch errors in trio members |

**Interpretation:** if coverage is 80–90% with statistical phasing, re-run with pedigree-aware phasing. Coverage that rises to > 95% confirms the assignment. Coverage that stays low (< 80%) even with pedigree-aware phasing indicates a false parent pair, even if IBS0≈0.

---

## Interpreting statistics in related / plant breeding germplasm

### What breaks down with fixed thresholds

All statistics produced by vcfped, PLINK2 KING, and hap-IBD are calculated relative to the allele frequencies in the cohort. In outbred populations with no background relatedness, the expected values are well-calibrated. In closely related breeding pools (elite panels, F2 populations, backcross lines, inbred lines), this breaks down:

| Statistic | Outbred assumption | Reality in related pools |
|---|---|---|
| KING KINSHIP thresholds | Background ≈ 0 | Background kinship 0.10–0.30; many non-PO pairs exceed the "1st-degree" threshold of 0.177 |
| vcfped T1/T2 scores | AA+BB combos frequent | Inbred material is enriched for homozygotes; fewer AA+BB combinations → lower T1 even for true trios |
| IBD total (%) | PO shares exactly 50% | Background IBD inflates the baseline; true PO still highest but margin narrows |

**Higher statistical estimates still indicate a tighter relationship** — the ranking is preserved even if absolute thresholds are not. Use statistics comparatively within the cohort, not against outbred reference values.

### What remains valid in any population

| Metric | Why it holds | How to use |
|---|---|---|
| **IBS0 ≈ 0** | A parent MUST share ≥1 allele with their offspring at every variant (Mendelian law). IBS0 is physically impossible for true PO. | Hard filter: any IBS0 > 0.05 definitively rules out PO regardless of relatedness |
| **Cohort-relative kinship rank** | True PO pairs cluster at the top of the kinship distribution even if the absolute values are inflated | Sort all pairs by KINSHIP; PO candidates should be in the top percentiles |
| **Phased IBD block coverage** | The child's full genome must tile onto exactly two parental haplotypes. False parents leave gaps even if they share many alleles due to background relatedness | Coverage ≥ 90% = high confidence; < 70% = likely false assignment |

### Evidence hierarchy (most to least reliable)

1. **Phased IBD block coverage** (gold standard): after pedigree-aware BEAGLE phasing, check that both proposed parents cover the child's entire genome with no gaps. Works in any population.
2. **IBS0** (hard criterion): rules out false parents definitively. Not affected by background relatedness.
3. **Mendelian error rate** (PLINK2 `--mendel`): variant-by-variant inheritance check. Low error rate confirms PO; high error rate rules it out.
4. **T1/T2 from vcfped**: higher scores indicate stronger allele-sharing signal. Treat as a relative ranking, not an absolute cutoff.
5. **KING KINSHIP**: use as a relative rank within the cohort. Absolute thresholds are unreliable in related pools.

### Practical approach for related germplasm

```bash
# Step 1: Run refine_pedigree.py with cohort-relative ranking enabled
# (default --kinship-min 0 uses IBS0 as the only hard filter)
python utilities/refine_pedigree.py step3_named_filtered king_ext \
    --step1-pair step1_named

# Step 2: After BEAGLE phasing, add IBD coverage check (gold standard)
bash step3_named_filtered_run_beagle.sh   # produces *_stat_ibd.ibd.gz

python utilities/refine_pedigree.py step3_named_filtered king_ext \
    --step1-pair step1_named \
    --ibd step3_named_filtered_stat_ibd.ibd.gz

# The refinement report will show for each trio:
#   IBS0 (hard criterion)
#   Cohort kinship percentile (relative ranking)
#   Genome coverage fraction (gold standard)
```

---

## KING output filter — critical requirement

PLINK2's `--make-king-table` **silently drops pairs below a minimum kinship threshold**. The default threshold is approximately 2^(−4.5) ≈ **0.0442**. Pairs more distantly related than 3rd degree are never written to the `.kin0` file, even though the computation was done.

The generated `_run_plink.sh` runs two KING passes to serve different purposes:

| Output file | `--king-table-filter` | Minimum relationship | Used by |
|---|---|---|---|
| `_king_1st.kin0` | 0.177 | 1st degree (parent–offspring, full sibling) | `refine_pedigree.py` PO confirmation |
| `_king_ext.kin0` | 0.044 | Up to 3rd degree (grandparent, 1st cousin) | `extend_pedigree.py` grandparent/offspring search |

**Use `_king_ext.kin0` with `extend_pedigree.py`**, not `_king_1st.kin0`. Passing the 1st-degree-only file to `extend_pedigree.py` would silently miss all grandparent candidates (KINSHIP 0.0884–0.177) and produce an empty extension.

### What the default filter cuts off

| Relationship | KINSHIP | In `_king_1st` | In `_king_ext` |
|---|---|---|---|
| Parent–offspring | ~0.25 | ✓ | ✓ |
| Full sibling | ~0.25 | ✓ | ✓ |
| Half sibling | ~0.125 | ✗ | ✓ |
| Grandparent–grandchild | ~0.125 | ✗ | ✓ |
| 1st cousin | ~0.0625 | ✗ | ✓ |
| Great-grandparent | ~0.031 | ✗ | ✗ |
| 2nd cousin | ~0.016 | ✗ | ✗ |

### Extending to great-grandparents

If you need a multi-generational pedigree (3+ generations up from current founders), lower the filter:

```bash
plink2 --bfile prefix_pruned \
       --chr-set 25 no-xy --allow-extra-chr \
       --make-king-table \
       --king-table-filter 0.022 \
       --out prefix_king_deep
```

Note: at 0.022, the output for a related breeding pool of 635 samples may include many pairs. For 4000+ samples this becomes large — use `--king-table-filter 0.044` and accept that great-grandparents won't be found automatically.

### Runtime checks in the utilities

Both `refine_pedigree.py` and `extend_pedigree.py` inspect the minimum kinship value in the provided `.kin0` file and print a warning if it appears too aggressively filtered for the intended analysis. If you see such a warning, regenerate the KING output with a lower `--king-table-filter`.

---

## Chromosome naming — PLINK 2.0 vs PLINK 1.9

**PLINK 1.9** requires numeric chromosome names (1–22, X, Y, MT) and would fail on `ch01`–`ch25`.

**PLINK 2.0** handles non-numeric names natively with two flags added to every command:
```bash
--chr-set 25 no-xy --allow-extra-chr
```
- `--chr-set 25 no-xy` tells PLINK 2.0 the organism has 25 autosomes and no sex chromosomes.
- `--allow-extra-chr` permits string-based chromosome names.

**Result: no VCF pre-processing or chromosome rename step is needed** when using PLINK 2.0. The `_chr_rename.txt` file produced by `prepare_plink_beagle.py` is retained as a reference but the generated `_run_plink.sh` script does not use it.

BEAGLE and hap-IBD accept any chromosome names natively — no rename needed there either.

---

## Tool overview

| Tool | What it adds | Key difference from PLINK 1.9 |
|---|---|---|
| **PLINK 2.0 `--mendel`** | Per-variant and per-family Mendelian error counts | Same concept; output format slightly updated |
| **PLINK 2.0 `--make-king-table`** | KING kinship coefficient + IBS0 per pair | Replaces `--genome` (Z0/Z1/Z2); IBS0 distinguishes PO from full sib |
| **BEAGLE phasing** | Haplotype-resolved genotypes using pedigree | Unchanged — accepts ch01-style names natively |
| **hap-IBD** | Exact IBD segment coordinates per pair | Unchanged — no chr rename needed |

---

## Step 0 — prepare input files

```bash
# 1. Filter ambiguous parent assignments
python utilities/filter_trios.py step3_named --strategy best -o step3_named_filtered

# 2. Generate PLINK 2.0 / BEAGLE inputs
python utilities/prepare_plink_beagle.py step3_named_filtered input.vcf.gz --chr-prefix ch
# Writes: step3_named_filtered.fam, _chr_rename.txt, _run_plink.sh, _run_beagle.sh
```

---

## Step 1 — PLINK 2.0: Mendelian error check

```bash
bash step3_named_filtered_run_plink.sh
```

The script runs `--mendel` which produces:

| File | Content |
|---|---|
| `.fmendel` | Errors **per family** — N = Mendelian errors across all variants |
| `.imendel` | Errors **per individual** |
| `.mendel` | Errors **per variant** |

### Interpreting Mendelian errors

| Scenario | Signature |
|---|---|
| **Correct trio, noisy variants** | < 1% error rate; errors scattered across many variants and chromosomes |
| **Wrong parent assigned** | Elevated `.imendel` errors in child; > 5% `.fmendel`; errors on many chromosomes |
| **Genotyping artefact** | High `.mendel` count for one variant across many families |
| **Sample swap** | Very high errors in one individual across ALL families it appears in |

**Rule of thumb:** a confirmed parent–offspring pair should show < 1–2% Mendelian errors after quality filtering. If a family exceeds 5%, the parent assignment is suspect.

---

## Step 2 — PLINK 2.0: KING kinship (replaces `--genome`)

### What changed from PLINK 1.9

PLINK 1.9's `--genome` output Z0/Z1/Z2 (proportion of genome with 0, 1, or 2 alleles IBD).
PLINK 2.0 removes `--genome` and replaces it with `--make-king-table`, which outputs the KING-robust kinship coefficient.

**Output columns of `.kin0`:**
```
#FID1  ID1  FID2  ID2  NSNP  HETHET  IBS0  KINSHIP
```

| Column | Meaning |
|---|---|
| `HETHET` | Proportion of variants where both samples are heterozygous |
| `IBS0` | Proportion with 0 alleles IBS (one sample 0/0, other 1/1) |
| `KINSHIP` | KING kinship coefficient φ |

### KINSHIP thresholds

| Relationship | KINSHIP (φ) |
|---|---|
| Duplicate / MZ twin | > 0.354 |
| 1st degree (PO **or** full sib) | 0.177 – 0.354 |
| 2nd degree (half sib, grandparent, avuncular) | 0.0884 – 0.177 |
| 3rd degree (1st cousin) | 0.0442 – 0.0884 |
| Unrelated | < 0.0442 |

### Distinguishing parent–offspring from full sibling using IBS0

Both relationships give KINSHIP ≈ 0.25, so KINSHIP alone cannot distinguish them.
**IBS0 resolves the ambiguity:**

| Relationship | KINSHIP | IBS0 |
|---|---|---|
| **Parent–offspring** | ~0.25 | **~0.00** |
| **Full sibling** | ~0.25 | **~0.06–0.12** |
| Half sibling | ~0.125 | ~0.03–0.06 |
| Unrelated | ~0 | ~0.08–0.15 |

The reasoning: a parent always shares at least one allele with their offspring at every variant (the child inherits exactly one allele from each parent). IBS0 — which requires both samples to be homozygous for *opposite* alleles — is therefore impossible for a parent–offspring pair. Any non-trivial IBS0 means the pair is **not** parent–offspring.

**Practical filters after `--make-king-table`:**
```bash
# All 1st-degree pairs
awk '$8 > 0.177' results_king.kin0

# Parent–offspring only (IBS0 ≈ 0)
awk '$8 > 0.177 && $7 < 0.02' results_king.kin0

# Full siblings only (IBS0 > 0.05)
awk '$8 > 0.177 && $7 > 0.05' results_king.kin0
```

### 2nd-degree relationships cannot be distinguished by KINSHIP alone

Half-sibling, grandparent–grandchild, and avuncular relationships all give KINSHIP ≈ 0.125 and similar IBS0. Use **hap-IBD segment patterns** (Step 4) to distinguish them.

---

## Step 3 — BEAGLE phasing: two strategies

```bash
bash step3_named_filtered_run_beagle.sh
```

The BEAGLE script runs **two phasing passes** in sequence. Understanding why matters for interpreting hap-IBD results.

### Step 3a — Statistical phasing (no pedigree)

BEAGLE is run first *without* any `ped=` argument. Every individual is phased using LD patterns across the cohort. This produces `*_stat_phased.vcf.gz`.

**Why run this first?**
- It phases *all* individuals — including founders, grandparent candidates, and individuals not yet assigned to any family.
- hap-IBD on this output (`*_stat_ibd.ibd.gz`) screens the full cohort for IBD, enabling `refine_pedigree.py` and `extend_pedigree.py` to find candidate relatives that are not yet in the pedigree.
- The quality per individual: switch error rate ~0.5–5% depending on marker density, cohort size, and LD structure.

**What IBD is reliable with statistical phasing?**

| Segment length | Reliability | Use case |
|---|---|---|
| > 20 Mb | High — survives switch errors | Detect 1st-degree relatives (PO, full sib) for any individual |
| 5–20 Mb | Moderate — some fragmentation | Detect 2nd-degree candidates (grandparents, half-sibs) for screening |
| < 5 Mb | Low — heavily fragmented | Not reliable without pedigree phasing |

**Practical implication:** statistical phasing is good enough to find grandparent and offspring *candidates* (long segments), but not good enough to distinguish grandparent from half-sibling (requires accurate short-segment detection).

### Step 3b — Pedigree-aware phasing (known trios — high accuracy)

BEAGLE is run with `ped=pedigree.fam`. Within confirmed parent–offspring trios, Mendelian constraints fix the phase of all three members simultaneously. This produces `*_phased.vcf.gz`.

**Quality per individual type:**

| Individual type | Phase accuracy | Reason |
|---|---|---|
| Member of complete trio (both parents known) | Near-zero switch errors | Mendelian constraint propagates phase from parents to child and back |
| Member of duo (one parent known) | Good; partial constraints | One parent's phase constrains the child's haplotype |
| Founder / unassigned | Same as statistical (Step 3a) | No pedigree constraint available |
| Grandparent candidate (not yet in .fam) | Same as statistical | Not in pedigree at this stage |

**Key implication:** grandparent candidates identified by `extend_pedigree.py` are *not* in the `.fam` when Step 3b runs, so they still get statistical phasing. To get high-accuracy phase for them, update the `.fam` with the confirmed grandparents and re-run Step 3b.

### Iterative phasing refinement

For the best IBD detection across all generations:

```
1. Statistical phase (Step 3a)  →  hap-IBD on all individuals
                                →  extend_pedigree.py finds candidates
                                →  refine_pedigree.py confirms with KING
2. Update .fam with confirmed grandparents/offspring
3. Re-run Step 3b (pedigree-aware) with expanded .fam
                                →  new grandparents now phased with pedigree constraints
4. hap-IBD again on new phased VCF
                                →  short segments for new members now reliable
5. Repeat from step 1 if further generations need to be added
```

The run script generates both phased VCFs in one pass. After `extend_pedigree.py`, re-run Step 3b only (not 3a) with the updated `.fam`:

```bash
java -Xmx16g -jar beagle.jar \
    gt=input.vcf.gz \
    ped=extended.fam \
    out=prefix_extended_phased \
    nthreads=4 iterations=10 burnin=3

java -Xmx16g -jar hap-ibd.jar \
    gt=prefix_extended_phased.vcf.gz \
    out=prefix_extended_ibd \
    min-seed=2.0 min-extend=1.0 min-output=2.0 nthreads=4
```

**What can be validated with a partially phased VCF?**

Even when many individuals have only statistical phasing (Step 3a), the following analyses are valid:

| Analysis | Statistically phased | Pedigree phased |
|---|---|---|
| Finding all 1st-degree relatives in cohort | ✓ (long segments reliable) | ✓ |
| Confirming PO vs. full sibling | Partial (may need KING IBS0) | ✓ (accurate Z1/Z2 distinction) |
| Grandparent screening (2nd degree) | ✓ (long segments, conservative threshold) | ✓ |
| Grandparent vs. half-sib distinction | ✗ (short segments fragmented) | ✓ |
| Inbreeding / HBD detection | Partial (longer HBD reliable) | ✓ |
| Cross-family IBD (population structure) | ✓ | ✓ |
| Individuals with NO pedigree assignment | ✓ (screening) | Same as statistical |

---

## Step 4 — hap-IBD: IBD segment detection

hap-IBD finds exact chromosomal segments where two haplotypes are identical by descent.

**Output columns of `.ibd.gz`:**
```
sample1  haplotype1  sample2  haplotype2  chromosome  start_bp  end_bp  LOD
```

### Interpreting IBD segments per relationship

| Relationship | IBD segment pattern |
|---|---|
| **Parent–offspring** | Long IBD segments covering ~50% of each chromosome; no chromosome entirely IBD-free; **no Z2 segments** (no chromosome fully shared) |
| **Full sibling** | ~50% total sharing but includes some fully-shared chromosome regions (Z2); visually mosaic |
| **Half sibling** | ~25% sharing; only single-haplotype (Z1) segments |
| **Grandparent–grandchild** | ~25% sharing; segments shorter than half-sib on average due to extra recombination generation |
| **Avuncular (uncle/aunt–niece/nephew)** | ~25% sharing; segment-length distribution similar to grandparent |
| **Unrelated** | Occasional short segments (< 3 cM) from population-level identity; nothing long |

### Distinguishing 2nd-degree relationships with segment length

Grandparent–grandchild and half-sibling both give ~25% IBD, but segment lengths differ:
- **Half sibling**: longer segments on average (one shared parent, recent common ancestor)
- **Grandparent–grandchild**: shorter segments (extra recombination generation)
- **Avuncular**: intermediate to half-sib

A species-specific genetic map (cM per bp) greatly improves this distinction. Without a map, use physical segment lengths as a proxy (longer = more recent common ancestry).

### Detecting inverted/generational trios from segments

If vcfped reported an "Inverted/generational trio" for (A, B, C), inspect hap-IBD:
- If C shares ~50% IBD with both A and B → C is likely the true offspring (regular trio missed by T2 filter)
- If A and B share a long IBD segment themselves → one may be parent of the other (A→B→C chain, not A+B→C)
- If C shares ~25% with only one of A or B → the other is more distant (grandparent, half-relative)

---

## Step 5 — inbreeding (HBD segments)

**Homozygous-by-descent (HBD)** segments occur where an individual's two haplotypes are IBD — indicating the same ancestral segment was inherited from both parents (inbreeding).

In an outcrossing species HBD segments should be rare. High HBD in specific individuals indicates consanguineous mating or population bottleneck.

```bash
Rscript step3_named_filtered_hbd_summary.R
```

---

## IBD block visualisation

Neither PLINK 2.0 nor BEAGLE includes built-in visualisation. Options:

### Option A — R with ggplot2 (recommended, included in scripts)

The BEAGLE run script generates `*_ibd_summary.R`. For chromosome-level painting:

```r
library(data.table); library(ggplot2)
ibd <- fread("prefix_ibd.ibd.gz",
    col.names=c("id1","h1","id2","h2","chrom","start","end","lod"))
pair <- ibd[id1=="SampleA" & id2=="SampleB"]

ggplot(pair, aes(xmin=start/1e6, xmax=end/1e6, ymin=0, ymax=1)) +
    geom_rect(fill="#4A90E2") +
    facet_wrap(~chrom, ncol=1, strip.position="left") +
    labs(x="Position (Mb)", title="IBD segments: SampleA vs SampleB") +
    theme_minimal()
```

### Option B — KING standalone (non-human friendly)

KING ([https://www.kingrelatedness.com/](https://www.kingrelatedness.com/)) is well-suited for non-human organisms and large cohorts. Use `--autosome-num 25` for a 25-chromosome species.

```bash
king --vcf input.vcf.gz --kinship --prefix king_out --autosome-num 25
king --vcf input.vcf.gz --ibdseg  --prefix king_ibd --autosome-num 25
```

KING's `--ibdseg` produces a segment file that can be visualised with the companion R scripts provided on the KING website.

### Option C — R packages `pedtools` + `ibdsim2`

Designed for pedigree-based IBD analysis. Simulates expected IBD patterns under a given pedigree and compares to observed data — useful for confirming complex or uncertain structures.

```r
library(pedtools); library(ibdsim2)
ped <- nuclearPed(nch=2)   # adjust to your pedigree
sims <- ibdsim(ped, N=1000, ids=c("father","child1","child2"))
plot(sims)
```

---

## Step 6 — pedigree refinement (`refine_pedigree.py`)

After PLINK2 KING is run, `refine_pedigree.py` cross-checks every vcfped parent assignment:

```bash
python utilities/refine_pedigree.py step3_named_filtered king_results \
    --step1-pair step1_named --include-inverted
```

### What it checks

1. **KING confirmation** — does each assigned parent have KINSHIP > 0.177 and IBS0 < 0.05 (PO signature)? Flags any that don't.
2. **MZ twin substitution** — if an MZ group member was purged before Step 3, the selected representative may not be the actual biological parent. `refine_pedigree.py` checks whether non-representative members of the MZ group have a better PO signal to the child. *Requires KING to be run on the original pre-purge VCF so all MZ group members appear in the kinship matrix.*
3. **Unchecked candidates** — finds individuals in KING with PO signal to a child that weren't assigned by vcfped (e.g. excluded at a lower stringency level).
4. **Inverted trio direction** — uses KING kinship to determine whether an inverted trio should be reclassified as regular, or represents a grandparent chain.

### Output

- `_refined.trio` — updated assignments (same format; compatible with `fixed_pedigree_viz.py`)
- `_refined.fam` — updated PLINK/BEAGLE pedigree file
- `_refinement_report.txt` — full audit trail with strategy recommendations

---

## Step 7 — pedigree extension (`extend_pedigree.py`)

Searches for grandparents (upward) and offspring (downward) of individuals at the boundaries of the current pedigree:

```bash
python utilities/extend_pedigree.py refined.fam king_results.kin0 \
    --ibd prefix_stat_ibd.ibd.gz --direction both --max-iter 2
```

### Grandparent candidate identification

1. Find all 2nd-degree relatives (KINSHIP 0.0884–0.177) of each founder.
2. Among those, find pairs where the two 2nd-degree relatives are themselves 1st-degree relatives (KINSHIP > 0.177, IBS0 < 0.05) → candidate grandparent pair.
3. If hap-IBD is provided: use mean segment length to distinguish grandparent–grandchild (shorter segments) from half-sibling (longer segments).

### Offspring candidate identification

Finds unassigned individuals with PO signal (KINSHIP > 0.177, IBS0 < 0.05) to known leaf individuals, then searches for a compatible second parent.

### Strategy recommendations (auto-selected)

`extend_pedigree.py` recommends a follow-up strategy based on cohort size and available data:

| Scenario | Recommended strategy |
|---|---|
| Few candidates, KING unambiguous | **KING-only** — accept the extension, update `.fam` |
| Many candidates or KING ambiguous | **vcfped rerun** — use `_vcfped_candidates.txt` to subset the VCF and re-run vcfped on just the candidate samples |
| 2nd-degree ambiguity (grandparent vs. half-sib) | **BEAGLE hap-IBD** — run `_grandparent_check.sh`; compare segment lengths |
| Large cohort with many generations | **Combined** — KING screening → vcfped subset rerun → BEAGLE confirmation |

### vcfped subset rerun (when recommended)

```bash
# Subset VCF to candidate samples
bcftools view -S prefix_vcfped_candidates.txt input.vcf.gz | bgzip > subset.vcf.gz
tabix -p vcf subset.vcf.gz

# Re-run vcfped on subset (fast — small N)
python -m vcfped.vcfped_optimized subset.vcf.gz --no-pairwise --no-gender -o extension_out
python utilities/vcfped_rename_samples.py extension_out
```

This gives exact T1/T2 Mendelian scores for all candidate triples — the most accurate confirmation short of experimental validation.

### MZ twins in the KING data — an approximate threshold, not proof of identity

`prepare_plink_beagle.py` deliberately runs on the *pre-purge* VCF (not the MZ-purged one used for the vcfped trio search), specifically so `refine_pedigree.py` can check whether a non-representative MZ twin would be a better parent candidate. This means `extend_pedigree.py`'s KING input legitimately contains duplicate individuals.

`extend_pedigree.py` detects candidate MZ-twin/duplicate groups in the KING data (KINSHIP ≥ 0.354) and, by default, collapses each group to one representative before searching — this both eliminates a large source of redundant/noisy candidates (one real run logged millions of duplicate-driven false candidates before this fix) and prevents the same biological individual appearing as two "different" competing candidates under separate sample IDs.

**KINSHIP ≥ 0.354 is an *approximate* threshold, not proof of literal genetic identity.** Two true MZ twins (or technical/clonal replicates) are genuinely indistinguishable by any genetic test — collapsing those is correct. But in closely related or repeatedly inbred germplasm (parents and offspring crossed again, extensive backcrossing or selfing), two genuinely **distinct** individuals can have kinship elevated into the nominal "MZ" range without being the same genotype: real meiotic recombination still occurred, it just pushed the genome-wide *average* relatedness above the threshold. Collapsing such a pair would be wrong — it erases a real candidate rather than deduplicating a technical one.

**hap-IBD segment-level structure resolves what KING-only statistics cannot.** Pass `--ibd <prefix>_stat_ibd.ibd.gz` to `extend_pedigree.py` and, for every KING_MZ_MIN-level pair, it computes each individual's actual *genome coverage* with the other: the merged length of all IBD segments between them, normalized against a self-calibrating proxy for that individual's "testable genome" (total IBD bp it has with *any* other individual in the index — the same pattern `refine_pedigree.py` uses for trio genome-coverage checks).

- **Coverage ≥ `--min-mz-coverage` (default 0.95):** near-complete IBD coverage, consistent with a true duplicate → collapsed as before.
- **Coverage below threshold:** the pair's *own* IBD evidence shows incomplete coverage — checked further (see "fragmented duplicates" below) before being trusted as genuinely distinct.
- **No IBD evidence for that specific pair (`--ibd` omitted, or hap-IBD found no segments):** falls back to the conservative default of collapsing on kinship alone, since there is no way to distinguish the two cases without segment data.

**Fragmented duplicates — when "incomplete coverage" is a phasing artifact, not real recombination.** `--ibd` is always `*_stat_ibd.ibd.gz` here (statistical phasing, no pedigree information — candidates aren't in the `.fam` yet), which has more phase-switch errors than pedigree-aware phasing. A switch error can fragment what should be ONE long continuous IBD block for a true duplicate into many short segments separated by small gaps, artificially deflating measured coverage below `--min-mz-coverage` even though the pair really is a duplicate. To catch this, `extend_pedigree.py` self-calibrates a "fragmentation noise floor" from the segment-length pattern of the CONFIRMED (coverage-passing) duplicates already found in the same run (median `mean_segment_Mb`, needs ≥5 confirmed pairs to be meaningful) and checks every below-threshold pair against it:

- **Segment pattern matches the noise floor** (`mean_segment_Mb` ≤ `--fragmentation-factor` × the reference, default factor 1.5): reported as `g1g2_relation = CLOSE_INBRED_POSSIBLE_DUP` — probably actually a duplicate, just under-measured by phasing noise. **Not auto-collapsed** (the heuristic isn't proof), but flagged with a stronger confidence penalty and a note recommending verification, since this pairing may really be one individual counted twice.
- **Segment pattern looks like real recombination blocks** (large segments, unlike the fragmentation noise floor) **or no reference available** (<5 confirmed duplicates in the run): reported as `g1g2_relation = CLOSE_INBRED` — the pair's own IBD evidence shows real, incomplete coverage with recombination breaks, consistent with a genuinely distinct, highly inbred relative, not a duplicate.

Both `CLOSE_INBRED` and `CLOSE_INBRED_POSSIBLE_DUP` candidates include the coverage fraction, segment count, and mean segment length in the report so the call can be evaluated rather than taken on faith.

**The representative chosen for a confirmed (collapsed) MZ-twin group is a bookkeeping convenience (whichever twin already survived `purge_mz_twins.py` and is in your working `.fam`), not a validated identity** — true MZ twins remain genetically indistinguishable from each other even with hap-IBD, so *which one* is the real grandparent still can't be determined from genetic data alone for a *confirmed* duplicate pair.

When a candidate involves a collapsed MZ-twin group, both the extension report and `_vcfped_candidates.txt` list the alternative twin ID(s) explicitly (flagged with ⚠) rather than hiding them. **If it matters which specific physical individual was the true parent of a confirmed duplicate pair, that can only be resolved with non-genetic evidence** — breeding records, sample tracking logs, planting dates, or similar metadata external to the VCF. `CLOSE_INBRED` and `CLOSE_INBRED_POSSIBLE_DUP` candidates are different: they are (or may be) two distinct, genotyped individuals, so ordinary genetic scrutiny (vcfped rerun, additional KING/hap-IBD evidence) still applies — be appropriately critical of them given the elevated kinship, but they are legitimate candidates, not artifacts to discard.

---

## Recommended workflow

```
vcfped output (Regular + Inverted trio calls)
    ↓
filter_trios.py --strategy best         → _filtered.trio
    ↓
fixed_pedigree_viz.py                   → pedigree.html (initial view)
    ↓
prepare_plink_beagle.py                 → .fam, _run_plink.sh, _run_beagle.sh
    ↓
bash _run_plink.sh
  ├─ --mendel            → flag wrong-parent families (fmendel error rate > 5%)
  └─ --make-king-table   → KINSHIP + IBS0 for all pairs → king.kin0
    ↓
bash _run_beagle.sh
  ├─ Step 3a: statistical phasing (all individuals)   → *_stat_phased.vcf.gz
  │   └─ hap-IBD                                      → *_stat_ibd.ibd.gz
  │       (screens ALL individuals for IBD, including unassigned)
  └─ Step 3b: pedigree-aware phasing (known trios)    → *_phased.vcf.gz
      └─ hap-IBD                                      → *_ibd.ibd.gz
          (high-accuracy IBD for confirmed family members)
    ↓
refine_pedigree.py (king.kin0 + step1 MZ pairs)
  ├─ Confirm each parent with KING PO signal
  ├─ Check MZ twin substitution candidates
  ├─ Flag unchecked PO candidates
  └─ Reclassify inverted trios                        → _refined.trio, _refined.fam
    ↓
extend_pedigree.py (refined.fam + king.kin0 + stat_ibd.ibd.gz)
  ├─ Find grandparent candidate pairs (2nd-degree who are 1st-degree to each other)
  ├─ Find offspring candidates (unassigned PO-signal individuals)
  └─ Recommend strategy: KING / vcfped rerun / BEAGLE hap-IBD / combined
    ↓
[if vcfped rerun recommended]
  bcftools view -S vcfped_candidates.txt input.vcf.gz → subset.vcf.gz
  python -m vcfped.vcfped_optimized subset.vcf.gz --no-pairwise --no-gender
    ↓
Update .fam with confirmed grandparents / offspring
Re-run Step 3b (pedigree-aware phasing) with extended .fam
Re-run hap-IBD on new phased VCF                     → shorter segments now reliable
    ↓
Iterate until pedigree is stable
    ↓
Final fixed_pedigree_viz.py on refined trio file      → final pedigree report

Resolution of common ambiguous cases:
  KINSHIP ≈ 0.25, IBS0 > 0.05      →  full sibling, not parent–offspring
  fmendel > 5%                      →  wrong parent; re-inspect filter_trios.py
  Inverted trio                     →  refine_pedigree.py --include-inverted
  2nd-degree (GP vs. half-sib)      →  hap-IBD mean segment length after Step 3b re-run
  MZ twin substitution              →  refine_pedigree.py --step1-pair + breeding records
```

---

## Quick reference: PLINK 2.0 expected values

| Metric | Confirmed parent–offspring | Doubt | Wrong assignment |
|---|---|---|---|
| Mendelian error rate (fmendel) | < 1% | 1–5% | > 5% |
| KINSHIP | 0.20 – 0.35 | 0.15 – 0.20 | < 0.15 |
| IBS0 | < 0.02 | 0.02 – 0.05 | > 0.05 (→ full sib) |
| Total IBD — hap-IBD (% genome) | 45 – 55% | 35 – 45% | < 35% |
