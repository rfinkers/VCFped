# VCFped

Tool for identifying trios and close pairwise relationships in multisample VCF files, using forced allele sharing statistics (IBS scores).

**Original author:** Magnus Dehli Vigeland
**License:** GPL-2

## Project structure

```
vcfped/
  vcfped.py                  # Original reference implementation
  vcfped_optimized.py        # Shared memory + multiprocessing + Numba JIT + online best-tracking + gzip
  checkped.py                # PED file validation
utilities/
  purge_mz_twins.py          # Remove MZ-twin duplicates from VCF before trio analysis
  vcfped_rename_samples.py   # Replace sample numbers with names in output files
  analyze_vcfped_pairs.py    # Diagnose MZ twin calls from a .pair file
  filter_trios.py            # Resolve ambiguous parent assignments in .trio file
  fixed_pedigree_viz.py      # Interactive Plotly pedigree from vcfped output (canonical)
  plot_vcfped_pedigree.py    # Earlier pedigree viz (no --debug flag)
  debug_pedigree_viz.py      # Debug wrapper around plot_vcfped_pedigree.py
  prepare_plink_beagle.py    # Generate PLINK2/BEAGLE input files from vcfped output
  refine_pedigree.py         # Refine parent assignments using PLINK2 KING kinship
  extend_pedigree.py         # Extend pedigree upward (grandparents) and downward (offspring)
  merge_refined_fam.py       # Merge a candidate-subset refined FAM back into the full-cohort pedigree
  analyze_mendel_errors.py   # Flag outlier families from a PLINK2 --mendel .fmendel report
  calibrate_error_rate.py    # Derive empirical --error-rate from confirmed MZ-twin pairs' KING IBS0
  analyze_hbd.py             # Flag inbreeding (autozygosity) outliers from BEAGLE/hap-IBD .hbd.gz
  check_genotype_quality.py  # Rule out poor genotyping quality as a confound for Mendelian repeat offenders
  fam_to_trio.py             # Convert a .fam's confirmed trios into a vcfped .trio file
  analyze_ibd_blocks.py      # Flag unexpectedly large shared hap-IBD blocks given pedigree distance
  run_validation_pipeline.sh # Run the pipeline stages end-to-end on a VCF with known truth, then score
  score_against_truth.py     # Score any pipeline outputs against the breeding_sim ground truth
vcfped/testfiles/
  generate_breeding_sim.py   # Simulates the 500-sample plant-breeding validation cohort + phased/IBD ground truth
  breeding_sim/              # Its output: VCF + truth .fam/.map/phased VCF/IBD/HBD/duplicates (see its README.md)
tests/                       # pytest suite covering vcfped/ and utilities/ (see Testing below)
setup.py                     # Package config (entry point: vcfped.vcfped:main)
```

## Dependencies

- **Required for `vcfped.py`:** `tabulate`
- **Required for `vcfped_optimized.py`:** `tabulate`, `numpy` (hard dependency — the genotype matrix is numpy-vectorized throughout, not an optional accelerator)
- **Optional for `vcfped_optimized.py`:** `numba` (JIT acceleration), `tqdm` (progress bars) — these two have real try/except graceful fallback
- **Visualization utilities:** `pandas`, `plotly`
- **Pedigree verification (PLINK2/BEAGLE workflow):** `plink2`, `beagle`, `king`, `bcftools`, `htslib` (tabix), `r-base` + `r-data.table` (+ `r-ggplot2` for IBD plots); `hap-ibd` has no conda package — download manually (see `environment.yml`)

Install everything with conda/mamba: `conda env create -f environment.yml && conda activate vcfped`

## Script version comparison

| Aspect | vcfped.py | vcfped_optimized.py |
|---|---|---|
| Vectorization | No | NumPy + optional Numba JIT |
| Multiprocessing | `Pool`, all cores by default, `-j/--jobs` to override | Shared memory + adaptive batch `imap_unordered`, `-j/--jobs` to override |
| Stores full variant lines | Yes | No (count-based genotype matrix + online best-tracking) |
| gzip (.vcf.gz) support | No | Yes |
| Thread env vars set | No | Yes (`OMP_NUM_THREADS` etc. forced to 1 to avoid oversubscription) |

Both scripts correctly handle missing genotypes (`-1`/`'-'`) by excluding them only from the
per-pair / per-trio counts that involve them, never by dropping a variant globally just
because some sample in the cohort has a missing call.

**`--all` discrepancy:** `vcfped_optimized.py`'s online best-tracking collapses each pair/triple to
one row per percentile *before* the final `--all`-specific selection logic runs, using a different
tie-break rule than `bestPairs()`/`bestTrios()` apply afterward. This is confirmed (via diff against
`vcfped.py`) to give a different reported percentile/score for `Other/unrelated` pairs under `--all`,
and flagged as the same structural risk (not yet confirmed) for `Not trio` triples — see
`_is_better_pair_result()`/`_is_better_trio_result()`'s docstrings. Everything else (MZ twins,
Parent-child, and all non-`--all` output) is unaffected. **Use `vcfped.py`, not
`vcfped_optimized.py`, for any `--all` run** (e.g. the candidate-subset rerun in the large-cohort
workflow below) — those reruns are on a small subset, so the reference engine's lack of
multiprocessing/JIT costs nothing.

## Build and run

```bash
pip install -e .
vcfped input.vcf -o output_prefix
# or for gzipped (vcfped_optimized.py only):
python -m vcfped.vcfped_optimized input.vcf.gz -o output_prefix
```

## Two-stage workflow: MZ-twin purge then trio analysis

Trio analysis is O(n^3) in sample count — even small reductions in n save dramatically
on compute and memory. If MZ twins or technical replicates are expected, run pairwise
detection first, purge duplicates, then run trios on the reduced set.

### Why this works

- **Step 1 is cheap:** pairwise analysis is O(n^2) and finishes in minutes
- **Cubic savings:** removing k samples saves ~3kn^2 triple combinations. For 4000 samples
  with 200 MZ duplicates removed: 10.6B triples drops to 8.7B (18% reduction)
- **Avoids false trios:** MZ twins share all alleles and can be mistaken for parent-child
  relationships, producing spurious trio calls

### Commands

```bash
# Step 1: Identify MZ twins (pairwise only, fast)
python -m vcfped.vcfped_optimized input.vcf.gz --no-trio --no-gender -o step1

# Inspect results (optional dry run)
python utilities/purge_mz_twins.py input.vcf.gz step1.pair -o /dev/null --dry-run

# Step 2: Create filtered VCF without MZ duplicates
python utilities/purge_mz_twins.py input.vcf.gz step1.pair -o input_no_mz.vcf.gz

# Step 3: Run trio analysis on reduced VCF
python -m vcfped.vcfped_optimized input_no_mz.vcf.gz --no-pairwise -o step3
# For non-human samples (no X-linked variants), add --no-gender:
python -m vcfped.vcfped_optimized input_no_mz.vcf.gz --no-pairwise --no-gender -o step3
```

## Utilities

### purge_mz_twins.py

Removes MZ-twin duplicate samples from a VCF file; see the two-stage workflow above for usage. Options:

- `--keep first|last` — which sample to retain per MZ group (default: `first`)
- `--dry-run` — report clusters and reduction stats without writing output
- Supports `.vcf.gz` input and output
- Handles transitive MZ clusters (A=B and B=C → keeps one of {A,B,C})

### filter_trios.py

Resolves ambiguous parent assignments in the `.trio` file. When the same child appears with
multiple parent pairs (because relatives of the true parent also pass the allele-sharing
test), this script selects the most reliable assignment. The output `.trio` file has the same
format as the input and is directly usable with `fixed_pedigree_viz.py`.

```bash
# Default: keep the highest-confidence (Perc, T1%) call per child
python utilities/filter_trios.py step3_named

# Strict threshold filtering (may still leave some children ambiguous)
python utilities/filter_trios.py step3_named --strategy strict --min-perc 30 --min-t1 95

# Consensus: like best, but also reports which competing parents are likely relatives
python utilities/filter_trios.py step3_named --strategy consensus --report-all
```

Strategies: `best` (default), `strict`, `consensus`. Output: `<prefix>_filtered.trio`.
Inverted/generational trios are passed through unchanged (use `--include-inverted` to
also filter them).

### vcfped_rename_samples.py

Replaces 1-based sample indices with actual sample names in `.gender`, `.pair`, and `.trio` output files.
Reads the sample name mapping from the `.log` file produced by vcfped.

```bash
python utilities/vcfped_rename_samples.py results           # writes results_named.*
python utilities/vcfped_rename_samples.py results --inplace # overwrites original files
python utilities/vcfped_rename_samples.py results -s _id    # custom suffix → results_id.*
```

### analyze_vcfped_pairs.py

Inspects a `.pair` file to diagnose unexpected or excessive MZ twin calls (e.g. after a run where many unrelated samples are flagged as MZ twins).

```bash
python utilities/analyze_vcfped_pairs.py results_named
python utilities/analyze_vcfped_pairs.py results_named --threshold 0.95
```

### fixed_pedigree_viz.py

Generates an HTML report from vcfped output with: summary statistics (families, founders, generations, sibling groups), a searchable parent1/parent2/child table, and per-family Plotly pedigree diagrams. Requires `plotly` and `pandas`.

- `.trio` — required
- `.pair` — optional (absent after `--no-pairwise` runs; MZ twins already removed before step 3)
- `.gender` — optional

```bash
python utilities/fixed_pedigree_viz.py results_named
python utilities/fixed_pedigree_viz.py results_named -o family_tree.html
python utilities/fixed_pedigree_viz.py results_named --keep-mz-separate --debug
python utilities/fixed_pedigree_viz.py results_named --max-families 30 --min-family-size 5
python utilities/fixed_pedigree_viz.py results_named --max-diagram-size 100
```

Options: `-o/--output`, `-t/--title`, `--no-browser`, `--keep-mz-separate`, `--debug`, `--max-families` (default 20), `--min-family-size` (default 3), `--max-diagram-size` (default 150)

**Diagram rendering at scale:** in a cohort where nearly the whole pedigree collapses into one connected
family (common once several rounds of `extend_pedigree.py`/`merge_refined_fam.py` have run — real example:
1066 of 1142 individuals in one family), the per-family Plotly diagrams need to handle hundreds/thousands
of nodes, not dozens:
- Node traces are batched by `(gender, is_mz_group)` into at most a handful of Plotly traces total, not one
  trace per individual; parent-child edges are batched into two traces (bars, drop-lines) via
  None-separated segments, not two traces per couple. The old one-trace-per-node/edge approach was the
  single biggest driver of slow, visually dense renders at scale.
- Figure `width` scales with the family's *widest* generation (not just `height` with generation depth) —
  without this, a generation with hundreds of members gets squeezed into a fixed-width box regardless of
  how many nodes it holds.
- On-plot text labels are suppressed above `LABEL_THRESHOLD` (40 individuals), relying on hover instead —
  crammed small-font labels are noise, not information, past that scale.
- Families larger than `--max-diagram-size` get a compact summary card (founder/leaf counts, per-generation
  breakdown) instead of a full diagram — no layout algorithm makes a diagram with hundreds/thousands of
  nodes readable as one picture; full per-individual detail is still in the searchable table above it.

`plot_vcfped_pedigree.py` is an earlier version without `--debug`. `debug_pedigree_viz.py` is a standalone debug wrapper around `plot_vcfped_pedigree.py`.

### fam_to_trio.py

Converts a `.fam` file's confirmed (PAT and MAT both known) parent-child links into a standard
vcfped `.trio` file. Needed because a working pedigree built through the iterative
extend/vcfped-rerun/merge cycle (`extend_pedigree.py` → vcfped rerun → `filter_trios.py` →
`refine_pedigree.py` → `merge_refined_fam.py`, repeated) accumulates confirmed trios from
*multiple* originating `.trio` files into one merged `.fam` — there is no single `.trio` file
representing that combined state. `fixed_pedigree_viz.py` and `refine_pedigree.py`'s `--ibd`
final-confirmation pass both only read `.trio`, so this bridges the gap.

Most rows in the merged FAM DO have real evidence behind them — it just lives in whichever
per-round `.trio` file the child originated from. `--trio <path>` (repeatable) looks up the
CURRENT `(child, {p1,p2})` against one or more original `*_refined.trio` files and carries over
the real `Perc/Autos/AA+BB/=AB/Test1/BB+BB/=BB/Test2` whenever it matches exactly — i.e.
`refine_pedigree.py` confirmed the original assignment as-is, or `merge_refined_fam.py` kept the
base side of a conflict. Only rows where a parent was *substituted* (`mz_swap`/`king_new`, or the
"new" side of a merge conflict) have no such match, since vcfped never scored that exact final
combination — for these, `--king <prefix>` supplies a KING kinship-based Test1 proxy instead of a
meaningless flat placeholder (the raw vcfped counts don't apply and stay `0`). `Perc` is always
set to `0` for these KING-only rows — a value real vcfped output never produces (its own `Perc` is
always one of the quality-filter percentiles, e.g. 10/30/50) — so it doubles as a visible,
self-documenting marker in `fixed_pedigree_viz.py`'s table distinguishing genuinely vcfped-tested
rows from KING-confirmed-only ones, without needing a different `Verdict` string (which would
break `refine_pedigree.py`'s default `Verdict == 'Regular trio'` filter downstream).

```bash
python utilities/fam_to_trio.py step3_named_filtered_refined_merged2.fam
python utilities/fam_to_trio.py step3_named_filtered_refined_merged2.fam \
    --trio step3_named_filtered_refined.trio \
    --trio extension_out_named_filtered_refined.trio \
    --trio extension_out2_named_filtered_refined.trio \
    --king revalidate_king_ext -o merged2_trio
```

Output: `<output>.trio`

### prepare_plink_beagle.py

Generates PLINK and BEAGLE input files from a VCF + filtered trio output for external pedigree verification. Reads `.trio` (required), `.gender` (optional for sex assignment); reads chromosome names and sample order from the VCF header.

`--from-fam <path>` is an alternative to the `trio_prefix` positional argument: instead of deriving
the pedigree from a vcfped `.trio` file, it reads an existing PLINK `.fam` directly — e.g.
`merge_refined_fam.py`'s output, which has already-decided PAT/MAT assignments (including
KING/IBD-confirmed links vcfped itself never scored, like reclassified-from-inverted trios) with no
vcfped Perc/Test1 confidence to read. Confirmed parent-child links (both PAT and MAT known) become
synthetic full-confidence trios; existing SEX values are carried over rather than re-derived.
Everything else (family grouping, role-derived sex for unknowns, Mendelian link repair for
hermaphrodite plant crossings) runs unchanged — this is the natural way to (re-)run PLINK2 `--mendel`
Mendelian-error validation against an updated/merged pedigree without touching any of that logic.
`-o/--output` is required with `--from-fam` (its natural prefix is the same path as the input FAM,
which would otherwise silently overwrite it).

```bash
python utilities/prepare_plink_beagle.py step3_named_filtered input.vcf.gz
python utilities/prepare_plink_beagle.py step3_named_filtered input.vcf.gz -o pedigree_check --chr-prefix ch
python utilities/prepare_plink_beagle.py --from-fam step3_named_filtered_refined_merged.fam input.vcf.gz -o revalidate --chr-prefix ch
```

Outputs:
- `<prefix>.fam` — PLINK/BEAGLE pedigree file (shared by both tools)
- `<prefix>_chr_rename.txt` — chromosome rename map (`ch01→1` …) for reference (not needed by PLINK 2.0)
- `<prefix>_run_plink.sh` — PLINK 2.0 workflow: `--mendel` → `--make-king-table` → `--het`/`--missing`
- `<prefix>_run_beagle.sh` — BEAGLE workflow: pedigree-aware phasing → hap-IBD segments

`_run_plink.sh`'s Step 4 runs `--het` (on the LD-pruned set used for KING — correlated SNPs bias the
inbreeding coefficient estimate) and `--missing` (on the full quality-filtered set, so restricting to
independent SNPs doesn't understate a genuinely poorly-sequenced sample's missingness), producing
`<prefix>_plink.het`/`<prefix>_plink.smiss`. See `check_genotype_quality.py` below for what consumes
them.

See `PEDIGREE_VERIFICATION.md` for interpretation of results and IBD visualisation options.

### refine_pedigree.py

Refines vcfped parent assignments using PLINK2 KING kinship as independent evidence.
Identifies: (1) assignments not supported by KING; (2) MZ-group members that would be better
parent candidates; (3) unchecked individuals with parent-offspring KING signal. Runs after
`prepare_plink_beagle.py` and `_run_plink.sh`.

`_run_plink.sh` writes KING output as `<prefix>_king_1st.kin0` (1st-degree only) and
`<prefix>_king_ext.kin0` (up to 3rd-degree) — **not** `<prefix>.kin0`. Pass `_king_ext`
as the second argument; it is a superset of `_king_1st` and is required for `--include-inverted`
(which needs 2nd-degree pairs to classify grandparent chains).

MZ-twin substitution (`check_mz_swap`) trusts every member of a `--step1-pair` MZ group as an
interchangeable substitute parent — valid only if the group really is one clonal individual. In
densely inbred/repeatedly-selfed germplasm, vcfped's raw pairwise "MZ twins" verdict can be
inflated by inbreeding-induced allele-sharing (confirmed on a real cohort: ~15x more raw "MZ
twins" pairs than hap-IBD-coverage validated true duplicates, with individual groups of 40+
members — implausible for literal clonal duplication). A contaminated group can silently
substitute in a genuinely *different* individual, not the same one under another label.
`--confirmed-duplicates <path>` (mutually exclusive with `--step1-pair`) uses `extend_pedigree.py`'s
stricter `<prefix>_confirmed_duplicates.tsv` instead — see that script's section below. When using
the raw `--step1-pair` verdict, a `WARNING` is printed if any single group exceeds 10 members.

`Assignment.confidence()` (the score used to rank alternative parent candidates) now takes the
calibrated `ibs0_max` in use as an argument instead of scaling against the hardcoded
`KING_PO_IBS0_MAX = 0.05` default. At a much lower calibrated ceiling (e.g. `--ibs0-max 0.0008`),
scoring against the old hardcoded constant flattened the IBS0 term to near-1.0 for every
near-zero-IBS0 candidate — silently neutralizing "IBS0: most reliable, weight highest" and leaving
kinship percentile as the real (much lower-weighted) discriminator instead.

`parents_are_compatible()` (rejects a candidate co-parent pair that look like parent-offspring of
*each other*, e.g. a grandparent/parent mixup) had the identical hardcoded-`0.05` problem —
confirmed by cross-referencing `check_inverted_direction()`, which already threads the calibrated
`ibs0_max` through `is_po_signal()` for the exact same question. Now also takes `ibs0_max` as an
argument (default `KING_PO_IBS0_MAX` preserved for callers that don't pass one); all three call
sites in `refine_child()` pass the calibrated value through.

```bash
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext --step1-pair step1_named
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext \
    --confirmed-duplicates ext_out_confirmed_duplicates.tsv
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext --include-inverted
```

Outputs: `_refined.trio`, `_refined.fam`, `_refinement_report.txt`

### extend_pedigree.py

Iteratively extends identified pedigrees upward (grandparents) and downward (grandchildren)
using PLINK2 KING kinship + optional hap-IBD segments. Finds grandparent candidate pairs
(2nd-degree relatives of founders that are themselves 1st-degree to each other) and offspring
candidates (unassigned individuals with PO signal to known leaves). Recommends follow-up
strategy (KING-only / vcfped rerun / BEAGLE hap-IBD / combined).

Use `_king_ext.kin0` here too (not `_king_1st.kin0`) — grandparent/2nd-degree candidates fall
below the 1st-degree-only filter. For `--ibd`, use `*_stat_ibd.ibd.gz` (statistical phasing),
not `*_ibd.ibd.gz` (pedigree-aware): candidates aren't in the `.fam` yet, so they only ever
get statistical phasing regardless of which IBD file you point at.

Since the KING input deliberately still contains MZ-twin duplicates (see `prepare_plink_beagle.py`
above), candidate MZ-twin groups (KINSHIP ≥ 0.354) are collapsed to one representative before
searching. That threshold is approximate, not proof of identity: in inbred/backcrossed germplasm,
genuinely distinct individuals can cross it without being true duplicates. When `--ibd` is given,
each MZ-level pair's hap-IBD genome coverage is checked (`--min-mz-coverage`, default 0.95) —
near-complete coverage confirms a true duplicate (collapsed, as before); incomplete coverage is
checked further before being trusted as genuinely distinct, since statistical phasing (no pedigree
info — the only option for unassigned candidates) can fragment a true duplicate's IBD into many
short segments and deflate its measured coverage. A self-calibrated "fragmentation noise floor"
(segment-length pattern of confirmed duplicates in the same run, `--fragmentation-factor`, default
1.5, needs ≥5 confirmed pairs) distinguishes this from real recombination breaks: a matching
pattern is reported as a flagged `CLOSE_INBRED_POSSIBLE_DUP` candidate (probably actually a
duplicate, not collapsed but penalized harder); a genuine mismatch (or no reference available) is
reported as `CLOSE_INBRED`. Neither is silently dropped. Confirmed-duplicate representatives are
still a bookkeeping choice, not a validated identity — no genetic test can tell which twin is the
real individual — so alternatives are reported (⚠ in the report and candidates file), never hidden.
See "MZ twins in the KING data" in `PEDIGREE_VERIFICATION.md`.

When `--ibd` is given (coverage-validated, not kinship-only), every collapsed true-duplicate pair
is also written to `<out_prefix>_confirmed_duplicates.tsv` — a stricter ground truth than vcfped's
own pairwise "MZ twins" verdict for `calibrate_error_rate.py`'s `--confirmed-duplicates` (see that
section below for why the raw verdict can be unreliable in this kind of cohort).

```bash
python utilities/extend_pedigree.py refined.fam step3_named_filtered_king_ext.kin0
python utilities/extend_pedigree.py refined.fam step3_named_filtered_king_ext.kin0 --ibd step3_named_filtered_stat_ibd.ibd.gz
python utilities/extend_pedigree.py refined.fam step3_named_filtered_king_ext.kin0 --direction up --max-iter 2
```

Outputs: `_extended.fam`, `_extension_report.txt`, `_vcfped_candidates.txt`, `_grandparent_check.sh`,
`_confirmed_duplicates.tsv` (only when `--ibd` is given)

### merge_refined_fam.py

Merges a candidate-subset refined FAM back into the full-cohort pedigree. For large cohorts,
`extend_pedigree.py`'s own strategy recommendation is to extract candidate samples into a subset
VCF and re-run `vcfped.py` (**not** `vcfped_optimized.py` — see Script version comparison below)
`--all` on just that subset (fast: O(k³) not O(N³)) for real T1/T2 verdicts, then `filter_trios.py`
+ `refine_pedigree.py` on the subset's output. That subset run's
`_refined.fam` only ever contains individuals who got a confirmed parent-pair *during that run* —
if invoked without `--fam`, `refine_pedigree.py` starts from an empty FAM, so every row in the
subset's `_refined.fam` already has parents by construction. Running `extend_pedigree.py` directly
on it will always report zero founders and zero grandparent candidates — an artifact of scope, not
a finding — because the founders that motivated the whole candidate search live in the full
cohort's FAM, not this subset one.

This script merges the subset's newly-confirmed assignments into the full-cohort FAM so a
subsequent `extend_pedigree.py` run has the actual founders/leaves to search from again. An
individual with *different* existing (non-founder) parents in both FAMs is a conflict, not
silently resolved — `--on-conflict keep-base` (default) or `keep-new`; every conflict is listed in
the report either way.

To judge a conflict on more than two bare parent-ID pairs, `--base-trio`/`--new-trio` (auto-derived
from `base_fam`/`new_fam` by replacing `.fam` with `.trio` — the same prefix `refine_pedigree.py`
always writes both under) join in each side's Verdict/Perc/Test1, and `--king <prefix>` joins in
KING kinship/IBS0 for child↔each proposed parent — an independent genetic check, since fixed
T1/Perc confidence can be inflated in closely related germplasm. All three are optional and
degrade to `NA` if not found; a `<output>_conflicts.tsv` with the full joined evidence is written
whenever there's at least one conflict, for sorting/filtering outside the narrative report.

`--score-conflicts` (requires `--king`) automates picking through that evidence: each proposed
parent is classified `STRONG`/`MODERATE`/`WEAK`/`UNTESTED`/`REJECTED` using the same
population-relative approach `refine_pedigree.py` already established for this kind of closely
related cohort — kinship percentiles (p25/p75) computed from the `--king` file itself rather than
hardcoded, and an IBS0 noise ceiling from `--error-rate` (same formula, default 0.025 → 0.075). A
side's overall confidence is its *worse* parent (a trio is only as strong as its weakest link); a
conflict is recommended `prefer_new`, `prefer_base`, or `tie` (or `tie_new_higher` for same-tier
ties with a clear kinship gap) accordingly, added as `Base_Overall`/`New_Overall`/`Recommendation`
columns in `_conflicts.tsv` and a dedicated "Flagged for review" section in the narrative report.

`--auto-resolve` (requires `--score-conflicts`) applies that recommendation per conflict instead
of one global `--on-conflict` choice — `prefer_new` keeps new, everything else (`prefer_base`,
`tie`, `tie_new_higher`) conservatively keeps base. `--on-conflict` is ignored when this is set.
A single global choice cannot express "keep base for most conflicts but new for these specific
ones"; `--auto-resolve` can, without hand-editing the FAM.

A fourth, independent signal can be layered on top: `--base-mendel`/`--new-mendel` (each side's own
PLINK2 `.fmendel` per-family error counts — not auto-derived, since there's no fixed naming
convention tying an `.fmendel` report to a specific FAM prefix, unlike `--base-trio`/`--new-trio`)
plus `--mendel-bim`/`--mendel-n-variants` (variant count, to turn the raw error count into a rate —
same formula as `analyze_mendel_errors.py`). If the side `--score-conflicts` currently favors has a
Mendelian error rate above `--mendel-absolute-multiplier` (default 2.0) × the expected baseline
(2 × `--error-rate`), the recommendation is downgraded to `tie_mendel_veto`. This can only *veto* an
automatic decision, never independently promote one — a plain `tie` is left as-is, and
`--auto-resolve`'s mapping doesn't change (only `prefer_new` keeps new; every other value, including
`tie_mendel_veto`, keeps base) — so the veto's only effect is making `--auto-resolve` conservative
when kinship and Mendelian evidence disagree, without a new automated code path.

```bash
python utilities/merge_refined_fam.py step3_named_filtered_refined.fam extension_out_refined.fam \
    --king step3_named_filtered_king_ext --score-conflicts --auto-resolve
python utilities/merge_refined_fam.py base.fam new.fam -o merged --on-conflict keep-new
python utilities/merge_refined_fam.py base.fam new.fam --king king_prefix --score-conflicts \
    --base-mendel base_mendel.fmendel --new-mendel new_mendel.fmendel --mendel-n-variants 8789
```

Outputs: `<output>.fam`, `<output>_merge_report.txt`, `<output>_conflicts.tsv` (only if there are conflicts)

### analyze_mendel_errors.py

Flags outlier families from a PLINK2 `--mendel` `.fmendel` report. `.fmendel`'s `N` column is a raw
error *count*, not a rate — it must be normalized by variant count and family size
(`rate = N / (n_variants × CHILD)`) before it means anything. A single fixed rate threshold is
unreliable across cohorts for the same reason fixed KING kinship thresholds are unreliable in this
project's closely related germplasm: the *expected* baseline rate for a genuinely correct trio
scales with the cohort's own genotyping error rate (`expected ≈ 2 × error rate`, documented in
`prepare_plink_beagle.py`'s generated `_run_plink.sh`). A cohort with an elevated error rate (e.g.
2.5%) has an expected baseline of ~5% — right at the generic "family error > 5% → wrong parent"
guideline, which was calibrated for cleaner data; applying it unmodified would flag many correct
trios. This script flags outliers two ways instead — cohort-relative percentile (default: 95th) and
absolute (multiple of the error-rate-derived theoretical baseline, default: 2×) — and reports which
signal(s) triggered for each flagged family, so nothing is hidden behind one number.

A flagged family only says "this specific PAT+MAT pairing has an elevated error rate" — it can't
distinguish a genuinely wrong assignment from an individual-level problem (sample mislabeling, an
MZ-twin representative that's actually the wrong twin, or a sample-specific genotyping quality
issue). The signature of the latter is one individual appearing as a parent across *many different*
co-parents, all showing elevated error — if the problem tracked with the specific pairing, different
co-parents would behave differently. The script aggregates every `.fmendel` row by individual
(across every family where they appear as either PAT or MAT) to surface this directly, rather than
relying on spotting repeated IDs by eye. `--min-co-parents` (default 2) excludes individuals with
only one pairing from this specific flagging — with just one pairing, "repeat offender" isn't yet
distinguishable from ordinary per-family noise, though they still appear in the full individual TSV.
Optionally cross-references PLINK2's own `.imendel` (per-CHILD error counts — a different,
complementary count from the PARENT-role aggregation above) via `--imendel`.

```bash
python utilities/analyze_mendel_errors.py revalidate_mendel.fmendel --bim revalidate_plink.bim
python utilities/analyze_mendel_errors.py revalidate_mendel.fmendel --n-variants 8789 \
    --error-rate 0.025 --outlier-percentile 95 --absolute-multiplier 2.0 \
    --imendel revalidate_mendel.imendel --min-co-parents 2
```

Outputs: `<output>_mendel_analysis.tsv` (every family: rate, cohort percentile rank, flagged, reason),
`<output>_mendel_report.txt` (distribution summary + flagged families sorted by rate descending +
repeat-offender parents section), `<output>_individual_analysis.tsv` (every individual who appears
as a parent: aggregate rate across all their families, number of distinct co-parents, flagged,
reason, `.imendel` cross-reference if given)

### calibrate_error_rate.py

Derives an empirical `--error-rate` from confirmed MZ-twin/duplicate pairs, replacing the assumed
`0.025` default used everywhere else in the pipeline (`refine_pedigree.py`'s IBS0 noise ceiling,
`analyze_mendel_errors.py`'s expected baseline rate) with a value measured against this cohort's
own data. vcfped's pairwise `Verdict == 'MZ twins'` call is an independent statistical signal
(allele-sharing based, not KING-derived) that two samples are the same individual. For a genuinely
identical pair, KING's IBS0 should be exactly 0 absent genotyping error — any IBS0 observed among
*confirmed* duplicates is therefore pure genotyping noise. Inverting the same
`IBS0 ≈ 2 × error rate` relationship `refine_pedigree.py`'s `ibs0_threshold_for_error_rate()` already
uses gives `error_rate ≈ median(IBS0 among confirmed MZ pairs) / 2`.

**This ground truth is only as clean as vcfped's "MZ twins" call itself.** In densely inbred/
repeatedly-selfed germplasm, extreme homozygosity can inflate allele-sharing statistics enough that
genuinely distinct-but-closely-related samples also get called "MZ twins" — the same inbreeding-
induced inflation `extend_pedigree.py`'s `CLOSE_INBRED`/`CLOSE_INBRED_POSSIBLE_DUP` distinction
exists to catch. A tell-tale sign: if double-digit percentages of *all possible pairs* in the cohort
get flagged "MZ twins", that's contamination, not that many literal duplicates exist. `--confirmed-
duplicates <path>` swaps in a stricter ground truth instead — a two-column `ID1`/`ID2` pair list
(e.g. `extend_pedigree.py`'s `<prefix>_confirmed_duplicates.tsv`, which is hap-IBD-coverage
validated, not just kinship-based) — mutually exclusive with the `step1_pair_prefix` positional
argument.

```bash
python utilities/calibrate_error_rate.py step1_named step3_named_filtered_king_ext
python utilities/calibrate_error_rate.py step1_named step3_named_filtered_king_ext -o calib
python utilities/calibrate_error_rate.py --confirmed-duplicates ext_out_confirmed_duplicates.tsv \
    step3_named_filtered_king_ext
```

Feed the result back in: `--error-rate <derived value>` to both `refine_pedigree.py` and
`analyze_mendel_errors.py`. Output: `<output>_calibration_report.txt` (pair counts, IBS0
distribution among confirmed duplicates, derived error rate).

### analyze_hbd.py

`prepare_plink_beagle.py`'s generated `_run_beagle.sh` already produces `<prefix>_ibd.hbd.gz`
(homozygosity-by-descent / autozygosity segments — same 8-column hap-IBD format as regular IBD, but
`id1==id2`) plus an R summary script; until now nothing in the Python pipeline consumed it. In a
cohort of deep-selfed lines, high genome-wide autozygosity is the *expected* baseline, so this data
is diagnostic two ways: (a) an individual with unexpectedly *low* autozygosity relative to the
cohort's own distribution is inconsistent with the claimed breeding history — a contamination/
mislabeling signal, complementary to (but independent of) PLINK2's frequency-based `--het` estimate;
(b) for an `extend_pedigree.py` `CLOSE_INBRED`/`CLOSE_INBRED_POSSIBLE_DUP` candidate pair, each
member's own autozygosity level is an extra independent data point — two genuinely distinct
deep-selfed founders should each show cohort-typical autozygosity. As elsewhere in this pipeline,
outlier flagging is population-relative (percentile *rank*, not a fixed Mb cutoff) — a value-based
percentile threshold can put the single most extreme individual in a small cohort exactly at its own
threshold, which a strict inequality then misses; rank-based comparison doesn't have that edge case.

```bash
python utilities/analyze_hbd.py step3_named_filtered_ibd.hbd.gz
python utilities/analyze_hbd.py step3_named_filtered_ibd.hbd.gz -o hbd --low-percentile 5 --outlier-percentile 95
python utilities/analyze_hbd.py step3_named_filtered_ibd.hbd.gz --check-pairs candidate_pairs.txt
```

`--check-pairs` takes a simple two-column file (`id1,id2` or `id1<TAB>id2` per line — e.g. hand-
extracted from `extend_pedigree.py`'s `CLOSE_INBRED` report lines) and adds a side-by-side
autozygosity comparison per pair to the report.

Outputs: `<output>_hbd_analysis.tsv` (every individual: HBD Mb, segment count, mean segment Mb,
percentile rank, flagged, reason), `<output>_hbd_report.txt` (distribution summary + flagged
individuals + `--check-pairs` comparisons if given)

### check_genotype_quality.py

Consumes `prepare_plink_beagle.py`'s `--het`/`--missing` output (Step 4 above) to rule out poor
genotyping quality as an alternative explanation for a `analyze_mendel_errors.py` "repeat offender"
parent. Elevated Mendelian error across every co-parent an individual appears with reads naturally
as a pedigree problem (wrong assignment, wrong MZ-twin representative) — but the *identical* symptom
also arises from poor genotyping quality alone (elevated missingness, or elevated heterozygosity from
sample contamination), which needs a completely different remedy (re-genotyping/QC exclusion, not
pedigree reassignment). As elsewhere in this pipeline, outlier flagging is population-relative
(percentile rank, not a fixed cutoff) — a fixed "F > 0.5 is suspicious" rule would misfire constantly
in a cohort of deep-selfed lines where high homozygosity is the norm. `.het`/`.smiss` columns are
matched by name (not fixed position), tolerant of minor PLINK2 version differences.

```bash
python utilities/check_genotype_quality.py step3_plink.het step3_plink.smiss
python utilities/check_genotype_quality.py step3_plink.het step3_plink.smiss -o qc \
    --het-low-percentile 5 --het-high-percentile 95 --miss-outlier-percentile 95
python utilities/check_genotype_quality.py step3_plink.het step3_plink.smiss \
    --mendel-individual-tsv step3_mendel_individual_analysis.tsv
```

`--mendel-individual-tsv` (`analyze_mendel_errors.py`'s `_individual_analysis.tsv`) cross-references
each flagged repeat-offender against this script's own het/missingness flags, splitting them into
"with a genotyping-quality confound" (elevated error could be explained by quality alone) vs. "no
confound" (the pedigree-error explanation isn't undermined by this check) in the report.

Outputs: `<output>_genotype_quality.tsv` (every individual: F, percentile rank, F_MISS, percentile
rank, flagged, reason), `<output>_genotype_quality_report.txt` (flagged individuals + Mendelian
cross-reference if `--mendel-individual-tsv` is given)

### analyze_ibd_blocks.py

Flags pairs of individuals with unexpectedly large shared hap-IBD haplotype blocks given their
*current* pedigree relationship (or lack of one) — a first pass at understanding germplasm
structure beyond what the confirmed pedigree records. A large shared IBD block indicates a recent
common ancestor; if the pedigree records no relationship at all between a pair (no path through
parent-child edges within a handful of generations) but their hap-IBD data shows a large shared
block anyway, that's worth investigating — an undocumented relationship (missing pedigree link,
e.g. an unrecorded shared parent) or a sample mix-up. Pedigree "connectedness" is a coarse,
undirected BFS hop-count through the FAM's parent-child edges (`--max-hops`, default 6) — it
answers "does the pedigree record ANY relationship here," not a full relationship-type classifier
distinguishing siblings from grandparent-grandchild (that's `refine_pedigree.py`/
`extend_pedigree.py`'s KING-based job). As elsewhere in this pipeline, "large" is population-
relative: outlier flagging is computed *separately* within each pedigree-status bucket
(connected/unconnected), since a block size unremarkable among connected (expected-related) pairs
could be a genuine outlier among unconnected (expected-unrelated) ones, and vice versa — general
background relatedness in this germplasm is unusually high, so a fixed Mb cutoff would misfire.

**Population-wide hotspot regions** (confirmed on real data): in a densely inbred cohort, one
specific genomic region can be broadly shared (near-fixed) across a large fraction of the
population — not because any two carriers are privately related, but because the whole population
descends from it. A real run flagged 147,214 pairs this way, nearly all reporting the *identical*
largest-segment length — the signature of a shared regional haplotype, not tens of thousands of
independent findings, and a large tied plateau at that value distorts the rank-based percentile
computation itself.

Detecting this by exact (chrom, start, end) — even coarsely binned — turned out too brittle for
CHROMOSOME-SCALE hotspots specifically: on the same real cohort, a single near-whole-chromosome
conservation event was fragmented into 15 "different" detected regions
(`chr01:4000000-44000000`, `chr01:4000000-45000000`, `chr01:2000000-44000000`, ...) that were all
really the same phenomenon, split apart by a few Mb of segment-boundary jitter between different
pairs. Grouping by **fraction of chromosome covered** instead (10% bins) turned out to have the
identical problem one level up: real coverage values like 99.6%/95.0%/96.0% of the *same*
chromosome still landed in different buckets and were still reported as separate hotspots — a
continuous, naturally-varying distribution will always straddle *some* bucket boundary, no matter
how the bucket is defined. The question that actually matters is "does an unusually large number of
distinct pairs share a huge chunk of *this* chromosome," not "do they share *precisely* the same
huge chunk" — so hotspot detection now groups by **chromosome alone**, once a pair's largest segment
clears `--hotspot-min-coverage` (default 30%; chromosome length self-calibrated from the largest
observed segment end on that chromosome, same "genome proxy" convention as `extend_pedigree.py`'s
`build_ibd_genome_proxy()`) — no further sub-bucketing by exact fraction, so there's no boundary
left to straddle. Segments below that floor are never hotspot candidates at all, regardless of how
often they recur — small segments recurring across many pairs is the *normal* background case, not
a chromosome-scale conservation event; without this floor, ordinary small segments from many
different pairs can inflate the chromosome-length proxy and spuriously cluster into a spurious
"hotspot". Any chromosome recurring across an unusually large number of *distinct* candidate pairs
is treated as a hotspot: those pairs are excluded from the ranking pool (so they don't distort
percentiles for the pairs that remain) and are never flagged `UNEXPECTED_LARGE_BLOCK` — but are
still reported, tagged `COMMON_HOTSPOT_REGION`, never hidden.

The distinct-pair-count threshold (`--min-hotspot-pairs`, default self-calibrated) is calibrated as
`--hotspot-pair-fraction` (default 1%) of the **total** pair count with IBD data — *not* a
percentile of the per-chromosome candidate-pair-count distribution. That was tried first and
confirmed degenerate on real data: a genome only has a handful of chromosomes (~15-20), and the
99th percentile of that few values just picks out the single largest chromosome's own count as the
threshold. On a real cohort this meant only chr01 (936,740 candidate pairs) ever cleared the bar,
even though chr02/chr03/chr04/etc. showed comparably strong widespread-sharing signals — those
pairs fell through to (and fragmented) the locus-scale mechanism below instead, since a percentile
computed over a handful of groups isn't statistically meaningful. Calibrating against the total
pair population — large enough for a fraction-based threshold to mean something — fixes this.

**Locus-scale hotspots** are a second, separate detection pass — confirmed necessary on the same
real cohort: after fixing the chromosome-scale case above, a ~39.6 Mb block at
`chr02:264222-39828158` was still shared (to within a few hundred bp) by 6,604 distinct pairs, all
still flagged `UNEXPECTED_LARGE_BLOCK`. It never reached the chromosome-scale mechanism at all —
that block covers well under 30% of chr02's much longer observed span, so it's a real conserved
region, just not a chromosome-scale one. The two cases need different grouping strategies because
their boundary jitter is a different order of magnitude: chromosome-scale events jitter by
*megabases* between pairs, while this locus-scale event jittered by only a few hundred *bp* — exact/
fixed-bin coordinate matching would misfire again on some other locus with wider jitter, so segments
are grouped by **reciprocal overlap** instead (`--locus-min-overlap`, default 50% — both segments
must cover at least this fraction of each other), which adapts to whatever jitter magnitude a given
real locus happens to have rather than assuming one bin width fits every locus. Segments are sorted
per chromosome and swept once, comparing each to the single currently-open cluster's *fixed* seed
interval (not a growing union, to avoid slow drift across a long chain of only-barely-overlapping
segments) — O(n log n), not the O(n²) an all-pairs comparison would cost at millions of pairs.
Detected loci are reported in their own section, separate from detected hotspot chromosomes, and
their pairs get the same excluded-from-ranking / `COMMON_HOTSPOT_REGION` treatment.

Locus detection deliberately **excludes every pair already captured by the chromosome-scale
mechanism** before clustering (`detect_locus_hotspots`'s `exclude_pairs`) — without this, the same
real cohort's already-explained chr01 near-whole-chromosome event was independently rediscovered
here too, fragmented into 9 different "loci" purely because the fixed-seed reciprocal-overlap sweep
breaks the chain once a segment's start has drifted far enough from the seed (the same
megabase-scale-jitter failure mode the chromosome-scale mechanism exists to handle, recurring one
level down). Excluding already-explained pairs keeps this pass scoped to genuinely sub-chromosomal
regions instead of re-reporting the same background sharing as spurious extra findings.

```bash
python utilities/analyze_ibd_blocks.py step3_named_filtered_refined_merged2.fam step3_named_filtered_stat_ibd.ibd.gz
python utilities/analyze_ibd_blocks.py merged2.fam stat_ibd.ibd.gz -o blocks --max-hops 6 --outlier-percentile 95
python utilities/analyze_ibd_blocks.py merged2.fam stat_ibd.ibd.gz --hotspot-min-coverage 0.4 --hotspot-pair-fraction 0.02
python utilities/analyze_ibd_blocks.py merged2.fam stat_ibd.ibd.gz --locus-min-overlap 0.6
```

Outputs: `<output>_ibd_block_analysis.tsv` (every pair with IBD data: pedigree status, total Mb,
largest segment Mb + region, segment count, percentile rank within its own bucket [hotspot pairs
excluded], flagged, hotspot flag [either kind], reason), `<output>_ibd_block_report.txt` (distribution
summary broken down by chromosome-scale vs. locus-scale hotspot pairs + detected hotspot chromosomes
+ detected hotspot loci, each with their distinct-pair counts + flagged pairs)

### run_validation_pipeline.sh and score_against_truth.py

`run_validation_pipeline.sh -o OUTDIR` runs the stages of the full workflow below (pairs, purge, trios,
filter, plink, beagle, refine, qc, score) on a VCF with known ground truth — by default the simulated
`vcfped/testfiles/breeding_sim/` cohort — logging each stage to `OUTDIR/logs/` and skipping finished
stages on rerun (`--force`, `--stages a,b`). It uses the generated PLINK/BEAGLE scripts as they are (patching only the
plink2 path, and falling back when a plink2 rejects `--mendel`), builds a `min-markers` wrapper for hap-IBD, and can
hand hap-IBD the true genetic map (`--true-map`). `BEAGLE_CMD`/`HAPIBD_CMD` enable the phasing/IBD stage.

`score_against_truth.py --truth <prefix> ...` compares outputs with the truth; each input switches on a section:
`--purge-log` (what the purge really removed, and which distinct individuals it lost), `--pair`, `--trio` (per truth kind:
simple / inbred child / related parents / backcross), `--fam` (final parent assignments, precision, what wrong parents
truly are), `--king` (degree bands, extra pairs), `--genotype-quality` / `--mendel-individual` / `--mendel-artifacts`
(planted quality problems), `--ibd` / `--hbd` (coverage of true IBD, false pairs, hotspot block). Duplicates are treated as
interchangeable and parent *sets* are compared; denominators use the original sample set (`--samples`), so wrongly purged
individuals count as misses. Outputs: `<out>_score_report.txt`, `_score_summary.json`, `_score_children.tsv`.

```bash
utilities/run_validation_pipeline.sh -o runs/seed42 -j 10
python utilities/score_against_truth.py --truth vcfped/testfiles/breeding_sim/breeding_sim --fam refined.fam --king king_ext.kin0
```

Backcross trios (and parents that are relatives of each other) share almost no opposite-homozygote sites, so
vcfped's trio test (minimum 100 informative sites) cannot see them at any realistic marker count; only the
KING/IBD-based steps can recover them. See `breeding_sim/README.md`.

## Full workflow (end-to-end)

```bash
# Stage 1: MZ twin purge
python -m vcfped.vcfped_optimized input.vcf.gz --no-trio --no-gender -o step1
python utilities/purge_mz_twins.py input.vcf.gz step1.pair -o input_no_mz.vcf.gz

# Stage 2: Trio analysis
python -m vcfped.vcfped_optimized input_no_mz.vcf.gz --no-pairwise --no-gender -o step3
python utilities/vcfped_rename_samples.py step3

# Stage 3: Post-processing
python utilities/filter_trios.py step3_named --strategy best
python utilities/fixed_pedigree_viz.py step3_named_filtered -o pedigree.html

# Stage 4: PLINK2/BEAGLE verification
python utilities/prepare_plink_beagle.py step3_named_filtered input.vcf.gz --chr-prefix ch
bash step3_named_filtered_run_plink.sh        # produces _king_1st.kin0 and _king_ext.kin0
bash step3_named_filtered_run_beagle.sh       # produces _stat_phased/_phased VCFs + hap-IBD segments

# Stage 5: Pedigree refinement + extension
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext \
    --step1-pair step1_named
python utilities/extend_pedigree.py step3_named_filtered_refined.fam \
    step3_named_filtered_king_ext.kin0 --ibd step3_named_filtered_stat_ibd.ibd.gz

# Stage 6: Final confirmation with pedigree-aware IBD, once candidates from
# Stage 5 are merged back in (see merge_refined_fam.py) -- re-phase with the
# UPDATED fam so already-confirmed trios get the higher-precision pass.
python utilities/prepare_plink_beagle.py --from-fam step3_named_filtered_refined_merged.fam \
    input.vcf.gz -o revalidate --chr-prefix ch
bash revalidate_run_beagle.sh   # only need Step 2b/3 here; KING doesn't need re-running
python utilities/refine_pedigree.py step3_named_filtered step3_named_filtered_king_ext \
    --ibd revalidate_ibd.ibd.gz   # NOT revalidate_stat_ibd.ibd.gz -- see note below
```

### Final confirmation: pedigree-aware vs. statistical IBD

Every `--ibd` use earlier in this pipeline (`extend_pedigree.py`'s candidate search, `refine_pedigree.py`'s
first refinement pass) correctly points at `*_stat_ibd.ibd.gz` — the *statistical* phasing pass — because
those candidates aren't confirmed in any `.fam` yet, so pedigree-aware phasing has nothing to work
with for them regardless of which IBD file is named. But for trios that are *already* confirmed and
merged into the FAM (via `merge_refined_fam.py`), that limitation no longer applies: re-running BEAGLE's
pedigree-aware phasing (`_run_beagle.sh`'s Step 2b, `ped="${FAM}"`) with the *updated* FAM propagates
Mendelian constraints through those specific trios, giving `*_ibd.ibd.gz` (not `*_stat_ibd.ibd.gz`)
near-zero switch errors for them — a strictly higher-precision genome-coverage confirmation than the
statistical pass ever gives.

This is a workflow *sequencing* point, not new code — `refine_pedigree.py --ibd` already accepts any
hap-IBD file generically. The trick is regenerating `_run_beagle.sh` against the merged FAM first:
`prepare_plink_beagle.py --from-fam <merged.fam> input.vcf.gz -o revalidate` (the same `--from-fam`
mode used to revalidate Mendelian errors against a merged pedigree — see that section above) also
regenerates `_run_beagle.sh`'s `ped="${FAM}"` line to point at the merged FAM. Only Step 2b/3 need
re-running for this — KING kinship doesn't depend on phasing, so the original `_king_ext.kin0` is
still valid and doesn't need regenerating.

## Testing

### Manual smoke test

```bash
# Run with test files included in package
python -m vcfped.vcfped_optimized vcfped/testfiles/trioHG002_22X.vcf -o test_out
# Validate against known pedigree
python -m vcfped.vcfped_optimized vcfped/testfiles/trioHG002_22X.vcf -o test_out -ped vcfped/testfiles/trio.ped
```

### Validation cohort with known ground truth

`vcfped/testfiles/breeding_sim/` holds a simulated plant-breeding cohort (500 samples, 3000 SNPs,
20 chromosomes, multi-generation families with backcrosses/selfing/sib mating, 30 clone samples with
independent genotyping noise, ungenotyped ancestors, planted poor-quality samples and bad markers)
plus error-free ground truth: `.fam`, phased VCF, true genetic map, true hap-IBD/HBD segments,
duplicate pairs, pairwise relationships with expected and realised kinship. It is the target for
reconstructing pedigrees *from scratch* and scoring the result. `breeding_sim/README.md` documents
the design, every edge case, the file formats, how to score (compare parent *sets*; clones are
interchangeable; backcross trios separately) and its limitations (the 3000-SNP default clears vcfped's hard-coded 100-informative-site trio floor; at 1000 SNPs
only ~4/52 simple trios are called; hap-IBD's default `min-markers=100` needs lowering). Regenerate with
`python vcfped/testfiles/generate_breeding_sim.py [--seed N ...]`; the generator is covered by
`tests/test_generate_breeding_sim.py` (fast, runs in the default `pytest`).

### Automated test suite

```bash
pip install pytest   # or: conda env create -f environment.yml (includes pytest)
pytest                # run everything (~380 tests, ~45s)
pytest tests/test_vcfped_core.py -v       # one module
pytest -k "filter_trios"                  # by keyword
```

Structure (`tests/`):

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: bundled test VCF/PED, `vcfped_run` (pre-built vcfped output), dynamic module loader for the `utilities/` scripts (not an importable package) |
| `test_vcfped_core.py` | Both engines against the bundled HG002/HG003/HG004 trio: known pedigree confirmation, exact `.pair`/`.trio` values, byte-identical cross-engine output, gzip input, CLI flags |
| `test_checkped.py` | PED file parsing and trio validation |
| `test_purge_mz_twins.py` | MZ-twin grouping (transitive closure), VCF column purging, end-to-end detection+purge on a synthetic duplicate-sample VCF |
| `test_filter_trios.py` | `best`/`strict`/`consensus` strategies, single-header-no-duplication regression |
| `test_vcfped_rename_samples.py` | Numeric-ID → sample-name substitution, `--inplace`, custom suffix |
| `test_analyze_vcfped_pairs.py` | Smoke tests (requires pandas/numpy — skips cleanly if absent) |
| `test_fixed_pedigree_viz.py` | File validation (missing/wrong-column errors), `PedigreeBuilder` (trios, MZ collapse, generations, family detection, sibling stats), `_family_plot` trace batching (node/edge trace count independent of family size), width-scales-with-widest-generation, label-hiding above `LABEL_THRESHOLD`, `_family_summary_card` stats, end-to-end HTML report generation (including the `--max-diagram-size` summary-card fallback for oversized families), optional `.pair` file |
| `test_fam_to_trio.py` | Confirmed-only (PAT and MAT both known) trio extraction from `.fam`, `--trio` real-statistics lookup (first-match-wins across sources, `Not trio` row exclusion), `--king` kinship-confidence fallback (averaging, partial-data, no-data cases), Perc=0 sentinel for KING-only rows, int-parseable raw counts (regression guard for the `int(NaN)` crash), compatibility with `refine_pedigree.py`'s `load_trio_file()`/`REQUIRED_TRIO_COLS` and an end-to-end `fixed_pedigree_viz.py` run, end-to-end CLI |
| `test_prepare_plink_beagle.py` | Chromosome renaming, hermaphrodite/plant-crossing sex-conflict resolution for PLINK2 `--mendel`, generated-script regressions (`--set-all-var-ids`, two-pass KING thresholds, quality filters, R-heredoc variable expansion, synthetic genetic map, `--het`/`--missing` present and `--het` running on the LD-pruned set), `--from-fam` alternative input mode (full-trio extraction, sex carried over from an existing FAM, mutual exclusivity with `trio_prefix`, required `-o` to prevent overwriting the input FAM) |
| `test_refine_pedigree.py` | Error-rate-aware IBS0 thresholds, cohort-relative KING percentiles, genome-coverage PO confirmation, MZ-group loading from both `--step1-pair` and `--confirmed-duplicates` (mutual exclusivity, missing-column error, end-to-end console reporting, implausibly-large-group `WARNING` regression guard), `Assignment.confidence(ibs0_max)` and `parents_are_compatible(ibs0_max)` scaling against the calibrated ceiling instead of the hardcoded default (regression guards for the flattened-IBS0-score and over-rejection bugs), reclassified-inverted trios reaching `_refined.fam` (not just `_refined.trio`) with correct highest-Test1 tie-break and no-overwrite-of-existing-assignment behavior |
| `test_extend_pedigree.py` | KING indexing performance guard, founder/leaf bookkeeping, candidate-pool truncation regression, `--direction up/down` + `--max-iter` crash regression, generated-script path/map regressions, MZ-twin collapse + hap-IBD coverage discrimination (true duplicate vs. `CLOSE_INBRED` vs. `CLOSE_INBRED_POSSIBLE_DUP`), genome-proxy build performance guard (no O(M×P) rescans), fragmentation noise-floor self-calibration, `_confirmed_duplicates.tsv` (all-pairwise-combination expansion per group, written only when `--ibd` is given, excludes `CLOSE_BUT_DISTINCT` pairs) |
| `test_merge_refined_fam.py` | FAM load/write round-trip, founder-receives-new-parents, conflict handling (`keep-base`/`keep-new`), new-individual appending, base-order preservation, sex fill-in, trio/KING evidence joining with `NA` fallback and auto-derived trio paths, `_conflicts.tsv` written only when conflicts exist, `--score-conflicts` classification/recommendation logic (prefer_new/prefer_base/tie/tie_new_higher, IBS0-ceiling and percentile formulas matching `refine_pedigree.py`, `--king` requirement), `--auto-resolve` per-conflict resolution (mixed prefer_new/prefer_base outcome in one run, `--score-conflicts` requirement, base-conflict-tuple-corruption regression guard), Mendelian veto (`.fmendel` sorted-pair lookup, rate normalization, elevated-rate classification, `apply_mendel_veto`'s winning-side-only/never-promotes logic, `--base-mendel`/`--new-mendel` CLI validation, end-to-end downgrade to `tie_mendel_veto` and its effect on `--auto-resolve`), end-to-end CLI |
| `test_analyze_mendel_errors.py` | `.fmendel` parsing (header skipping, malformed-row tolerance), rate computation (`N / (n_variants × CHILD)`), cohort-relative percentile values/ranks, error-rate-derived expected-baseline formula matching `prepare_plink_beagle.py`'s documented `≈2×` relationship, dual outlier flagging (cohort-percentile/absolute/both/neither), `--bim`/`--n-variants` mutual exclusivity, per-individual repeat-offender aggregation across all co-parents (reproducing a real multi-co-parent pattern found in production data), `--min-co-parents` threshold exclusion, `.imendel` cross-reference, end-to-end CLI |
| `test_calibrate_error_rate.py` | `.pair` MZ-twins verdict filtering, `.kin0` IBS0 lookup (order-independent), unmatched-pair skipping, median-IBS0-halving derivation, no-matching-pairs CLI error, default output prefix, `--confirmed-duplicates` alternative ground-truth loading (ID1/ID2 header, missing-column error) and mutual exclusivity with `step1_pair_prefix` (both/neither given -> error), end-to-end CLI for both ground-truth sources |
| `test_analyze_hbd.py` | `.hbd.gz` parsing (per-individual grouping, LOD filtering, id1≠id2 mismatch counting, plain-text fallback), per-individual Mb/segment aggregation, rank-based outlier flagging (low and high, including the small-cohort boundary case a value-threshold approach misses), `--check-pairs` side-by-side comparison (missing data, low-flagged member, both-typical), end-to-end CLI |
| `test_check_genotype_quality.py` | `.het`/`.smiss` name-based column parsing (leading-`#` stripping, missing-column error), rank-based het/missingness outlier flagging (both-direction for F, high-only for F_MISS), Mendelian repeat-offender TSV filtering, genotyping-quality-confound cross-reference (confounded vs. clean split), end-to-end CLI |
| `test_analyze_ibd_blocks.py` | Pedigree-graph BFS connectivity (direct/sibling/max-hops/no-path/memoization), hap-IBD pair aggregation (LOD filtering, self-segment exclusion, sorted-key ordering), per-bucket (connected/unconnected) outlier flagging regression guard (a block unremarkable among connected pairs must not be judged against unconnected pairs' distribution or vice versa), per-chromosome hotspot detection (recurring-chromosome identification, jittered-boundary unification regression guard now trivial since there's only one grouping key per chromosome, small-segment exclusion floor, fraction-of-total-pairs self-calibration regression guard for the degenerate-percentile-of-few-chromosomes bug, hotspot pairs excluded from ranking and never flagged as unexpected, non-hotspot large blocks still flagged), reciprocal-overlap locus-scale hotspot detection (recurring sub-chromosomal region identification independent of chromosome coverage, min-overlap threshold behavior, exclusion of pairs already explained by the chromosome-scale mechanism, no-data case), end-to-end CLI including real-world chromosome-scale and locus-scale hotspot reproductions |
| `test_generate_breeding_sim.py` | Breeding-cohort generator: gamma-interference crossover rate and under-dispersion (nu=1 is Poisson), real gametes recombine at the map rate and are mosaics of the parent's two haplotypes, determinism per seed, tabular kinship/inbreeding against analytic values (PO and sib ¼, backcross ⅜, full-sib-mating F ¼, selfed F ½), relationship classes, true IBD/HBD, then one full run: size, anonymous IDs with no family signal in column order, monotone map matching the VCF, zero Mendelian violations in the truth, duplicates identical in truth but noisy as observed, planted QC problems, edge-case coverage, hub role switching, realised-vs-expected kinship, true trios keep a margin above vcfped's 100-informative-site floor (guard for the SNP-count default), truth files parse with the project's own loaders (`analyze_ibd_blocks`, `analyze_hbd`, `calibrate_error_rate`, `fam_to_trio`), HBD tracks inbreeding |
| `test_breeding_sim_validation.py` | (`integration`) vcfped scored against the breeding cohort's truth: every true duplicate called MZ twins, simple parent-offspring recall, full sibs not mistaken for parent-offspring, false MZ calls confined to deeply inbred lines (printed, not locked in), and simple outbred trios found as `Regular trio` on a ~100-sample subset of whole families |
| `test_score_against_truth.py` | Scorer and runner: a hand-built 15-sample truth with every expected number worked out (kind classification, fam outcomes incl. clone-equivalent parents and precision, trio unique/ambiguous/wrong/inverted/missed/spurious, purge losses counted as misses, pair verdicts, numeric-ID + `--log` resolution, KING bands incl. exact boundaries, quality plants, empty Mendel list skipped, IBD recall/false pairs/conserved block, HBD recovery, `--samples` restriction); the shipped truth scored against itself is perfect; runner syntax/help/required `-o`, and its `--stages score` glue and resume. Seven deliberate scorer bugs (clone equivalence, purge accounting, pair-order, min-cM, KING thresholds, ambiguity, lost classes) were each caught |

Optional dependencies (`numpy`, `numba`, `tqdm`, `pandas`, `plotly`) are detected at collection time — tests requiring them skip cleanly rather than failing when absent (see `requires_pandas`/`requires_plotly`/`HAS_NUMPY` in `conftest.py`). Note: `vcfped_optimized.py` itself has a **hard** numpy dependency (not optional — see Dependencies above), so its tests in `test_vcfped_core.py` skip individually when numpy is unavailable while `vcfped.py`'s tests still run.

Most utility tests use `vcfped.py` (not `vcfped_optimized.py`) to produce their input fixtures, since they only need *some* valid vcfped output — engine-specific correctness is covered exhaustively in `test_vcfped_core.py` alone.


**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
