# IDEAS for vcfped (1.5.0-optimized)

Collected while validating 12 vcfped runs (11 small populations, 11–106 samples; and one joint run, 424 samples,
16.8k variants, 12.6M trios, `-p 0 10 25 50 75 -j 64 --no-gender`). Every output number (pair rows, trio rows, all 37.9M
log rows of the joint run) was reproduced with an independent numpy/BLAS re-implementation with 0 mismatches, so the
observations below are about behaviour and design, not about wrong arithmetic. Items are ordered roughly by value.
Line numbers refer to `vcfped_optimized.py`.

Suggested way to use this file: open the vcfped project directory with Claude and ask it to go through the sections one
by one, starting with "Behaviour worth deciding on".

## 1. Behaviour worth deciding on

1. **First data record is not analysed.** `get_file_info` reads the first variant for format detection (l.152), and the
   analysis loop skips another line after `seek(startpos)` (l.327, plain files). Result: vcfped analyses `records - 1`
   variants in every run we validated (8094 → 8093 etc.). For `.vcf.gz` the loop at l.318–324 already consumes the first
   data line (the `break` happens after reading it) and then `next(f, None)` skips a *second* record, so gz input appears
   to lose two records. This is from reading the code; not tested on gz. Suggest: iterate from the first data line
   deterministically and add a test that `Autos(p0) == number of PASS autosomal biallelic records`.
2. **Best-of-percentiles selection inflates calls.** For each pair/trio only the best row over all `-p` values is kept
   (verdict priority, then POscore/T1). More percentiles = more chances to cross a threshold. On our data, extra
   percentiles raised mother–child calls for true mothers from 94/288 to 266/288 (good), and with `-po 98` produced one
   call for a child whose mother is demonstrably wrong. Suggest: report all percentiles in the .pair/.trio files (or at
   least the percentile that was chosen *and* the number of percentiles tried), and mention the bias in the docs.
3. **`Inverted/generational trio` label.** For the joint run 8061 of 8062 trio calls are labelled "Inverted" and 1 "Regular",
   because the label only means "Test2 (BB×BB→BB) < 95%". In inbreeding/selfing material T2 almost never reaches 95%.
   The name suggests a reversed relationship. Suggest renaming (e.g. `Trio (T2 low)`) or making T2 optional.
4. **Trio needs ≥100 AA×BB sites but the percentile filter removes them.** With the site-level cumulative filter, the
   median number of informative AA×BB sites for confirmed trios drops from 372 (p0) to 88 (p50) to 3 (p75), so T1 ≥ 90 is
   often reached only where `t1 < 100` (no call) and fails where there are enough sites. Only 13/67 confirmed trios got a
   call; with a genotype pre-filter (set DP<8 or GQ<20 genotypes to `./.` before running) 44/67 did. Suggest an optional
   `--min-dp/--min-gq` genotype mask inside vcfped (currently only site-level percentile filters exist), and/or report the
   best-T1 row for *each* percentile.
5. **`-po` inconsistency.** We ran populations with `-po 99` and the joint set with `-po 98`; scores printed in the logs are
   not comparable between those. Suggest printing all thresholds in the header of every output file, not only the log.
6. **No `_named` files for the joint run** (and they are not mentioned in the log). Names must be reconstructed from the
   VCF column order. Suggest always writing them (or writing names instead of indices in the main files).
7. **MZ label for selfed offspring.** A child that is nearly identical to a selfing mother is reported as `MZ twins`
   (63 of 266 correct mother–child calls). Perhaps `MZ twins / clone / selfed offspring` in the verdict text, or a
   separate IBS2 column warning.
8. **Heterozygosity outliers.** One sample (LA000006) with 26–67 % heterozygous calls (others 5–20 %) got a
   Parent-child call with 260 of 411 children in the joint run. A sample-level QC section in the log (call rate,
   heterozygosity, outlier flag) would have made this visible immediately.

## 2. Performance and parallelisation

Measured (joint run, 64 jobs requested on the server): wall 1115 s, user 15834 s (**1451 % CPU of 6400 %**), max RSS 26.3 GB,
log 2.1 GB (37.9M rows), per-percentile trio time 190/185/170/150/136 s for 16.3k→3.6k sites, pair phase 3 s in total,
about 260 s outside the pair/trio phases. Populations (≤106 samples): 3.7–20.6 s, of which ~3 s is worker start-up.

Reference implementation (see `scripts/v2_vcfped_replica.py` and `scripts/v2_common.py` of the offspring project): the same
numbers in **78 s and ~3 GB on a 10-core M1 Pro with one Python process**, using numpy matmul on 0/1 indicator matrices.

Ideas, with the expected effect:

1. **Matrix formulation for pairs and trios.** For a block of parent pairs and all children, the counts are products of
   indicator matrices (sites × samples): for pairs `AA`, `BB`, `AB` indicators give IBS0/IBS2/EitherBB with a few GEMMs; for
   trios, the AA×BB→AB and BB×BB→BB counts are `(parentAA·parentBBᵀ) ∘ childAB` summed over sites, i.e. also GEMMs after
   stacking parent combinations. Replaces per-trio loops; the per-percentile time then scales with the number of sites
   instead of staying flat at ~190 s as it does now.
2. **Per-percentile time is flat while sites shrink 4.5×.** That means the cost is overhead (task distribution, result
   transfer, Python-level loops), not arithmetic. Profile with `py-spy` / `cProfile` in one worker before adding more cores.
3. **Memory.** RSS 26 GB vs 3 GB for the single-process replica. vcfped already uses `shared_memory` for the genotype
   matrix, so look at per-worker result buffers and the trio state arrays (n_pairs × n_children, int16/float arrays).
4. **Percentile masks are nested** (cumulative filter). Counts for p75 can be derived from p50 by *subtracting* the removed
   sites, so only p0 needs a full pass. Untested idea.
5. **Pre-pruning trios.** Trios whose child is incompatible with one of the parents at the pair level (high IBS0) cannot be
   a regular trio; dropping them early would remove most of the 12.6M trios. Verify with the replica that the output rows
   are unchanged before adopting.
6. **The log.** 37.9M rows (all pairs and all triple×pivot rows) = 2.1 GB, and writing it costs minutes. Make it opt-in
   (`--full-log`) or write compact binary/parquet; the .pair/.trio files already hold the selected rows.
7. **First batch JIT.** The first batch of every process pays for numba compilation (we saw ~24 s for the first batch in a
   joint run). `cache=True` is set; check that the cache directory is writable on the cluster and shared by workers, or
   compile once in the parent before forking.
8. **Worker count.** Small runs spend most of their 3–4 s on starting 64 workers; use `min(jobs, n_batches)`.
9. **Progress bars** are written by tqdm into the stderr file (one update per batch). One line per 10 % is enough.
10. **Incremental / resumable output.** For many populations, an SQLite or parquet store with upserts keyed by
    (sample set, settings) would allow adding one population without rerunning everything.

## 3. Documentation

- State the exact selection rules in the README: first record skipped; percentile filter is cumulative and **site-level**
  (a site is dropped when *any* sample fails DP/GQ, or any heterozygote has an AD ratio outside the percentile threshold);
  threshold = `sorted(values)[int(p/100*len)]`; the best row over percentiles is kept; the trio pivot with the highest T1
  (1 decimal), ties → lowest percentile, is written; verdict priority MZ > PC > other.
- Document what `Autos` means in each file (sites left after the filter at that percentile).
- Document that genotype-level DP/GQ are not used unless the VCF was filtered beforehand, and give the pre-filter recipe
  (`bcftools +setGT -- -t q -n . -i 'FMT/DP<8 | FMT/GQ<20'`) as the recommended input for plant/inbred material.
- Explain the `-po`, `-mz`, `-t1`, `-t2` defaults, and that PO 99 % is strict for low-coverage panel data.
- A "known quirks" section (items 1–8 above).

## 4. Tests and validation

- Add a tiny synthetic VCF with known pedigree (selfed, outcross, clone, wrong mother) and expected verdicts.
- Reuse the independent replica (`v2_vcfped_replica.py`, ~200 lines, exact on all 12 runs, 0 mismatches) as a regression
  test: run both, compare `.pair`, `.trio` and every log row. This is also how we studied what other `-p`, `-po` and
  genotype-filter settings would have given without re-running the cluster job.
- Property tests: scores invariant to sample order; symmetry of pair scores; monotone site counts over percentiles.

## 5. Interpretation helpers that would be nice inside vcfped

- Per-child summary table: best mother candidate, best father candidate given the mother, and a plain-language
  verdict (mother OK / mother mismatch / father determined / father not in set).
- A relatedness warning when two candidate parents are nearly identical (our 24LA0029 mother and father share 83 % of genotypes),
  because trio and "selfed" cannot be distinguished then.
- Flag samples with extreme heterozygosity or very low call rate before any pair/trio analysis.

## 6. Additional insights (depth/quality, percentiles, scaling)

1. **Sample depth decides usability, and vcfped does not report it.** In the 11 population VCFs, children with median DP ≤ 20×
   were unusable in 35/56 cases (62 %), 21–40×: 1/11, above 40×: 0/344. Unusable samples show apparent heterozygosity
   of ~67 % (vs 24 %) and call rate ~14 % (vs 82 %). Idea: a per-sample QC block in the log (median DP, call rate,
   heterozygosity, flag) and an optional `--min-sample-dp` / `--exclude-flagged` so such samples cannot generate
   pair/trio rows.
2. **Only the highest percentile changed pair calls.** On our data `-p 0 50 75` gives exactly the same mother–child calls as
   `-p 0 10 25 50 75` (266/288 vs 94/288 with `-p 0 50`), at roughly half the run time. p10/p25 added nothing.
   Idea: document which percentiles matter, or add an adaptive mode (add percentiles only while new calls appear).
3. **Site-level percentile filter scales badly with sample count.** A site is dropped when *any* sample fails the DP/GQ/AD
   threshold, so with many samples of uneven depth almost no sites survive (already ~3.6k of 16.8k at p75 for 424 samples;
   informative AA×BB trio sites fall to a median of 3). Hypothesis for ~4000 samples: nearly empty site sets. Idea: pairwise
   or per-trio site masks (use genotypes that pass DP/GQ for the samples involved, with a minimum number of shared sites)
   instead of one global mask.
4. **Cubic scaling.** Trio work scales with n³: 424 → 4000 samples is ~840× (≈ 11 days at the observed 1115 s; extrapolation,
   not measured), and the (pair × child) state arrays do not fit in memory. A candidate-parent approach (pair scores for all
   pairs via GEMM, then trios only for candidate parents per child) would be linear in n; needs to be verified against the
   full result on a subset using the replica.
5. **Direction and cycles.** Pair/trio scores are symmetric (parent→child vs child→parent give the same IBS0). For pedigrees with
   backcrosses or circular crosses, an optional metadata input (generation/date) and an acyclicity constraint would be
   needed to turn scores into a directed pedigree.
6. **Report uncertainty, not just the best call.** Add margin to the second-best candidate and flag near-identical candidate
   parents (in 24LA0029 mother and father share 83 % of genotypes).
7. **Calibrate thresholds on a truth set.** The confirmed mother–child pairs here had IBS0 ≤ 0.9 %, non-matching ones ≥ 1.7 %;
   a calibration step (or a "report distributions" mode) would make thresholds such as PO 99 % and T1 90 % less arbitrary.
