#!/usr/bin/env bash
#
# Run the VCFped pipeline end-to-end on a VCF with known ground truth and score it.
#
#   utilities/run_validation_pipeline.sh -o OUTDIR [options]
#
# Stages (in order; each is skipped if already done, so a killed job resumes):
#   pairs    vcfped pairwise stage + sample renaming
#   purge    purge_mz_twins.py -> no_mz.vcf.gz
#   trios    vcfped trio stage on the purged VCF + renaming        (the slow one: ~40 min for 500 samples with numba)
#   filter   filter_trios.py --strategy best
#   plink    prepare_plink_beagle.py, then the generated PLINK2 script (KING, QC)
#   beagle   generated BEAGLE + hap-IBD script (only if BEAGLE_CMD and HAPIBD_CMD are set)
#   refine   refine_pedigree.py (with --include-inverted unless --no-inverted)
#   qc       check_genotype_quality.py, analyze_mendel_errors.py (if Mendel output exists)
#   score    utilities/score_against_truth.py on everything that exists
#
# Options:
#   --vcf FILE                input VCF   (default: the breeding_sim cohort)
#   --truth PREFIX            truth prefix (default: breeding_sim/breeding_sim)
#   -j N                      vcfped worker processes (default: min(cores, 10); the trio stage's
#                             result writing is single-process, so more workers rarely help)
#   --stages a,b,c            run only these stages
#   --force                   rerun stages even if marked done
#   --true-map                give hap-IBD the true genetic map instead of the synthetic uniform one
#   --hapibd-min-markers N    hap-IBD min-markers (default 10; its default of 100 is too many here)
#   --chr-prefix P            chromosome prefix for prepare_plink_beagle.py (default: ch)
#   --error-rate X            --error-rate for analyze_mendel_errors.py (default 0.005)
#   --no-inverted             do not pass --include-inverted to refine_pedigree.py
#
# Environment:
#   PYTHON      interpreter (default python3)      PLINK2      plink2 binary (default plink2)
#   BEAGLE_CMD  e.g. "java -Xmx16g -jar beagle.jar"   HAPIBD_CMD  e.g. "java -Xmx16g -jar hap-ibd.jar"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VCF="$ROOT/vcfped/testfiles/breeding_sim/breeding_sim.vcf.gz"
TRUTH="$ROOT/vcfped/testfiles/breeding_sim/breeding_sim"
OUT=""
CORES=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) | head -n1)
JOBS=$(( CORES > 10 ? 10 : CORES ))
ALL_STAGES="pairs,purge,trios,filter,plink,beagle,refine,qc,score"
STAGES=""
FORCE=0
TRUE_MAP=0
INVERTED=1
MIN_MARKERS=10
CHR_PREFIX=ch
ERROR_RATE=0.005
PYTHON="${PYTHON:-python3}"
PLINK2="${PLINK2:-plink2}"
BEAGLE_CMD="${BEAGLE_CMD:-}"
HAPIBD_CMD="${HAPIBD_CMD:-}"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUT="$2"; shift 2;;
    --vcf) VCF="$2"; shift 2;;
    --truth) TRUTH="$2"; shift 2;;
    -j) JOBS="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --true-map) TRUE_MAP=1; shift;;
    --hapibd-min-markers) MIN_MARKERS="$2"; shift 2;;
    --chr-prefix) CHR_PREFIX="$2"; shift 2;;
    --error-rate) ERROR_RATE="$2"; shift 2;;
    --no-inverted) INVERTED=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ -n "$OUT" ]] || { echo "Error: -o OUTDIR is required" >&2; usage >&2; exit 2; }
if [[ -z "$STAGES" ]]; then
  STAGES="$ALL_STAGES"
  if [[ -z "$BEAGLE_CMD" || -z "$HAPIBD_CMD" ]]; then
    STAGES="${STAGES/beagle,/}"           # no BEAGLE/hap-IBD configured: skip that stage
  fi
fi

mkdir -p "$OUT/logs" "$OUT/bin"
OUT="$(cd "$OUT" && pwd)"
PFX="$OUT/step3_named_filtered"
cd "$ROOT"

# Per-stage logging: status lines go to the terminal (fd 3/4), tool output to logs/<stage>.log.
exec 3>&1 4>&2
CURRENT=""
trap 'rc=$?; if [[ $rc -ne 0 && -n "$CURRENT" ]]; then echo "[$CURRENT] FAILED (exit $rc); last lines of $OUT/logs/$CURRENT.log:" >&4; tail -n 15 "$OUT/logs/$CURRENT.log" >&4; fi' EXIT

begin() {   # returns 0 if the stage should run
  [[ ",$STAGES," == *",$1,"* ]] || return 1
  if [[ -f "$OUT/.done_$1" && $FORCE -eq 0 ]]; then
    echo "[$1] already done (use --force to rerun)" >&3
    return 1
  fi
  CURRENT="$1"
  rm -f "$OUT/.done_$1"                 # a failed rerun must not leave a stale "done" marker behind
  echo "[$1] start $(date +%T)" >&3
  exec >"$OUT/logs/$1.log" 2>&1
}
end() {
  exec >&3 2>&4
  touch "$OUT/.done_$1"
  echo "[$1] done $(date +%T)" >&3
  CURRENT=""
}

if [[ -n "$HAPIBD_CMD" ]]; then       # wrapper so the generated script's hap-IBD calls get min-markers
  printf '#!/usr/bin/env bash\nexec %s "$@" min-markers=%s\n' "$HAPIBD_CMD" "$MIN_MARKERS" > "$OUT/bin/hap-ibd"
  chmod +x "$OUT/bin/hap-ibd"
fi

if begin pairs; then
  "$PYTHON" -m vcfped.vcfped_optimized "$VCF" --no-trio --no-gender --quiet -j "$JOBS" -o "$OUT/step1"
  "$PYTHON" utilities/vcfped_rename_samples.py "$OUT/step1"
  end pairs
fi

if begin purge; then
  "$PYTHON" utilities/purge_mz_twins.py "$VCF" "$OUT/step1.pair" -o "$OUT/no_mz.vcf.gz"
  end purge
fi

if begin trios; then
  "$PYTHON" -m vcfped.vcfped_optimized "$OUT/no_mz.vcf.gz" --no-pairwise --no-gender --quiet -j "$JOBS" -o "$OUT/step3"
  "$PYTHON" utilities/vcfped_rename_samples.py "$OUT/step3"
  end trios
fi

if begin filter; then
  "$PYTHON" utilities/filter_trios.py "$OUT/step3_named" --strategy best
  end filter
fi

if begin plink; then
  command -v "$PLINK2" >/dev/null || { echo "plink2 not found (set PLINK2=/path/to/plink2)"; exit 1; }
  prep=()
  [[ -n "$BEAGLE_CMD" ]] && prep+=(--beagle "$BEAGLE_CMD")
  [[ -n "$HAPIBD_CMD" ]] && prep+=(--hapibd "$OUT/bin/hap-ibd")
  "$PYTHON" utilities/prepare_plink_beagle.py "$PFX" "$VCF" --chr-prefix "$CHR_PREFIX" ${prep[@]+"${prep[@]}"}
  RUN="${PFX}_run_plink.sh"
  sed "s#^PLINK2=.*#PLINK2=\"$PLINK2\"#" "$RUN" > "$RUN.tmp" && mv "$RUN.tmp" "$RUN"
  if ! bash "$RUN"; then
    if grep -q "Unrecognized flag ('--mendel')" "$OUT/logs/plink.log"; then
      echo "NOTE: this plink2 has no --mendel. Rerunning with that step replaced; there will be no Mendelian-error"
      echo "      output, so the marker/sample Mendel scoring is skipped."
      printf 'FID\tPAT\tMAT\tCHLD\tCHR\tSNP\tCODE\tERROR\n' > "${PFX}_mendel.mendel"
      sed 's#^    --mendel \\$#    --freq \\#' "$RUN" > "$RUN.nomendel.sh"
      bash "$RUN.nomendel.sh"
    else
      exit 1
    fi
  fi
  end plink
fi

if begin beagle; then
  [[ -n "$BEAGLE_CMD" && -n "$HAPIBD_CMD" ]] || { echo "set BEAGLE_CMD and HAPIBD_CMD"; exit 1; }
  if [[ $TRUE_MAP -eq 1 ]]; then
    cp "${TRUTH}_truth.map" "${PFX}_synthetic.map"      # the generated script only builds a map if this is absent
  fi
  bash "${PFX}_run_beagle.sh"
  end beagle
fi

if begin refine; then
  ref=(--step1-pair "$OUT/step1_named")
  if [[ $INVERTED -eq 1 ]]; then ref+=(--include-inverted); fi
  if [[ -f "${PFX}_stat_ibd.ibd.gz" ]]; then ref+=(--ibd "${PFX}_stat_ibd.ibd.gz"); fi
  "$PYTHON" utilities/refine_pedigree.py "$PFX" "${PFX}_king_ext" "${ref[@]}"
  end refine
fi

if begin qc; then
  "$PYTHON" utilities/check_genotype_quality.py "${PFX}_plink.het" "${PFX}_plink.smiss" -o "$OUT/qc"
  if [[ -f "${PFX}_mendel.fmendel" ]] && [[ "$(wc -l < "${PFX}_mendel.fmendel")" -gt 1 ]]; then
    "$PYTHON" utilities/analyze_mendel_errors.py "${PFX}_mendel.fmendel" --bim "${PFX}_plink.bim" \
        --error-rate "$ERROR_RATE" -o "$OUT/mendel"
  else
    echo "No Mendelian-error output (.fmendel); skipping analyze_mendel_errors.py"
  fi
  end qc
fi

if begin score; then
  sc=(--truth "$TRUTH" --samples "$VCF" -o "$OUT/validation")
  [[ -f "$OUT/step1_named.pair" ]] && sc+=(--pair "$OUT/step1_named.pair")
  [[ -f "$OUT/logs/purge.log" ]] && sc+=(--purge-log "$OUT/logs/purge.log")
  [[ -f "$OUT/step3_named.trio" ]] && sc+=(--trio "$OUT/step3_named.trio")
  [[ -f "${PFX}_refined.fam" ]] && sc+=(--fam "${PFX}_refined.fam")
  [[ -f "${PFX}_king_ext.kin0" ]] && sc+=(--king "${PFX}_king_ext.kin0")
  [[ -f "$OUT/qc_genotype_quality.tsv" ]] && sc+=(--genotype-quality "$OUT/qc_genotype_quality.tsv")
  [[ -f "$OUT/mendel_individual_analysis.tsv" ]] && sc+=(--mendel-individual "$OUT/mendel_individual_analysis.tsv")
  [[ -f "${PFX}_mendel_artifact_snps.txt" ]] && sc+=(--mendel-artifacts "${PFX}_mendel_artifact_snps.txt")
  [[ -f "${PFX}_stat_ibd.ibd.gz" ]] && sc+=(--ibd "${PFX}_stat_ibd.ibd.gz")
  [[ -f "${PFX}_ibd.ibd.gz" ]] && sc+=(--ibd "${PFX}_ibd.ibd.gz")
  [[ -f "${PFX}_ibd.hbd.gz" ]] && sc+=(--hbd "${PFX}_ibd.hbd.gz")
  "$PYTHON" utilities/score_against_truth.py "${sc[@]}"
  end score
  cat "$OUT/validation_score_report.txt"
fi
