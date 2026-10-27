"""
End-to-end integration tests for the complete VCFped workflow.

These tests run the full pipeline from VCF input through trio analysis,
filtering, MZ purge, pedigree refinement, and extension.  They are marked
``integration`` and are EXCLUDED from the default ``pytest`` run to keep CI fast.

Run the full suite explicitly:
    pytest -m integration -v

Or a single stage:
    pytest -m integration -v -k "Stage1"

Scenarios in the synthetic VCF (vcfped/testfiles/integration.vcf):
  clean_trio   TRI_P1 + TRI_P2 → TRI_C1       (outbred parents)
  three_gen    GEN3_GP1+GEN3_GP2 → GEN3_P1; GEN3_P1+GEN3_P2 → GEN3_C1
  inbred       F6-selfed INBRD_L1 × INBRD_L2 → INBRD_F1
  mz_twin      MZ_A; MZ_B = clone of MZ_A with ~0.1% sequencing error
  half_sib     HS_DAD × HS_MOM1 → HS_C1; HS_DAD × HS_MOM2 → HS_C2
  selfing      SELF_P1 × SELF_P1 → SELF_C1  (plant self-fertilisation)
  control      CTRL1 (unrelated negative control)

Fixture pipeline (module-scoped to avoid re-running vcfped):
  vcfped_run → named_run → filter_run
                         → purge_run
  filter_run → refine_run → extend tests
  filter_run → prepare_plink tests

Re-generate test files:
    python vcfped/testfiles/generate_integration_vcf.py
"""

import csv
import os
import shutil
import subprocess
import sys

import pytest

TESTFILES        = os.path.join(os.path.dirname(__file__), "..", "vcfped", "testfiles")
VCFPED_PY        = os.path.join(os.path.dirname(__file__), "..", "vcfped", "vcfped.py")
VCFPED_OPT_PY    = os.path.join(os.path.dirname(__file__), "..", "vcfped", "vcfped_optimized.py")
UTILITIES        = os.path.join(os.path.dirname(__file__), "..", "utilities")

VCF_PATH   = os.path.join(TESTFILES, "integration.vcf")
TRUE_PED   = os.path.join(TESTFILES, "integration_true.ped")
KING_KIN0  = os.path.join(TESTFILES, "integration_king.kin0")
GENERATOR  = os.path.join(TESTFILES, "generate_integration_vcf.py")

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(script, args, cwd=None):
    cmd = [sys.executable, script] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=cwd or os.path.dirname(TESTFILES))


def _read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _verdict_pairs(rows, verdict):
    return [r for r in rows if verdict.lower() in r.get("Verdict", "").lower()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_test_files():
    """Generate integration VCF + KING + PED if any are missing."""
    if not os.path.exists(VCF_PATH):
        r = subprocess.run([sys.executable, GENERATOR], capture_output=True, text=True)
        assert r.returncode == 0, f"Generator failed:\n{r.stdout}\n{r.stderr}"
    for p in (VCF_PATH, TRUE_PED, KING_KIN0):
        assert os.path.exists(p), f"Missing test file: {p}"


def _vcfped_script():
    """Return vcfped_optimized.py if numpy is available, else vcfped.py."""
    try:
        import numpy  # noqa: F401
        return VCFPED_OPT_PY
    except ImportError:
        return VCFPED_PY


@pytest.fixture(scope="module")
def vcfped_run(tmp_path_factory):
    """Stage 1a: vcfped on integration.vcf → numeric-index .pair/.trio/.log."""
    prefix = str(tmp_path_factory.mktemp("s1") / "out")
    script = _vcfped_script()
    r = _run(script, [VCF_PATH, "--no-gender", "-o", prefix])
    assert r.returncode == 0, f"vcfped failed:\n{r.stderr}"
    return prefix


@pytest.fixture(scope="module")
def named_run(tmp_path_factory, vcfped_run):
    """
    Stage 1b: vcfped_rename_samples.py → sample-name .pair/.trio/.log.
    Writes to <prefix>_named.* by default.
    """
    work = str(tmp_path_factory.mktemp("s1b"))
    src = os.path.join(work, "out")
    for ext in (".pair", ".trio", ".log"):
        shutil.copy(vcfped_run + ext, src + ext)
    r = _run(os.path.join(UTILITIES, "vcfped_rename_samples.py"), [src])
    assert r.returncode == 0, f"vcfped_rename_samples failed:\n{r.stderr}"
    return src + "_named"   # named.pair, named.trio, named.log exist


@pytest.fixture(scope="module")
def filter_run(tmp_path_factory, named_run):
    """
    Stage 2: filter_trios on named vcfped output.
    Uses -o so the output prefix is explicit and predictable.
    """
    out = str(tmp_path_factory.mktemp("s2") / "filtered")
    r = _run(os.path.join(UTILITIES, "filter_trios.py"),
             [named_run, "--strategy", "best", "-o", out])
    assert r.returncode == 0, f"filter_trios failed:\n{r.stderr}"
    return out   # out.trio, out_inverted.trio (if inverted rows existed)


@pytest.fixture(scope="module")
def purge_run(tmp_path_factory, vcfped_run):
    """Stage 3: purge_mz_twins on integration VCF using numeric pair file."""
    out_vcf = str(tmp_path_factory.mktemp("s3") / "purged.vcf")
    r = _run(os.path.join(UTILITIES, "purge_mz_twins.py"),
             [VCF_PATH, vcfped_run + ".pair", "-o", out_vcf])
    assert r.returncode == 0, f"purge_mz_twins failed:\n{r.stderr}"
    return out_vcf


@pytest.fixture(scope="module")
def refine_run(tmp_path_factory, filter_run):
    """
    Stage 4: refine_pedigree using the synthetic KING file.

    The KING file uses sample names (generated from integration.vcf).
    filter_run output also uses sample names (via named_run).
    TRUE_PED is passed as the starting FAM so all 21 samples are present.
    """
    work = str(tmp_path_factory.mktemp("s4"))
    out = os.path.join(work, "refined")
    fam_in = os.path.join(work, "input.fam")
    shutil.copy(TRUE_PED, fam_in)
    king_prefix = KING_KIN0.replace(".kin0", "")
    r = _run(os.path.join(UTILITIES, "refine_pedigree.py"),
             [filter_run, king_prefix, "--fam", fam_in, "-o", out])
    assert r.returncode == 0, f"refine_pedigree failed:\n{r.stderr}\n{r.stdout}"
    return out   # out_refined.trio, out_refined.fam, out_refinement_report.txt


# ---------------------------------------------------------------------------
# Stage 1a — vcfped pair/trio output (numeric indices)
# ---------------------------------------------------------------------------

class TestStage1aVcfPed:

    def test_output_files_created(self, vcfped_run):
        for ext in (".pair", ".trio", ".log"):
            assert os.path.exists(vcfped_run + ext), f"Missing {ext}"

    def test_mz_twins_by_index(self, vcfped_run):
        """MZ_A (12) and MZ_B (13) must be called MZ twins."""
        rows = _read_tsv(vcfped_run + ".pair")
        mz = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "MZ twins")}
        assert frozenset(["12", "13"]) in mz, (
            f"MZ_A/MZ_B (indices 12,13) not in MZ calls: {mz}"
        )

    def test_clean_trio_parents_po(self, vcfped_run):
        """TRI_P1 (1) and TRI_P2 (2) must each show PO with TRI_C1 (3)."""
        rows = _read_tsv(vcfped_run + ".pair")
        po = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "Parent-child")}
        assert frozenset(["1", "3"]) in po, f"TRI_P1-TRI_C1 PO not found. PO: {po}"
        assert frozenset(["2", "3"]) in po, f"TRI_P2-TRI_C1 PO not found. PO: {po}"

    def test_three_gen_all_po_pairs(self, vcfped_run):
        """All four PO links in the three-generation chain must be detected."""
        rows = _read_tsv(vcfped_run + ".pair")
        po = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "Parent-child")}
        for e in [("4", "6"), ("5", "6"), ("6", "8"), ("7", "8")]:
            assert frozenset(e) in po, f"Three-gen PO pair {e} not found in {po}"

    def test_half_sibling_shared_father(self, vcfped_run):
        """HS_DAD (14) must appear in PO with both HS_C1 (17) and HS_C2 (18)."""
        rows = _read_tsv(vcfped_run + ".pair")
        po = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "Parent-child")}
        assert frozenset(["14", "17"]) in po, "HS_DAD-HS_C1 PO not found"
        assert frozenset(["14", "18"]) in po, "HS_DAD-HS_C2 PO not found"

    def test_selfed_offspring_po(self, vcfped_run):
        """SELF_P1 (19) must be called PO with SELF_C1 (20)."""
        rows = _read_tsv(vcfped_run + ".pair")
        po = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "Parent-child")}
        assert frozenset(["19", "20"]) in po, "SELF_P1-SELF_C1 PO not found"

    def test_inbrd_regular_trio(self, vcfped_run):
        """
        After F6 selfing, INBRD_L1 (9) × INBRD_L2 (10) → INBRD_F1 (11) has many
        AA+BB sites and should score as the best Regular trio.
        """
        rows = _read_tsv(vcfped_run + ".trio")
        triples = {frozenset(r["Triple"].split(","))
                   for r in rows if "Regular trio" in r.get("Verdict", "")}
        assert frozenset(["9", "10", "11"]) in triples, (
            f"INBRD Regular trio not found. Triples: {triples}"
        )

    def test_ctrl1_not_in_po(self, vcfped_run):
        """CTRL1 (21) is unrelated; must not appear in any Parent-child call."""
        rows = _read_tsv(vcfped_run + ".pair")
        ctrl_po = [r for r in _verdict_pairs(rows, "Parent-child")
                   if "21" in r["Pair"].split(",")]
        assert not ctrl_po, f"CTRL1 in PO calls: {ctrl_po}"

    def test_ped_validation_succeeds(self, tmp_path):
        """vcfped -ped must validate the ground-truth pedigree without error.
        Note: --no-gender is incompatible with -ped in vcfped_optimized.py, so
        gender analysis runs (produces empty result for chr1-only VCF) but vcfped
        must still exit 0 and validate the trio assignments that ARE found.
        """
        out = str(tmp_path / "ped_val")
        r = _run(_vcfped_script(), [VCF_PATH, "-o", out, "-ped", TRUE_PED])
        assert r.returncode == 0, f"vcfped -ped failed:\n{r.stderr}"


# ---------------------------------------------------------------------------
# Stage 1b — renamed output (sample names instead of indices)
# ---------------------------------------------------------------------------

class TestStage1bRenameSamples:

    def test_named_files_exist(self, named_run):
        # vcfped_rename_samples.py writes .pair and .trio (and .gender if present);
        # it does NOT write a renamed .log — only .pair/.trio are checked here.
        for ext in (".pair", ".trio"):
            assert os.path.exists(named_run + ext), f"Missing {named_run + ext}"

    def test_mz_twins_by_name(self, named_run):
        rows = _read_tsv(named_run + ".pair")
        mz = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "MZ twins")}
        assert frozenset(["MZ_A", "MZ_B"]) in mz, f"MZ_A/MZ_B not in named MZ calls: {mz}"

    def test_clean_trio_parents_named(self, named_run):
        rows = _read_tsv(named_run + ".pair")
        po = {frozenset(r["Pair"].split(",")) for r in _verdict_pairs(rows, "Parent-child")}
        assert frozenset(["TRI_P1", "TRI_C1"]) in po, "TRI_P1-TRI_C1 not in named PO"
        assert frozenset(["TRI_P2", "TRI_C1"]) in po, "TRI_P2-TRI_C1 not in named PO"

    def test_inbrd_trio_named(self, named_run):
        rows = _read_tsv(named_run + ".trio")
        triples = {frozenset(r["Triple"].split(","))
                   for r in rows if "Regular trio" in r.get("Verdict", "")}
        assert frozenset(["INBRD_L1", "INBRD_L2", "INBRD_F1"]) in triples, (
            f"Named INBRD trio not found. Triples: {triples}"
        )


# ---------------------------------------------------------------------------
# Stage 2 — filter_trios
# ---------------------------------------------------------------------------

class TestStage2FilterTrios:

    def test_filtered_trio_written(self, filter_run):
        assert os.path.exists(filter_run + ".trio"), f"Missing {filter_run}.trio"

    def test_no_duplicate_header(self, filter_run):
        lines = open(filter_run + ".trio").read().splitlines()
        n_hdr = sum(1 for l in lines if l.startswith("Triple\t"))
        assert n_hdr == 1, f"Expected exactly 1 header line, found {n_hdr}"

    def test_inbrd_trio_in_filtered_output(self, filter_run):
        rows = _read_tsv(filter_run + ".trio")
        triples = {frozenset(r["Triple"].split(",")) for r in rows}
        assert frozenset(["INBRD_L1", "INBRD_L2", "INBRD_F1"]) in triples, (
            "INBRD trio absent from filter_trios output"
        )

    def test_inverted_trio_file_written_when_applicable(self, named_run, filter_run):
        """
        _inverted.trio must be written by filter_trios when inverted/generational
        rows appear in vcfped output.  With 2000 SNPs the scenario may or may not
        trigger; test is conditional on actual vcfped output content.
        """
        named_rows = _read_tsv(named_run + ".trio")
        has_inverted = any(
            "Inverted" in r.get("Verdict", "") or
            "generational" in r.get("Verdict", "").lower()
            for r in named_rows
        )
        if not has_inverted:
            pytest.skip("No inverted trios in vcfped output for this VCF/seed")
        assert os.path.exists(filter_run + "_inverted.trio"), (
            "_inverted.trio not written despite inverted rows present"
        )


# ---------------------------------------------------------------------------
# Stage 3 — purge_mz_twins
# ---------------------------------------------------------------------------

class TestStage3PurgeMzTwins:

    def test_output_vcf_created(self, purge_run):
        assert os.path.exists(purge_run)

    def test_exactly_one_mz_twin_retained(self, purge_run):
        """Exactly one of MZ_A / MZ_B must remain in the purged VCF."""
        header = next(l for l in open(purge_run) if l.startswith("#CHROM"))
        samples = header.rstrip().split("\t")[9:]
        both   = "MZ_A" in samples and "MZ_B" in samples
        neither = "MZ_A" not in samples and "MZ_B" not in samples
        assert not both,    "Both MZ twins still present after purge"
        assert not neither, "Neither MZ twin retained — one must survive"

    def test_non_mz_samples_present(self, purge_run):
        header = next(l for l in open(purge_run) if l.startswith("#CHROM"))
        samples = set(header.rstrip().split("\t")[9:])
        for s in ("TRI_P1", "TRI_P2", "TRI_C1", "CTRL1", "SELF_P1", "SELF_C1"):
            assert s in samples, f"{s} missing from purged VCF"

    def test_dry_run_writes_nothing(self, vcfped_run, tmp_path):
        out = str(tmp_path / "dry.vcf")
        r = _run(os.path.join(UTILITIES, "purge_mz_twins.py"),
                 [VCF_PATH, vcfped_run + ".pair", "-o", out, "--dry-run"])
        assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
        assert not os.path.exists(out), "--dry-run must not write VCF"


# ---------------------------------------------------------------------------
# Stage 4 — refine_pedigree with synthetic KING
# ---------------------------------------------------------------------------

class TestStage4RefinePedigree:

    def test_output_files_created(self, refine_run):
        for suf in ("_refined.trio", "_refined.fam", "_refinement_report.txt"):
            assert os.path.exists(refine_run + suf), f"Missing: {refine_run + suf}"

    def test_report_contains_status_summary(self, refine_run):
        txt = open(refine_run + "_refinement_report.txt").read()
        assert "Status summary" in txt or "confirmed" in txt.lower()

    def test_refined_fam_parseable(self, refine_run):
        rows = [l.split() for l in open(refine_run + "_refined.fam")
                if l.strip() and not l.startswith("#")]
        assert rows, "Refined FAM is empty"
        for row in rows:
            assert len(row) >= 4, f"FAM row has fewer than 4 columns: {row}"

    def test_inbrd_f1_has_parents(self, refine_run):
        """
        INBRD_F1 appeared as a Regular trio; after refinement it must retain
        at least one parent in the refined FAM.
        """
        fam = {}
        for l in open(refine_run + "_refined.fam"):
            parts = l.split()
            if len(parts) >= 4:
                fam[parts[1]] = (parts[2], parts[3])
        if "INBRD_F1" not in fam:
            pytest.skip("INBRD_F1 not in refined FAM (may have been dropped by KING check)")
        pat, mat = fam["INBRD_F1"]
        assert pat != "0" or mat != "0", "INBRD_F1 has both parents set to 0 in refined FAM"


# ---------------------------------------------------------------------------
# Stage 5 — extend_pedigree (smoke test)
# ---------------------------------------------------------------------------

class TestStage5ExtendPedigree:

    def test_extend_runs_cleanly(self, refine_run, tmp_path):
        out = str(tmp_path / "ext")
        r = _run(os.path.join(UTILITIES, "extend_pedigree.py"),
                 [refine_run + "_refined.fam", KING_KIN0, "-o", out])
        assert r.returncode == 0, f"extend_pedigree failed:\n{r.stderr}\n{r.stdout}"

    def test_extended_fam_and_report_created(self, refine_run, tmp_path):
        out = str(tmp_path / "ext2")
        _run(os.path.join(UTILITIES, "extend_pedigree.py"),
             [refine_run + "_refined.fam", KING_KIN0, "-o", out])
        assert os.path.exists(out + "_extended.fam"), "Missing _extended.fam"
        assert os.path.exists(out + "_extension_report.txt"), "Missing _extension_report.txt"

    def test_extension_report_non_empty(self, refine_run, tmp_path):
        out = str(tmp_path / "ext3")
        r = _run(os.path.join(UTILITIES, "extend_pedigree.py"),
                 [refine_run + "_refined.fam", KING_KIN0, "-o", out])
        assert r.returncode == 0
        assert os.path.getsize(out + "_extension_report.txt") > 0


# ---------------------------------------------------------------------------
# Stage 6 — prepare_plink_beagle (smoke test, no external PLINK2/BEAGLE needed)
# ---------------------------------------------------------------------------

class TestStage6PreparePlinkBeagle:

    def test_fam_and_scripts_created(self, filter_run, tmp_path):
        out = str(tmp_path / "plink_prep")
        r = _run(os.path.join(UTILITIES, "prepare_plink_beagle.py"),
                 [filter_run, VCF_PATH, "-o", out])
        assert r.returncode == 0, f"prepare_plink_beagle failed:\n{r.stderr}"
        assert os.path.exists(out + ".fam"), "Missing .fam"
        assert os.path.exists(out + "_run_plink.sh"), "Missing _run_plink.sh"
        assert os.path.exists(out + "_run_beagle.sh"), "Missing _run_beagle.sh"

    def test_fam_contains_inbrd_trio_members(self, filter_run, tmp_path):
        out = str(tmp_path / "plink_prep2")
        r = _run(os.path.join(UTILITIES, "prepare_plink_beagle.py"),
                 [filter_run, VCF_PATH, "-o", out])
        assert r.returncode == 0
        fam_text = open(out + ".fam").read()
        for name in ("INBRD_L1", "INBRD_L2", "INBRD_F1"):
            assert name in fam_text, f"{name} missing from prepared .fam"
