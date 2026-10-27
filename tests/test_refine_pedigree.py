"""
Tests for utilities/refine_pedigree.py.

Covers the population-relatedness-aware logic that was extensively
redesigned during development:
  - ibs0_threshold_for_error_rate: genotyping-error-aware noise ceiling
  - build_king_percentiles: cohort-relative ranking (population-agnostic,
    needed because fixed KING thresholds are unreliable in closely related
    plant breeding germplasm)
  - compute_trio_genome_coverage: the gold-standard PO confirmation check
    (phased IBD segment coverage of the child's genome by both parents)
  - load_mz_groups: transitive MZ-twin grouping for substitution checking
  - end-to-end refinement on synthetic KING + trio data
"""
import os
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

RP_SCRIPT = os.path.join(UTILITIES_DIR, 'refine_pedigree.py')


@pytest.fixture
def rp():
    return import_module_from_path('refine_pedigree', RP_SCRIPT)


# ---------------------------------------------------------------------------
# ibs0_threshold_for_error_rate
# ---------------------------------------------------------------------------

def test_ibs0_threshold_scales_with_error_rate(rp):
    """Higher genotype error rate -> higher IBS0 noise tolerance (~2x error rate)."""
    low = rp.ibs0_threshold_for_error_rate(0.001)   # high-quality WGS
    high = rp.ibs0_threshold_for_error_rate(0.025)  # low-quality / GBS
    assert low < high
    assert low == pytest.approx(0.003, abs=0.001)


def test_ibs0_threshold_capped(rp):
    """Threshold must not exceed a sane ceiling even for extreme error rates."""
    assert rp.ibs0_threshold_for_error_rate(10.0) <= 0.10


# ---------------------------------------------------------------------------
# parents_are_compatible -- must use the calibrated ibs0_max in use, not the
# hardcoded KING_PO_IBS0_MAX default (same class of bug as
# Assignment.confidence(), found by cross-referencing check_inverted_direction()'s
# already-correct is_po_signal() call for the identical "are these two
# co-parents PO to each other?" question)
# ---------------------------------------------------------------------------

def test_parents_are_compatible_uses_calibrated_ibs0_max(rp):
    """
    At the hardcoded KING_PO_IBS0_MAX=0.05 default, an IBS0=0.01 pair --
    nowhere near true PO-level noise at this cohort's real calibrated scale
    -- is incorrectly rejected as "looks like PO to each other". With the
    calibrated ibs0_max threaded through, it's correctly classified
    compatible.
    """
    king = {('P1', 'P2'): {'kinship': 0.05, 'ibs0': 0.01}}
    assert rp.parents_are_compatible('P1', 'P2', king) is False  # old hardcoded default
    assert rp.parents_are_compatible('P1', 'P2', king, ibs0_max=0.0008) is True  # calibrated


def test_parents_are_compatible_still_rejects_true_po_level_ibs0(rp):
    """A pair whose mutual IBS0 IS within the calibrated PO noise floor must
    still be rejected as incompatible -- the fix must not stop catching
    genuine grandparent/parent conflicts."""
    king = {('P1', 'P2'): {'kinship': 0.30, 'ibs0': 0.0002}}
    assert rp.parents_are_compatible('P1', 'P2', king, ibs0_max=0.0008) is False


def test_parents_are_compatible_missing_pair_defaults_compatible(rp):
    assert rp.parents_are_compatible('X', 'Y', {}, ibs0_max=0.0008) is True


# ---------------------------------------------------------------------------
# load_king + KING helper functions
# ---------------------------------------------------------------------------

@pytest.fixture
def king_kin0_file(tmp_path):
    f = tmp_path / "king.kin0"
    f.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tFATHER\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD\tF1\tMOTHER\t1000\t0.3\t0.005\t0.24\n"
        "F1\tFATHER\tF1\tUNCLE\t1000\t0.3\t0.06\t0.13\n"
    )
    return str(f)


def test_load_king_parses_kin0(rp, king_kin0_file):
    king, all_ids = rp.load_king(king_kin0_file)
    assert ('CHILD', 'FATHER') in king
    assert king[('CHILD', 'FATHER')]['kinship'] == 0.26
    assert 'CHILD' in all_ids
    assert 'FATHER' in all_ids


def test_get_king_order_independent(rp, king_kin0_file):
    king, _ = rp.load_king(king_kin0_file)
    a = rp.get_king(king, 'CHILD', 'FATHER')
    b = rp.get_king(king, 'FATHER', 'CHILD')
    assert a == b


def test_load_king_missing_file_exits(rp, tmp_path):
    with pytest.raises(SystemExit):
        rp.load_king(str(tmp_path / "nonexistent.kin0"))


def test_load_king_warns_on_aggressive_filter(rp, tmp_path, capsys):
    """If the .kin0 file's minimum kinship is much higher than what's
    needed, warn (e.g. user passed _king_1st.kin0 instead of _king_ext.kin0)."""
    f = tmp_path / "filtered.kin0"
    f.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tA\tF1\tB\t1000\t0.3\t0.005\t0.26\n"  # only 1st-degree present
    )
    rp.load_king(str(f), required_min_kinship=0.0884)
    captured = capsys.readouterr()
    assert 'WARNING' in captured.out


# ---------------------------------------------------------------------------
# build_king_percentiles
# ---------------------------------------------------------------------------

def test_king_percentiles_relative_ranking(rp, king_kin0_file):
    """The highest-kinship pair should rank at/near the 100th percentile,
    regardless of the absolute kinship values (population-agnostic)."""
    king, _ = rp.load_king(king_kin0_file)
    pct = rp.build_king_percentiles(king)
    top_pair_pct = pct[('CHILD', 'FATHER')]  # kinship=0.26, highest in this set
    lowest_pair_pct = pct[('FATHER', 'UNCLE')]  # kinship=0.13, lowest
    assert top_pair_pct > lowest_pair_pct
    assert top_pair_pct == 100  # highest of 3 -> 100th percentile


def test_king_percentiles_empty_king(rp):
    assert rp.build_king_percentiles({}) == {}


# ---------------------------------------------------------------------------
# compute_trio_genome_coverage (gold-standard PO confirmation)
# ---------------------------------------------------------------------------

def test_genome_coverage_full_coverage_true_trio(rp):
    """A true parent-offspring trio: child's genome fully covered by the
    union of both parents' IBD segments -> coverage near 100%."""
    ibd_index = {
        ('CHILD', 'FATHER'): [('1', 0, 1000000), ('1', 1000000, 2000000)],
        ('CHILD', 'MOTHER'): [('1', 0, 1000000)],  # overlapping segment from MOTHER
    }
    # Need to use sorted-tuple keys matching load_ibd_index's convention
    ibd_index = {tuple(sorted(k)): v for k, v in ibd_index.items()}
    result = rp.compute_trio_genome_coverage('CHILD', 'FATHER', 'MOTHER', ibd_index)
    assert result['coverage_frac'] is not None
    assert result['coverage_frac'] > 0.5


def test_genome_coverage_no_segments_returns_none(rp):
    result = rp.compute_trio_genome_coverage('CHILD', 'FATHER', 'MOTHER', {})
    assert result['coverage_frac'] is None
    assert 'No IBD segments' in result['interpretation']


def test_genome_coverage_low_coverage_flagged(rp):
    """A false parent pair with minimal genome coverage should be flagged
    as low confidence in the interpretation text."""
    ibd_index = {}
    key_child_father = tuple(sorted(('CHILD', 'FATHER')))
    key_child_mother = tuple(sorted(('CHILD', 'MOTHER')))
    key_other = tuple(sorted(('CHILD', 'STRANGER')))
    ibd_index[key_child_father] = [('1', 0, 100000)]    # tiny segment
    ibd_index[key_other] = [('1', 0, 50000000)]          # most of child's genome is elsewhere
    result = rp.compute_trio_genome_coverage('CHILD', 'FATHER', 'MOTHER', ibd_index)
    assert result['coverage_frac'] is not None
    assert result['coverage_frac'] < 0.3
    assert 'LOW' in result['interpretation']


# ---------------------------------------------------------------------------
# Assignment.confidence() -- must scale against the calibrated ibs0_max in
# use, not the hardcoded KING_PO_IBS0_MAX default
# ---------------------------------------------------------------------------

def test_confidence_uses_calibrated_ibs0_max_not_hardcoded_default(rp):
    """
    Regression guard for a real production bug: at a very low calibrated
    error rate (e.g. 0.0002 -> ibs0_max~0.0008), scoring ibs0_score against
    the hardcoded KING_PO_IBS0_MAX=0.05 default flattens it to near-1.0 for
    every near-zero-IBS0 candidate, silently neutralizing "IBS0: most
    reliable, weight highest" and leaving kinship percentile as the real
    discriminator. Passing the calibrated ibs0_max restores real spread.
    """
    # Two candidates with the same small kinship/T1/percentile but a
    # meaningfully different IBS0 (0.0002 vs 0.0007) -- both tiny in
    # absolute terms, but a 3.5x difference relative to a 0.0008 ceiling.
    weaker = rp.Assignment('P1', 'P2', 'CHILD', 'king_new', perc=50, t1=90.0,
                           ibs_p1=0.0007, ibs_p2=0.0007, pct_p1=80, pct_p2=80)
    stronger = rp.Assignment('P1b', 'P2', 'CHILD', 'king_new', perc=50, t1=90.0,
                             ibs_p1=0.0002, ibs_p2=0.0002, pct_p1=80, pct_p2=80)

    calibrated_ibs0_max = 0.0008
    conf_weaker_calibrated = weaker.confidence(calibrated_ibs0_max)
    conf_stronger_calibrated = stronger.confidence(calibrated_ibs0_max)
    assert conf_stronger_calibrated > conf_weaker_calibrated

    # At the hardcoded default (0.05), the same pair's IBS0 difference is
    # negligible relative to the scale -- scores nearly collapse together.
    conf_weaker_default = weaker.confidence()
    conf_stronger_default = stronger.confidence()
    assert (conf_stronger_default - conf_weaker_default) < (
        conf_stronger_calibrated - conf_weaker_calibrated)


def test_confidence_default_argument_matches_hardcoded_constant(rp):
    """confidence() with no argument must still work (backward compatible)
    and match the module's KING_PO_IBS0_MAX constant explicitly."""
    a = rp.Assignment('P1', 'P2', 'CHILD', 'king_new', t1=90.0,
                      ibs_p1=0.001, ibs_p2=0.001, pct_p1=50, pct_p2=50)
    assert a.confidence() == a.confidence(rp.KING_PO_IBS0_MAX)


# ---------------------------------------------------------------------------
# load_mz_groups (transitive closure for twin substitution checking)
# ---------------------------------------------------------------------------

def test_load_mz_groups_transitive(rp, tmp_path):
    pair_file = tmp_path / "step1.pair"
    pair_file.write_text(
        "Pair\tPerc\tAutos\tNeitherAA\tIBS2\tMZscore\tEitherBB\tIBS>0\tPOscore\tVerdict\n"
        "1,2\t10\t100\t50\t50\t100.0\t10\t10\t100.0\tMZ twins\n"
        "2,3\t10\t100\t50\t50\t100.0\t10\t10\t100.0\tMZ twins\n"
    )
    # load_mz_groups expects a prefix (it appends .pair itself)
    prefix = str(tmp_path / "step1")
    mz_of = rp.load_mz_groups(prefix)
    # All numeric "samples" 1,2,3 here are placeholders -- check transitivity directly
    assert mz_of['1'] == mz_of['2'] == mz_of['3']
    assert len(mz_of['1']) == 3


def test_load_mz_groups_no_file_returns_empty(rp, tmp_path):
    mz_of = rp.load_mz_groups(str(tmp_path / "nonexistent"))
    assert mz_of == {}


# ---------------------------------------------------------------------------
# load_mz_groups_from_confirmed_duplicates (stricter, hap-IBD-coverage-
# validated ground truth, as an alternative to the raw --step1-pair verdict)
# ---------------------------------------------------------------------------

def test_load_mz_groups_from_confirmed_duplicates_transitive(rp, tmp_path):
    f = tmp_path / "confirmed.tsv"
    f.write_text("ID1\tID2\nA\tB\nB\tC\n")
    mz_of = rp.load_mz_groups_from_confirmed_duplicates(str(f))
    assert mz_of['A'] == mz_of['B'] == mz_of['C']
    assert len(mz_of['A']) == 3


def test_load_mz_groups_from_confirmed_duplicates_no_file_returns_empty(rp, tmp_path):
    assert rp.load_mz_groups_from_confirmed_duplicates(None) == {}
    assert rp.load_mz_groups_from_confirmed_duplicates(str(tmp_path / "nope.tsv")) == {}


def test_load_mz_groups_from_confirmed_duplicates_missing_columns_exits(rp, tmp_path):
    f = tmp_path / "bad.tsv"
    f.write_text("FOO\tBAR\nA\tB\n")
    with pytest.raises(SystemExit):
        rp.load_mz_groups_from_confirmed_duplicates(str(f))


def test_step1_pair_and_confirmed_duplicates_mutually_exclusive(tmp_path):
    trio_path = tmp_path / "test.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER,MOTHER,CHILD\tCHILD\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    king_path = tmp_path / "king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tFATHER\t1000\t0.3\t0.005\t0.26\n"
    )
    pair_path = tmp_path / "step1.pair"
    pair_path.write_text("Pair\tVerdict\n")
    confirmed_path = tmp_path / "confirmed.tsv"
    confirmed_path.write_text("ID1\tID2\n")

    result = run_script(RP_SCRIPT, [
        str(tmp_path / "test"), str(tmp_path / "king"),
        '--step1-pair', str(tmp_path / "step1"),
        '--confirmed-duplicates', str(confirmed_path),
    ])
    assert result.returncode != 0
    assert 'at most one' in (result.stdout + result.stderr)


def test_end_to_end_uses_confirmed_duplicates_for_mz_groups(synthetic_refine_inputs, tmp_path):
    """
    --confirmed-duplicates should be usable in place of --step1-pair and
    reported in the console output, confirming the stricter ground truth
    actually drives MZ-group loading (not silently ignored).
    """
    trio_prefix, king_prefix = synthetic_refine_inputs
    confirmed_path = tmp_path / "confirmed.tsv"
    confirmed_path.write_text("ID1\tID2\nX\tY\n")
    out_prefix = str(tmp_path / "refined_out")
    result = run_script(RP_SCRIPT, [
        trio_prefix, king_prefix, '-o', out_prefix,
        '--confirmed-duplicates', str(confirmed_path),
    ])
    assert result.returncode == 0, result.stderr
    assert 'hap-IBD-coverage validated' in result.stdout
    assert '2 samples in 1 MZ groups' in result.stdout


def test_end_to_end_warns_on_implausibly_large_raw_mz_group(tmp_path):
    """
    Regression guard for the real production finding: a single vcfped raw
    "MZ twins" group with an implausible number of members (inbreeding-
    induced allele-sharing inflation, not literal clonal duplication) should
    print a WARNING pointing at --confirmed-duplicates as the fix -- but
    only when using the raw --step1-pair source, never for
    --confirmed-duplicates (which is already the trusted, stricter source).
    """
    trio_path = tmp_path / "test.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER,MOTHER,CHILD\tCHILD\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    king_path = tmp_path / "king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tFATHER\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD\tF1\tMOTHER\t1000\t0.3\t0.005\t0.24\n"
    )
    pair_path = tmp_path / "step1.pair"
    header = "Pair\tVerdict\n"
    # 12-member clique -- chain of pairs all called "MZ twins", well past a
    # plausible clonal-duplicate cluster size.
    rows = "".join(f"S{i},S{i+1}\tMZ twins\n" for i in range(11))
    pair_path.write_text(header + rows)

    out_prefix = str(tmp_path / "refined_out")
    result = run_script(RP_SCRIPT, [
        str(tmp_path / "test"), str(tmp_path / "king"),
        '--step1-pair', str(tmp_path / "step1"), '-o', out_prefix,
    ])
    assert result.returncode == 0, result.stderr
    assert 'WARNING' in result.stdout
    assert '--confirmed-duplicates' in result.stdout


# ---------------------------------------------------------------------------
# End-to-end: full refinement run on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_refine_inputs(tmp_path):
    trio_path = tmp_path / "test.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER,MOTHER,CHILD\tCHILD\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    king_path = tmp_path / "king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tFATHER\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD\tF1\tMOTHER\t1000\t0.3\t0.005\t0.24\n"
    )
    return str(tmp_path / "test"), str(tmp_path / "king")


def test_end_to_end_confirms_valid_trio(synthetic_refine_inputs, tmp_path):
    trio_prefix, king_prefix = synthetic_refine_inputs
    out_prefix = str(tmp_path / "refined_out")
    result = run_script(RP_SCRIPT, [trio_prefix, king_prefix, '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{out_prefix}_refined.trio")
    assert os.path.exists(f"{out_prefix}_refined.fam")
    assert os.path.exists(f"{out_prefix}_refinement_report.txt")
    report = open(f"{out_prefix}_refinement_report.txt").read()
    assert 'confirmed' in report.lower()


def test_end_to_end_flags_unconfirmed_trio(tmp_path):
    """A trio where KING shows IBS0 too high (not PO) must be flagged,
    not silently confirmed."""
    trio_path = tmp_path / "bad.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER,MOTHER,CHILD\tCHILD\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    king_path = tmp_path / "bad_king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD\tF1\tFATHER\t1000\t0.3\t0.20\t0.26\n"  # IBS0 way too high for PO
        "F1\tCHILD\tF1\tMOTHER\t1000\t0.3\t0.005\t0.24\n"
    )
    out_prefix = str(tmp_path / "refined_out")
    result = run_script(RP_SCRIPT, [str(tmp_path / "bad"), str(tmp_path / "bad_king"),
                                    '-o', out_prefix])
    assert result.returncode == 0, result.stderr
    report = open(f"{out_prefix}_refinement_report.txt").read()
    assert 'NON-PO' in report or 'REJECTION' in report


def test_end_to_end_fam_includes_reclassified_inverted(tmp_path):
    """
    Regression test: _refined.fam must include children whose ONLY
    confirmation came from inverted-trio reclassification
    (cls == LIKELY_REGULAR_TRIO), not just from the `results` (originally
    Regular-trio) pool. Previously the FAM-writing loop only iterated
    `results`, so a --include-inverted run could confirm orders of magnitude
    more trios in _refined.trio than ever reached _refined.fam -- the file
    extend_pedigree.py actually consumes.

    Also verifies:
      - When multiple inverted rows compete for the same child, the
        highest-Test1 candidate wins (matching filter_trios.py's own
        best-per-child tie-break).
      - A child already assigned via `results` (regular-trio evidence) is
        never overwritten by a lower-confidence reclassified-inverted row
        for the same child.
    """
    trio_path = tmp_path / "mix.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER1,MOTHER1,CHILD1\tCHILD1\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    inv_path = tmp_path / "mix_inverted.trio"
    inv_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "FATHER2A,MOTHER2A,CHILD2\tCHILD2\t10\t1905\t100\t80\t80.0\t50\t10\t20.0\tInverted/generational trio\n"
        "FATHER2B,MOTHER2B,CHILD2\tCHILD2\t10\t1905\t120\t115\t95.8\t60\t5\t8.3\tInverted/generational trio\n"
        "FATHER1,MOTHER1,CHILD1\tCHILD1\t10\t1905\t100\t60\t60.0\t40\t2\t5.0\tInverted/generational trio\n"
    )
    king_path = tmp_path / "mix_king.kin0"
    king_path.write_text(
        "#FID1\tID1\tFID2\tID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
        "F1\tCHILD1\tF1\tFATHER1\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD1\tF1\tMOTHER1\t1000\t0.3\t0.005\t0.24\n"
        "F1\tCHILD2\tF1\tFATHER2A\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD2\tF1\tMOTHER2A\t1000\t0.3\t0.005\t0.24\n"
        "F1\tCHILD2\tF1\tFATHER2B\t1000\t0.3\t0.005\t0.26\n"
        "F1\tCHILD2\tF1\tMOTHER2B\t1000\t0.3\t0.005\t0.24\n"
    )
    out_prefix = str(tmp_path / "out")
    result = run_script(RP_SCRIPT, [str(tmp_path / "mix"), str(tmp_path / "mix_king"),
                                    '--include-inverted', '-o', out_prefix])
    assert result.returncode == 0, result.stderr

    fam = {}
    for line in open(f"{out_prefix}_refined.fam"):
        parts = line.split()
        fam[parts[1]] = (parts[2], parts[3])

    # CHILD1: results-based assignment must win, not be overwritten by the
    # competing (lower-confidence) inverted-trio row for the same child.
    assert fam['CHILD1'] == ('FATHER1', 'MOTHER1')

    # CHILD2: only ever appeared via inverted candidates -- must still reach
    # the FAM (the bug this test guards against), and the higher-Test1
    # candidate (T1=95.8) must win over the lower one (T1=80.0).
    assert 'CHILD2' in fam, "Reclassified-inverted-only child missing from refined FAM"
    assert fam['CHILD2'] == ('FATHER2B', 'MOTHER2B')
