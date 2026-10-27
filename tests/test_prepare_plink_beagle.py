"""
Tests for utilities/prepare_plink_beagle.py.

This script accumulated several real bugs during development that are
worth permanent regression coverage:
  - chromosome rename logic (ch01->1 etc.)
  - --set-all-var-ids missing from the VCF->PLINK conversion step (caused
    "--indep-pairwise requires unique variant IDs" when the VCF used '.'
    for all variant IDs)
  - sex derivation for PLINK2 --mendel (SEX=1 for PAT, SEX=2 for MAT),
    including the hermaphrodite/plant-crossing conflict case where the same
    individual is used as both father and mother in different trios
  - the R-script heredoc quoting bug (quoted 'RSCRIPT' delimiter prevented
    ${OUT} from being bash-expanded, writing the literal string into the .R
    file instead of the real filename)
  - PLINK2/BEAGLE/hap-IBD command configurability (--plink2/--beagle/--hapibd)
  - the synthetic genetic map generation required by hap-IBD (mandatory
    map= argument, unlike BEAGLE phasing where it's optional)
"""
import os
import re
import pytest

from conftest import import_module_from_path, run_script, UTILITIES_DIR

PPB_SCRIPT = os.path.join(UTILITIES_DIR, 'prepare_plink_beagle.py')


@pytest.fixture
def ppb():
    return import_module_from_path('prepare_plink_beagle', PPB_SCRIPT)


# ---------------------------------------------------------------------------
# build_chr_rename
# ---------------------------------------------------------------------------

def test_chr_rename_ch_prefix(ppb):
    chroms = ['ch01', 'ch02', 'ch10', 'ch25']
    rename = ppb.build_chr_rename(chroms, 'ch')
    rename_dict = dict(rename)
    assert rename_dict['ch01'] == '1'
    assert rename_dict['ch02'] == '2'
    assert rename_dict['ch10'] == '10'
    assert rename_dict['ch25'] == '25'


def test_chr_rename_already_numeric_skipped(ppb):
    """Chromosomes already numeric ('1', '2', ...) need no rename entry."""
    rename = ppb.build_chr_rename(['1', '2', '3'], 'ch')
    assert rename == []


def test_chr_rename_mixed(ppb):
    rename = ppb.build_chr_rename(['ch01', '5', 'ch03'], 'ch')
    rename_dict = dict(rename)
    assert '5' not in rename_dict  # already numeric, no rename needed
    assert rename_dict['ch01'] == '1'
    assert rename_dict['ch03'] == '3'


def test_chr_rename_non_numeric_unprefixed(ppb):
    """Names that don't match the prefix pattern get assigned sequential IDs."""
    rename = ppb.build_chr_rename(['scaffold_A', 'scaffold_B'], 'ch')
    rename_dict = dict(rename)
    assert set(rename_dict.values()) == {'1', '2'}


# ---------------------------------------------------------------------------
# build_pedigree: family detection
# ---------------------------------------------------------------------------

def test_build_pedigree_family_grouping(ppb):
    trios = [
        {'p1': 'A', 'p2': 'B', 'child': 'C', 'perc': 50, 't1': 100.0},
    ]
    samples = ['A', 'B', 'C', 'D']  # D is unrelated singleton
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, {}, samples)
    assert fam_of['A'] == fam_of['B'] == fam_of['C']
    assert fam_of['D'] != fam_of['A']


def test_build_pedigree_uses_gender_when_known(ppb):
    trios = [{'p1': 'DAD', 'p2': 'MOM', 'child': 'KID', 'perc': 50, 't1': 100.0}]
    gender = {'DAD': 'Male', 'MOM': 'Female'}
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, gender, ['DAD', 'MOM', 'KID'])
    assert pat_of['KID'] == 'DAD'
    assert mat_of['KID'] == 'MOM'


def test_build_pedigree_swaps_when_sex_reversed(ppb):
    """If p1 is recorded female and p2 male, father/mother must be assigned
    correctly regardless of trio column order."""
    trios = [{'p1': 'MOM', 'p2': 'DAD', 'child': 'KID', 'perc': 50, 't1': 100.0}]
    gender = {'DAD': 'Male', 'MOM': 'Female'}
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, gender, ['DAD', 'MOM', 'KID'])
    assert pat_of['KID'] == 'DAD'
    assert mat_of['KID'] == 'MOM'


# ---------------------------------------------------------------------------
# build_pedigree + write_fam: sex derivation for PLINK2 --mendel
# (SEX must be 1 for any PAT, 2 for any MAT, with no contradictions)
# ---------------------------------------------------------------------------

def test_sex_derived_from_parent_role_no_gender_file(ppb, tmp_path):
    """With no biological sex known, PAT individuals get SEX=1, MAT get SEX=2
    — required for PLINK2 --mendel to accept the .fam without error."""
    trios = [{'p1': 'DAD', 'p2': 'MOM', 'child': 'KID', 'perc': 50, 't1': 100.0}]
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, {}, ['DAD', 'MOM', 'KID'])
    assert sex['DAD'] == 'Male'
    assert sex['MOM'] == 'Female'

    fam_path = str(tmp_path / "test.fam")
    ppb.write_fam(fam_path, ordered, fam_of, pat_of, mat_of, sex)
    lines = open(fam_path).read().splitlines()
    fam_dict = {l.split('\t')[1]: l.split('\t') for l in lines}
    assert fam_dict['DAD'][4] == '1'   # SEX=1 (male)
    assert fam_dict['MOM'][4] == '2'   # SEX=2 (female)


def test_sex_role_conflict_hermaphrodite_resolved_by_majority(ppb, tmp_path):
    """
    Regression guard: in plant breeding, the same accession can be the
    pollen donor (father) in one cross and seed parent (mother) in another.
    Without resolution, PLINK2 errors with "Mother/Father has unspecified or
    inconsistent sex." The fix: majority-vote sex, and clear the MINORITY
    parent link (set to '0') for that specific child, rather than leaving a
    sex-inconsistent assignment in the .fam.
    """
    trios = [
        # X used as FATHER twice, MOTHER once -> majority = Male
        {'p1': 'X', 'p2': 'P1', 'child': 'C1', 'perc': 50, 't1': 100.0},
        {'p1': 'X', 'p2': 'P2', 'child': 'C2', 'perc': 50, 't1': 100.0},
        {'p1': 'P3', 'p2': 'X', 'child': 'C3', 'perc': 50, 't1': 100.0},  # X as mother (minority)
    ]
    samples = ['X', 'P1', 'P2', 'P3', 'C1', 'C2', 'C3']
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, {}, samples)
    assert sex['X'] == 'Male'  # majority role

    fam_path = str(tmp_path / "test.fam")
    ppb.write_fam(fam_path, ordered, fam_of, pat_of, mat_of, sex)
    lines = open(fam_path).read().splitlines()
    rows = {l.split('\t')[1]: l.split('\t') for l in lines}

    # X's own row must have SEX=1 (male, majority)
    assert rows['X'][4] == '1'

    # C3's MAT was X, but X is SEX=1 (male) -> that link must be CLEARED
    # to '0', not left as a sex-inconsistent assignment.
    assert rows['C3'][3] == '0', \
        "Minority-role parent link must be cleared to avoid PLINK2 sex inconsistency"

    # C1/C2's PAT=X assignments (majority role) must remain intact
    assert rows['C1'][2] == 'X'
    assert rows['C2'][2] == 'X'


def test_no_sex_inconsistency_in_any_written_fam_row(ppb, tmp_path):
    """
    Exhaustive check: for ANY individual appearing as PAT somewhere, their
    own SEX must be '1'; for MAT, SEX must be '2'. This is PLINK2's hard
    requirement for --mendel to run without error.
    """
    trios = [
        {'p1': 'X', 'p2': 'P1', 'child': 'C1', 'perc': 50, 't1': 100.0},
        {'p1': 'X', 'p2': 'P2', 'child': 'C2', 'perc': 50, 't1': 100.0},
        {'p1': 'P3', 'p2': 'X', 'child': 'C3', 'perc': 50, 't1': 100.0},
        {'p1': 'Y', 'p2': 'Z', 'child': 'C4', 'perc': 50, 't1': 100.0},
    ]
    samples = ['X', 'Y', 'Z', 'P1', 'P2', 'P3', 'C1', 'C2', 'C3', 'C4']
    fam_of, pat_of, mat_of, ordered, warnings, sex = ppb.build_pedigree(
        trios, {}, samples)
    fam_path = str(tmp_path / "test.fam")
    ppb.write_fam(fam_path, ordered, fam_of, pat_of, mat_of, sex)

    rows = {}
    for line in open(fam_path):
        parts = line.rstrip('\n').split('\t')
        rows[parts[1]] = parts  # FID IID PAT MAT SEX PHENO

    for iid, row in rows.items():
        pat, mat, own_sex = row[2], row[3], row[4]
        if pat != '0' and pat in rows:
            assert rows[pat][4] == '1', f"{pat} is PAT of {iid} but SEX={rows[pat][4]}"
        if mat != '0' and mat in rows:
            assert rows[mat][4] == '2', f"{mat} is MAT of {iid} but SEX={rows[mat][4]}"


# ---------------------------------------------------------------------------
# scan_vcf_header
# ---------------------------------------------------------------------------

def test_scan_vcf_header_standard_vcf(ppb, tmp_path):
    f = tmp_path / "test.vcf"
    f.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=ch01,length=1000>\n"
        "##contig=<ID=ch02,length=2000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSampleA\tSampleB\n"
        "ch01\t100\t.\tA\tT\t.\t.\t.\tGT\t0/0\t0/1\n"
    )
    chroms, samples = ppb.scan_vcf_header(str(f))
    assert chroms == ['ch01', 'ch02']
    assert samples == ['SampleA', 'SampleB']


def test_scan_vcf_header_no_contig_lines_falls_back_to_data(ppb, tmp_path):
    f = tmp_path / "test.vcf"
    f.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSampleA\n"
        "ch01\t100\t.\tA\tT\t.\t.\t.\tGT\t0/0\n"
        "ch02\t200\t.\tA\tT\t.\t.\t.\tGT\t0/1\n"
    )
    chroms, samples = ppb.scan_vcf_header(str(f))
    assert set(chroms) == {'ch01', 'ch02'}


def test_scan_vcf_header_gzip(ppb, tmp_path):
    import gzip
    f = tmp_path / "test.vcf.gz"
    with gzip.open(f, 'wt') as gz:
        gz.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\n"
            "1\t100\t.\tA\tT\t.\t.\t.\tGT\t0/0\n"
        )
    chroms, samples = ppb.scan_vcf_header(str(f))
    assert samples == ['A']


# ---------------------------------------------------------------------------
# read_trio_file
# ---------------------------------------------------------------------------

def test_read_trio_file_filters_to_regular_trio_only(ppb, tmp_path):
    f = tmp_path / "test.trio"
    f.write_text(
        "Triple\tPivot\tPerc\tTest1\tVerdict\n"
        "A,B,C\tC\t50\t100.0\tRegular trio\n"
        "D,E,F\tF\t10\t90.0\tInverted/generational trio\n"
    )
    trios = ppb.read_trio_file(str(f))
    assert len(trios) == 1
    assert trios[0]['child'] == 'C'


# ---------------------------------------------------------------------------
# fam_to_trios / read_fam_sex: --from-fam input mode
# ---------------------------------------------------------------------------

def test_fam_to_trios_only_full_trios_included(ppb, tmp_path):
    f = tmp_path / "merged.fam"
    f.write_text(
        "0\tCHILD\tDAD\tMOM\t0\t-9\n"     # both parents known -> trio
        "0\tFOUNDER\t0\t0\t1\t-9\n"        # no parents -> not a trio
        "0\tPARTIAL\tDAD\t0\t0\t-9\n"      # only one parent -> not a trio
    )
    trios = ppb.fam_to_trios(str(f))
    assert len(trios) == 1
    assert trios[0] == {'p1': 'DAD', 'p2': 'MOM', 'child': 'CHILD', 'perc': 100, 't1': 100.0}


def test_read_fam_sex_maps_codes(ppb, tmp_path):
    f = tmp_path / "merged.fam"
    f.write_text(
        "0\tDAD\t0\t0\t1\t-9\n"
        "0\tMOM\t0\t0\t2\t-9\n"
        "0\tCHILD\tDAD\tMOM\t0\t-9\n"
    )
    sex = ppb.read_fam_sex(str(f))
    assert sex == {'DAD': 'Male', 'MOM': 'Female', 'CHILD': None}


# ---------------------------------------------------------------------------
# write_plink_script: regression checks for fixes made during development
# ---------------------------------------------------------------------------

def test_plink_script_includes_set_all_var_ids(ppb, tmp_path):
    """
    Regression guard: --indep-pairwise requires unique variant IDs, but VCF
    files commonly use '.' for every variant ID. Without --set-all-var-ids,
    PLINK2 errors out partway through the pipeline (after Mendel/KING have
    already run, wasting that work).
    """
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='./plink2')
    content = open(out_path).read()
    assert '--set-all-var-ids' in content


def test_plink_script_uses_configurable_plink2_executable(ppb, tmp_path):
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='./plink2')
    content = open(out_path).read()
    assert 'PLINK2="./plink2"' in content
    # All plink2 invocations should use the variable, not a hardcoded "plink2"
    assert re.search(r'^plink2 \\', content, re.MULTILINE) is None


def test_plink_script_two_king_passes_different_thresholds(ppb, tmp_path):
    """
    Regression guard: PLINK2 --make-king-table silently drops pairs below
    its default filter (~0.044). A single pass at the 1st-degree threshold
    (0.177) would make extend_pedigree.py's grandparent search (which needs
    2nd-degree pairs down to ~0.0884) silently see zero candidates.
    """
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='plink2')
    content = open(out_path).read()
    assert '--king-table-filter 0.177' in content
    assert '--king-table-filter 0.044' in content
    assert 'king_1st' in content
    assert 'king_ext' in content


def test_plink_script_quality_filters_before_king(ppb, tmp_path):
    """Genotyping-error-aware filtering must happen before KING/Mendel."""
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='plink2')
    content = open(out_path).read()
    assert '--geno' in content
    assert '--maf' in content
    assert '--hwe' in content


def test_plink_script_includes_het_and_missing(ppb, tmp_path):
    """
    Regression guard: without --het/--missing there's no way to rule out poor
    genotyping quality (elevated missingness or heterozygosity) as the cause
    of a "repeat offender" parent flagged by analyze_mendel_errors.py, before
    concluding it's a pedigree assignment problem.
    """
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='plink2')
    content = open(out_path).read()
    assert '--het \\' in content
    assert '--missing \\' in content
    # --het must run on the LD-pruned set (correlated SNPs bias the estimate,
    # same reason KING itself is computed on the pruned set).
    het_cmd_idx = content.index('--het \\')
    assert '_pruned' in content[max(0, het_cmd_idx - 100):het_cmd_idx]


def test_plink_script_valid_bash_syntax(ppb, tmp_path):
    import subprocess
    out_path = str(tmp_path / "run_plink.sh")
    ppb.write_plink_script(out_path, 'input.vcf.gz', 'test.fam', None,
                           'prefix', [('ch01', '1')], 25, plink2_exe='plink2')
    result = subprocess.run(['bash', '-n', out_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# write_beagle_script: regression checks
# ---------------------------------------------------------------------------

def test_beagle_script_uses_configurable_executables(ppb, tmp_path):
    out_path = str(tmp_path / "run_beagle.sh")
    ppb.write_beagle_script(out_path, 'input.vcf.gz', 'test.fam', None,
                            'prefix', [], 25,
                            beagle_exe='beagle', hapibd_exe='hap-ibd')
    content = open(out_path).read()
    assert 'BEAGLE="beagle"' in content
    assert 'HAPIBD="hap-ibd"' in content


def test_beagle_script_generates_synthetic_map_for_hapibd(ppb, tmp_path):
    """
    Regression guard: hap-IBD's map= argument is MANDATORY (unlike BEAGLE
    phasing, where it's optional). Without it: 'Exception ... missing map
    argument'. The script must generate a synthetic map and pass it.
    """
    out_path = str(tmp_path / "run_beagle.sh")
    ppb.write_beagle_script(out_path, 'input.vcf.gz', 'test.fam', None,
                            'prefix', [], 25, cm_per_mb=1.0)
    content = open(out_path).read()
    assert 'synthetic.map' in content
    assert re.search(r'map="\$\{MAP\}"', content), \
        "hap-IBD calls must pass map=, which is mandatory"


def test_beagle_script_two_phasing_passes(ppb, tmp_path):
    """Statistical phasing (screens all individuals) + pedigree-aware
    phasing (high accuracy for known trios) must both be present."""
    out_path = str(tmp_path / "run_beagle.sh")
    ppb.write_beagle_script(out_path, 'input.vcf.gz', 'test.fam', None,
                            'prefix', [], 25)
    content = open(out_path).read()
    assert 'stat_phased' in content
    assert re.search(r'ped="\$\{FAM\}"', content)  # pedigree-aware pass present


def test_beagle_script_r_heredoc_variables_expand(ppb, tmp_path):
    """
    Regression guard: `cat << 'RSCRIPT'` (quoted delimiter) prevents bash
    variable expansion inside the heredoc, so ${OUT} would be written
    LITERALLY into the generated .R file instead of the real path. The fix
    uses an unquoted `cat << RSCRIPT` delimiter.
    """
    out_path = str(tmp_path / "run_beagle.sh")
    ppb.write_beagle_script(out_path, 'input.vcf.gz', 'test.fam', None,
                            'prefix', [], 25)
    content = open(out_path).read()
    assert "cat << 'RSCRIPT'" not in content, \
        "Quoted heredoc delimiter prevents ${OUT} expansion"
    assert "cat << RSCRIPT" in content


def test_beagle_script_valid_bash_syntax(ppb, tmp_path):
    import subprocess
    out_path = str(tmp_path / "run_beagle.sh")
    ppb.write_beagle_script(out_path, 'input.vcf.gz', 'test.fam', None,
                            'prefix', [], 25)
    result = subprocess.run(['bash', '-n', out_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_beagle_script_heredoc_actually_expands_when_run(ppb, tmp_path):
    """End-to-end proof: simulate the heredoc in bash and verify ${OUT}
    resolves to the real prefix, not the literal string '${OUT}'."""
    import subprocess
    script = '''
OUT="myprefix"
cat << RSCRIPT > /dev/stdout
ibd <- fread("${OUT}_ibd_summary.ibd.gz")
RSCRIPT
'''
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert 'myprefix_ibd_summary.ibd.gz' in result.stdout
    assert '${OUT}' not in result.stdout


# ---------------------------------------------------------------------------
# End-to-end: full CLI run on the bundled test VCF
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_vcf_and_trio(tmp_path):
    """
    The bundled trioHG002_22X.vcf is a non-standard annotated VCF (no
    #CHROM line in the format prepare_plink_beagle.py expects) — it's only
    usable by vcfped itself. This fixture builds a minimal, properly
    formatted VCF + matching .trio file so prepare_plink_beagle.py can be
    exercised end-to-end.
    """
    vcf_path = tmp_path / "standard.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=ch01,length=50000000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG002\tHG003\tHG004\n"
        "ch01\t100\t.\tA\tT\t100\tPASS\t.\tGT\t0/1\t0/0\t1/1\n"
        "ch01\t200\t.\tG\tC\t100\tPASS\t.\tGT\t0/1\t0/0\t1/1\n"
    )
    trio_path = tmp_path / "test.trio"
    trio_path.write_text(
        "Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n"
        "HG003,HG004,HG002\tHG002\t10\t1905\t136\t136\t100.0\t269\t267\t99.3\tRegular trio\n"
    )
    return str(vcf_path), str(tmp_path / "test")


def test_end_to_end_cli_run(standard_vcf_and_trio, tmp_path):
    vcf_path, trio_prefix = standard_vcf_and_trio
    out_prefix = str(tmp_path / "pb_out")
    result = run_script(PPB_SCRIPT, [trio_prefix, vcf_path, '-o', out_prefix,
                                     '--chr-prefix', 'ch'])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{out_prefix}.fam")
    assert os.path.exists(f"{out_prefix}_run_plink.sh")
    assert os.path.exists(f"{out_prefix}_run_beagle.sh")

    # The trio relationship should appear in the .fam
    fam_content = open(f"{out_prefix}.fam").read()
    assert 'HG002' in fam_content
    assert 'HG003' in fam_content
    assert 'HG004' in fam_content


# ---------------------------------------------------------------------------
# --from-fam: end-to-end alternative input mode
# ---------------------------------------------------------------------------

def test_from_fam_and_trio_prefix_mutually_exclusive(standard_vcf_and_trio, tmp_path):
    vcf_path, trio_prefix = standard_vcf_and_trio
    fam_path = tmp_path / "existing.fam"
    fam_path.write_text("0\tHG002\tHG003\tHG004\t0\t-9\n")
    result = run_script(PPB_SCRIPT, [trio_prefix, vcf_path, '--from-fam', str(fam_path)])
    assert result.returncode != 0
    assert 'not both' in (result.stderr + result.stdout)


def test_neither_trio_prefix_nor_from_fam_errors(standard_vcf_and_trio):
    vcf_path, _ = standard_vcf_and_trio
    result = run_script(PPB_SCRIPT, [vcf_path])
    assert result.returncode != 0


def test_from_fam_missing_file_errors_cleanly(standard_vcf_and_trio, tmp_path):
    vcf_path, _ = standard_vcf_and_trio
    result = run_script(PPB_SCRIPT, [vcf_path, '--from-fam', str(tmp_path / "nope.fam")])
    assert result.returncode != 0


def test_from_fam_end_to_end_run(standard_vcf_and_trio, tmp_path):
    """
    --from-fam should produce the same kind of output as the trio_prefix
    path (a .fam reflecting the confirmed relationship, plus PLINK/BEAGLE
    scripts including --mendel), but reading an existing FAM (e.g.
    merge_refined_fam.py's output) instead of a vcfped .trio file.
    """
    vcf_path, _ = standard_vcf_and_trio
    existing_fam = tmp_path / "existing.fam"
    existing_fam.write_text(
        "0\tHG002\tHG003\tHG004\t0\t-9\n"
        "0\tHG003\t0\t0\t1\t-9\n"
        "0\tHG004\t0\t0\t2\t-9\n"
    )
    out_prefix = str(tmp_path / "from_fam_out")
    result = run_script(PPB_SCRIPT, [vcf_path, '--from-fam', str(existing_fam),
                                     '-o', out_prefix, '--chr-prefix', 'ch'])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{out_prefix}.fam")
    assert os.path.exists(f"{out_prefix}_run_plink.sh")
    assert os.path.exists(f"{out_prefix}_run_beagle.sh")

    fam_content = open(f"{out_prefix}.fam").read()
    assert 'HG002' in fam_content
    assert 'HG003' in fam_content
    assert 'HG004' in fam_content
    # Sex carried over from the existing FAM's SEX column (not re-derived as '0')
    fam_rows = {l.split()[1]: l.split() for l in fam_content.splitlines() if l.strip()}
    assert fam_rows['HG003'][4] == '1'
    assert fam_rows['HG004'][4] == '2'

    # The generated plink script must still include the Mendelian check.
    plink_script = open(f"{out_prefix}_run_plink.sh").read()
    assert '--mendel' in plink_script


def test_from_fam_requires_explicit_output(standard_vcf_and_trio, tmp_path):
    """
    Regression guard: --from-fam's natural prefix (stripping .fam) is the
    SAME path as the input FAM itself, unlike trio_prefix (whose <prefix>.trio
    input never collides with a <prefix>.fam output). Silently defaulting
    would overwrite the caller's input FAM (e.g. merge_refined_fam.py's
    carefully-built merged pedigree) with a regenerated one. -o must be
    required, not defaulted.
    """
    vcf_path, _ = standard_vcf_and_trio
    existing_fam = tmp_path / "cohort_refined_merged.fam"
    original_content = "0\tHG002\tHG003\tHG004\t0\t-9\n"
    existing_fam.write_text(original_content)

    result = run_script(PPB_SCRIPT, [vcf_path, '--from-fam', str(existing_fam)])
    assert result.returncode != 0
    assert '-o/--output' in (result.stderr + result.stdout)
    # Must not have touched the input file.
    assert existing_fam.read_text() == original_content
