"""
Tests for utilities/vcfped_rename_samples.py.

End-to-end: run vcfped on the bundled test VCF, then rename — verify the
numeric sample indices (1,2,3) in .pair/.trio/.gender are correctly
replaced with HG002/HG003/HG004.
"""
import os
import pytest

from conftest import run_script, run_module, UTILITIES_DIR

RENAME_SCRIPT = os.path.join(UTILITIES_DIR, 'vcfped_rename_samples.py')


def test_rename_replaces_numeric_ids_with_names(vcfped_run):
    result = run_script(RENAME_SCRIPT, [vcfped_run])
    assert result.returncode == 0, result.stderr

    named_trio = f"{vcfped_run}_named.trio"
    assert os.path.exists(named_trio)
    content = open(named_trio).read()
    assert 'HG002' in content
    assert 'HG003' in content
    assert 'HG004' in content
    # No bare numeric sample refs like a standalone "1," should remain in the Triple column
    assert '1,2,3' not in content


def test_rename_preserves_pair_and_gender_files(vcfped_run):
    result = run_script(RENAME_SCRIPT, [vcfped_run])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{vcfped_run}_named.pair")
    assert os.path.exists(f"{vcfped_run}_named.gender")
    pair_content = open(f"{vcfped_run}_named.pair").read()
    assert 'HG002' in pair_content


def test_custom_suffix(vcfped_run):
    result = run_script(RENAME_SCRIPT, [vcfped_run, '-s', '_id'])
    assert result.returncode == 0, result.stderr
    assert os.path.exists(f"{vcfped_run}_id.trio")


def test_inplace_overwrites_original(work_dir, test_vcf):
    prefix = os.path.join(work_dir, 'inplace_test')
    result = run_module(['vcfped.vcfped', test_vcf,
                         '-o', prefix, '-j', '2'])
    assert result.returncode == 0, result.stderr

    before = open(f"{prefix}.trio").read()
    assert 'HG002' not in before  # still numeric before renaming

    result = run_script(RENAME_SCRIPT, [prefix, '--inplace'], stdin_input='y\n')
    assert result.returncode == 0, result.stderr

    after = open(f"{prefix}.trio").read()
    assert 'HG002' in after
    # The --inplace variant should not create a separate _named file
    assert not os.path.exists(f"{prefix}_named.trio")


def test_rename_requires_log_file(tmp_path):
    """Without a .log file (sample name mapping source), rename should fail
    cleanly rather than silently producing garbage output."""
    prefix = str(tmp_path / "nolog")
    # Create the bare minimum trio file but no .log
    with open(f"{prefix}.trio", 'w') as f:
        f.write("Triple\tPivot\tPerc\tAutos\tAA+BB\t=AB\tTest1\tBB+BB\t=BB\tTest2\tVerdict\n")
    result = run_script(RENAME_SCRIPT, [prefix])
    assert result.returncode != 0
