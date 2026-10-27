"""Tests for vcfped/checkped.py — PED file parsing and trio validation."""
import os
import pytest

from conftest import import_module_from_path, VCFPED_DIR, TEST_PED

checkped = import_module_from_path('checkped', os.path.join(VCFPED_DIR, 'checkped.py'))


def test_readped_bundled_trio():
    """Bundled trio.ped should parse to HG002=child, HG003=father, HG004=mother."""
    ped = checkped.readped(TEST_PED)
    assert ped['child'] == 'HG002'
    assert ped['father'] == 'HG003'
    assert ped['mother'] == 'HG004'
    assert ped['child_gender'] == 'Male'
    assert set(ped['sampleset']) == {'HG002', 'HG003', 'HG004'}


def test_readped_rejects_non_trio(tmp_path):
    """A pedfile with != 3 lines must raise."""
    bad = tmp_path / "bad.ped"
    bad.write_text("FAM\tID1\t0\t0\t1\t1\nFAM\tID2\t0\t0\t2\t1\n")  # only 2 lines
    with pytest.raises(RuntimeError, match="exactly 3 lines"):
        checkped.readped(str(bad))


def test_readped_rejects_non_trio_structure(tmp_path):
    """3 lines that don't actually describe a parent-parent-child trio."""
    bad = tmp_path / "bad.ped"
    # Three unrelated founders, no child relationship
    bad.write_text(
        "FAM\tID1\t0\t0\t1\t1\n"
        "FAM\tID2\t0\t0\t2\t1\n"
        "FAM\tID3\t0\t0\t1\t1\n"
    )
    with pytest.raises(RuntimeError, match="does not describe a trio"):
        checkped.readped(str(bad))


def test_readped_handles_header_row(tmp_path):
    """A header row containing FID/MID/SEX should be stripped."""
    f = tmp_path / "with_header.ped"
    f.write_text(
        "FID\tIID\tPID\tMID\tSEX\n"
        "FAM\tCHILD\tFATHER\tMOTHER\t1\n"
        "FAM\tFATHER\t0\t0\t1\n"
        "FAM\tMOTHER\t0\t0\t2\n"
    )
    ped = checkped.readped(str(f))
    assert ped['child'] == 'CHILD'
    assert ped['father'] == 'FATHER'
    assert ped['mother'] == 'MOTHER'
