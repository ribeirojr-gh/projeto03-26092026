"""Stage 1 tests: ffield parsing, addressing and exact round trip."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from ga_reaxff.ffield import ForceField

REF = Path(__file__).parents[1] / "data" / "ffield.reax.cho"
REF_SHA256 = (REF.parent / "ffield.reax.cho.sha256").read_text().split()[0]


def test_reference_file_integrity():
    """The shipped reference force field must be byte-identical to the documented one."""
    assert hashlib.sha256(REF.read_bytes()).hexdigest() == REF_SHA256


@pytest.fixture(scope="module")
def ff():
    return ForceField.read(REF)


def test_block_sizes(ff):
    assert ff.elements == ["C", "H", "O"]
    assert len(ff.general) == 39
    sizes = {k: len(v) for k, v in ff.blocks.items()}
    assert sizes == {"atom": 3, "bond": 6, "offdiag": 3, "angle": 18, "torsion": 26, "hbond": 1}


def test_known_values(ff):
    # values read by eye from the file (Chenoweth 2008)
    assert ff.get("general:0") == pytest.approx(50.0)
    assert ff.get("atom:C:r_s") == pytest.approx(1.3825)
    assert ff.get("atom:O:chi") == pytest.approx(8.5)
    assert ff.get("bond:C-O:De_sigma") == pytest.approx(160.4802)
    assert ff.get("bond:C-O:p_bo2") == pytest.approx(5.2913)
    assert ff.get("offdiag:C-O:r_vdw") == pytest.approx(1.8523)
    assert ff.get("angle:C-O-C:theta_00") == pytest.approx(74.3994)
    assert ff.get("torsion:X-C-C-X:V2") == pytest.approx(50.0)
    assert ff.get("hbond:O-H-O:r0_hb") == pytest.approx(1.9682)


def test_symmetric_addressing(ff):
    assert ff.get("bond:O-C:De_sigma") == ff.get("bond:C-O:De_sigma")
    assert ff.get("offdiag:O-C:D") == ff.get("offdiag:C-O:D")
    assert ff.get("angle:O-C-C:theta_00") == ff.get("angle:C-C-O:theta_00")


def test_bad_keys(ff):
    for key in ["bond:C-O:nonsense", "bond:C-N:De_sigma", "foo:C:r_s", "bond:C-O"]:
        with pytest.raises(KeyError):
            ff.get(key)


def test_set_and_with_values_do_not_alias(ff):
    new = ff.with_values(["bond:C-O:De_sigma"], [123.0])
    assert new.get("bond:C-O:De_sigma") == 123.0
    assert ff.get("bond:C-O:De_sigma") == pytest.approx(160.4802)  # original untouched
    assert set(ff.diff(new)) == {"bond:C-O:De_sigma"}


def test_exact_round_trip(ff, tmp_path):
    p = ff.write(tmp_path / "ffield.rt")
    ff2 = ForceField.read(p)
    assert ff.diff(ff2, tol=0.0) == {}
    assert ff2.elements == ff.elements
    # second round trip is byte-identical (writer is deterministic)
    p2 = ff2.write(tmp_path / "ffield.rt2")
    assert p.read_bytes() == p2.read_bytes()


def test_all_keys_count(ff):
    n = 39 + 3 * 32 + 6 * 16 + 3 * 6 + 18 * 7 + 26 * 7 + 1 * 4
    keys = ff.all_keys()
    assert len(keys) == n and len(set(keys)) == n
    assert all(np.isfinite(ff.get(k)) for k in keys)
