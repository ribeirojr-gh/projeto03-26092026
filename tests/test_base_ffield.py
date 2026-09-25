"""Stage 12 tests: merging element blocks into a base force field (Zn + C/H/N/O)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from ga_reaxff.engine import _mass, relax, single_point
from ga_reaxff.ffield import ForceField, merge_elements, missing_interactions

SRC = Path(__file__).parents[1] / "data" / "ffields" / "sources"


@pytest.fixture(scope="module")
def zn():
    return ForceField.read(SRC / "ffield.reax.ZnOH")


@pytest.fixture(scope="module")
def fc():
    return ForceField.read(SRC / "ffield.reax.FC")


@pytest.fixture(scope="module")
def merged(fc, zn):
    return merge_elements(fc, zn, ["Zn"])


def test_source_files_integrity():
    for line in (SRC / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == digest, name


def test_merge_appends_element_and_keeps_base(merged, fc, zn):
    ff, rep = merged
    assert ff.elements == fc.elements + ["Zn"]
    # base parameters untouched
    assert ff.get("bond:C-O:De_sigma") == fc.get("bond:C-O:De_sigma")
    assert ff.get("atom:O:r_s") == fc.get("atom:O:r_s")
    assert np.array_equal(ff.general, fc.general)
    # Zn block and Zn cross terms come from the donor
    assert ff.get("atom:Zn:r_s") == zn.get("atom:Zn:r_s")
    assert ff.get("bond:Zn-O:De_sigma") == zn.get("bond:O-Zn:De_sigma")
    assert ("bond", ("O", "Zn")) in rep.added or ("bond", ("Zn", "O")) in rep.added
    assert all("Zn" in labs for _, labs in rep.added)
    assert len(fc.elements) == 10, "merge must not modify its inputs"


def test_merge_reports_differences(merged):
    _, rep = merged
    assert "O" in rep.shared_atom_differences and "H" in rep.shared_atom_differences
    assert all(not name.startswith("nu") for d in rep.shared_atom_differences.values()
               for name in d)


def test_merge_rejects_bad_requests(fc, zn):
    with pytest.raises(ValueError, match="not in the donor"):
        merge_elements(fc, zn, ["Cu"])
    with pytest.raises(ValueError, match="already in the base"):
        merge_elements(fc, zn, ["O"])


def test_missing_interactions(merged):
    ff, _ = merged
    m = missing_interactions(ff, ["Zn"])
    assert ("C", "Zn") in m["bond"] and ("N", "Zn") in m["bond"]
    assert ("O", "Zn") not in m["bond"]
    assert ("Zn", "Zn") not in m["offdiag"], "same-element pairs need no off-diagonal"
    assert all("X" not in t for v in m.values() for t in v), "dummy type is not an element"
    assert ("N", "Zn", "N") in m["angle"]


def test_merged_file_round_trip(merged, tmp_path):
    ff, _ = merged
    back = ForceField.read(ff.write(tmp_path / "ffield"))
    assert back.elements == ff.elements
    assert back.diff(ff) == {}


def test_wildcard_label_outside_torsions_is_rejected(fc, tmp_path):
    ff = fc.copy()
    ff.blocks["bond"][0].labels = ("X", "C")
    with pytest.raises(ValueError, match="ambiguous"):
        ff.write(tmp_path / "bad")


def test_masses():
    assert _mass("C") == 12.011
    assert abs(_mass("Zn") - 65.38) < 0.01
    assert _mass("X") == 1.0


def test_zn_cluster_is_finite_and_stays_bound(merged):
    """Merged FC + Zn: a Zn(OH)(H2O) fragment evaluates and Zn-O stays bonded."""
    from ase import Atoms
    ff, _ = merged
    a = Atoms("ZnOHOHH", positions=[[0, 0, 0], [1.95, 0, 0], [2.3, 0.9, 0],
                                    [-1.2, 1.7, 0], [-1.9, 2.3, 0.3], [-0.6, 2.3, -0.5]],
              cell=[20, 20, 20], pbc=True)
    a.center()
    e, f = single_point(a, ff)
    assert np.isfinite(e) and np.all(np.isfinite(f))
    r = relax(a, ff, ftol=1e-2, maxiter=2000)
    assert r.get_distance(0, 1, mic=True) < 2.3
