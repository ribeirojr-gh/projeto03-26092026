"""Stage 14 tests: triclinic cells and cell relaxation in the LAMMPS engine."""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from ga_reaxff import qmof
from ga_reaxff.engine import relax, single_point
from ga_reaxff.ffield import ForceField

ROOT = Path(__file__).parents[1]
BASE = ROOT / "data" / "ffields" / "base" / "ffield.Zn-FC"
CHO = ROOT / "data" / "ffield.reax.cho"


@pytest.fixture(scope="module")
def ff_zn():
    return ForceField.read(BASE)


@pytest.fixture(scope="module")
def ff_init(ff_zn):
    """Base + placeholder C-Zn / N-Zn entries (needed for finite forces)."""
    from ga_reaxff.ffield import add_placeholder_pairs
    return add_placeholder_pairs(ff_zn, [("C", "Zn"), ("N", "Zn")], ("O", "Zn"))[0]


@pytest.fixture(scope="module")
def mof():
    """Smallest triclinic Zn MOF of the QMOF Zn + C/H/N/O family."""
    s = qmof.load_structures(ROOT / "data" / "qmof" / "structures_Zn-CHNO.extxyz.gz")
    tri = [a for a in s if not np.allclose(a.cell.angles(), 90)]
    return min(tri, key=len)


def _rotation(seed=3):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_triclinic_mof_evaluates(mof, ff_init):
    e, f = single_point(mof, ff_init)
    assert np.isfinite(e) and f.shape == (len(mof), 3) and np.all(np.isfinite(f))


def test_rigid_rotation_invariance(mof, ff_init):
    """Rotating cell and atoms together: same energy, forces rotate with the structure."""
    R = _rotation()
    rot = mof.copy()
    rot.set_cell(mof.cell.array @ R.T, scale_atoms=False)
    rot.positions = mof.positions @ R.T
    e0, f0 = single_point(mof, ff_init)
    e1, f1 = single_point(rot, ff_init)
    assert e1 == pytest.approx(e0, abs=1e-5)
    np.testing.assert_allclose(f1, f0 @ R.T, atol=1e-5)


def test_equivalent_cell_basis(ff_zn):
    """Same crystal described by a sheared basis (b -> b + a): identical energy and forces."""
    ff = ForceField.read(CHO)
    a = bulk("C", "diamond", a=3.57, cubic=True)
    a.rattle(0.05, seed=1)
    b = a.copy()
    c = a.cell.array.copy()
    b.set_cell([c[0], c[1] + c[0], c[2]], scale_atoms=False)
    e0, f0 = single_point(a, ff)
    e1, f1 = single_point(b, ff)
    assert e1 == pytest.approx(e0, abs=1e-5)
    np.testing.assert_allclose(f1, f0, atol=1e-5)


def test_fixed_cell_relax_keeps_cell_and_frame(mof, ff_init):
    r = relax(mof, ff_init, ftol=1.0, maxiter=50)
    np.testing.assert_allclose(r.cell.array, mof.cell.array)
    assert np.abs(r.positions - mof.positions).max() < 1.0   # original frame, not rotated


def test_cell_relax_diamond():
    """Strained diamond relaxes to the force field's lattice constant."""
    ff = ForceField.read(CHO)
    a = bulk("C", "diamond", a=3.75, cubic=True)
    r = relax(a, ff, ftol=1e-4, maxiter=5000, cell=True)
    assert r.get_volume() < a.get_volume()
    lat = r.cell.cellpar()
    assert np.allclose(lat[:3], lat[0], atol=0.01) and np.allclose(lat[3:], 90, atol=0.1)
    assert 3.4 < lat[0] < 3.7
    # at the relaxed cell the structure is a minimum: small forces
    _, f = single_point(r, ff)
    assert np.abs(f).max() < 0.5


# --- placeholders and per-MOF validation --------------------------------------

def _zn_carbon_fragment():
    """Zn bound to two formate O atoms in a triclinic box (contains Zn-C pairs)."""
    sym = ["Zn", "O", "O", "C", "C", "H", "H", "O", "O"]
    pos = [[0, 0, 0], [1.95, 0, 0], [-1.95, 0, 0], [2.65, 1.05, 0], [-2.65, -1.05, 0],
           [3.75, 0.95, 0], [-3.75, -0.95, 0], [2.2, 2.25, 0], [-2.2, -2.25, 0]]
    a = Atoms(sym, positions=pos, cell=[[14, 0, 0], [3, 13, 0], [1, 2, 12]], pbc=True)
    a.center()
    return a


def test_missing_bond_entry_gives_nan_and_is_caught(mof, ff_zn):
    """Zn carboxylate MOF without a C-Zn bond entry: LAMMPS returns NaN forces
    (found in stage 14); single_point turns that into an EvaluationError."""
    from ga_reaxff.engine import EvaluationError
    with pytest.raises(EvaluationError, match="non-finite"):
        single_point(mof, ff_zn)


def test_placeholder_pairs_fix_nan(mof, ff_zn):
    from ga_reaxff.ffield import add_placeholder_pairs
    ff, added = add_placeholder_pairs(ff_zn, [("C", "Zn"), ("N", "Zn")], ("O", "Zn"))
    assert ("bond", ("C", "Zn")) in added and ("offdiag", ("N", "Zn")) in added
    assert ff.get("bond:C-Zn:De_sigma") == ff_zn.get("bond:O-Zn:De_sigma")
    e, f = single_point(_zn_carbon_fragment(), ff)
    assert np.isfinite(e) and np.all(np.isfinite(f))
    e, f = single_point(mof, ff)
    assert np.isfinite(e) and np.all(np.isfinite(f)) and e < 0
    # idempotent and does not touch the input
    ff2, added2 = add_placeholder_pairs(ff, [("C", "Zn")], ("O", "Zn"))
    assert added2 == [] and not any(set(e.labels) == {"C", "Zn"} for e in ff_zn.blocks["bond"])
    with pytest.raises(ValueError, match="template"):
        add_placeholder_pairs(ff_zn, [("C", "Zn")], ("Cu", "O"))


def test_scorecard_identity_and_translation():
    from ga_reaxff.validate import compare
    a = bulk("ZnO", "wurtzite", a=3.25, c=5.2) * (2, 2, 2)
    a.info["identifier"] = "zno"
    s = compare(a, a, ligands=("O",))
    assert s.passed and s.volume_change == 0 and s.displacement_rms < 1e-12
    assert s.metal_coordination_ref == 4.0 and s.ligands_lost == 0
    b = a.copy()
    b.translate([0.7, -0.3, 0.2])                     # rigid translation is not a change
    assert compare(b, a, ligands=("O",)).displacement_rms < 1e-9


def test_scorecard_detects_expansion_and_lost_ligands():
    from ga_reaxff.validate import compare
    a = bulk("ZnO", "wurtzite", a=3.25, c=5.2) * (2, 2, 2)
    big = a.copy()
    big.set_cell(a.cell * 1.2, scale_atoms=True)      # +73 % volume, Zn-O 2.4 A
    s = compare(big, a, ligands=("O",))
    assert s.volume_change == pytest.approx(1.2 ** 3 - 1)
    assert not s.passed
    bigger = a.copy()
    bigger.set_cell(a.cell * 1.4, scale_atoms=True)   # Zn-O ~2.8 A > cutoff
    assert compare(bigger, a, ligands=("O",)).ligands_lost > 0
