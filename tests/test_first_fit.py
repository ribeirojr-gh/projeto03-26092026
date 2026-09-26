"""Stage 17 tests: stress, sharded evaluation, stress loss, MOF training configurations."""
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, molecule

from ga_reaxff import qmof
from ga_reaxff.dataset import TrainingSet
from ga_reaxff.engine import LammpsEngine, ShardedEngine
from ga_reaxff.ffield import ForceField
from ga_reaxff.fitness import compute_loss
from ga_reaxff.mofdata import bond_scan_family, equilibrium_config, first_contact, strain_family

ROOT = Path(__file__).parents[1]
INIT = ROOT / "data" / "ffields" / "init" / "ffield.Zn-FC.init"
CHO = ROOT / "data" / "ffield.reax.cho"
KCAL_A3_TO_GPA = 6.947695


@pytest.fixture(scope="module")
def mof():
    s = qmof.load_structures(ROOT / "data" / "qmof" / "structures_Zn-CHNO.extxyz.gz")
    return next(a for a in s if a.info["identifier"] == "qmof-69d6aa4")


def _energy(atoms, ff):
    with LammpsEngine([atoms], ff, qeq_tol=1e-10) as eng:
        return eng.evaluate(ff)[0][0]


@pytest.mark.parametrize("component", [0, 3])          # xx and yz on a triclinic MOF
def test_stress_matches_strain_derivative(mof, component):
    ff = ForceField.read(INIT)
    with LammpsEngine([mof], ff, qeq_tol=1e-10) as eng:
        _, _, s = eng.evaluate(ff, stress=True)[0]
    i, j = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)][component]
    h = 2e-4
    eps = np.zeros((3, 3))
    eps[i, j] = eps[j, i] = h / (1 if i == j else 2)
    e = []
    for sgn in (1, -1):
        a = mof.copy()
        a.set_cell(mof.cell.array @ (np.eye(3) + sgn * eps), scale_atoms=True)
        e.append(_energy(a, ff))
    numeric = (e[0] - e[1]) / (2 * h) / mof.get_volume() * KCAL_A3_TO_GPA
    assert s[component] == pytest.approx(numeric, rel=5e-3, abs=0.05)


def test_sharded_engine_matches_serial():
    ff = ForceField.read(CHO)
    configs = []
    for k in range(5):
        a = bulk("C", "diamond", a=3.57, cubic=True)
        a.rattle(0.03, seed=k)
        configs.append(a)
    with LammpsEngine(configs, ff) as eng:
        serial = eng.evaluate(ff, stress=True)
    with ShardedEngine(configs, ff, n_workers=3) as sh:
        par = sh.evaluate(ff, stress=True)
    for (e0, f0, s0), (e1, f1, s1) in zip(serial, par):
        assert e1 == pytest.approx(e0, abs=1e-6)
        np.testing.assert_allclose(f1, f0, atol=1e-6)
        np.testing.assert_allclose(s1, s0, atol=1e-6)


def test_stress_term_in_loss():
    a = molecule("H2O", vacuum=4.0, pbc=True)
    a.info.update(identifier="w")
    eq = equilibrium_config(a, w_f=0.0, w_s=2.0)
    ts = TrainingSet([eq])
    r = compute_loss(ts, [0.0], [np.zeros((3, 3))], stresses=[[1.0, 0, 0, 0, 0, 0]],
                     sigma_s=0.5, stress_weight=1.0)
    assert r.stress_rmse == pytest.approx(np.sqrt(1 / 6))
    assert r.loss_terms["stress"] == pytest.approx(2.0 * (1 / 6) / 0.25)
    assert compute_loss(ts, [0.0], [np.zeros((3, 3))]).loss_terms["stress"] == 0.0


def test_equilibrium_and_strain_configs(mof):
    eq = equilibrium_config(mof)
    assert eq.info["is_ref"] and eq.info["w_e"] == 0 and eq.info["ref_stress"] == [0.0] * 6
    assert np.all(eq.arrays["ref_forces"] == 0) and "pbe_ddec_charge" not in eq.arrays
    fam = strain_family(mof, [-0.02, 0.0, 0.02])
    assert [c.info["is_ref"] for c in fam] == [False, True, False]
    assert fam[2].get_volume() == pytest.approx(mof.get_volume() * 1.02 ** 3)
    assert all(c.info["w_f"] == 0.0 for c in fam)
    TrainingSet([eq] + fam)            # families are consistent


def test_bond_scan_moves_metal_along_bond(mof):
    zn, o = first_contact(mof, "Zn", "O")
    d0 = mof.get_distance(zn, o, mic=True)
    fam = bond_scan_family(mof, zn, o, [-0.1, 0.0, 0.2])
    d = [c.get_distance(zn, o, mic=True) for c in fam]
    assert d == pytest.approx([d0 - 0.1, d0, d0 + 0.2], abs=1e-9)
    assert sum(c.info["is_ref"] for c in fam) == 1
    assert first_contact(mof, "Zn", "N") is None      # carboxylate MOF: no Zn-N


def test_linear_molecule_bend():
    sys.path.insert(0, str(ROOT / "scripts"))
    from molecule_refs import _bend
    a = molecule("CO2")
    b = _bend(a, 1, 0, 2, 150.0)
    assert b.get_angle(1, 0, 2) == pytest.approx(150.0, abs=1e-6)
    assert b.get_distance(0, 2) == pytest.approx(a.get_distance(0, 2))
