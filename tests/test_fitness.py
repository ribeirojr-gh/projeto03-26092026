"""Stage 5 tests: loss definition (pure numpy) and LAMMPS-driven objective."""
from pathlib import Path

import numpy as np
import pytest

from ga_reaxff.dataset import TrainingSet, displacement_scan, rattle_series, strain_series
from ga_reaxff.fitness import PENALTY, compute_loss
from ga_reaxff.structures import graphene_oxide_model

REF = Path(__file__).parents[1] / "data" / "ffield.reax.cho"


def _toy_set():
    go = graphene_oxide_model(a=2.51)
    confs = strain_series(go, [-0.01, 0.0, 0.01]) + rattle_series(go, 2, 0.03, seed=3)
    for k, c in enumerate(confs):
        c.info["ref_energy"] = 100.0 + 10.0 * k
        c.arrays["ref_forces"] = np.zeros((len(c), 3))
    return TrainingSet(confs)


# ------------------------------------------------------------ pure numpy
def test_perfect_prediction_gives_zero():
    ts = _toy_set()
    e = [c.info["ref_energy"] for c in ts.configs]
    f = [c.arrays["ref_forces"] for c in ts.configs]
    r = compute_loss(ts, e, f)
    assert r.loss == 0.0 and r.energy_rmse == 0.0 and r.force_rmse == 0.0


def test_constant_energy_shift_per_family_is_invisible():
    """Relative energies: shifting a whole family must not change the loss."""
    ts = _toy_set()
    e = np.array([c.info["ref_energy"] for c in ts.configs])
    f = [c.arrays["ref_forces"] for c in ts.configs]
    e[:3] += 1234.5           # strain family
    e[3:] -= 77.0             # rattle family
    assert compute_loss(ts, e, f).loss == pytest.approx(0.0, abs=1e-18)


def test_hand_computed_loss():
    ts = _toy_set()
    e = np.array([c.info["ref_energy"] for c in ts.configs])
    f = [c.arrays["ref_forces"].copy() for c in ts.configs]
    e[0] += 2.0                          # strain_-0.010 off by 2 kcal/mol
    f[4] += 0.5                          # rattle_001 forces off by 0.5 everywhere
    r = compute_loss(ts, e, f, sigma_e=1.0, sigma_f=1.0, force_weight=2.0)
    n_e = 4                              # non-reference configs: 2 strain + 2 rattle
    expected_e = (2.0 ** 2) / n_e
    assert len(ts) == 6                  # 3 strain + (1 reference + 2 rattled)
    expected_f = (0.5 ** 2) / 6          # mean-squared force error, averaged over 6 configs
    assert r.loss == pytest.approx(expected_e + 2.0 * expected_f)
    assert r.family_energy_rmse["strain"] == pytest.approx(np.sqrt(4.0 / 2))
    assert r.family_energy_rmse["rattle"] == pytest.approx(0.0)


def test_weights_scale_contributions():
    ts = _toy_set()
    e = np.array([c.info["ref_energy"] for c in ts.configs])
    e[0] += 1.0
    base = compute_loss(ts, e, None).loss
    ts.set_weights("strain", w_e=3.0)
    assert compute_loss(ts, e, None).loss == pytest.approx(3.0 * base)
    ts.set_weights("strain", w_e=0.0)
    assert compute_loss(ts, e, None).loss == 0.0


# ------------------------------------------------------ LAMMPS objective
lammps = pytest.importorskip("lammps")
from ga_reaxff.engine import label_with_forcefield  # noqa: E402
from ga_reaxff.ffield import ForceField  # noqa: E402
from ga_reaxff.fitness import Objective  # noqa: E402
from ga_reaxff.parameters import ParameterSpace  # noqa: E402

KEYS = ["bond:C-O:De_sigma", "bond:C-O:p_bo2", "angle:C-O-C:theta_00", "offdiag:C-O:r_vdw"]


@pytest.fixture(scope="module")
def setup():
    ff = ForceField.read(REF)
    go = graphene_oxide_model(a=2.51)
    epox_o = 32
    confs = (strain_series(go, [-0.01, 0.0, 0.01])
             + displacement_scan(go, [epox_o], [0, 0, 1], [-0.1, 0.0, 0.2], family="epoxide_z")
             + rattle_series(go, 2, 0.03, seed=5))
    label_with_forcefield(confs, ff)
    ts = TrainingSet(confs)
    space = ParameterSpace.relative(ff, KEYS, rel=0.2)
    obj = Objective(ts, ff, space)
    yield ff, space, obj
    obj.close()


def test_reference_parameters_are_a_global_zero(setup):
    ff, space, obj = setup
    r = obj.result(np.full(len(space), 0.5))      # genome center == reference values
    assert r.loss < 1e-6 and not r.failed


def test_loss_grows_away_from_reference(setup):
    _, space, obj = setup
    losses = [obj(np.r_[0.5 + t, 0.5, 0.5, 0.5]) for t in (0.0, 0.1, 0.25, 0.5)]
    assert losses[0] < losses[1] < losses[2] < losses[3]


def test_cache_avoids_reevaluation(setup):
    _, space, obj = setup
    g = np.array([0.3, 0.6, 0.4, 0.7])
    obj(g)
    n = obj.n_calls
    assert obj(g.copy()) == obj(g) and obj.n_calls == n


def test_crashing_parameters_get_penalty(setup):
    ff, space, obj = setup
    bad = ff.with_values(["bond:C-O:p_bo2"], [-50.0])
    r = obj.evaluate_ff(bad)
    assert r.failed and r.loss == PENALTY
    # and the objective keeps working afterwards
    assert obj.evaluate_ff(ff).loss < 1e-6
