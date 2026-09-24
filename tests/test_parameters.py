"""Stage 2 tests: parameter space bounds, encode/decode, perturbation."""
from pathlib import Path

import numpy as np
import pytest

from ga_reaxff.ffield import ForceField
from ga_reaxff.parameters import ParameterSpace, ParameterSpec, perturb

REF = Path(__file__).parents[1] / "data" / "ffield.reax.cho"
KEYS = ["bond:C-O:De_sigma", "bond:C-O:p_bo2", "offdiag:C-O:r_vdw",
        "angle:C-O-C:theta_00", "angle:C-O-C:p_coa1"]  # last one is 0.0 in the file


@pytest.fixture(scope="module")
def ff():
    return ForceField.read(REF)


def test_relative_bounds(ff):
    ps = ParameterSpace.relative(ff, KEYS, rel=0.2, min_width=0.1)
    v = ps.values_from(ff)
    assert np.allclose(ps.upper - v, np.maximum(0.2 * np.abs(v), 0.1))
    assert np.allclose(v - ps.lower, np.maximum(0.2 * np.abs(v), 0.1))
    # the zero-valued parameter gets the minimum width, not a degenerate interval
    i = KEYS.index("angle:C-O-C:p_coa1")
    assert ps.lower[i] == pytest.approx(-0.1) and ps.upper[i] == pytest.approx(0.1)


def test_initial_is_center(ff):
    ps = ParameterSpace.relative(ff, KEYS)
    assert np.allclose(ps.encode(ps.values_from(ff)), 0.5)
    assert ps.contains(ff)


def test_encode_decode_inverse(ff):
    ps = ParameterSpace.relative(ff, KEYS)
    g = np.random.default_rng(0).random((50, len(ps)))
    for row in g:
        assert np.allclose(ps.encode(ps.decode(row)), row)


def test_decode_clips(ff):
    ps = ParameterSpace.relative(ff, KEYS)
    vals = ps.decode(np.array([-1.0, 2.0, 0.5, 0.5, 0.5]))
    assert vals[0] == ps.lower[0] and vals[1] == ps.upper[1]


def test_apply_changes_only_selected(ff):
    ps = ParameterSpace.relative(ff, KEYS)
    new = ps.apply(ff, np.zeros(len(ps)))
    assert set(ff.diff(new)) <= set(KEYS)
    assert np.allclose(ps.values_from(new), ps.lower)


def test_from_config_overrides(ff):
    cfg = {"keys": KEYS, "rel": 0.3, "bounds": {"angle:C-O-C:theta_00": [60.0, 90.0]}}
    ps = ParameterSpace.from_config(ff, cfg)
    i = KEYS.index("angle:C-O-C:theta_00")
    assert (ps.lower[i], ps.upper[i]) == (60.0, 90.0)
    with pytest.raises(KeyError):
        ParameterSpace.from_config(ff, {"keys": KEYS, "bounds": {"bond:C-C:De_pi": [1, 2]}})


def test_validation():
    with pytest.raises(ValueError):
        ParameterSpec("x", 1.0, 1.0)
    with pytest.raises(ValueError):
        ParameterSpace([ParameterSpec("a", 0, 1), ParameterSpec("a", 0, 2)])


def test_perturb_moves_every_parameter_within_range(ff):
    rng = np.random.default_rng(42)
    keys = KEYS[:4]
    new = perturb(ff, keys, rel=0.15, rng=rng)
    for k in keys:
        r = abs(new.get(k) / ff.get(k) - 1)
        assert 0.075 - 1e-12 <= r <= 0.15 + 1e-12
    assert set(ff.diff(new)) == set(keys)
    # reproducible with the same seed
    again = perturb(ff, keys, rel=0.15, rng=np.random.default_rng(42))
    assert again.diff(new) == {}
