"""Stage 3 tests: GO model geometry, configuration generators, dataset I/O."""
import numpy as np
import pytest

from ga_reaxff.dataset import TrainingSet, displacement_scan, rattle_series, strain_series
from ga_reaxff.structures import (D_CC, carbon_neighbors, graphene_oxide_model, graphene_sheet,
                                  min_interatomic_distance)


def test_graphene_geometry():
    g = graphene_sheet(4, 2)
    assert len(g) == 32
    assert min_interatomic_distance(g) == pytest.approx(D_CC, abs=1e-9)
    # every carbon has exactly three nearest neighbours (periodic, defect-free)
    assert all(len(carbon_neighbors(g, i)) == 3 for i in range(len(g)))
    frac = g.get_scaled_positions(wrap=False)
    assert np.all((frac >= 0) & (frac < 1))


def test_go_model_composition_and_sanity():
    go = graphene_oxide_model()
    syms = go.get_chemical_symbols()
    assert (syms.count("C"), syms.count("O"), syms.count("H")) == (32, 3, 2)
    assert min_interatomic_distance(go) > 0.9          # no overlapping atoms
    sites = go.info["sites"]
    used = set(sites["epoxide_C"]) | {sites["hydroxyl_top_C"], sites["hydroxyl_bottom_C"]}
    assert len(used) == 4                               # four distinct functionalized carbons
    z_sheet = go.positions[0, 2]
    o_z = go.positions[[i for i, s in enumerate(syms) if s == "O"], 2] - z_sheet
    assert (o_z > 0).sum() == 2 and (o_z < 0).sum() == 1  # two groups on top, one below


def test_strain_series():
    go = graphene_oxide_model()
    confs = strain_series(go, [-0.02, 0.0, 0.02])
    assert [c.info["is_ref"] for c in confs] == [False, True, False]
    assert confs[2].cell[0, 0] == pytest.approx(go.cell[0, 0] * 1.02)
    # fractional coordinates are preserved in-plane
    assert np.allclose(confs[2].get_scaled_positions(wrap=False)[:, :2],
                       go.get_scaled_positions(wrap=False)[:, :2])


def test_displacement_scan_moves_only_selected():
    go = graphene_oxide_model()
    confs = displacement_scan(go, [32], [0, 0, 2.0], [0.0, 0.3], family="epox")
    dx = confs[1].positions - go.positions
    assert np.allclose(dx[32], [0, 0, 0.3]) and np.allclose(np.delete(dx, 32, 0), 0)


def test_rattle_reproducible():
    go = graphene_oxide_model()
    a = rattle_series(go, 3, 0.05, seed=7)
    b = rattle_series(go, 3, 0.05, seed=7)
    assert len(a) == 4 and a[0].info["is_ref"]
    assert all(np.allclose(x.positions, y.positions) for x, y in zip(a, b))


def test_dataset_roundtrip_and_validation(tmp_path):
    go = graphene_oxide_model()
    confs = strain_series(go, [-0.01, 0.0, 0.01]) + rattle_series(go, 2, 0.05, seed=1)
    for k, c in enumerate(confs):
        c.info["ref_energy"] = -1000.0 + k
        c.arrays["ref_forces"] = np.full((len(c), 3), 0.1 * k)
    ts = TrainingSet(confs)
    ts.set_weights("rattle", w_f=2.0)
    ts2 = TrainingSet.read(ts.write(tmp_path / "ts.extxyz"))
    assert ts2.families == ["strain", "rattle"] and ts2.is_labeled()
    for x, y in zip(ts.configs, ts2.configs):
        assert np.allclose(x.positions, y.positions)
        assert x.info["ref_energy"] == pytest.approx(y.info["ref_energy"])
        assert np.allclose(x.arrays["ref_forces"], y.arrays["ref_forces"])
        assert x.info["w_f"] == y.info["w_f"] and x.info["is_ref"] == y.info["is_ref"]


def test_dataset_rejects_duplicate_names_and_double_references():
    go = graphene_oxide_model()
    with pytest.raises(ValueError, match="unique"):
        TrainingSet(strain_series(go, [0.0]) + strain_series(go, [0.0]))
    a, b = strain_series(go, [0.0]), strain_series(go, [0.0])
    b[0].info["name"] = "another_name"
    with pytest.raises(ValueError, match="exactly one reference"):
        TrainingSet(a + b)
