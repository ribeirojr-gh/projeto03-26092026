"""Stage 13 tests: QMOF structures of the Zn family and teacher-check utilities."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from ga_reaxff import qmof
from ga_reaxff.teacher import (energy_consistency, reference_errors, relax_with_cell,
                               sample_indices)

DATA = Path(__file__).parents[1] / "data" / "qmof"
STRUCT = DATA / "structures_Zn-CHNO.extxyz.gz"


def _reduced(counts):
    from math import gcd
    from functools import reduce
    g = reduce(gcd, counts.values())
    return {el: n // g for el, n in counts.items()}


@pytest.fixture(scope="module")
def structures():
    return qmof.load_structures(STRUCT)      # verifies SHA-256


def test_zn_family_structures(structures):
    assert len(structures) == 2958
    ids = [s.info["identifier"] for s in structures]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for s in structures:
        els = set(s.get_chemical_symbols())
        assert "Zn" in els and els <= {"Zn", "C", "H", "N", "O"}


def test_structures_match_snapshot_metadata(structures):
    rows = {r["identifier"]: r for r in qmof.load_snapshot(DATA)}
    for s in structures[::97]:
        r = rows[s.info["identifier"]]
        assert len(s) == r["natoms"]
        assert _reduced(Counter(s.get_chemical_symbols())) == \
            _reduced(Counter(Atoms(r["reducedFormula"]).get_chemical_symbols()))
        assert s.info["energy_pbe_d3_eV"] == pytest.approx(r["outputsDFT.PBE.energy.total"])


def test_ddec_charges_are_neutral_and_physical(structures):
    for s in structures[::50]:
        q = s.arrays["pbe_ddec_charge"]
        assert abs(q.sum()) < 1e-3
        zn = q[np.array(s.get_chemical_symbols()) == "Zn"]
        assert np.all((zn > 0.5) & (zn < 2.0)), "Zn(II) DDEC charges are ~+1"


def test_structure_file_tampering_detected(tmp_path, structures):
    p = tmp_path / "s.extxyz.gz"
    qmof.save_structures(structures[:2], p)
    assert [a.info["identifier"] for a in qmof.load_structures(p)] == \
        [a.info["identifier"] for a in structures[:2]]
    import gzip
    p.write_bytes(gzip.compress(b"1\n\nH 0 0 0\n"))
    with pytest.raises(ValueError, match="hash mismatch"):
        qmof.load_structures(p)


def test_sample_indices_reproducible():
    a, b = sample_indices(100, 10, 7), sample_indices(100, 10, 7)
    assert a == b and len(set(a)) == 10 and a == sorted(a)
    assert sample_indices(5, 10, 0) == [0, 1, 2, 3, 4]


def test_reference_errors_with_emt():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.info = {"identifier": "cu", "energy_pbe_d3_eV": 0.0}
    r = reference_errors(a, EMT())
    assert r["force_rms"] < 1e-8                       # perfect crystal: zero forces
    assert r["energy_diff_per_atom"] == pytest.approx(r["energy"] / len(a))
    assert r["pressure_GPa"] is not None
    a.positions[0] += 0.1
    assert reference_errors(a, EMT())["force_max"] > 0.1


def test_relax_with_cell_recovers_emt_lattice():
    a = bulk("Cu", "fcc", a=3.7, cubic=True)           # strained start
    a.info = {"identifier": "cu"}
    relaxed, r = relax_with_cell(a, EMT(), fmax=1e-3, steps=2000)
    assert r["converged"]
    assert r["volume_change"] < -0.02                  # EMT Cu minimum is ~3.59 A
    assert relaxed.cell.cellpar()[0] == pytest.approx(3.59, abs=0.02)


def test_energy_consistency_removes_per_element_offsets():
    comps = [{"Zn": 1, "C": 4}, {"Zn": 2, "C": 3}, {"Zn": 1, "C": 8}, {"Zn": 3, "C": 1}]
    ref = np.array([-10.0, -20.0, -30.0, -5.0])
    mu = {"Zn": 0.3, "C": -0.1}
    pred = ref + np.array([c["Zn"] * mu["Zn"] + c["C"] * mu["C"] for c in comps])
    r = energy_consistency(pred, ref, comps, ["C", "Zn"])
    assert r["residual_rmse_per_atom"] < 1e-12
    assert r["offsets_eV"]["Zn"] == pytest.approx(0.3)
    assert r["raw_std_per_atom"] > 0.01
