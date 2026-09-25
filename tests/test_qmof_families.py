"""Stage 11 tests: QMOF snapshot integrity and chemical-family analysis."""
import gzip
import hashlib
from pathlib import Path

import pytest

from ga_reaxff import qmof
from ga_reaxff.families import (Family, coverage, elements, greedy_cover, node_types,
                                split_elements, split_smiles_list)

DATA = Path(__file__).parents[1] / "data" / "qmof"


# --- flattening of MPContribs records ---------------------------------------

def test_flatten_reduces_quantities_to_values():
    data = {"natoms": {"display": "58", "value": 58.0, "unit": ""},
            "mofid": {"smilesNodes": "[Zn]"},
            "outputsDFT": {"PBE": {"energy": {"total": {"display": "-1 eV", "value": -1.0,
                                                        "unit": "eV"}}}}}
    assert qmof.flatten(data) == {"natoms": 58.0, "mofid.smilesNodes": "[Zn]",
                                  "outputsDFT.PBE.energy.total": -1.0}


def test_to_row_drops_excluded_subtrees():
    c = {"identifier": "qmof-x", "id": "abc", "formula": "ZnC",
         "data": {"chemsys": "C-Zn",
                  "outputsGCMC": {"CO2": {"henry": {"display": "1", "value": 1.0}}},
                  "outputsDFT": {"HSE06": {"Eg": {"display": "3", "value": 3.0}},
                                 "PBE": {"Eg": {"display": "2", "value": 2.0}}}}}
    row = qmof.to_row(c)
    assert row == {"identifier": "qmof-x", "contribution_id": "abc", "formula": "ZnC",
                   "chemsys": "C-Zn", "outputsDFT.PBE.Eg": 2.0}


# --- snapshot I/O --------------------------------------------------------------

def test_snapshot_round_trip_is_deterministic(tmp_path):
    rows = [{"identifier": "b", "x": 1}, {"identifier": "a", "x": 2}]
    h1 = qmof.save_snapshot(rows, {"source": "test"}, tmp_path / "s1")
    h2 = qmof.save_snapshot(rows[::-1], {"source": "test"}, tmp_path / "s2")
    assert h1 == h2, "hash must not depend on input order"
    assert (tmp_path / "s1" / qmof.SNAPSHOT).read_bytes() == \
        (tmp_path / "s2" / qmof.SNAPSHOT).read_bytes(), "gzip output must be byte-identical"
    assert [r["identifier"] for r in qmof.load_snapshot(tmp_path / "s1")] == ["a", "b"]


def test_snapshot_tampering_is_detected(tmp_path):
    qmof.save_snapshot([{"identifier": "a"}], {}, tmp_path)
    (tmp_path / qmof.SNAPSHOT).write_bytes(gzip.compress(b'{"identifier": "z"}\n'))
    with pytest.raises(ValueError, match="hash mismatch"):
        qmof.load_snapshot(tmp_path)


def test_committed_snapshot_integrity():
    """The committed QMOF snapshot matches its recorded hash and size."""
    rows = qmof.load_snapshot(DATA)          # verifies SHA-256
    assert len(rows) == 20375
    assert len({r["identifier"] for r in rows}) == len(rows)
    assert all(r.get("chemsys") and r.get("natoms") for r in rows)


# --- families -------------------------------------------------------------------

def test_split_elements_metals_vs_nonmetals():
    assert split_elements(elements("C-H-O-Zn")) == (("Zn",), ("C", "H", "O"))
    # metalloids go with the non-metals
    assert split_elements({"B", "Si", "Cu", "C"}) == (("Cu",), ("B", "C", "Si"))


def test_family_key():
    assert Family.from_chemsys("C-H-N-O-Zn").key == "Zn|C-H-N-O"
    assert Family.from_chemsys("C-Co-H-O-Zn").metals == ("Co", "Zn")
    assert Family.from_chemsys("B-C-H-O").key == "-|B-C-H-O"


def test_coverage_is_subset_relation():
    sets = [elements(s) for s in ("C-H-O-Zn", "C-H-N-O-Zn", "C-Cu-H-O", "C-H-O")]
    assert coverage({"C", "H", "O", "Zn"}, sets) == [0, 3]
    assert coverage({"C", "H", "N", "O", "Zn"}, sets) == [0, 1, 3]


def test_greedy_cover_on_toy_data():
    sets = [elements(s) for s in ["C-H-O-Zn"] * 3 + ["C-Cu-H-O"] * 2 + ["C-Cd-H-O"]]
    picks = greedy_cover(sets, [{"C", "H", "O", m} for m in ("Zn", "Cu", "Cd")])
    assert [p[0] for p in picks] == [("C", "H", "O", "Zn"), ("C", "Cu", "H", "O"),
                                     ("C", "Cd", "H", "O")]
    assert [p[2] for p in picks] == [3, 5, 6]
    assert greedy_cover(sets, [{"C", "H", "O", "Zn"}], n_max=1)[0][1] == 3


def test_greedy_cover_cumulative_is_monotonic_on_qmof():
    rows = qmof.load_snapshot(DATA)
    fams = [Family.from_chemsys(r["chemsys"]) for r in rows]
    metals = {f.metals[0] for f in fams if len(f.metals) == 1}
    picks = greedy_cover([f.elements for f in fams],
                         [{"C", "H", "N", "O", m} for m in metals], n_max=10)
    cum = [p[2] for p in picks]
    assert cum == sorted(cum) and all(p[1] > 0 for p in picks)
    assert picks[0][0] == ("C", "H", "N", "O", "Zn")   # largest family in QMOF


def test_node_types_and_smiles_split():
    assert split_smiles_list("[Zn],[O]([Zn])[Zn]") == ["[Zn]", "[O]([Zn])[Zn]"]
    assert split_smiles_list(None) == []
    rows = [{"chemsys": "C-H-O-Zn", "mofid.smilesNodes": "[Zn],[Zn]"},
            {"chemsys": "C-H-O-Zn", "mofid.smilesNodes": "[Zn]"},
            {"chemsys": "C-Co-H-O-Zn", "mofid.smilesNodes": "[Co]"}]
    assert node_types(rows, "Zn") == [("[Zn]", 2)]


def test_node_types_ties_are_deterministic():
    rows = [{"chemsys": "C-Mo-O", "mofid.smilesNodes": "[O]"},
            {"chemsys": "C-Mo-O", "mofid.smilesNodes": "[Mo]"}]
    assert node_types(rows, "Mo") == [("[Mo]", 1), ("[O]", 1)]
