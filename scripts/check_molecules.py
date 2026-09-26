"""Relax the application molecules with a ReaxFF force field and compare.

    python scripts/check_molecules.py <ffield> [--out docs/<fit>_validation/molecules.json]

Reference geometries: SIESTA (stage-16 settings, the level used in the
training data) and experiment (NIST CCCBDB, as in stage 12). Molecules the
force field must keep right for water splitting and CO2 capture/splitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.build import molecule

from ga_reaxff.engine import relax
from ga_reaxff.ffield import ForceField

EXP = {"CO2": 1.160, "H2O": 0.958, "CH4": 1.087, "CO": 1.128, "H2": 0.741, "NH3": 1.012}
SIESTA_RUNS = Path("runs/molecule_refs")


def siesta_reference(name: str):
    """Bond 0-1 of the SIESTA-relaxed molecule, from the committed training set."""
    from ga_reaxff.dataset import TrainingSet
    ts = TrainingSet.read("data/training/molecules_siesta/molecules.extxyz")
    eq = next((c for c in ts.configs if c.info["name"] == f"{name}_eq"), None)
    return None if eq is None else float(eq.get_distance(0, 1, mic=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ffield")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ff = ForceField.read(a.ffield)
    rows = []
    for name, exp in EXP.items():
        r = relax(molecule(name, vacuum=8.0, pbc=True), ff, ftol=1e-3, maxiter=5000)
        d = float(r.get_distance(0, 1))
        ref = siesta_reference(name)
        row = {"molecule": name, "bond": f"{r[0].symbol}-{r[1].symbol}", "reaxff": d,
               "siesta": ref, "exp": exp, "error_vs_siesta": None if ref is None else d / ref - 1,
               "error_vs_exp": d / exp - 1}
        if name == "H2O":
            row["angle_reaxff"] = float(r.get_angle(1, 0, 2))
        rows.append(row)
        print(f"{name:4s} {row['bond']:4s} ReaxFF {d:.3f}  SIESTA {ref if ref is None else round(ref, 3)}  "
              f"exp {exp:.3f}  ({100 * row['error_vs_exp']:+.1f} % vs exp)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ffield": a.ffield, "molecules": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
