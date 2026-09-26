"""SIESTA reference data for the molecules of the target applications (stage 17).

    python scripts/molecule_refs.py [--out data/training/molecules_siesta]

Water splitting and CO2 capture/splitting need the molecules themselves to
be right, not only the frameworks: fit 01 (MOF data only) shortened the CO2
C=O bond by 7 %. Each molecule is relaxed with SIESTA (stage-16 settings,
Gamma point, 8 A vacuum), then scanned around the SIESTA minimum:

    CO2   symmetric stretch, one-bond stretch (towards CO + O), bend
    H2O   one O-H stretch (towards OH + H), bend
    CH4, H2, CO, NH3, HCOOH (carboxylic acid, model of the linkers): minimum
    and one bond stretch

Every configuration keeps SIESTA energy (kcal/mol) and forces
(kcal/mol/A); the relaxed geometry is the reference of its family. The
scans stay in the closed-shell region (PBE here is not spin-polarized, so
full dissociation to radicals is not attempted). Local only, sized with
resources.plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

from ga_reaxff import siesta
from ga_reaxff.dataset import EV_TO_KCALMOL, TrainingSet, _tag
from ga_reaxff.resources import plan

VACUUM = 8.0
SETTINGS = siesta.SiestaSettings(kgrid_cutoff_ang=0.0)


def _stretch(a: Atoms, i: int, j: int, d: float) -> Atoms:
    """Move atom j (and nothing else) so that |r_ij| = d."""
    b = a.copy()
    v = b.positions[j] - b.positions[i]
    b.positions[j] = b.positions[i] + v / np.linalg.norm(v) * d
    return b


def _sym_stretch(a: Atoms, center: int, ends, d: float) -> Atoms:
    b = a.copy()
    for j in ends:
        v = b.positions[j] - b.positions[center]
        b.positions[j] = b.positions[center] + v / np.linalg.norm(v) * d
    return b


def _bend(a: Atoms, i: int, center: int, j: int, angle: float) -> Atoms:
    """Rotate atom j about `center` so that angle i-center-j = `angle` (degrees).

    Works for linear molecules too (ASE's set_angle needs a non-zero normal):
    the rotation axis is any vector perpendicular to the i-center bond.
    """
    b = a.copy()
    u = b.positions[i] - b.positions[center]
    v = b.positions[j] - b.positions[center]
    n = np.cross(u, v)
    if np.linalg.norm(n) < 1e-8:
        trial = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 * np.linalg.norm(u) else np.array([0.0, 1.0, 0.0])
        n = np.cross(u, trial)
    n /= np.linalg.norm(n)
    uu = u / np.linalg.norm(u)
    w = np.cross(n, uu)                       # in-plane, perpendicular to u
    t = np.radians(angle)
    b.positions[j] = b.positions[center] + np.linalg.norm(v) * (np.cos(t) * uu + np.sin(t) * w)
    return b


def scans(name: str, a: Atoms) -> list[tuple[str, Atoms]]:
    out = []
    if name == "CO2":            # C index 0, O 1, 2
        d0 = a.get_distance(0, 1)
        out += [(f"sym{dd:+.2f}", _sym_stretch(a, 0, [1, 2], d0 + dd)) for dd in (-0.08, -0.04, 0.04, 0.08, 0.16)]
        out += [(f"one{dd:+.2f}", _stretch(a, 0, 1, d0 + dd)) for dd in (0.1, 0.2, 0.4, 0.6)]
        out += [(f"bend{t:.0f}", _bend(a, 1, 0, 2, t)) for t in (170.0, 160.0, 150.0, 140.0)]
    elif name == "H2O":          # O 0, H 1, 2
        d0, t0 = a.get_distance(0, 1), a.get_angle(1, 0, 2)
        out += [(f"oh{dd:+.2f}", _stretch(a, 0, 1, d0 + dd)) for dd in (-0.08, 0.08, 0.2, 0.4, 0.7)]
        out += [(f"bend{t:.0f}", _bend(a, 1, 0, 2, t)) for t in (t0 - 15, t0 - 7, t0 + 7, t0 + 15)]
    else:                        # one bond stretch between atoms 0 and 1
        d0 = a.get_distance(0, 1)
        out += [(f"b01{dd:+.2f}", _stretch(a, 0, 1, d0 + dd)) for dd in (-0.06, 0.06, 0.15, 0.3)]
    return out


def main(argv=None):  # pragma: no cover - runs SIESTA
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/training/molecules_siesta")
    ap.add_argument("--molecules", default="CO2,H2O,CH4,H2,CO,NH3,HCOOH")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    configs, log = [], []
    for name in a.molecules.split(","):
        p = plan(mem_per_process_gb=1.0, max_processes=8)
        if p.processes < 1:
            raise SystemExit(f"machine busy, not starting: {p.reason}")
        m = molecule(name, vacuum=VACUUM, pbc=True)
        rdir = Path("runs/molecule_refs") / name / "relax"
        siesta.run(m, rdir, SETTINGS, processes=min(4, p.processes), label="m",
                   relax={"cell": False, "fmax": 0.005, "steps": 300}, resource_plan=p.as_dict())
        eq = siesta.read_final_structure(rdir, "m", m)
        items = [("eq", eq)] + scans(name, eq)
        for tag, atoms in items:
            d = Path("runs/molecule_refs") / name / tag
            r = siesta.run(atoms, d, SETTINGS, processes=min(4, p.processes), label="m",
                           resource_plan=p.as_dict())
            if not (r["normal_exit"] and r["scf_converged"]):
                log.append({"molecule": name, "config": tag, "status": "failed"})
                print(name, tag, "FAILED", flush=True)
                continue
            c = atoms.copy()
            c.info = {}
            _tag(c, f"{name}_{tag}", f"mol_{name}", is_ref=(tag == "eq"))
            c.info["ref_energy"] = r["energy_total"] * EV_TO_KCALMOL
            c.arrays["ref_forces"] = np.array(r["forces"]) * EV_TO_KCALMOL
            configs.append(c)
            log.append({"molecule": name, "config": tag, "energy_eV": r["energy_total"],
                        "force_max_eV_A": float(np.linalg.norm(r["forces"], axis=1).max()),
                        "wall_time_s": r["wall_time_s"]})
            print(name, tag, f"E {r['energy_total']:.4f} eV", flush=True)
    ts = TrainingSet(configs)
    path = ts.write(out / "molecules.extxyz")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "molecules.extxyz.sha256").write_text(f"{digest}  molecules.extxyz\n")
    (out / "manifest.json").write_text(json.dumps(
        {"settings": SETTINGS.as_dict(), "vacuum_A": VACUUM, "n_configs": len(ts),
         "pseudopotentials": {el: siesta.pseudo_sha256(el, SETTINGS.pseudo_dirs())
                              for el in sorted({s for c in configs for s in c.get_chemical_symbols()})},
         "runs": log}, indent=1) + "\n")
    print(f"{len(ts)} configurations -> {path}")


if __name__ == "__main__":
    main()
