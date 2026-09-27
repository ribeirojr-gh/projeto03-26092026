"""Calibrate SIESTA (PBE-D3(BJ), PSML) against the QMOF (VASP PBE-D3(BJ)) structures.

    python scripts/siesta_calibration.py sweep   [--mof qmof-69d6aa4]
    python scripts/siesta_calibration.py check   --settings <json>   [--n 6]
    python scripts/siesta_calibration.py relax   [--mof qmof-69d6aa4]
    python scripts/siesta_calibration.py molecules

sweep : one-parameter-at-a-time convergence study on one small Zn MOF
        (basis, PAO energy shift, mesh cutoff, k-grid cutoff). At the QMOF
        geometry the VASP forces and stress are ~0, so SIESTA's residual
        forces and pressure there measure the disagreement between the two
        DFT setups, and their change with the settings shows convergence.
check : chosen settings on several Zn MOFs (forces, pressure, Hirshfeld vs
        DDEC6 charges, cost per atom).
molecules : CH4, H2O, CO2 relaxed with each pseudopotential choice.
relax : fixed-cell SIESTA relaxation from the QMOF structure for each
        pseudopotential choice; bond lengths compared with VASP (QMOF).

Runs locally only (project rule), sized with resources.plan. Outputs:
runs/siesta_calibration/<case>/ (full SIESTA directories, not versioned) and
docs/siesta_calibration/<step>.jsonl (one record per calculation).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ga_reaxff import qmof, siesta
from ga_reaxff.resources import plan
from ga_reaxff.teacher import sample_indices

STRUCTURES = "data/qmof/structures_Zn-CHNO.extxyz.gz"
EV_A3_TO_GPA = 160.21766
BASE = siesta.SiestaSettings()
SWEEP = [
    ("basis", ["DZP", "TZP"]),   # generic TZ2P stops (split norm too small)
    ("energy_shift_ry", [0.01, 0.005, 0.002]),
    ("mesh_cutoff_ry", [200.0, 300.0, 500.0]),
    ("kgrid_cutoff_ang", [10.0, 15.0]),
    ("pseudo_families", [{"default": "ATOM-TABLE", "Zn": "DOJO-PSML"},
                         {"default": "DOJO-PSML"}, {"default": "ATOM-TABLE"}]),
]


def metrics(res: dict, atoms) -> dict:
    if not res.get("forces") or not res.get("normal_exit"):
        out = Path(res["run_dir"]) / "mof.out" if res.get("run_dir") else None
        tail = out.read_text().strip().splitlines()[-12:] if out and out.exists() else []
        reason = next((ln for ln in tail if "too small" in ln or "Error" in ln), "no forces")
        return {"identifier": atoms.info["identifier"], "natoms": len(atoms), "failed": True,
                "reason": reason.strip(), "wall_time_s": res.get("wall_time_s")}
    f = np.array(res["forces"])
    st = np.array(res["stress"])
    q = np.array(res.get("hirshfeld_charges") or [np.nan] * len(atoms))
    return {
        "identifier": atoms.info["identifier"], "natoms": len(atoms),
        "force_rms": float(np.sqrt((f ** 2).mean())),
        "force_max": float(np.linalg.norm(f, axis=1).max()),
        "pressure_GPa": float(-np.trace(st) / 3 * EV_A3_TO_GPA),
        "energy_total": res["energy_total"], "energy_d3": res["energy_d3"],
        "charge_corr_ddec6": float(np.corrcoef(q, atoms.arrays["pbe_ddec_charge"])[0, 1]),
        "zn_hirshfeld_mean": float(np.mean(q[np.array(atoms.get_chemical_symbols()) == "Zn"])),
        "zn_ddec6_mean": float(np.mean(atoms.arrays["pbe_ddec_charge"][
            np.array(atoms.get_chemical_symbols()) == "Zn"])),
        "wall_time_s": res["wall_time_s"], "processes": res["processes"],
        "scf_converged": res["scf_converged"], "normal_exit": res["normal_exit"],
    }


def _append(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def sweep(mof_id: str, out: Path):
    structures = qmof.load_structures(STRUCTURES)
    mof = next(a for a in structures if a.info["identifier"] == mof_id)
    cases = [("base", BASE)]
    for key, values in SWEEP:
        cases += [(f"{key}={json.dumps(v, sort_keys=True) if isinstance(v, dict) else v}",
                   replace(BASE, **{key: v})) for v in values if v != getattr(BASE, key)]
    done = set()
    if (out / "sweep.jsonl").exists():
        done = {json.loads(ln)["case"] for ln in (out / "sweep.jsonl").read_text().splitlines()}
    for name, st in cases:
        if name in done:
            continue
        p = plan(mem_per_process_gb=1.5, max_processes=8)
        if p.processes < 1:
            raise SystemExit(f"machine busy, not starting: {p.reason}")
        run_dir = f"runs/siesta_calibration/sweep/{mof_id}/{name}"
        res = siesta.run(mof, run_dir, st, processes=p.processes, label="mof",
                         resource_plan=p.as_dict())
        res["run_dir"] = run_dir
        rec = {"case": name, "settings": st.as_dict(), **metrics(res, mof)}
        _append(out / "sweep.jsonl", rec)
        if rec.get("failed"):
            print(f"{name:24s} FAILED: {rec['reason']}", flush=True)
        else:
            print(f"{name:24s} Frms {rec['force_rms']:.3f} P {rec['pressure_GPa']:+.2f} GPa "
                  f"t {rec['wall_time_s']:.0f} s", flush=True)


def check(settings: siesta.SiestaSettings, n: int, max_atoms: int, out: Path):
    structures = qmof.load_structures(STRUCTURES)
    small = [s for s in structures if len(s) <= max_atoms]
    sample = [small[i] for i in sample_indices(len(small), n, 2028)]
    for mof in sample:
        p = plan(mem_per_process_gb=2.0, max_processes=8)
        if p.processes < 1:
            raise SystemExit(f"machine busy, not starting: {p.reason}")
        mid = mof.info["identifier"]
        res = siesta.run(mof, f"runs/siesta_calibration/check/{mid}", settings,
                         processes=p.processes, label="mof", resource_plan=p.as_dict())
        rec = {"settings": settings.as_dict(), **metrics(res, mof)}
        _append(out / "check.jsonl", rec)
        print(f"{mid} {len(mof):4d} atoms Frms {rec['force_rms']:.3f} "
              f"P {rec['pressure_GPa']:+.2f} GPa t {rec['wall_time_s']:.0f} s", flush=True)


PSEUDO_CHOICES = {"mixed (ATOM-TABLE + Zn DOJO)": {"default": "ATOM-TABLE", "Zn": "DOJO-PSML"},
                  "DOJO-PSML": {"default": "DOJO-PSML"},
                  "ATOM-TABLE": {"default": "ATOM-TABLE"}}


def relax_bonds(mof_id: str, out: Path):
    from ga_reaxff.validate import compare
    structures = qmof.load_structures(STRUCTURES)
    mof = next(a for a in structures if a.info["identifier"] == mof_id)
    for name, fam in PSEUDO_CHOICES.items():
        p = plan(mem_per_process_gb=1.0, max_processes=8)
        if p.processes < 1:
            raise SystemExit(f"machine busy, not starting: {p.reason}")
        st = replace(BASE, pseudo_families=fam)
        d = f"runs/siesta_calibration/relax/{mof_id}/{fam['default']}-{fam.get('Zn', fam['default'])}"
        res = siesta.run(mof, d, st, processes=p.processes, label="mof",
                         relax={"cell": False, "fmax": 0.02, "steps": 300},
                         resource_plan=p.as_dict())
        c = compare(siesta.read_final_structure(d, "mof", mof), mof)
        rec = {"pseudos": name, "settings": st.as_dict(), "identifier": mof_id,
               "wall_time_s": res["wall_time_s"], "displacement_rms": c.displacement_rms,
               "pressure_after_GPa": float(-np.trace(np.array(res["stress"])) / 3 * EV_A3_TO_GPA),
               "bonds": {k: {**v, "relative_change": v["relaxed"] / v["ref"] - 1}
                         for k, v in c.bonds.items()}}
        _append(out / "relax_bonds.jsonl", rec)
        print(name, {k: f"{100 * v['relative_change']:+.2f}%" for k, v in rec["bonds"].items()},
              flush=True)


def molecules(out: Path):
    from ase.build import molecule
    for name in ("CH4", "H2O", "CO2"):
        for label, fam in PSEUDO_CHOICES.items():
            st = replace(BASE, pseudo_families=fam, kgrid_cutoff_ang=0.0)
            d = f"runs/siesta_calibration/molecules_final/{name}/{fam['default']}"
            siesta.run(molecule(name, vacuum=6.0, pbc=True), d, st, processes=4, label="m",
                       relax={"cell": False, "fmax": 0.005, "steps": 200})
            fin = siesta.read_final_structure(d, "m", molecule(name, vacuum=6.0, pbc=True))
            rec = {"molecule": name, "pseudos": label, "bond": float(fin.get_distance(0, 1))}
            if name == "H2O":
                rec["angle"] = float(fin.get_angle(1, 0, 2))
            _append(out / "molecules.jsonl", rec)
            print(rec, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sweep", "check", "relax", "molecules"])
    ap.add_argument("--mof", default="qmof-69d6aa4")
    ap.add_argument("--settings", default="{}", help="JSON overrides of SiestaSettings")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-atoms", type=int, default=100)
    ap.add_argument("--out", default="docs/siesta_calibration")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.step == "sweep":
        sweep(a.mof, out)
    elif a.step == "relax":
        relax_bonds(a.mof, out)
    elif a.step == "molecules":
        molecules(out)
    else:
        check(replace(BASE, **json.loads(a.settings)), a.n, a.max_atoms, out)


if __name__ == "__main__":
    main()
