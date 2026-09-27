"""Per-MOF validation of the *initial* Zn + C/H/N/O force field (before any fit).

    python scripts/baseline_validation.py [--n 200] [--max-atoms 150] [--procs 12]
    python scripts/baseline_validation.py --ffield runs/<fit>/ffield.best --out docs/<fit>_validation

Initial force field = data/ffields/base/ffield.Zn-FC + placeholder C-Zn and
N-Zn entries copied from O-Zn (ffield.add_placeholder_pairs). Every MOF of a
reproducible sample (seed 2026, <= max-atoms) is relaxed with LAMMPS
(positions + triclinic cell, zero pressure) from its QMOF PBE-D3(BJ)
structure and scored with validate.compare. The sample contains the 30 MOFs
of the teacher check (docs/teacher_check) so both can be compared directly.

Outputs in docs/baseline_validation/: scorecards.jsonl, summary.json,
fig_baseline.{png,svg}; the initial force field is written to
data/ffields/init/ffield.Zn-FC.init.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ga_reaxff import qmof
from ga_reaxff.engine import relax
from ga_reaxff.ffield import ForceField, add_placeholder_pairs
from ga_reaxff.teacher import sample_indices
from ga_reaxff.validate import (COORDINATION_CUTOFF, MAX_DISPLACEMENT_RMS, MAX_VOLUME_CHANGE,
                                compare, coordination)

STRUCTURES = "data/qmof/structures_Zn-CHNO.extxyz.gz"
BASE = "data/ffields/base/ffield.Zn-FC"
INIT = "data/ffields/init/ffield.Zn-FC.init"
SEED = 2026
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

_FF = None


def _init_worker(path):
    global _FF
    _FF = ForceField.read(path)


def _one(atoms):
    t = time.time()
    try:
        r = relax(atoms, _FF, ftol=0.5, maxiter=3000, cell=True)
        rec = compare(r, atoms).as_dict()
        rec["error"] = None
    except Exception as e:   # LAMMPS failure = the force field cannot hold this MOF
        r = None
        rec = {"identifier": atoms.info.get("identifier"), "natoms": len(atoms),
               "passed": False, "error": str(e)[:200]}
    rec["seconds"] = time.time() - t
    rec["zn_n_bonded"] = any(len(v) for v in coordination(atoms, ligands=("N",)).values())
    return rec, r


def build_initial_ff() -> tuple[Path, list]:
    ff, added = add_placeholder_pairs(ForceField.read(BASE), [("C", "Zn"), ("N", "Zn")], ("O", "Zn"))
    ff.header = ("ga-reaxff initial: ffield.Zn-FC + placeholder C-Zn, N-Zn bond/off-diagonal "
                 "entries copied from O-Zn (starting point for the fit, not physical)")
    Path(INIT).parent.mkdir(parents=True, exist_ok=True)
    return ff.write(INIT), added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-atoms", type=int, default=150)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="docs/baseline_validation")
    ap.add_argument("--ffield", default=None,
                    help="force field to validate (default: build and use the initial one)")
    ap.add_argument("--label", default="Initial Zn force field")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from ga_reaxff.resources import plan
    p = plan(mem_per_process_gb=0.5, max_processes=a.procs)
    if p.processes < 1:
        raise SystemExit(f"machine busy, not starting: {p.reason}")
    a.procs = p.processes
    if a.ffield:
        init_path, added = Path(a.ffield), []
    else:
        init_path, added = build_initial_ff()
    structures = qmof.load_structures(STRUCTURES)
    # the teacher-check relaxation sample (same selection as scripts/teacher_check.py)
    sp = [structures[i] for i in sample_indices(len(structures), 300, SEED)]
    small_sp = [s for s in sp if len(s) <= 150]
    teacher_ids = {small_sp[i].info["identifier"] for i in sample_indices(len(small_sp), 30, SEED)}
    small = [s for s in structures if len(s) <= a.max_atoms]
    ids = {small[i].info["identifier"] for i in sample_indices(len(small), a.n, SEED + 1)}
    sample = [s for s in structures if s.info["identifier"] in ids | teacher_ids]

    t = time.time()
    with Pool(a.procs, initializer=_init_worker, initargs=(str(init_path),)) as pool:
        results = pool.map(_one, sample, chunksize=1)
    wall = time.time() - t
    cards = [c for c, _ in results]
    # relaxed structures kept for re-analysis (runs/ is not versioned)
    from ase.io import write
    run_dir = Path("runs") / Path(a.out).name
    run_dir.mkdir(parents=True, exist_ok=True)
    write(run_dir / "relaxed.extxyz", [r for _, r in results if r is not None], format="extxyz")
    for c in cards:
        c["in_teacher_sample"] = c["identifier"] in teacher_ids
    (out / "scorecards.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n"
                                                  for c in cards))
    ok = [c for c in cards if c["error"] is None]
    v = np.array([c["volume_change"] for c in ok])
    summary = {
        "initial_ffield": str(init_path),
        "initial_ffield_sha256": hashlib.sha256(Path(init_path).read_bytes()).hexdigest(),
        "placeholders": ["{}:{}".format(b, "-".join(p)) for b, p in added],
        "structures_sha256": Path(STRUCTURES + ".sha256").read_text().split()[0],
        "thresholds": {"max_volume_change": MAX_VOLUME_CHANGE,
                       "max_displacement_rms_A": MAX_DISPLACEMENT_RMS,
                       "coordination_cutoff_A": COORDINATION_CUTOFF},
        "n": len(cards), "n_failed_lammps": len(cards) - len(ok),
        "n_passed": sum(c["passed"] for c in cards),
        "n_ligands_lost_any": sum(c.get("ligands_lost", 0) > 0 for c in ok),
        "volume_change_median": float(np.median(v)),
        "volume_change_abs_median": float(np.median(np.abs(v))),
        "volume_change_p10_p90": [float(np.percentile(v, 10)), float(np.percentile(v, 90))],
        "displacement_rms_median": float(np.median([c["displacement_rms"] for c in ok])),
        "by_group": {},
        "wall_time_s": wall, "procs": a.procs, "resource_plan": p.as_dict(),
        "cpu_seconds_per_mof_median": float(np.median([c["seconds"] for c in cards])),
    }
    for name, grp in (("Zn-N bonded", [c for c in ok if c["zn_n_bonded"]]),
                      ("Zn-O only", [c for c in ok if not c["zn_n_bonded"]])):
        g = {"n": len(grp), "n_passed": sum(c["passed"] for c in grp),
             "volume_change_median": float(np.median([c["volume_change"] for c in grp])),
             "bond_change_median": {}}
        for pair in sorted({k for c in grp for k in c["bonds"]}):
            rel = [c["bonds"][pair]["relaxed"] / c["bonds"][pair]["ref"] - 1
                   for c in grp if pair in c["bonds"]]
            ref = [c["bonds"][pair]["ref"] for c in grp if pair in c["bonds"]]
            g["bond_change_median"][pair] = {"n_mofs": len(rel), "ref_median_A": float(np.median(ref)),
                                             "relative_change_median": float(np.median(rel))}
        summary["by_group"][name] = g
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lost = np.array([c["ligands_lost"] > 0 for c in ok])
    ax.hist([100 * v[~lost], 100 * v[lost]], bins=40, stacked=True, color=[BLUE, ORANGE],
            edgecolor="white", linewidth=0.8,
            label=["metal coordination kept", "metal lost at least one ligand"])
    ax.axvspan(-100 * MAX_VOLUME_CHANGE, 100 * MAX_VOLUME_CHANGE, color=GRID, zorder=0,
               label=f"acceptance band (|dV| < {100 * MAX_VOLUME_CHANGE:.0f} %)")
    ax.set_xlabel("volume change after ReaxFF relaxation (%)", color=INK)
    ax.set_ylabel("MOFs", color=INK)
    ax.set_title(f"{a.label} on {len(ok)} Zn MOFs",
                 loc="left", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_baseline.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
