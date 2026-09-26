"""Fit the Zn + C/H/N/O force field (stage 17).

    python scripts/fit_zn.py build configs/zn_fit_01.toml     # GPU teacher labels (local)
    python scripts/fit_zn.py fit   configs/zn_fit_01.toml     # GA + Nelder-Mead (local CPU)

build : select the training MOFs (disjoint from the validation MOFs of the
        stage-14 baseline), make the equilibrium and strain configurations,
        label the strain energies with the MLIP teacher, and write
        data/training/<name>/train.extxyz (+ .sha256, selection.json).
fit   : GA from the start force field, then bounded Nelder-Mead; outputs in
        runs/<name>/ (manifest.json, history.jsonl, result.json, ffield.best).
        The number of LAMMPS worker processes comes from resources.plan.

Validation of the result: scripts/baseline_validation.py --ffield runs/<name>/ffield.best
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path

import numpy as np

from ga_reaxff import qmof
from ga_reaxff.dataset import TrainingSet
from ga_reaxff.families import Family
from ga_reaxff.mofdata import (bond_scan_family, equilibrium_config, first_contact,
                               label_energies, strain_family)
from ga_reaxff.teacher import sample_indices
from ga_reaxff.validate import coordination


def _load(cfg_path):
    cfg = tomllib.loads(Path(cfg_path).read_text())
    cfg["_path"] = str(cfg_path)
    return cfg


def select_training(cfg: dict):
    t = cfg["training"]
    structures = qmof.load_structures(cfg["structures"])
    excluded = {json.loads(ln)["identifier"]
                for ln in Path(cfg["exclude_scorecards"]).read_text().splitlines()}
    pool = [s for s in structures if len(s) <= t["max_atoms"] and s.info["identifier"] not in excluded]
    zn_n = [s for s in pool if any(len(v) for v in coordination(s, ligands=("N",)).values())]
    zn_o = [s for s in pool if s not in zn_n]
    pick_n = [zn_n[i] for i in sample_indices(len(zn_n), t["n_zn_n"], cfg["seed"])]
    pick_o = [zn_o[i] for i in sample_indices(len(zn_o), t["n_zn_o_only"], cfg["seed"] + 1)]
    train = pick_n + pick_o
    k = t["n_strain"]
    strained = pick_n[: k - k // 2] + pick_o[: k // 2]
    return train, strained, {"pool_size": len(pool), "pool_zn_n": len(zn_n), "pool_zn_o_only": len(zn_o)}


def build(cfg: dict):  # pragma: no cover - needs the GPU teacher
    from ga_reaxff.resources import snapshot
    from ga_reaxff.teacher import load_mace
    t = cfg["training"]
    train, strained, pool = select_training(cfg)
    configs = [equilibrium_config(s) for s in train]
    strain_cfgs = []
    for s in strained:
        strain_cfgs += strain_family(s, t["strains"])
    n_scans = 0
    scan_mofs = train if t.get("bond_scan_mofs") == "all" else strained
    for s in scan_mofs if t.get("bond_scans") else []:
        for lig in ("N", "O"):
            pair = first_contact(s, "Zn", lig)
            if pair:
                strain_cfgs += bond_scan_family(s, *pair, t["bond_scans"],
                                                w_e=t.get("bond_scan_w_e", 1.0))
                n_scans += 1
    snap = snapshot().as_dict()
    calc = load_mace(t["teacher_model"])
    t0 = time.time()
    label_energies(strain_cfgs, calc)
    t_label = time.time() - t0
    mol_cfgs, mol_sha = [], None
    if t.get("molecules"):
        mpath = Path(t["molecules"])
        mol_sha = mpath.with_suffix(".extxyz.sha256").read_text().split()[0] \
            if mpath.with_suffix(".extxyz.sha256").exists() else \
            Path(str(mpath) + ".sha256").read_text().split()[0]
        if hashlib.sha256(mpath.read_bytes()).hexdigest() != mol_sha:
            raise SystemExit("molecule reference set does not match its hash")
        mol_cfgs = TrainingSet.read(mpath).configs
        for c in mol_cfgs:
            c.info["w_e"] = t.get("molecule_w_e", 1.0)
            c.info["w_f"] = t.get("molecule_w_f", 1.0)
    ts = TrainingSet(configs + strain_cfgs + mol_cfgs)
    out = Path("data/training") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = ts.write(out / "train.extxyz")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "train.extxyz.sha256").write_text(f"{digest}  train.extxyz\n")
    sel = {"config": cfg["_path"], "training_mofs": [s.info["identifier"] for s in train],
           "strain_mofs": [s.info["identifier"] for s in strained],
           "zn_n": [s.info["identifier"] for s in train[: t["n_zn_n"]]],
           "zn_o_only": [s.info["identifier"] for s in train[t["n_zn_n"]:]],
           "natoms": {s.info["identifier"]: len(s) for s in train}, **pool,
           "n_configs": len(ts), "n_bond_scans": n_scans, "n_molecule_configs": len(mol_cfgs),
           "molecules_sha256": mol_sha,
           "teacher": t["teacher_model"], "teacher_seconds": t_label,
           "machine": snap, "structures_sha256":
               Path(cfg["structures"] + ".sha256").read_text().split()[0]}
    (out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(json.dumps({k: v for k, v in sel.items() if k not in ("natoms", "machine")}, indent=1))


def fit(cfg: dict):  # pragma: no cover - long LAMMPS run
    from ga_reaxff.audit import RunRecorder
    from ga_reaxff.ffield import ForceField
    from ga_reaxff.fitness import Objective
    from ga_reaxff.ga import GAConfig, GeneticAlgorithm, local_refine
    from ga_reaxff.parameters import ParameterSpace
    from ga_reaxff.resources import plan

    out = Path("runs") / cfg["name"]
    rec = RunRecorder(out)
    data = Path("data/training") / cfg["name"] / "train.extxyz"
    expected = (data.parent / "train.extxyz.sha256").read_text().split()[0]
    if hashlib.sha256(data.read_bytes()).hexdigest() != expected:
        raise SystemExit("training set does not match its recorded hash")
    ts = TrainingSet.read(data)
    for c in ts.configs:        # extxyz stores ref_stress as an array
        if "ref_stress" in c.info:
            c.info["ref_stress"] = [float(x) for x in np.ravel(c.info["ref_stress"])]
    ff0 = ForceField.read(cfg["start_ffield"])
    space = ParameterSpace.from_config(ff0, cfg["parameters"])
    p = plan(cfg["run"]["mem_per_worker_gb"], cfg["run"]["max_workers"])
    if p.processes < 1:
        raise SystemExit(f"machine busy, not starting: {p.reason}")
    L = cfg["loss"]
    obj = Objective(ts, ff0, space, sigma_e=L["sigma_e"], sigma_f=L["sigma_f"],
                    force_weight=L["force_weight"], sigma_s=L["sigma_s"],
                    stress_weight=L["stress_weight"], n_workers=p.processes)
    ga_cfg = GAConfig(seed=cfg["seed"], **cfg["ga"])
    ga_cfg.inject = [space.encode(space.values_from(ff0))]
    rec.manifest(cfg, {"start_ffield": cfg["start_ffield"], "training_set": data},
                 extra={"parameter_space": space.to_records(), "ga_config": ga_cfg.as_dict(),
                        "n_configs": len(ts), "resource_plan": p.as_dict()})
    g0 = ga_cfg.inject[0]
    start = obj.result(g0)
    print(f"workers {p.processes} | start loss {start.loss:.4g} {start.loss_terms} | "
          f"F-RMSE {start.force_rmse:.2f} kcal/mol/A | S-RMSE {start.stress_rmse:.2f} GPa", flush=True)

    def on_gen(r):
        best = obj.result(np.array(r["best_genome"]))
        rec.log_generation({**{k: r[k] for k in ("generation", "best", "mean", "median", "std", "diversity")},
                            "loss_terms": best.loss_terms, "energy_rmse": best.energy_rmse,
                            "force_rmse": best.force_rmse, "stress_rmse": best.stress_rmse,
                            "n_objective_calls": obj.n_calls, "n_failed": obj.n_failed,
                            "best_params": dict(zip(space.keys, space.decode(r["best_genome"])))})
        print(f"gen {r['generation']:3d} best {r['best']:.4g} mean {r['mean']:.4g} "
              f"div {r['diversity']:.3f} calls {obj.n_calls} fail {obj.n_failed} "
              f"t {rec.elapsed():.0f} s", flush=True)

    t0 = time.time()
    ga_res = GeneticAlgorithm(ga_cfg).run(obj, len(space), callback=on_gen)
    t_ga = time.time() - t0
    t0 = time.time()
    g_best, f_best, loc = local_refine(obj, ga_res.best_genome, maxiter=cfg["refine"]["maxiter"])
    t_loc = time.time() - t0
    ff_best = space.apply(ff0, g_best)
    ff_best.header = f"ga-reaxff {cfg['name']}: fitted from {cfg['start_ffield']} (see manifest.json)"
    ff_best.write(out / "ffield.best")
    end = obj.result(g_best)
    v0, v1 = space.values_from(ff0), space.values_from(ff_best)
    g_final = space.encode(v1)
    table = [{"key": k, "lower": space.lower[i], "upper": space.upper[i], "start": v0[i],
              "final": v1[i], "at_bound": bool(g_final[i] < 0.01 or g_final[i] > 0.99)}
             for i, k in enumerate(space.keys)]
    result = {"parameters": table, "start": start.as_dict(), "final": end.as_dict(),
              "ga": {"best_loss": ga_res.best_loss, "generations": len(ga_res.history) - 1,
                     "time_s": t_ga},
              "local_refinement": {**loc, "time_s": t_loc},
              "objective_calls": obj.n_calls, "failed_evaluations": obj.n_failed,
              "workers": p.processes, "total_time_s": rec.elapsed()}
    rec.write_json("result.json", result)
    obj.close()
    print(f"final loss {end.loss:.4g} {end.loss_terms} | F-RMSE {end.force_rmse:.2f} | "
          f"S-RMSE {end.stress_rmse:.2f} | E-RMSE {end.energy_rmse:.2f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "fit"])
    ap.add_argument("config")
    a = ap.parse_args(argv)
    cfg = _load(a.config)
    {"build": build, "fit": fit}[a.step](cfg)


if __name__ == "__main__":
    main()
