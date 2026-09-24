"""End-to-end workflow: training set -> GA -> local refinement -> validation.

Two modes (config key `reference.mode`):
  * "synthetic": reference data are produced with a known force field (here
    Chenoweth 2008 C/H/O). The starting force field is that same file with
    selected parameters perturbed. Because the truth is known, this is a
    *recovery benchmark* that validates the whole machinery before any DFT
    data is used.
  * "file": reference energies/forces are read from an extxyz training set
    (e.g. converted from VASP / Quantum ESPRESSO / SIESTA with ASE). This is
    the production mode for real parametrizations.
"""
from __future__ import annotations

import time
import tomllib
from pathlib import Path

import numpy as np

from .audit import RunRecorder
from .dataset import TrainingSet, displacement_scan, rattle_series, strain_series
from .engine import graphene_lattice_constant, label_with_forcefield, relax
from .ffield import ForceField
from .fitness import Objective
from .ga import GAConfig, GeneticAlgorithm, local_refine
from .parameters import ParameterSpace, perturb
from .structures import graphene_oxide_model

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


# ------------------------------------------------------------ training data
def go_configurations(relaxed, spec: dict, seed: int, tag: str = "") -> list:
    """Geometric families around a relaxed GO model (see docs/methodology.md)."""
    syms = relaxed.get_chemical_symbols()
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    h_idx = [i for i, s in enumerate(syms) if s == "H"]
    epox_o, oh_o, oh_h = o_idx[0], o_idx[1], h_idx[0]           # builder order: epoxide, OH(top), OH(bottom)
    c_oh = int(np.argmin(np.where(np.array(syms) == "C",
                                  relaxed.get_distances(oh_o, range(len(relaxed)), mic=True), np.inf)))
    co_vec = relaxed.get_distance(c_oh, oh_o, mic=True, vector=True)

    confs = []
    confs += strain_series(relaxed, spec["strains"], family=f"strain{tag}")
    confs += displacement_scan(relaxed, [epox_o], [0, 0, 1], spec["epoxide_z"], family=f"epoxide_z{tag}")
    confs += displacement_scan(relaxed, [oh_o, oh_h], co_vec, spec["hydroxyl_stretch"],
                               family=f"hydroxyl_stretch{tag}")
    confs += displacement_scan(relaxed, [oh_h], [1, 0, 0], spec["hydroxyl_bend"], family=f"hydroxyl_bend{tag}")
    confs += rattle_series(relaxed, spec["n_rattle"], spec["rattle_stdev"], seed=seed, family=f"rattle{tag}")
    for c in confs:
        fam = c.info["family"]
        c.info["w_e"] = float(spec.get("weights", {}).get(fam.removesuffix(tag), 1.0))
    return confs


def build_go_training_sets(ff_ref: ForceField, cfg: dict, log=print):
    """Relax GO at the reference force field's lattice constant; build train + validation sets."""
    ts_cfg = cfg["training_set"]
    t = time.time()
    a_eq = graphene_lattice_constant(ff_ref)
    log(f"graphene lattice constant (reference ffield): a = {a_eq:.4f} A")
    go = relax(graphene_oxide_model(a=a_eq), ff_ref, ftol=ts_cfg.get("relax_ftol", 1e-3))
    log(f"relaxed GO model ({len(go)} atoms) in {time.time() - t:.1f} s")
    train = go_configurations(go, ts_cfg["train"], seed=ts_cfg["seed"])
    valid = go_configurations(go, ts_cfg["validation"], seed=ts_cfg["seed"] + 1000, tag="_val")
    return go, a_eq, train, valid


# ---------------------------------------------------------------- main run
def run(cfg: dict, outdir: str | Path, log=print) -> dict:
    rec = RunRecorder(outdir)
    out = Path(outdir)
    t_start = time.time()

    ff_ref = ForceField.read(_resolve(cfg["forcefield"]["reference"]))
    keys = list(cfg["parameters"]["keys"])
    mode = cfg["reference"]["mode"]

    # ---- reference data
    if mode == "synthetic":
        go, a_eq, train, valid = build_go_training_sets(ff_ref, cfg, log)
        label_with_forcefield(train, ff_ref)
        label_with_forcefield(valid, ff_ref)
        rng = np.random.default_rng(cfg["reference"]["perturb_seed"])
        ff_start = perturb(ff_ref, keys, cfg["reference"]["perturb_rel"], rng)
        truth = {k: ff_ref.get(k) for k in keys}
    elif mode == "file":
        train = TrainingSet.read(_resolve(cfg["reference"]["train_file"])).configs
        valid = TrainingSet.read(_resolve(cfg["reference"]["validation_file"])).configs
        ff_start = ff_ref
        truth, a_eq = None, None
    else:
        raise ValueError(f"unknown reference mode {mode!r}")

    ts_train, ts_valid = TrainingSet(train), TrainingSet(valid)
    train_path = ts_train.write(out / "training_set.extxyz")
    valid_path = ts_valid.write(out / "validation_set.extxyz")
    start_path = ff_start.write(out / "ffield.start")

    # ---- parameter space is centered on the START force field (the truth is unknown to the optimizer)
    space = ParameterSpace.from_config(ff_start, cfg["parameters"])
    truth_in_bounds = None
    if truth is not None:
        g_true = space.encode(np.array([truth[k] for k in keys]))
        truth_in_bounds = bool(np.all((g_true >= 0) & (g_true <= 1)))

    fit = cfg.get("fitness", {})
    obj_kw = dict(sigma_e=fit.get("sigma_e", 1.0), sigma_f=fit.get("sigma_f", 1.0),
                  force_weight=fit.get("force_weight", 1.0))
    obj = Objective(ts_train, ff_start, space, **obj_kw)
    obj_val = Objective(ts_valid, ff_start, space, **obj_kw)

    ga_cfg = GAConfig(**{k: v for k, v in cfg["ga"].items()})
    ga_cfg.inject = [space.encode(space.values_from(ff_start))]      # the GA can only improve on the start

    rec.manifest(cfg, {"reference_ffield": _resolve(cfg["forcefield"]["reference"]),
                       "start_ffield": start_path, "training_set": train_path,
                       "validation_set": valid_path},
                 extra={"parameter_space": space.to_records(), "ga_config": ga_cfg.as_dict(),
                        "n_train_configs": len(ts_train), "n_valid_configs": len(ts_valid),
                        "graphene_a_eq": a_eq, "truth_in_bounds": truth_in_bounds})

    g0 = ga_cfg.inject[0]
    start_train, start_valid = obj.result(g0), obj_val.result(g0)
    log(f"start: train loss {start_train.loss:.4g} | E-RMSE {start_train.energy_rmse:.3f} kcal/mol"
        f" | F-RMSE {start_train.force_rmse:.3f} kcal/mol/A")

    def on_generation(r):
        best = obj.result(np.array(r["best_genome"]))
        rec.log_generation({**{k: r[k] for k in ("generation", "best", "mean", "median", "std", "diversity")},
                            "energy_rmse": best.energy_rmse, "force_rmse": best.force_rmse,
                            "n_objective_calls": obj.n_calls, "n_failed": obj.n_failed,
                            "best_params": dict(zip(keys, space.decode(r["best_genome"])))})
        if r["generation"] % max(1, ga_cfg.n_generations // 10) == 0:
            log(f"gen {r['generation']:4d}  best {r['best']:.4g}  mean {r['mean']:.4g}  "
                f"diversity {r['diversity']:.3f}  calls {obj.n_calls}")

    t_ga = time.time()
    ga_res = GeneticAlgorithm(ga_cfg).run(obj, len(space), callback=on_generation)
    t_ga = time.time() - t_ga

    t_loc = time.time()
    loc_cfg = cfg.get("local", {"enabled": True, "maxiter": 300})
    if loc_cfg.get("enabled", True):
        g_best, f_best, loc_info = local_refine(obj, ga_res.best_genome, maxiter=loc_cfg.get("maxiter", 300))
    else:
        g_best, f_best, loc_info = ga_res.best_genome, ga_res.best_loss, {"skipped": True}
    t_loc = time.time() - t_loc

    ff_best = space.apply(ff_start, g_best)
    ff_best.write(out / "ffield.best")
    end_train, end_valid = obj.result(g_best), obj_val.result(g_best)
    log(f"final: train loss {end_train.loss:.4g} | E-RMSE {end_train.energy_rmse:.3f} | "
        f"F-RMSE {end_train.force_rmse:.3f} | validation loss {end_valid.loss:.4g}")

    table = []
    v_start, v_end = space.values_from(ff_start), space.values_from(ff_best)
    for i, k in enumerate(keys):
        row = {"key": k, "lower": space.lower[i], "upper": space.upper[i],
               "start": v_start[i], "final": v_end[i]}
        if truth is not None:
            row.update(truth=truth[k],
                       start_rel_error=abs(v_start[i] - truth[k]) / max(abs(truth[k]), 1e-12),
                       final_rel_error=abs(v_end[i] - truth[k]) / max(abs(truth[k]), 1e-12))
        table.append(row)

    result = {"mode": mode, "parameters": table,
              "train": {"start": start_train.as_dict(), "final": end_train.as_dict()},
              "validation": {"start": start_valid.as_dict(), "final": end_valid.as_dict()},
              "ga": {"best_loss": ga_res.best_loss, "generations": len(ga_res.history) - 1,
                     "time_s": t_ga},
              "local_refinement": {**loc_info, "time_s": t_loc},
              "objective_calls": obj.n_calls, "failed_evaluations": obj.n_failed,
              "total_time_s": time.time() - t_start}
    rec.write_json("result.json", result)
    obj.close()
    obj_val.close()
    return result
