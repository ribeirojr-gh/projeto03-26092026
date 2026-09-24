"""Figures for a finished GO recovery run (reproducible from the run directory).

    python scripts/make_figures.py runs/go_recovery docs/figures

Regenerates the (deterministic, unversioned) training/validation sets, checks
their SHA-256 against the run manifest, re-evaluates the reference, start and
optimized force fields with LAMMPS, and writes PNG (for reading) and SVG
(vector, small) versions of four figures:

  fig1_convergence   GA loss and RMSE vs generation (+ Nelder-Mead end point)
  fig2_parameters    parameter recovery: error start vs final, trajectories
  fig3_scans         energy along the physical scans: reference / start / final
  fig4_parity        relative energies and forces, ReaxFF vs reference
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ga_reaxff.audit import read_history
from ga_reaxff.dataset import TrainingSet, file_sha256
from ga_reaxff.engine import LammpsEngine, label_with_forcefield
from ga_reaxff.ffield import ForceField
from ga_reaxff.pipeline import _resolve, build_go_training_sets, load_config

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
C_REF, C_START, C_FINAL = "#222222", "#d95f02", "#1b9e77"


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def regenerate_sets(run: Path, manifest: dict):
    cfg = manifest["config"]
    ff_ref = ForceField.read(_resolve(cfg["forcefield"]["reference"]))
    _, _, train, valid = build_go_training_sets(ff_ref, cfg, log=lambda *_: None)
    label_with_forcefield(train, ff_ref)
    label_with_forcefield(valid, ff_ref)
    ts, vs = TrainingSet(train), TrainingSet(valid)
    with tempfile.TemporaryDirectory() as d:
        for name, s in (("training_set", ts), ("validation_set", vs)):
            h = file_sha256(s.write(Path(d) / f"{name}.extxyz"))
            ok = h == manifest["inputs"][name]["sha256"]
            print(f"{name}: regenerated, SHA-256 {'matches' if ok else 'DIFFERS FROM'} manifest")
    return ff_ref, ts, vs


def evaluate(ts: TrainingSet, ff: ForceField):
    with LammpsEngine(ts.configs, ff) as eng:
        res = eng.evaluate(ff)
    return np.array([e for e, _ in res]), [f for _, f in res]


def relative(ts: TrainingSet, energies):
    """Energy of each config relative to its family reference, grouped by family."""
    out = {}
    for fam in ts.families:
        idx = [i for i, c in enumerate(ts.configs) if c.info["family"] == fam]
        iref = next(i for i in idx if ts.configs[i].info["is_ref"])
        out[fam] = (idx, np.array([energies[i] - energies[iref] for i in idx]))
    return out


def main(run_dir: str, out_dir: str):
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    hist = read_history(run / "history.jsonl")
    keys = [p["key"] for p in result["parameters"]]
    truth = {p["key"]: p["truth"] for p in result["parameters"]}

    ff_ref, ts, vs = regenerate_sets(run, manifest)
    ff_start = ForceField.read(run / "ffield.start")
    ff_best = ForceField.read(run / "ffield.best")
    E, F = {}, {}
    for label, ff in (("ref", ff_ref), ("start", ff_start), ("final", ff_best)):
        E[label, "train"], F[label, "train"] = evaluate(ts, ff)
        E[label, "valid"], F[label, "valid"] = evaluate(vs, ff)

    # ---------------------------------------------------------- figure 1
    g = np.array([h["generation"] for h in hist])
    n_gen = g[-1]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].semilogy(g, [h["best"] for h in hist], "-o", ms=3, color=C_FINAL, label="GA best")
    ax[0].semilogy(g, [h["median"] for h in hist], "-", color="0.6", label="population median")
    ax[0].axhline(result["train"]["start"]["loss"], ls="--", color=C_START, label="start")
    ax[0].semilogy([n_gen + 3], [result["train"]["final"]["loss"]], "*", ms=12, color="C0",
                   label="after Nelder-Mead")
    ax[0].set(xlabel="generation", ylabel="training loss", title="(a) loss")
    ax[0].legend(fontsize=7)
    ax[1].semilogy(g, [h["energy_rmse"] for h in hist], "-o", ms=3, color="C3", label="energy RMSE (kcal/mol)")
    ax[1].semilogy(g, [h["force_rmse"] for h in hist], "-s", ms=3, color="C0", label="force RMSE (kcal/mol/A)")
    ax[1].semilogy([n_gen + 3] * 2, [result["train"]["final"]["energy_rmse"],
                   result["train"]["final"]["force_rmse"]], "*", ms=12, color="k", label="after Nelder-Mead")
    ax[1].axhline(1.0, ls=":", color="k", lw=0.8)
    ax[1].set(xlabel="generation", ylabel="RMSE of GA best", title="(b) errors")
    ax[1].legend(fontsize=7)
    fig.suptitle("GO recovery benchmark: convergence (start: loss 48 480, E-RMSE 145 kcal/mol)", fontsize=9)
    save(fig, out, "fig1_convergence")

    # ---------------------------------------------------------- figure 2
    short = [k.replace("bond:", "").replace("offdiag:", "vdW/").replace("angle:", "") for k in keys]
    e0 = np.array([100 * p["start_rel_error"] for p in result["parameters"]])
    e1 = np.array([100 * p["final_rel_error"] for p in result["parameters"]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(keys))
    ax[0].barh(y + 0.2, e0, 0.4, color=C_START, label="start")
    ax[0].barh(y - 0.2, e1, 0.4, color=[C_FINAL if v <= 6 else "#7570b3" for v in e1], label="final")
    ax[0].axvline(5, ls=":", color="k", lw=0.8)
    ax[0].set_yticks(y, short, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="|relative error| vs truth (%)", title="(a) parameter error: start vs final")
    ax[0].legend(fontsize=7)
    traj = np.array([[abs(h["best_params"][k] / truth[k] - 1) * 100 for k in keys] for h in hist])
    mat = np.column_stack([traj.T, e1])                     # last column: after Nelder-Mead
    im = ax[1].imshow(mat, aspect="auto", cmap="viridis_r", vmin=0, vmax=40, interpolation="nearest")
    ax[1].set_yticks(y, short, fontsize=7)
    xt = list(range(0, n_gen - 4, 10)) + [n_gen + 1]
    ax[1].set_xticks(xt, [str(t) for t in xt[:-1]] + ["NM"], fontsize=7)
    ax[1].axvline(n_gen + 0.5, color="w", lw=1.5)
    ax[1].grid(False)
    ax[1].set(xlabel="generation of the GA best (NM = after Nelder-Mead)",
              title="(b) |relative error| (%) of each parameter over the run")
    fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02, label="% (clipped at 40)")
    fig.tight_layout()
    save(fig, out, "fig2_parameters")

    # ---------------------------------------------------------- figure 3
    scans = [("epoxide_z", "epoxide O height shift (A)"), ("hydroxyl_stretch", "C-O(H) stretch (A)"),
             ("hydroxyl_bend", "H displacement (A)"), ("strain", "in-plane strain (%)")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    for ax_, (fam, xl) in zip(axs, scans):
        for split, s, mk in (("train", ts, "o"), ("valid", vs, "^")):
            famname = fam if split == "train" else f"{fam}_val"
            x = []
            for c in s.family(famname):
                v = float(c.info["name"].rsplit("_", 1)[1])
                x.append(100 * v if fam == "strain" else v)
            x = np.array(x)
            rel = {lab: relative(s, E[lab, split])[famname][1] for lab in ("ref", "start", "final")}
            o = np.argsort(x)
            ls = "-" if split == "train" else "none"
            ax_.plot(x[o], rel["ref"][o], mk, ls=ls, color=C_REF, ms=4, label="reference" if split == "train" else None)
            ax_.plot(x[o], rel["start"][o], mk, ls=ls, color=C_START, ms=4, label="start" if split == "train" else None)
            ax_.plot(x[o], rel["final"][o], mk, ls=ls, color=C_FINAL, ms=4, mfc="none",
                     label="final" if split == "train" else None)
        ax_.set(xlabel=xl, title=fam)
    axs[0].set_ylabel("dE (kcal/mol)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Energy along physical scans (circles: training, triangles: held-out validation)", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig3_scans")

    # ---------------------------------------------------------- figure 4
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for j, lab in enumerate(("start", "final")):
        for split, s, mk in (("train", ts, "o"), ("valid", vs, "^")):
            r_ref, r_ff = relative(s, E["ref", split]), relative(s, E[lab, split])
            x = np.concatenate([r_ref[f][1] for f in s.families])
            yv = np.concatenate([r_ff[f][1] for f in s.families])
            ax[j].plot(x, yv, mk, ms=3.5, alpha=0.8, color=C_START if lab == "start" else C_FINAL,
                       label=split)
        lim = ax[j].get_xlim()
        ax[j].plot(lim, lim, "k-", lw=0.7)
        rm_t = result["train"][lab]["energy_rmse"]
        rm_v = result["validation"][lab]["energy_rmse"]
        ax[j].set(xlabel="reference dE (kcal/mol)", ylabel="ReaxFF dE (kcal/mol)",
                  title=f"({'ab'[j]}) energies, {lab}: RMSE {rm_t:.1f} / {rm_v:.1f}")
        ax[j].legend(fontsize=7)
    rng = np.random.default_rng(0)
    fr = np.concatenate([f.ravel() for f in F["ref", "valid"]])
    for lab, col in (("start", C_START), ("final", C_FINAL)):
        ff_ = np.concatenate([f.ravel() for f in F[lab, "valid"]])
        sel = rng.choice(len(fr), size=min(600, len(fr)), replace=False)
        ax[2].plot(fr[sel], ff_[sel], ".", ms=2.5, alpha=0.6, color=col,
                   label=f"{lab} (RMSE {result['validation'][lab]['force_rmse']:.1f})")
    lim = [-150, 150]
    ax[2].plot(lim, lim, "k-", lw=0.7)
    ax[2].set(xlim=lim, ylim=[-400, 400], xlabel="reference F (kcal/mol/A)", ylabel="ReaxFF F",
              title="(c) force components, validation (600 sampled)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig4_parity")
    print("figures written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
