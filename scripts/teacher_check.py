"""Check MLIP teachers against QMOF DFT for the Zn + C/H/N/O family.

    python scripts/teacher_check.py [--n-sp 300] [--n-relax 30] [--out docs/teacher_check]

1. Single points at the DFT-relaxed geometries of a random sample (seed 2026)
   for every model: force error, pressure, energy consistency.
2. Cell + position relaxation with the best two models (lowest median force
   error) on a smaller sample of structures with <= 150 atoms.
Outputs: sp_<model>.jsonl, relax_<model>.jsonl, summary.json, fig_*.png/svg.
Runs on the local GPU (MACE, float64, D3(BJ)).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

# large MOFs + D3 fragment GPU memory; must be set before torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ga_reaxff import qmof  # noqa: E402
from ga_reaxff.teacher import (energy_consistency, load_mace, reference_errors,  # noqa: E402
                               relax_with_cell, sample_indices)


class Teacher:
    """GPU calculator with a CPU fallback for structures that do not fit in GPU memory."""

    def __init__(self, model):
        self.model = model
        self.gpu = load_mace(model)
        self.cpu = None

    def run(self, fn, atoms, **kw):
        import torch
        try:
            r = fn(atoms, self.gpu, **kw)
            device = "cuda"
        except RuntimeError as e:   # TorchScript wraps CUDA OOM in a plain RuntimeError
            if "out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            if self.cpu is None:
                torch.set_num_threads(CPU_THREADS)   # OMP_NUM_THREADS may be 1 in the shell
                self.cpu = load_mace(self.model, device="cpu")
            r = fn(atoms, self.cpu, **kw)
            device = "cpu"
        if isinstance(r, tuple):
            atoms_out, rec = r
            rec["device"] = device
            return atoms_out, rec
        r["device"] = device
        return r

    def close(self):
        import torch
        del self.gpu, self.cpu
        torch.cuda.empty_cache()

MODELS = ["medium-mpa-0", "medium-0b3", "mace-matpes-pbe-0", "medium-omat-0"]
STRUCTURES = "data/qmof/structures_Zn-CHNO.extxyz.gz"
SEED = 2026
CPU_THREADS = 16
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _jsonl(path, records):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(summary, sp, relax, out):
    import matplotlib.pyplot as plt
    colors = dict(zip(MODELS, (BLUE, ORANGE, AQUA, VIOLET)))
    # force error distribution per model
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = [[r["force_rms"] for r in sp[m]] for m in MODELS]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops={"markersize": 3, "markeredgecolor": INK2})
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(colors[m]); patch.set_edgecolor(INK2)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_yticks(range(1, len(MODELS) + 1), MODELS)
    ax.set_xlabel("force RMS at the DFT minimum (eV/A)", color=INK)
    ax.set_title(f"Teacher force error on {len(data[0])} Zn MOFs (DFT forces = 0 there)",
                 loc="left", fontsize=10, color=INK)
    _style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_force_error.{ext}", dpi=200)
    plt.close(fig)
    # volume change on relaxation
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ms = list(relax)
    for k, m in enumerate(ms):
        v = 100 * np.array([r["volume_change"] for r in relax[m]])
        y = k + np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
        ax.scatter(v, y, s=16, color=colors[m], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(ms)), ms)
    ax.set_xlabel("volume change after teacher relaxation (%)", color=INK)
    ax.set_title("Cell relaxation with the teacher, starting from the DFT structure",
                 loc="left", fontsize=10, color=INK)
    _style(ax); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_relax_volume.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sp", type=int, default=300)
    ap.add_argument("--n-relax", type=int, default=30)
    ap.add_argument("--max-atoms-relax", type=int, default=150)
    ap.add_argument("--out", default="docs/teacher_check")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    structures = qmof.load_structures(STRUCTURES)
    idx = sample_indices(len(structures), a.n_sp, SEED)
    sample = [structures[i] for i in idx]
    small = [s for s in sample if len(s) <= a.max_atoms_relax]
    relax_sample = [small[i] for i in sample_indices(len(small), a.n_relax, SEED)]
    elements = sorted({el for s in sample for el in s.get_chemical_symbols()})
    comps = [dict(Counter(s.get_chemical_symbols())) for s in sample]

    summary = {"structures_sha256": Path(STRUCTURES + ".sha256").read_text().split()[0],
               "seed": SEED, "n_single_point": len(sample), "n_relax": len(relax_sample),
               "relax_max_atoms": a.max_atoms_relax, "models": {}}
    sp = {}
    for m in MODELS:
        teacher = Teacher(m)
        t = time.time()
        sp[m] = []
        for k, s in enumerate(sample, 1):
            sp[m].append(teacher.run(reference_errors, s))
            if k % 25 == 0:
                print(f"[{m}] single points {k}/{len(sample)} {time.time() - t:.0f} s", flush=True)
        dt = time.time() - t
        _jsonl(out / f"sp_{m}.jsonl", sp[m])
        fr = np.array([r["force_rms"] for r in sp[m]])
        pr = np.array([r["pressure_GPa"] for r in sp[m]])
        by_el = {el: float(np.median([r["force_rms_by_element"][el] for r in sp[m]
                                      if el in r["force_rms_by_element"]])) for el in elements}
        summary["models"][m] = {
            "force_rms_median": float(np.median(fr)), "force_rms_p90": float(np.percentile(fr, 90)),
            "force_rms_max": float(fr.max()), "force_rms_median_by_element": by_el,
            "pressure_GPa_median": float(np.median(pr)),
            "pressure_GPa_abs_p90": float(np.percentile(np.abs(pr), 90)),
            "energy": energy_consistency([r["energy"] for r in sp[m]],
                                         [r["energy_ref"] for r in sp[m]], comps, elements),
            "seconds_per_structure": dt / len(sample),
            "n_cpu_fallback": sum(r["device"] == "cpu" for r in sp[m]),
        }
        teacher.close()
        print(m, json.dumps({k: v for k, v in summary["models"][m].items() if k != "energy"}))

    best = sorted(MODELS, key=lambda m: summary["models"][m]["force_rms_median"])[:2]
    relax = {}
    for m in best:
        teacher = Teacher(m)
        relax[m], relaxed, t = [], [], time.time()
        for k, s in enumerate(relax_sample, 1):
            atoms_out, rec = teacher.run(relax_with_cell, s)
            relax[m].append(rec)
            relaxed.append(atoms_out)
            print(f"[{m}] relax {k}/{len(relax_sample)} {time.time() - t:.0f} s", flush=True)
        # relaxed structures kept for re-analysis (not versioned; runs/ is ignored)
        run_dir = Path("runs/teacher_check")
        run_dir.mkdir(parents=True, exist_ok=True)
        for at in relaxed:
            at.calc = None
        from ase.io import write
        write(run_dir / f"relaxed_{m}.extxyz", relaxed, format="extxyz")
        _jsonl(out / f"relax_{m}.jsonl", relax[m])
        v = np.array([r["volume_change"] for r in relax[m]])
        summary["models"][m]["relax"] = {
            "converged": int(sum(r["converged"] for r in relax[m])),
            "volume_change_median": float(np.median(v)),
            "volume_change_abs_median": float(np.median(np.abs(v))),
            "volume_change_abs_max": float(np.abs(v).max()),
            "length_change_max_median": float(np.median([r["length_change_max"] for r in relax[m]])),
            "displacement_rms_median": float(np.median([r["displacement_rms"] for r in relax[m]])),
        }
        teacher.close()
        print(m, json.dumps(summary["models"][m]["relax"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures(summary, sp, relax, out)


if __name__ == "__main__":
    main()
