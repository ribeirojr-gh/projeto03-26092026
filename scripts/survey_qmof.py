"""Survey of the QMOF database for ReaxFF parametrization planning.

    python scripts/survey_qmof.py [--fetch] [--data data/qmof] [--out docs/qmof_survey]

--fetch downloads a fresh snapshot from the Materials Project (MPContribs
project 'mofexplorer') into --data; without it the committed snapshot is used
and verified against its SHA-256. Outputs in --out:

    summary.json          all numbers quoted in docs/results_qmof_survey.md
    metals.md             per-metal table (single-metal MOFs)
    coverage.md           greedy force-field coverage for each non-metal base
    fig_metals.{png,svg}  single-metal MOFs per metal, closed-shell vs spin-polarized
    fig_coverage.{png,svg} cumulative fraction of the database covered vs number of force fields
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ga_reaxff import qmof
from ga_reaxff.families import Family, greedy_cover, node_types

# Non-metal bases a force field could be built on (the metal is added per family).
BASES = {
    "C-H-O": ("C", "H", "O"),
    "C-H-N-O": ("C", "H", "N", "O"),
    "C-H-N-O-S": ("C", "H", "N", "O", "S"),
    "C-H-N-O-S + halogens": ("C", "H", "N", "O", "S", "F", "Cl", "Br", "I"),
}
N_FF = 30          # force fields considered in the coverage curves
N_METALS_FIG = 20  # metals shown in the bar chart

# Reference palette of the project figures (categorical slots 1-4, light mode)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ranked(counter):
    """Counter items by decreasing count, ties broken by key (deterministic)."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))


def per_metal_table(rows, fams):
    by_metal = defaultdict(list)
    for r, f in zip(rows, fams):
        if len(f.metals) == 1:
            by_metal[f.metals[0]].append(r)
    table = []
    for m, rs in by_metal.items():
        natoms = np.array([r["natoms"] for r in rs])
        nodes = node_types(rows, m, top=1)
        table.append({
            "metal": m,
            "n_mofs": len(rs),
            "spin_polarized_frac": float(np.mean(
                [r.get("outputsDFT.PBE.magnetism.spinPolarized") == "Yes" for r in rs])),
            "synthesized_frac": float(np.mean([r.get("synthesized") == "Yes" for r in rs])),
            "natoms_median": float(np.median(natoms)),
            "natoms_p90": float(np.percentile(natoms, 90)),
            "chno_only": sum(1 for r in rs
                             if Family.from_chemsys(r["chemsys"]).elements
                             <= {m, "C", "H", "N", "O"}),
            "top_node": nodes[0][0] if nodes else None,
            "top_node_count": nodes[0][1] if nodes else 0,
        })
    return sorted(table, key=lambda t: (-t["n_mofs"], t["metal"]))


def coverage_curves(fams, metals):
    sets = [f.elements for f in fams]
    curves = {}
    for name, base in BASES.items():
        picks = greedy_cover(sets, [set(base) | {m} for m in metals], n_max=N_FF)
        curves[name] = [{"elements": "-".join(c), "new": n, "cumulative": cum}
                        for c, n, cum in picks]
    return curves


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_metals(table, out: Path):
    import matplotlib.pyplot as plt
    t = table[:N_METALS_FIG][::-1]
    names = [x["metal"] for x in t]
    total = np.array([x["n_mofs"] for x in t])
    spin = np.round(total * np.array([x["spin_polarized_frac"] for x in t])).astype(int)
    closed = total - spin
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y = np.arange(len(t))
    ax.barh(y, closed, height=0.72, color=BLUE, edgecolor="white", linewidth=1.5,
            label="closed shell (PBE not spin-polarized)")
    ax.barh(y, spin, left=closed, height=0.72, color=ORANGE, edgecolor="white",
            linewidth=1.5, label="spin-polarized")
    for yi, n in zip(y, total):
        ax.text(n + total.max() * 0.01, yi, f"{n}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlabel("single-metal MOFs in QMOF", color=INK)
    ax.set_title("MOFs per metal (top 20 of the single-metal MOFs)", color=INK,
                 loc="left", fontsize=10)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_metals.{ext}", dpi=200)
    plt.close(fig)


def fig_coverage(curves, n_total, out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (name, c), color in zip(curves.items(), (BLUE, ORANGE, AQUA, VIOLET)):
        k = np.arange(1, len(c) + 1)
        frac = 100 * np.array([p["cumulative"] for p in c]) / n_total
        ax.plot(k, frac, color=color, linewidth=2, marker="o", markersize=3.5,
                label=f"metal + {name}")
        ax.annotate(f"{frac[-1]:.0f} %", (k[-1], frac[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("number of force fields (one metal each, chosen greedily)", color=INK)
    ax.set_ylabel("QMOF structures covered (%)", color=INK)
    ax.set_title(f"Database coverage vs. number of ReaxFF force fields (N = {n_total})",
                 color=INK, loc="left", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, N_FF + 3)
    _style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_coverage.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--data", default="data/qmof")
    ap.add_argument("--out", default="docs/qmof_survey")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.fetch:  # pragma: no cover - network
        rows, prov = qmof.fetch()
        qmof.save_snapshot(rows, prov, a.data)
    rows = qmof.load_snapshot(a.data)
    fams = [Family.from_chemsys(r["chemsys"]) for r in rows]
    n = len(rows)

    n_metals = Counter(len(f.metals) for f in fams)
    metals = sorted({f.metals[0] for f in fams if len(f.metals) == 1})
    table = per_metal_table(rows, fams)
    curves = coverage_curves(fams, metals)
    natoms = np.array([r["natoms"] for r in rows])

    summary = {
        "snapshot_sha256": (Path(a.data) / qmof.SNAPSHOT_HASH).read_text().split()[0],
        "n_structures": n,
        "n_metals_per_structure": {str(k): v for k, v in sorted(n_metals.items())},
        "n_distinct_single_metals": len(metals),
        "nonmetal_frequency": dict(_ranked(Counter(e for f in fams for e in f.nonmetals))),
        "family_keys_top30": _ranked(Counter(f.key for f in fams))[:30],
        "natoms_percentiles": dict(zip(["p10", "p50", "p90", "max"],
                                       np.percentile(natoms, [10, 50, 90, 100]).tolist())),
        "source": dict(_ranked(Counter(r.get("source") for r in rows))),
        "synthesized": dict(Counter(r.get("synthesized") for r in rows)),
        "spin_polarized": dict(Counter(r.get("outputsDFT.PBE.magnetism.spinPolarized")
                                       for r in rows)),
        "missing_mofid_nodes": sum(1 for r in rows if not r.get("mofid.smilesNodes")),
        "per_metal": table,
        "coverage": curves,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = ["| metal | MOFs | metal + C/H/N/O only | spin-polarized | synthesized "
             "| median atoms | p90 atoms | most common node (MOFid) |",
             "|---|---|---|---|---|---|---|---|"]
    for t in table[:30]:
        lines.append(f"| {t['metal']} | {t['n_mofs']} | {t['chno_only']} "
                     f"| {100 * t['spin_polarized_frac']:.0f} % | {100 * t['synthesized_frac']:.0f} % "
                     f"| {t['natoms_median']:.0f} | {t['natoms_p90']:.0f} "
                     f"| `{t['top_node']}` ({t['top_node_count']}) |")
    (out / "metals.md").write_text("\n".join(lines) + "\n")

    lines = []
    for name, c in curves.items():
        lines += [f"### metal + {name}", "", "| # | force field | new MOFs | cumulative | % of QMOF |",
                  "|---|---|---|---|---|"]
        for i, p in enumerate(c[:15], 1):
            lines.append(f"| {i} | {p['elements']} | {p['new']} | {p['cumulative']} "
                         f"| {100 * p['cumulative'] / n:.1f} |")
        lines.append("")
    (out / "coverage.md").write_text("\n".join(lines))

    fig_metals(table, out)
    fig_coverage(curves, n, out)
    print(f"{n} structures; outputs in {out}")


if __name__ == "__main__":
    main()
