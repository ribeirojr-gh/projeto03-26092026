"""Figures of the ga-reaxff paper from the CSV files written by paper_data_v2.py (version 2, QC round).

Differences from make_paper_figures.py: workflow label moved off the dashed arrow (fig 1), markers and
legend position in fig 2b, histogram range -40..30 % (fig 4a, the original clipped 10 MOFs), interquartile
ranges in fig 5a, signed energy-term contributions (fig 6b) and the threshold-sensitivity figure S2.

    python make_paper_figures_v2.py --data data --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_style import C, DOUBLE, FF_COLORS, FF_LABELS, SINGLE, panel_label, plt, save

FFS = ["initial", "fit01", "fit02", "fit03"]


# ---------------------------------------------------------------- figure 1
def fig1_workflow(data: Path, out: Path):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=color, ec="k", lw=0.5))
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=7.5)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.5, linespacing=1.3)

    def arrow(x0, y0, x1, y1, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.6, color="k", linestyle=ls, shrinkA=0, shrinkB=0))

    top, h = 23, 16
    w, gap = 18.0, 2.0
    xs = [0.6 + k * (w + gap) for k in range(5)]
    items = [
        (r"\textbf{QMOF database}", "20\\,375 MOFs\nPBE-D3(BJ) structures\nDDEC6 charges", C["pale_blue"]),
        (r"\textbf{Chemical families}", "metal + non-metal sets\ngreedy coverage\npilot: Zn + C/H/N/O", C["pale_blue"]),
        (r"\textbf{Base force field}", "organic FC + Zn block\nof ZnOH (merged)\nZn--N, Zn--C placeholders", C["pale_yellow"]),
        (r"\textbf{Reference data}", "QMOF: $\\mathbf{F}=0$, $\\sigma=0$\nMACE: strain, Zn--L scans\nSIESTA: molecules", C["pale_yellow"]),
        (r"\textbf{GA + Nelder--Mead}", "sharded LAMMPS\n12 worker processes\n24--38 parameters", C["sand"]),
    ]
    for x, (t, b, c) in zip(xs, items):
        box(x, top, w, h, t, b, c)
    for k in range(4):
        arrow(xs[k] + w + 0.4, top + h / 2, xs[k + 1] - 0.4, top + h / 2)
    box(55.0, 1.0, 32.0, 15, r"\textbf{Per-MOF validation}",
        "228 unseen Zn MOFs, cell + positions relaxed\n$|\\Delta V/V|<5\\%$, rms displacement $<0.3$\\,\\AA\nno Zn ligand lost", C["light_green"])
    box(13.0, 1.0, 32.0, 15, r"\textbf{Diagnosis}",
        "bond lengths per element pair\nmolecules of the applications\nenergy-term decomposition", C["grey"])
    arrow(xs[4] + w / 2, top - 0.4, 80.0, 16.4)
    arrow(54.6, 8.5, 45.4, 8.5)
    arrow(29.0, 16.4, xs[3] + w / 2 - 3, top - 0.4, ls="--")
    # v2: the label sat on the dashed arrow; it is now placed below the arrow, clear of the boxes
    ax.text(50.5, 14.2, "new targets\nand parameters", fontsize=6.5, ha="center", va="center", linespacing=1.15)
    save(fig, "Figure1_workflow", out)


# ---------------------------------------------------------------- figure 2
def fig2_survey(data: Path, out: Path):
    m = pd.read_csv(data / "fig2a_mofs_per_metal.csv").head(20).iloc[::-1]
    cov = pd.read_csv(data / "fig2b_coverage.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    spin = np.round(m.n_mofs * m.spin_polarized_fraction).astype(int)
    closed = m.n_mofs - spin
    y = np.arange(len(m))
    a.barh(y, closed, color=C["steelblue"], height=0.72, lw=0, label="closed shell")
    a.barh(y, spin, left=closed, color=C["chocolate"], height=0.72, lw=0, label="spin-polarized (PBE)")
    a.set_yticks(y, m.metal)
    a.tick_params(axis="y", length=0, labelsize=8)
    a.set_xlabel("Single-metal MOFs")
    a.legend(loc="lower right")
    a.set_ylim(-0.7, len(m) - 0.3)
    panel_label(a, "a", x=-0.12)
    colors = {"C-H-O": C["grey"], "C-H-N-O": C["steelblue"], "C-H-N-O-S": C["sand"],
              "C-H-N-O-S + halogens": C["darkred"]}
    labels = {"C-H-O": "metal + C/H/O", "C-H-N-O": "metal + C/H/N/O",
              "C-H-N-O-S": "metal + C/H/N/O/S", "C-H-N-O-S + halogens": "metal + C/H/N/O/S + halogens"}
    # v2: one marker shape per base (readable in grayscale) and a legend placed above the lowest curve
    markers = {"C-H-O": "o", "C-H-N-O": "s", "C-H-N-O-S": "^", "C-H-N-O-S + halogens": "D"}
    for base, g in cov.groupby("nonmetal_base", sort=False):
        b.plot(g.n_force_fields, 100 * g.cumulative_fraction, "-", marker=markers[base], ms=2.6, lw=1.0,
               color=colors[base], label=labels[base])
    b.set_xlabel("Number of force fields")
    b.set_ylabel(r"QMOF structures covered [\%]")
    b.set_xlim(0, 31)
    b.set_ylim(0, 100)
    b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=7.5)
    panel_label(b, "b", x=-0.14)
    fig.tight_layout(w_pad=2.0)
    save(fig, "Figure2_qmof_survey", out)


# ---------------------------------------------------------------- figure 3
def fig3_references(data: Path, out: Path):
    t = pd.read_csv(data / "fig3a_teacher_force_error.csv")
    r = pd.read_csv(data / "fig3b_teacher_relaxation.csv")
    s = pd.read_csv(data / "fig3c_siesta_bonds_vs_vasp.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.5),
                                  gridspec_kw={"width_ratios": [1.25, 0.9, 1.15]})
    models = ["medium-mpa-0", "medium-0b3", "mace-matpes-pbe-0", "medium-omat-0"]
    names = ["MPA-0", "MP-0b3", "MatPES-PBE-0", "OMAT-0"]
    cols = [C["darkred"], C["steelblue"], C["sand"], C["grey"]]
    parts = a.violinplot([t[t.model == mm].force_rms_eV_A for mm in models], showmedians=True,
                         widths=0.8)
    for pc, col in zip(parts["bodies"], cols):
        pc.set_facecolor(col)
        pc.set_edgecolor("k")
        pc.set_linewidth(0.4)
        pc.set_alpha(0.85)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color("k")
        parts[k].set_linewidth(0.6)
    a.set_xticks(range(1, 5), names, rotation=20, fontsize=8)
    a.set_ylabel(r"Force rms [eV/\AA]")
    a.set_ylim(0, 0.8)
    panel_label(a, "a", x=-0.26, y=1.04)
    for k, (mm, col) in enumerate(zip(["medium-mpa-0", "medium-0b3"], cols[:2])):
        v = 100 * r[r.model == mm].volume_change.values
        x = k + np.random.default_rng(0).uniform(-0.14, 0.14, len(v))
        b.scatter(x, v, s=9, color=col, edgecolor="k", linewidth=0.3, zorder=3)
    b.axhline(0, color="k", lw=0.5, ls="--")
    b.set_xticks([0, 1], names[:2], fontsize=8)
    b.set_xlim(-0.6, 1.6)
    b.set_ylabel(r"Volume change [\%]")
    panel_label(b, "b", x=-0.34, y=1.04)
    pairs = ["Zn-O", "C-O", "C-C", "C-H"]
    choices = [("DOJO-PSML", "PseudoDojo", C["grey"]), ("ATOM-TABLE", "ATOM table", C["sand"]),
               ("mixed (ATOM-TABLE + Zn DOJO)", "Mixed (final)", C["darkred"])]
    w = 0.26
    for k, (key, lab, col) in enumerate(choices):
        vals = [100 * s[(s.pseudopotentials == key) & (s.pair == p)].relative_change.iloc[0] for p in pairs]
        c.bar(np.arange(4) + (k - 1) * w, vals, w, color=col, ec="k", lw=0.4, label=lab)
    c.set_xticks(range(4), [p.replace("-", "--") for p in pairs], fontsize=8)
    c.set_ylabel(r"SIESTA vs.\ VASP [\%]")
    c.legend(loc="upper right", fontsize=7)
    c.set_ylim(0, 3.4)
    panel_label(c, "c", x=-0.26, y=1.04)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Figure3_reference_quality", out)


# ---------------------------------------------------------------- figure 4
def fig4_validation(data: Path, out: Path):
    v = pd.read_csv(data / "fig4_per_mof_validation.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    a.axvspan(-5, 5, color=C["light_green"], alpha=0.35, lw=0, zorder=0,
              label=r"$|\Delta V/V|<5\%$")
    # v2: range widened from -30..30 to -40..30; the original axis silently dropped 10 MOFs
    # (6 in fit 01, 3 in fit 02, 1 in fit 03) whose volume change is below -30 %
    bins = np.arange(-40.5, 31.5, 1.5)
    for ff in FFS:
        dv = 100 * v[v.force_field == ff].volume_change.dropna()
        assert dv.min() > bins[0] and dv.max() < bins[-1], "histogram range would clip data"
        a.hist(dv, bins=bins, histtype="step", lw=1.1, color=FF_COLORS[ff] if ff != "initial" else "0.35",
               label=FF_LABELS[ff])
    a.set_xlabel(r"Volume change after ReaxFF relaxation [\%]")
    a.set_ylabel("MOFs")
    a.set_xlim(-40, 30)
    a.legend(loc="upper left", fontsize=7.5)
    panel_label(a, "a", x=-0.12)
    groups = [(True, "Zn--N bonded (182)"), (False, "carboxylate-only (46)")]
    w = 0.2
    for k, ff in enumerate(FFS):
        frac = []
        for flag, _ in groups:
            g = v[(v.force_field == ff) & (v.zn_n_bonded == flag)]
            frac.append(100 * g.passed.mean())
        tot = 100 * v[v.force_field == ff].passed.mean()
        b.bar(np.arange(3) + (k - 1.5) * w, frac + [tot], w, color=FF_COLORS[ff], ec="k", lw=0.4,
              label=FF_LABELS[ff])
    b.set_xticks(range(3), ["Zn--N\nbonded", "carboxylate-\nonly", "all\n(228)"], fontsize=8)
    b.set_ylabel(r"MOFs passing the scorecard [\%]")
    b.set_ylim(0, 75)
    b.legend(loc="upper left", fontsize=7.5, ncol=2)
    panel_label(b, "b", x=-0.18)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Figure4_per_mof_validation", out)


# ---------------------------------------------------------------- figure 5
def fig5_bonds_molecules(data: Path, out: Path):
    bc = pd.read_csv(data / "fig5a_bond_changes.csv")
    mol = pd.read_csv(data / "fig5b_molecules.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))
    pairs = [("Zn-N", True), ("Zn-O", False), ("C-O", True), ("C-N", True), ("C-C", True),
             ("N-H", True), ("O-H", True), ("C-H", True)]
    x = np.arange(len(pairs))
    for k, ff in enumerate(FFS):
        med, lo, hi = [], [], []
        for p, flag in pairs:
            g = bc[(bc.force_field == ff) & (bc.pair == p) & (bc.zn_n_bonded == flag)]
            med.append(100 * g.relative_change.median())
            lo.append(100 * g.relative_change.quantile(0.25))
            hi.append(100 * g.relative_change.quantile(0.75))
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        # v2: interquartile range added (the text quotes IQRs but the original figure showed medians only)
        a.errorbar(x + (k - 1.5) * 0.14, med, yerr=[med - lo, hi - med], fmt="none", ecolor="k",
                   elinewidth=0.5, capsize=0, zorder=2)
        a.plot(x + (k - 1.5) * 0.14, med, "o", ms=3.5, color=FF_COLORS[ff],
               mec="k", mew=0.3, label=FF_LABELS[ff], zorder=3)
    a.axhline(0, color="k", lw=0.5)
    a.axhspan(-2, 2, color=C["light_green"], alpha=0.3, lw=0, zorder=0)
    a.set_xticks(x, [p.replace("-", "--") for p, _ in pairs], fontsize=8)
    a.set_ylabel(r"Median bond-length change [\%]")
    a.set_ylim(-40, 16)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=4,
             handletextpad=0.2, columnspacing=0.8)
    panel_label(a, "a", x=-0.13, y=1.1)
    mols = ["CO2", "CO", "H2O", "CH4", "NH3", "H2"]
    tex = {"CO2": r"CO$_2$", "CO": "CO", "H2O": r"H$_2$O", "CH4": r"CH$_4$", "NH3": r"NH$_3$", "H2": r"H$_2$"}
    xm = np.arange(len(mols))
    for k, ff in enumerate(FFS):
        e = [100 * mol[(mol.force_field == ff) & (mol.molecule == m_)].relative_error_vs_experiment.iloc[0]
             for m_ in mols]
        b.plot(xm + (k - 1.5) * 0.14, e, "o", ms=3.5, color=FF_COLORS[ff], mec="k", mew=0.3,
               label=FF_LABELS[ff], zorder=3)
    sref = [100 * (mol[(mol.force_field == "initial") & (mol.molecule == m_)].siesta_A.iloc[0]
                   / mol[(mol.force_field == "initial") & (mol.molecule == m_)].experiment_A.iloc[0] - 1)
            for m_ in mols]
    b.plot(xm, sref, "_", ms=12, mew=1.2, color=C["dark_green"], label="SIESTA", zorder=2)
    b.axhline(0, color="k", lw=0.5)
    b.set_xticks(xm, [tex[m_] for m_ in mols])
    b.set_ylabel(r"Bond length vs.\ experiment [\%]")
    b.set_ylim(-22, 16)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=5,
             handletextpad=0.2, columnspacing=0.8)
    panel_label(b, "b", x=-0.13, y=1.1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Figure5_bonds_and_molecules", out)


# ---------------------------------------------------------------- figure 6
TERM_LABELS = {"ea": "over/under-coordination", "eb": "bond", "ep": "Coulomb", "eqeq": "QEq self",
               "ew": "van der Waals", "ev": "valence angle", "epen": "penalty", "elp": "lone pair"}
TERM_COLORS = {"ea": C["darkred"], "eb": C["steelblue"], "ep": C["sand"], "eqeq": C["pale_yellow"],
               "ew": C["mid_green"], "ev": C["purple"], "other": C["grey"]}


def fig6_terms(data: Path, out: Path):
    d = pd.read_csv(data / "fig6_term_decomposition.csv")
    cols = [c for c in d.columns if c.startswith("abs_dE_")]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    ffs = ["initial", "fit02", "fit03"]
    shown = ["ea", "eb", "ew", "ep", "eqeq", "ev"]
    left = np.zeros(len(ffs))
    shares = {}
    for ff in ffs:
        s = d[d.force_field == ff][cols].sum()
        shares[ff] = s / s.sum()
    for t in shown + ["other"]:
        vals = np.array([100 * (shares[ff][f"abs_dE_{t}_dx"] if t != "other"
                                else sum(shares[ff][f"abs_dE_{u}_dx"] for u in
                                         [c[7:-3] for c in cols] if u not in shown)) for ff in ffs])
        ax.barh(np.arange(len(ffs)), vals, left=left, color=TERM_COLORS[t], ec="k", lw=0.3, height=0.6,
                label=TERM_LABELS.get(t, "other"))
        for i, vv in enumerate(vals):
            if vv >= 9:
                ax.text(left[i] + vv / 2, i, f"{vv:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if t in ("ea", "eb", "ev") else "k")
        left += vals
    ax.set_yticks(range(len(ffs)), [FF_LABELS[f] for f in ffs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"Share of $|\partial E/\partial x|$ on the worst atom [\%]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7)
    ax.tick_params(axis="y", length=0)
    save(fig, "Figure6_energy_terms", out)


# ---------------------------------------------------------------- figure 6, signed version (v2, QC round)
def fig6_terms_signed(data: Path, out: Path):
    """Signed contribution of each ReaxFF energy term to the force on the worst atom.

    The original Figure 6 shows shares of sum |dE_t/dx|. Because the terms partly cancel, those shares
    are not shares of the force. Here each bar is the mean over the eight MOFs of -dE_t/dx / |F|
    (100 % = the whole force on the worst atom); bars can be negative or exceed 100 %."""
    d = pd.read_csv(data / "fig6_term_decomposition.csv")
    ffs = ["initial", "fit02", "fit03"]
    shown = ["ea", "eb", "ev", "ew", "ep", "eqeq"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    h = 0.24
    for k, ff in enumerate(ffs):
        g = d[d.force_field == ff]
        for j, t in enumerate(shown):
            frac = -g[f"dE_{t}_dx"] / g["force_norm_kcal_mol_A"] * 100
            y = j + (k - 1) * h
            ax.barh(y, frac.mean(), height=h * 0.9, color=FF_COLORS[ff], ec="k", lw=0.3,
                    label=FF_LABELS[ff] if j == 0 else None, zorder=2)
            ax.plot([frac.quantile(0.25), frac.quantile(0.75)], [y, y], color="k", lw=0.6, zorder=3)  # IQR, n = 8
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(100, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(len(shown)), [TERM_LABELS[t] for t in shown], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"Contribution to the force on the worst atom [\% of $|\mathbf{F}|$]")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "Figure6b_energy_terms_signed", out)


# ---------------------------------------------------------------- SI figure S2 (v2, QC round)
def figS2_threshold(data: Path, out: Path):
    """Pass fraction versus the volume-change threshold (displacement < 0.3 A and no lost ligand kept).

    Shows whether the ranking of the force fields depends on the 5 % acceptance window."""
    v = pd.read_csv(data / "fig4_per_mof_validation.csv")
    thr = np.arange(1.0, 15.01, 0.5)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for ff in FFS:
        g = v[v.force_field == ff]
        ok = (g.displacement_rms_A < 0.3) & (g.ligands_lost == 0)
        frac = [100 * float(((g.volume_change.abs() * 100 < t) & ok).mean()) for t in thr]
        ax.plot(thr, frac, "-", lw=1.1, color=FF_COLORS[ff] if ff != "initial" else "0.35", label=FF_LABELS[ff])
    ax.axvline(5, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"Volume-change threshold $|\Delta V/V|$ [\%]")
    ax.set_ylabel(r"MOFs passing [\%]")
    ax.set_xlim(1, 15)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "FigureS2_threshold_sensitivity", out)


# ---------------------------------------------------------------- SI figure
def figS1_ga(data: Path, out: Path):
    h = pd.read_csv(data / "figS_ga_history.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for fit, ff in (("zn_fit_01", "fit01"), ("zn_fit_02", "fit02"), ("zn_fit_03", "fit03")):
        g = h[h.fit == fit]
        ax.plot(g.generation, g.best_loss, "-", lw=1.1, color=FF_COLORS[ff], label=f"{FF_LABELS[ff]}, best")
        ax.plot(g.generation, g.median_loss, ":", lw=0.9, color=FF_COLORS[ff], label=f"{FF_LABELS[ff]}, median")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    save(fig, "FigureS1_ga_convergence", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    for f in (fig1_workflow, fig2_survey, fig3_references, fig4_validation, fig5_bonds_molecules,
              fig6_terms, fig6_terms_signed, figS1_ga, figS2_threshold):
        f(data, out)


if __name__ == "__main__":
    main()
