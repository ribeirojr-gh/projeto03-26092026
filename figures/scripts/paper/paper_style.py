"""Figure style of the paper, taken from the authors' plotting scripts.

Reference scripts (lab folder FIGURES-SCRIPTS/20128410 in Google Drive): exciton_lifetimes_2D_systems_v2.py,
phase_diagram_plot.py, plot_alpha-plus-spectrum.py, plot_psiavg.py. Common elements kept here:
LaTeX text in Times (mathptmx), 10 pt, 0.5 pt axes and ticks, frameless legends, units in
square brackets, tab: colors plus the authors' custom colors, SVG + high-dpi PNG output.
Widths follow J. Chem. Theory Comput. (single column 3.25 in, double column 7.0 in).
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE = 3.25
DOUBLE = 7.0

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.size": 10,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colors from the authors' scripts
C = {
    "purple": "tab:purple", "orange": "tab:orange", "blue": "tab:blue", "red": "tab:red",
    "green": "tab:green", "chocolate": "chocolate", "steelblue": "steelblue",
    "darkred": "#8b0000", "grey": "#c5d0d5", "deep_red": "#B2182B", "strong_orange": "#EF8A00",
    "sand": "#E6BE6A", "pale_yellow": "#F2E8B6", "light_green": "#A6C96A",
    "mid_green": "#5AA14F", "dark_green": "#0B6B2E", "pale_blue": "#b5d8ef",
}

# One color per force field, used in every figure
FF_COLORS = {"initial": C["grey"], "fit01": C["sand"], "fit02": C["steelblue"], "fit03": C["darkred"]}
FF_LABELS = {"initial": "Initial", "fit01": "Fit 01", "fit02": "Fit 02", "fit03": "Fit 03"}


def panel_label(ax, letter, x=-0.18, y=1.02):
    ax.text(x, y, rf"\textbf{{({letter})}}", transform=ax.transAxes, fontsize=10,
            ha="left", va="bottom")


def save(fig, name, outdir="figures"):
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=500)
    plt.close(fig)
    print(f"saved {name}")
