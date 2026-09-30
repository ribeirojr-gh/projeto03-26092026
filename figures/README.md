# Figure scripts

Every figure of the ga-reaxff work can be regenerated from files committed in this
repository. This directory holds the scripts of the paper figures; the scripts of the
figures in `docs/` are listed below because they live elsewhere.

**Not stored here:** the manuscript, its supporting-information sources and the figure
files themselves (SVG/PDF/PNG). They stay in Google Drive only. `figures/build/` is
ignored by git and is the place to write regenerated data and figures.

## Paper figures (`figures/scripts/paper/`)

| Figure | Panel(s) | Data file written by `paper_data.py` | Script function | Source in the repository |
|---|---|---|---|---|
| 1 | workflow schematic | (text only) | `fig1_workflow` | none |
| 2 | (a) metals, (b) coverage | `fig2a_mofs_per_metal.csv`, `fig2b_coverage.csv` | `fig2_survey` | `docs/qmof_survey/summary.json` |
| 3 | (a) teacher force error, (b) teacher relaxation, (c) SIESTA bonds | `fig3a_*`, `fig3b_*`, `fig3c_*` | `fig3_references` | `docs/teacher_check/`, `docs/siesta_calibration/` |
| 4 | (a) volume-change histograms, (b) pass fractions | `fig4_per_mof_validation.csv` | `fig4_validation` | `docs/*validation*/scorecards.jsonl` |
| 5 | (a) bond-length changes, (b) molecules | `fig5a_bond_changes.csv`, `fig5b_molecules.csv` | `fig5_bonds_molecules` | scorecards, `docs/*validation*/molecules.json` |
| 6 | energy-term shares | `fig6_term_decomposition.csv` (reruns LAMMPS, about 1 min) | `fig6_terms` | `data/training/zn_fit_02/train.extxyz`, `runs/zn_fit_0[23]/ffield.best`, initial force field |
| S1 | GA convergence | `figS_ga_history.csv` | `figS1_ga` | `runs/zn_fit_0N/history.jsonl` |

Files:

* `paper_style.py` — figure style (LaTeX text in Times, 10 pt, frameless legends, units in
  square brackets), derived from the lab's reference plotting scripts.
* `paper_data.py` + `make_paper_figures.py` — the scripts that produced the figures of the
  submitted manuscript (unchanged apart from a docstring and the default `--repo` path).
* `paper_data_v2.py` + `make_paper_figures_v2.py` — corrected version proposed in the
  quality-check round of 2026-09-30 (see below). Not used for the submitted manuscript.

```bash
cd figures/build                                   # created on first use, ignored by git
python ../scripts/paper/paper_data.py --out data   # about 1 min (fig 6 runs LAMMPS)
PYTHONPATH=../scripts/paper python ../scripts/paper/make_paper_figures.py --data data --out fig
```

Requirements: the project environment plus `pandas`; a LaTeX installation with `mathptmx`
(`text.usetex` is on). Without LaTeX, the figures need a change of `paper_style.py`
(`text.usetex: False`, `mathtext.fontset: stix`) and are then only style-approximate.

## Verification

On 2026-09-30, `paper_data.py` and `make_paper_figures.py` were run from a clean checkout of
`develop` (v0.3.0 content): the 14 data files were byte-identical to those used for the
manuscript and the 7 figures were pixel-identical (difference 0 at 100 dpi).

## Changes in version 2 (`*_v2.py`)

| Figure | Problem in the original | Change |
|---|---|---|
| 1 | label "new targets and parameters" crossed by the dashed arrow | label moved off the arrow |
| 2b | legend overlapped the lowest curve; same marker for all curves | legend moved; one marker shape per curve |
| 4a | axis range -30..30 % silently dropped 10 MOFs (6 fit 01, 3 fit 02, 1 fit 03) | range -40..30 % and an assertion that nothing is clipped |
| 5a | the text quotes interquartile ranges but the figure showed medians only | interquartile whiskers added |
| 6b (new) | shares of summed absolute derivatives are not shares of the force (the terms partly cancel) | signed contribution of each term to the force on the worst atom |
| S2 (new) | the 5 % volume window is a choice | pass fraction versus threshold |

## Figures in `docs/`

| Figures | Script |
|---|---|
| `docs/figures/fig[1-4]_*` (graphene-oxide recovery) | `scripts/make_figures.py` |
| `docs/qmof_survey/fig_*` | `scripts/survey_qmof.py` |
| `docs/teacher_check/fig_*` | `scripts/teacher_check.py` |
| `docs/baseline_validation/fig_baseline.*`, `docs/zn_fit_0N_validation/fig_baseline.*` | `scripts/baseline_validation.py` |
