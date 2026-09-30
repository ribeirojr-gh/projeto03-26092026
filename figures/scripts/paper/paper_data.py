"""Extract the data behind every figure and table of the ga-reaxff paper.

    python paper_data.py --out data/        # run from figures/build/ (ignored by git)

Reads the ga-reaxff repository (read only) and writes one CSV per figure panel
plus provenance.json (repository commit, SHA-256 of every source file). The
only computation that is not a plain read is the energy-term decomposition of
the largest equilibrium force (figure 6), which reruns LAMMPS locally on the
committed training set and force fields.

Stored here since stage 19 so that every figure of the paper can be regenerated
from the committed run records. The manuscript and the figure files stay in
Google Drive only. Script used for the figures of the submitted manuscript
(unchanged apart from this docstring and the default --repo path).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCES: dict[str, str] = {}


def _src(repo: Path, rel: str) -> Path:
    p = repo / rel
    SOURCES[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write(out: Path, name: str, header: list[str], rows: list[list]):
    with open(out / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def fig2_survey(repo: Path, out: Path):
    s = json.loads(_src(repo, "docs/qmof_survey/summary.json").read_text())
    _write(out, "fig2a_mofs_per_metal.csv",
           ["metal", "n_mofs", "n_metal_plus_CHNO", "spin_polarized_fraction", "synthesized_fraction",
            "natoms_median", "natoms_p90"],
           [[t["metal"], t["n_mofs"], t["chno_only"], t["spin_polarized_frac"], t["synthesized_frac"],
             t["natoms_median"], t["natoms_p90"]] for t in s["per_metal"]])
    rows = []
    for base, picks in s["coverage"].items():
        for k, p in enumerate(picks, 1):
            rows.append([base, k, p["elements"], p["new"], p["cumulative"],
                         p["cumulative"] / s["n_structures"]])
    _write(out, "fig2b_coverage.csv",
           ["nonmetal_base", "n_force_fields", "force_field_elements", "new_mofs", "cumulative_mofs",
            "cumulative_fraction"], rows)


def fig3_references(repo: Path, out: Path):
    rows = []
    for model in ("medium-mpa-0", "medium-0b3", "mace-matpes-pbe-0", "medium-omat-0"):
        for r in _jsonl(_src(repo, f"docs/teacher_check/sp_{model}.jsonl")):
            rows.append([model, r["identifier"], r["natoms"], r["force_rms"], r["force_max"],
                         r["pressure_GPa"], r.get("energy_diff_per_atom")]
                        + [r["force_rms_by_element"].get(el) for el in ("C", "H", "N", "O", "Zn")])
    _write(out, "fig3a_teacher_force_error.csv",
           ["model", "identifier", "natoms", "force_rms_eV_A", "force_max_eV_A", "pressure_GPa",
            "energy_diff_per_atom_eV", "force_rms_C", "force_rms_H", "force_rms_N", "force_rms_O",
            "force_rms_Zn"], rows)
    rows = []
    for model in ("medium-mpa-0", "medium-0b3"):
        for r in _jsonl(_src(repo, f"docs/teacher_check/relax_{model}.jsonl")):
            rows.append([model, r["identifier"], r["natoms"], r["converged"], r["volume_change"],
                         r["length_change_max"], r["displacement_rms"]])
    _write(out, "fig3b_teacher_relaxation.csv",
           ["model", "identifier", "natoms", "converged", "volume_change", "length_change_max",
            "displacement_rms_A"], rows)
    rows = []
    for r in _jsonl(_src(repo, "docs/siesta_calibration/relax_bonds.jsonl")):
        for pair, b in r["bonds"].items():
            rows.append([r["pseudos"], pair, b["n"], b["ref"], b["relaxed"], b["relative_change"]])
    _write(out, "fig3c_siesta_bonds_vs_vasp.csv",
           ["pseudopotentials", "pair", "n_bonds", "vasp_A", "siesta_A", "relative_change"], rows)
    sweep = _jsonl(_src(repo, "docs/siesta_calibration/sweep.jsonl"))
    _write(out, "fig3d_siesta_sweep.csv",
           ["case", "force_rms_eV_A", "pressure_GPa", "wall_time_s"],
           [[r["case"], r.get("force_rms"), r.get("pressure_GPa"), r.get("wall_time_s")] for r in sweep])
    chk = _jsonl(_src(repo, "docs/siesta_calibration/check.jsonl"))
    _write(out, "tab_siesta_check.csv",
           ["identifier", "natoms", "force_rms_eV_A", "pressure_GPa", "charge_corr_ddec6",
            "zn_hirshfeld", "zn_ddec6", "wall_time_s", "processes"],
           [[r["identifier"], r["natoms"], r["force_rms"], r["pressure_GPa"], r["charge_corr_ddec6"],
             r["zn_hirshfeld_mean"], r["zn_ddec6_mean"], r["wall_time_s"], r["processes"]] for r in chk])


FITS = [("initial", "docs/baseline_validation"), ("fit01", "docs/zn_fit_01_validation"),
        ("fit02", "docs/zn_fit_02_validation"), ("fit03", "docs/zn_fit_03_validation")]


def fig4_validation(repo: Path, out: Path):
    rows = []
    for label, d in FITS:
        for c in _jsonl(_src(repo, f"{d}/scorecards.jsonl")):
            rows.append([label, c["identifier"], c["natoms"], c.get("zn_n_bonded"), c["passed"],
                         c.get("volume_change"), c.get("displacement_rms"), c.get("ligands_lost"),
                         c.get("error")])
    _write(out, "fig4_per_mof_validation.csv",
           ["force_field", "identifier", "natoms", "zn_n_bonded", "passed", "volume_change",
            "displacement_rms_A", "ligands_lost", "lammps_error"], rows)


def fig5_bonds_molecules(repo: Path, out: Path):
    rows = []
    for label, d in FITS:
        for c in _jsonl(_src(repo, f"{d}/scorecards.jsonl")):
            for pair, b in (c.get("bonds") or {}).items():
                rows.append([label, c["identifier"], c.get("zn_n_bonded"), pair, b["n"], b["ref"],
                             b["relaxed"], b["relaxed"] / b["ref"] - 1])
    _write(out, "fig5a_bond_changes.csv",
           ["force_field", "identifier", "zn_n_bonded", "pair", "n_bonds", "dft_A", "reaxff_A",
            "relative_change"], rows)
    rows = []
    for label, d in FITS:
        for m in json.loads(_src(repo, f"{d}/molecules.json").read_text())["molecules"]:
            rows.append([label, m["molecule"], m["bond"], m["reaxff"], m["siesta"], m["exp"],
                         m["error_vs_exp"], m.get("angle_reaxff")])
    _write(out, "fig5b_molecules.csv",
           ["force_field", "molecule", "bond", "reaxff_A", "siesta_A", "experiment_A",
            "relative_error_vs_experiment", "h2o_angle_reaxff_deg"], rows)


def tab_fits(repo: Path, out: Path):
    rows, prow = [], []
    for fit in ("zn_fit_01", "zn_fit_02", "zn_fit_03"):
        r = json.loads(_src(repo, f"runs/{fit}/result.json").read_text())
        m = json.loads(_src(repo, f"runs/{fit}/manifest.json").read_text())
        rows.append([fit, m["n_configs"], len(m["parameter_space"]), r["workers"], r["total_time_s"],
                     r["objective_calls"], r["failed_evaluations"], r["start"]["loss"], r["final"]["loss"],
                     r["final"]["force_rmse"], r["final"]["stress_rmse"], r["final"]["energy_rmse"],
                     sum(p["at_bound"] for p in r["parameters"])])
        for p in r["parameters"]:
            prow.append([fit, p["key"], p["lower"], p["upper"], p["start"], p["final"], p["at_bound"]])
    _write(out, "tab_fits.csv",
           ["fit", "n_configs", "n_parameters", "workers", "wall_time_s", "objective_calls",
            "failed_evaluations", "loss_start", "loss_final", "force_rmse_kcal_mol_A",
            "stress_rmse_GPa", "energy_rmse_kcal_mol", "n_at_bound"], rows)
    _write(out, "tab_fit_parameters.csv",
           ["fit", "parameter", "lower", "upper", "start", "final", "at_bound"], prow)
    h = []
    for fit in ("zn_fit_01", "zn_fit_02", "zn_fit_03"):
        for g in _jsonl(_src(repo, f"runs/{fit}/history.jsonl")):
            h.append([fit, g["generation"], g["best"], g["mean"], g["median"], g["diversity"],
                      g.get("force_rmse"), g.get("stress_rmse"), g.get("energy_rmse")])
    _write(out, "figS_ga_history.csv",
           ["fit", "generation", "best_loss", "mean_loss", "median_loss", "diversity",
            "force_rmse", "stress_rmse", "energy_rmse"], h)


TERMS = ["eb", "ea", "elp", "emol", "ev", "epen", "ecoa", "ehb", "et", "eco", "ew", "ep", "efi", "eqeq"]


def fig6_term_decomposition(repo: Path, out: Path, n_mofs: int = 8, h: float = 0.005):
    """|dE_term/dx| along the largest force, for the first n_mofs equilibrium configurations."""
    sys.path.insert(0, str(repo / "src"))
    import tempfile

    from ga_reaxff.dataset import TrainingSet
    from ga_reaxff.engine import _Instance
    from ga_reaxff.ffield import ForceField

    ts = TrainingSet.read(_src(repo, "data/training/zn_fit_02/train.extxyz"))
    eqs = [c for c in ts.configs if c.info["family"].endswith("_eq")][:n_mofs]
    rows = []
    for label, rel in (("initial", "data/ffields/init/ffield.Zn-FC.init"),
                       ("fit02", "runs/zn_fit_02/ffield.best"), ("fit03", "runs/zn_fit_03/ffield.best")):
        ff = ForceField.read(_src(repo, rel))
        with tempfile.TemporaryDirectory() as d:
            path = str(ff.write(Path(d) / "ffield"))
            for c in eqs:
                inst = _Instance(c, list(ff.elements), path, 1e-8)
                _, f = inst.compute(path)
                inst.close()
                i = int(np.argmax(np.linalg.norm(f, axis=1)))
                u = f[i] / np.linalg.norm(f[i])
                vals = []
                for s in (+h, -h):
                    a = c.copy()
                    a.positions[i] += s * u
                    j = _Instance(a, list(ff.elements), path, 1e-8)
                    j.L.commands_list(["compute reax all pair reaxff"])
                    j.compute(path)
                    vals.append(np.array([j.L.extract_compute("reax", 0, 1)[k] for k in range(14)]))
                    j.close()
                dedx = (vals[0] - vals[1]) / (2 * h)
                rows.append([label, c.info["identifier"], c.get_chemical_symbols()[i],
                             float(np.linalg.norm(f[i]))] + [float(abs(x)) for x in dedx])
    _write(out, "fig6_term_decomposition.csv",
           ["force_field", "identifier", "worst_atom", "force_norm_kcal_mol_A"]
           + [f"abs_dE_{t}_dx" for t in TERMS], rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip-lammps", action="store_true")
    a = ap.parse_args(argv)
    repo, out = Path(a.repo).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig2_survey(repo, out)
    fig3_references(repo, out)
    fig4_validation(repo, out)
    fig5_bonds_molecules(repo, out)
    tab_fits(repo, out)
    if not a.skip_lammps:
        fig6_term_decomposition(repo, out)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    tag = subprocess.run(["git", "-C", str(repo), "describe", "--tags", "--always"], capture_output=True,
                         text=True).stdout.strip()
    (out / "provenance.json").write_text(json.dumps(
        {"repository": "https://github.com/ribeirojr-gh/projeto03-26092026", "commit": commit,
         "describe": tag, "sources_sha256": SOURCES}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
