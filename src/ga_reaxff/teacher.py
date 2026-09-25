"""Machine-learning "teacher" potentials checked against QMOF DFT data.

The plan (docs/results_qmof_survey.md) uses a universal machine-learning
interatomic potential (MLIP) to label the many off-equilibrium configurations
a ReaxFF fit needs, with DFT as the anchor. Before trusting a teacher, it is
checked on data it was not trained on: the QMOF structures, relaxed with
PBE-D3(BJ). At those geometries the DFT forces are (close to) zero, so

    * the teacher's forces there are its force error;
    * its total energies, compared with the QMOF energies, show whether
      energy differences *between* MOFs are consistent (an overall offset is
      irrelevant; the spread is what matters);
    * relaxing the cell with the teacher shows how far its minimum is from
      the DFT minimum (volume, lattice, atomic positions).

Functions take any ASE calculator, so they are tested without a GPU.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms


def load_mace(model: str = "medium-mpa-0", dispersion: bool = True,
              dtype: str = "float64", device: str = "cuda"):  # pragma: no cover - GPU/model
    """MACE foundation model (+ D3(BJ) for PBE, as used by QMOF)."""
    from mace.calculators import mace_mp
    return mace_mp(model=model, dispersion=dispersion, damping="bj", dispersion_xc="pbe",
                   default_dtype=dtype, device=device)


def sample_indices(n: int, k: int, seed: int) -> list[int]:
    """Reproducible random sample of k out of n indices, sorted."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))


def reference_errors(atoms: Atoms, calc, energy_key: str = "energy_pbe_d3_eV") -> dict:
    """Teacher vs. DFT at the DFT-relaxed geometry (DFT forces taken as zero)."""
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = a.get_forces()
    fn = np.linalg.norm(f, axis=1)
    sym = np.array(a.get_chemical_symbols())
    out = {
        "identifier": atoms.info.get("identifier"),
        "natoms": len(a),
        "energy": e,
        "energy_ref": atoms.info.get(energy_key),
        "force_rms": float(np.sqrt((f ** 2).mean())),
        "force_max": float(fn.max()),
        "force_rms_by_element": {el: float(np.sqrt((f[sym == el] ** 2).mean()))
                                 for el in sorted(set(sym))},
    }
    if out["energy_ref"] is not None:
        out["energy_diff_per_atom"] = (e - out["energy_ref"]) / len(a)
    try:
        s = a.get_stress(voigt=True)            # eV/A^3, ASE sign convention
        out["pressure_GPa"] = float(-s[:3].mean() * 160.21766)
    except Exception:  # calculator without stress
        out["pressure_GPa"] = None
    return out


def relax_with_cell(atoms: Atoms, calc, fmax: float = 0.02, steps: int = 500) -> tuple[Atoms, dict]:
    """Relax positions and cell with the teacher; compare with the DFT structure."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    a = atoms.copy()
    a.calc = calc
    opt = FIRE(FrechetCellFilter(a), logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    v0, v1 = atoms.get_volume(), a.get_volume()
    p0, p1 = atoms.cell.cellpar(), a.cell.cellpar()
    # displacement in fractional coordinates, minimum image, expressed in the relaxed cell
    ds = a.get_scaled_positions(wrap=False) - atoms.get_scaled_positions(wrap=False)
    ds -= np.round(ds)
    ds -= ds.mean(axis=0)   # a rigid translation of the crystal is not a structural change
    disp = np.linalg.norm(ds @ a.cell.array, axis=1)
    return a, {
        "identifier": atoms.info.get("identifier"),
        "natoms": len(a),
        "converged": converged,
        "steps": int(opt.nsteps),
        "volume_change": float(v1 / v0 - 1),
        "length_change_max": float(np.max(np.abs(p1[:3] / p0[:3] - 1))),
        "angle_change_max_deg": float(np.max(np.abs(p1[3:] - p0[3:]))),
        "displacement_rms": float(np.sqrt((disp ** 2).mean())),
        "displacement_max": float(disp.max()),
    }


def energy_consistency(energy: np.ndarray, energy_ref: np.ndarray, compositions: list[dict],
                       elements: list[str]) -> dict:
    """Spread of teacher-vs-DFT energies after removing per-element offsets.

    A universal potential and a DFT code can differ by a constant per element
    (reference energies, pseudopotentials). Fitting d_i = sum_el n_el * mu_el
    by least squares removes that part; the residual per atom measures how
    consistently the teacher reproduces energy differences between MOFs.
    """
    d = np.asarray(energy) - np.asarray(energy_ref)
    X = np.array([[c.get(el, 0) for el in elements] for c in compositions], dtype=float)
    mu, *_ = np.linalg.lstsq(X, d, rcond=None)
    n = X.sum(axis=1)
    resid = (d - X @ mu) / n
    raw = d / n
    return {"offsets_eV": dict(zip(elements, mu.tolist())),
            "raw_mean_per_atom": float(raw.mean()), "raw_std_per_atom": float(raw.std()),
            "residual_mae_per_atom": float(np.abs(resid).mean()),
            "residual_rmse_per_atom": float(np.sqrt((resid ** 2).mean()))}
