"""Objective function: ReaxFF predictions vs reference data.

    L = (1/N_E) * sum_i w_e,i * ((dE_ff,i - dE_ref,i) / sigma_E)^2
      + lambda_F * (1/N_F) * sum_i w_f,i * sum_atoms |F_ff - F_ref|^2 / (3 sigma_F^2)
      + lambda_S * (1/N_S) * sum_i w_s,i * sum_voigt (S_ff - S_ref)^2 / (6 sigma_S^2)

* dE are energies relative to the family reference configuration, so the
  (method-dependent) absolute energy zero never enters the fit.
* sigma_E, sigma_F set the "acceptable error" scale (default 1 kcal/mol and
  1 kcal/mol/A); a loss ~1 means errors are of that size on average.
* N_E counts non-reference configurations, N_F configurations with forces,
  N_S configurations with a reference stress (info["ref_stress"], Voigt
  xx yy zz yz xz xy in GPa; sigma_S default 1 GPa). The stress term is off
  unless stresses are passed.
* Parameter sets that crash LAMMPS get a fixed large penalty.

The pure-numpy `compute_loss` is separated from the LAMMPS-driven `Objective`
so it can be unit-tested (and audited) without an MD engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import TrainingSet
from .engine import EvaluationError, LammpsEngine
from .ffield import ForceField
from .parameters import ParameterSpace

PENALTY = 1.0e6


@dataclass
class FitnessResult:
    loss: float
    energy_rmse: float            # kcal/mol, relative energies, unweighted
    force_rmse: float             # kcal/mol/A per component, unweighted
    family_energy_rmse: dict = field(default_factory=dict)
    failed: bool = False
    stress_rmse: float = 0.0      # GPa per Voigt component, unweighted
    loss_terms: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"loss": self.loss, "energy_rmse": self.energy_rmse, "force_rmse": self.force_rmse,
                "stress_rmse": self.stress_rmse, "loss_terms": self.loss_terms,
                "family_energy_rmse": self.family_energy_rmse, "failed": self.failed}


def compute_loss(ts: TrainingSet, energies, forces, sigma_e: float = 1.0,
                 sigma_f: float = 1.0, force_weight: float = 1.0, stresses=None,
                 sigma_s: float = 1.0, stress_weight: float = 1.0) -> FitnessResult:
    """Loss for predicted energies/forces (lists aligned with ts.configs)."""
    energies = np.asarray(energies, float)
    name_to_idx = {c.info["name"]: k for k, c in enumerate(ts.configs)}

    e_terms, e_err_all, fam_rmse = [], [], {}
    for fam in ts.families:
        members = ts.family(fam)
        ref = next(c for c in members if c.info["is_ref"])
        i_ref = name_to_idx[ref.info["name"]]
        errs = []
        for c in members:
            if c.info["is_ref"]:
                continue
            i = name_to_idx[c.info["name"]]
            d_ff = energies[i] - energies[i_ref]
            d_ref = c.info["ref_energy"] - ref.info["ref_energy"]
            err = d_ff - d_ref
            errs.append(err)
            e_terms.append(c.info["w_e"] * (err / sigma_e) ** 2)
        if errs:
            fam_rmse[fam] = float(np.sqrt(np.mean(np.square(errs))))
            e_err_all += errs

    f_terms, f_sq = [], []
    for k, c in enumerate(ts.configs):
        if "ref_forces" not in c.arrays or forces is None:
            continue
        diff = np.asarray(forces[k]) - c.arrays["ref_forces"]
        sq = np.mean(diff ** 2)                       # mean over atoms and components
        f_sq.append(sq)
        f_terms.append(c.info["w_f"] * sq / sigma_f ** 2)

    s_terms, s_sq = [], []
    if stresses is not None:
        for k, c in enumerate(ts.configs):
            if "ref_stress" not in c.info:
                continue
            diff = np.asarray(stresses[k]) - np.asarray(c.info["ref_stress"], float)
            sq = np.mean(diff ** 2)
            s_sq.append(sq)
            s_terms.append(c.info.get("w_s", 1.0) * sq / sigma_s ** 2)

    loss_e = float(np.mean(e_terms)) if e_terms else 0.0
    loss_f = float(np.mean(f_terms)) if f_terms else 0.0
    loss_s = float(np.mean(s_terms)) if s_terms else 0.0
    return FitnessResult(
        loss=loss_e + force_weight * loss_f + stress_weight * loss_s,
        energy_rmse=float(np.sqrt(np.mean(np.square(e_err_all)))) if e_err_all else 0.0,
        force_rmse=float(np.sqrt(np.mean(f_sq))) if f_sq else 0.0,
        family_energy_rmse=fam_rmse,
        stress_rmse=float(np.sqrt(np.mean(s_sq))) if s_sq else 0.0,
        loss_terms={"energy": loss_e, "force": force_weight * loss_f, "stress": stress_weight * loss_s},
    )


class Objective:
    """Callable genome -> loss, with caching and failure accounting."""

    def __init__(self, ts: TrainingSet, ff_template: ForceField, space: ParameterSpace,
                 sigma_e: float = 1.0, sigma_f: float = 1.0, force_weight: float = 1.0,
                 qeq_tol: float = 1e-6, cache_decimals: int = 12, sigma_s: float = 1.0,
                 stress_weight: float = 0.0, n_workers: int = 1):
        if not ts.is_labeled():
            raise ValueError("training set has configurations without reference energies")
        self.ts, self.ff_template, self.space = ts, ff_template, space
        self.sigma_e, self.sigma_f, self.force_weight = sigma_e, sigma_f, force_weight
        self.sigma_s, self.stress_weight = sigma_s, stress_weight
        self.use_stress = stress_weight > 0 and any("ref_stress" in c.info for c in ts.configs)
        if n_workers > 1:
            from .engine import ShardedEngine
            self.engine = ShardedEngine(ts.configs, ff_template, n_workers, qeq_tol=qeq_tol)
        else:
            self.engine = LammpsEngine(ts.configs, ff_template, qeq_tol=qeq_tol)
        self.cache: dict[bytes, FitnessResult] = {}
        self.cache_decimals = cache_decimals
        self.n_calls = 0
        self.n_failed = 0

    def evaluate_ff(self, ff: ForceField) -> FitnessResult:
        try:
            preds = self.engine.evaluate(ff, stress=True) if self.use_stress else self.engine.evaluate(ff)
        except EvaluationError:
            self.n_failed += 1
            return FitnessResult(PENALTY, np.inf, np.inf, {}, failed=True)
        res = compute_loss(self.ts, [r[0] for r in preds], [r[1] for r in preds],
                           self.sigma_e, self.sigma_f, self.force_weight,
                           stresses=[r[2] for r in preds] if self.use_stress else None,
                           sigma_s=self.sigma_s, stress_weight=self.stress_weight)
        if not np.isfinite(res.loss):          # e.g. overflow on absurd parameters
            self.n_failed += 1
            return FitnessResult(PENALTY, np.inf, np.inf, {}, failed=True)
        return res

    def result(self, genome: np.ndarray) -> FitnessResult:
        key = np.round(np.clip(genome, 0, 1), self.cache_decimals).tobytes()
        if key not in self.cache:
            self.n_calls += 1
            self.cache[key] = self.evaluate_ff(self.space.apply(self.ff_template, genome))
        return self.cache[key]

    def __call__(self, genome: np.ndarray) -> float:
        return self.result(genome).loss

    def close(self):
        self.engine.close()
