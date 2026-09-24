"""Training sets: configurations + reference energies/forces, grouped in families.

Each configuration is an ase.Atoms object carrying:
    info["name"]      unique label
    info["family"]    group label; energies are compared *relative to the
                      family reference* (info["is_ref"] = True), which removes
                      the arbitrary energy zero that differs between DFT and ReaxFF
    info["is_ref"]    True for the family's zero-energy configuration
    info["ref_energy"]  reference energy (kcal/mol, LAMMPS "real" units)
    info["w_e"], info["w_f"]  per-configuration weights for energy / forces
    arrays["ref_forces"]  reference forces (kcal/mol/Angstrom), optional

The extended-XYZ format is plain text, diffable in git and readable by ASE,
OVITO and most DFT post-processing tools. A DFT training set later replaces the
synthetic one simply by providing a file in this format (see docs/methodology.md).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

EV_TO_KCALMOL = 23.060548  # 1 eV = 23.0605 kcal/mol


# ---------------------------------------------------------------- generators
def strain_series(atoms: Atoms, strains, family: str = "strain") -> list[Atoms]:
    """Isotropic in-plane (x, y) strain; atoms scaled with the cell. strain 0 is the reference."""
    out = []
    for s in strains:
        a = atoms.copy()
        cell = a.cell.array.copy()
        cell[0] *= 1 + s
        cell[1] *= 1 + s
        a.set_cell(cell, scale_atoms=False)
        frac_xy = atoms.get_scaled_positions(wrap=False)[:, :2]
        a.positions[:, :2] = frac_xy @ cell[:2, :2]
        _tag(a, f"{family}_{s:+.3f}", family, is_ref=abs(s) < 1e-12)
        out.append(a)
    return out


def displacement_scan(atoms: Atoms, indices, direction, amplitudes, family: str) -> list[Atoms]:
    """Rigidly move the atoms in `indices` by amplitude * unit(direction). amplitude 0 is the reference."""
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    out = []
    for amp in amplitudes:
        a = atoms.copy()
        a.positions[list(indices)] += amp * u
        _tag(a, f"{family}_{amp:+.3f}", family, is_ref=abs(amp) < 1e-12)
        out.append(a)
    return out


def rattle_series(atoms: Atoms, n: int, stdev: float, seed: int, family: str = "rattle") -> list[Atoms]:
    """n configurations with Gaussian displacements; the unrattled structure is added as reference."""
    rng = np.random.default_rng(seed)
    ref = atoms.copy()
    _tag(ref, f"{family}_ref", family, is_ref=True)
    out = [ref]
    for k in range(n):
        a = atoms.copy()
        a.positions += rng.normal(0.0, stdev, size=a.positions.shape)
        _tag(a, f"{family}_{k:03d}", family, is_ref=False)
        out.append(a)
    return out


def _tag(a: Atoms, name: str, family: str, is_ref: bool, w_e: float = 1.0, w_f: float = 1.0):
    a.info.update(name=name, family=family, is_ref=bool(is_ref), w_e=w_e, w_f=w_f)
    a.info.pop("sites", None)  # dict-valued info does not serialize to extxyz


# -------------------------------------------------------------- the dataset
@dataclass
class TrainingSet:
    configs: list[Atoms]

    def __post_init__(self):
        names = [c.info["name"] for c in self.configs]
        if len(set(names)) != len(names):
            raise ValueError("configuration names must be unique")
        for fam in self.families:
            n_ref = sum(c.info["is_ref"] for c in self.family(fam))
            if n_ref != 1:
                raise ValueError(f"family {fam!r} must have exactly one reference config (has {n_ref})")

    def __len__(self):
        return len(self.configs)

    @property
    def families(self) -> list[str]:
        seen = []
        for c in self.configs:
            if c.info["family"] not in seen:
                seen.append(c.info["family"])
        return seen

    def family(self, name: str) -> list[Atoms]:
        return [c for c in self.configs if c.info["family"] == name]

    def set_weights(self, family: str, w_e: float | None = None, w_f: float | None = None):
        for c in self.family(family):
            if w_e is not None:
                c.info["w_e"] = float(w_e)
            if w_f is not None:
                c.info["w_f"] = float(w_f)

    def is_labeled(self) -> bool:
        return all("ref_energy" in c.info for c in self.configs)

    # ---- I/O
    def write(self, path: str | Path) -> Path:
        path = Path(path)
        write(path, self.configs, format="extxyz")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "TrainingSet":
        configs = read(path, index=":", format="extxyz")
        for c in configs:
            c.info["is_ref"] = bool(c.info["is_ref"])
        return cls(configs)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
