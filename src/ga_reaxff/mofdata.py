"""Training configurations built from QMOF structures.

Two kinds of targets, chosen so that no two reference sources are mixed point
by point (see docs/results_siesta_calibration.md, section 6):

equilibrium  one family per MOF with a single configuration: the QMOF
             PBE-D3(BJ) structure, where the DFT forces and stress vanish.
             Targets: forces = 0, stress = 0. No energy term (a family of
             one has no energy difference).
strain       isotropic scaling of a MOF (cell and atoms together), labelled
             with the machine-learning teacher. Targets: energy differences
             within the family only (the teacher's forces at the QMOF
             geometry are not zero, so they would contradict the
             equilibrium targets and are not used).

Energies in kcal/mol, forces kcal/mol/A, stress GPa (ReaxFF/LAMMPS units).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

from .dataset import EV_TO_KCALMOL, _tag


def equilibrium_config(atoms: Atoms, w_f: float = 1.0, w_s: float = 1.0) -> Atoms:
    """QMOF structure as a zero-force, zero-stress target."""
    a = atoms.copy()
    ident = atoms.info["identifier"]
    keep = {"identifier": ident}
    a.info = keep
    _tag(a, f"{ident}_eq", f"{ident}_eq", is_ref=True, w_e=0.0, w_f=w_f)
    a.info["w_s"] = float(w_s)
    a.info["ref_energy"] = 0.0
    a.info["ref_stress"] = [0.0] * 6
    a.arrays["ref_forces"] = np.zeros((len(a), 3))
    for k in list(a.arrays):
        if k not in ("numbers", "positions", "ref_forces"):
            del a.arrays[k]
    return a


def strain_family(atoms: Atoms, strains, w_e: float = 1.0) -> list[Atoms]:
    """Isotropic linear strains (cell and atoms scaled); strain 0 is the reference."""
    ident = atoms.info["identifier"]
    fam = f"{ident}_strain"
    out = []
    for s in strains:
        a = atoms.copy()
        a.info = {"identifier": ident}
        for k in list(a.arrays):
            if k not in ("numbers", "positions"):
                del a.arrays[k]
        a.set_cell(atoms.cell.array * (1 + s), scale_atoms=True)
        _tag(a, f"{fam}_{s:+.4f}", fam, is_ref=abs(s) < 1e-12, w_e=w_e, w_f=0.0)
        a.info["strain"] = float(s)
        out.append(a)
    return out


def label_energies(configs: list[Atoms], calc) -> None:
    """Set info['ref_energy'] (kcal/mol) from an ASE calculator in eV (in place)."""
    for c in configs:
        a = c.copy()
        a.calc = calc
        c.info["ref_energy"] = float(a.get_potential_energy()) * EV_TO_KCALMOL


def bond_scan_family(atoms: Atoms, metal: int, ligand: int, displacements,
                     w_e: float = 1.0) -> list[Atoms]:
    """Move the metal atom along the metal->ligand bond by each displacement (A).

    Negative values shorten the bond. Displacement 0 is the reference. Energy
    targets only (teacher labels); the family pins where the minimum of the
    metal-ligand interaction lies, which equilibrium forces alone did not
    (fit 01: Zn-N collapsed by 35 %).
    """
    from ase.geometry import find_mic
    ident = atoms.info["identifier"]
    lig = atoms.get_chemical_symbols()[ligand]
    met = atoms.get_chemical_symbols()[metal]
    fam = f"{ident}_scan_{met}{metal}-{lig}{ligand}"
    v = find_mic(atoms.positions[ligand] - atoms.positions[metal], atoms.cell, atoms.pbc)[0]
    u = v / np.linalg.norm(v)
    out = []
    for d in displacements:
        a = atoms.copy()
        a.info = {"identifier": ident}
        for k in list(a.arrays):
            if k not in ("numbers", "positions"):
                del a.arrays[k]
        a.positions[metal] -= d * u          # moving the metal away from the ligand stretches the bond
        _tag(a, f"{fam}_{d:+.3f}", fam, is_ref=abs(d) < 1e-12, w_e=w_e, w_f=0.0)
        a.info["bond_change"] = float(d)
        out.append(a)
    return out


def first_contact(atoms: Atoms, metal: str, ligand: str, cutoff: float = 2.6):
    """(metal index, ligand index) of the shortest metal-ligand contact, or None."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    sym = np.array(atoms.get_chemical_symbols())
    m = (sym[i] == metal) & (sym[j] == ligand)
    if not m.any():
        return None
    k = int(np.argmin(np.where(m, d, np.inf)))
    return int(i[k]), int(j[k])
