"""Per-MOF validation of a force field against the DFT-relaxed structure.

Each MOF is relaxed (positions and cell) with the force field, starting from
its QMOF PBE-D3(BJ) structure, and the result is compared with that
structure. A force field that describes a MOF well keeps it close to the DFT
minimum; a missing or wrong interaction shows up as a large volume change,
atoms drifting away, or metal atoms losing their ligands.

The scorecard thresholds are deliberately simple and explicit; they are
defaults, recorded with every result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

# Default acceptance thresholds
MAX_VOLUME_CHANGE = 0.05          # |V/V_DFT - 1|
MAX_DISPLACEMENT_RMS = 0.30       # Angstrom, after removing the cell change
COORDINATION_CUTOFF = 2.6         # Angstrom, metal-ligand contact

# Bonded pairs tracked in the scorecard (pair -> cutoff in Angstrom)
BOND_PAIRS = {("Zn", "O"): 2.6, ("Zn", "N"): 2.6, ("C", "O"): 1.6, ("C", "N"): 1.6,
              ("C", "C"): 1.7, ("C", "H"): 1.2, ("N", "H"): 1.2, ("O", "H"): 1.2}


def bond_lengths(atoms: Atoms, pairs=BOND_PAIRS) -> dict[str, list[float]]:
    """All bond lengths per element pair, using the reference bonding list.

    Returns {"A-B": [d, ...]} for pairs within their cutoff.
    """
    out = {}
    sym = np.array(atoms.get_chemical_symbols())
    for (a, b), cut in pairs.items():
        i, j, d = neighbor_list("ijd", atoms, cut)
        m = (sym[i] == a) & (sym[j] == b) & ((a != b) | (i < j))
        if m.any():
            out[f"{a}-{b}"] = d[m].tolist()
    return out


def _bond_changes(relaxed: Atoms, reference: Atoms, pairs=BOND_PAIRS) -> dict[str, dict]:
    """Mean length of the *same* bonds (reference bonding list) before and after."""
    sym = np.array(reference.get_chemical_symbols())
    out = {}
    for (a, b), cut in pairs.items():
        i, j, d0, D = neighbor_list("ijdD", reference, cut)
        m = (sym[i] == a) & (sym[j] == b) & ((a != b) | (i < j))
        if not m.any():
            continue
        # same atom pairs in the relaxed structure, minimum image
        v = relaxed.positions[j[m]] - relaxed.positions[i[m]]
        from ase.geometry import find_mic
        d1 = np.linalg.norm(find_mic(v, relaxed.cell, relaxed.pbc)[0], axis=1)
        out[f"{a}-{b}"] = {"n": int(m.sum()), "ref": float(d0[m].mean()),
                           "relaxed": float(d1.mean())}
    return out


def coordination(atoms: Atoms, metals=("Zn",), ligands=("O", "N"),
                 cutoff: float = COORDINATION_CUTOFF) -> dict[int, list[int]]:
    """Ligand atoms within `cutoff` of each metal atom (periodic)."""
    i, j = neighbor_list("ij", atoms, cutoff)
    sym = np.array(atoms.get_chemical_symbols())
    out = {int(k): [] for k in np.where(np.isin(sym, metals))[0]}
    for a, b in zip(i, j):
        if a in out and sym[b] in ligands:
            out[int(a)].append(int(b))
    return {k: sorted(set(v)) for k, v in out.items()}


@dataclass
class Scorecard:
    identifier: str | None
    natoms: int
    volume_change: float
    length_change_max: float
    angle_change_max_deg: float
    displacement_rms: float
    displacement_max: float
    metal_coordination_ref: float      # mean number of ligands per metal, DFT
    metal_coordination: float          # after relaxation
    ligands_lost: int                  # metal-ligand contacts present in DFT, absent after
    passed: bool
    bonds: dict                        # pair -> {n, ref, relaxed}: mean length of DFT bonds

    def as_dict(self):
        return asdict(self)


def compare(relaxed: Atoms, reference: Atoms, metals=("Zn",), ligands=("O", "N")) -> Scorecard:
    """Scorecard of a relaxed structure against its DFT reference (same atom order)."""
    v0, v1 = reference.get_volume(), relaxed.get_volume()
    p0, p1 = reference.cell.cellpar(), relaxed.cell.cellpar()
    ds = relaxed.get_scaled_positions(wrap=False) - reference.get_scaled_positions(wrap=False)
    ds -= np.round(ds)
    ds -= ds.mean(axis=0)   # a rigid translation of the crystal is not a structural change
    disp = np.linalg.norm(ds @ relaxed.cell.array, axis=1)
    c0 = coordination(reference, metals, ligands)
    c1 = coordination(relaxed, metals, ligands)
    lost = sum(len(set(c0[k]) - set(c1.get(k, []))) for k in c0)
    dv = v1 / v0 - 1
    rms = float(np.sqrt((disp ** 2).mean()))
    return Scorecard(
        identifier=reference.info.get("identifier"), natoms=len(reference),
        volume_change=float(dv),
        length_change_max=float(np.max(np.abs(p1[:3] / p0[:3] - 1))),
        angle_change_max_deg=float(np.max(np.abs(p1[3:] - p0[3:]))),
        displacement_rms=rms, displacement_max=float(disp.max()),
        metal_coordination_ref=float(np.mean([len(v) for v in c0.values()])) if c0 else 0.0,
        metal_coordination=float(np.mean([len(v) for v in c1.values()])) if c1 else 0.0,
        ligands_lost=int(lost),
        passed=bool(abs(dv) < MAX_VOLUME_CHANGE and rms < MAX_DISPLACEMENT_RMS and lost == 0),
        bonds=_bond_changes(relaxed, reference),
    )
