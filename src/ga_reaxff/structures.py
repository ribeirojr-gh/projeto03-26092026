"""Atomistic models for the graphene-oxide (GO) test case.

Geometry conventions
--------------------
* Graphene is built in an orthogonal (rectangular) cell so that the LAMMPS box
  is a simple `block` (no tilt factors). Rectangular unit: 4 C atoms,
  a x sqrt(3)a, with a = 2.46 Angstrom (C-C = a/sqrt(3) = 1.420 Angstrom).
* The sheet lies at z = z0 in a cell with vacuum along z (periodic in x, y).
* Functional groups follow the Lerf-Klinowski picture of GO:
    - epoxide: O bridging two neighbouring C atoms, above the C-C midpoint;
    - hydroxyl: O on top of one C, H tilted away from the sheet normal.
  Starting heights are approximate; the model is relaxed with the reference
  force field before any training data is generated (see engine.relax).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

A_GRAPHENE = 2.46
D_CC = A_GRAPHENE / np.sqrt(3.0)


def graphene_sheet(nx: int = 4, ny: int = 2, vacuum: float = 20.0,
                   a: float = A_GRAPHENE) -> Atoms:
    """Periodic graphene sheet in an orthogonal cell of nx x ny rectangular units."""
    d = a / np.sqrt(3.0)
    basis = np.array([[0.0, 0.0], [0.0, d], [a / 2, 1.5 * d], [a / 2, 2.5 * d]])
    xy = np.vstack([basis + [i * a, j * 3 * d] for i in range(nx) for j in range(ny)])
    # shift so that no atom sits exactly on a cell face
    xy += [a / 4, d / 4]
    z0 = vacuum / 2
    pos = np.column_stack([xy, np.full(len(xy), z0)])
    return Atoms(f"C{len(xy)}", positions=pos, cell=[nx * a, ny * 3 * d, vacuum], pbc=True)


def carbon_neighbors(atoms: Atoms, i: int, cutoff: float = 1.6) -> list[int]:
    """Indices of C atoms bonded to C atom i (minimum-image convention)."""
    c_idx = [k for k, s in enumerate(atoms.get_chemical_symbols()) if s == "C" and k != i]
    d = atoms.get_distances(i, c_idx, mic=True)
    return [c_idx[k] for k in np.where(d < cutoff)[0]]


def add_epoxide(atoms: Atoms, i: int, j: int, height: float = 1.25, side: int = +1) -> Atoms:
    """Add an O bridging C_i and C_j at `height` above their midpoint (side=+1 top, -1 bottom)."""
    vec = atoms.get_distance(i, j, mic=True, vector=True)
    mid = atoms.positions[i] + vec / 2
    atoms += Atoms("O", positions=[mid + [0, 0, side * height]])
    return atoms


def add_hydroxyl(atoms: Atoms, i: int, side: int = +1, d_co: float = 1.45,
                 d_oh: float = 0.97, tilt_deg: float = 70.0) -> Atoms:
    """Add an OH group on C_i. The O-H bond makes `tilt_deg` with the sheet normal."""
    o = atoms.positions[i] + [0, 0, side * d_co]
    t = np.radians(tilt_deg)
    h = o + [d_oh * np.sin(t), 0.0, side * d_oh * np.cos(t)]
    atoms += Atoms("OH", positions=[o, h])
    return atoms


def graphene_oxide_model(nx: int = 4, ny: int = 2, vacuum: float = 20.0) -> Atoms:
    """Small GO model: C32 sheet with one epoxide (top), one OH (top), one OH (bottom).

    Composition C32 O3 H2 -> C/O ~ 10.7, in the range of mildly oxidized GO.
    Sites are fixed (deterministic), well separated from each other and from
    their periodic images.
    """
    g = graphene_sheet(nx, ny, vacuum)
    # epoxide on a C-C bond near the cell origin
    i = 0
    j = carbon_neighbors(g, i)[0]
    add_epoxide(g, i, j, side=+1)
    # hydroxyl on a C roughly half a cell away (top side)
    top_oh = _farthest_carbon(g, [i, j])
    add_hydroxyl(g, top_oh, side=+1)
    # hydroxyl on the bottom side, far from both
    bot_oh = _farthest_carbon(g, [i, j, top_oh])
    add_hydroxyl(g, bot_oh, side=-1)
    g.info["sites"] = {"epoxide_C": [i, j], "hydroxyl_top_C": top_oh, "hydroxyl_bottom_C": bot_oh}
    return g


def _farthest_carbon(atoms: Atoms, taken: list[int]) -> int:
    c_idx = [k for k, s in enumerate(atoms.get_chemical_symbols()) if s == "C" and k not in taken]
    dmin = np.array([min(atoms.get_distances(k, taken, mic=True)) for k in c_idx])
    return int(c_idx[int(np.argmax(dmin))])


def min_interatomic_distance(atoms: Atoms) -> float:
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())
