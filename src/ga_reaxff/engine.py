"""LAMMPS evaluation engine for ReaxFF single points, relaxations and labeling.

Performance design
------------------
Creating a LAMMPS instance costs ~20 ms, a ReaxFF+QEq single point on ~40
atoms ~5 ms. The GA evaluates the *same* geometries with *many* parameter
sets, so the engine keeps one persistent LAMMPS instance per configuration
and, for each candidate force field, only re-issues

    pair_coeff * * <new ffield> C H O
    run 0

This was verified to reproduce a fresh instance to ~1e-8 kcal/mol (QEq
tolerance); tests/test_engine.py keeps checking it.

Robustness
----------
Unphysical parameter sets can make LAMMPS raise an error or return NaN.
Both are converted into EvaluationError; the instance is rebuilt so the next
candidate starts from a clean state. The fitness layer turns the error into a
penalty, so the GA simply discards such individuals.

Cells
-----
Any cell is accepted. LAMMPS needs a *restricted* triclinic box (a along x,
b in the xy plane), so the structure is rotated into ASE's standard form,
rcell = cell @ Q.T, before it is handed to LAMMPS; positions follow the same
rotation and forces (and relaxed positions and cells) are rotated back with Q.
Energies are rotation invariant. Orthogonal cells give Q = identity.

Units: LAMMPS "real" -> energies kcal/mol, forces kcal/mol/Angstrom.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

from .ffield import ForceField

MASSES = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}


def _mass(el: str) -> float:
    """Atomic mass (amu); ASE's table for elements not listed in MASSES.

    Dummy atom types of some published force fields (e.g. "X") never occur in
    a structure; they get mass 1 so that LAMMPS accepts the type.
    """
    if el in MASSES:
        return MASSES[el]
    from ase.data import atomic_masses, chemical_symbols
    return float(atomic_masses[chemical_symbols.index(el)]) if el in chemical_symbols else 1.0


class EvaluationError(RuntimeError):
    """Raised when LAMMPS fails or returns non-finite results for a force field."""


def _import_lammps():
    try:
        import lammps  # noqa: F401
        return lammps
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the LAMMPS Python module is required: pip install lammps") from exc


class _Instance:
    """One LAMMPS instance bound to one fixed geometry."""

    def __init__(self, atoms: Atoms, elements: list[str], ffield_path: str, qeq_tol: float):
        if not all(atoms.pbc):
            raise ValueError("fully periodic cells are required")
        rcell, Q = atoms.cell.standard_form()
        # standard_form is lower triangular; snap round-off in the zero entries
        self.rcell = np.tril(rcell)
        self.Q = np.asarray(Q)
        self.atoms = atoms
        self.elements = elements
        self.qeq_tol = qeq_tol
        self._build(ffield_path)

    def _build(self, ffield_path: str):
        lammps = _import_lammps()
        L = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        a = self.atoms
        r = self.rcell
        lx, ly, lz = float(r[0, 0]), float(r[1, 1]), float(r[2, 2])
        xy, xz, yz = float(r[1, 0]), float(r[2, 0]), float(r[2, 1])
        if max(abs(xy), abs(xz), abs(yz)) < 1e-10:
            region = f"region box block 0 {lx!r} 0 {ly!r} 0 {lz!r}"
        else:
            region = f"region box prism 0 {lx!r} 0 {ly!r} 0 {lz!r} {xy!r} {xz!r} {yz!r}"
        cmds = ["units real", "atom_style charge", "atom_modify map array sort 0 0",
                "boundary p p p", region, f"create_box {len(self.elements)} box"]
        cmds += [f"mass {i + 1} {_mass(el)}" for i, el in enumerate(self.elements)]
        L.commands_list(cmds)
        types = [self.elements.index(s) + 1 for s in a.get_chemical_symbols()]
        wrapped = a.copy()
        wrapped.wrap()   # LAMMPS needs atoms inside the periodic box; forces are unaffected
        x = wrapped.positions @ self.Q.T
        L.create_atoms(len(a), None, types, x.flatten().tolist())
        if L.get_natoms() != len(a):
            raise RuntimeError("LAMMPS dropped atoms during creation")
        L.commands_list(["pair_style reaxff NULL checkqeq yes safezone 3.0 mincap 200",
                         f"pair_coeff * * {ffield_path} {' '.join(self.elements)}",
                         f"fix qeq all qeq/reaxff 1 0.0 10.0 {self.qeq_tol} reaxff",
                         "thermo_style custom step pe"])
        self.L = L

    def compute(self, ffield_path: str) -> tuple[float, np.ndarray]:
        L = self.L
        L.command(f"pair_coeff * * {ffield_path} {' '.join(self.elements)}")
        L.command("run 0 post no")
        e = float(L.get_thermo("pe"))
        f = np.array(L.gather_atoms("f", 1, 3), dtype=float).reshape(-1, 3)
        return e, f @ self.Q

    def positions(self) -> np.ndarray:
        """Current LAMMPS positions in the original (unrotated) frame."""
        x = np.array(self.L.gather_atoms("x", 1, 3), dtype=float).reshape(-1, 3)
        return x @ self.Q

    def cell(self) -> np.ndarray:
        """Current LAMMPS box as cell vectors in the original frame."""
        lo, hi, xy, yz, xz, _, _ = self.L.extract_box()
        lx, ly, lz = (h - l for h, l in zip(hi, lo))
        rcell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
        return rcell @ self.Q

    def close(self):
        try:
            self.L.close()
        except Exception:  # pragma: no cover
            pass


class LammpsEngine:
    """Evaluate a list of fixed configurations for arbitrary ReaxFF parameter sets."""

    def __init__(self, configs: list[Atoms], ff_template: ForceField,
                 qeq_tol: float = 1e-6, workdir: str | None = None):
        self.configs = configs
        self.elements = list(ff_template.elements)
        self.qeq_tol = qeq_tol
        self._tmp = tempfile.TemporaryDirectory(prefix="ga_reaxff_", dir=workdir)
        self._ff_path = os.path.join(self._tmp.name, "ffield.current")
        self._template_path = os.path.join(self._tmp.name, "ffield.template")
        ff_template.write(self._template_path)
        self._inst = [_Instance(c, self.elements, self._template_path, qeq_tol) for c in configs]
        self.n_evaluations = 0
        self.n_failures = 0

    def evaluate(self, ff: ForceField) -> list[tuple[float, np.ndarray]]:
        """Energies and forces of every configuration with force field `ff`."""
        ff.write(self._ff_path)
        self.n_evaluations += 1
        results = []
        for k, inst in enumerate(self._inst):
            try:
                e, f = inst.compute(self._ff_path)
            except Exception as exc:
                self._rebuild(k)
                raise EvaluationError(f"LAMMPS error on config {k}: {exc}") from exc
            if not (np.isfinite(e) and np.all(np.isfinite(f))):
                self._rebuild(k)
                raise EvaluationError(f"non-finite energy/forces on config {k}")
            results.append((e, f))
        return results

    def _rebuild(self, k: int):
        """Replace a possibly corrupted instance, using the known-good template ffield."""
        old = self._inst[k]
        old.close()
        self._inst[k] = _Instance(old.atoms, self.elements, self._template_path, self.qeq_tol)
        self.n_failures += 1

    def close(self):
        for inst in self._inst:
            inst.close()
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- utilities
def single_point(atoms: Atoms, ff: ForceField, qeq_tol: float = 1e-6) -> tuple[float, np.ndarray]:
    """Energy/forces with a fresh LAMMPS instance (slow, used for cross-checks).

    Raises EvaluationError on non-finite results: LAMMPS can return a finite
    energy with NaN forces (e.g. an element pair without a bond entry).
    """
    with tempfile.TemporaryDirectory() as d:
        p = ff.write(Path(d) / "ffield")
        inst = _Instance(atoms, list(ff.elements), str(p), qeq_tol)
        try:
            e, f = inst.compute(str(p))
        finally:
            inst.close()
    if not (np.isfinite(e) and np.all(np.isfinite(f))):
        raise EvaluationError("non-finite energy/forces")
    return e, f


def relax(atoms: Atoms, ff: ForceField, ftol: float = 1e-3, maxiter: int = 2000,
          qeq_tol: float = 1e-8, cell: bool = False, pressure: float = 0.0) -> Atoms:
    """Conjugate-gradient minimization; returns a relaxed copy.

    With `cell=True` the cell relaxes too (``fix box/relax``, all six
    components for triclinic boxes, target `pressure` in atm), so the result
    can be compared with a DFT-relaxed structure.

    Notes
    -----
    * The quadratic line search is used because ReaxFF energies contain small
      discontinuities (bond-order cutoff, taper), on which the default
      backtracking search stalls far from the minimum.
    * Stopping is by force tolerance only (etol = 0), in kcal/mol/Angstrom.
    """
    with tempfile.TemporaryDirectory() as d:
        p = ff.write(Path(d) / "ffield")
        inst = _Instance(atoms, list(ff.elements), str(p), qeq_tol)
        try:
            cmds = []
            if cell:
                tri = np.any(np.abs(inst.rcell[np.tril_indices(3, -1)]) > 1e-10)
                cmds.append(f"fix relax all box/relax {'tri' if tri else 'aniso'} {pressure} "
                            "vmax 0.001")
            cmds += ["min_style cg", "min_modify line quadratic",
                     f"minimize 0.0 {ftol} {maxiter} {10 * maxiter}"]
            inst.L.commands_list(cmds)
            x = inst.positions()
            new_cell = inst.cell() if cell else atoms.cell.array
        finally:
            inst.close()
    out = atoms.copy()
    out.set_cell(new_cell, scale_atoms=False)
    out.positions = x
    return out


def graphene_lattice_constant(ff: ForceField, a_min: float = 2.44, a_max: float = 2.58,
                              n: int = 15) -> float:
    """Equilibrium in-plane lattice constant of pristine graphene for force field `ff`.

    Grid scan of E(a) on a C32 sheet followed by a parabola through the three
    lowest-energy grid points. A grid scan (instead of a gradient-based cell
    relaxation) is deliberate: ReaxFF E(a) is not smooth (see docs).
    Models must be built at the lattice constant of the force field used to
    relax them; otherwise the sheet stores spurious strain energy.
    """
    from .structures import graphene_sheet

    grid = np.linspace(a_min, a_max, n)
    energies = np.array([single_point(graphene_sheet(4, 2, a=a), ff)[0] for a in grid])
    k = int(np.clip(np.argmin(energies), 1, n - 2))
    c = np.polyfit(grid[k - 1:k + 2], energies[k - 1:k + 2], 2)
    a_eq = -c[1] / (2 * c[0]) if c[0] > 0 else grid[k]
    return float(np.clip(a_eq, grid[k - 1], grid[k + 1]))


def label_with_forcefield(configs: list[Atoms], ff: ForceField, qeq_tol: float = 1e-6) -> None:
    """Write ReaxFF energies/forces into info['ref_energy'] / arrays['ref_forces'] (in place).

    This produces a *synthetic* reference (used for the recovery benchmark).
    For real parametrizations these fields come from DFT instead.
    """
    with LammpsEngine(configs, ff, qeq_tol=qeq_tol) as eng:
        for c, (e, f) in zip(configs, eng.evaluate(ff)):
            c.info["ref_energy"] = e
            c.arrays["ref_forces"] = f
