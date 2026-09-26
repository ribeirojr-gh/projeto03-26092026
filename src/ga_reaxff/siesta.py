"""SIESTA (DFT) single points and relaxations for reference data.

Project rule: DFT runs on the local workstation only (never on GitHub
Actions), sized to the free cores and memory (`resources.plan`).

Setup (stage 16)
----------------
* SIESTA 5.4.2 (MPI, DFT-D3 via s-dftd3, libxc), `siesta` on PATH.
* PBE + D3 with Becke-Johnson damping and SIESTA's PBE defaults - the same
  functional as QMOF (PBE-D3(BJ), VASP). Basis and code differ, so absolute
  energies are not comparable with QMOF; geometries and energy differences
  are, and the calibration measures how far.
* Norm-conserving PBE pseudopotentials from ~/Pacotes/PSEUDOS, chosen per
  element (`SiestaSettings.pseudo_families`): SIESTA's ATOM-TABLE (.psf)
  for H/C/N/O, PseudoDojo (.psml, semicore) for Zn. They are copied into
  each run directory and their SHA-256 recorded.

Each call writes a self-contained run directory:

    <run_dir>/<label>.fdf, <El>.psml, <label>.out, <label>.FA, ...
    <run_dir>/result.json   energies, forces, stress, charges, timings,
                            settings, pseudopotential hashes, resource plan

Units of results: eV, eV/Angstrom, eV/Angstrom^3 (stress), e (charges).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

PSEUDO_ROOT = Path(os.environ.get("GA_REAXFF_PSEUDO_ROOT", Path.home() / "Pacotes" / "PSEUDOS"))
PSEUDO_DIR = PSEUDO_ROOT / "DOJO-PSML"


# Grimme D3 with Becke-Johnson damping, PBE parameters (a2 in Bohr)
D3BJ_PBE = {"s6": 1.0, "a1": 0.4289, "s8": 0.7875, "a2": 4.4407}


@dataclass
class SiestaSettings:
    """DFT settings; every field ends up in result.json."""
    xc_functional: str = "GGA"
    xc_authors: str = "PBE"
    d3: bool = True                       # DFT-D3, BJ damping, PBE defaults
    basis: str = "DZP"
    energy_shift_ry: float = 0.01
    mesh_cutoff_ry: float = 300.0
    kgrid_cutoff_ang: float = 10.0        # SIESTA kgrid.Cutoff (0 = Gamma only)
    electronic_temperature_k: float = 300.0
    dm_tolerance: float = 1e-5
    max_scf: int = 300
    mixing_weight: float = 0.1
    mixing_history: int = 8
    spin: str = "non-polarized"
    # pseudopotential family (sub-directory of PSEUDO_ROOT) per element;
    # "default" applies to elements not listed. Stage-16 calibration: SIESTA's
    # own table (Troullier-Martins, PBE) for the light elements gives the
    # organic bond lengths closest to QMOF, PseudoDojo (with 3s3p semicore)
    # for Zn gives the Zn-O bond; the table's Zn (no semicore) makes Zn-O
    # 2.4 % too long.
    pseudo_families: dict = field(default_factory=lambda: {"default": "ATOM-TABLE",
                                                            "Zn": "DOJO-PSML"})
    extra: dict = field(default_factory=dict)   # any further fdf key -> value

    def pseudo_dirs(self) -> dict:
        return {k: PSEUDO_ROOT / v for k, v in self.pseudo_families.items()}

    def as_dict(self):
        return asdict(self)


def pseudo_file(element: str, pseudo_dir=PSEUDO_DIR) -> Path:
    """<element>.psml, else <element>.psf, in `pseudo_dir`.

    `pseudo_dir` is a directory, or a mapping element -> directory (key
    "default" for the rest), to combine pseudopotential families.
    """
    if isinstance(pseudo_dir, dict):
        pseudo_dir = pseudo_dir.get(element, pseudo_dir.get("default", PSEUDO_DIR))
    for ext in (".psml", ".psf"):
        f = Path(pseudo_dir) / f"{element}{ext}"
        if f.exists():
            return f
    raise FileNotFoundError(f"no pseudopotential for {element} in {pseudo_dir}")


def pseudo_sha256(element: str, pseudo_dir=PSEUDO_DIR) -> str:
    return hashlib.sha256(pseudo_file(element, pseudo_dir).read_bytes()).hexdigest()


def write_fdf(atoms: Atoms, label: str, settings: SiestaSettings,
              relax: dict | None = None) -> str:
    """fdf input text for `atoms` (any periodic cell, Cartesian Angstrom)."""
    species = sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers[s])
    idx = {s: i + 1 for i, s in enumerate(species)}
    s = settings
    lines = [f"SystemLabel {label}", f"NumberOfAtoms {len(atoms)}",
             f"NumberOfSpecies {len(species)}", "%block ChemicalSpeciesLabel"]
    lines += [f" {idx[e]} {atomic_numbers[e]} {e}" for e in species]
    lines += ["%endblock ChemicalSpeciesLabel", "LatticeConstant 1.0 Ang",
              "%block LatticeVectors"]
    lines += [" " + " ".join(f"{x:.10f}" for x in v) for v in atoms.cell.array]
    lines += ["%endblock LatticeVectors", "AtomicCoordinatesFormat Ang",
              "%block AtomicCoordinatesAndAtomicSpecies"]
    lines += [f" {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {idx[e]}"
              for p, e in zip(atoms.positions, atoms.get_chemical_symbols())]
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies",
              f"XC.functional {s.xc_functional}", f"XC.authors {s.xc_authors}",
              f"PAO.BasisSize {s.basis}", f"PAO.EnergyShift {s.energy_shift_ry} Ry",
              f"MeshCutoff {s.mesh_cutoff_ry} Ry",
              f"kgrid.Cutoff {s.kgrid_cutoff_ang} Ang",
              f"ElectronicTemperature {s.electronic_temperature_k} K",
              f"DM.Tolerance {s.dm_tolerance}", f"MaxSCFIterations {s.max_scf}",
              "SCF.Mixer.Method Pulay", f"SCF.Mixer.Weight {s.mixing_weight}",
              f"SCF.Mixer.History {s.mixing_history}",
              f"Spin {s.spin}",
              "WriteForces true", "WriteHirshfeldPop true", "WriteMullikenPop 0",
              "SaveHS false", "XML.Write true"]
    if s.d3:
        # Explicit Grimme D3(BJ) parameters for PBE. Found in stage 16:
        # "DFTD3.UseXCDefaults true" left SIESTA 5.4.2 on its generic defaults
        # (a1 0.4, s8 1.0, a2 5.0), giving -1.768 eV instead of the -2.143 eV of
        # QMOF (VASP IVDW=12) and of torch-dftd on the same structure.
        # SIESTA also adds the three-body (Axilrod-Teller-Muto) term, which
        # VASP IVDW=12 does not; a vanishing 3-body cutoff removes it
        # (verified: -2.0211 eV with it = torch-dftd abc=True -2.0215 eV).
        lines += ["DFTD3 true", "DFTD3.UseXCDefaults false", "DFTD3.BJdamping true",
                  f"DFTD3.s6 {D3BJ_PBE['s6']}", f"DFTD3.a1 {D3BJ_PBE['a1']}",
                  f"DFTD3.s8 {D3BJ_PBE['s8']}", f"DFTD3.a2 {D3BJ_PBE['a2']}",
                  "DFTD3.3BodyCutOff 0.01 Bohr"]
    if relax:
        lines += [f"MD.TypeOfRun {relax.get('method', 'CG')}",
                  f"MD.NumCGsteps {relax.get('steps', 200)}",
                  f"MD.MaxForceTol {relax.get('fmax', 0.02)} eV/Ang",
                  f"MD.VariableCell {'true' if relax.get('cell') else 'false'}",
                  f"MD.MaxStressTol {relax.get('smax_gpa', 0.1)} GPa",
                  "WriteCoorXmol true"]
    for k, v in s.extra.items():
        lines.append(f"{k} {v}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ parsing
_FLOAT = r"[-+]?\d+\.\d*(?:[eE][-+]?\d+)?"


def parse_output(out_text: str, fa_text: str | None = None) -> dict:
    """Final energies, forces, stress and Hirshfeld charges from SIESTA output."""
    res = {}
    for key, pat in (("energy_total", r"siesta:\s+Total =\s+(" + _FLOAT + ")"),
                     ("free_energy", r"siesta: FreeEng =\s+(" + _FLOAT + ")"),
                     ("energy_d3", r"siesta: D3 dispersion =\s+(" + _FLOAT + ")")):
        m = re.findall(pat, out_text)
        res[key] = float(m[-1]) if m else None
    # stress (static), last occurrence, eV/Ang^3
    blocks = re.findall(r"siesta: Stress tensor \(static\) \(eV/Ang\*\*3\):\n"
                        r"((?:siesta:\s+.*\n){3})", out_text)
    if blocks:
        rows = [[float(x) for x in ln.split()[1:4]] for ln in blocks[-1].strip().splitlines()]
        res["stress"] = rows
    # forces: .FA file if given (final geometry), else last block in the output
    if fa_text:
        lines = fa_text.strip().splitlines()
        n = int(lines[0])
        res["forces"] = [[float(x) for x in ln.split()[1:4]] for ln in lines[1:1 + n]]
    else:
        fb = re.findall(r"siesta: Atomic forces \(eV/Ang\):\n((?:\s+\d+\s+.*\n)+)", out_text)
        if fb:
            res["forces"] = [[float(x) for x in ln.split()[1:4]]
                             for ln in fb[-1].strip().splitlines()]
    hb = re.findall(r"Hirshfeld Atomic Populations:\nAtom #.*\n((?:\s+\d+\s+.*\n)+)", out_text)
    if hb:
        res["hirshfeld_charges"] = [float(ln.split()[1]) for ln in hb[-1].strip().splitlines()]
    # "SCF Convergence by ..." is printed after every converged SCF cycle; with
    # the default (abort on failure) a non-converged SCF never reaches "Job completed"
    res["scf_converged"] = "SCF Convergence by" in out_text
    res["normal_exit"] = "Job completed" in out_text
    return res


def read_final_structure(run_dir: Path, label: str, atoms: Atoms) -> Atoms:
    """Final geometry of a relaxation from <label>.XV (Bohr)."""
    bohr = 0.529177210903
    lines = (Path(run_dir) / f"{label}.XV").read_text().split("\n")
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(3)]) * bohr
    n = int(lines[3])
    pos = np.array([[float(x) for x in lines[4 + i].split()[2:5]] for i in range(n)]) * bohr
    out = atoms.copy()
    out.set_cell(cell, scale_atoms=False)
    out.positions = pos
    return out


def run(atoms: Atoms, run_dir: str | Path, settings: SiestaSettings | None = None,
        processes: int = 4, label: str = "calc", relax: dict | None = None,
        resource_plan: dict | None = None, pseudo_dir=None,
        timeout_s: float | None = None) -> dict:  # pragma: no cover - runs SIESTA
    """Run SIESTA locally with `processes` MPI ranks; returns and writes result.json."""
    settings = settings or SiestaSettings()
    if pseudo_dir is None:
        pseudo_dir = settings.pseudo_dirs()
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    elements = sorted(set(atoms.get_chemical_symbols()))
    for el in elements:
        src = pseudo_file(el, pseudo_dir)
        shutil.copyfile(src, d / src.name)
    (d / f"{label}.fdf").write_text(write_fdf(atoms, label, settings, relax))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t = time.time()
    with open(d / f"{label}.out", "w") as out:
        proc = subprocess.run(["mpirun", "-np", str(processes), "siesta", f"{label}.fdf"],
                              cwd=d, stdout=out, stderr=subprocess.STDOUT, env=env,
                              timeout=timeout_s)
    wall = time.time() - t
    out_text = (d / f"{label}.out").read_text()
    fa = d / f"{label}.FA"
    res = parse_output(out_text, fa.read_text() if fa.exists() else None)
    res.update({
        "identifier": atoms.info.get("identifier"), "natoms": len(atoms),
        "symbols": atoms.get_chemical_symbols(), "returncode": proc.returncode,
        "wall_time_s": wall, "processes": processes, "settings": settings.as_dict(),
        "relax": relax, "siesta_version": _siesta_version(out_text),
        "pseudopotentials": {el: {"file": str(pseudo_file(el, pseudo_dir)),
                                  "sha256": pseudo_sha256(el, pseudo_dir)} for el in elements},
        "resource_plan": resource_plan,
    })
    if relax and (d / f"{label}.XV").exists():
        fin = read_final_structure(d, label, atoms)
        res["final_cell"] = fin.cell.array.tolist()
        res["final_positions"] = fin.positions.tolist()
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def _siesta_version(out_text: str) -> str | None:
    m = re.search(r"Version\s*:\s*(\S+)", out_text) or re.search(r"Siesta Version:\s*(\S+)", out_text)
    return m.group(1) if m else None
