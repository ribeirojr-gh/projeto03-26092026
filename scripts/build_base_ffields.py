"""Build and screen candidate base force fields for the Zn + C/H/N/O MOF family.

    python scripts/build_base_ffields.py [--out-ff data/ffields/base] [--out docs/base_ffield]

For each organic base (FC, budzien, mattsson) the Zn block of ffield.reax.ZnOH
(Raymand et al. 2010) is merged in with `ffield.merge_elements`. The merged
files, the merge reports and the list of missing Zn interactions are written,
then every candidate is screened with LAMMPS:

* gas molecules relevant to water splitting and CO2/CH4 separation
  (H2, O2, N2, CO, H2O, CO2, CH4, NH3) and benzene (linker core): relaxed
  geometry vs. experimental equilibrium geometry;
* a Zn(OH)2(H2O)2 cluster: does Zn stay four-coordinated, at what Zn-O
  distance.

The screen is a sanity check of the *starting point*, not a validation: the
Zn-N and Zn-C terms do not exist yet and are the first targets of the fit.
Outputs: <out-ff>/ffield.Zn-<base>, <out>/screen.json, <out>/screen.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

from ga_reaxff.engine import EvaluationError, relax
from ga_reaxff.ffield import ForceField, merge_elements, missing_interactions

SOURCES = Path(__file__).parents[1] / "data" / "ffields" / "sources"
BASES = ["FC", "budzien", "mattsson"]
BOX = 25.0

# Experimental equilibrium geometries (Angstrom, degrees), NIST CCCBDB.
REFERENCE = {
    "H2": {"H-H": 0.741},
    "N2": {"N-N": 1.098},
    "O2": {"O-O": 1.208},
    "CO": {"C-O": 1.128},
    "H2O": {"H-O": 0.958, "angle": 104.5},
    "CO2": {"C-O": 1.160, "angle": 180.0},
    "CH4": {"C-H": 1.087},
    "NH3": {"H-N": 1.012, "angle": 106.7},
    "C6H6": {"C-C": 1.397, "C-H": 1.084},
}
BOND_CUTOFF = 1.6  # Angstrom, to identify bonded pairs in the relaxed molecules


def boxed(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.cell = [BOX, BOX, BOX]
    a.pbc = True
    a.center()
    return a


def zn_cluster() -> Atoms:
    """Tetrahedral Zn(OH)2(H2O)2, Zn-O 2.0 A (start geometry)."""
    d = 2.0
    t = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / np.sqrt(3)
    sym, pos = ["Zn"], [[0, 0, 0]]
    for k, u in enumerate(t):
        o = d * u
        sym.append("O"), pos.append(o)
        perp = np.cross(u, [0, 0, 1] if abs(u[2]) < 0.9 else [1, 0, 0])
        perp /= np.linalg.norm(perp)
        h1 = o + 0.96 * (0.33 * u + 0.94 * perp)
        sym.append("H"), pos.append(h1)
        if k >= 2:  # two water ligands
            h2 = o + 0.96 * (0.33 * u - 0.94 * perp)
            sym.append("H"), pos.append(h2)
    return boxed(Atoms(sym, positions=pos))


def geometry(atoms: Atoms) -> dict[str, float]:
    """Mean bond length per element pair and the angle of triatomics / NH3."""
    out, d = {}, atoms.get_all_distances(mic=True)
    sym = atoms.get_chemical_symbols()
    pairs: dict[str, list[float]] = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if d[i, j] < BOND_CUTOFF:
                pairs.setdefault("-".join(sorted((sym[i], sym[j]))), []).append(d[i, j])
    out.update({k: float(np.mean(v)) for k, v in pairs.items()})
    heavy = [i for i, s in enumerate(sym) if s in ("O", "C", "N")]
    if len(atoms) in (3, 4) and len(heavy) >= 1:
        c = heavy[0] if sym.count(sym[heavy[0]]) == 1 else [i for i in range(len(sym))
                                                             if sym.count(sym[i]) == 1][0]
        others = [i for i in range(len(atoms)) if i != c][:2]
        out["angle"] = float(atoms.get_angle(others[0], c, others[1], mic=True))
    return out


def screen(ff: ForceField) -> dict:
    res = {"molecules": {}, "zn_cluster": None}
    for name, ref in REFERENCE.items():
        try:
            g = geometry(relax(boxed(molecule(name)), ff, ftol=1e-4, maxiter=5000))
            res["molecules"][name] = {k: {"reaxff": g.get(k), "exp": v,
                                          "error": None if g.get(k) is None else g[k] - v}
                                      for k, v in ref.items()}
        except EvaluationError as e:
            res["molecules"][name] = {"error": str(e)}
    at = relax(zn_cluster(), ff, ftol=1e-3, maxiter=5000)
    d = at.get_distances(0, [i for i, s in enumerate(at.get_chemical_symbols()) if s == "O"],
                         mic=True)
    res["zn_cluster"] = {"zn_o": sorted(float(x) for x in d), "coordination_2.5A":
                         int((d < 2.5).sum())}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-ff", default="data/ffields/base")
    ap.add_argument("--out", default="docs/base_ffield")
    a = ap.parse_args(argv)
    out_ff, out = Path(a.out_ff), Path(a.out)
    out_ff.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    donor_path = SOURCES / "ffield.reax.ZnOH"
    donor = ForceField.read(donor_path)
    summary = {}
    for base_name in BASES:
        base_path = SOURCES / f"ffield.reax.{base_name}"
        base = ForceField.read(base_path)
        header = (f"ga-reaxff base: {base_path.name} + Zn block of {donor_path.name} "
                  f"(merge_elements; base general and shared-element parameters kept)")
        ff, rep = merge_elements(base, donor, ["Zn"], header=header)
        path = ff.write(out_ff / f"ffield.Zn-{base_name}")
        summary[base_name] = {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "elements": [e for e in ff.elements],
            "added_from_donor": ["{}:{}".format(b, "-".join(l)) for b, l in rep.added],
            "general_differences": {str(k): v for k, v in rep.general_differences.items()},
            "shared_atom_differences": rep.shared_atom_differences,
            "missing": {k: ["-".join(t) for t in v]
                        for k, v in missing_interactions(ff, ["Zn"]).items()},
            "screen": screen(ff),
        }
    (out / "screen.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = ["| molecule | quantity | exp. | " + " | ".join(BASES) + " |",
             "|---|---|---|" + "---|" * len(BASES)]
    for name, ref in REFERENCE.items():
        for q, v in ref.items():
            cells = []
            for b in BASES:
                m = summary[b]["screen"]["molecules"][name]
                cells.append("fail" if "error" in m and not isinstance(m["error"], dict)
                             and q not in m else
                             "-" if m[q]["reaxff"] is None else f"{m[q]['reaxff']:.3f}")
            lines.append(f"| {name} | {q} | {v} | " + " | ".join(cells) + " |")
    lines += ["", "| Zn(OH)2(H2O)2 | " + " | ".join(BASES) + " |", "|---|" + "---|" * len(BASES)]
    lines.append("| Zn-O (A) | " + " | ".join(
        ", ".join(f"{x:.2f}" for x in summary[b]["screen"]["zn_cluster"]["zn_o"]) for b in BASES) + " |")
    lines.append("| O within 2.5 A | " + " | ".join(
        str(summary[b]["screen"]["zn_cluster"]["coordination_2.5A"]) for b in BASES) + " |")
    lines += ["", "| | " + " | ".join(BASES) + " |", "|---|" + "---|" * len(BASES)]
    lines.append("| general parameters differing from ZnOH | " + " | ".join(
        str(len(summary[b]["general_differences"])) for b in BASES) + " |")
    lines.append("| O / H atom parameters differing from ZnOH | " + " | ".join(
        f"{len(summary[b]['shared_atom_differences'].get('O', {}))} / "
        f"{len(summary[b]['shared_atom_differences'].get('H', {}))}" for b in BASES) + " |")
    lines.append("| missing Zn bonds | " + " | ".join(
        ", ".join(summary[b]["missing"]["bond"]) for b in BASES) + " |")
    lines.append("| missing Zn angles | " + " | ".join(
        str(len(summary[b]["missing"]["angle"])) for b in BASES) + " |")
    (out / "screen.md").write_text("\n".join(lines) + "\n")
    print((out / "screen.md").read_text())


if __name__ == "__main__":
    main()
