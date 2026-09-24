"""Reading, writing and addressing ReaxFF force-field files (LAMMPS "ffield" format).

Design
------
A ReaxFF file is a sequence of blocks (general, atoms, bonds, off-diagonal,
angles, torsions, hydrogen bonds). Each block entry is stored as a small
record holding the element labels and a numpy vector of floats. Every single
number in the file therefore has a unique, human-readable *key*::

    general:<index>                  e.g. general:0          (p_boc1)
    atom:<El>:<name>                 e.g. atom:O:r_s
    bond:<El1>-<El2>:<name>          e.g. bond:C-O:De_sigma
    offdiag:<El1>-<El2>:<name>       e.g. offdiag:C-O:r_vdw
    angle:<El1>-<El2>-<El3>:<name>   e.g. angle:C-O-C:theta_00
    torsion:<El1>-..-<El4>:<name>    e.g. torsion:X-C-C-X:V2   (X = wildcard 0)
    hbond:<El1>-<El2>-<El3>:<name>   e.g. hbond:O-H-O:r0_hb

Keys make the optimization auditable: logs and result files state exactly
which physical parameter changed, not an anonymous vector index.

Symmetric entries (bonds, off-diagonals) can be addressed in either order;
angles are symmetric under reversal of the outer atoms (A-B-C == C-B-A),
torsions under full reversal (A-B-C-D == D-C-B-A).

Formatting: LAMMPS tokenizes the numeric lines, so column alignment is
cosmetic. Values are written with 6 decimals (the reference file uses 4),
so write -> read round trips are exact for the original data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parameter names per block (order = order in the file). "nu*" = not used.
# Names follow the LAMMPS reaxff documentation / van Duin's conventions.
# ---------------------------------------------------------------------------
ATOM_NAMES = [
    "r_s", "valency", "mass", "r_vdw", "epsilon", "gamma", "r_pi", "valency_e",
    "alpha", "gamma_w", "valency_boc", "p_ovun5", "nu1", "chi", "eta", "p_hbond",
    "r_pi_pi", "p_lp2", "nu2", "b_o_131", "b_o_132", "b_o_133", "nu3", "nu4",
    "p_ovun2", "p_val3", "nu5", "valency_val", "p_val5", "rcore2", "ecore2", "acore2",
]
BOND_NAMES = [
    "De_sigma", "De_pi", "De_pipi", "p_be1", "p_bo5", "v13cor", "p_bo6", "p_ovun1",
    "p_be2", "p_bo3", "p_bo4", "nu1", "p_bo1", "p_bo2", "ovc", "nu2",
]
OFFDIAG_NAMES = ["D", "r_vdw", "alpha", "r_s", "r_p", "r_pp"]
ANGLE_NAMES = ["theta_00", "p_val1", "p_val2", "p_coa1", "p_val7", "p_pen1", "p_val4"]
TORSION_NAMES = ["V1", "V2", "V3", "p_tor1", "p_cot1", "nu1", "nu2"]
HBOND_NAMES = ["r0_hb", "p_hb1", "p_hb2", "p_hb3"]

BLOCK_NAMES = {
    "atom": ATOM_NAMES, "bond": BOND_NAMES, "offdiag": OFFDIAG_NAMES,
    "angle": ANGLE_NAMES, "torsion": TORSION_NAMES, "hbond": HBOND_NAMES,
}
WILDCARD = "X"  # label used for atom-type index 0 in torsions


@dataclass
class Entry:
    """One record of a block: element labels + parameter vector."""
    labels: tuple[str, ...]
    values: np.ndarray


@dataclass
class ForceField:
    """In-memory ReaxFF force field.

    Attributes
    ----------
    header : first (free-text) line of the file, kept verbatim.
    general : general parameters (vector) and their trailing comments.
    elements : element symbols in file order (type index = position + 1).
    blocks : mapping block name -> list of Entry.
    block_headers : the comment lines of each block, kept verbatim so the
        written file stays readable for humans.
    """
    header: str
    general: np.ndarray
    general_comments: list[str]
    elements: list[str]
    blocks: dict[str, list[Entry]] = field(default_factory=dict)
    block_headers: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def read(cls, path: str | Path) -> "ForceField":
        lines = Path(path).read_text().splitlines()
        it = iter(lines)
        header = next(it)

        # --- general parameters
        n_gen = int(next(it).split()[0])
        gen_vals, gen_comments = [], []
        for _ in range(n_gen):
            line = next(it)
            num, _, comment = line.partition("!")
            gen_vals.append(float(num.split()[0]))
            gen_comments.append(comment.strip())
        general = np.array(gen_vals)

        # --- atoms: count line + 3 comment lines, then 4 lines per atom
        count_line = next(it)
        n_atoms = int(count_line.split()[0])
        atom_hdr = [count_line] + [next(it) for _ in range(3)]
        elements, atom_entries = [], []
        for _ in range(n_atoms):
            first = next(it).split()
            sym = first[0]
            vals = [float(v) for v in first[1:9]]
            for _ in range(3):
                vals += [float(v) for v in next(it).split()[:8]]
            elements.append(sym)
            atom_entries.append(Entry((sym,), np.array(vals)))

        def label(i: int) -> str:
            return WILDCARD if i == 0 else elements[i - 1]

        # --- bonds: count line + 1 comment line, 2 lines per entry
        count_line = next(it)
        n = int(count_line.split()[0])
        bond_hdr = [count_line, next(it)]
        bonds = []
        for _ in range(n):
            t1 = next(it).split()
            t2 = next(it).split()
            i, j = int(t1[0]), int(t1[1])
            vals = [float(v) for v in t1[2:10]] + [float(v) for v in t2[:8]]
            bonds.append(Entry((label(i), label(j)), np.array(vals)))

        def simple_block(n_idx: int, n_vals: int):
            count_line = next(it)
            n = int(count_line.split()[0])
            out = []
            for _ in range(n):
                t = next(it).split()
                labs = tuple(label(int(x)) for x in t[:n_idx])
                out.append(Entry(labs, np.array([float(v) for v in t[n_idx:n_idx + n_vals]])))
            return [count_line], out

        off_hdr, offdiag = simple_block(2, 6)
        ang_hdr, angles = simple_block(3, 7)
        tor_hdr, torsions = simple_block(4, 7)
        hb_hdr, hbonds = simple_block(3, 4)

        return cls(
            header=header, general=general, general_comments=gen_comments,
            elements=elements,
            blocks={"atom": atom_entries, "bond": bonds, "offdiag": offdiag,
                    "angle": angles, "torsion": torsions, "hbond": hbonds},
            block_headers={"atom": atom_hdr, "bond": bond_hdr, "offdiag": off_hdr,
                           "angle": ang_hdr, "torsion": tor_hdr, "hbond": hb_hdr},
        )

    def write(self, path: str | Path) -> Path:
        """Write the force field in LAMMPS-readable ffield format."""
        idx = {el: i + 1 for i, el in enumerate(self.elements)}
        idx[WILDCARD] = 0

        def fmt(vals) -> str:
            return "".join(f"{v:12.6f}" for v in vals)

        def ids(labels) -> str:
            return "".join(f"{idx[l]:3d}" for l in labels)

        out = [self.header, f"{len(self.general):3d}       ! Number of general parameters"]
        for v, c in zip(self.general, self.general_comments):
            out.append(f"{v:12.6f} !{c}")

        # atoms: header lines are rewritten with the (unchanged) count
        hdr = self.block_headers["atom"]
        out.append(_replace_count(hdr[0], len(self.blocks["atom"])))
        out += hdr[1:]
        for e in self.blocks["atom"]:
            v = e.values
            out.append(f" {e.labels[0]:<2s}" + fmt(v[0:8]))
            for k in (8, 16, 24):
                out.append("   " + fmt(v[k:k + 8]))

        hdr = self.block_headers["bond"]
        out.append(_replace_count(hdr[0], len(self.blocks["bond"])))
        out += hdr[1:]
        for e in self.blocks["bond"]:
            out.append(ids(e.labels) + fmt(e.values[:8]))
            out.append("      " + fmt(e.values[8:]))

        for name in ("offdiag", "angle", "torsion", "hbond"):
            out.append(_replace_count(self.block_headers[name][0], len(self.blocks[name])))
            for e in self.blocks[name]:
                out.append(ids(e.labels) + fmt(e.values))

        path = Path(path)
        path.write_text("\n".join(out) + "\n")
        return path

    # ------------------------------------------------------------ addressing
    def _locate(self, key: str) -> tuple[np.ndarray, int]:
        """Return (vector, index) that stores the parameter named by `key`."""
        parts = key.split(":")
        block = parts[0]
        if block == "general":
            if len(parts) != 2:
                raise KeyError(f"malformed key {key!r}")
            return self.general, int(parts[1])
        if block not in BLOCK_NAMES or len(parts) != 3:
            raise KeyError(f"malformed key {key!r}")
        labels = tuple(parts[1].split("-"))
        names = BLOCK_NAMES[block]
        if parts[2] not in names:
            raise KeyError(f"unknown parameter {parts[2]!r} for block {block!r}")
        k = names.index(parts[2])
        for e in self.blocks[block]:
            if _same_labels(block, e.labels, labels):
                return e.values, k
        raise KeyError(f"no {block} entry for labels {labels}")

    def get(self, key: str) -> float:
        vec, k = self._locate(key)
        return float(vec[k])

    def set(self, key: str, value: float) -> None:
        vec, k = self._locate(key)
        vec[k] = float(value)

    def copy(self) -> "ForceField":
        return copy.deepcopy(self)

    def with_values(self, keys: list[str], values) -> "ForceField":
        """Return a copy with the given parameters replaced (self untouched)."""
        ff = self.copy()
        for k, v in zip(keys, values):
            ff.set(k, v)
        return ff

    def all_keys(self) -> list[str]:
        """Every addressable parameter key (useful for auditing / diffs)."""
        keys = [f"general:{i}" for i in range(len(self.general))]
        for block, entries in self.blocks.items():
            for e in entries:
                lab = "-".join(e.labels)
                keys += [f"{block}:{lab}:{n}" for n in BLOCK_NAMES[block]]
        return keys

    def diff(self, other: "ForceField", tol: float = 1e-9) -> dict[str, tuple[float, float]]:
        """Parameters that differ between two force fields: key -> (self, other)."""
        out = {}
        for k in self.all_keys():
            a, b = self.get(k), other.get(k)
            if abs(a - b) > tol:
                out[k] = (a, b)
        return out


def _same_labels(block: str, a: tuple, b: tuple) -> bool:
    if a == b:
        return True
    # all multi-body terms are invariant under reversal of the label sequence
    return block in ("bond", "offdiag", "angle", "torsion") and a == tuple(reversed(b))


def _replace_count(line: str, n: int) -> str:
    _, _, rest = line.partition("!")
    return f"{n:3d}    !{rest}" if rest else f"{n:3d}"
