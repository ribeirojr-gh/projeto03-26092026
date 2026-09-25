"""Chemical families of MOFs, as seen by a ReaxFF parametrization.

ReaxFF parameters belong to elements and to pairs/triplets/quadruplets of
elements, not to individual structures. A force field with element set E can
therefore describe every MOF whose element set is a subset of E. This module
groups MOFs accordingly:

    split_elements   metals vs. non-metals of a chemical system
    Family           one metal set + one non-metal set (the key of a force field)
    coverage         which MOFs a given force-field element set can describe
    greedy_cover     choose force-field element sets that cover the most MOFs
    node_types       secondary building units (MOFid nodes) per metal

"Metal" follows pymatgen's `Element.is_metal`; metalloids (B, Si, Ge, As, Sb,
Te) are counted with the non-metals, because in MOFs they sit in linkers or
counter-ions and are handled by the organic part of a force field.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=None)
def is_metal(symbol: str) -> bool:
    from pymatgen.core import Element
    return bool(Element(symbol).is_metal)


def elements(chemsys: str) -> frozenset[str]:
    """'C-H-O-Zn' -> {'C', 'H', 'O', 'Zn'}."""
    return frozenset(s for s in chemsys.split("-") if s)


def split_elements(els) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sorted (metals, non-metals)."""
    els = sorted(els)
    return (tuple(e for e in els if is_metal(e)),
            tuple(e for e in els if not is_metal(e)))


@dataclass(frozen=True)
class Family:
    metals: tuple[str, ...]
    nonmetals: tuple[str, ...]

    @classmethod
    def from_chemsys(cls, chemsys: str) -> "Family":
        return cls(*split_elements(elements(chemsys)))

    @property
    def key(self) -> str:
        """'Zn|C-H-N-O'; metal-free frameworks get '-|...'."""
        return f"{'-'.join(self.metals) or '-'}|{'-'.join(self.nonmetals)}"

    @property
    def elements(self) -> frozenset[str]:
        return frozenset(self.metals + self.nonmetals)


def coverage(ff_elements, mof_element_sets) -> list[int]:
    """Indices of the MOFs whose element set is contained in `ff_elements`."""
    ff = frozenset(ff_elements)
    return [i for i, s in enumerate(mof_element_sets) if s <= ff]


def greedy_cover(mof_element_sets, candidates, n_max: int | None = None):
    """Greedy maximum coverage.

    Repeatedly picks the candidate element set that describes the largest
    number of not-yet-covered MOFs. Returns a list of
    (candidate, newly_covered, cumulative_covered). Greedy is within a factor
    (1 - 1/e) of the optimal coverage for any number of picks, which is enough
    to rank force fields by how much of the database they unlock.
    """
    sets = [frozenset(s) for s in mof_element_sets]
    cands = sorted({frozenset(c) for c in candidates}, key=lambda c: sorted(c))
    covered: set[int] = set()
    picks = []
    while cands and (n_max is None or len(picks) < n_max):
        best, best_new = None, set()
        for c in cands:
            new = {i for i in coverage(c, sets) if i not in covered}
            if len(new) > len(best_new):
                best, best_new = c, new
        if not best_new:
            break
        covered |= best_new
        cands.remove(best)
        picks.append((tuple(sorted(best)), len(best_new), len(covered)))
    return picks


def split_smiles_list(s: str | None) -> list[str]:
    """MOFid node/linker fields are comma-separated SMILES."""
    if not s:
        return []
    return [x for x in s.split(",") if x]


def node_types(rows, metal: str, top: int = 10) -> list[tuple[str, int]]:
    """Most common MOFid nodes among single-metal MOFs of `metal`.

    Ties are broken alphabetically, so the result does not depend on string
    hash randomization.
    """
    c = Counter()
    for r in rows:
        fam = Family.from_chemsys(r["chemsys"])
        if fam.metals == (metal,):
            c.update(set(split_smiles_list(r.get("mofid.smilesNodes"))))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
