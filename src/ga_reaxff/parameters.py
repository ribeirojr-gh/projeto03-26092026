"""Parameter space: which ReaxFF parameters form the GA genome, and their bounds.

The GA works on a normalized genome g in [0, 1]^d. Each gene maps linearly to
one physical parameter:

    p_i = lower_i + g_i * (upper_i - lower_i)

Working in the unit cube makes crossover / mutation operators scale-free
(parameters range from ~0.1 to ~200 in ReaxFF), and makes the bounds an
explicit, logged part of the experiment rather than an implicit choice.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .ffield import ForceField


@dataclass(frozen=True)
class ParameterSpec:
    key: str        # ffield key, e.g. "bond:C-O:De_sigma"
    lower: float
    upper: float

    def __post_init__(self):
        if not self.upper > self.lower:
            raise ValueError(f"{self.key}: upper ({self.upper}) must exceed lower ({self.lower})")


class ParameterSpace:
    """Ordered list of optimized parameters with bounds."""

    def __init__(self, specs: list[ParameterSpec]):
        keys = [s.key for s in specs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate parameter keys in parameter space")
        self.specs = list(specs)
        self.keys = keys
        self.lower = np.array([s.lower for s in specs])
        self.upper = np.array([s.upper for s in specs])

    def __len__(self) -> int:
        return len(self.specs)

    # --------------------------------------------------------- construction
    @classmethod
    def relative(cls, ff: ForceField, keys: list[str], rel: float = 0.3,
                 min_width: float = 0.05) -> "ParameterSpace":
        """Bounds = initial value +/- rel*|initial| (at least +/- min_width).

        `min_width` avoids a degenerate interval for parameters that are 0.
        """
        specs = []
        for k in keys:
            v = ff.get(k)
            half = max(rel * abs(v), min_width)
            specs.append(ParameterSpec(k, v - half, v + half))
        return cls(specs)

    @classmethod
    def from_config(cls, ff: ForceField, cfg: dict) -> "ParameterSpace":
        """Build from a config section:

        [parameters]
        rel = 0.3                       # default relative half-width
        keys = ["bond:C-O:De_sigma", ...]
        [parameters.bounds]             # optional absolute overrides
        "angle:C-O-C:theta_00" = [60.0, 90.0]
        """
        base = cls.relative(ff, list(cfg["keys"]), rel=cfg.get("rel", 0.3),
                            min_width=cfg.get("min_width", 0.05))
        overrides = cfg.get("bounds", {})
        unknown = set(overrides) - set(base.keys)
        if unknown:
            raise KeyError(f"bounds given for keys not in the parameter list: {sorted(unknown)}")
        specs = [ParameterSpec(s.key, *overrides[s.key]) if s.key in overrides else s
                 for s in base.specs]
        return cls(specs)

    # -------------------------------------------------------- transformation
    def decode(self, genome: np.ndarray) -> np.ndarray:
        """Unit-cube genome -> physical parameter values (clipped to bounds)."""
        g = np.clip(np.asarray(genome, float), 0.0, 1.0)
        return self.lower + g * (self.upper - self.lower)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Physical values -> unit-cube genome (no clipping: out-of-box shows up as <0 or >1)."""
        return (np.asarray(values, float) - self.lower) / (self.upper - self.lower)

    def values_from(self, ff: ForceField) -> np.ndarray:
        return np.array([ff.get(k) for k in self.keys])

    def apply(self, ff: ForceField, genome: np.ndarray) -> ForceField:
        """Copy of `ff` with the genome's parameters written in."""
        return ff.with_values(self.keys, self.decode(genome))

    def contains(self, ff: ForceField) -> bool:
        g = self.encode(self.values_from(ff))
        return bool(np.all((g >= 0) & (g <= 1)))

    def to_records(self) -> list[dict]:
        """Plain dicts for JSON audit logs."""
        return [asdict(s) for s in self.specs]


def perturb(ff: ForceField, keys: list[str], rel: float, rng: np.random.Generator) -> ForceField:
    """Multiply each parameter by (1 + u), u ~ U(-rel, rel) with |u| >= rel/2.

    Used to build synthetic "wrong" starting force fields for recovery tests.
    The |u| >= rel/2 floor guarantees every chosen parameter is really moved.
    """
    out = ff.copy()
    for k in keys:
        u = rng.uniform(rel / 2, rel) * rng.choice([-1.0, 1.0])
        out.set(k, ff.get(k) * (1.0 + u))
    return out
