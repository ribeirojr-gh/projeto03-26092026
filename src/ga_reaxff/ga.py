"""Real-coded genetic algorithm on the unit cube, plus local refinement.

Algorithm (per generation)
--------------------------
1. Elitism: the `n_elite` best individuals are copied unchanged.
2. Selection: tournament of size `tournament_k` (minimization).
3. Crossover: simulated binary crossover, SBX (Deb & Agrawal 1995), with
   probability `p_crossover`, distribution index `eta_c`.
4. Mutation: bounded polynomial mutation (Deb & Goyal 1996), per-gene
   probability `p_mutation` (default 1/d), distribution index `eta_m`.
5. Evaluation of the new individuals (cached by the objective).

Initialization: Latin hypercube sampling of the unit cube; optionally the
initial force field (genome 0.5 when bounds are symmetric) is injected as
one individual so the GA can never end worse than the starting point.

Everything random flows from ONE numpy Generator seeded by `seed`, so a run
is exactly reproducible given (config, seed, objective).

Why hand-written instead of a library: each operator is ~15 lines, has its
own unit test, and the full algorithm can be audited without reading a
third-party code base. The objective interface (genome -> float) is the same
one used by pymoo/DEAP, so switching later is trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass
class GAConfig:
    pop_size: int = 40
    n_generations: int = 40
    tournament_k: int = 3
    p_crossover: float = 0.9
    eta_c: float = 15.0
    p_mutation: float | None = None     # None -> 1/d
    eta_m: float = 5.0                  # tuned: see docs/development_log.md (stage 6)
    n_elite: int = 2
    seed: int = 12345
    init: str = "lhs"                   # "lhs" or "uniform"
    inject: list | None = None          # genomes to insert in the initial population
    patience: int | None = None         # stop after this many generations without improvement
    tol: float = 1e-12

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["inject"] is not None:
            d["inject"] = [list(map(float, g)) for g in d["inject"]]
        return d


@dataclass
class GAResult:
    best_genome: np.ndarray
    best_loss: float
    history: list = field(default_factory=list)
    final_population: np.ndarray | None = None
    final_losses: np.ndarray | None = None


# ------------------------------------------------------------------ operators
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0,1]^d with exactly one point per 1/n stratum in every dimension."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def tournament_select(losses: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Index of the best (lowest loss) among k individuals drawn with replacement."""
    idx = rng.integers(0, len(losses), size=k)
    return int(idx[np.argmin(losses[idx])])


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator,
                  p_gene: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover; each gene crosses with probability p_gene.

    Children satisfy c1 + c2 = p1 + p2 before clipping to [0, 1].
    """
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    mask = rng.random(p1.shape) < p_gene
    beta = np.where(mask, beta, 1.0)
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def polynomial_mutation(x: np.ndarray, eta: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """Bounded polynomial mutation on [0, 1]; the result never leaves the bounds."""
    y = x.copy()
    mask = rng.random(x.shape) < p
    if not mask.any():
        return y
    u = rng.random(x.shape)
    d1, d2 = y, 1.0 - y                  # distances to lower / upper bound
    mpow = 1.0 / (eta + 1.0)
    left = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** mpow - 1
    right = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** mpow
    delta = np.where(u < 0.5, left, right)
    y = np.where(mask, y + delta, y)
    return np.clip(y, 0.0, 1.0)


# ---------------------------------------------------------------- algorithm
class GeneticAlgorithm:
    def __init__(self, config: GAConfig):
        self.cfg = config

    def run(self, objective: Callable[[np.ndarray], float], dim: int,
            callback: Callable[[dict], None] | None = None) -> GAResult:
        cfg = self.cfg
        if cfg.pop_size < 4 or cfg.pop_size % 2:
            raise ValueError("pop_size must be even and >= 4")
        if not 0 <= cfg.n_elite < cfg.pop_size:
            raise ValueError("n_elite must be in [0, pop_size)")
        rng = np.random.default_rng(cfg.seed)
        pm = cfg.p_mutation if cfg.p_mutation is not None else 1.0 / dim

        pop = latin_hypercube(cfg.pop_size, dim, rng) if cfg.init == "lhs" else rng.random((cfg.pop_size, dim))
        for k, g in enumerate(cfg.inject or []):
            pop[k] = np.clip(np.asarray(g, float), 0, 1)
        losses = np.array([objective(g) for g in pop])

        history, best_so_far, stall = [], np.inf, 0
        for gen in range(cfg.n_generations + 1):
            order = np.argsort(losses, kind="stable")
            pop, losses = pop[order], losses[order]
            rec = {"generation": gen, "best": float(losses[0]), "mean": float(np.mean(losses)),
                   "median": float(np.median(losses)), "std": float(np.std(losses)),
                   "diversity": float(np.mean(np.std(pop, axis=0))),
                   "best_genome": pop[0].tolist()}
            history.append(rec)
            if callback:
                callback(rec)

            if losses[0] < best_so_far - cfg.tol:
                best_so_far, stall = losses[0], 0
            else:
                stall += 1
            if gen == cfg.n_generations or (cfg.patience and stall >= cfg.patience):
                break

            # --- build next generation
            children = [pop[i].copy() for i in range(cfg.n_elite)]
            while len(children) < cfg.pop_size:
                a = pop[tournament_select(losses, cfg.tournament_k, rng)]
                b = pop[tournament_select(losses, cfg.tournament_k, rng)]
                if rng.random() < cfg.p_crossover:
                    c1, c2 = sbx_crossover(a, b, cfg.eta_c, rng)
                else:
                    c1, c2 = a.copy(), b.copy()
                children.append(polynomial_mutation(c1, cfg.eta_m, pm, rng))
                if len(children) < cfg.pop_size:
                    children.append(polynomial_mutation(c2, cfg.eta_m, pm, rng))
            new = np.array(children)
            new_losses = np.empty(cfg.pop_size)
            new_losses[:cfg.n_elite] = losses[:cfg.n_elite]          # elites are not re-evaluated
            for i in range(cfg.n_elite, cfg.pop_size):
                new_losses[i] = objective(new[i])
            pop, losses = new, new_losses

        return GAResult(pop[0].copy(), float(losses[0]), history, pop, losses)


def local_refine(objective: Callable[[np.ndarray], float], x0: np.ndarray,
                 maxiter: int = 200, xatol: float = 1e-4, fatol: float = 1e-8) -> tuple[np.ndarray, float, dict]:
    """Bounded Nelder-Mead polish of a GA solution (derivative-free: ReaxFF is not smooth).

    Returns (x, f, info); never returns a point worse than x0.
    """
    f0 = objective(x0)
    res = minimize(objective, x0, method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(x0),
                   options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol,
                            "initial_simplex": _simplex(x0, 0.05)})
    x, f = (np.clip(res.x, 0, 1), float(res.fun)) if res.fun < f0 else (x0.copy(), float(f0))
    return x, f, {"nit": int(res.nit), "nfev": int(res.nfev), "success": bool(res.success),
                  "message": str(res.message), "f_start": float(f0), "f_end": f}


def _simplex(x0: np.ndarray, step: float) -> np.ndarray:
    """Initial simplex inside the unit cube (steps point inward near a bound)."""
    s = [x0.copy()]
    for j in range(len(x0)):
        v = x0.copy()
        v[j] += step if x0[j] + step <= 1 else -step
        s.append(v)
    return np.array(s)
