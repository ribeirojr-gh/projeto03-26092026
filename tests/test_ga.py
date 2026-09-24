"""Stage 6 tests: GA operators (statistical properties) and optimizer behaviour.

These tests use analytic functions only, so the optimizer is validated
independently of LAMMPS.
"""
import numpy as np
import pytest

from ga_reaxff.ga import (GAConfig, GeneticAlgorithm, latin_hypercube, local_refine,
                          polynomial_mutation, sbx_crossover, tournament_select)

OPT = np.array([0.2, 0.8, 0.35, 0.6, 0.5])


def sphere(x):
    return float(np.sum((x - OPT) ** 2))


def rastrigin(x):
    """Highly multimodal; global minimum 0 at OPT (scaled so ~10 local minima per axis)."""
    z = 5.12 * (x - OPT) / 2
    return float(10 * len(z) + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))


# ---------------------------------------------------------------- operators
def test_lhs_stratification():
    rng = np.random.default_rng(0)
    u = latin_hypercube(20, 4, rng)
    for j in range(4):
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))


def test_tournament_prefers_better():
    rng = np.random.default_rng(1)
    losses = np.arange(10.0)
    picks = [tournament_select(losses, 3, rng) for _ in range(5000)]
    counts = np.bincount(picks, minlength=10)
    assert counts[0] > counts[4] > counts[9]
    assert counts[9] == 0 or counts[9] < 50  # worst wins only if drawn 3 times


def test_sbx_preserves_mean_and_bounds():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        p1, p2 = rng.random(6), rng.random(6)
        c1, c2 = sbx_crossover(p1, p2, 15.0, rng)
        assert np.all((0 <= c1) & (c1 <= 1) & (0 <= c2) & (c2 <= 1))
        inside = (c1 > 0) & (c1 < 1) & (c2 > 0) & (c2 < 1)   # not clipped
        assert np.allclose((c1 + c2)[inside], (p1 + p2)[inside])


def test_polynomial_mutation_rate_and_bounds():
    rng = np.random.default_rng(3)
    x = np.full((20000, 5), 0.5)
    y = np.array([polynomial_mutation(r, 20.0, 0.2, rng) for r in x])
    changed = np.mean(y != x)
    assert changed == pytest.approx(0.2, abs=0.01)
    assert np.all((y >= 0) & (y <= 1))
    # at a bound, mutation stays feasible
    z = np.array([polynomial_mutation(np.array([0.0, 1.0]), 5.0, 1.0, rng) for _ in range(2000)])
    assert np.all((z >= 0) & (z <= 1))


# ---------------------------------------------------------------- algorithm
def test_ga_then_local_refine_solves_sphere():
    """Division of labour: GA finds the basin, Nelder-Mead polishes it.

    With the exploration-oriented default (eta_m = 5) the GA alone stops at
    ~1e-4 on the sphere; the local step must bring it to < 1e-8.
    """
    res = GeneticAlgorithm(GAConfig(pop_size=30, n_generations=60, seed=4)).run(sphere, 5)
    assert res.best_loss < 1e-3
    assert np.allclose(res.best_genome, OPT, atol=0.02)
    x, f, _ = local_refine(sphere, res.best_genome, maxiter=1000, xatol=1e-6, fatol=1e-12)
    assert f < 1e-8 and np.allclose(x, OPT, atol=1e-3)


def test_ga_finds_global_minimum_of_rastrigin():
    """Statistical test: global minimum in >= 5 of 6 independent seeds.

    Seeds 200-205 are disjoint from the seeds (100-111) used to tune the
    default operators, so this is an out-of-sample check.
    """
    hits = 0
    for s in range(200, 206):
        res = GeneticAlgorithm(GAConfig(pop_size=60, n_generations=120, seed=s)).run(rastrigin, 5)
        hits += res.best_loss < 0.1 and np.allclose(res.best_genome, OPT, atol=0.05)
    assert hits >= 5


def test_elitism_monotonic_best():
    res = GeneticAlgorithm(GAConfig(pop_size=20, n_generations=30, seed=6)).run(rastrigin, 5)
    best = [h["best"] for h in res.history]
    assert all(b2 <= b1 + 1e-15 for b1, b2 in zip(best, best[1:]))


def test_reproducible_with_seed_and_sensitive_to_seed():
    run = lambda s: GeneticAlgorithm(GAConfig(pop_size=16, n_generations=10, seed=s)).run(rastrigin, 5)
    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a.best_genome, b.best_genome) and a.history == b.history
    assert not np.array_equal(a.best_genome, c.best_genome)


def test_injected_individual_bounds_the_result():
    start = OPT + 0.01
    res = GeneticAlgorithm(GAConfig(pop_size=10, n_generations=0, seed=9, inject=[start])).run(sphere, 5)
    assert res.best_loss <= sphere(start)


def test_patience_stops_early():
    res = GeneticAlgorithm(GAConfig(pop_size=10, n_generations=500, seed=10, patience=5)).run(lambda x: 1.0, 3)
    assert len(res.history) == 6


def test_invalid_config():
    with pytest.raises(ValueError):
        GeneticAlgorithm(GAConfig(pop_size=7)).run(sphere, 5)
    with pytest.raises(ValueError):
        GeneticAlgorithm(GAConfig(pop_size=10, n_elite=10)).run(sphere, 5)


def test_local_refine_improves_and_respects_bounds():
    x0 = OPT + 0.05
    x, f, info = local_refine(sphere, x0, maxiter=500)
    assert f < sphere(x0) * 1e-3 and np.all((x >= 0) & (x <= 1))
    # never worse than the start, even from a bound
    x, f, _ = local_refine(sphere, np.ones(5), maxiter=5)
    assert f <= sphere(np.ones(5))
