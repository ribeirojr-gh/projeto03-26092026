# Development log (audit trail)

Each stage was developed on its own `feature/*` branch, tested with `pytest`,
the full test output was committed to `docs/test_logs/`, and only then merged
into `develop` with `--no-ff` (so every stage remains visible in the history:
`git log --graph --oneline --all`).

Environment: Python 3.12.3, LAMMPS 22 Jul 2025 (PyPI wheel, REAXFF + QEQ packages), MPICH 4, NumPy, SciPy, ASE.

## Stage 1 - `feature/01-ffield-io`
ReaxFF reader/writer with named keys. Tests: 8 passed.
* Issue found by the tests: the integrity test read the SHA-256 from the wrong
  place in `data/README.md`. Fixed by storing the hash in
  `data/ffield.reax.cho.sha256`.

## Stage 2 - `feature/02-parameter-space`
Unit-cube genome, bounds, perturbation helper. Tests: 16 passed (cumulative).

## Stage 3 - `feature/03-structures-dataset`
GO model and training-set generators. Tests: 23 passed (cumulative).
* Review finding: one test raised the expected `ValueError` for the wrong
  reason (duplicate names instead of two family references). Split into two
  tests that match on the error message.

## Stage 4 - `feature/04-lammps-engine`
Persistent-instance LAMMPS engine. Tests: 31 passed (cumulative).
Findings recorded here because they affect the physics, not just the code:
1. **Persistent instance == fresh instance.** Re-issuing `pair_coeff` with a
   new ffield reproduces a freshly built instance to ~1e-8 kcal/mol, at ~6.5
   ms per single point instead of ~24 ms (37-atom GO, 1 core).
2. **Graphene lattice constant of Chenoweth 2008 C/H/O.** A C32 E(a) scan
   gives a_eq ~ 2.50-2.52 A (C-C ~ 1.45 A), not 2.46 A. Building GO at 2.46 A
   stored ~10 kcal/mol per C of strain (~330 kcal/mol in total). GO models are
   now built at the force field's own a_eq (`graphene_lattice_constant`).
3. **ReaxFF E(a) is not smooth** (irregular steps between grid points),
   consistent with bond-order cutoff/taper discontinuities. This supports a
   derivative-free global optimizer (GA) for the parametrization.
4. **Minimizer.** Default CG line search stalls at fmax ~ 2-5 kcal/mol/A on
   GO; the quadratic line search reaches < 1 kcal/mol/A.
5. **Force consistency.** Central finite differences (h = 1e-3 and 1e-4)
   agree with LAMMPS forces to < 0.01 kcal/mol/A everywhere except the
   hydroxyl O-H pair: an equal-and-opposite, h-independent offset of ~0.17
   kcal/mol/A (~0.0075 eV/A). This is a small genuine non-conservative
   contribution, below typical DFT force noise; the test tolerance is set to
   0.25 kcal/mol/A with this measurement documented.
6. **Failure containment.** Unphysical parameters (e.g. `bond:C-O:p_bo2 = -50`)
   make LAMMPS abort with `bondchk failed`; the engine converts this into
   `EvaluationError`, rebuilds the instance from the template force field,
   and later evaluations are unaffected. Some absurd parameters (e.g.
   `atom:O:eta = -100`) do not crash but give nonsensical energies; these
   are handled by the fitness function, not the engine.

## Stage 5 - `feature/05-fitness`
Relative-energy + force objective, penalty, caching. Tests: 39 passed (cumulative).
* The hand-computed reference value in `test_hand_computed_loss` was wrong
  (the test author counted 5 configurations; the rattle family also contains
  its unrattled reference, giving 6). The implementation was correct; the
  test now asserts the configuration count explicitly.
* Verified: reference parameters are an exact global zero of the loss on
  synthetic data (< 1e-6), loss grows monotonically along a parameter line,
  per-family constant energy shifts are invisible (relative energies), and
  crashing parameter sets return the penalty without breaking later calls.

## Stage 6 - `feature/06-genetic-algorithm`
Real-coded GA (LHS init, tournament, SBX, bounded polynomial mutation,
elitism) + bounded Nelder-Mead refinement. Tests: 51 passed (cumulative).
* **Premature convergence caught by the tests.** With the textbook default
  `eta_m = 20` the GA got trapped in a local minimum of a 5-D Rastrigin
  function (loss 1.99). Operator tuning over 12 seeds (100-111), pop 60,
  120 generations (script: `scripts/tune_ga_operators.py`):

  | tournament k | eta_m | p_mut | success (loss < 0.1) | median best |
  |---|---|---|---|---|
  | 3 | 5  | 1/d | **12/12** | 0.001 |
  | 3 | 10 | 1/d | 9/12 | 0.004 |
  | 3 | 20 | 1/d | 0/12 | 2.985 |
  | 2 | 5  | 1/d | 9/12 | 0.027 |
  | 3 | 5  | 0.4 | 3/12 | 0.382 |

  New default: `eta_m = 5`, `k = 3`, `p_mut = 1/d`. The Rastrigin test is now
  statistical (>= 5/6 successes) on seeds 200-205, disjoint from the tuning
  seeds (out-of-sample).
* **Exploration/exploitation trade-off.** The wider mutation makes the GA
  alone less precise on a smooth bowl (sphere: 1.6e-4 instead of < 1e-4).
  This is the intended division of labour: the GA locates the basin and
  `local_refine` polishes it (sphere: < 1e-8). Tested explicitly.

## Stage 7 - `feature/07-pipeline-go-benchmark`
Audit records, pipeline, CLI, benchmark. Tests: 56 passed (cumulative).
* Operational finding: background processes are killed when the controlling
  session ends; the first two benchmark attempts were interrupted (at
  generation 3 and 20). The third run completed. Their partial histories
  matched the completed run value-for-value, an unplanned reproducibility check.
* Benchmark results and interpretation: `docs/results_go_recovery.md`.

## Stage 8 - `feature/08-figures`
Figure script (`scripts/make_figures.py`) and four figures in `docs/figures/`.
* The script regenerates the unversioned training/validation sets; both
  matched the SHA-256 recorded in the run manifest (reproducibility check).
* **Finding: the reference force field has a cusp at the graphene lattice
  constant.** Pristine graphene, uniform in-plane strain, reference ffield:

  | strain | dE per atom (kcal/mol), C32 | C128 |
  |---|---|---|
  | +0.25 % | 0.590 | 0.590 |
  | +0.50 % | 1.079 | 1.079 |
  | +1.00 % | 1.726 | 1.726 |

  Doubling the strain multiplies dE by ~1.6-1.8 instead of 4 (harmonic): E(eps)
  is non-analytic (a kink) at a_eq, and ~10x stiffer than the harmonic estimate
  from graphene's biaxial modulus (Y = 340 N/m, nu = 0.17 -> ~0.16 kcal/mol per
  atom at +1 %). Identical per-atom values in C32 and C128 rule out a cell-size
  artifact. Under compression the sheet buckles (0.6-0.9 A) and the energy
  drops by ~6x once positions are relaxed.
* **Methodological consequence.** The `strain` family uses rigidly scaled
  (unrelaxed) configurations, so under compression it measures a physically
  irrelevant flat-sheet energy. It must be rebuilt with internal relaxation at
  each strain before DFT data are generated.
* **Correlated parameters seen directly.** In fig. 2b, `p_be1` stays near the
  truth until generation ~29 and then drifts to 37 % error, and `offdiag D`
  moves from ~2 % to 12 % during Nelder-Mead, while the loss keeps decreasing:
  flat directions of the loss in parameter space.
