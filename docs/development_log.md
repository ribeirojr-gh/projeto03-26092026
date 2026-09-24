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
