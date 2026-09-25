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

## Stage 9 - `chore/09-repository-infrastructure`
Repository infrastructure for publication on GitHub; no change to the
scientific code or to the committed results. Tests: 59 passed (cumulative).
* Restored from `ga-reaxff-v0.1.1.bundle` (SHA-256
  `fe67300fdc685f5f611084367bcbce46641b2606d48cd9c376acb7bed2f244ff`, matches
  the value recorded at the end of stage 8). All 56 existing tests passed on
  the new machine before any change.
* **Branch invariant repaired.** `develop` did not contain the three release
  commits on `main` (44dad77, 552f726, 7138daf). Back-merged `main` into
  `develop`; the release procedure in `CONTRIBUTING.md` now includes this step.
* **Version mismatch fixed.** `pyproject.toml` and `__init__.py` still said
  `0.1.0` after the `v0.1.1` tag. New `tests/test_metadata.py` fails whenever
  the package, `CHANGELOG.md` and `CITATION.cff` disagree on the version.
* Added `CONTRIBUTING.md` (branching model, per-stage protocol, releases,
  provenance rules), `CHANGELOG.md`, `CITATION.cff`, `CODEOWNERS`, PR and
  issue templates.
* **CI set to manual trigger.** On the GitHub free plan the jobs were refused
  before starting (account spending limit). GitHub is used for storage and
  structure only; the committed pytest logs are the test evidence. The
  workflow keeps the `pytest -v` log and `pip freeze` as artifacts if run.
* Tests now run against the locally compiled LAMMPS (22 Jul 2025 update 6,
  KOKKOS/CUDA build) instead of the PyPI wheel; same 59 tests pass.
* Environment note: without root access, the MPI runtime required by the
  LAMMPS wheel can be installed from PyPI (`pip install mpich`, then
  `LD_LIBRARY_PATH=$VIRTUAL_ENV/lib`).

## Stage 10 - `chore/10-github-pro-ci`
Account moved to GitHub Pro. Tests: 59 passed (cumulative; no code change).
* First automatic CI run on this branch succeeded (all steps green), which
  confirms that the stage-9 failures were only the free-plan spending limit.
* GitHub warned that checkout v4, setup-python v5 and upload-artifact v4
  target the deprecated Node 20 runtime; updated to checkout v5 and
  setup-python v6. upload-artifact v5 still triggered the warning (Node 20);
  v7 runs on Node 24.
* Branch protection enabled on `main` and `develop` (pull request with a
  passing `pytest` check required), now available on private repositories.
* Actions runners are used for CI and light batch jobs only: private
  repositories get 2-vCPU Linux runners without GPU; heavy LAMMPS and GPU
  work stays on the local workstation.

## Stage 11 - `feature/11-qmof-survey`
Change of target: from the graphene-oxide benchmark to ReaxFF force fields for
the MOFs of the QMOF database. Tests: 71 passed (cumulative; 12 new).
* **Data source identified.** The Materials Project "MOF Explorer" app is
  backed by the MPContribs project `mofexplorer` (Rosen, Jablonka), built on
  QMOF: 20 375 MOFs. The metadata table (no structures) downloads in ~40 s;
  GCMC and hybrid-functional columns are dropped. Snapshot committed (3.9 MB
  gzip) with the SHA-256 of the uncompressed JSON lines, so the hash does not
  depend on gzip headers; the gzip itself is written with mtime 0 and is
  byte-identical across runs. The API key is read from the pymatgen settings
  and never written to the repository.
* **Unit of work changed from "one force field per MOF" to "one force field
  per chemical family".** ReaxFF parameters belong to elements, so a force
  field with element set E covers every MOF whose elements are a subset of E.
  Coverage is computed with greedy maximum coverage.
* **Non-determinism found and fixed.** The first two runs of the survey
  produced different `summary.json` files: tied counts (`Counter.most_common`)
  were ordered by string-hash iteration order, which Python randomizes per
  process. All rankings now break ties by key; outputs verified byte-identical
  for PYTHONHASHSEED 1, 2, 3, and a test covers the tie case.
* **A claim corrected before commit.** The draft said the Zn family contains
  MOF-74 and ZIF-8; checking the data showed the QMOF MOF-74 set is Mg only,
  and ZIF-8 refcodes are not in the Zn + C/H/N/O family. The text now
  reports the verified topology counts.
* Results and pilot-family choice: `docs/results_qmof_survey.md`.

## Stage 12 - `feature/12-base-forcefield`
Base force field for the pilot family Zn + C/H/N/O. Tests: 80 passed
(cumulative; 9 new). Target applications fixed by the project owner: water
splitting (H2 production) and CO2 capture/selectivity/splitting in CH4/CO2
mixtures.
* **Sources.** Only files distributed with LAMMPS were used (hashes and
  references in `data/ffields/sources/`); none contains Zn-N or Zn-C terms.
  `ffield.reax.lg` and `.rdx` are not readable by `ffield.py` (different
  format); the HNS example file has no citation and was not used.
* **`merge_elements`** copies the donor's atom block and every term that
  involves the new element and only elements of the merged file; general
  and shared-element parameters stay those of the base, and all differences
  are reported rather than resolved silently.
* **Found: dummy atom type "X".** ZnOH and FC contain an atom type literally
  named X, the same label `ffield.py` uses for the torsion wildcard. It only
  appears in torsions in these files, so reading and writing are correct;
  `write` now raises if the label appears elsewhere, the merge report skips
  it, and `missing_interactions` ignores it.
* **Found: masses were hard-coded for C/H/O/N/S** in `engine.py`, so Zn and F
  failed with a KeyError on the first screening run. Masses now come from
  ASE for other elements (values for C/H/O/N/S unchanged).
* **Screening result** (`docs/results_base_ffield.md`): FC + Zn keeps the Zn
  cluster four-coordinated, describes H2 and H2O best, and differs from the
  Zn donor in only 2 of 39 general parameters (18 for budzien, 12 for
  mattsson). CO2 is too long in every candidate (FC: C=O 1.253 A vs 1.160 A),
  so the C-O terms join the Zn terms in the first fit.
* Build outputs verified identical across two runs.
