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

## Stage 13 - `feature/13-structures-mlip-teacher`
Structures of the Zn family and check of MLIP teachers. Tests: 88 passed
(cumulative; 8 new).
* **Structures.** 2 958 MOFs downloaded in 167 s with 8 threads; each carries
  per-atom DDEC6/CM5 charges, DDEC6 bond-order sums and magnetic moments.
  All neutral in DDEC6. Stored as deterministic gzipped extxyz with hash.
* **A test was wrong, not the data.** The first composition check compared
  `get_chemical_formula(mode="reduce")` strings; in ASE that mode only merges
  repeated symbols and does not divide by the greatest common divisor, so
  Zn2C24... did not match ZnC12... . The test now compares reduced counts.
* **GPU memory.** The first teacher run crashed with CUDA out-of-memory on a
  500-atom MOF (float64 + D3 on 8 GB). TorchScript wraps the CUDA error in a
  plain RuntimeError, so catching `torch.OutOfMemoryError` did not work;
  the script now catches the wrapped error and falls back to the CPU, and
  sets `expandable_segments`. In the final run no structure needed the
  fallback (1 260 of 1 260 evaluations on the GPU).
* **Environment.** `OMP_NUM_THREADS=1` is set in the shell, so the CPU
  fallback first ran on one thread; it now sets 16 threads itself.
* **Measurement bug fixed before the final run.** The atomic displacement
  after relaxation did not remove a rigid translation of the whole crystal
  (a zero mode of periodic relaxations). Fixed in `teacher.py` (and in the
  stage-14 scorecard) and the relaxations re-run; relaxed structures are
  now written to `runs/teacher_check/` for re-analysis.
* **Result** (`docs/results_teacher_check.md`): best model medium-mpa-0 -
  force rms 0.152 eV/A at the DFT minima (worst on C and N), energy spread
  13 meV/atom after per-element offsets, cell volume within 0.65 % (median)
  after relaxation. Good for cells and energy differences, not accurate
  enough as the only force reference: own DFT is needed for fine-tuning and
  for reaction paths.

## Stage 14 - `feature/14-triclinic-engine`
ReaxFF on real MOF cells and per-MOF validation of the initial force field.
Tests: 98 passed (cumulative; 10 new). Developed in a separate git worktree
while the stage-13 teacher check occupied the GPU; rebased on stage 13.
* **Triclinic cells.** 94 % of the Zn MOFs are triclinic and the engine only
  accepted orthogonal boxes. Structures are rotated into ASE's standard
  (lower-triangular) form for LAMMPS and forces, positions and cells rotated
  back. Verified: rigid rotation of a MOF (same energy, rotated forces to
  1e-5) and an equivalent sheared basis (identical energy and forces). The
  80 earlier tests, including the graphene-oxide pipeline, pass unchanged.
* **Found: NaN forces with a finite energy.** The stage-12 base (no C-Zn
  bond entry) gives a finite energy (+77 kcal/mol) but NaN forces on a Zn
  carboxylate MOF. An isolated Zn...CH4 pair at 1.8-3.6 A does not trigger
  it, so the bonded environment does. Adding a C-Zn bond entry fixes it
  (off-diagonal alone does not). `single_point` returned the NaN silently;
  it now raises EvaluationError. `add_placeholder_pairs` copies O-Zn
  entries to C-Zn and N-Zn as starting values.
* **Measurement fix.** The displacement metric now removes a rigid
  translation of the crystal (same fix as in stage 13).
* **Result** (`docs/results_baseline_validation.md`): 228 MOFs relaxed in
  10 min on 12 cores, no LAMMPS failure; 96 pass (42 %). Failures are volume
  expansions (129 of 132) caused by bond lengths, not by missing dispersion:
  carboxylate C-O +6.2 %, Zn-O +3-4 %, N-H +13 %, consistent with the gas
  molecules of stage 12 (CO2 C=O +8 %, NH3 N-H +13 %). Zn-O-only frameworks
  expand most (+11 %, 2 of 46 pass).

## Stage 15 - `chore/15-rename-and-run-policy`
Repository renamed and run policy fixed by the project owner. Tests: 102
passed (cumulative; 4 new).
* Repository renamed on GitHub to `projeto03-26092026` (now public). Local
  remote, README badge, changelog links, citation and publishing script
  updated. A scan of the full history found no API key or personal e-mail.
* Run policy (CONTRIBUTING.md): DFT only locally (SIESTA by default, GPAW as
  alternative, GPAW on Actions only as a last resort); local runs are sized
  to the free resources; local fallback when Actions credits run out.
* `resources.py` measures free cores (1-min load and a measured busy
  fraction), available memory and GPU use, and sizes runs with a reserve of
  4 cores and 4 GB. First test expectation was wrong (it ignored the measured
  busy fraction); corrected.
* SIESTA 5.4.2 checked: MPI, DFT-D3 (s-dftd3), libxc, ELPA, NetCDF. PBE
  PseudoDojo PSML pseudopotentials available for Zn, C, H, N, O. GPAW is
  not installed.

## Stage 16 - `feature/16-siesta-setup`
Local DFT reference with SIESTA (project rule: DFT only locally). Tests: 109
passed (cumulative; 7 new). All runs sized with `resources.plan` (8 ranks,
18-26 free cores and 17-22 GB available at the time).
* **D3 was not what it claimed to be.** `DFTD3.UseXCDefaults true` left the
  generic parameters in place (E_D3 -1.768 eV vs -2.143 eV in QMOF on the
  test MOF). An independent implementation (torch-dftd) reproduced QMOF
  exactly without the three-body term (-2.1433 eV) and SIESTA's value with
  it (-2.0215 vs -2.0211 eV). Explicit PBE D3(BJ) parameters and a vanishing
  three-body cutoff give -2.1429 eV. The sweep and check made before the fix
  are kept as `*_d3-generic.jsonl` and were redone.
* **Convergence:** mesh (200-500 Ry) and k-grid (10-15 A) converged; TZP
  gains little; the generic TZ2P basis stops ("split norm too small").
  Smaller PAO energy shifts *increase* the disagreement with QMOF, which
  pointed to the pseudopotentials.
* **Pseudopotentials decide.** Fixed-cell relaxations from the QMOF
  structure: PseudoDojo for all elements gives C-H +1.8 %; SIESTA's
  ATOM-TABLE gives better organic bonds but Zn-O +2.4 % (its Zn has no
  semicore states). Final choice: ATOM-TABLE for H/C/N/O and PseudoDojo for
  Zn (Zn-O +0.38 %, C-O +0.66 %, C-C +0.67 %, C-H +1.23 %). `siesta.py` now
  takes per-element pseudopotential families (recorded with each run).
* **Result:** at the QMOF geometries SIESTA shows 0.2 eV/A residual forces
  and +0.5 to +4.4 GPa, almost all removed by relaxing the positions (cell
  unchanged). Consistency rule for the fit: SIESTA labels only on
  SIESTA-relaxed structures. Cost: 0.6-0.9 s per atom per single point on
  8 ranks.
* **Parser fix:** SCF convergence was first detected by the absence of
  "SCF convergence failure", which also appears in SIESTA's echo of the
  options; it now looks for "SCF Convergence by".
* GPAW (alternative code) is not installed; not needed so far.

## Stage 17 - `feature/17-first-fit`
First GA fits of the Zn + C/H/N/O force field. Tests: 116 passed
(cumulative; 7 new). All runs local, sized with `resources.plan` (12 LAMMPS
workers; teacher labels on the GPU; SIESTA molecules on 4 ranks).
* **Stress** added to the engine and checked against finite-difference
  strain derivatives on a triclinic MOF (six components within 0.1 %). At the
  QMOF geometries the initial force field gives stresses up to -160 GPa and
  forces up to 1300 kcal/mol/A on carboxylate O.
* **`ShardedEngine`** (spawned workers, configurations dealt round-robin)
  reproduces the serial engine to 1e-6; 12 workers.
* **Fit 01 failed on validation (45/228 vs 96).** Equilibrium forces alone
  let the GA shorten all C-O and Zn-N bonds (CO2 -7 %, CO -20 %, Zn-N -35 %).
* **Fit 02** added Zn-ligand bond scans (teacher), SIESTA molecules and
  bond-level over-coordination terms: 85/228, Zn-N still -22 %.
* **Diagnosis by energy-term decomposition:** finite differences of each
  ReaxFF energy component on the atom with the largest force showed the atom
  over/under-coordination term at 55 % (initial) and 30 % (fit 02) of the
  total; it is controlled by atom-level parameters of O/N (FC) and Zn (ZnOH).
* **Fit 03** (atom-level terms, bond scans in all 30 training MOFs, fresh
  start): **128/228 (56 %)**, Zn-N -4 %, carboxylate C-O +0.9 %; CO2 +5 % and
  CO +10 % remain, frameworks contract 3.8 %, 24 of 38 parameters moved by
  more than 15 % (bounds +/-30 % limit the search).
* **Found along the way:** ASE's `set_angle` cannot bend a linear molecule
  (CO2); `molecule_refs._bend` rotates about an explicit perpendicular axis.
  Molecule-check outputs of fits 01-02 were first written to misnamed
  folders (`zn_fit01_validation`) and moved to the right ones before commit.

## Stage 18 - `chore/18-ci-flake`
Pending items before the paper. Tests: 116 passed (no code change).
* **Repository visibility.** Public, confirmed by the project owner;
  recorded in CONTRIBUTING.md with the rule of never committing secrets.
* **Intermittent CI failure explained and fixed.** Three runs (PR #6, PR #10
  and a push to develop) died with exit code 15 at the first test that
  creates a LAMMPS instance; re-runs passed. A temporary probe workflow on
  this branch (history kept in its commits) created LAMMPS instances in
  fresh processes on many runners:
  - run 36333300706: 2 of 6 jobs died, message
    `ucx_init.c 38 init_worker Input/output error` (MPICH's UCX layer),
    with and without `mpiexec`;
  - run 36333433684: failures are per runner: one runner 200/200, the other
    four 0/200;
  - run 36333572250: both variants on the same 12 runners: default MPICH
    failed 20/20 on 2 runners and 0/20 on 10; `UCX_TLS=self,sm` 0/20 on all
    12, including the two bad runners.
  A re-run "fixed" the failure only because it landed on another machine.
  CI now sets `UCX_TLS=self,sm`; LAMMPS runs as a single process there, so
  only local transports are needed. The probe workflow was removed.

## Stage 19 - `chore/19-figure-scripts`
Every script used to generate a figure of the paper is now stored in the repository
(`figures/scripts/paper/`, documented in `figures/README.md`). No scientific code, run record or
result changed. Tests: 130 passed (cumulative; 14 new, no LaTeX needed).
* **Origin.** The paper figures had been produced by scripts kept only in the Google Drive paper
  folder. A quality-check round (2026-09-30, independent reviewer run) asked for them to be stored
  with the code. The manuscript and the figure files stay in Drive; the test
  `test_no_manuscript_or_figure_files_in_tree` guards this.
* **Reproduction verified.** From a clean checkout of `develop`, `paper_data.py` reproduced the 14
  data files of the manuscript byte for byte (the energy-term decomposition, which reruns LAMMPS,
  included) and `make_paper_figures.py` reproduced the 7 figures pixel for pixel.
* **Scripts.** `paper_data.py`, `make_paper_figures.py` and `paper_style.py` are the versions used
  for the submitted manuscript (only a docstring and the default `--repo` path changed).
  `paper_data_v2.py` and `make_paper_figures_v2.py` correct defects found on inspection: a clipped
  histogram range in fig 4a (10 MOFs below -30 % were dropped without warning), a legend over a
  curve in fig 2b, a label crossed by an arrow in fig 1, missing interquartile ranges in fig 5a; and
  add fig 6b (signed contribution of each energy term, because the terms cancel and shares of the
  summed absolute derivatives are not shares of the force) and fig S2 (pass fraction versus the
  volume window).
* **Findings that shape the paper text (not changed here).** The quality-check round also
  recomputed the validation from the scorecards: 96/128/45/85 passes, Wilson intervals and McNemar
  tests agree with `docs/results_first_fit.md`. Two additions: at a volume window of 10 % or more
  fit 03 no longer beats the initial force field (its failures are contractions, 97 of 100), and on
  60 fresh Zn MOFs that were neither in training nor in validation the pass count rose from 20 to
  42 (exact McNemar p = 1e-4).
