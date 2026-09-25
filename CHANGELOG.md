# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Each release is a tag on `main`;
the stage-by-stage audit trail is in [`docs/development_log.md`](docs/development_log.md).

## [Unreleased]
### Added
- LAMMPS engine: triclinic cells (rotation to LAMMPS's restricted form and
  back) and cell relaxation (`relax(cell=True)`).
- `ffield.add_placeholder_pairs`: starting entries for element pairs that
  have none (a missing bond entry gives NaN forces in LAMMPS).
- `validate.py`: per-MOF scorecard against the DFT structure (volume, cell,
  displacements, metal coordination, bond lengths per element pair).
- `scripts/baseline_validation.py`, `docs/results_baseline_validation.md`:
  the initial Zn force field on 228 Zn MOFs (42 % pass).
- Relaxed structures of the Zn + C/H/N/O family (2 958 MOFs) with per-atom
  DDEC6/CM5 charges and bond-order sums (`qmof.fetch_structures`,
  `data/qmof/structures_Zn-CHNO.extxyz.gz`).
- `teacher.py` and `scripts/teacher_check.py`: four MACE foundation models
  checked against QMOF DFT (force error at the DFT minima, energy
  consistency, cell relaxation); `docs/results_teacher_check.md`.
- Optional dependency group `mlip` (mace-torch, torch-dftd).
### Added
- `ffield.merge_elements` and `ffield.missing_interactions`: add element
  blocks from a donor force field to a base one, with an audit report of
  copied terms, differing general/shared-element parameters and missing
  interactions.
- `data/ffields/sources/`: four ReaxFF files from the LAMMPS distribution
  (ZnOH, FC, budzien, mattsson) with hashes and references.
- `scripts/build_base_ffields.py` and `docs/results_base_ffield.md`: three
  candidate Zn + C/H/N/O bases, screened on gas molecules (H2O, CO2, CH4,
  H2, ...) and a Zn aqua-hydroxo cluster; FC + Zn selected.

### Changed
- `engine.single_point` raises on non-finite energies or forces.
- `engine`: masses for any element (ASE table), dummy atom types get mass 1.
- `ffield.write` refuses a literal "X" label outside torsions (ambiguous with
  the torsion wildcard).

## [0.2.0] - 2026-09-25
First step toward ReaxFF force fields for the MOFs of the QMOF database;
GitHub Pro CI and branch protection.

### Added
- `qmof.py`: download of the Materials Project MOF Explorer / QMOF metadata
  (MPContribs `mofexplorer`) into a deterministic, hash-verified snapshot;
  snapshot of 20 375 MOFs committed in `data/qmof/`.
- `families.py`: MOF chemical families (metal set + non-metal set), force-field
  coverage and greedy maximum-coverage selection.
- `scripts/survey_qmof.py` and `docs/results_qmof_survey.md`: database survey,
  coverage of the database vs. number of force fields, pilot-family choice
  (Zn + C/H/N/O).
- Optional dependency group `mof` (pymatgen, mpcontribs-client, matplotlib).

### Changed
- CI runs automatically again on every push and pull request (GitHub Pro);
  actions updated to Node 24 versions (checkout v5, setup-python v6,
  upload-artifact v7).
- `main` and `develop` protected: pull request with a passing `pytest`
  check required.

## [0.1.2] - 2026-09-25
Repository infrastructure; no change to the scientific code or results.

### Added
- `CONTRIBUTING.md`: branching model, commit conventions, per-stage test and
  audit protocol, release procedure.
- `CHANGELOG.md`, `CITATION.cff`, `.github/CODEOWNERS`, pull-request and
  issue templates.
- `tests/test_metadata.py`: version consistency between the package,
  `pyproject.toml`, this changelog and `CITATION.cff` (59 tests in total).
- CI workflow keeps the `pytest -v` log and `pip freeze` as artifacts.

### Changed
- CI runs on manual trigger only (GitHub free plan, used for storage); the
  committed `docs/test_logs/` remain the test evidence.
- `develop` back-merged from `main`; it was missing the v0.1.0/v0.1.1
  release commits.

### Fixed
- Package version was still `0.1.0` in `pyproject.toml` and `__init__.py`
  after the `v0.1.1` tag.

## [0.1.1] - 2026-09-24
### Added
- `scripts/make_figures.py` and four result figures (`docs/figures/`).
- Finding: the reference force field has a non-analytic cusp at the graphene
  lattice constant; the `strain` family must be rebuilt with internal relaxation.

### Changed
- Regenerable `.extxyz` datasets are no longer versioned; their SHA-256 hashes
  are kept in the run manifest.

## [0.1.0] - 2026-09-24
### Added
- ReaxFF ffield reader/writer with named parameter keys (stage 1).
- Unit-cube parameter space with explicit bounds (stage 2).
- Graphene-oxide model and training-set generators (stage 3).
- Persistent-instance LAMMPS ReaxFF evaluator (stage 4).
- Relative-energy and force objective with penalty and cache (stage 5).
- Real-coded GA with bounded Nelder-Mead refinement (stage 6).
- End-to-end pipeline, audit records, CLI and GO recovery benchmark (stage 7).

[Unreleased]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ribeirojr-gh/ga-reaxff/releases/tag/v0.1.0
