# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Each release is a tag on `main`;
the stage-by-stage audit trail is in [`docs/development_log.md`](docs/development_log.md).

## [0.1.2] - 2026-09-25
Repository infrastructure; no change to the scientific code or results.

### Added
- `CONTRIBUTING.md`: branching model, commit conventions, per-stage test and
  audit protocol, release procedure.
- `CHANGELOG.md`, `CITATION.cff`, `.github/CODEOWNERS`, pull-request and
  issue templates.
- `tests/test_metadata.py`: version consistency between the package,
  `pyproject.toml`, this changelog and `CITATION.cff` (59 tests in total).
- CI uploads the full `pytest -v` log of every run as a build artifact.

### Changed
- CI runs on every branch of the branching model (`feature/**`, `fix/**`,
  `chore/**`, `release/**`) and on pull requests.
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

[0.1.2]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ribeirojr-gh/ga-reaxff/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ribeirojr-gh/ga-reaxff/releases/tag/v0.1.0
