# Contributing

This project is developed so that every result can be traced back to the
code, data and tests that produced it. The rules below keep that property.

## Branching model

| Branch | Purpose | Merges into |
|---|---|---|
| `main` | Released versions only. Every commit on `main` is a release merge and carries a tag `vX.Y.Z`. | — |
| `develop` | Integration branch. Always passes the full test suite. | `main` (at release) |
| `feature/NN-short-name` | One development stage. `NN` is the stage number, continuing the sequence in `docs/development_log.md`. | `develop` |
| `fix/short-name` | Bug fix to code already on `develop`. | `develop` |
| `chore/NN-short-name` | Repository infrastructure (CI, docs, packaging) without scientific changes. | `develop` |

Stage branches are kept after merging; they are part of the audit trail.

## Per-stage protocol

1. Branch from an up-to-date `develop`.
2. Commit the code and its tests (one test file per stage in `tests/`).
3. Run the suite and commit the full log:
   ```bash
   pytest -v 2>&1 | tee docs/test_logs/NN-short-name.log
   ```
4. Add a section to `docs/development_log.md`: what was built, the test count,
   and every failure found and how it was resolved, including physics findings.
5. Open a pull request into `develop` (the template lists the checklist) and
   merge it with a merge commit (`--no-ff`), never squash or rebase, so the
   stage stays visible in `git log --graph`. CI must pass; the committed test
   log remains the permanent record (CI artifacts expire).

## Where things run

- **GitHub** (Pro plan): storage and structure (branches, pull requests,
  tags, releases), CI on every push and pull request, and light batch jobs
  on Actions runners (Linux, 2 vCPU for private repositories, no GPU,
  6 h per job, monthly minutes quota). Suited to many small independent
  runs, e.g. relaxations split with a job matrix.
- **Cloud sandbox** (Claude): development, tests and light runs.
- **Local workstation** (32 threads, RTX 4070 8 GB): heavy simulations
  (LAMMPS with MPI/KOKKOS-CUDA) and GPU inference.

`main` and `develop` are protected: changes arrive only through pull
requests with a passing `pytest` check.

Every run's `manifest.json` records software versions and input hashes, so
results from different machines stay comparable.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) with the module
as scope, e.g. `feat(engine): ...`, `fix(dataset): ...`, `test: ...`,
`docs: ...`, `results: ...`, `chore: ...`, `merge: ...`, `release: ...`.
The body explains *why*, not only *what*.

## Releases

1. Bump the version in `pyproject.toml`, `src/ga_reaxff/__init__.py` and
   `CITATION.cff`; add the `CHANGELOG.md` entry (`tests/test_metadata.py`
   checks that these agree).
2. Open a pull request `develop` → `main`; merge with a merge commit.
3. Tag the merge commit with an annotated tag and publish a GitHub release
   whose notes are the changelog entry:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z: summary"
   git push origin vX.Y.Z
   ```
4. Back-merge `main` into `develop`, so that `develop` always contains `main`.

## Provenance rules

- Reference data in `data/` ship with a SHA-256 file that the tests check.
- Every run writes `manifest.json` (software versions, git commit, input
  hashes, configuration, seed). Results quoted in `docs/` must come from a
  committed run directory.
- Large regenerable files (e.g. `.extxyz` datasets) are not versioned; their
  hashes are recorded in the manifest and `scripts/make_figures.py` verifies them.
- History on `main` and `develop` is never rewritten.

## Environment

```bash
sudo apt-get install -y libmpich12   # MPI runtime needed by the LAMMPS wheel
pip install -e ".[dev,lammps]"
pytest -v
```

Without root access, the MPI runtime can come from PyPI instead:
`pip install mpich` and `export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib`.

To use a locally compiled LAMMPS (shared library, `PKG_PYTHON` and
`PKG_REAXFF` on) instead of the PyPI wheel, install without the `lammps`
extra and point Python at the build:

```bash
pip install -e ".[dev]"
ln -sfn "$LAMMPS_DIR/build/liblammps.so" "$LAMMPS_DIR/python/lammps/liblammps.so"
echo "$LAMMPS_DIR/python" > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/lammps-local-build.pth"
```

The LAMMPS version actually used is recorded in each run's `manifest.json`.

Code, comments and documentation are written in English.
