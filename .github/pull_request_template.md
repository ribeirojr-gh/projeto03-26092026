## Summary
<!-- What this branch does and why. For a stage: which pipeline module. -->

## Stage / type
- Branch: `feature/NN-...` | `fix/...` | `chore/NN-...` | `develop` → `main` (release)

## Checklist
- [ ] Tests added or updated for the change
- [ ] `pytest -v` passes locally; log committed in `docs/test_logs/`
- [ ] Section added to `docs/development_log.md` (failures found and how they were resolved)
- [ ] Results quoted in `docs/` come from a committed run directory with its `manifest.json`
- [ ] `CHANGELOG.md` updated (releases: version bumped in `pyproject.toml`, `__init__.py`, `CITATION.cff`)

## Test result
<!-- e.g. 59 passed in 15 s -->
