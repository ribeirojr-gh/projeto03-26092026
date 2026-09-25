"""Stage 9 tests: release metadata is consistent across the repository."""
import re
import tomllib
from pathlib import Path

import ga_reaxff

ROOT = Path(__file__).parents[1]


def _pyproject_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_package_version_matches_pyproject():
    """`ga_reaxff.__version__` and pyproject.toml must name the same release."""
    assert ga_reaxff.__version__ == _pyproject_version()


def test_changelog_documents_current_version():
    """Every released version must have a CHANGELOG entry."""
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.search(rf"^## \[{re.escape(_pyproject_version())}\]", changelog, re.M)


def test_citation_version_matches_pyproject():
    """CITATION.cff must cite the current release."""
    cff = (ROOT / "CITATION.cff").read_text()
    assert f'version: "{_pyproject_version()}"' in cff
