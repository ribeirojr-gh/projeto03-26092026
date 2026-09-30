"""Stage 19: the figure scripts of the paper are stored in the repository.

These tests need no LaTeX and no pandas: they check that the scripts compile, that the README
documents each of them, that the default --repo path points to the repository root, and that
no manuscript or figure file was added to the tree.
"""
import ast
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "figures" / "scripts" / "paper"
SCRIPTS = sorted(PAPER.glob("*.py"))
README = (ROOT / "figures" / "README.md").read_text()


def test_expected_scripts_present():
    names = {p.name for p in SCRIPTS}
    assert {"paper_style.py", "paper_data.py", "make_paper_figures.py",
            "paper_data_v2.py", "make_paper_figures_v2.py"} <= names


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_readme_documents_script(path):
    assert path.name in README


@pytest.mark.parametrize("name", ["paper_data.py", "paper_data_v2.py"])
def test_default_repo_is_repository_root(name):
    text = (PAPER / name).read_text()
    ast.parse(text)
    assert "parents[3]" in text, "default --repo must be derived from the script location"
    # parents[3] of <root>/figures/scripts/paper/<script>.py is the repository root
    assert (PAPER / name).resolve().parents[3] == ROOT and (ROOT / "pyproject.toml").exists()


def test_no_manuscript_or_figure_files_in_tree():
    forbidden = {".tex", ".bib", ".docx", ".pdf", ".png", ".svg", ".jpg", ".jpeg"}
    found = [p for p in (ROOT / "figures").rglob("*")
             if p.is_file() and p.suffix.lower() in forbidden and "build" not in p.parts]
    assert not found, f"paper files and figure images stay in Google Drive only: {found}"
