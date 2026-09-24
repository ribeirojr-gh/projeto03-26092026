"""Stage 7 tests: miniature end-to-end run of the GO recovery pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from ga_reaxff.audit import read_history
from ga_reaxff.cli import main as cli_main
from ga_reaxff.ffield import ForceField
from ga_reaxff.pipeline import load_config, run

ROOT = Path(__file__).parents[1]


def tiny_config():
    cfg = load_config(ROOT / "configs" / "go_recovery.toml")
    cfg["parameters"]["keys"] = ["bond:C-O:De_sigma", "angle:C-O-C:theta_00", "offdiag:C-O:r_vdw"]
    cfg["ga"].update(pop_size=8, n_generations=3, patience=None)
    cfg["local"] = {"enabled": True, "maxiter": 20}
    t = cfg["training_set"]
    t["train"].update(strains=[-0.01, 0.0, 0.01], epoxide_z=[-0.1, 0.0, 0.2],
                      hydroxyl_stretch=[0.0, 0.2], hydroxyl_bend=[0.0, 0.2], n_rattle=2)
    t["validation"].update(n_rattle=1)
    t["relax_ftol"] = 5e-2
    return cfg


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return out, run(tiny_config(), out, log=lambda *_: None)


@pytest.mark.slow
def test_outputs_exist_and_are_consistent(tiny_run):
    out, res = tiny_run
    for f in ("manifest.json", "history.jsonl", "result.json", "ffield.best", "ffield.start",
              "training_set.extxyz", "validation_set.extxyz"):
        assert (out / f).exists(), f
    m = json.loads((out / "manifest.json").read_text())
    assert m["truth_in_bounds"] is True
    assert len(read_history(out / "history.jsonl")) == 4          # generations 0..3
    # ffield.best really contains the reported final values
    ff = ForceField.read(out / "ffield.best")
    for row in res["parameters"]:
        assert ff.get(row["key"]) == pytest.approx(row["final"], rel=1e-6)


@pytest.mark.slow
def test_never_worse_than_start(tiny_run):
    _, res = tiny_run
    assert res["train"]["final"]["loss"] <= res["train"]["start"]["loss"]
    assert res["train"]["final"]["loss"] < 0.5 * res["train"]["start"]["loss"]


@pytest.mark.slow
def test_cli_report(tiny_run, capsys):
    out, _ = tiny_run
    cli_main(["report", str(out)])
    text = capsys.readouterr().out
    assert "bond:C-O:De_sigma" in text and "validation" in text
