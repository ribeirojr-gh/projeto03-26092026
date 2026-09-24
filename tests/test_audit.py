"""Stage 7 tests: provenance records."""
import json

import numpy as np

from ga_reaxff.audit import RunRecorder, read_history, software_versions


def test_manifest_and_history(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("hello")
    rec = RunRecorder(tmp_path / "run")
    m = rec.manifest({"a": 1}, {"input": f})
    assert m["inputs"]["input"]["sha256"] == \
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"   # sha256("hello")
    assert {"python", "numpy", "lammps", "ga_reaxff"} <= set(m["software"])
    assert "git_commit" in m
    rec.log_generation({"generation": 0, "best": np.float64(1.5), "g": np.arange(3)})
    rec.log_generation({"generation": 1, "best": float("inf")})
    h = read_history(tmp_path / "run" / "history.jsonl")
    assert [r["generation"] for r in h] == [0, 1]
    assert h[0]["g"] == [0, 1, 2] and h[1]["best"] == "inf"      # JSON-safe conversion
    json.loads((tmp_path / "run" / "manifest.json").read_text())  # valid JSON


def test_software_versions_has_lammps():
    assert software_versions()["lammps"] != ""
