"""Provenance and run records.

Every run directory contains:
    manifest.json   inputs: config, seed, software versions, git commit,
                    SHA-256 of the starting force field and training set
    history.jsonl   one JSON line per GA generation (loss statistics and the
                    physical values of the best parameter set)
    result.json     per-parameter table (start, final, bounds[, truth]),
                    loss breakdown, validation metrics, timings
    ffield.best     the optimized force field, ready for LAMMPS

With these files, anyone can check *what* was run, *on what data*, and
*reproduce it* (same config + same seed + same versions -> same history).
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__
from .dataset import file_sha256


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def software_versions() -> dict:
    import ase
    import scipy
    v = {"ga_reaxff": __version__, "python": platform.python_version(),
         "numpy": np.__version__, "scipy": scipy.__version__, "ase": ase.__version__,
         "platform": platform.platform()}
    try:
        import lammps
        L = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        v["lammps"] = str(L.version())
        L.close()
    except Exception:  # pragma: no cover
        v["lammps"] = "unavailable"
    return v


def git_commit(repo_dir: str | Path = ".") -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class RunRecorder:
    def __init__(self, outdir: str | Path):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hist = self.dir / "history.jsonl"
        self._hist.write_text("")
        self.t0 = time.time()

    def manifest(self, config: dict, files: dict[str, str | Path], extra: dict | None = None):
        m = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "git_commit": git_commit(Path(__file__).parent),
             "software": software_versions(),
             "config": config,
             "inputs": {name: {"path": str(p), "sha256": file_sha256(p)} for name, p in files.items()}}
        if extra:
            m.update(extra)
        self.write_json("manifest.json", m)
        return m

    def log_generation(self, record: dict):
        record = dict(record, elapsed_s=round(time.time() - self.t0, 3))
        with self._hist.open("a") as fh:
            fh.write(json.dumps(_jsonable(record)) + "\n")

    def write_json(self, name: str, obj) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")
        return p

    def elapsed(self) -> float:
        return time.time() - self.t0


def read_history(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
