"""QMOF database ingestion (Materials Project MOF Explorer).

The MOF Explorer app of the Materials Project is backed by the MPContribs
project ``mofexplorer`` (A. S. Rosen, K. M. Jablonka), built on the QMOF
database: ~20k MOFs relaxed with periodic DFT (VASP, PBE-D3(BJ)), with
energies, DDEC6/CM5 charges, MOFid decomposition (nodes, linkers, topology)
and textural properties. Reference: A. S. Rosen et al., Matter 4, 1578 (2021).

This module downloads the per-MOF metadata table (no structures), flattens it
to one dict per MOF and stores it as a deterministic snapshot:

    <dir>/qmof_snapshot.jsonl.gz    one JSON line per MOF, sorted by identifier
    <dir>/qmof_snapshot.sha256      SHA-256 of the *uncompressed* JSON lines
    <dir>/provenance.json           source, query, client versions, date, counts

The hash is taken over the uncompressed text so that it does not depend on
gzip metadata; the same database content always gives the same hash.
The API key is read from the pymatgen settings (``PMG_MAPI_KEY`` in
``~/.config/.pmgrc.yaml``) or the ``MP_API_KEY`` environment variable and is
never written to disk by this module.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

PROJECT = "mofexplorer"
REFERENCES = ["https://doi.org/10.1016/j.matt.2021.02.015"]

# Sub-trees of `data` that are dropped from the snapshot: adsorption (GCMC)
# results and hybrid-functional outputs are not needed to build force fields.
EXCLUDED_PREFIXES = ("outputsGCMC.", "outputsDFT.HSE06", "outputsDFT.HLE17")

SNAPSHOT = "qmof_snapshot.jsonl.gz"
SNAPSHOT_HASH = "qmof_snapshot.sha256"
PROVENANCE = "provenance.json"


def api_key() -> str:
    """API key from the environment or the pymatgen settings file."""
    key = os.environ.get("MP_API_KEY")
    if not key:
        from pymatgen.core import SETTINGS
        key = SETTINGS.get("PMG_MAPI_KEY")
    if not key:
        raise RuntimeError("no Materials Project API key: set MP_API_KEY or "
                           "PMG_MAPI_KEY in ~/.config/.pmgrc.yaml")
    return key


def flatten(data: dict, prefix: str = "") -> dict:
    """Flatten an MPContribs `data` tree to dotted keys.

    Quantities stored as {"display", "value", "unit", ...} are reduced to their
    numeric value (units are fixed per column and recorded in the provenance).
    """
    out = {}
    for k, v in data.items():
        key = prefix + k
        if isinstance(v, dict) and "value" in v and "display" in v:
            out[key] = v["value"]
        elif isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def to_row(contribution: dict) -> dict:
    """One flat record per MOF, without the excluded sub-trees."""
    row = {"identifier": contribution["identifier"],
           "contribution_id": contribution["id"],
           "formula": contribution.get("formula")}
    for k, v in flatten(contribution.get("data", {})).items():
        if not k.startswith(EXCLUDED_PREFIXES):
            row[k] = v
    return row


def _jsonl(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r["identifier"])
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)


def fetch(timeout: int = 3600) -> tuple[list[dict], dict]:  # pragma: no cover - network
    """Download all MOF Explorer records; returns (rows, provenance)."""
    import mpcontribs.client
    from mpcontribs.client import Client

    client = Client(apikey=api_key(), project=PROJECT)
    info = client.get_project()
    t0 = time.time()
    result = client.query_contributions(fields=["identifier", "id", "formula", "data"],
                                        paginate=True, timeout=timeout)
    rows = [to_row(c) for c in result["data"]]
    units = {c["path"].removeprefix("data."): c.get("unit")
             for c in info.get("columns", []) if c.get("path", "").startswith("data.")}
    provenance = {
        "source": f"MPContribs project '{PROJECT}' (Materials Project MOF Explorer)",
        "project_title": info.get("title"),
        "project_authors": info.get("authors"),
        "references": [r.get("url") for r in info.get("references", [])] or REFERENCES,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "download_time_s": round(time.time() - t0, 1),
        "mpcontribs_client": mpcontribs.client.__version__
        if hasattr(mpcontribs.client, "__version__") else "unknown",
        "total_count_reported": result.get("total_count"),
        "n_rows": len(rows),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "units": {k: u for k, u in units.items() if not k.startswith(EXCLUDED_PREFIXES)},
    }
    return rows, provenance


def save_snapshot(rows: list[dict], provenance: dict, directory: str | Path) -> str:
    """Write the snapshot, its hash and the provenance record; return the hash."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    text = _jsonl(rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    # mtime=0 and no file name in the gzip header: byte-identical output
    with open(d / SNAPSHOT, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh,
                                                       mtime=0) as gz:
        gz.write(text.encode())
    (d / SNAPSHOT_HASH).write_text(f"{digest}  {SNAPSHOT} (uncompressed JSON lines)\n")
    (d / PROVENANCE).write_text(json.dumps({**provenance, "sha256_uncompressed": digest},
                                           indent=2, sort_keys=True) + "\n")
    return digest


def load_snapshot(directory: str | Path, verify: bool = True) -> list[dict]:
    """Read the snapshot; with `verify`, fail if it does not match its hash."""
    d = Path(directory)
    text = gzip.decompress((d / SNAPSHOT).read_bytes()).decode()
    if verify:
        expected = (d / SNAPSHOT_HASH).read_text().split()[0]
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != expected:
            raise ValueError(f"QMOF snapshot hash mismatch: {actual} != {expected}")
    return [json.loads(line) for line in text.splitlines()]
