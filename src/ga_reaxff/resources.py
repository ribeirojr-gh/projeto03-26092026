"""Check the local machine before starting a calculation.

The workstation is shared with other projects. Before a local run (DFT,
large LAMMPS batches, GPU jobs) the available cores and memory are measured
and the run is sized to what is free, never to the whole machine:

    >>> from ga_reaxff.resources import snapshot, plan
    >>> p = plan(mem_per_process_gb=2.0, max_processes=16)
    >>> p.processes, p.reason

`snapshot` reads /proc (Linux) and, when present, nvidia-smi; `plan` keeps a
safety margin of cores and memory for the other users. Every run records the
snapshot and the plan in its manifest.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass
class Snapshot:
    time_utc: str
    cores_logical: int
    load_1min: float
    load_5min: float
    cpu_busy_fraction: float          # measured over `interval` seconds
    mem_total_gb: float
    mem_available_gb: float
    gpu_name: str | None = None
    gpu_mem_total_gb: float | None = None
    gpu_mem_used_gb: float | None = None
    gpu_util_percent: float | None = None

    def as_dict(self):
        return asdict(self)


def _cpu_times():
    with open("/proc/stat") as fh:
        vals = [int(x) for x in fh.readline().split()[1:]]
    idle = vals[3] + vals[4]            # idle + iowait
    return idle, sum(vals)


def _meminfo() -> dict[str, float]:
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":")
            out[k] = int(v.split()[0]) / 1024 ** 2   # kB -> GB
    return out


def _gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10, check=True).stdout.strip().splitlines()[0]
        name, tot, used, util = [x.strip() for x in q.split(",")]
        return name, float(tot) / 1024, float(used) / 1024, float(util)
    except Exception:
        return None


def snapshot(interval: float = 1.0) -> Snapshot:
    """Current usage of the machine (CPU busy fraction measured over `interval`)."""
    i0, t0 = _cpu_times()
    time.sleep(interval)
    i1, t1 = _cpu_times()
    busy = 1.0 - (i1 - i0) / max(t1 - t0, 1)
    la1, la5, _ = os.getloadavg()
    m = _meminfo()
    s = Snapshot(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 cores_logical=os.cpu_count() or 1, load_1min=la1, load_5min=la5,
                 cpu_busy_fraction=round(busy, 3), mem_total_gb=round(m["MemTotal"], 2),
                 mem_available_gb=round(m["MemAvailable"], 2))
    g = _gpu()
    if g:
        s.gpu_name, s.gpu_mem_total_gb, s.gpu_mem_used_gb, s.gpu_util_percent = g
    return s


@dataclass
class Plan:
    processes: int
    free_cores: int
    usable_mem_gb: float
    reason: str
    snapshot: dict

    def as_dict(self):
        return asdict(self)


def plan(mem_per_process_gb: float, max_processes: int | None = None,
         reserve_cores: int = 4, reserve_mem_gb: float = 4.0,
         snap: Snapshot | None = None) -> Plan:
    """How many processes can run now without starving the other users.

    free cores = logical cores - busy cores (max of 1-min load and measured
    busy fraction) - `reserve_cores`; usable memory = available - reserve.
    Returns 0 processes (with the reason) when the machine is too busy: the
    caller must then wait, or fall back as the project rules say.
    """
    s = snap or snapshot()
    busy = max(s.load_1min, s.cpu_busy_fraction * s.cores_logical)
    free_cores = max(0, int(s.cores_logical - busy + 1e-9) - reserve_cores)
    usable = max(0.0, s.mem_available_gb - reserve_mem_gb)
    by_mem = int(usable // mem_per_process_gb) if mem_per_process_gb > 0 else free_cores
    n = min(free_cores, by_mem, max_processes if max_processes else free_cores)
    limit = ("cores" if n == free_cores else "memory" if n == by_mem else "max_processes")
    reason = (f"{n} processes: {free_cores} free cores (reserve {reserve_cores}), "
              f"{usable:.1f} GB usable (reserve {reserve_mem_gb} GB, "
              f"{mem_per_process_gb} GB each); limited by {limit}")
    return Plan(processes=max(n, 0), free_cores=free_cores, usable_mem_gb=round(usable, 2),
                reason=reason, snapshot=s.as_dict())


def main():  # pragma: no cover - CLI
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Show free resources and a process plan.")
    ap.add_argument("--mem-per-process", type=float, default=2.0)
    ap.add_argument("--max-processes", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(plan(a.mem_per_process, a.max_processes).as_dict(), indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
