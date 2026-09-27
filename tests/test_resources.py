"""Stage 15 tests: machine resource snapshot and run sizing."""
from ga_reaxff.resources import Snapshot, plan, snapshot


def _snap(cores=32, load=0.5, busy=0.02, avail=20.0):
    return Snapshot(time_utc="t", cores_logical=cores, load_1min=load, load_5min=load,
                    cpu_busy_fraction=busy, mem_total_gb=31.0, mem_available_gb=avail)


def test_snapshot_reads_the_machine():
    s = snapshot(interval=0.1)
    assert s.cores_logical >= 1 and 0.0 <= s.cpu_busy_fraction <= 1.0
    assert 0 < s.mem_available_gb <= s.mem_total_gb


def test_plan_idle_machine_limited_by_cores_or_max():
    p = plan(2.0, snap=_snap())
    # busy = max(load 0.5, 0.02 * 32 = 0.64) -> int(31.36) = 31 cores, minus 4 reserved
    assert p.free_cores == 27 and p.processes == 8   # 16 GB usable / 2 GB each
    assert "memory" in p.reason
    assert plan(0.5, max_processes=6, snap=_snap()).processes == 6


def test_plan_busy_machine():
    p = plan(1.0, snap=_snap(load=27.5))
    assert p.free_cores == 0 and p.processes == 0 and "cores" in p.reason
    # measured busy fraction counts even if the load average is still low
    assert plan(1.0, snap=_snap(load=1.0, busy=0.75)).free_cores == 32 - 24 - 4


def test_plan_low_memory():
    p = plan(4.0, snap=_snap(avail=6.0))
    assert p.usable_mem_gb == 2.0 and p.processes == 0 and "memory" in p.reason
