"""Command-line interface.

    ga-reaxff run configs/go_recovery.toml --out runs/go_recovery
    ga-reaxff report runs/go_recovery
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _report(run_dir: Path) -> str:
    r = json.loads((run_dir / "result.json").read_text())
    lines = [f"run: {run_dir}", f"mode: {r['mode']}", ""]
    hdr = f"{'parameter':32s} {'start':>11s} {'final':>11s}"
    synth = "truth" in r["parameters"][0]
    if synth:
        hdr += f" {'truth':>11s} {'err0 %':>7s} {'err %':>7s}"
    lines.append(hdr)
    for p in r["parameters"]:
        s = f"{p['key']:32s} {p['start']:11.4f} {p['final']:11.4f}"
        if synth:
            s += f" {p['truth']:11.4f} {100 * p['start_rel_error']:7.2f} {100 * p['final_rel_error']:7.2f}"
        lines.append(s)
    lines.append("")
    for split in ("train", "validation"):
        a, b = r[split]["start"], r[split]["final"]
        lines.append(f"{split:10s} loss {a['loss']:.4g} -> {b['loss']:.4g} | "
                     f"E-RMSE {a['energy_rmse']:.3f} -> {b['energy_rmse']:.3f} kcal/mol | "
                     f"F-RMSE {a['force_rmse']:.3f} -> {b['force_rmse']:.3f} kcal/mol/A")
    lines.append(f"objective calls {r['objective_calls']}, failed {r['failed_evaluations']}, "
                 f"total time {r['total_time_s']:.0f} s")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ga-reaxff", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run an optimization from a TOML config")
    p_run.add_argument("config")
    p_run.add_argument("--out", required=True)
    p_rep = sub.add_parser("report", help="print a summary of a finished run")
    p_rep.add_argument("run_dir")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from .pipeline import load_config, run
        run(load_config(args.config), args.out)
        print(_report(Path(args.out)))
    elif args.cmd == "report":
        print(_report(Path(args.run_dir)))


if __name__ == "__main__":
    main()
