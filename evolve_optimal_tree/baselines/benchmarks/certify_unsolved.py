"""Run the reference GOSDT with no time cap on every pair that no exact solver
has certified yet, until it certifies or exceeds the machine's memory.

Uses the certificate-fixed reference binary (``baselines/gosdt/build/gosdt_patched``,
built from ``baselines/gosdt_patches/scope-lowerbound.patch``), because the
unpatched binary issues false certificates.  One pair at a time, single thread.
A certificate is accepted only if its objective is not worse than the best
known incumbent by more than 1e-6 (a certificate above a known tree would be a
false certificate); accepted certificates update ``src/known_optima.csv``.

Outputs ``results/certify_unsolved.csv`` (one row per attempt) and, on
acceptance, the refreshed known optima.

Usage (from ``evolve_optimal_tree``)::

    uv run baselines/benchmarks/certify_unsolved.py [--pairs dataset:lam,...] [--binary PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import evaluate_tree  # noqa: E402
from suite import DATA, load_dataset, load_known_optima  # noqa: E402
import suite  # noqa: E402

BINARY = ROOT / "baselines" / "gosdt" / "build" / "gosdt_patched"
OUT = HERE / "results" / "certify_unsolved.csv"
KNOWN = ROOT / "src" / "known_optima.csv"
COLS = ["dataset", "lam", "objective", "errors", "leaves", "seconds", "wall", "status", "exit_code",
        "peak_rss_gb", "graph_size", "iterations", "known_before", "accepted"]
TOL = 1e-6


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) * 1024
    except (ValueError, OSError):
        return 0


def run_pair(binary: Path, name: str, lam: float, frame: pd.DataFrame, memory_cap: int = 0) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "data.csv"
        frame.to_csv(csv_path, index=False)
        model_path = Path(tmp) / "model.json"
        cfg = Path(tmp) / "cfg.json"
        cfg.write_text(json.dumps({"regularization": lam, "verbose": True, "worker_limit": 1,
                                   "model_limit": 1, "time_limit": 0, "model": str(model_path)}))
        with open(csv_path, "rb") as fh, open(Path(tmp) / "out.txt", "w+") as out_fh:
            t0 = time.perf_counter()
            proc = subprocess.Popen([str(binary), str(cfg)], stdin=fh, stdout=out_fh,
                                    stderr=subprocess.STDOUT, text=True)
            peak = 0
            killed = False
            while proc.poll() is None:
                time.sleep(1.0)
                peak = max(peak, _rss_bytes(proc.pid))
                if memory_cap and peak > memory_cap:
                    proc.kill()
                    killed = True
            wall = time.perf_counter() - t0
            out_fh.seek(0)
            out = out_fh.read()
        m = re.search(r"Training Duration: ([0-9.eE+-]+) seconds", out)
        seconds = float(m.group(1)) if m else float("nan")
        m = re.search(r"Optimality Gap: ([0-9.eE+-]+)", out)
        gap = float(m.group(1)) if m else float("nan")
        m = re.search(r"Size of Graph: (\d+)", out)
        graph = int(m.group(1)) if m else ""
        m = re.search(r"Number of Iterations: (\d+)", out)
        iters = int(m.group(1)) if m else ""
        row = {"dataset": name, "lam": lam, "seconds": seconds, "wall": round(wall, 1),
               "exit_code": proc.returncode, "peak_rss_gb": round(peak / (1 << 30), 2),
               "graph_size": graph, "iterations": iters, "objective": "", "errors": "", "leaves": ""}
        if killed:
            row["status"] = "memory_cap"
        elif proc.returncode < 0 or proc.returncode == 137:
            row["status"] = "oom_killed"
        elif not model_path.exists():
            row["status"] = "no_model"
        else:
            models = json.loads(model_path.read_text())
            e, l = evaluate_tree(models[0], frame)
            row.update(objective=e / len(frame) + lam * l, errors=e, leaves=l,
                       status="optimal" if gap == 0.0 else "time")
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="", help="dataset:lam,... (default: every uncertified pair)")
    ap.add_argument("--binary", default=str(BINARY))
    ap.add_argument("--memory-cap-gb", type=float, default=12.0,
                    help="kill a run whose resident memory exceeds this (0 = none; the reference exhausts a 16 GB "
                         "machine within minutes on these pairs, which kills the driver too)")
    args = ap.parse_args()
    memory_cap = int(args.memory_cap_gb * (1 << 30))
    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(f"binary not found: {binary} (apply baselines/gosdt_patches/scope-lowerbound.patch and build)")
    suite.DATASETS.append(("sine_10k", DATA / "sine" / "ten_thousand.csv"))
    known = load_known_optima()
    if args.pairs:
        pairs = [(p.split(":")[0], float(p.split(":")[1])) for p in args.pairs.split(",")]
    else:
        pairs = [(d, lam) for (d, lam), (_, cert) in known.items() if not cert]
        # easier pairs first: larger λ within a dataset, smaller datasets first
        sizes = {d: 0 for d, _ in pairs}
        for d in sizes:
            sizes[d] = len(load_dataset(d))
        pairs.sort(key=lambda p: (sizes[p[0]], -p[1]))
    os.makedirs(OUT.parent, exist_ok=True)
    for name, lam in pairs:
        frame = load_dataset(name)
        k_obj, k_cert = known.get((name, lam), (float("nan"), False))
        print(f"\n=== {name} λ={lam:g}  (best known {k_obj:.6f}{'*' if k_cert else ''}) ===", flush=True)
        row = run_pair(binary, name, lam, frame, memory_cap)
        row["known_before"] = k_obj
        accepted = False
        if row["status"] == "optimal":
            if row["objective"] > k_obj + TOL:
                row["status"] = "false_certificate"
            else:
                accepted = True
        row["accepted"] = accepted
        print("  " + ", ".join(f"{k}={row[k]}" for k in ["status", "objective", "leaves", "seconds", "wall", "peak_rss_gb", "graph_size"]),
              flush=True)
        new = not OUT.exists()
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow(row)
        if accepted:
            table = pd.read_csv(KNOWN)
            m = (table["dataset"] == name) & (table["lam"] == lam)
            table.loc[m, ["objective", "certified", "source"]] = [round(float(row["objective"]), 6), True, "gosdt_patched"]
            table.to_csv(KNOWN, index=False)
            known[(name, lam)] = (float(row["objective"]), True)
            print(f"  certified: known_optima.csv updated ({name} λ={lam:g} = {row['objective']:.6f})", flush=True)


if __name__ == "__main__":
    main()
