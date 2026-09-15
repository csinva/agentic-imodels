"""Repeat a baseline on the development suite to estimate timing variability.

Each repeat runs the solver through ``src/evaluate.evaluate_solver`` on the
suite (same cap as the loop) and appends per-pair rows, tagged with the repeat
number, to ``results/repeats.csv``.  ``--summary`` prints and writes
``results/repeats_summary.csv``: per model, the metrics of every run
(repeat 0 is the cached baseline row from ``pair_results.csv`` converted to the
suite cap, like ``run_baselines.py`` does) with mean and standard deviation.

Usage (from ``evolve_optimal_tree``)::

    uv run baselines/benchmarks/run_repeats.py --models gosdt --repeats 2
    uv run baselines/benchmarks/run_repeats.py --summary
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from evaluate import PAIR_CSV_COLS, evaluate_solver, print_summary, rows_from_benchmark, summarize  # noqa: E402
from suite import MEMORY_LIMIT, TIME_LIMIT  # noqa: E402

RESULTS = os.path.join(HERE, "results")
REPEATS_CSV = os.path.join(RESULTS, "repeats.csv")
SUMMARY_CSV = os.path.join(RESULTS, "repeats_summary.csv")
COLS = ["repeat"] + PAIR_CSV_COLS


def factories():
    import reference_solver
    from pygosdt_v1 import GOSDTClassifier
    out = {
        "gosdt": lambda lam, tl: reference_solver.ReferenceGOSDT(lam, tl, memory_limit=MEMORY_LIMIT),
        "pygosdt_v1": lambda lam, tl: GOSDTClassifier(regularization=lam, time_limit=tl, memory_limit=MEMORY_LIMIT),
    }

    def make_streed(lam, tl):
        import streed_solver
        return streed_solver.STreeD(lam, tl)

    out["streed"] = make_streed
    return out


def next_repeat(model: str) -> int:
    if not os.path.exists(REPEATS_CSV):
        return 1
    done = [int(r["repeat"]) for r in csv.DictReader(open(REPEATS_CSV, newline="")) if r["model"] == model]
    return max(done, default=0) + 1


def append_row(row: dict, repeat: int):
    os.makedirs(RESULTS, exist_ok=True)
    new = not os.path.exists(REPEATS_CSV)
    with open(REPEATS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"repeat": repeat, **row})


def write_summary():
    reps = pd.read_csv(REPEATS_CSV) if os.path.exists(REPEATS_CSV) else pd.DataFrame(columns=COLS)
    out = []
    for model in sorted(set(reps["model"]) | {"gosdt", "pygosdt_v1", "streed"}):
        runs = []
        cached = rows_from_benchmark(model)
        if cached is not None:
            runs.append((0, summarize(cached)))
        for rep, d in reps[reps["model"] == model].groupby("repeat"):
            rows = d.to_dict("records")
            for r in rows:
                r["seconds"] = float(r["seconds"]) if not pd.isna(r["seconds"]) else TIME_LIMIT
            runs.append((int(rep), summarize(rows)))
        if not runs:
            continue
        for rep, s in runs:
            out.append({"model": model, "repeat": rep, "n_solved": s["n_solved"],
                        "geo_mean_time": round(s["geo_mean_time"], 4), "n_wrong": s["n_wrong"],
                        "total_time": round(sum(float(r["seconds"]) for r in s["rows"]), 1)})
        if len(runs) > 1:
            df = pd.DataFrame(out)
            d = df[df["model"] == model]
            out.append({"model": model, "repeat": "mean", "n_solved": round(d["n_solved"].mean(), 2),
                        "geo_mean_time": round(d["geo_mean_time"].mean(), 4), "n_wrong": round(d["n_wrong"].mean(), 2),
                        "total_time": round(d["total_time"].mean(), 1)})
            out.append({"model": model, "repeat": "std", "n_solved": round(d["n_solved"].std(ddof=1), 2),
                        "geo_mean_time": round(d["geo_mean_time"].std(ddof=1), 4), "n_wrong": round(d["n_wrong"].std(ddof=1), 2),
                        "total_time": round(d["total_time"].std(ddof=1), 1)})
    df = pd.DataFrame(out)
    df.to_csv(SUMMARY_CSV, index=False)
    print(df.to_string(index=False))
    print(f"-> {SUMMARY_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gosdt")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--summary", action="store_true", help="only (re)write the summary")
    args = ap.parse_args()
    if not args.summary:
        f = factories()
        for name in [m for m in args.models.split(",") if m]:
            for _ in range(args.repeats):
                rep = next_repeat(name)
                print("\n" + "=" * 60 + f"\n  {name}  repeat {rep}  (cap {args.time_limit:g}s)\n" + "=" * 60, flush=True)
                t0 = time.time()
                s = evaluate_solver(f[name], name, time_limit=args.time_limit, on_row=lambda r, rep=rep: append_row(r, rep))
                print_summary(name, s)
                print(f"repeat {rep} wall: {time.time() - t0:.0f}s", flush=True)
    write_summary()
