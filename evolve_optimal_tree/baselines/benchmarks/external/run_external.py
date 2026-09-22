"""Time the key solvers on TabArena-14, the held-out mirror of the development suite.

Runs every solver in ``MODELS`` on the 14 datasets built by build_tabarena14.py at the
suite's five penalties and 30 s cap, three times each. Rounds are interleaved (every
solver's first run, then every solver's second, ...) so a partial sweep is already a
like-for-like comparison, and rows are appended as each problem finishes so an
interrupted sweep resumes where it stopped.

Timings are the measurement, so before each solver's run the script waits until no
other compute job has used a core for a minute, and records the load average it started
at (interactive apps are not waited for, so that load average is the record of them).

There are no certified optima for these datasets, so correctness is judged after the
fact against the best tree any run of any solver returned for the problem (every
objective is recomputed from the returned tree, so that tree exists): a run that
certifies optimality with a worse objective issued a false certificate.

    uv run baselines/benchmarks/external/run_external.py            # the sweep
    uv run baselines/benchmarks/external/run_external.py --summary  # summary only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from evaluate import NoModel, evaluate_tree  # noqa: E402
from suite import LAMBDAS, TIME_LIMIT  # noqa: E402

DATA = os.path.join(HERE, "data")
PAIRS_CSV = os.path.join(HERE, "results", "external_pairs.csv")
SUMMARY_CSV = os.path.join(HERE, "results", "external_summary.csv")
CAP = TIME_LIMIT
COLS = ["model", "repeat", "dataset", "n", "p", "lam", "objective", "errors", "leaves", "seconds",
        "wall", "status", "iterations", "binary_features", "lb", "ub", "load1"]
MODELS = ["gosdt", "pygosdt_v1", "streed", "gosdt_guesses", "gosdt_guesses_guided", "split",
          "autoopttree", "autoopttree_v40", "autoopttree_v46", "autoopttree_v46_anytime100",
          "autoopttree_v49"]
REPEATS = 3
TOL = 1e-9


def factories():
    spec = importlib.util.spec_from_file_location(
        "run_repeats_all", os.path.join(os.path.dirname(HERE), "run_repeats_all.py"))
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    f = rr.factories()

    def make_split(lam, tl):
        import split_solver
        return split_solver.Split(lam, tl)

    f["split"] = make_split
    return f


def datasets():
    manifest = pd.read_csv(os.path.join(DATA, "manifest.csv"))
    return list(manifest["name"])


def done_pairs():
    if not os.path.exists(PAIRS_CSV):
        return set()
    d = pd.read_csv(PAIRS_CSV)
    return set(zip(d["model"], d["repeat"], d["dataset"], d["lam"]))


def append(row):
    os.makedirs(os.path.dirname(PAIRS_CSV), exist_ok=True)
    new = not os.path.exists(PAIRS_CSV)
    with open(PAIRS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# compute jobs that would distort a timing; interactive apps are not waited for, since the
# machine may be in normal use, but the load average at each block's start is recorded
COMPUTE = ("python", "uv", "gosdt", "streed", "split", "node", "java", "cc1", "clang", "rustc", "cargo")


def wait_for_idle(quiet_seconds=60):
    """Block until no other compute job has used more than 15% of a core for `quiet_seconds`."""
    mine = {os.getpid(), os.getppid()}
    quiet = 0
    while quiet < quiet_seconds:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pcpu=,comm="], capture_output=True, text=True).stdout
        busy = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3 or int(parts[0]) in mine or float(parts[1]) <= 15:
                continue
            if any(k in os.path.basename(parts[2]).lower() for k in COMPUTE):
                busy.append(line)
        quiet = quiet + 10 if not busy else 0
        time.sleep(10)


def run(models, repeats):
    f = factories()
    todo = done_pairs()
    names = datasets()
    for rep in range(1, repeats + 1):
        for model in models:
            pending = [(d, lam) for d in names for lam in LAMBDAS if (model, rep, d, lam) not in todo]
            if not pending:
                continue
            wait_for_idle()
            load1 = os.getloadavg()[0]
            print(f"\n=== {model} repeat {rep}: {len(pending)} problems (load {load1:.2f})", flush=True)
            t_block = time.time()
            for name in names:
                lams = [lam for d, lam in pending if d == name]
                if not lams:
                    continue
                frame = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
                X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
                for lam in lams:
                    row = {"model": model, "repeat": rep, "dataset": name, "n": len(frame),
                           "p": frame.shape[1] - 1, "lam": lam, "load1": round(load1, 2)}
                    t0 = time.perf_counter()
                    try:
                        m = f[model](lam, CAP)
                        m.fit(X, y)
                        wall = time.perf_counter() - t0
                        seconds = float(getattr(m, "time_", wall))
                        errors, leaves = evaluate_tree(m.tree_, frame)
                        optimal = bool(getattr(m, "optimal_", False))
                        row.update(objective=errors / len(frame) + lam * leaves, errors=errors, leaves=leaves,
                                   seconds=min(seconds, CAP), wall=round(wall, 3),
                                   status="optimal" if optimal else str(getattr(m, "stop_reason_", "time") or "time"),
                                   iterations=getattr(m, "iterations_", ""),
                                   binary_features=getattr(m, "n_binary_features_", ""),
                                   lb=getattr(m, "lowerbound_", ""), ub=getattr(m, "upperbound_", ""))
                    except NoModel as exc:
                        row.update(objective=float("nan"), seconds=CAP,
                                   wall=round(time.perf_counter() - t0, 3), status=exc.status)
                    except Exception:  # noqa: BLE001 - a crashing solver must not stop the sweep
                        traceback.print_exc()
                        row.update(objective=float("nan"), seconds=CAP,
                                   wall=round(time.perf_counter() - t0, 3), status="crash")
                    append(row)
                    print(f"  {name:20s} λ={lam:<6} {row['status']:8s} obj={row['objective']:.6f} "
                          f"t={row['seconds']:6.2f}s", flush=True)
            print(f"=== {model} repeat {rep} wall {time.time() - t_block:.0f}s", flush=True)


def summarize():
    d = pd.read_csv(PAIRS_CSV)
    best = d.groupby(["dataset", "lam"])["objective"].min().rename("best")
    d = d.join(best, on=["dataset", "lam"])
    d["wrong"] = (d["status"] == "optimal") & (d["objective"] > d["best"] + TOL)
    # a crash counts as wrong too, as it does on the development suite
    d.loc[d["status"] == "crash", "wrong"] = True
    out = []
    for (model, rep), g in d.groupby(["model", "repeat"]):
        secs = g["seconds"].fillna(CAP).clip(lower=1e-3, upper=CAP)
        out.append({"model": model, "repeat": rep, "pairs": len(g),
                    "n_solved": int((g["status"] == "optimal").sum()),
                    "geo_mean_time": round(float(np.exp(np.log(secs).mean())), 5),
                    "n_wrong": int(g["wrong"].sum()), "no_tree": int(g["objective"].isna().sum()),
                    "load1_at_start": g["load1"].iloc[0]})
    s = pd.DataFrame(out)
    s.to_csv(SUMMARY_CSV, index=False)
    print(s.to_string(index=False))
    wrong = d[d["wrong"]]
    if len(wrong):
        print("\nfalse certificates / crashes:")
        print(wrong[["model", "repeat", "dataset", "lam", "status", "objective", "best"]].to_string(index=False))
    print("->", SUMMARY_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data", default="data", help="dataset folder under external/ (e.g. data_full)")
    ap.add_argument("--cap", type=float, default=CAP, help="seconds per problem")
    ap.add_argument("--tag", default="", help="suffix for the results files")
    ap.add_argument("--lams", default="", help="comma-separated penalties (default: the suite's five)")
    args = ap.parse_args()
    DATA = os.path.join(HERE, args.data)
    if args.lams:
        LAMBDAS = [float(x) for x in args.lams.split(",")]
    CAP = args.cap
    if args.tag:
        PAIRS_CSV = os.path.join(HERE, "results", f"external_pairs_{args.tag}.csv")
        SUMMARY_CSV = os.path.join(HERE, "results", f"external_summary_{args.tag}.csv")
    if not args.summary:
        run([m for m in args.models.split(",") if m], args.repeats)
    summarize()
