"""TabArena-14 without the 30 s cap: every solver on the 70 hidden problems until it
certifies, stops at the 6 GB memory budget, or reaches a wall of hours.

The 30 s sweep (run_external.py) asks how much each solver proves in a fixed budget.
This asks the other question: given hours rather than seconds, what stops it? The wall
(default 4 h a problem, 480 times the suite's cap) is handed to each solver as its time
limit, so a run that reaches it returns the best tree it has, as at 30 s; the driver
only kills a run that overshoots it. Each problem runs in its own child process pinned
to its own cores (2 for single-threaded solvers, 8 for the multicore ones), many at
once, so a sweep that would take weeks one problem at a time takes hours; timings are
the solver's own clock, and since the problems are CPU bound and each has its cores to
itself they are comparable across solvers on this machine, though not to the 30 s
panel, which was measured on a different one.

Rows are appended as each problem finishes, so an interrupted sweep resumes. Status is
``optimal``, ``memory`` (the solver or this driver stopped it at the memory budget),
``time`` (the wall), ``unfinished`` (killed for overshooting the wall), ``crash``, or
the wrapper's own status. ``seconds`` is the solver's reported optimisation time, or
the wall time at the stop for a run that returned no tree.

    uv run baselines/benchmarks/external/run_external_nolimit.py            # the sweep
    uv run baselines/benchmarks/external/run_external_nolimit.py --summary  # summary only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import signal
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

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
MODELS = ["gosdt", "pygosdt_v1", "streed", "gosdt_guesses", "gosdt_guesses_guided", "split",
          "gg_exact_simsup", "gg_guided_e60d2",
          "autoopttree", "autoopttree_v40", "autoopttree_v46", "autoopttree_v46_anytime100",
          "autoopttree_v49"]
MULTICORE = {"autoopttree_v46", "autoopttree_v46_anytime100", "autoopttree_v49"}
LAMBDAS = [0.1, 0.05, 0.02, 0.01, 0.005]
COLS = ["model", "repeat", "dataset", "n", "p", "lam", "objective", "errors", "leaves", "seconds",
        "wall", "status", "iterations", "binary_features", "lb", "ub", "peak_rss_gb", "cores"]
TOL = 1e-9


def _rx():
    spec = importlib.util.spec_from_file_location("run_external", os.path.join(HERE, "run_external.py"))
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    return rx


# ----------------------------------------------------------------------------- worker

def worker(model, dataset, lam, out_path, wall):
    from evaluate import NoModel, evaluate_tree
    rx = _rx()
    frame = pd.read_csv(os.path.join(DATA, f"{dataset}.csv"))
    X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
    row = {"n": len(frame), "p": frame.shape[1] - 1}
    t0 = time.perf_counter()
    try:
        m = rx.factories()[model](lam, wall)
        m.fit(X, y)
        wall = time.perf_counter() - t0
        seconds = float(getattr(m, "time_", wall))
        errors, leaves = evaluate_tree(m.tree_, frame)
        optimal = bool(getattr(m, "optimal_", False))
        row.update(objective=errors / len(frame) + lam * leaves, errors=errors, leaves=leaves,
                   seconds=seconds, wall=round(wall, 3),
                   status="optimal" if optimal else str(getattr(m, "stop_reason_", "time") or "time"),
                   iterations=getattr(m, "iterations_", ""),
                   binary_features=getattr(m, "n_binary_features_", ""),
                   lb=getattr(m, "lowerbound_", ""), ub=getattr(m, "upperbound_", ""))
    except NoModel as exc:
        wall = time.perf_counter() - t0
        row.update(objective=float("nan"), seconds=wall, wall=round(wall, 3), status=exc.status)
    except Exception:  # noqa: BLE001 - a crashing solver is a row, not the end of the sweep
        traceback.print_exc()
        wall = time.perf_counter() - t0
        row.update(objective=float("nan"), seconds=wall, wall=round(wall, 3), status="crash")
    with open(out_path, "w") as fh:
        json.dump(row, fh)


# ----------------------------------------------------------------------------- driver

def _tree_rss(pid):
    """Largest resident set (bytes) of any process in pid's tree, and the pids."""
    out = subprocess.run(["ps", "-o", "pid=,ppid=,rss=", "-A"], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.splitlines():
        p, pp, r = line.split()
        kids.setdefault(int(pp), []).append(int(p))
        rss[int(p)] = int(r) * 1024
    seen, stack, peak = [], [pid], 0
    while stack:
        p = stack.pop()
        seen.append(p)
        peak = max(peak, rss.get(p, 0))
        stack.extend(kids.get(p, []))
    return peak, seen


def _kill_tree(pid):
    _, pids = _tree_rss(pid)
    for p in reversed(pids):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def hard_first(models, names, pairs30):
    """Problems a solver did not certify at 30 s go first, widest dataset first, so the
    long runs start early and the sweep's tail is short."""
    sizes = {n: os.path.getsize(os.path.join(DATA, f"{n}.csv")) for n in names}
    jobs = []
    for model in models:
        for name in names:
            for lam in LAMBDAS:
                easy = pairs30.get((model, name, lam)) == "optimal"
                jobs.append((easy, -sizes[name], lam, model, name))
    jobs.sort()
    return [(m, n, lam) for _, _, lam, m, n in jobs]


def run(args):
    rx = _rx()
    names = rx.datasets()
    if args.datasets:
        names = [n for n in names if n in args.datasets.split(",")]
    pairs_csv = os.path.join(RESULTS, f"external_pairs_{args.tag}.csv")
    done = set()
    if os.path.exists(pairs_csv):
        d = pd.read_csv(pairs_csv)
        done = set(zip(d["model"], d["dataset"], d["lam"]))
    pairs30 = {}
    p30 = os.path.join(RESULTS, "external_pairs.csv")
    if os.path.exists(p30):
        d = pd.read_csv(p30)
        d = d[d["repeat"] == 1]
        pairs30 = dict(zip(zip(d["model"], d["dataset"], d["lam"]), d["status"]))
    models = [m for m in args.models.split(",") if m]
    jobs = [j for j in hard_first(models, names, pairs30) if j not in done]
    print(f"{len(jobs)} problems to run, {len(done)} already done -> {pairs_csv}", flush=True)

    free = list(range(args.core_lo, args.core_hi + 1))
    running = {}   # pid -> dict(job, cores, t0, out, proc, peak)
    mem_cap = args.memory_gb * (1 << 30)
    work = os.path.join(RESULTS, "work_nolimit")
    os.makedirs(work, exist_ok=True)
    # numba's disk cache records the evolved module under a dynamic name that a fresh
    # process cannot re-import, so each worker gets an empty cache directory of its own and
    # compiles (about a minute, outside the timed fit)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def launch(job):
        model, name, lam = job
        k = 8 if model in MULTICORE else 2
        cores, free[:] = free[:k], free[k:]
        out = os.path.join(work, f"{model}__{name}__{lam}.json")
        if os.path.exists(out):
            os.remove(out)
        cmd = ["taskset", "-c", ",".join(map(str, cores)), sys.executable, __file__,
               "--worker", model, name, str(lam), out, "--wall", str(args.wall)]
        log = open(os.path.join(work, f"{model}__{name}__{lam}.log"), "w")
        cache = os.path.join(work, "numba_cache", f"{model}__{name}__{lam}")
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                env=dict(env, NUMBA_CACHE_DIR=cache))
        running[proc.pid] = dict(job=job, cores=cores, t0=time.time(), out=out, proc=proc, peak=0,
                                 log=log, cache=cache)

    def finish(pid, killed=""):
        r = running.pop(pid)
        free.extend(r["cores"])
        r["log"].close()
        shutil.rmtree(r["cache"], ignore_errors=True)
        model, name, lam = r["job"]
        wall = time.time() - r["t0"]
        row = {"model": model, "repeat": 1, "dataset": name, "lam": lam,
               "peak_rss_gb": round(r["peak"] / (1 << 30), 2), "cores": len(r["cores"])}
        if killed:
            frame = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
            row.update(n=len(frame), p=frame.shape[1] - 1, objective=float("nan"),
                       seconds=round(wall, 3), wall=round(wall, 3), status=killed)
        elif os.path.exists(r["out"]):
            row.update(json.load(open(r["out"])))
        else:
            frame = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
            row.update(n=len(frame), p=frame.shape[1] - 1, objective=float("nan"),
                       seconds=round(wall, 3), wall=round(wall, 3),
                       status="memory" if r["proc"].returncode < 0 else "crash")
        new = not os.path.exists(pairs_csv)
        with open(pairs_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        obj = row.get("objective", float("nan"))
        print(f"  {model:28s} {name:20s} λ={lam:<6} {row['status']:10s} obj={obj:.6f} "
              f"t={row['seconds']:9.2f}s peak={row['peak_rss_gb']:.2f}GB  [{len(running)} running, {len(jobs)} queued]",
              flush=True)

    while jobs or running:
        while jobs and len(running) < args.jobs:
            # the first queued job whose cores are free, so a 2-core job is not held
            # behind an 8-core one
            fits = [i for i, j in enumerate(jobs) if len(free) >= (8 if j[0] in MULTICORE else 2)]
            if not fits:
                break
            launch(jobs.pop(fits[0]))
        time.sleep(1.0)
        for pid in list(running):
            r = running[pid]
            if r["proc"].poll() is not None:
                finish(pid)
                continue
            peak, _ = _tree_rss(pid)
            r["peak"] = max(r["peak"], peak)
            if peak > mem_cap:
                _kill_tree(pid)
                r["proc"].wait()
                finish(pid, "memory")
            elif time.time() - r["t0"] > 1.1 * args.wall + 300:
                _kill_tree(pid)
                r["proc"].wait()
                finish(pid, "unfinished")


def summarize(tag, wall):
    pairs_csv = os.path.join(RESULTS, f"external_pairs_{tag}.csv")
    d = pd.read_csv(pairs_csv)
    best = d.groupby(["dataset", "lam"])["objective"].min().rename("best")
    d = d.join(best, on=["dataset", "lam"])
    d["wrong"] = (d["status"] == "optimal") & (d["objective"] > d["best"] + TOL)
    d.loc[d["status"] == "crash", "wrong"] = True
    out = []
    for model, g in d.groupby("model"):
        secs = g["seconds"].clip(lower=1e-3)
        out.append({"model": model, "pairs": len(g),
                    "n_solved": int((g["status"] == "optimal").sum()),
                    "geo_mean_time": round(float(np.exp(np.log(secs).mean())), 4),
                    "max_time": round(float(secs.max()), 1),
                    "total_hours": round(float(g["wall"].sum()) / 3600, 2),
                    "n_wrong": int(g["wrong"].sum()), "no_tree": int(g["objective"].isna().sum()),
                    "memory": int((g["status"] == "memory").sum()),
                    "at_wall": int(((g["status"] == "time") & (g["seconds"] >= 0.99 * wall)).sum()
                                   + (g["status"] == "unfinished").sum()),
                    "peak_rss_gb": round(float(g["peak_rss_gb"].max()), 2)})
    s = pd.DataFrame(out)
    path = os.path.join(RESULTS, f"external_summary_{tag}.csv")
    s.to_csv(path, index=False)
    print(s.to_string(index=False))
    wrong = d[d["wrong"]]
    if len(wrong):
        print("\nfalse certificates / crashes:")
        print(wrong[["model", "dataset", "lam", "status", "objective", "best"]].to_string(index=False))
    print("->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, metavar=("MODEL", "DATASET", "LAM", "OUT"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default="", help="comma-separated subset (default: all 14)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tag", default="nolimit")
    ap.add_argument("--wall", type=float, default=4 * 3600,
                    help="seconds a problem may take; given to the solver as its time limit")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    ap.add_argument("--jobs", type=int, default=32, help="problems run at once")
    ap.add_argument("--core-lo", type=int, default=8)
    ap.add_argument("--core-hi", type=int, default=167, help="cpu ids handed to the workers")
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], float(args.worker[2]), args.worker[3], args.wall)
    elif args.summary:
        summarize(args.tag, args.wall)
    else:
        run(args)
        summarize(args.tag, args.wall)
