"""Scoring of an optimal-tree solver on the development suite.

A solver is any callable ``make_model(regularization, time_limit) -> estimator``
returning an object with ``fit(X, y)`` and, after fitting, ``tree_`` (the
reference JSON tree schema), ``optimal_`` (bool) and ``time_`` (optimisation
seconds).  ``evaluate_solver`` fits it on every (dataset, λ) pair of the suite,
recomputes the objective of the returned tree independently from the tree and
the raw data, checks it against the best known objective, and returns the
three leaderboard metrics:

* ``n_wrong``       pairs where a *certified* result disagrees with a certified
                    known optimum, or the solver crashed (must be 0: the solver
                    is not exact);
* ``n_solved``      pairs certified optimal within the time cap (higher is better);
* ``geo_mean_time`` geometric mean over pairs of the optimisation time, with
                    unsolved pairs counted at the cap (lower is better).

Do not modify.
"""

from __future__ import annotations

import csv
import math
import os
import time
import traceback

import numpy as np
import pandas as pd

from suite import (DATASETS, LAMBDAS, MEMORY_LIMIT, RESULTS_DIR, TIME_LIMIT, load_dataset,
                   load_known_optima)

OVERALL_CSV_COLS = ["commit", "n_solved", "geo_mean_time", "n_wrong", "status", "model_name", "description"]
PAIR_CSV_COLS = ["model", "dataset", "lam", "objective", "errors", "leaves", "seconds", "status",
                 "known_objective", "known_certified", "verdict"]
TOL = 1e-6


# ------------------------------------------------------------- objective
def evaluate_tree(node: dict, frame: pd.DataFrame):
    """Independent (errors, leaves) of a JSON tree on a frame whose last column is the label."""
    if "prediction" in node:
        labels = frame.iloc[:, -1].to_numpy()
        pred = node["prediction"]
        try:
            errors = int(np.sum(labels != pred))
        except TypeError:
            errors = int(np.sum(labels.astype(str) != str(pred)))
        return errors, 1
    col = frame.iloc[:, node["feature"]]
    rel, ref = node["relation"], node["reference"]
    if rel == ">=":
        mask = (pd.to_numeric(col, errors="coerce") >= ref).to_numpy()
    elif rel == "<=":
        mask = (pd.to_numeric(col, errors="coerce") <= ref).to_numpy()
    elif isinstance(ref, str):
        mask = (col.astype(str) == ref).to_numpy()
    else:
        mask = (pd.to_numeric(col, errors="coerce") == ref).to_numpy()
    e1, l1 = evaluate_tree(node["true"], frame[mask])
    e2, l2 = evaluate_tree(node["false"], frame[~mask])
    return e1 + e2, l1 + l2


# ---------------------------------------------------------------- scoring
def verdict_for(objective: float, claimed_optimal: bool, k_obj: float, k_cert: bool) -> str:
    """Compare a returned objective with the best known one.

    ``WRONG`` (an exactness violation) is reserved for *certified* results that
    disagree with a certified known optimum in either direction: a solver that
    certifies a worse tree has an unsound bound, one that certifies a better
    value than a certified optimum has a false certificate.  An uncertified
    incumbent (time cap) is never wrong, only ``worse_than_known``.
    """
    if math.isnan(k_obj):
        return "ok"
    if objective > k_obj + TOL:
        return "WRONG" if (claimed_optimal and k_cert) else "worse_than_known"
    if objective < k_obj - TOL:
        return "WRONG" if (claimed_optimal and k_cert) else "better_than_known"
    return "ok"


def evaluate_solver(make_model, model_name: str, datasets=None, lambdas=None, time_limit=None,
                    verbose=True) -> dict:
    """Fit ``make_model`` on the suite and return metrics plus per-pair rows."""
    datasets = datasets or [d for d, _ in DATASETS]
    lambdas = lambdas or LAMBDAS
    time_limit = TIME_LIMIT if time_limit is None else time_limit
    known = load_known_optima()
    rows = []
    for name in datasets:
        frame = load_dataset(name)
        X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
        n = frame.shape[0]
        for lam in lambdas:
            k_obj, k_cert = known.get((name, lam), (float("nan"), False))
            row = {"model": model_name, "dataset": name, "lam": lam, "known_objective": k_obj,
                   "known_certified": k_cert}
            t0 = time.perf_counter()
            try:
                model = make_model(lam, time_limit)
                model.fit(X, y)
                seconds = float(getattr(model, "time_", time.perf_counter() - t0))
                errors, leaves = evaluate_tree(model.tree_, frame)
                objective = errors / n + lam * leaves
                optimal = bool(getattr(model, "optimal_", False))
                row.update(objective=objective, errors=errors, leaves=leaves,
                           seconds=min(seconds, time_limit), status="optimal" if optimal else "time")
                row["verdict"] = verdict_for(objective, optimal, k_obj, k_cert)
            except Exception:  # noqa: BLE001 - a crashing solver must not abort the whole suite
                traceback.print_exc()
                row.update(objective=float("nan"), errors="", leaves="", seconds=time_limit,
                           status="crash", verdict="WRONG")
            rows.append(row)
            if verbose:
                print(f"  {name:17s} λ={lam:<6} {row['status']:7s} obj={row['objective']:.6f} "
                      f"t={row['seconds']:6.2f}s known={k_obj:.6f}{'*' if k_cert else ''} {row['verdict']}",
                      flush=True)
    return summarize(rows)


def summarize(rows) -> dict:
    n_wrong = sum(r["verdict"] == "WRONG" for r in rows)
    n_solved = sum(r["status"] == "optimal" for r in rows)
    times = [max(float(r["seconds"]), 1e-3) for r in rows]
    geo = float(math.exp(np.mean(np.log(times)))) if times else float("nan")
    return {"n_wrong": n_wrong, "n_solved": n_solved, "n_pairs": len(rows), "geo_mean_time": geo, "rows": rows}


def print_summary(model_name: str, s: dict):
    print("---")
    print(f"model:          {model_name}")
    print(f"n_solved:       {s['n_solved']}/{s['n_pairs']} certified optimal within {TIME_LIMIT:g}s")
    print(f"geo_mean_time:  {s['geo_mean_time']:.3f}s")
    print(f"n_wrong:        {s['n_wrong']}  (certified result disagreeing with a certified optimum, or a crash; must be 0)")
    worse = [r for r in s["rows"] if r["verdict"] == "worse_than_known"]
    if worse:
        print(f"uncertified incumbent worse than best known on {len(worse)} pairs: " +
              ", ".join(f"{r['dataset']} λ={r['lam']}" for r in worse))
    better = [r for r in s["rows"] if r["verdict"] == "better_than_known"]
    if better:
        print(f"better than best known on {len(better)} pairs: " +
              ", ".join(f"{r['dataset']} λ={r['lam']}" for r in better))


# --------------------------------------------------------------- results
def upsert_overall_results(rows, results_dir=RESULTS_DIR):
    """Write or update overall_results.csv, replacing rows with the same (model_name, description)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    existing = []
    new_keys = {(r["model_name"], r.get("description", "")) for r in rows}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_name"), row.get("description", "")) not in new_keys:
                    existing.append(row)
    all_rows = existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Overall results saved → {path}")


def upsert_pair_results(rows, results_dir=RESULTS_DIR):
    """Write or update pair_results.csv (one row per model × dataset × λ)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "pair_results.csv")
    models = {r["model"] for r in rows}
    existing = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = [row for row in csv.DictReader(f) if row.get("model") not in models]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    print(f"Per-pair results saved → {path}")


def record(model_name: str, description: str, s: dict, commit: str, status: str = ""):
    upsert_pair_results(s["rows"])
    upsert_overall_results([{
        "commit": commit,
        "n_solved": s["n_solved"],
        "geo_mean_time": f"{s['geo_mean_time']:.3f}",
        "n_wrong": s["n_wrong"],
        "status": status,
        "model_name": model_name,
        "description": description,
    }])


# --------------------------------------------------- cached baseline rows
BENCHMARK_SUMMARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "benchmarks", "results", "summary.csv")
BENCHMARK_COLUMNS = {"gosdt": "ref", "pygosdt_v1": "py"}


def rows_from_benchmark(model_name: str, time_limit=None) -> list[dict] | None:
    """Per-pair rows for a baseline taken from the full benchmark in ``benchmarks/results``.

    The benchmark ran with a 600 s cap; a pair counts as solved here only if it was
    certified within the suite's cap, and its time is capped at the suite's cap, so
    the rows are exactly what a run of the suite would have produced (times differ
    only by machine noise).  Returns None if the benchmark summary is missing.
    """
    if not os.path.exists(BENCHMARK_SUMMARY):
        return None
    time_limit = TIME_LIMIT if time_limit is None else time_limit
    prefix = BENCHMARK_COLUMNS[model_name]
    table = pd.read_csv(BENCHMARK_SUMMARY)
    known = load_known_optima()
    rows = []
    for name, _ in DATASETS:
        n = None
        for lam in LAMBDAS:
            hit = table[(table["dataset"] == name) & (table["lam"] == lam)]
            if not len(hit):
                return None
            r = hit.iloc[0]
            if n is None:
                n = int(r["n"])
            k_obj, k_cert = known.get((name, lam), (float("nan"), False))
            row = {"model": model_name, "dataset": name, "lam": lam, "known_objective": k_obj,
                   "known_certified": k_cert}
            objective = r[f"{prefix}_objective"]
            seconds = r[f"{prefix}_time"]
            stop = str(r[f"{prefix}_stop"]) if not pd.isna(r[f"{prefix}_stop"]) else ""
            if pd.isna(objective):
                row.update(objective=float("nan"), errors="", leaves="", seconds=time_limit,
                           status=stop or "crash", verdict="no_tree")
            else:
                solved = stop == "optimal" and float(seconds) <= time_limit
                if solved:
                    status = "optimal"
                elif stop == "memory":
                    status = "memory"
                else:
                    status = "time"
                seconds = time_limit if pd.isna(seconds) else min(float(seconds), time_limit)
                row.update(objective=float(objective), errors=int(r[f"{prefix}_errors"]),
                           leaves=int(r[f"{prefix}_leaves"]), seconds=seconds, status=status)
                row["verdict"] = verdict_for(float(objective), stop == "optimal", k_obj, k_cert)
            rows.append(row)
    return rows
