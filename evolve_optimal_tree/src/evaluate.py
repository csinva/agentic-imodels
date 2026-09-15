"""Scoring of an optimal-tree solver on the development suite.

A solver is any callable ``make_model(regularization, time_limit) -> estimator``
returning an object with ``fit(X, y)`` and, after fitting, ``tree_`` (the
reference JSON tree schema), ``optimal_`` (bool) and ``time_`` (optimisation
seconds).  ``evaluate_solver`` fits it on every (dataset, λ) pair of the suite,
recomputes the objective of the returned tree independently from the tree and
the raw data, checks it against the best known objective, and returns the
three leaderboard metrics:

* ``n_wrong``       pairs whose objective is worse than a *certified* known
                    optimum (must be 0: the solver is not exact);
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
                if not math.isnan(k_obj) and objective > k_obj + TOL and k_cert:
                    verdict = "WRONG"
                elif not math.isnan(k_obj) and objective > k_obj + TOL:
                    verdict = "worse_than_incumbent"
                elif not math.isnan(k_obj) and objective < k_obj - TOL:
                    verdict = "better_than_known"
                else:
                    verdict = "ok"
                if optimal and not math.isnan(k_obj) and objective < k_obj - TOL and k_cert:
                    verdict = "WRONG"  # a certificate below a certified optimum is a false certificate
                row["verdict"] = verdict
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
    print(f"n_wrong:        {s['n_wrong']}  (objective worse than a certified optimum, or a false certificate; must be 0)")
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
