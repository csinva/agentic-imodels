"""Scoring for the exact track: certified sparse integer risk scores. Do not modify.

A solver may set ``lower_bound_`` after ``fit``: a proven lower bound on the minimum calibrated
training loss over EVERY feasible point vector (integers in [-5, 5], at most k nonzero). The
harness recomputes the loss of the returned points as usual and then

* certified   lower_bound_ >= loss - CERT_TOL, returned within the time limit: the points are
              proven optimal;
* WRONG       lower_bound_ > (a loss some point vector is known to reach) + CERT_TOL, i.e. above
              the returned loss, above the problem's best known loss (src/best_known.csv), or, on
              the tiny suite, above the true optimum (src/tiny_optima.csv); or a certified answer
              on the tiny suite whose loss is not the optimum; or an invalid answer or a crash.

``n_wrong`` must be 0. The visible suite gives ``n_certified`` (of 70) and the usual time and
regret; the tiny suite (30 problems x k in {2, 3}, 10 features each, optima by enumeration) is the
correctness gate and gives ``n_certified_tiny``.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from evaluate import evaluate_solver, load_best_known, _upsert
from suite import K_VALUES, RESULTS_DIR, SRC_DIR, TIME_LIMIT

CERT_TOL = 1e-7
TINY_KS = [2, 3]
OVERALL_COLS = ["commit", "n_certified", "n_certified_tiny", "n_wrong", "geo_mean_time", "mean_regret",
                "mean_test_auc", "status", "model_name", "description"]


def load_tiny_optima():
    out = {}
    with open(os.path.join(SRC_DIR, "tiny_optima.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], int(r["k"]))] = float(r["loss"])
    return out


def judge(rows, suite):
    """Add certified / verdict to each row; returns the rows."""
    best = load_best_known()
    opt = load_tiny_optima() if suite == "tiny" else {}
    for r in rows:
        lb = r.get("lower_bound", float("nan"))
        lb = float("nan") if lb in ("", None) else float(lb)
        loss = float(r["loss"])
        known = opt.get((r["dataset"], r["k"])) if suite == "tiny" else best.get((suite, r["dataset"], r["k"]))
        known = float("nan") if known is None else known
        verdict = "ok"
        if r["status"] in ("invalid", "crash"):
            verdict = "WRONG"
        elif not math.isnan(lb):
            if lb > loss + CERT_TOL or (not math.isnan(known) and lb > known + CERT_TOL):
                verdict = "WRONG"  # a bound above a loss that is actually reached
        certified = (not math.isnan(lb)) and r["status"] == "ok" and lb >= loss - CERT_TOL  # within the limit
        if suite == "tiny" and certified and not math.isnan(known) and loss > known + CERT_TOL:
            verdict = "WRONG"  # certified a point vector that is not optimal
        r["certified"] = bool(certified and verdict == "ok")
        r["verdict"] = verdict
        r["known"] = known
    return rows


def evaluate_exact(spec, model_name, datasets=None, ks=None, jobs=14, tiny=True, time_limit=TIME_LIMIT):
    vis = evaluate_solver(spec, model_name, suite="visible", datasets=datasets, ks=ks or K_VALUES,
                          time_limit=time_limit, jobs=jobs)
    vis_rows = judge(vis["rows"], "visible")
    tiny_rows = []
    if tiny:
        t = evaluate_solver(spec, model_name, suite="tiny", ks=TINY_KS, time_limit=time_limit, jobs=jobs,
                            verbose=False)
        tiny_rows = judge(t["rows"], "tiny")
    return {**vis, "rows": vis_rows, "tiny_rows": tiny_rows,
            "n_certified": sum(r["certified"] for r in vis_rows),
            "n_certified_tiny": sum(r["certified"] for r in tiny_rows),
            "n_wrong": sum(r["verdict"] == "WRONG" for r in vis_rows + tiny_rows)}


def print_exact(model_name, s):
    print("---")
    print(f"model:            {model_name}")
    print(f"n_wrong:          {s['n_wrong']}  (bound above a reached loss, false certificate, invalid or crash; "
          f"must be 0)")
    print(f"n_certified:      {s['n_certified']}/{len(s['rows'])} visible problems proven optimal")
    print(f"n_certified_tiny: {s['n_certified_tiny']}/{len(s['tiny_rows'])} tiny problems proven optimal")
    print(f"geo_mean_time:    {s['geo_mean_time']:.3f}s")
    print(f"mean_regret:      {s['mean_regret']:.5f}")
    print(f"mean_test_auc:    {s['mean_test_auc']:.4f}")
    for r in [r for r in s["rows"] + s["tiny_rows"] if r["verdict"] == "WRONG"][:10]:
        print(f"  WRONG: {r['suite']} {r['dataset']} k={r['k']} loss={r['loss']:.9f} "
              f"lb={r.get('lower_bound')} known={r['known']:.9f} {r['status']} {str(r.get('note'))[:120]}")


def record_exact(model_name, description, s, commit="", status="", results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    cols = ["model", "suite", "dataset", "k", "status", "seconds", "loss", "lower_bound", "known", "certified",
            "verdict", "auc_test", "points"]
    _upsert(os.path.join(results_dir, "problem_results.csv"), cols,
            [{c: r.get(c, "") for c in cols} for r in s["rows"] + s["tiny_rows"]], key=lambda r: r["model"])
    _upsert(os.path.join(results_dir, "overall_results.csv"), OVERALL_COLS, [{
        "commit": commit, "n_certified": s["n_certified"], "n_certified_tiny": s["n_certified_tiny"],
        "n_wrong": s["n_wrong"], "geo_mean_time": f"{s['geo_mean_time']:.3f}",
        "mean_regret": f"{s['mean_regret']:.5f}", "mean_test_auc": f"{s['mean_test_auc']:.4f}",
        "status": status, "model_name": model_name, "description": description}], key=lambda r: r["model_name"])
    print(f"results saved -> {results_dir}")
