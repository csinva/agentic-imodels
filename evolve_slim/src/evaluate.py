"""Scoring of a sparse integer linear model solver on a suite. Do not modify.

A solver is any callable ``make_model(k, time_limit) -> estimator`` whose ``fit(X, y)``
(``X`` a float matrix, ``y`` in {0, 1}) leaves integer points in ``coef_``, one per
column of ``X``. Nothing else the model reports is used. For every problem the
harness

1. checks the points: finite, integers, in [-COEF_BOUND, COEF_BOUND], at most k nonzero
   (a violation, or a crash, is ``invalid``: ``n_invalid`` must be 0);
2. computes the score ``s = X @ coef_`` and fits the score-to-risk map itself,
   ``P(y = 1) = sigmoid(a * s + b)``, by minimising the training logistic loss over
   the two real numbers ``a`` and ``b``. That minimum is the problem's criterion
   (mean training log loss, lower is better). A solver that returns no model is
   charged the loss of ``a = 0``, the base rate;
3. reports the test AUC of ``s`` and the test log loss of the fitted map.

``a`` plays the role of FasterRisk's 1 / multiplier and ``b`` of its intercept over
the multiplier (here not constrained to an integer). Scoring every solver's points
with the same optimal map means the criterion depends only on the points.

Fits run in parallel worker processes, one problem at a time per worker, each
worker single-threaded (BLAS, OpenMP and numba set to one thread). A worker first
fits a tiny warm-up problem, untimed, so import and JIT compilation are excluded.
Time is the wall-clock seconds of ``fit``. A fit still running at
``HARD_LIMIT_FACTOR`` times the limit is killed and counted as no model at that time.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

from suite import (COEF_BOUND, DATA_DIR, HARD_LIMIT_FACTOR, K_VALUES, RESULTS_DIR, SRC_DIR, TIME_LIMIT,
                   load_problem, suite_datasets)

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
OVERALL_CSV_COLS = ["commit", "mean_regret", "geo_mean_time", "mean_test_auc", "mean_loss", "n_invalid",
                    "n_killed", "n_over_limit", "integer", "status", "model_name", "description"]
PROBLEM_CSV_COLS = ["model", "suite", "dataset", "k", "n_train", "d", "status", "seconds", "nnz", "loss",
                    "best_known", "regret", "auc_train", "auc_test", "loss_test", "a", "b", "points", "note"]
BEST_KNOWN = os.path.join(SRC_DIR, "best_known.csv")


# ------------------------------------------------------------------ scoring
def calibrate(scores, y):
    """(a, b, loss): the minimum over real a, b of mean log(1 + exp(-(2y-1)(a s + b))).

    Convex in (a, b); solved by damped Newton on the distinct score values."""
    y = np.asarray(y, float)
    u, inv = np.unique(np.round(np.asarray(scores, float), 9), return_inverse=True)
    pos = np.bincount(inv, weights=y, minlength=len(u))
    cnt = np.bincount(inv, minlength=len(u)).astype(float)
    n = cnt.sum()
    p = min(max(pos.sum() / n, 1e-12), 1 - 1e-12)
    a, b = 0.0, math.log(p / (1 - p))

    def f(a_, b_):
        z = a_ * u + b_
        return float((pos * np.logaddexp(0, -z) + (cnt - pos) * np.logaddexp(0, z)).sum() / n)

    cur = f(a, b)
    if len(u) > 1:
        for _ in range(200):
            z = a * u + b
            q = 1 / (1 + np.exp(-z))
            r = cnt * q - pos                    # d loss / d z, summed per distinct score
            w = cnt * q * (1 - q)
            g = np.array([r @ u, r.sum()]) / n
            H = np.array([[w @ (u * u), w @ u], [w @ u, w.sum()]]) / n + 1e-12 * np.eye(2)
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-10:
                na, nb = a - t * step[0], b - t * step[1]
                new = f(na, nb)
                if new <= cur - 1e-4 * t * float(g @ step):
                    break
                t *= 0.5
            if t <= 1e-10 or cur - new < 1e-13:
                if t > 1e-10:
                    a, b, cur = na, nb, new
                break
            a, b, cur = na, nb, new
    return a, b, cur


def auc(scores, y):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(scores, float)
    if np.ptp(s) == 0:
        return 0.5
    return float(roc_auc_score(y, s))


def check_points(coef, d, k):
    """Return (points as int array, None) or (None, reason)."""
    try:
        c = np.asarray(coef, dtype=float).ravel()
    except Exception:  # noqa: BLE001
        return None, "coef_ not numeric"
    if c.shape[0] != d:
        return None, f"coef_ has {c.shape[0]} entries for {d} features"
    if not np.all(np.isfinite(c)):
        return None, "non-finite coefficient"
    if np.max(np.abs(c - np.round(c)), initial=0) > 1e-6:
        return None, "non-integer coefficient"
    c = np.round(c).astype(np.int64)
    if np.max(np.abs(c), initial=0) > COEF_BOUND:
        return None, f"coefficient outside [-{COEF_BOUND}, {COEF_BOUND}]"
    if np.count_nonzero(c) > k:
        return None, f"{np.count_nonzero(c)} nonzero coefficients for k={k}"
    return c, None


def score_points(points, Xtr, ytr, Xte, yte):
    s_tr, s_te = Xtr @ points, Xte @ points
    a, b, loss = calibrate(s_tr, ytr)
    z = a * s_te + b
    loss_test = float(np.mean(np.logaddexp(0, -(2 * yte - 1) * z)))
    sign = -1.0 if a < 0 else 1.0
    return {"loss": loss, "a": a, "b": b, "auc_train": auc(sign * s_tr, ytr), "auc_test": auc(sign * s_te, yte),
            "loss_test": loss_test}


def load_best_known():
    out = {}
    if os.path.exists(BEST_KNOWN):
        with open(BEST_KNOWN, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["suite"], r["dataset"], int(r["k"]))] = float(r["loss"])
    return out


# ------------------------------------------------------------------ workers
def _set_single_thread():
    for v in THREAD_VARS:
        os.environ[v] = "1"


def load_make_model(spec):
    """``spec`` is ("file", path) for a solver file defining make_model, or ("baseline", name)."""
    kind, key = spec
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    if kind == "file":
        mod_spec = importlib.util.spec_from_file_location("candidate_solver", key)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules["candidate_solver"] = mod
        mod_spec.loader.exec_module(mod)
        return mod.make_model
    from baselines import make_baseline
    return make_baseline(key)


def _worker(conn, spec, suite):
    _set_single_thread()
    try:
        make_model = load_make_model(spec)
        rng = np.random.default_rng(0)
        Xw = (rng.random((300, 12)) < 0.4).astype(float)
        yw = (Xw[:, 0] + Xw[:, 1] + rng.random(300) > 1.2).astype(np.int64)
        try:
            make_model(3, 5.0).fit(Xw, yw)
        except Exception:  # noqa: BLE001 - a failing warm-up must not stop the worker
            pass
        conn.send(("ready", None))
    except Exception:  # noqa: BLE001
        conn.send(("dead", traceback.format_exc()))
        return
    while True:
        task = conn.recv()
        if task is None:
            return
        name, k, time_limit = task
        Xtr, ytr, _, _, _ = load_problem(name, suite)
        t0 = time.perf_counter()
        try:
            model = make_model(k, time_limit)
            model.fit(Xtr, ytr)
            sec = time.perf_counter() - t0
            if getattr(model, "no_model_", False):  # the solver stopped without a model (e.g. a solver licence cap)
                conn.send(("no_model", (sec, None, str(getattr(model, "stop_reason_", "") or ""))))
                continue
            coef = np.asarray(model.coef_, dtype=float).ravel().tolist()
            conn.send(("ok", (sec, coef, str(getattr(model, "stop_reason_", "") or ""))))
        except Exception:  # noqa: BLE001 - a crashing solver must not abort the suite
            err = [ln.strip() for ln in traceback.format_exc().strip().splitlines() if ln.strip() and set(ln.strip()) != {"^"}]
            conn.send(("crash", (time.perf_counter() - t0, None, " | ".join(err[-3:])[-300:])))


def run_problems(spec, problems, suite="visible", time_limit=TIME_LIMIT, jobs=14, verbose=True):
    """Fit ``spec`` on ``problems`` [(dataset, k)] in parallel. Returns raw results
    {(dataset, k): (status, seconds, coef or None, note)}."""
    _set_single_thread()
    ctx = mp.get_context("spawn")
    hard = time_limit * HARD_LIMIT_FACTOR
    sizes = {}
    for name, k in problems:
        if name not in sizes:
            z = np.load(os.path.join(DATA_DIR, suite, f"{name}.npz"))
            sizes[name] = z["Xtr"].shape[0] * z["Xtr"].shape[1]
    todo = sorted(problems, key=lambda p: -sizes[p[0]] * p[1])  # largest first
    results, workers = {}, []

    def spawn():
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker, args=(child, spec, suite), daemon=True)
        proc.start()
        workers.append({"proc": proc, "conn": parent, "task": None, "t0": 0.0, "ready": False})

    for _ in range(min(jobs, len(todo))):
        spawn()
    while todo or any(w["task"] for w in workers):
        for w in list(workers):
            if w["conn"].poll():
                try:
                    kind, payload = w["conn"].recv()
                except EOFError:
                    kind, payload = "died", None
                if kind == "ready":
                    w["ready"] = True
                elif kind == "dead":
                    raise RuntimeError(f"solver failed to load:\n{payload}")
                elif kind in ("ok", "crash", "no_model"):
                    sec, coef, note = payload
                    results[w["task"]] = (kind, sec, coef, note)
                    if verbose:
                        _print_raw(w["task"], results[w["task"]])
                    w["task"] = None
                elif kind == "died":
                    if w["task"]:
                        results[w["task"]] = ("crash", time.perf_counter() - w["t0"], None, "worker died")
                    w["proc"].kill(); workers.remove(w); spawn()
                    continue
            if w["task"] and time.perf_counter() - w["t0"] > hard:
                results[w["task"]] = ("killed", hard, None, f"killed at {hard:g}s")
                if verbose:
                    _print_raw(w["task"], results[w["task"]])
                w["proc"].kill(); workers.remove(w); spawn()
                continue
            if w["task"] is None and w["ready"] and todo:
                w["task"] = todo.pop(0)
                w["t0"] = time.perf_counter()
                w["conn"].send((w["task"][0], w["task"][1], time_limit))
            if w["task"] is None and not todo and w["ready"]:
                w["conn"].send(None); workers.remove(w)
        time.sleep(0.01)
    for w in workers:
        w["proc"].kill()
    return results


def _print_raw(task, res):
    kind, sec, coef, note = res
    nnz = int(np.count_nonzero(np.round(coef))) if coef is not None else 0
    print(f"  {task[0]:14s} k={task[1]:<3d} {kind:7s} t={sec:7.2f}s nnz={nnz} {note[-120:]}",
          flush=True)


def evaluate_solver(spec, model_name, suite="visible", datasets=None, ks=None, time_limit=TIME_LIMIT, jobs=14,
                    integer=True, verbose=True):
    """Fit and score; returns the summary dict with per-problem ``rows``."""
    datasets = datasets or suite_datasets(suite)
    ks = ks or K_VALUES
    problems = [(d, k) for d in datasets for k in ks]
    raw = run_problems(spec, problems, suite=suite, time_limit=time_limit, jobs=jobs, verbose=verbose)
    best = load_best_known()
    rows = []
    cache = {}
    for name, k in problems:
        if name not in cache:
            cache[name] = load_problem(name, suite)
        Xtr, ytr, Xte, yte, feats = cache[name]
        kind, sec, coef, note = raw[(name, k)]
        status = kind
        points = None
        if kind == "ok":
            if integer:
                points, reason = check_points(coef, Xtr.shape[1], k)
            else:  # a real-valued reference model: only sparsity is checked
                c = np.asarray(coef, float).ravel()
                points, reason = (c, None) if np.count_nonzero(c) <= k else (None, "too many nonzeros")
            if points is None:
                status, note = "invalid", reason
            elif sec > time_limit:
                status = "over_limit"
        if points is None:
            points = np.zeros(Xtr.shape[1], dtype=np.int64)
        sc = score_points(points, Xtr, ytr, Xte, yte)
        bk = best.get((suite, name, k), float("nan"))
        nz = np.flatnonzero(points)
        rows.append({"model": model_name, "suite": suite, "dataset": name, "k": k, "n_train": Xtr.shape[0],
                     "d": Xtr.shape[1], "status": status, "seconds": round(float(sec), 4),
                     "nnz": len(nz), "loss": sc["loss"], "best_known": bk, "regret": sc["loss"] - bk,
                     "auc_train": sc["auc_train"], "auc_test": sc["auc_test"], "loss_test": sc["loss_test"],
                     "a": sc["a"], "b": sc["b"],
                     "points": json.dumps({feats[j]: (int(points[j]) if integer else round(float(points[j]), 4))
                                           for j in nz}),
                     "note": note if status != "ok" else ""})
    return summarize(rows)


def summarize(rows):
    times = [max(float(r["seconds"]), 1e-3) for r in rows]
    regrets = [r["regret"] for r in rows if not math.isnan(r["regret"])]
    return {"rows": rows, "n_problems": len(rows),
            "n_invalid": sum(r["status"] in ("invalid", "crash") for r in rows),
            "n_killed": sum(r["status"] in ("killed", "no_model") for r in rows),
            "n_over_limit": sum(r["status"] == "over_limit" for r in rows),
            "geo_mean_time": float(math.exp(np.mean(np.log(times)))),
            "mean_loss": float(np.mean([r["loss"] for r in rows])),
            "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
            "mean_test_auc": float(np.mean([r["auc_test"] for r in rows]))}


def print_summary(model_name, s, time_limit=TIME_LIMIT):
    print("---")
    print(f"model:          {model_name}")
    print(f"mean_regret:    {s['mean_regret']:.5f}  (training log loss minus the best known, mean over "
          f"{s['n_problems']} problems; lower is better)")
    print(f"geo_mean_time:  {s['geo_mean_time']:.3f}s  (geometric mean of fit seconds; lower is better)")
    print(f"mean_test_auc:  {s['mean_test_auc']:.4f}  (test AUC of the score, mean over problems)")
    print(f"mean_loss:      {s['mean_loss']:.5f}")
    print(f"n_invalid:      {s['n_invalid']}  (points not integer / out of bounds / more than k nonzero, or a "
          f"crash; must be 0)")
    print(f"n_killed:       {s['n_killed']}  (no model: still running at {HARD_LIMIT_FACTOR * time_limit:g}s, or the "
          f"solver stopped without one; scored as the base rate)")
    print(f"n_over_limit:   {s['n_over_limit']}  (returned after the {time_limit:g}s limit)")
    bad = [r for r in s["rows"] if r["status"] in ("invalid", "crash")]
    for r in bad[:10]:
        print(f"  {r['status']}: {r['dataset']} k={r['k']}: {r['note'][:200]}")


# ------------------------------------------------------------------ results
def _upsert(path, cols, rows, key):
    existing = []
    new_keys = {key(r) for r in rows}
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if key(r) not in new_keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
        w.writerows(rows)


def record(model_name, description, s, commit="", status="", results_dir=RESULTS_DIR, integer=True,
           prefix=""):
    """Write the per-problem rows and the leaderboard row (``prefix`` names a non-default suite file)."""
    os.makedirs(results_dir, exist_ok=True)
    fmt = lambda v: "" if isinstance(v, float) and math.isnan(v) else v  # noqa: E731
    _upsert(os.path.join(results_dir, f"{prefix}problem_results.csv"), PROBLEM_CSV_COLS,
            [{k: fmt(v) for k, v in r.items()} for r in s["rows"]], key=lambda r: r["model"])
    _upsert(os.path.join(results_dir, f"{prefix}overall_results.csv"), OVERALL_CSV_COLS, [{
        "commit": commit, "mean_regret": f"{s['mean_regret']:.5f}", "geo_mean_time": f"{s['geo_mean_time']:.3f}",
        "mean_test_auc": f"{s['mean_test_auc']:.4f}", "mean_loss": f"{s['mean_loss']:.5f}",
        "n_invalid": s["n_invalid"], "n_killed": s["n_killed"], "n_over_limit": s["n_over_limit"],
        "integer": "true" if integer else "false", "status": status, "model_name": model_name,
        "description": description}], key=lambda r: r["model_name"])
    print(f"results saved -> {results_dir}")
