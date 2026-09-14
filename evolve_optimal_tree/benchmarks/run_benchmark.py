"""Benchmark pygosdt against the reference C++ GOSDT binary.

For every (dataset, regularization) pair both implementations are run on the
same CSV.  The objective of each returned tree is recomputed independently from
the JSON tree and the raw data so that the two results are compared on equal
footing.

Usage (from ``evolve_optimal_tree``)::

    uv run python benchmarks/run_benchmark.py [--datasets a,b] [--lams 0.1,0.05]
        [--time-limit 600] [--out benchmarks/results]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REF = ROOT / "GeneralizedOptimalSparseDecisionTreesReference"
DATA = REF / "experiments" / "datasets"
BINARY = REF / "build" / "gosdt"

sys.path.insert(0, str(ROOT))
from pygosdt import GOSDTClassifier  # noqa: E402

# name -> (csv path, preprocessing)
DATASETS = {
    "monk_1": DATA / "monk_1" / "data.csv",
    "monk_2": DATA / "monk_2" / "data.csv",
    "monk_3": DATA / "monk_3" / "data.csv",
    "chudi": DATA / "chudi" / "data.csv",
    "iris": DATA / "iris" / "data.csv",
    "tic-tac-toe": DATA / "tic-tac-toe" / "data.csv",
    "car_evaluation": DATA / "car_evaluation" / "data.csv",
    "sine_1k": DATA / "sine" / "thousand.csv",
    "gaussian_1k": DATA / "gaussian" / "thousand.csv",
    "fico_1k": DATA / "fico" / "data.csv",
    "coupon_bar": DATA / "coupon" / "bar-7.csv",
    "compas_binned": DATA / "compas" / "binned.csv",
    "fico_binary": DATA / "fico" / "fico-binary.csv",
    "sine_10k": DATA / "sine" / "ten_thousand.csv",
    "compas_processed": DATA / "compas" / "processed.csv",
}

DEFAULT_ORDER = list(DATASETS)


def load_dataset(name: str, workdir: Path) -> Path:
    """Return a CSV path usable by both implementations (missing values -> 0)."""
    src = DATASETS[name]
    frame = pd.read_csv(src)
    if frame.isna().any().any():
        frame = frame.fillna(0)
    out = workdir / f"{name}.csv"
    frame.to_csv(out, index=False)
    return out


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
    else:
        if isinstance(ref, str):
            mask = (col.astype(str) == ref).to_numpy()
        else:
            mask = (pd.to_numeric(col, errors="coerce") == ref).to_numpy()
    e1, l1 = evaluate_tree(node["true"], frame[mask])
    e2, l2 = evaluate_tree(node["false"], frame[~mask])
    return e1 + e2, l1 + l2


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) * 1024
    except (ValueError, OSError):
        return 0


def run_reference(csv: Path, lam: float, time_limit: int, workdir: Path, workers: int = 1,
                  memory_limit_bytes: int = 0) -> dict:
    model_path = workdir / f"ref_{csv.stem}_{lam}.json"
    cfg = {
        "regularization": lam, "verbose": True, "worker_limit": workers,
        "model_limit": 1, "time_limit": int(time_limit), "model": str(model_path),
    }
    cfg_path = workdir / f"ref_{csv.stem}_{lam}.cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    killed = ""
    with open(csv, "rb") as fh, open(workdir / f"ref_{csv.stem}_{lam}.out", "w+") as out_fh:
        t0 = time.perf_counter()
        proc = subprocess.Popen([str(BINARY), str(cfg_path)], stdin=fh, stdout=out_fh,
                                stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            time.sleep(0.5)
            if time.perf_counter() - t0 > time_limit * 2 + 120:
                proc.kill()
                killed = "timeout"
            elif memory_limit_bytes and _rss_bytes(proc.pid) > memory_limit_bytes:
                proc.kill()
                killed = "memory"
        wall = time.perf_counter() - t0
        out_fh.seek(0)
        out = out_fh.read()
    res = {"ref_wall": wall, "ref_status": proc.returncode, "ref_killed": killed}
    m = re.search(r"Training Duration: ([0-9.eE+-]+) seconds", out)
    res["ref_time"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"Size of Graph: (\d+)", out)
    res["ref_size"] = int(m.group(1)) if m else -1
    m = re.search(r"Number of Iterations: (\d+)", out)
    res["ref_iterations"] = int(m.group(1)) if m else -1
    m = re.search(r"Objective Boundary: \[([0-9.eE+-]+), ([0-9.eE+-]+)\]", out)
    if m:
        res["ref_lb"], res["ref_ub"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Optimality Gap: ([0-9.eE+-]+)", out)
    res["ref_gap"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"Binary Dataset Dimension: (\d+) x (\d+)", out)
    res["ref_binary_features"] = int(m.group(2)) if m else -1
    res["ref_tree"] = None
    if model_path.exists():
        models = json.loads(model_path.read_text())
        if models:
            res["ref_tree"] = models[0]
    return res


def run_python(csv: Path, lam: float, time_limit: float, engine: str = "auto",
               memory_limit_bytes: int = 0) -> dict:
    frame = pd.read_csv(csv)
    X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = GOSDTClassifier(regularization=lam, time_limit=time_limit, engine=engine,
                                memory_limit=memory_limit_bytes).fit(X, y)
    wall = time.perf_counter() - t0
    res = {
        "py_wall": wall, "py_time": model.time_, "py_encode_time": model.encoding_time_,
        "py_size": model.size_, "py_iterations": model.iterations_,
        "py_optimal": model.optimal_, "py_stop_reason": model.stop_reason_,
        "py_binary_features": model.n_binary_features_, "py_lb": model.lowerbound_,
        "py_ub": model.upperbound_, "py_tree": model.tree_,
    }
    del model
    gc.collect()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--lams", default="0.1,0.05,0.02,0.01,0.005")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--engine", default="auto", help="pygosdt engine: auto, numba or python")
    ap.add_argument("--workers", type=int, default=1, help="reference worker_limit")
    ap.add_argument("--memory-limit-gb", type=float, default=6.0,
                    help="stop either implementation when its resident memory exceeds this")
    ap.add_argument("--resume", action="store_true", help="skip (dataset, lam) pairs already in the output CSV")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workdir = out / "work"
    workdir.mkdir(exist_ok=True)
    datasets = [d for d in args.datasets.split(",") if d]
    lams = [float(v) for v in args.lams.split(",") if v]
    csv_out = out / f"benchmark{args.tag}.csv"
    rows = []
    done = set()
    if args.resume and csv_out.exists():
        previous = pd.read_csv(csv_out)
        rows = previous.to_dict("records")
        done = {(r["dataset"], float(r["lam"])) for r in rows}
    mem_bytes = int(args.memory_limit_gb * (1 << 30))

    for name in datasets:
        csv = load_dataset(name, workdir)
        frame = pd.read_csv(csv)
        n, p = frame.shape[0], frame.shape[1] - 1
        for lam in lams:
            if (name, lam) in done:
                continue
            row = {"dataset": name, "n": n, "p": p, "lam": lam}
            if not args.skip_reference:
                r = run_reference(csv, lam, int(args.time_limit), workdir, workers=args.workers,
                                  memory_limit_bytes=mem_bytes)
                if r["ref_tree"] is not None:
                    e, l = evaluate_tree(r["ref_tree"], frame)
                    row.update(ref_errors=e, ref_leaves=l, ref_objective=e / n + lam * l)
                row.update({k: v for k, v in r.items() if k != "ref_tree"})
            if not args.skip_python:
                r = run_python(csv, lam, args.time_limit, engine=args.engine, memory_limit_bytes=mem_bytes)
                e, l = evaluate_tree(r["py_tree"], frame)
                row.update(py_errors=e, py_leaves=l, py_objective=e / n + lam * l)
                row.update({k: v for k, v in r.items() if k != "py_tree"})
            if "ref_objective" in row and "py_objective" in row:
                row["objective_diff"] = row["py_objective"] - row["ref_objective"]
                row["speedup_cpp_over_py"] = row["py_time"] / row["ref_time"] if row["ref_time"] > 0 else float("nan")
            rows.append(row)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
