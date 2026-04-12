"""
Repair script: re-evaluate only the (model, dataset) pairs that have nan/empty
RMSE in performance_results.csv, then update all result files.
"""
import csv
import gc
import importlib.util
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evolve", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evolve"))

from performance_eval import subsample_dataset, MAX_SAMPLES, MAX_FEATURES, MIN_SAMPLES, MIN_FEATURES, OVERALL_CSV_COLS
from visualize import plot_interp_vs_performance

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "interpretable_regressors_lib" / "success"
NEW_RESULTS_DIR = SCRIPT_DIR / "new_results"

# Baseline imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV as SkLassoCV, LinearRegression, RidgeCV as SkRidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM
from imodels import FIGSRegressor, HSTreeRegressorCV, RuleFitRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from tabpfn import TabPFNRegressor

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")

# ---------------------------------------------------------------------------
# Dataset loader (same as evaluate_new_generalization.py)
# ---------------------------------------------------------------------------
import openml
import pandas as pd

def _load_openml_dataset_by_id(dataset_id):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder

    path = os.path.join(_CACHE_DIR, f"openml_id_{dataset_id}.parquet")
    if not os.path.exists(path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        openml.config.cache_directory = os.path.join(_CACHE_DIR, "openml")
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        df["__target__"] = y
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)

    y_raw = df["__target__"].values
    X_raw = df.drop(columns=["__target__"])
    y = pd.to_numeric(pd.Series(y_raw), errors="coerce").values.astype(float)
    valid = ~np.isnan(y)
    y = y[valid]
    X_raw = X_raw[valid]

    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    for col in num_cols:
        median = X_tr[col].median()
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(median)
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(median)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))

    return X_tr.astype(np.float32).values, X_te.astype(np.float32).values, y_tr, y_te


NEW_OPENML_IDS = [44065, 44066, 44068, 44069, 45048, 45041, 45043, 45047, 45045, 45046, 44055, 44056, 44059, 44061, 44062, 44063]
DS_NAME_TO_ID = {f"openml/{did}": did for did in NEW_OPENML_IDS}

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_evolved_models():
    model_map = {}
    for py_file in sorted(LIB_DIR.glob("*.py")):
        mod_name = py_file.stem
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'fit') and hasattr(obj, 'predict'):
                obj.__module__ = mod_name
        name = getattr(mod, "model_shorthand_name", None)
        defs = getattr(mod, "model_defs", None)
        if name and defs:
            model_map[name] = defs[0][1]
    return model_map


BASELINE_MAP = {
    "PyGAM": LinearGAM(n_splines=10),
    "DT_mini": DecisionTreeRegressor(max_leaf_nodes=8, random_state=42),
    "DT_large": DecisionTreeRegressor(max_leaf_nodes=20, random_state=42),
    "OLS": LinearRegression(),
    "LassoCV": SkLassoCV(cv=3),
    "RidgeCV": SkRidgeCV(cv=3),
    "RF": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
    "GBM": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    "MLP": MLPRegressor(random_state=42),
    "FIGS_mini": FIGSRegressor(max_rules=8, random_state=42),
    "FIGS_large": FIGSRegressor(max_rules=20, random_state=42),
    "RuleFit": RuleFitRegressor(max_rules=20, random_state=42),
    "HSTree_mini": HSTreeRegressorCV(max_leaf_nodes=8, random_state=42),
    "HSTree_large": HSTreeRegressorCV(max_leaf_nodes=20, random_state=42),
    "EBM": ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000),
    "TabPFN": TabPFNRegressor(device="cpu", random_state=42),
}


if __name__ == "__main__":
    t0 = time.time()

    # 1. Read existing performance_results.csv and find failures
    perf_csv = str(NEW_RESULTS_DIR / "performance_results.csv")
    rows = []
    failures = []  # list of (ds_name, model_name)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if row["rmse"] in ("", None):
                failures.append((row["dataset"], row["model"]))

    print(f"Found {len(failures)} failed (dataset, model) pairs")
    if not failures:
        print("Nothing to repair!")
        sys.exit(0)

    # 2. Identify unique datasets and models needed
    failed_datasets = sorted(set(ds for ds, _ in failures))
    failed_models = sorted(set(m for _, m in failures))
    print(f"  Datasets: {failed_datasets}")
    print(f"  Models: {failed_models}")

    # 3. Load only the needed models
    print("\nLoading models...")
    evolved = load_evolved_models()
    all_models = {**evolved, **BASELINE_MAP}
    needed_models = {name: all_models[name] for name in failed_models if name in all_models}
    print(f"  Loaded {len(needed_models)} models")

    # 4. Re-evaluate each failed pair ONE AT A TIME with gc between
    fixed = {}
    for ds_name in failed_datasets:
        did = DS_NAME_TO_ID.get(ds_name)
        if did is None:
            print(f"  Skipping unknown dataset: {ds_name}")
            continue
        print(f"\n  Loading dataset {ds_name}...")
        X_tr, X_te, y_tr, y_te = _load_openml_dataset_by_id(did)
        X_tr, X_te, y_tr, y_te = subsample_dataset(X_tr, X_te, y_tr, y_te)
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
        if y_std > 0:
            y_tr_n = (y_tr - y_mean) / y_std
            y_te_n = (y_te - y_mean) / y_std
        else:
            y_tr_n, y_te_n = y_tr, y_te

        models_for_ds = [m for d, m in failures if d == ds_name]
        for model_name in models_for_ds:
            if model_name not in needed_models:
                print(f"    {model_name:<25}: SKIPPED (model not found)")
                continue
            gc.collect()
            try:
                reg = needed_models[model_name]
                m = deepcopy(reg)
                m.fit(X_tr, y_tr_n)
                preds = m.predict(X_te)
                rmse = float(np.sqrt(mean_squared_error(y_te_n, preds)))
                fixed[(ds_name, model_name)] = rmse
                print(f"    {model_name:<25}: {rmse:.4f} (FIXED)")
                del m
            except Exception as e:
                print(f"    {model_name:<25}: ERROR — {e}")
            gc.collect()

    print(f"\nFixed {len(fixed)}/{len(failures)} pairs")

    # 5. Patch performance_results.csv
    for row in rows:
        key = (row["dataset"], row["model"])
        if key in fixed:
            row["rmse"] = f"{fixed[key]:.6f}"
            row["rank"] = ""  # will recompute

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for ds_name, ds_rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in ds_rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        for r in ds_rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "rmse", "rank"])
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results patched -> {perf_csv}")

    # 6. Recompute overall_results.csv with lenient ranking
    dataset_rmses = defaultdict(dict)
    for row in rows:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            dataset_rmses[row["dataset"]][row["model"]] = float("nan")

    # Lenient rank: average over available datasets
    all_model_names = set()
    for d in dataset_rmses.values():
        all_model_names.update(d.keys())
    ranks_per_model = {n: [] for n in all_model_names}
    for ds_name, model_rmses in dataset_rmses.items():
        valid = [(n, v) for n, v in model_rmses.items() if not np.isnan(v)]
        sorted_models = sorted(valid, key=lambda x: x[1])
        rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
        for name in all_model_names:
            if name in model_rmses and not np.isnan(model_rmses[name]):
                ranks_per_model[name].append(rank_map[name])
    avg_rank = {n: float(np.mean(v)) if v else float("nan") for n, v in ranks_per_model.items()}

    # Read interp pass rates from interpretability_results.csv
    interp_csv = str(NEW_RESULTS_DIR / "interpretability_results.csv")
    model_interp = defaultdict(lambda: {"passed": 0, "total": 0})
    with open(interp_csv, newline="") as f:
        for row in csv.DictReader(f):
            model_interp[row["model"]]["total"] += 1
            if row["passed"] == "True":
                model_interp[row["model"]]["passed"] += 1

    # Read existing overall to preserve descriptions and status
    overall_csv = str(NEW_RESULTS_DIR / "overall_results.csv")
    old_overall = {}
    with open(overall_csv, newline="") as f:
        for row in csv.DictReader(f):
            old_overall[row["model_name"]] = row

    # Write updated overall
    overall_rows = []
    for name in sorted(all_model_names, key=lambda n: avg_rank.get(n, 999)):
        old = old_overall.get(name, {})
        mi = model_interp[name]
        frac = mi["passed"] / mi["total"] if mi["total"] > 0 else float("nan")
        rank = avg_rank.get(name, float("nan"))
        overall_rows.append({
            "commit": old.get("commit", ""),
            "mean_rank": f"{rank:.2f}" if not np.isnan(rank) else "nan",
            "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
            "status": old.get("status", ""),
            "model_name": name,
            "description": old.get("description", ""),
        })

    with open(overall_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(overall_rows)
    print(f"Overall results updated -> {overall_csv}")

    # 7. Replot
    plot_interp_vs_performance(overall_csv, str(NEW_RESULTS_DIR / "interpretability_vs_performance.png"))

    # 8. Summary
    remaining_nans = sum(1 for r in overall_rows if r["mean_rank"] == "nan")
    print(f"\nRemaining nan ranks: {remaining_nans}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # Print final leaderboard
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for row in overall_rows:
        status = " [baseline]" if row["status"] == "baseline" else ""
        print(f"  {row['model_name']:<25}  rank={row['mean_rank']:>6}  interp={row['frac_interpretability_tests_passed']}{status}")
