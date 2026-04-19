"""Interpretable regressor autoresearch script. Usage: uv run interpretable_regressor.py"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance

# =========== CLASS ============


class ForwardOLSRegressor(BaseEstimator, RegressorMixin):
    """Forward stepwise selection (greedy) up to K features, then OLS refit on selected support."""

    def __init__(self, max_features=6):
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.d_ = d
        selected = []
        remaining = list(range(d))
        resid = y - y.mean()
        k = min(self.max_features, d)
        for _ in range(k):
            best = None; best_improve = -np.inf; best_idx = None
            for i in remaining:
                feats = selected + [i]
                ols = LinearRegression().fit(X[:, feats], y)
                r = y - ols.predict(X[:, feats])
                rss = float((r * r).sum())
                improve = -rss
                if improve > best_improve:
                    best_improve = improve; best_idx = i
            if best_idx is None: break
            selected.append(best_idx); remaining.remove(best_idx)
        self.support_ = sorted(selected) if selected else [0]
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [f"ForwardOLS (greedy-selected {len(self.support_)} features, OLS refit):",
                 f"  y = {self.intercept_:+.4f} + {terms}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for c, j in zip(self.coef_, self.support_): lines.append(f"  x{j}: {c:.4f}")
        if excluded: lines.append(f"Features excluded (not selected): {', '.join(excluded)}")
        return "\n".join(lines)


class _OldThresholdOLS(BaseEstimator, RegressorMixin):
    """Fit full OLS, drop coefs with |coef_std| < threshold * max (threshold=0.05), refit OLS."""

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        full = LinearRegression().fit(Xs, y)
        std_coef = np.abs(full.coef_)
        if std_coef.max() > 0:
            keep_mask = std_coef >= self.threshold * std_coef.max()
        else:
            keep_mask = np.zeros(d, bool); keep_mask[0] = True
        self.support_ = sorted(np.where(keep_mask)[0].tolist())
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [
            f"ThresholdOLS (full OLS then drop features with |std-coef| < {self.threshold}*max):",
            f"  y = {self.intercept_:+.4f} + {terms}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for c, j in zip(self.coef_, self.support_):
            lines.append(f"  x{j}: {c:.4f}")
        if excluded:
            lines.append(f"Features excluded (negligible effect): {', '.join(excluded)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ForwardOLSRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ForwardOLS6_v1"
model_description = "Greedy forward stepwise: pick 6 features minimizing RSS, OLS refit"
model_defs = [(model_shorthand_name, ForwardOLSRegressor())]


# ============ EVAL ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o')
    args = parser.parse_args()
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results); total = len(interp_results)
    dataset_rmses = evaluate_all_regressors(model_defs)
    try: git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    def _suite(t):
        if t.startswith("insight_"): return "insight"
        if t.startswith("hard_"):    return "hard"
        return "standard"
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name: existing_interp.append(row)
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
                   "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
                   "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(existing_interp + new_interp)

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name: existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
                              "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf: by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1): r["rank"] = rank_idx
        for r in rows:
            if r["rmse"] in ("", None): r["rank"] = ""
    with open(perf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=perf_fields); w.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]: w.writerow(row)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None): all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else: all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))
    upsert_overall_results([{"commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed/total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description}], RESULTS_DIR)
    recompute_all_mean_ranks(RESULTS_DIR)
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
