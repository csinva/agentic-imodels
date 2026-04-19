"""Interpretable regressor autoresearch script.

Usage: uv run interpretable_regressor.py
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this class freely)
# ---------------------------------------------------------------------------


class HingeLinearRegressor(BaseEstimator, RegressorMixin):
    """Augment features with hinges at per-feature median, Lasso-select, OLS-refit on support.
    Reports as piecewise-linear formula y = intercept + sum a_j*x_j + b_j*max(0, x_j - m_j)."""

    def __init__(self, max_features=10, cv=3):
        self.max_features = max_features
        self.cv = cv

    def _basis(self, X):
        Xh = np.maximum(0.0, X - self.medians_)
        return np.hstack([X, Xh])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.d_ = d
        self.medians_ = np.median(X, axis=0)
        B = self._basis(X)
        scaler = StandardScaler().fit(B)
        Bs = scaler.transform(B)
        lasso = LassoCV(cv=min(self.cv, max(2, n // 20)), n_alphas=25, max_iter=5000).fit(Bs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_features]
        if not kept:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(B[:, self.support_], y)
        self.coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=float)
        B = self._basis(X)
        return B[:, self.support_] @ self.coef_ + self.intercept_

    def _basis_name(self, idx):
        if idx < self.d_:
            return f"x{idx}"
        j = idx - self.d_
        return f"max(0, x{j} - {self.medians_[j]:+.3f})"

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*{self._basis_name(j)}" for c, j in zip(self.coef_, self.support_))
        lines = [
            "HingeLinear: y = intercept + sum_j c_j * basis_j  (linear + ReLU hinges at median)",
            f"  y = {self.intercept_:+.4f} + {terms}",
            "",
            "Active basis terms (each is either x_j or max(0, x_j - median_j)):",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for c, j in zip(self.coef_, self.support_):
            lines.append(f"  {self._basis_name(j)}: {c:.4f}")
        # List inactive features (no active basis for them)
        active_feats = {(j if j < self.d_ else j - self.d_) for j in self.support_}
        inactive = [f"x{i}" for i in range(self.d_) if i not in active_feats]
        if inactive:
            lines.append(f"Features with no active term: {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeLinear_v1"
model_description = "Linear + ReLU hinges at median, Lasso-select top-10, OLS refit"
model_defs = [(model_shorthand_name, HingeLinearRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(existing_interp + new_interp)

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
                              "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""
    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)
    recompute_all_mean_ranks(RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
