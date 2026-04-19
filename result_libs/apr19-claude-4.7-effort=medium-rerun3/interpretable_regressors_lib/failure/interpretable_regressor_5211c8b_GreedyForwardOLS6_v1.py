"""Interpretable regressor autoresearch script. Usage: uv run interpretable_regressor.py"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge, Lars
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance

# =========== CLASS ============


class GreedyForwardOLSRegressor(BaseEstimator, RegressorMixin):
    """Greedy forward selection up to max_feat; OLS refit."""

    def __init__(self, max_feat=6):
        self.max_feat = max_feat

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        remaining = list(range(d)); selected = []
        best_sse = float("inf")
        for _ in range(min(self.max_feat, d)):
            best_j, best_new = -1, best_sse
            for j in remaining:
                cols = selected + [j]
                Xs = np.column_stack([np.ones(n), X[:, cols]])
                b, *_ = np.linalg.lstsq(Xs, y, rcond=None)
                sse = np.sum((y - Xs @ b) ** 2)
                if sse < best_new - 1e-10: best_new = sse; best_j = j
            if best_j < 0: break
            selected.append(best_j); remaining.remove(best_j); best_sse = best_new
        if not selected: selected = [0]
        self.support_ = sorted(selected)
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excl = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [f"Greedy Forward OLS ({len(self.support_)} features chosen by greedy forward selection):",
                 f"  y = {self.intercept_:+.4f} + {terms}", "",
                 "Coefficients:", f"  intercept: {self.intercept_:.4f}"]
        for c, j in zip(self.coef_, self.support_): lines.append(f"  x{j}: {c:.4f}")
        if excl: lines.append(f"Features excluded: {', '.join(excl)}")
        return "\n".join(lines)


class _UnusedWinsor(BaseEstimator, RegressorMixin):
    """OLS after winsorizing each feature to [5th, 95th] percentile (robust to outliers)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.lo_ = np.percentile(X, 5, axis=0); self.hi_ = np.percentile(X, 95, axis=0)
        Xc = np.clip(X, self.lo_, self.hi_)
        ols = LinearRegression().fit(Xc, y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        Xc = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        return Xc @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*clip(x{j})" for j, c in enumerate(self.coef_))
        lines = ["Winsorized OLS: each feature clipped to its training [5th, 95th] percentile, then plain linear regression.",
                 f"  y = {self.intercept_:+.4f} + {terms}", "",
                 "Coefficients:", f"  intercept: {self.intercept_:.4f}"]
        for j, c in enumerate(self.coef_):
            lines.append(f"  x{j}: {c:.4f}  (clipped to [{self.lo_[j]:.4g}, {self.hi_[j]:.4g}])")
        return "\n".join(lines)


class _UnusedOLSR(BaseEstimator, RegressorMixin):
    """Plain OLS; __str__ additionally ranks features by |standardized coefficient|."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.sd_ = X.std(axis=0); self.sd_[self.sd_ == 0] = 1.0
        ols = LinearRegression().fit(X, y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for j, c in enumerate(self.coef_))
        std_imp = np.abs(self.coef_ * self.sd_)
        order = np.argsort(-std_imp)
        lines = ["OLS Linear Regression (ordinary least squares; dense linear formula):",
                 f"  y = {self.intercept_:+.4f} + {terms}", "",
                 "Coefficients (original feature scale):",
                 f"  intercept: {self.intercept_:.4f}"]
        for j, c in enumerate(self.coef_): lines.append(f"  x{j}: {c:.4f}")
        lines += ["", "Feature importance (ranked by |coef * sd(x)|):"]
        for rk, j in enumerate(order, 1):
            lines.append(f"  {rk}. x{j}: |std-coef|={std_imp[j]:.4f}  (coef={self.coef_[j]:+.4f})")
        return "\n".join(lines)


class _UnusedMinMax(BaseEstimator, RegressorMixin):
    """OLS after rescaling each feature to [0,1] by its training min/max; print coefs as effect per full-range change (then back-transform for predict)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.mn_ = X.min(axis=0); self.mx_ = X.max(axis=0)
        self.rng_ = (self.mx_ - self.mn_); self.rng_[self.rng_ == 0] = 1.0
        Z = (X - self.mn_) / self.rng_
        ols = LinearRegression().fit(Z, y)
        self.coef_scaled_ = ols.coef_
        self.intercept_scaled_ = float(ols.intercept_)
        # back-transform to original space
        self.coef_ = self.coef_scaled_ / self.rng_
        self.intercept_ = float(self.intercept_scaled_ - np.dot(self.coef_scaled_, self.mn_ / self.rng_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for j, c in enumerate(self.coef_))
        lines = ["MinMaxOLS: linear regression fit on min-max-scaled features; coefs below shown in ORIGINAL feature scale:",
                 f"  y = {self.intercept_:+.4f} + {terms}", "",
                 "Coefficients (original feature scale):",
                 f"  intercept: {self.intercept_:.4f}"]
        for j, c in enumerate(self.coef_): lines.append(f"  x{j}: {c:.4f}")
        return "\n".join(lines)


class _UnusedHuber(BaseEstimator, RegressorMixin):
    """Robust HuberRegressor on standardized X; coefs translated to original scale."""

    def fit(self, X, y):
        from sklearn.linear_model import HuberRegressor
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.mu_ = X.mean(axis=0); self.sd_ = X.std(axis=0); self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        h = HuberRegressor(max_iter=200, alpha=1e-3).fit(Xs, y)
        self.coef_ = h.coef_ / self.sd_
        self.intercept_ = float(h.intercept_ - np.dot(h.coef_, self.mu_ / self.sd_))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for j, c in enumerate(self.coef_))
        lines = ["Huber Robust Linear Regression (outlier-robust loss; dense linear formula):",
                 f"  y = {self.intercept_:+.4f} + {terms}", "", "Coefficients:",
                 f"  intercept: {self.intercept_:.4f}"]
        for j, c in enumerate(self.coef_): lines.append(f"  x{j}: {c:.4f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GreedyForwardOLSRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "GreedyForwardOLS6_v1"
model_description = "Greedy forward feature selection up to 6, then OLS refit"
model_defs = [(model_shorthand_name, GreedyForwardOLSRegressor())]


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
