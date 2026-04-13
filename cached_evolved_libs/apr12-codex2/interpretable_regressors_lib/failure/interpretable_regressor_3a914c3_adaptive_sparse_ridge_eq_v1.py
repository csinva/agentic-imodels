"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class AdaptiveSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Validation-calibrated sparse ridge in raw features.

    Steps:
    1. Fit ridge on standardized features with holdout-selected alpha.
    2. Convert to raw-feature equation.
    3. Select a sparse active set by validation (top-k and relative-threshold candidates).
    4. Refit an exact linear equation on active raw features.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0),
        topk_grid=(2, 4, 6, 8, 12, 16),
        rel_thresh_grid=(0.0, 0.01, 0.02, 0.05, 0.1),
        sparsity_penalty=0.015,
        coef_eps=1e-8,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.topk_grid = topk_grid
        self.rel_thresh_grid = rel_thresh_grid
        self.sparsity_penalty = sparsity_penalty
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _ridge_fit_no_intercept(X, y, alpha):
        p = X.shape[1]
        gram = X.T @ X + float(alpha) * np.eye(p, dtype=float)
        rhs = X.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs
        return np.asarray(coef, dtype=float)

    @staticmethod
    def _ols_fit(X, y):
        n = X.shape[0]
        Z = np.column_stack([np.ones(n, dtype=float), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(Z) @ y
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _train_val_split(self, n):
        if n < 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    def _fit_ridge_raw_equation(self, X, y, alpha):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std > 1e-12, x_std, 1.0)
        y_mean = float(np.mean(y))
        Xs = (X - x_mean) / x_std
        yc = y - y_mean
        w_std = self._ridge_fit_no_intercept(Xs, yc, alpha)
        raw_coef = w_std / x_std
        intercept = y_mean - float(np.dot(raw_coef, x_mean))
        return intercept, np.asarray(raw_coef, dtype=float)

    def _select_alpha(self, X_tr, y_tr, X_va, y_va):
        best = None
        for alpha in self.alpha_grid:
            b0, w = self._fit_ridge_raw_equation(X_tr, y_tr, float(alpha))
            pred = b0 + X_va @ w
            mse = float(np.mean((y_va - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, w)
        return best

    def _sparse_candidates(self, coef):
        p = coef.shape[0]
        order = np.argsort(np.abs(coef))[::-1]
        cand = {tuple(range(p))}

        for k in self.topk_grid:
            kk = int(min(max(1, k), p))
            idx = tuple(sorted(int(i) for i in order[:kk]))
            cand.add(idx)

        max_abs = float(np.max(np.abs(coef))) if p > 0 else 0.0
        for rel in self.rel_thresh_grid:
            if max_abs <= 0:
                idx = tuple()
            else:
                keep = np.where(np.abs(coef) >= float(rel) * max_abs)[0]
                idx = tuple(sorted(int(i) for i in keep))
            cand.add(idx)
        return [np.array(c, dtype=int) for c in cand]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, va_idx = self._train_val_split(n)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        _, self.alpha_, _, dense_coef = self._select_alpha(X_tr, y_tr, X_va, y_va)
        dense_b0, dense_coef = self._fit_ridge_raw_equation(X, y, self.alpha_)

        best = None
        for active in self._sparse_candidates(dense_coef):
            if active.size == 0:
                b0 = float(np.mean(y_tr))
                w = np.zeros(0, dtype=float)
                pred_va = np.full(len(y_va), b0, dtype=float)
            else:
                b0, w = self._ols_fit(X_tr[:, active], y_tr)
                pred_va = b0 + X_va[:, active] @ w
            mse = float(np.mean((y_va - pred_va) ** 2))
            complexity = float(active.size) / max(1.0, float(p))
            objective = mse * (1.0 + float(self.sparsity_penalty) * complexity)
            if best is None or objective < best[0]:
                best = (objective, mse, active, b0, w)

        _, _, active, _, _ = best
        full_coef = np.zeros(p, dtype=float)
        if active.size == 0:
            self.intercept_ = float(np.mean(y))
        else:
            b0, w = self._ols_fit(X[:, active], y)
            self.intercept_ = float(b0)
            full_coef[active] = w

        full_coef[np.abs(full_coef) < float(self.coef_eps)] = 0.0
        self.coef_ = full_coef
        self.active_features_ = np.where(self.coef_ != 0.0)[0]
        self.dense_intercept_ = float(dense_b0)
        self.dense_coef_ = np.asarray(dense_coef, dtype=float)

        fi = np.abs(self.coef_)
        fi_total = float(np.sum(fi))
        self.feature_importance_ = fi / fi_total if fi_total > 0 else fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        return float(self.intercept_) + X @ self.coef_

    @staticmethod
    def _fmt_signed(value):
        if value >= 0:
            return f"+ {value:.6f}"
        return f"- {abs(value):.6f}"

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "active_features_", "feature_importance_", "n_features_in_"])

        lines = [
            "Adaptive Sparse Ridge Regressor",
            f"selected_ridge_alpha: {self.alpha_:.4g}",
            f"active_features: {len(self.active_features_)}/{self.n_features_in_}",
            "equation:",
        ]

        eq = f"  y = {self.intercept_:.6f}"
        for j in self.active_features_:
            eq += f" {self._fmt_signed(float(self.coef_[j]))}*x{int(j)}"
        if len(self.active_features_) == 0:
            eq += "  (constant model)"
        lines.append(eq)

        lines.append("")
        lines.append("nonzero_coefficients:")
        if len(self.active_features_) == 0:
            lines.append("  none")
        else:
            for j in self.active_features_:
                lines.append(f"  x{int(j)}: {self.coef_[j]:+.6f}")

        zero_feats = [f"x{j}" for j, c in enumerate(self.coef_) if c == 0.0]
        if zero_feats:
            lines.append("")
            lines.append("zero_or_irrelevant_features:")
            lines.append("  " + ", ".join(zero_feats))

        lines.append("")
        lines.append("normalized_feature_importance:")
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            score = float(self.feature_importance_[j])
            if score <= 0:
                continue
            lines.append(f"  x{int(j)}: {score:.4f}")
            shown += 1
            if shown >= 12:
                break
        if shown == 0:
            lines.append("  all zero")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
AdaptiveSparseRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveSparseRidgeEq_v1"
model_description = "Holdout-selected ridge converted to raw-feature equation, then validation-guided sparsification (top-k/threshold candidates) with exact refit on active features for concise arithmetic simulation"
model_defs = [(model_shorthand_name, AdaptiveSparseRidgeRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
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

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
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
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in ALL_TESTS})
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in HARD_TESTS})
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in INSIGHT_TESTS})
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else "") +
          f"  [std {std_passed}/{len(ALL_TESTS)}  hard {hard_passed}/{len(HARD_TESTS)}  insight {insight_passed}/{len(INSIGHT_TESTS)}]")
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
