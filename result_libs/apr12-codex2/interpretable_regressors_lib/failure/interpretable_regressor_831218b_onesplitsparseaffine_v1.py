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


class OneSplitSparseAffineRegressor(BaseEstimator, RegressorMixin):
    """
    Global sparse affine model with an optional single feature split:
        if x_j <= t: y = b_L + w_L^T x
        else:        y = b_R + w_R^T x

    The split is selected by validation MSE gain. Each leaf is a ridge fit that
    is hard-thresholded to top-|coef| features for concise arithmetic.
    """

    def __init__(
        self,
        alpha=1e-2,
        val_frac=0.2,
        max_split_features=10,
        n_thresholds=7,
        min_leaf_frac=0.12,
        max_terms_per_leaf=6,
        min_split_gain=5e-3,
        min_coef=1e-4,
        random_state=42,
    ):
        self.alpha = alpha
        self.val_frac = val_frac
        self.max_split_features = max_split_features
        self.n_thresholds = n_thresholds
        self.min_leaf_frac = min_leaf_frac
        self.max_terms_per_leaf = max_terms_per_leaf
        self.min_split_gain = min_split_gain
        self.min_coef = min_coef
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        D = np.column_stack([np.ones(n, dtype=float), X])
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(p + 1, dtype=float) * float(alpha)
        pen[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    @staticmethod
    def _sparsify(coef, max_terms):
        coef = np.asarray(coef, dtype=float).copy()
        k = int(max_terms)
        if k <= 0 or coef.size <= k:
            return coef
        keep = np.argsort(np.abs(coef))[::-1][:k]
        mask = np.zeros(coef.size, dtype=bool)
        mask[keep] = True
        coef[~mask] = 0.0
        return coef

    @staticmethod
    def _predict_linear(X, intercept, coef):
        return float(intercept) + X @ np.asarray(coef, dtype=float)

    def _fit_sparse_linear(self, X, y):
        b, w = self._ridge_fit(X, y, alpha=float(self.alpha))
        w = self._sparsify(w, self.max_terms_per_leaf)
        # Refit intercept after sparsification.
        b = float(np.mean(y - X @ w))
        return b, w

    def _leaf_mse(self, X_tr, y_tr, X_val, y_val):
        b, w = self._fit_sparse_linear(X_tr, y_tr)
        pred = self._predict_linear(X_val, b, w)
        return float(np.mean((y_val - pred) ** 2)), b, w

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(20, int(float(self.val_frac) * n))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size == 0:
            tr_idx = perm
            val_idx = perm[: max(1, min(20, n))]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Baseline sparse global affine model.
        global_mse, global_b, global_w = self._leaf_mse(X_tr, y_tr, X_val, y_val)

        # Candidate split features screened by |corr(x_j, y)|.
        x_center = X_tr - X_tr.mean(axis=0)
        y_center = y_tr - y_tr.mean()
        denom = (np.sqrt(np.sum(x_center ** 2, axis=0)) * np.sqrt(np.sum(y_center ** 2)) + 1e-12)
        corr = np.abs((x_center.T @ y_center) / denom)
        kf = min(int(max(1, self.max_split_features)), p)
        split_features = np.argsort(corr)[::-1][:kf]

        best = {
            "use_split": False,
            "mse": global_mse,
            "feature": -1,
            "threshold": 0.0,
            "left": (global_b, global_w),
            "right": (global_b, global_w),
        }

        min_leaf = max(8, int(float(self.min_leaf_frac) * X_tr.shape[0]))
        quantiles = np.linspace(0.15, 0.85, int(max(2, self.n_thresholds)))

        for j in split_features:
            vals = X_tr[:, j]
            thrs = np.quantile(vals, quantiles)
            thrs = np.unique(np.asarray(thrs, dtype=float))
            for thr in thrs:
                ltr = vals <= thr
                rtr = ~ltr
                if ltr.sum() < min_leaf or rtr.sum() < min_leaf:
                    continue

                lval = X_val[:, j] <= thr
                rval = ~lval
                if lval.sum() == 0 or rval.sum() == 0:
                    continue

                l_mse, lb, lw = self._leaf_mse(X_tr[ltr], y_tr[ltr], X_val[lval], y_val[lval])
                r_mse, rb, rw = self._leaf_mse(X_tr[rtr], y_tr[rtr], X_val[rval], y_val[rval])
                split_mse = float((lval.mean() * l_mse) + (rval.mean() * r_mse))

                if split_mse + 1e-12 < best["mse"]:
                    best = {
                        "use_split": True,
                        "mse": split_mse,
                        "feature": int(j),
                        "threshold": float(thr),
                        "left": (lb, lw),
                        "right": (rb, rw),
                    }

        gain = global_mse - best["mse"]
        if (not best["use_split"]) or gain < float(self.min_split_gain):
            self.use_split_ = False
            self.split_feature_ = -1
            self.split_threshold_ = 0.0
            self.left_intercept_, self.left_coef_ = self._fit_sparse_linear(X, y)
            self.right_intercept_, self.right_coef_ = self.left_intercept_, self.left_coef_.copy()
        else:
            self.use_split_ = True
            self.split_feature_ = int(best["feature"])
            self.split_threshold_ = float(best["threshold"])

            left_mask = X[:, self.split_feature_] <= self.split_threshold_
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                self.use_split_ = False
                self.split_feature_ = -1
                self.split_threshold_ = 0.0
                self.left_intercept_, self.left_coef_ = self._fit_sparse_linear(X, y)
                self.right_intercept_, self.right_coef_ = self.left_intercept_, self.left_coef_.copy()
            else:
                self.left_intercept_, self.left_coef_ = self._fit_sparse_linear(X[left_mask], y[left_mask])
                self.right_intercept_, self.right_coef_ = self._fit_sparse_linear(X[right_mask], y[right_mask])

        imp = np.abs(self.left_coef_) + np.abs(self.right_coef_)
        self.feature_importance_ = imp / (imp.sum() + 1e-12)
        return self

    def predict(self, X):
        check_is_fitted(self, [
            "use_split_", "split_feature_", "split_threshold_",
            "left_intercept_", "left_coef_", "right_intercept_", "right_coef_",
            "n_features_in_",
        ])
        X = np.asarray(X, dtype=float)
        if not self.use_split_:
            return self._predict_linear(X, self.left_intercept_, self.left_coef_)

        out = np.empty(X.shape[0], dtype=float)
        left_mask = X[:, self.split_feature_] <= self.split_threshold_
        out[left_mask] = self._predict_linear(X[left_mask], self.left_intercept_, self.left_coef_)
        out[~left_mask] = self._predict_linear(X[~left_mask], self.right_intercept_, self.right_coef_)
        return out

    def _equation_str(self, intercept, coef):
        terms = [f"{float(intercept):+.4f}"]
        for j, c in enumerate(coef):
            if abs(float(c)) >= float(self.min_coef):
                terms.append(f"({float(c):+.4f})*x{j}")
        return " + ".join(terms)

    def __str__(self):
        check_is_fitted(self, [
            "use_split_", "split_feature_", "split_threshold_",
            "left_intercept_", "left_coef_", "right_intercept_", "right_coef_", "feature_importance_",
        ])

        lines = ["One-Split Sparse Affine Regressor", "equation:"]
        if self.use_split_:
            lines.append(f"  if x{self.split_feature_} <= {self.split_threshold_:.4f}:")
            lines.append(f"    y = {self._equation_str(self.left_intercept_, self.left_coef_)}")
            lines.append("  else:")
            lines.append(f"    y = {self._equation_str(self.right_intercept_, self.right_coef_)}")
        else:
            lines.append(f"  y = {self._equation_str(self.left_intercept_, self.left_coef_)}")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")

        lines.append("")
        if self.use_split_:
            lines.append("manual_prediction: choose branch by split condition, then evaluate that affine equation.")
        else:
            lines.append("manual_prediction: plug feature values into the affine equation.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
OneSplitSparseAffineRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "OneSplitSparseAffine_v1"
model_description = "Validation-selected one-split sparse affine model: global linear fallback plus two sparse linear leaves with explicit branch equations"
model_defs = [(model_shorthand_name, OneSplitSparseAffineRegressor())]


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
