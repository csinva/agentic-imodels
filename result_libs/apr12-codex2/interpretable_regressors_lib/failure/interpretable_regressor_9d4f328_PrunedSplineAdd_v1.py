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


class PrunedSplineAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Additive piecewise-linear regressor with feature pruning.

    For each raw feature x_j we use a 3-term spline basis on standardized z_j:
      z_j, relu(z_j - k1_j), relu(z_j - k2_j)
    where k1_j/k2_j are standardized quartile knots.

    Fit procedure:
      1) choose ridge alpha on a validation split,
      2) keep only the most influential features,
      3) refit ridge on all data with selected features.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        max_active_features=7,
        keep_frac_of_top=0.2,
        min_coef=2e-4,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.keep_frac_of_top = keep_frac_of_top
        self.min_coef = min_coef
        self.random_state = random_state

    @staticmethod
    def _ridge_solve(D, y, alpha):
        n, p = D.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        Z = np.column_stack([np.ones(n, dtype=float), D])
        gram = Z.T @ Z
        rhs = Z.T @ y
        pen = np.eye(p + 1, dtype=float) * float(alpha)
        pen[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _standardize(self, X):
        Z = (X - self.x_mean_) / self.x_scale_
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def _basis_for_features(self, Z, feats):
        cols = []
        for j in feats:
            z = Z[:, j]
            k1, k2 = self.knot_z_[j]
            cols.append(z)
            cols.append(np.maximum(0.0, z - k1))
            cols.append(np.maximum(0.0, z - k2))
        if not cols:
            return np.zeros((Z.shape[0], 0), dtype=float)
        return np.column_stack(cols).astype(float)

    @staticmethod
    def _feature_scores(coef, feats):
        scores = {}
        for i, j in enumerate(feats):
            block = coef[3 * i: 3 * i + 3]
            scores[int(j)] = float(np.sum(np.abs(block)))
        return scores

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0) + 1e-8
        Z = self._standardize(X)

        self.knot_raw_ = np.column_stack([
            np.percentile(X, 25, axis=0),
            np.percentile(X, 75, axis=0),
        ]).astype(float)
        self.knot_z_ = (self.knot_raw_ - self.x_mean_[:, None]) / self.x_scale_[:, None]

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(24, int(float(self.val_frac) * n))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size == 0:
            tr_idx = perm
            val_idx = perm[: max(1, min(24, n))]

        all_feats = list(range(p))
        D_tr = self._basis_for_features(Z[tr_idx], all_feats)
        D_val = self._basis_for_features(Z[val_idx], all_feats)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        best = None
        for alpha in self.alpha_grid:
            b0, b = self._ridge_solve(D_tr, y_tr, alpha)
            pred = b0 + D_val @ b
            mse = float(np.mean((y_val - pred) ** 2))
            if (best is None) or (mse < best[0]):
                best = (mse, float(alpha), b0, b)
        _, self.alpha_, _, coef_full = best

        scores = self._feature_scores(coef_full, all_feats)
        if len(scores) == 0:
            selected = []
        else:
            sorted_feats = sorted(scores, key=lambda j: scores[j], reverse=True)
            top_score = scores[sorted_feats[0]]
            thresh = float(self.keep_frac_of_top) * top_score
            selected = [j for j in sorted_feats if scores[j] >= thresh]
            selected = selected[: int(max(1, self.max_active_features))]
            if len(selected) == 0:
                selected = [sorted_feats[0]]
        self.selected_features_ = sorted(selected)

        D_all = self._basis_for_features(Z, self.selected_features_)
        self.intercept_, self.coef_ = self._ridge_solve(D_all, y, self.alpha_)
        self.coef_[np.abs(self.coef_) < float(self.min_coef)] = 0.0

        # Re-center intercept after coefficient thresholding.
        if D_all.shape[1] > 0:
            self.intercept_ = float(np.mean(y - D_all @ self.coef_))
        else:
            self.intercept_ = float(np.mean(y))

        self.feature_importance_ = np.zeros(p, dtype=float)
        for i, j in enumerate(self.selected_features_):
            block = self.coef_[3 * i: 3 * i + 3]
            self.feature_importance_[j] = float(np.sum(np.abs(block)))
        total_imp = float(np.sum(self.feature_importance_))
        if total_imp > 0:
            self.feature_importance_ /= total_imp
        return self

    def predict(self, X):
        check_is_fitted(self, ["selected_features_", "intercept_", "coef_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        Z = self._standardize(X)
        D = self._basis_for_features(Z, self.selected_features_)
        return float(self.intercept_) + D @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["selected_features_", "intercept_", "coef_", "feature_importance_"])
        lines = ["Pruned Spline Additive Regressor", "equation:"]
        lines.append(f"  y = {float(self.intercept_):+.4f} + sum_j g_j(x_j)")
        lines.append("  where only the listed features have non-zero g_j")
        lines.append("")
        lines.append("active_feature_functions:")
        if len(self.selected_features_) == 0:
            lines.append("  none (constant model)")
        else:
            for i, j in enumerate(self.selected_features_):
                a, b, c = self.coef_[3 * i: 3 * i + 3]
                mu = self.x_mean_[j]
                sd = self.x_scale_[j]
                k1, k2 = self.knot_z_[j]
                lines.append(f"  x{j}: z=(x{j}-{mu:+.3f})/{sd:.3f}")
                lines.append(
                    f"      g_{j}(x{j}) = ({a:+.4f})*z + ({b:+.4f})*max(0, z-{k1:+.3f}) + ({c:+.4f})*max(0, z-{k2:+.3f})"
                )
        lines.append("")
        lines.append("feature_importance_order:")
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")
        lines.append("")
        lines.append("manual_prediction: compute each active g_j from x-values, then add intercept.")
        return "\n".join(lines)
# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
PrunedSplineAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "PrunedSplineAdd_v1"
model_description = "Pruned additive spline ridge: per-feature 3-term piecewise-linear basis with validation-selected shrinkage and top-feature retention"
model_defs = [(model_shorthand_name, PrunedSplineAdditiveRegressor())]


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
