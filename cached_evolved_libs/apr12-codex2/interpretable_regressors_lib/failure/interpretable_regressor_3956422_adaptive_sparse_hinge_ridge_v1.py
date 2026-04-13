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


class AdaptiveSparseHingeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Linear ridge backbone plus a tiny set of per-feature hinge corrections.

    - Keeps all linear terms for predictive stability.
    - Adds only a few validation-selected hinge terms to capture thresholds.
    - Exposes a compact explicit equation in raw feature space.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_hinges=3,
        min_hinge_strength_ratio=0.12,
        coef_eps=1e-8,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.max_hinges = max_hinges
        self.min_hinge_strength_ratio = min_hinge_strength_ratio
        self.coef_eps = coef_eps
        self.random_state = random_state

    def _make_split(self, n):
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = int(round(float(self.val_frac) * n))
        n_val = min(max(1, n_val), max(1, n - 1))
        return perm[n_val:], perm[:n_val]

    @staticmethod
    def _std_params(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return mu, sigma

    @staticmethod
    def _ridge_fit(D, y, alpha):
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

    def _best_alpha(self, D_tr, y_tr, D_val, y_val):
        best = None
        for alpha in self.alpha_grid:
            b0, coef = self._ridge_fit(D_tr, y_tr, alpha)
            mse = float(np.mean((y_val - (b0 + D_val @ coef)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best

    @staticmethod
    def _design_matrix(Z):
        return np.column_stack([Z, np.maximum(0.0, Z)])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        mu_tr, scale_tr = self._std_params(X_tr)
        Ztr = (X_tr - mu_tr) / scale_tr
        Zval = (X_val - mu_tr) / scale_tr
        Dtr_full = self._design_matrix(Ztr)
        Dval_full = self._design_matrix(Zval)

        _, alpha_full = self._best_alpha(Dtr_full, y_tr, Dval_full, y_val)
        _, coef_full = self._ridge_fit(Dtr_full, y_tr, alpha_full)
        hinge_coef = coef_full[p:]

        hinge_scale = np.std(np.maximum(0.0, Ztr), axis=0)
        hinge_strength = np.abs(hinge_coef) * hinge_scale
        order = np.argsort(hinge_strength)[::-1]
        max_strength = float(np.max(hinge_strength)) if p > 0 else 0.0
        min_strength = float(self.min_hinge_strength_ratio) * max_strength
        selected_hinges = []
        for j in order:
            j = int(j)
            if len(selected_hinges) >= int(self.max_hinges):
                break
            if hinge_strength[j] <= min_strength or hinge_strength[j] <= 1e-12:
                continue
            selected_hinges.append(j)
        self.hinge_features_ = sorted(selected_hinges)

        active_cols = list(range(p)) + [p + j for j in self.hinge_features_]
        Dtr = Dtr_full[:, active_cols]
        Dval = Dval_full[:, active_cols]
        _, alpha_reduced = self._best_alpha(Dtr, y_tr, Dval, y_val)

        self.x_mean_, self.x_scale_ = self._std_params(X)
        Z = (X - self.x_mean_) / self.x_scale_
        D_full = self._design_matrix(Z)[:, active_cols]
        b0_std, coef_reduced = self._ridge_fit(D_full, y, alpha_reduced)

        coef_linear_z = np.asarray(coef_reduced[:p], dtype=float)
        coef_hinge_z = np.zeros(p, dtype=float)
        for k, j in enumerate(self.hinge_features_):
            coef_hinge_z[int(j)] = float(coef_reduced[p + k])

        self.linear_coef_ = coef_linear_z / self.x_scale_
        self.hinge_coef_ = coef_hinge_z / self.x_scale_
        self.hinge_thresholds_ = self.x_mean_.copy()

        self.intercept_ = float(b0_std - np.sum(coef_linear_z * self.x_mean_ / self.x_scale_))

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        self.hinge_coef_[np.abs(self.hinge_coef_) < float(self.coef_eps)] = 0.0
        self.hinge_features_ = [j for j in self.hinge_features_ if abs(float(self.hinge_coef_[j])) > 0.0]

        fi = np.abs(self.linear_coef_) + np.abs(self.hinge_coef_)
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def _predict_no_check(self, X):
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ self.linear_coef_
        for j in self.hinge_features_:
            j = int(j)
            yhat = yhat + float(self.hinge_coef_[j]) * np.maximum(0.0, X[:, j] - float(self.hinge_thresholds_[j]))
        return yhat

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_", "hinge_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        return self._predict_no_check(X)

    def __str__(self):
        check_is_fitted(self, ["linear_coef_", "hinge_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        terms = [f"{float(self.intercept_):+.6f}"]
        nz_linear = np.where(np.abs(self.linear_coef_) > 0)[0]
        for j in nz_linear:
            terms.append(f"({float(self.linear_coef_[j]):+.6f})*x{int(j)}")
        for j in self.hinge_features_:
            j = int(j)
            terms.append(
                f"({float(self.hinge_coef_[j]):+.6f})*max(0, x{j} - {float(self.hinge_thresholds_[j]):+.6f})"
            )

        lines = [
            "Adaptive Sparse Hinge-Ridge Regressor",
            "Prediction formula:",
            "  y = " + " + ".join(terms),
            "Only listed terms are active; all omitted features have zero contribution.",
            "",
            "Feature contributions (|linear| + |hinge|):",
        ]
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{int(j)}: {float(self.feature_importance_[j]):.4f}")
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
AdaptiveSparseHingeRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveSparseHingeRidge_v1"
model_description = "Full linear ridge backbone plus validation-selected sparse per-feature hinge corrections at feature means, with compact explicit equation output"
model_defs = [(model_shorthand_name, AdaptiveSparseHingeRidgeRegressor())]


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
