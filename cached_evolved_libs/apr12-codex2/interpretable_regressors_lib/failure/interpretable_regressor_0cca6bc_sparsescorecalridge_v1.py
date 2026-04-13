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


class SparseScoreCalibratedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse ridge base equation with a tiny piecewise-linear score calibration.

    The model is intentionally two-stage but still arithmetic:
      1) score = b + sum_j w_j * x_j  (sparse linear base)
      2) y = c0 + c1*score + c2*max(0, score-k1) + c3*max(0, score-k2)
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        cal_alpha_grid=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
        max_active_features=12,
        min_rel_weight=0.04,
        coef_eps=1e-8,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.cal_alpha_grid = cal_alpha_grid
        self.max_active_features = max_active_features
        self.min_rel_weight = min_rel_weight
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

    def _best_alpha(self, D_tr, y_tr, D_val, y_val, grid):
        best = None
        for alpha in grid:
            b0, coef = self._ridge_fit(D_tr, y_tr, alpha)
            mse = float(np.mean((y_val - (b0 + D_val @ coef)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best

    def _select_active(self, coef_z):
        strength = np.abs(np.asarray(coef_z, dtype=float))
        if strength.size == 0:
            return []
        order = np.argsort(strength)[::-1]
        max_s = float(np.max(strength))
        min_s = float(self.min_rel_weight) * max_s
        active = []
        for j in order:
            j = int(j)
            if len(active) >= int(self.max_active_features):
                break
            if strength[j] < min_s or strength[j] <= 1e-12:
                continue
            active.append(j)
        if not active:
            active = [int(order[0])]
        return sorted(active)

    @staticmethod
    def _cal_design(score, knot1, knot2):
        score = np.asarray(score, dtype=float).reshape(-1)
        return np.column_stack(
            [
                score,
                np.maximum(0.0, score - float(knot1)),
                np.maximum(0.0, score - float(knot2)),
            ]
        )

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

        _, alpha_full = self._best_alpha(Ztr, y_tr, Zval, y_val, self.alpha_grid)
        _, coef_full_z = self._ridge_fit(Ztr, y_tr, alpha_full)
        self.active_features_ = self._select_active(coef_full_z)

        Ztr_a = Ztr[:, self.active_features_]
        Zval_a = Zval[:, self.active_features_]
        _, alpha_base = self._best_alpha(Ztr_a, y_tr, Zval_a, y_val, self.alpha_grid)
        b0_tr, w_tr_z = self._ridge_fit(Ztr_a, y_tr, alpha_base)

        score_tr = b0_tr + Ztr_a @ w_tr_z
        score_val = b0_tr + Zval_a @ w_tr_z
        knot1 = float(np.quantile(score_tr, 0.35))
        knot2 = float(np.quantile(score_tr, 0.70))
        Ctr = self._cal_design(score_tr, knot1, knot2)
        Cval = self._cal_design(score_val, knot1, knot2)
        _, alpha_cal = self._best_alpha(Ctr, y_tr, Cval, y_val, self.cal_alpha_grid)

        self.x_mean_, self.x_scale_ = self._std_params(X)
        Zall = (X - self.x_mean_) / self.x_scale_
        Zall_a = Zall[:, self.active_features_]
        b0_std, w_all_z = self._ridge_fit(Zall_a, y, alpha_base)

        self.linear_coef_ = np.zeros(p, dtype=float)
        self.linear_coef_[self.active_features_] = w_all_z / self.x_scale_[self.active_features_]
        self.base_intercept_ = float(b0_std - np.sum(w_all_z * self.x_mean_[self.active_features_] / self.x_scale_[self.active_features_]))

        score_all = self.base_intercept_ + X @ self.linear_coef_
        self.score_knot1_ = float(np.quantile(score_all, 0.35))
        self.score_knot2_ = float(np.quantile(score_all, 0.70))
        Call = self._cal_design(score_all, self.score_knot1_, self.score_knot2_)
        self.cal_intercept_, self.cal_coef_ = self._ridge_fit(Call, y, alpha_cal)

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        self.cal_coef_[np.abs(self.cal_coef_) < float(self.coef_eps)] = 0.0

        fi = np.abs(self.linear_coef_) * float(np.sum(np.abs(self.cal_coef_)))
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def _predict_no_check(self, X):
        X = np.asarray(X, dtype=float)
        score = float(self.base_intercept_) + X @ self.linear_coef_
        yhat = float(self.cal_intercept_) + self.cal_coef_[0] * score
        yhat = yhat + self.cal_coef_[1] * np.maximum(0.0, score - float(self.score_knot1_))
        yhat = yhat + self.cal_coef_[2] * np.maximum(0.0, score - float(self.score_knot2_))
        return yhat

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "linear_coef_",
                "base_intercept_",
                "cal_intercept_",
                "cal_coef_",
                "score_knot1_",
                "score_knot2_",
                "feature_importance_",
                "n_features_in_",
            ],
        )
        return self._predict_no_check(X)

    def __str__(self):
        check_is_fitted(
            self,
            [
                "linear_coef_",
                "base_intercept_",
                "cal_intercept_",
                "cal_coef_",
                "score_knot1_",
                "score_knot2_",
                "feature_importance_",
                "n_features_in_",
            ],
        )
        active = np.where(np.abs(self.linear_coef_) > 0)[0]
        base_terms = [f"{float(self.base_intercept_):+.6f}"]
        for j in active:
            base_terms.append(f"({float(self.linear_coef_[j]):+.6f})*x{int(j)}")

        lines = [
            "Sparse Score-Calibrated Ridge Regressor",
            "Step 1 (base score):",
            "  score = " + " + ".join(base_terms),
            "Step 2 (piecewise score calibration):",
            "  y = "
            f"{float(self.cal_intercept_):+.6f} + ({float(self.cal_coef_[0]):+.6f})*score"
            f" + ({float(self.cal_coef_[1]):+.6f})*max(0, score - {float(self.score_knot1_):+.6f})"
            f" + ({float(self.cal_coef_[2]):+.6f})*max(0, score - {float(self.score_knot2_):+.6f})",
            "Only listed base features are active; omitted features have zero contribution.",
            "",
            "Feature contributions (normalized):",
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
SparseScoreCalibratedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseScoreCalRidge_v1"
model_description = "Sparse ridge base equation on selected raw features, followed by a compact two-knot piecewise-linear calibration of the scalar base score"
model_defs = [(model_shorthand_name, SparseScoreCalibratedRidgeRegressor())]


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
