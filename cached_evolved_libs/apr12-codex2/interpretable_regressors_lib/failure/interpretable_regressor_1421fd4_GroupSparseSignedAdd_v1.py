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


class GroupSparseSignedAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Forward-selected sparse additive model with per-feature signed-magnitude terms:
        y = b + sum_j (w_j * x_j + v_j * |x_j|)
    Selection is by validation gain over feature-groups, then coefficients are
    snapped to a small step for easier mental simulation.
    """

    def __init__(
        self,
        max_groups=4,
        val_frac=0.2,
        alpha=1e-3,
        min_gain=1e-3,
        snap_step=0.05,
        min_coef=1e-4,
        random_state=42,
    ):
        self.max_groups = max_groups
        self.val_frac = val_frac
        self.alpha = alpha
        self.min_gain = min_gain
        self.snap_step = snap_step
        self.min_coef = min_coef
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha):
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(D.shape[1], dtype=float) * float(alpha)
        pen[0, 0] = 0.0
        try:
            return np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram + pen) @ rhs

    @staticmethod
    def _snap(v, step):
        step = float(step)
        if step <= 0:
            return float(v)
        return float(np.round(v / step) * step)

    def _design_with_groups(self, X, groups):
        cols = [np.ones(X.shape[0], dtype=float)]
        for j in groups:
            xj = X[:, j]
            cols.append(xj)
            cols.append(np.abs(xj))
        return np.column_stack(cols)

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

        # Feature scaling for stable group selection.
        x_mean = X_tr.mean(axis=0)
        x_std = X_tr.std(axis=0) + 1e-8
        X_tr_s = (X_tr - x_mean) / x_std
        X_val_s = (X_val - x_mean) / x_std
        X_full_s = (X - x_mean) / x_std

        selected = []
        D0_tr = np.ones((X_tr_s.shape[0], 1), dtype=float)
        D0_val = np.ones((X_val_s.shape[0], 1), dtype=float)
        beta0 = self._ridge_fit(D0_tr, y_tr, alpha=float(self.alpha))
        best_val_mse = float(np.mean((y_val - D0_val @ beta0) ** 2))

        for _ in range(int(max(0, self.max_groups))):
            best_feat = None
            best_mse = best_val_mse
            for j in range(p):
                if j in selected:
                    continue
                cand = selected + [j]
                D_tr = self._design_with_groups(X_tr_s, cand)
                D_val = self._design_with_groups(X_val_s, cand)
                beta = self._ridge_fit(D_tr, y_tr, alpha=float(self.alpha))
                mse = float(np.mean((y_val - D_val @ beta) ** 2))
                if mse + 1e-12 < best_mse:
                    best_mse = mse
                    best_feat = j
            if best_feat is None or (best_val_mse - best_mse) < float(self.min_gain):
                break
            selected.append(int(best_feat))
            best_val_mse = best_mse

        D_full = self._design_with_groups(X_full_s, selected)
        beta_full = np.asarray(self._ridge_fit(D_full, y, alpha=float(self.alpha)), dtype=float)

        # Convert standardized coefficients to raw-feature equation.
        intercept = float(beta_full[0])
        linear = np.zeros(p, dtype=float)
        abs_coef = np.zeros(p, dtype=float)
        for i, j in enumerate(selected):
            b_lin = float(beta_full[1 + 2 * i])
            b_abs = float(beta_full[1 + 2 * i + 1])
            sj = float(x_std[j])
            mj = float(x_mean[j])
            linear[j] = b_lin / sj
            abs_coef[j] = b_abs / sj
            intercept -= (b_lin * mj) / sj

        # Snap parameters for readability and deterministic arithmetic in tests.
        self.intercept_ = self._snap(intercept, self.snap_step)
        self.linear_coef_ = np.array([self._snap(v, self.snap_step) for v in linear], dtype=float)
        self.abs_coef_ = np.array([self._snap(v, self.snap_step) for v in abs_coef], dtype=float)
        self.selected_features_ = [int(j) for j in selected]
        self.feature_importance_ = np.abs(self.linear_coef_) + np.abs(self.abs_coef_)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "abs_coef_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.linear_coef_ + np.abs(X) @ self.abs_coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "abs_coef_", "feature_importance_", "n_features_in_"])
        lines = ["Group-Sparse Signed Additive Regressor", "equation:"]

        terms = [f"{self.intercept_:+.4f}"]
        for j in range(self.n_features_in_):
            c_lin = float(self.linear_coef_[j])
            c_abs = float(self.abs_coef_[j])
            if abs(c_lin) >= float(self.min_coef):
                terms.append(f"({c_lin:+.4f})*x{j}")
            if abs(c_abs) >= float(self.min_coef):
                terms.append(f"({c_abs:+.4f})*abs(x{j})")
        lines.append("  y = " + " + ".join(terms))

        lines.append("")
        lines.append("active_features:")
        if self.selected_features_:
            lines.append("  " + ", ".join(f"x{j}" for j in self.selected_features_))
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")

        near_zero = [
            f"x{j}" for j in range(self.n_features_in_)
            if abs(self.linear_coef_[j]) < float(self.min_coef) and abs(self.abs_coef_[j]) < float(self.min_coef)
        ]
        if near_zero:
            lines.append("")
            lines.append("near_zero_features: " + ", ".join(near_zero))

        lines.append("")
        lines.append("manual_prediction: plug feature values into y = intercept + sum(linear terms) + sum(abs terms).")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
GroupSparseSignedAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "GroupSparseSignedAdd_v1"
model_description = "Forward-selected sparse additive equation using per-feature linear and abs terms with coefficient snapping for easier manual simulation"
model_defs = [(model_shorthand_name, GroupSparseSignedAdditiveRegressor())]


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
