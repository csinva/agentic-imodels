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


class SparseLinearResidualStumpsRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse ridge linear backbone + a few one-feature threshold rules.

    The model is intentionally compact and arithmetic:
      y = intercept + sum_j w_j * x_j + sum_k c_k * I(x_{f_k} > t_k)

    Fit:
      1) choose ridge alpha by validation and score feature weights,
      2) keep only strongest linear features,
      3) greedily add a few residual threshold rules (single feature each),
      4) jointly refit linear + rule coefficients with ridge.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        max_linear_features=8,
        max_stumps=3,
        threshold_quantiles=(20, 35, 50, 65, 80),
        min_stump_gain=1e-4,
        min_coef=1e-4,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.max_linear_features = max_linear_features
        self.max_stumps = max_stumps
        self.threshold_quantiles = threshold_quantiles
        self.min_stump_gain = min_stump_gain
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

    def _make_split(self, n):
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
        return tr_idx, val_idx

    def _select_alpha(self, D_tr, y_tr, D_val, y_val):
        best = None
        for alpha in self.alpha_grid:
            b0, b = self._ridge_solve(D_tr, y_tr, alpha)
            pred = b0 + D_val @ b
            mse = float(np.mean((y_val - pred) ** 2))
            if (best is None) or (mse < best[0]):
                best = (mse, float(alpha), b0, b)
        return best

    def _candidate_thresholds(self, x):
        vals = [np.percentile(x, q) for q in self.threshold_quantiles]
        uniq = []
        for v in vals:
            fv = float(v)
            if np.isfinite(fv) and (len(uniq) == 0 or abs(fv - uniq[-1]) > 1e-10):
                uniq.append(fv)
        return uniq

    def _best_residual_stump(self, X, residual, blocked_features):
        n, p = X.shape
        base_mse = float(np.mean(residual ** 2))
        min_leaf = max(10, int(0.05 * n))
        best = None

        for j in range(p):
            if j in blocked_features:
                continue
            x = X[:, j]
            for t in self._candidate_thresholds(x):
                right = x > t
                n_right = int(np.sum(right))
                n_left = n - n_right
                if n_right < min_leaf or n_left < min_leaf:
                    continue
                left_mean = float(np.mean(residual[~right]))
                right_mean = float(np.mean(residual[right]))
                piece = np.where(right, right_mean, left_mean)
                mse = float(np.mean((residual - piece) ** 2))
                gain = base_mse - mse
                if (best is None) or (gain > best["gain"]):
                    best = {
                        "feature": int(j),
                        "threshold": float(t),
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "gain": float(gain),
                    }
        return best

    def _design_matrix(self, X):
        cols = []
        for j in self.selected_linear_features_:
            cols.append(X[:, j])
        for stump in self.stumps_:
            cols.append((X[:, stump["feature"]] > stump["threshold"]).astype(float))
        if not cols:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack(cols).astype(float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)

        # Stage 1: full linear ridge for feature screening.
        D_tr_full = X[tr_idx]
        D_val_full = X[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]
        _, alpha_linear, _, coef_full = self._select_alpha(D_tr_full, y_tr, D_val_full, y_val)

        abs_coef = np.abs(coef_full)
        if abs_coef.size == 0:
            selected_linear = []
        else:
            order = list(np.argsort(abs_coef)[::-1])
            max_keep = int(max(1, min(self.max_linear_features, p)))
            top = abs_coef[order[0]]
            rel_cut = max(float(self.min_coef), 0.12 * float(top))
            selected_linear = [j for j in order if abs_coef[j] >= rel_cut][:max_keep]
            if len(selected_linear) == 0:
                selected_linear = order[:max_keep]
        self.selected_linear_features_ = sorted(int(j) for j in selected_linear)

        # Stage 2: sparse linear refit.
        D_tr_lin = X[tr_idx][:, self.selected_linear_features_]
        D_val_lin = X[val_idx][:, self.selected_linear_features_]
        _, alpha_sparse, b0_lin, w_lin = self._select_alpha(D_tr_lin, y_tr, D_val_lin, y_val)
        self.alpha_linear_ = float(alpha_sparse)

        if len(self.selected_linear_features_) > 0:
            y_hat_lin = float(b0_lin) + X[:, self.selected_linear_features_] @ w_lin
        else:
            y_hat_lin = np.full(n, float(b0_lin), dtype=float)
        residual = y - y_hat_lin

        # Stage 3: greedy residual threshold rules.
        stumps = []
        blocked = set()
        for _ in range(int(max(0, self.max_stumps))):
            best = self._best_residual_stump(X, residual, blocked)
            if best is None or best["gain"] < float(self.min_stump_gain):
                break
            stumps.append(best)
            blocked.add(best["feature"])
            piece = np.where(
                X[:, best["feature"]] > best["threshold"],
                best["right_mean"],
                best["left_mean"],
            )
            residual = residual - piece
        self.stumps_ = stumps

        # Stage 4: joint ridge refit on sparse linear terms + stump indicators.
        D_all = self._design_matrix(X)
        D_tr = self._design_matrix(X[tr_idx])
        D_val = self._design_matrix(X[val_idx])
        _, self.alpha_, self.intercept_, coef = self._select_alpha(D_tr, y_tr, D_val, y_val)
        self.coef_ = np.asarray(coef, dtype=float)
        self.coef_[np.abs(self.coef_) < float(self.min_coef)] = 0.0

        if D_all.shape[1] > 0:
            self.intercept_ = float(np.mean(y - D_all @ self.coef_))
        else:
            self.intercept_ = float(np.mean(y))

        n_lin = len(self.selected_linear_features_)
        self.linear_coefs_ = self.coef_[:n_lin].copy()
        self.stump_coefs_ = self.coef_[n_lin:].copy()

        self.feature_importance_ = np.zeros(p, dtype=float)
        for idx, j in enumerate(self.selected_linear_features_):
            self.feature_importance_[j] += float(abs(self.linear_coefs_[idx]))
        for idx, stump in enumerate(self.stumps_):
            self.feature_importance_[stump["feature"]] += float(abs(self.stump_coefs_[idx]))
        total_imp = float(np.sum(self.feature_importance_))
        if total_imp > 0:
            self.feature_importance_ /= total_imp
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["selected_linear_features_", "stumps_", "intercept_", "coef_", "n_features_in_"],
        )
        X = np.asarray(X, dtype=float)
        D = self._design_matrix(X)
        return float(self.intercept_) + D @ self.coef_

    def __str__(self):
        check_is_fitted(
            self,
            [
                "selected_linear_features_",
                "stumps_",
                "intercept_",
                "linear_coefs_",
                "stump_coefs_",
                "feature_importance_",
            ],
        )
        lines = ["Sparse Linear + Residual Stumps Regressor", "equation:"]
        lines.append(f"  y = {float(self.intercept_):+.4f} + linear_terms + threshold_rule_terms")
        lines.append("")

        lines.append("linear_terms:")
        shown_linear = 0
        for j, c in sorted(
            zip(self.selected_linear_features_, self.linear_coefs_),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            if abs(c) <= 0:
                continue
            lines.append(f"  ({float(c):+.4f}) * x{int(j)}")
            shown_linear += 1
        if shown_linear == 0:
            lines.append("  none")
        lines.append("  all unlisted features have linear coefficient 0")
        lines.append("")

        lines.append("threshold_rule_terms:")
        if len(self.stumps_) == 0:
            lines.append("  none")
        else:
            for i, stump in enumerate(self.stumps_):
                c = float(self.stump_coefs_[i])
                j = int(stump["feature"])
                t = float(stump["threshold"])
                lines.append(f"  rule_{i+1}: add ({c:+.4f}) if x{j} > {t:+.4f}, else add (+0.0000)")
        lines.append("")

        lines.append("feature_importance_order:")
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")
        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) start from intercept")
        lines.append("  2) add each listed linear term c*xj")
        lines.append("  3) for each rule, add its coefficient only when the condition is true")
        return "\n".join(lines)

# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseLinearResidualStumpsRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseLinStumps_v1"
model_description = "Sparse ridge linear backbone with greedy residual single-feature threshold rules and explicit arithmetic if-then equation"
model_defs = [(model_shorthand_name, SparseLinearResidualStumpsRegressor())]


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
