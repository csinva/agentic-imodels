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


class RidgeWithStepShiftRegressor(BaseEstimator, RegressorMixin):
    """
    Raw-feature ridge equation with one optional threshold-shift term.

    Model:
      y = b0 + sum_j w_j * x_j + g * I[x_k > t]
    where (k, t) and ridge alpha are selected on a validation split.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        threshold_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        max_threshold_features=6,
        min_rel_improvement=0.005,
        coef_eps=1e-8,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.threshold_quantiles = threshold_quantiles
        self.max_threshold_features = max_threshold_features
        self.min_rel_improvement = min_rel_improvement
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha):
        n, p = D.shape
        Z = np.column_stack([np.ones(n, dtype=float), D])
        gram = Z.T @ Z
        rhs = Z.T @ y
        reg = np.eye(p + 1, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + reg, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + reg) @ rhs
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

    @staticmethod
    def _corr_screen(X, y):
        yc = y - float(np.mean(y))
        y_norm = float(np.linalg.norm(yc)) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j] - float(np.mean(X[:, j]))
            scores[j] = abs(float(np.dot(xj, yc)) / ((float(np.linalg.norm(xj)) + 1e-12) * y_norm))
        return np.argsort(scores)[::-1]

    def _select_alpha(self, D_tr, y_tr, D_va, y_va):
        best = None
        for a in self.alpha_grid:
            b0, w = self._ridge_fit(D_tr, y_tr, float(a))
            mse = float(np.mean((y_va - (b0 + D_va @ w)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(a))
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr, va = self._train_val_split(n)
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        base_mse, base_alpha = self._select_alpha(X_tr, y_tr, X_va, y_va)
        best = (base_mse, base_alpha, None)

        screened = self._corr_screen(X_tr, y_tr)[: min(int(self.max_threshold_features), p)]
        for j in screened:
            xj_tr = X_tr[:, int(j)]
            for q in self.threshold_quantiles:
                thr = float(np.quantile(xj_tr, float(q)))
                step_tr = (xj_tr > thr).astype(float).reshape(-1, 1)
                step_va = (X_va[:, int(j)] > thr).astype(float).reshape(-1, 1)
                D_tr = np.column_stack([X_tr, step_tr])
                D_va = np.column_stack([X_va, step_va])
                mse, alpha = self._select_alpha(D_tr, y_tr, D_va, y_va)
                if mse < best[0]:
                    best = (mse, alpha, (int(j), thr))

        rel_improvement = (base_mse - best[0]) / (base_mse + 1e-12)
        if best[2] is not None and rel_improvement >= float(self.min_rel_improvement):
            self.step_term_ = best[2]
            self.alpha_ = float(best[1])
            step_col = (X[:, self.step_term_[0]] > self.step_term_[1]).astype(float).reshape(-1, 1)
            D_full = np.column_stack([X, step_col])
            self.intercept_, coef = self._ridge_fit(D_full, y, self.alpha_)
            self.linear_coef_ = np.asarray(coef[:p], dtype=float)
            self.step_coef_ = float(coef[p])
        else:
            self.step_term_ = None
            self.alpha_ = float(base_alpha)
            self.intercept_, coef = self._ridge_fit(X, y, self.alpha_)
            self.linear_coef_ = np.asarray(coef, dtype=float)
            self.step_coef_ = 0.0

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        if abs(self.step_coef_) < float(self.coef_eps):
            self.step_coef_ = 0.0
            self.step_term_ = None

        fi = np.abs(self.linear_coef_)
        if self.step_term_ is not None:
            fi[int(self.step_term_[0])] += abs(float(self.step_coef_))
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_", "intercept_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ self.linear_coef_
        if self.step_term_ is not None and self.step_coef_ != 0.0:
            j, thr = self.step_term_
            yhat = yhat + float(self.step_coef_) * (X[:, int(j)] > float(thr)).astype(float)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["linear_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        lines = ["Ridge With One Step Shift", f"chosen_ridge_alpha: {self.alpha_:.4g}", "equation:"]
        terms = [f"{float(self.intercept_):+.6f}"]
        for j, c in enumerate(self.linear_coef_):
            if c != 0.0:
                terms.append(f"({float(c):+.6f})*x{int(j)}")
        if self.step_term_ is not None and self.step_coef_ != 0.0:
            j, thr = self.step_term_
            terms.append(f"({float(self.step_coef_):+.6f})*I[x{int(j)} > {float(thr):.6f}]")
        lines.append("  y = " + " + ".join(terms))

        lines.append("")
        lines.append("top_features:")
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            val = float(self.feature_importance_[j])
            if val <= 0:
                continue
            lines.append(f"  x{int(j)}: {val:.4f}")
            shown += 1
            if shown >= 12:
                break
        if shown == 0:
            lines.append("  all zero")

        zero_like = [f"x{j}" for j, c in enumerate(self.linear_coef_) if c == 0.0]
        if zero_like:
            lines.append("")
            lines.append("zero_or_tiny_linear_features:")
            lines.append("  " + ", ".join(zero_like))

        if self.step_term_ is not None and self.step_coef_ != 0.0:
            j, thr = self.step_term_
            lines.append("")
            lines.append(f"step_term: add {self.step_coef_:+.6f} when x{int(j)} > {float(thr):.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
RidgeWithStepShiftRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RidgeStepShift_v1"
model_description = "Dense raw-feature ridge equation with validation-selected regularization and one optional binary threshold shift term I[x_j > t] for explicit nonlinearity"
model_defs = [(model_shorthand_name, RidgeWithStepShiftRegressor())]


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
