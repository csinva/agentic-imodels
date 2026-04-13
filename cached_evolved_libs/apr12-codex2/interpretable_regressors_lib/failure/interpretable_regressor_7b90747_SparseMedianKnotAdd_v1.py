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


class SparseMedianKnotAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Additive linear + one-median-hinge per feature with ridge shrinkage and pruning."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0, 100.0),
        val_frac=0.2,
        max_active_features=12,
        feature_rel_threshold=0.04,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.max_active_features = max_active_features
        self.feature_rel_threshold = feature_rel_threshold
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _fit_ridge_with_intercept(X, y, alpha):
        n, p = X.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        Z = np.column_stack([np.ones(n, dtype=float), X])
        gram = Z.T @ Z
        pen = np.eye(p + 1, dtype=float) * float(alpha)
        pen[0, 0] = 0.0
        rhs = Z.T @ y
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _train_val_split(self, n):
        if n < 25:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(1, int(float(self.val_frac) * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    def _select_alpha(self, X_tr, y_tr, X_va, y_va):
        best = None
        for alpha in self.alpha_grid:
            b0, w = self._fit_ridge_with_intercept(X_tr, y_tr, float(alpha))
            mse = float(np.mean((y_va - (b0 + X_va @ w)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, w)
        return best

    def _make_design(self, X):
        X = np.asarray(X, dtype=float)
        hinge = np.maximum(0.0, X - self.medians_[None, :])
        return np.column_stack([X, hinge])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape

        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)
        self.medians_ = np.median(X, axis=0).astype(float)

        Phi = self._make_design(X)
        tr_idx, va_idx = self._train_val_split(n)
        Phi_tr, Phi_va = Phi[tr_idx], Phi[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        _, self.alpha_, _, _ = self._select_alpha(Phi_tr, y_tr, Phi_va, y_va)
        b0_full, w_full = self._fit_ridge_with_intercept(Phi, y, self.alpha_)

        w_lin_full = w_full[:p]
        w_hinge_full = w_full[p:]
        feature_strength = np.abs(w_lin_full) + np.abs(w_hinge_full)
        max_strength = float(np.max(feature_strength)) if p > 0 else 0.0
        keep = np.where(feature_strength >= float(self.feature_rel_threshold) * max(max_strength, 1e-12))[0]
        if len(keep) > int(self.max_active_features):
            order = np.argsort(feature_strength[keep])[::-1]
            keep = keep[order[: int(self.max_active_features)]]
        keep = np.asarray(sorted(set(int(j) for j in keep)), dtype=int)
        if len(keep) == 0 and p > 0:
            keep = np.array([int(np.argmax(feature_strength))], dtype=int)

        cols = np.concatenate([keep, keep + p])
        if len(cols) > 0:
            b0, w_red = self._fit_ridge_with_intercept(Phi[:, cols], y, self.alpha_)
            self.linear_coef_ = np.zeros(p, dtype=float)
            self.hinge_coef_ = np.zeros(p, dtype=float)
            self.linear_coef_[keep] = w_red[: len(keep)]
            self.hinge_coef_[keep] = w_red[len(keep):]
            self.intercept_ = float(b0)
        else:
            self.linear_coef_ = np.zeros(p, dtype=float)
            self.hinge_coef_ = np.zeros(p, dtype=float)
            self.intercept_ = float(b0_full)

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        self.hinge_coef_[np.abs(self.hinge_coef_) < float(self.coef_eps)] = 0.0
        self.active_features_ = np.where((np.abs(self.linear_coef_) + np.abs(self.hinge_coef_)) > 0)[0]

        imp = np.abs(self.linear_coef_) + np.abs(self.hinge_coef_)
        total = float(np.sum(imp))
        self.feature_importance_ = imp / total if total > 0 else imp
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_coef_", "medians_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ self.linear_coef_
        yhat += np.maximum(0.0, X - self.medians_[None, :]) @ self.hinge_coef_
        return yhat

    @staticmethod
    def _fmt_signed(v):
        return f"+ {v:.6f}" if v >= 0 else f"- {abs(v):.6f}"

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "linear_coef_", "hinge_coef_", "medians_", "feature_importance_", "n_features_in_"],
        )
        lines = [
            "Sparse Median-Knot Additive Regressor",
            f"alpha: {self.alpha_:.4g}",
            "",
            "equation:",
        ]
        eq = f"  y = {self.intercept_:.6f}"
        for j in self.active_features_:
            a = float(self.linear_coef_[j])
            h = float(self.hinge_coef_[j])
            m = float(self.medians_[j])
            if a != 0.0:
                eq += f" {self._fmt_signed(a)}*x{int(j)}"
            if h != 0.0:
                eq += f" {self._fmt_signed(h)}*max(0, x{int(j)} - {m:.6f})"
        lines.append(eq)

        lines.append("")
        lines.append("active_features:")
        if len(self.active_features_) == 0:
            lines.append("  none")
        else:
            lines.append("  " + ", ".join(f"x{int(j)}" for j in self.active_features_))

        lines.append("")
        lines.append("per_feature_effect_form:")
        if len(self.active_features_) == 0:
            lines.append("  y = intercept only")
        else:
            for j in self.active_features_:
                a = float(self.linear_coef_[j])
                h = float(self.hinge_coef_[j])
                m = float(self.medians_[j])
                if h == 0.0:
                    lines.append(f"  x{int(j)}: contribution = ({a:+.4f})*x{int(j)}")
                else:
                    slope_lo = a
                    slope_hi = a + h
                    lines.append(
                        f"  x{int(j)}: knot at {m:.4f}; slope {slope_lo:+.4f} below knot, {slope_hi:+.4f} above knot"
                    )

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
SparseMedianKnotAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseMedianKnotAdd_v1"
model_description = "Adaptive additive ridge with one median hinge per retained feature (linear + max(0, xj-median)) and explicit compact equation output"
model_defs = [(model_shorthand_name, SparseMedianKnotAdditiveRegressor())]


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
