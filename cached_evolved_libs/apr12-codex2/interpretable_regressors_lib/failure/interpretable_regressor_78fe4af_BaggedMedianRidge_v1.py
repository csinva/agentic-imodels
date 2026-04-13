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


class BaggedMedianRidgeRegressor(BaseEstimator, RegressorMixin):
    """Bootstrap-aggregated ridge with a single explicit linear equation."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0),
        n_bootstrap=9,
        bootstrap_frac=0.8,
        min_sign_stability=0.6,
        coef_prune_rel=0.01,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.n_bootstrap = n_bootstrap
        self.bootstrap_frac = bootstrap_frac
        self.min_sign_stability = min_sign_stability
        self.coef_prune_rel = coef_prune_rel
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _train_val_split(n, seed, val_frac=0.2):
        if n < 40:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    @staticmethod
    def _fit_ridge_raw(X, y, alpha):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std > 1e-12, x_std, 1.0)
        y_mean = float(np.mean(y))

        Xs = (X - x_mean) / x_std
        yc = y - y_mean

        p = Xs.shape[1]
        gram = Xs.T @ Xs + float(alpha) * np.eye(p, dtype=float)
        rhs = Xs.T @ yc
        try:
            coef_std = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef_std = np.linalg.pinv(gram) @ rhs

        coef = coef_std / x_std
        intercept = y_mean - float(np.dot(coef, x_mean))
        return float(intercept), np.asarray(coef, dtype=float)

    def _select_alpha(self, X_tr, y_tr, X_va, y_va):
        best = None
        for alpha in self.alpha_grid:
            b0, w = self._fit_ridge_raw(X_tr, y_tr, float(alpha))
            pred = b0 + X_va @ w
            mse = float(np.mean((y_va - pred) ** 2))
            coef_scale = float(np.mean(np.abs(w)))
            obj = mse + 1e-3 * coef_scale
            if best is None or obj < best[0]:
                best = (obj, float(alpha))
        return best[1]

    def _prune_coef(self, coef):
        coef = np.asarray(coef, dtype=float).copy()
        max_abs = float(np.max(np.abs(coef))) if coef.size > 0 else 0.0
        thr = max(float(self.coef_prune_rel) * max_abs, float(self.coef_eps))
        coef[np.abs(coef) < thr] = 0.0
        return coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, va_idx = self._train_val_split(n, self.random_state, val_frac=0.2)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        self.alpha_ = self._select_alpha(X_tr, y_tr, X_va, y_va)

        rng = np.random.RandomState(self.random_state)
        boot_coefs = []
        boot_intercepts = []
        n_boot = max(1, int(self.n_bootstrap))
        sample_n = max(8, min(n, int(max(0.2, float(self.bootstrap_frac)) * n)))
        for _ in range(n_boot):
            idx = rng.choice(n, size=sample_n, replace=True)
            b0, w = self._fit_ridge_raw(X[idx], y[idx], self.alpha_)
            boot_intercepts.append(float(b0))
            boot_coefs.append(np.asarray(w, dtype=float))

        coef_mat = np.vstack(boot_coefs)
        coef_med = np.median(coef_mat, axis=0)
        sign_match = np.mean(np.sign(coef_mat) == np.sign(coef_med[None, :]), axis=0)
        stable = sign_match >= float(self.min_sign_stability)

        coef_med = coef_med * stable
        coef_med = self._prune_coef(coef_med)
        intercept = float(np.median(np.asarray(boot_intercepts, dtype=float)))
        intercept = float(np.mean(y) - np.dot(np.mean(X, axis=0), coef_med))

        self.coef_ = np.asarray(coef_med, dtype=float)
        self.intercept_ = intercept
        self.sign_stability_ = np.asarray(sign_match, dtype=float)
        self.active_features_ = np.where(self.coef_ != 0.0)[0]

        imp = np.abs(self.coef_)
        total = float(np.sum(imp))
        self.feature_importance_ = imp / total if total > 0 else imp

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "coef_",
                "intercept_",
                "n_features_in_",
            ],
        )
        X = np.asarray(X, dtype=float)
        return float(self.intercept_) + X @ np.asarray(self.coef_, dtype=float)

    @staticmethod
    def _fmt_signed(v):
        if v >= 0:
            return f"+ {v:.6f}"
        return f"- {abs(v):.6f}"

    def _eq_from_coef(self, intercept, coef):
        eq = f"{float(intercept):.6f}"
        for j in np.where(np.asarray(coef) != 0.0)[0]:
            eq += f" {self._fmt_signed(float(coef[j]))}*x{int(j)}"
        return eq

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_importance_", "n_features_in_"])

        lines = [
            "Bagged-Median Ridge Regressor",
            f"alpha: {self.alpha_:.4g}",
            "prediction_rule:",
            "  y = " + self._eq_from_coef(self.intercept_, self.coef_),
            "",
            "active_features:",
            "  " + (", ".join(f"x{int(j)}" for j in self.active_features_) if len(self.active_features_) else "none"),
        ]
        zero_feats = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.coef_[j] == 0.0
        ]
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
BaggedMedianRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BaggedMedianRidge_v1"
model_description = "Bootstrap-aggregated ridge with median coefficient stabilization, mild sign-consistency pruning, and a single explicit linear prediction rule"
model_defs = [(model_shorthand_name, BaggedMedianRidgeRegressor())]


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
