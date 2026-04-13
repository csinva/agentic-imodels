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


class SplitAffineRidgeRegressor(BaseEstimator, RegressorMixin):
    """Validation-gated one-split piecewise affine regressor with ridge leaf fits."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        split_feature_screen=16,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_leaf_frac=0.08,
        min_leaf_count=20,
        split_complexity_penalty=0.015,
        min_relative_gain=0.01,
        coef_prune_rel=0.015,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.split_feature_screen = split_feature_screen
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_leaf_count = min_leaf_count
        self.split_complexity_penalty = split_complexity_penalty
        self.min_relative_gain = min_relative_gain
        self.coef_prune_rel = coef_prune_rel
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _train_val_split(n, seed):
        if n < 30:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
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
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best[1]

    @staticmethod
    def _safe_corr(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2)) + 1e-12
        return float(np.dot(a0, b0) / denom)

    def _prune_coef(self, coef):
        coef = np.asarray(coef, dtype=float).copy()
        max_abs = float(np.max(np.abs(coef))) if coef.size > 0 else 0.0
        thr = max(float(self.coef_prune_rel) * max_abs, float(self.coef_eps))
        coef[np.abs(coef) < thr] = 0.0
        return coef

    @staticmethod
    def _predict_piecewise(X, split_feat, split_thr, left_b0, left_w, right_b0, right_w):
        mask = X[:, split_feat] <= split_thr
        out = np.empty(X.shape[0], dtype=float)
        if np.any(mask):
            out[mask] = left_b0 + X[mask] @ left_w
        if np.any(~mask):
            out[~mask] = right_b0 + X[~mask] @ right_w
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, va_idx = self._train_val_split(n, self.random_state)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        self.base_alpha_ = self._select_alpha(X_tr, y_tr, X_va, y_va)
        base_b0, base_w = self._fit_ridge_raw(X_tr, y_tr, self.base_alpha_)
        base_pred_va = base_b0 + X_va @ base_w
        base_mse_va = float(np.mean((y_va - base_pred_va) ** 2))

        corr = np.array([abs(self._safe_corr(X_tr[:, j], y_tr)) for j in range(p)], dtype=float)
        order = np.argsort(corr)[::-1]
        m = int(min(max(1, self.split_feature_screen), p))
        screened_features = [int(j) for j in order[:m] if corr[j] > 0]
        if len(screened_features) == 0:
            screened_features = list(range(min(p, m)))

        min_leaf = max(int(self.min_leaf_frac * len(X_tr)), int(self.min_leaf_count))
        min_leaf = min(min_leaf, max(2, len(X_tr) // 2))

        best = None
        for feat in screened_features:
            vals = X_tr[:, feat]
            thrs = np.unique(np.quantile(vals, self.split_quantiles))
            for thr in thrs:
                left_tr = vals <= thr
                right_tr = ~left_tr
                if int(np.sum(left_tr)) < min_leaf or int(np.sum(right_tr)) < min_leaf:
                    continue

                left_va = X_va[:, feat] <= thr
                right_va = ~left_va
                if int(np.sum(left_va)) == 0 or int(np.sum(right_va)) == 0:
                    continue

                alpha_l = self._select_alpha(X_tr[left_tr], y_tr[left_tr], X_va[left_va], y_va[left_va])
                alpha_r = self._select_alpha(X_tr[right_tr], y_tr[right_tr], X_va[right_va], y_va[right_va])
                b0_l, w_l = self._fit_ridge_raw(X_tr[left_tr], y_tr[left_tr], alpha_l)
                b0_r, w_r = self._fit_ridge_raw(X_tr[right_tr], y_tr[right_tr], alpha_r)

                pred_va = self._predict_piecewise(X_va, int(feat), float(thr), b0_l, w_l, b0_r, w_r)
                mse = float(np.mean((y_va - pred_va) ** 2))
                obj = mse * (1.0 + float(self.split_complexity_penalty))

                if best is None or obj < best[0]:
                    best = (obj, mse, int(feat), float(thr), float(alpha_l), float(alpha_r))

        self.has_split_ = False
        self.split_feature_ = -1
        self.split_threshold_ = 0.0

        self.left_alpha_ = float(self.base_alpha_)
        self.right_alpha_ = float(self.base_alpha_)

        use_split = False
        if best is not None:
            _, split_mse_va, feat, thr, alpha_l, alpha_r = best
            rel_gain = (base_mse_va - split_mse_va) / (abs(base_mse_va) + 1e-12)
            use_split = rel_gain > float(self.min_relative_gain)
            if use_split:
                self.has_split_ = True
                self.split_feature_ = int(feat)
                self.split_threshold_ = float(thr)
                self.left_alpha_ = float(alpha_l)
                self.right_alpha_ = float(alpha_r)

        if not use_split:
            b0, w = self._fit_ridge_raw(X, y, self.base_alpha_)
            w = self._prune_coef(w)
            self.global_intercept_ = float(b0)
            self.global_coef_ = np.asarray(w, dtype=float)

            self.left_intercept_ = float(b0)
            self.left_coef_ = np.asarray(w, dtype=float)
            self.right_intercept_ = float(b0)
            self.right_coef_ = np.asarray(w, dtype=float)
        else:
            left = X[:, self.split_feature_] <= self.split_threshold_
            right = ~left

            b0_l, w_l = self._fit_ridge_raw(X[left], y[left], self.left_alpha_)
            b0_r, w_r = self._fit_ridge_raw(X[right], y[right], self.right_alpha_)
            w_l = self._prune_coef(w_l)
            w_r = self._prune_coef(w_r)

            self.left_intercept_ = float(b0_l)
            self.left_coef_ = np.asarray(w_l, dtype=float)
            self.right_intercept_ = float(b0_r)
            self.right_coef_ = np.asarray(w_r, dtype=float)

            self.global_intercept_ = float(np.mean(y) - np.dot(np.mean(X, axis=0), 0.5 * (w_l + w_r)))
            self.global_coef_ = 0.5 * (w_l + w_r)

        self.coef_ = np.asarray(self.global_coef_, dtype=float)
        self.intercept_ = float(self.global_intercept_)

        imp = 0.5 * np.abs(self.left_coef_) + 0.5 * np.abs(self.right_coef_)
        if self.has_split_:
            imp[self.split_feature_] += float(np.mean(np.abs(y - np.mean(y)))) * 0.05
        total = float(np.sum(imp))
        self.feature_importance_ = imp / total if total > 0 else imp

        self.left_active_features_ = np.where(self.left_coef_ != 0.0)[0]
        self.right_active_features_ = np.where(self.right_coef_ != 0.0)[0]

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "has_split_",
                "left_intercept_",
                "left_coef_",
                "right_intercept_",
                "right_coef_",
                "n_features_in_",
            ],
        )
        X = np.asarray(X, dtype=float)

        if not self.has_split_:
            return float(self.left_intercept_) + X @ self.left_coef_

        return self._predict_piecewise(
            X,
            int(self.split_feature_),
            float(self.split_threshold_),
            float(self.left_intercept_),
            np.asarray(self.left_coef_, dtype=float),
            float(self.right_intercept_),
            np.asarray(self.right_coef_, dtype=float),
        )

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
        check_is_fitted(self, ["has_split_", "left_coef_", "right_coef_", "feature_importance_", "n_features_in_"])

        lines = [
            "Split-Affine Ridge Regressor",
            f"base_alpha: {self.base_alpha_:.4g}",
            f"uses_split: {self.has_split_}",
        ]

        if not self.has_split_:
            lines.append("equation:")
            lines.append("  y = " + self._eq_from_coef(self.left_intercept_, self.left_coef_))
        else:
            lines.append(
                f"rule: if x{int(self.split_feature_)} <= {float(self.split_threshold_):.6f} then LEFT else RIGHT"
            )
            lines.append(f"left_alpha: {self.left_alpha_:.4g}  right_alpha: {self.right_alpha_:.4g}")
            lines.append("equations:")
            lines.append("  LEFT:  y = " + self._eq_from_coef(self.left_intercept_, self.left_coef_))
            lines.append("  RIGHT: y = " + self._eq_from_coef(self.right_intercept_, self.right_coef_))

        lines.append("")
        lines.append("active_features_by_region:")
        left_feats = ", ".join(f"x{int(j)}" for j in self.left_active_features_) or "none"
        right_feats = ", ".join(f"x{int(j)}" for j in self.right_active_features_) or "none"
        lines.append(f"  left: {left_feats}")
        lines.append(f"  right: {right_feats}")

        zero_feats = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.left_coef_[j] == 0.0 and self.right_coef_[j] == 0.0
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
SplitAffineRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SplitAffineRidge_v1"
model_description = "Validation-gated one-split piecewise affine ridge model with explicit if-then leaf equations and coefficient pruning for compact simulation"
model_defs = [(model_shorthand_name, SplitAffineRidgeRegressor())]


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
