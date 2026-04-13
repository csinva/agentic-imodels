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


class SparseMedianHingeGAMRegressor(BaseEstimator, RegressorMixin):
    """Dense ridge linear equation plus a tiny set of median-hinge residual terms."""

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        max_hinges=3,
        hinge_screen=12,
        hinge_complexity_penalty=0.03,
        min_hinge_gain=1e-4,
        linear_prune_rel=0.002,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_hinges = max_hinges
        self.hinge_screen = hinge_screen
        self.hinge_complexity_penalty = hinge_complexity_penalty
        self.min_hinge_gain = min_hinge_gain
        self.linear_prune_rel = linear_prune_rel
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _train_val_split(n, seed):
        if n < 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    @staticmethod
    def _ridge_fit_no_intercept(X, y, alpha):
        p = X.shape[1]
        gram = X.T @ X + float(alpha) * np.eye(p, dtype=float)
        rhs = X.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs
        return np.asarray(coef, dtype=float)

    def _fit_ridge_raw_equation(self, X, y, alpha):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std > 1e-12, x_std, 1.0)
        y_mean = float(np.mean(y))

        Xs = (X - x_mean) / x_std
        yc = y - y_mean
        coef_std = self._ridge_fit_no_intercept(Xs, yc, alpha)

        coef = coef_std / x_std
        intercept = y_mean - float(np.dot(coef, x_mean))
        return float(intercept), np.asarray(coef, dtype=float)

    def _select_alpha(self, X_tr, y_tr, X_va, y_va):
        best = None
        for alpha in self.alpha_grid:
            b0, w = self._fit_ridge_raw_equation(X_tr, y_tr, float(alpha))
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

    @staticmethod
    def _hinge_col(x, thr):
        return np.maximum(0.0, x - float(thr))

    @staticmethod
    def _fit_small_ridge(Phi, y, alpha=1e-8):
        if Phi.size == 0:
            return np.zeros(0, dtype=float)
        p = Phi.shape[1]
        gram = Phi.T @ Phi + float(alpha) * np.eye(p, dtype=float)
        rhs = Phi.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs
        return np.asarray(coef, dtype=float)

    def _forward_select_hinges(self, X_tr, r_tr, X_va, r_va):
        p = X_tr.shape[1]
        med = np.median(X_tr, axis=0)

        scores = []
        for j in range(p):
            phi = self._hinge_col(X_tr[:, j], med[j])
            scores.append(abs(self._safe_corr(phi, r_tr)))

        order = np.argsort(scores)[::-1]
        m = int(min(max(1, self.hinge_screen), p))
        cand = [int(j) for j in order[:m] if scores[j] > 0]

        selected = []
        centers = np.zeros(0, dtype=float)
        coefs = np.zeros(0, dtype=float)
        best_obj = float(np.mean(r_va ** 2))

        for _ in range(int(max(0, self.max_hinges))):
            best_step = None
            for j in cand:
                if j in selected:
                    continue
                trial = selected + [j]

                Phi_tr = np.column_stack([self._hinge_col(X_tr[:, k], med[k]) for k in trial])
                ctr = np.mean(Phi_tr, axis=0)
                beta = self._fit_small_ridge(Phi_tr - ctr, r_tr)

                Phi_va = np.column_stack([self._hinge_col(X_va[:, k], med[k]) for k in trial])
                pred_va = (Phi_va - ctr) @ beta
                mse = float(np.mean((r_va - pred_va) ** 2))
                obj = mse * (1.0 + float(self.hinge_complexity_penalty) * len(trial))

                if best_step is None or obj < best_step[0]:
                    best_step = (obj, trial, ctr, beta)

            if best_step is None:
                break

            obj, trial, ctr, beta = best_step
            if best_obj - obj <= float(self.min_hinge_gain):
                break

            best_obj = obj
            selected = trial
            centers = ctr
            coefs = beta

        if len(selected) == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        return (
            np.asarray(selected, dtype=int),
            np.asarray([med[j] for j in selected], dtype=float),
            np.asarray(centers, dtype=float),
            np.asarray(coefs, dtype=float),
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, va_idx = self._train_val_split(n, self.random_state)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        self.alpha_ = self._select_alpha(X_tr, y_tr, X_va, y_va)
        self.intercept_, coef = self._fit_ridge_raw_equation(X, y, self.alpha_)

        max_abs = float(np.max(np.abs(coef))) if p > 0 else 0.0
        thr = max(float(self.linear_prune_rel) * max_abs, float(self.coef_eps))
        coef[np.abs(coef) < thr] = 0.0
        self.coef_ = coef
        self.active_linear_features_ = np.where(self.coef_ != 0.0)[0]

        b0_tr, w_tr = self._fit_ridge_raw_equation(X_tr, y_tr, self.alpha_)
        resid_tr = y_tr - (b0_tr + X_tr @ w_tr)
        resid_va = y_va - (b0_tr + X_va @ w_tr)

        (
            self.hinge_features_,
            self.hinge_thresholds_,
            self.hinge_centers_,
            _,
        ) = self._forward_select_hinges(X_tr, resid_tr, X_va, resid_va)

        if len(self.hinge_features_) > 0:
            Phi_full = np.column_stack([
                self._hinge_col(X[:, j], t)
                for j, t in zip(self.hinge_features_, self.hinge_thresholds_)
            ])
            self.hinge_centers_ = np.mean(Phi_full, axis=0)
            resid_full = y - (self.intercept_ + X @ self.coef_)
            self.hinge_coefs_ = self._fit_small_ridge(Phi_full - self.hinge_centers_, resid_full)
        else:
            self.hinge_coefs_ = np.zeros(0, dtype=float)

        lin_imp = np.abs(self.coef_)
        hinge_imp = np.zeros(p, dtype=float)
        for j, c in zip(self.hinge_features_, self.hinge_coefs_):
            hinge_imp[int(j)] += abs(float(c))
        imp = lin_imp + hinge_imp
        total = float(np.sum(imp))
        self.feature_importance_ = imp / total if total > 0 else imp

        return self

    def _hinge_matrix(self, X):
        if len(self.hinge_features_) == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack([
            self._hinge_col(X[:, j], t)
            for j, t in zip(self.hinge_features_, self.hinge_thresholds_)
        ])

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "coef_",
                "intercept_",
                "hinge_features_",
                "hinge_thresholds_",
                "hinge_centers_",
                "hinge_coefs_",
                "n_features_in_",
            ],
        )
        X = np.asarray(X, dtype=float)

        pred = float(self.intercept_) + X @ self.coef_
        if len(self.hinge_features_) > 0:
            Phi = self._hinge_matrix(X)
            pred = pred + (Phi - self.hinge_centers_) @ self.hinge_coefs_
        return pred

    @staticmethod
    def _fmt_signed(v):
        if v >= 0:
            return f"+ {v:.6f}"
        return f"- {abs(v):.6f}"

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_importance_", "n_features_in_"])

        lines = [
            "Sparse Median-Hinge GAM Regressor",
            f"selected_ridge_alpha: {self.alpha_:.4g}",
            f"linear_active_features: {len(self.active_linear_features_)}/{self.n_features_in_}",
            f"selected_hinge_terms: {len(self.hinge_features_)}",
            "equation:",
        ]

        eq = f"  y = {self.intercept_:.6f}"
        for j in np.where(self.coef_ != 0.0)[0]:
            eq += f" {self._fmt_signed(float(self.coef_[j]))}*x{int(j)}"
        for idx, j in enumerate(self.hinge_features_):
            coef = float(self.hinge_coefs_[idx])
            if abs(coef) <= 0:
                continue
            thr = float(self.hinge_thresholds_[idx])
            ctr = float(self.hinge_centers_[idx])
            eq += f" {self._fmt_signed(coef)}*(max(0, x{int(j)} - {thr:.6f}) - {ctr:.6f})"
        lines.append(eq)

        lines.append("")
        lines.append("coefficients:")
        nz = np.where(self.coef_ != 0.0)[0]
        if len(nz) == 0:
            lines.append("  linear: none")
        else:
            for j in nz:
                lines.append(f"  x{int(j)}: {self.coef_[j]:+.6f}")

        if len(self.hinge_features_) == 0:
            lines.append("  hinge_terms: none")
        else:
            lines.append("  hinge_terms:")
            for idx, j in enumerate(self.hinge_features_):
                lines.append(
                    f"    {self.hinge_coefs_[idx]:+.6f} * "
                    f"(max(0, x{int(j)} - {self.hinge_thresholds_[idx]:.6f}) - {self.hinge_centers_[idx]:.6f})"
                )

        hinge_set = set(self.hinge_features_.tolist())
        zero_feats = [f"x{j}" for j, c in enumerate(self.coef_) if c == 0.0 and j not in hinge_set]
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
SparseMedianHingeGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseMedianHingeGAM_v1"
model_description = "GA2M-style explicit equation: dense ridge linear backbone plus forward-selected median-hinge terms fit on residuals for compact threshold nonlinearities"
model_defs = [(model_shorthand_name, SparseMedianHingeGAMRegressor())]


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
