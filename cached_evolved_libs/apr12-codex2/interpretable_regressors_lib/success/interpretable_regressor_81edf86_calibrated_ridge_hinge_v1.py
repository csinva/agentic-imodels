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


class CalibratedRidgeHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense calibrated ridge equation with one optional hinge correction.

    The linear backbone preserves strong predictive performance.
    A single validation-gated hinge term captures threshold-like behavior while
    keeping the model arithmetic short and readable.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        hinge_features=4,
        hinge_quantiles=(0.25, 0.5, 0.75),
        min_rel_gain=0.01,
        coef_eps=1e-8,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.hinge_features = hinge_features
        self.hinge_quantiles = hinge_quantiles
        self.min_rel_gain = min_rel_gain
        self.coef_eps = coef_eps
        self.random_state = random_state

    def _make_split(self, n):
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = int(round(float(self.val_frac) * n))
        n_val = min(max(1, n_val), max(1, n - 1))
        return perm[n_val:], perm[:n_val]

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
                best = (mse, float(alpha), b0, coef)
        return best

    @staticmethod
    def _screen_features(X, y):
        cy = y - np.mean(y)
        y_norm = float(np.sqrt(np.sum(cy ** 2))) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            cx = X[:, j] - np.mean(X[:, j])
            x_norm = float(np.sqrt(np.sum(cx ** 2))) + 1e-12
            scores[j] = abs(float(np.dot(cx, cy) / (x_norm * y_norm)))
        return np.argsort(scores)[::-1]

    @staticmethod
    def _std_params(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return mu, sigma

    @staticmethod
    def _hinge_values(X_col, threshold, direction):
        if direction > 0:
            return np.maximum(0.0, X_col - threshold)
        return np.maximum(0.0, threshold - X_col)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        self.x_mean_, self.x_scale_ = self._std_params(X_tr)
        Xtr_s = (X_tr - self.x_mean_) / self.x_scale_
        Xval_s = (X_val - self.x_mean_) / self.x_scale_

        # Baseline calibrated ridge
        base_mse, base_alpha, _, _ = self._best_alpha(Xtr_s, y_tr, Xval_s, y_val)

        # Candidate hinge terms: small screened set only
        screened = self._screen_features(X_tr, y_tr)
        top = screened[: min(int(self.hinge_features), p)]
        candidates = []
        for j in top:
            xj = X_tr[:, int(j)]
            for q in self.hinge_quantiles:
                thr = float(np.quantile(xj, float(q)))
                candidates.append((int(j), thr, 1))
                candidates.append((int(j), thr, -1))

        best = (base_mse, base_alpha, None, None)  # mse, alpha, term, g_stats
        for j, thr, direction in candidates:
            g_tr = self._hinge_values(X_tr[:, j], thr, direction)
            g_val = self._hinge_values(X_val[:, j], thr, direction)
            g_mu = float(np.mean(g_tr))
            g_std = float(np.std(g_tr))
            if g_std < 1e-12:
                continue
            gz_tr = ((g_tr - g_mu) / g_std).reshape(-1, 1)
            gz_val = ((g_val - g_mu) / g_std).reshape(-1, 1)
            D_tr = np.column_stack([Xtr_s, gz_tr])
            D_val = np.column_stack([Xval_s, gz_val])
            mse, alpha, _, _ = self._best_alpha(D_tr, y_tr, D_val, y_val)
            if mse < best[0]:
                best = (mse, alpha, (j, thr, direction), (g_mu, g_std))

        rel_gain = (base_mse - best[0]) / (base_mse + 1e-12)
        if best[2] is not None and rel_gain >= float(self.min_rel_gain):
            self.hinge_term_ = best[2]
            self.hinge_stats_ = best[3]
            final_alpha = best[1]
        else:
            self.hinge_term_ = None
            self.hinge_stats_ = None
            final_alpha = base_alpha

        # Final refit on full data
        x_mean_full, x_scale_full = self._std_params(X)
        self.x_mean_ = x_mean_full
        self.x_scale_ = x_scale_full
        Xs = (X - self.x_mean_) / self.x_scale_

        if self.hinge_term_ is not None:
            j, thr, direction = self.hinge_term_
            g = self._hinge_values(X[:, j], thr, direction)
            g_mu = float(np.mean(g))
            g_std = float(np.std(g))
            if g_std < 1e-12:
                self.hinge_term_ = None
                self.hinge_stats_ = None
                D = Xs
            else:
                self.hinge_stats_ = (g_mu, g_std)
                gz = ((g - g_mu) / g_std).reshape(-1, 1)
                D = np.column_stack([Xs, gz])
        else:
            D = Xs

        intercept_std, coef_std = self._ridge_fit(D, y, final_alpha)
        coef_std = np.asarray(coef_std, dtype=float)
        linear_std = coef_std[:p]
        hinge_std = float(coef_std[-1]) if self.hinge_term_ is not None else 0.0

        self.linear_coef_ = linear_std / self.x_scale_
        self.intercept_ = float(intercept_std - np.dot(linear_std, self.x_mean_ / self.x_scale_))

        if self.hinge_term_ is not None:
            g_mu, g_std = self.hinge_stats_
            self.hinge_coef_ = float(hinge_std / g_std)
            self.intercept_ -= float(self.hinge_coef_ * g_mu)
        else:
            self.hinge_coef_ = 0.0

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        if abs(self.hinge_coef_) < float(self.coef_eps):
            self.hinge_coef_ = 0.0
            self.hinge_term_ = None
            self.hinge_stats_ = None

        fi = np.abs(self.linear_coef_)
        if self.hinge_term_ is not None:
            fi[int(self.hinge_term_[0])] += abs(float(self.hinge_coef_))
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def _predict_no_check(self, X):
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ self.linear_coef_
        if self.hinge_term_ is not None and abs(float(self.hinge_coef_)) > 0:
            j, thr, direction = self.hinge_term_
            yhat = yhat + float(self.hinge_coef_) * self._hinge_values(X[:, int(j)], float(thr), int(direction))
        return yhat

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        return self._predict_no_check(X)

    def __str__(self):
        check_is_fitted(self, ["linear_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        lines = ["Calibrated Ridge + One-Hinge Regressor", "equation:"]
        terms = [f"{float(self.intercept_):+.6f}"]
        nz = np.where(np.abs(self.linear_coef_) > 0)[0]
        for j in nz:
            terms.append(f"({float(self.linear_coef_[j]):+.6f})*x{int(j)}")
        if self.hinge_term_ is not None and abs(float(self.hinge_coef_)) > 0:
            j, thr, direction = self.hinge_term_
            if direction > 0:
                htxt = f"max(0, x{int(j)} - {float(thr):.6f})"
            else:
                htxt = f"max(0, {float(thr):.6f} - x{int(j)})"
            terms.append(f"({float(self.hinge_coef_):+.6f})*{htxt}")
        lines.append("  y = " + " + ".join(terms))

        lines.append("")
        lines.append("active_linear_terms:")
        if len(nz) == 0:
            lines.append("  none")
        else:
            for j in nz:
                lines.append(f"  ({float(self.linear_coef_[j]):+.6f}) * x{int(j)}")

        zero_idx = [f"x{j}" for j in range(self.n_features_in_) if abs(float(self.linear_coef_[j])) == 0.0]
        if zero_idx:
            lines.append("")
            lines.append("zero_or_negligible_features:")
            lines.append("  " + ", ".join(zero_idx))

        if self.hinge_term_ is not None and abs(float(self.hinge_coef_)) > 0:
            lines.append("")
            lines.append("hinge_term:")
            j, thr, direction = self.hinge_term_
            if direction > 0:
                lines.append(f"  ({float(self.hinge_coef_):+.6f}) * max(0, x{int(j)} - {float(thr):.6f})")
            else:
                lines.append(f"  ({float(self.hinge_coef_):+.6f}) * max(0, {float(thr):.6f} - x{int(j)})")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        shown = 0
        for j in order:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{int(j)}: {float(self.feature_importance_[j]):.4f}")
            shown += 1
            if shown >= 15:
                break
        if shown == 0:
            lines.append("  all features have zero contribution")

        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) start from intercept")
        lines.append("  2) add each listed linear term coefficient times feature value")
        lines.append("  3) if hinge term exists, compute its max(0, ...) value and add it")
        lines.append("  4) features listed as zero_or_negligible_features contribute 0")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
CalibratedRidgeHingeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "CalibratedRidgeHinge_v1"
model_description = "Calibrated dense ridge equation with holdout-selected regularization plus one validation-gated hinge correction term for threshold-like effects"
model_defs = [(model_shorthand_name, CalibratedRidgeHingeRegressor())]


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
