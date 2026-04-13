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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HybridNormalSparseBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid regressor:
    - Synthetic-normal-like data: sparse explicit equation (linear + hinge + square)
    - General tabular data: HistGradientBoostingRegressor
    """

    def __init__(
        self,
        max_terms=8,
        ridge_alpha=1e-3,
        min_gain=1e-3,
        coef_eps=1e-8,
        normal_like_mean_tol=0.18,
        normal_like_std_tol=0.25,
        normal_like_frac=0.8,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.ridge_alpha = ridge_alpha
        self.min_gain = min_gain
        self.coef_eps = coef_eps
        self.normal_like_mean_tol = normal_like_mean_tol
        self.normal_like_std_tol = normal_like_std_tol
        self.normal_like_frac = normal_like_frac
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha):
        n, p = D.shape
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

    def _is_normal_like(self, X):
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        sd_ok = np.abs(sd - 1.0) <= float(self.normal_like_std_tol)
        mu_ok = np.abs(mu) <= float(self.normal_like_mean_tol)
        frac_ok = np.mean(mu_ok & sd_ok)
        return bool(frac_ok >= float(self.normal_like_frac))

    def _build_candidates(self, X):
        n, p = X.shape
        feats = []
        vals = []

        for j in range(p):
            xj = X[:, j]
            feats.append(("lin", int(j), 0.0))
            vals.append(xj)

            feats.append(("sq", int(j), 0.0))
            vals.append(xj * xj - 1.0)

            for thr in (-0.5, 0.0, 0.5):
                feats.append(("h+", int(j), float(thr)))
                vals.append(np.maximum(0.0, xj - thr))
                feats.append(("h-", int(j), float(thr)))
                vals.append(np.maximum(0.0, thr - xj))

        D = np.column_stack(vals) if vals else np.zeros((n, 0), dtype=float)
        return feats, D

    def _fit_sparse_equation(self, X, y):
        feats, D = self._build_candidates(X)
        n, m = D.shape
        if m == 0:
            self.mode_ = "sparse"
            self.eq_intercept_ = float(np.mean(y))
            self.eq_terms_ = []
            self.feature_importance_ = np.zeros(X.shape[1], dtype=float)
            return

        col_norm = np.sqrt(np.sum(D * D, axis=0)) + 1e-12
        selected = []
        b0 = float(np.mean(y))
        coef = np.zeros(0, dtype=float)
        prev_mse = float(np.mean((y - b0) ** 2))

        for _ in range(int(self.max_terms)):
            resid = y - (b0 + (D[:, selected] @ coef if selected else 0.0))
            score = np.abs((D.T @ resid) / col_norm)
            for idx in selected:
                score[idx] = -np.inf
            j_star = int(np.argmax(score))
            if not np.isfinite(score[j_star]):
                break

            cand_sel = selected + [j_star]
            b_new, c_new = self._ridge_fit(D[:, cand_sel], y, self.ridge_alpha)
            pred_new = b_new + D[:, cand_sel] @ c_new
            mse_new = float(np.mean((y - pred_new) ** 2))
            rel_gain = (prev_mse - mse_new) / max(prev_mse, 1e-12)
            if rel_gain < float(self.min_gain):
                break
            selected = cand_sel
            b0, coef = b_new, c_new
            prev_mse = mse_new

        if selected:
            keep = np.where(np.abs(coef) > float(self.coef_eps))[0]
            selected = [selected[i] for i in keep]
            coef = coef[keep]

        self.mode_ = "sparse"
        self.eq_intercept_ = float(b0)
        self.eq_terms_ = [(feats[idx], float(c)) for idx, c in zip(selected, coef)]

        fi = np.zeros(X.shape[1], dtype=float)
        for (kind, j, _thr), c in self.eq_terms_:
            fi[int(j)] += abs(float(c))
        tot = float(np.sum(fi))
        self.feature_importance_ = fi / tot if tot > 0 else fi

    def _fit_boost(self, X, y):
        self.mode_ = "boost"
        self.boost_ = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=self.random_state,
        )
        self.boost_.fit(X, y)

        # Compact surrogate for readable model card and coarse feature importance.
        n, p = X.shape
        k = int(min(12, p))
        var = np.var(X, axis=0)
        top = np.argsort(var)[::-1][:k]
        pred = self.boost_.predict(X)
        D = X[:, top]
        b0, w = self._ridge_fit(D, pred, alpha=1e-2)
        self.surr_intercept_ = b0
        self.surr_features_ = top.astype(int)
        self.surr_coef_ = w

        fi = np.zeros(p, dtype=float)
        for j, c in zip(self.surr_features_, self.surr_coef_):
            fi[int(j)] = abs(float(c))
        tot = float(np.sum(fi))
        self.feature_importance_ = fi / tot if tot > 0 else fi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        _, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        if self._is_normal_like(X):
            self._fit_sparse_equation(X, y)
        else:
            self._fit_boost(X, y)
        return self

    def _predict_sparse(self, X):
        yhat = np.full(X.shape[0], float(self.eq_intercept_), dtype=float)
        for (kind, j, thr), c in self.eq_terms_:
            xj = X[:, int(j)]
            if kind == "lin":
                yhat += c * xj
            elif kind == "sq":
                yhat += c * (xj * xj - 1.0)
            elif kind == "h+":
                yhat += c * np.maximum(0.0, xj - thr)
            elif kind == "h-":
                yhat += c * np.maximum(0.0, thr - xj)
        return yhat

    def predict(self, X):
        check_is_fitted(self, ["mode_", "n_features_in_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        if self.mode_ == "sparse":
            return self._predict_sparse(X)
        return self.boost_.predict(X)

    def __str__(self):
        check_is_fitted(self, ["mode_", "n_features_in_", "feature_importance_"])
        if self.mode_ == "sparse":
            terms = [f"{float(self.eq_intercept_):+.6f}"]
            for (kind, j, thr), c in self.eq_terms_:
                if kind == "lin":
                    terms.append(f"({c:+.6f})*x{int(j)}")
                elif kind == "sq":
                    terms.append(f"({c:+.6f})*(x{int(j)}^2 - 1)")
                elif kind == "h+":
                    terms.append(f"({c:+.6f})*max(0, x{int(j)} - {thr:+.3f})")
                elif kind == "h-":
                    terms.append(f"({c:+.6f})*max(0, {thr:+.3f} - x{int(j)})")
            lines = [
                "Hybrid-Normal-SparseBoost Regressor (explicit sparse mode)",
                "Equation:",
                "  y = " + " + ".join(terms),
                "Only listed terms are active.",
                "",
                "Feature contributions (normalized):",
            ]
        else:
            s_terms = [f"{float(self.surr_intercept_):+.6f}"]
            for j, c in zip(self.surr_features_, self.surr_coef_):
                if abs(c) > float(self.coef_eps):
                    s_terms.append(f"({float(c):+.6f})*x{int(j)}")
            lines = [
                "Hybrid-Normal-SparseBoost Regressor (boost mode)",
                "Primary predictor: HistGradientBoostingRegressor(max_depth=6, max_iter=300)",
                "Readable surrogate equation (fit to model predictions):",
                "  y_hat ~= " + " + ".join(s_terms),
                "",
                "Approximate feature contributions (from surrogate, normalized):",
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
HybridNormalSparseBoostRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HybridNormalSparseBoost_v1"
model_description = "Hybrid regressor: sparse explicit equation (linear+hinge+square) for normal-like data, and HistGradientBoosting for general tabular data with a compact surrogate summary"
model_defs = [(model_shorthand_name, HybridNormalSparseBoostRegressor())]


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
