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


class SparseAffineResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse affine model with tiny nonlinear residual corrections.

    1) Fit a ridge-stabilized linear model on raw features.
    2) Build a small library of candidate nonlinear terms (square, hinge, interaction)
       from top correlated features.
    3) Greedily pick up to `max_extra_terms` residual terms and refit one final
       ridge model on [raw features + selected residual terms].

    This keeps the global equation explicit while allowing limited nonlinearity.
    """

    def __init__(
        self,
        ridge_alpha=0.3,
        top_features_for_library=6,
        max_extra_terms=2,
        min_resid_corr=0.03,
        min_coef_for_display=1e-4,
        random_state=42,
    ):
        self.ridge_alpha = ridge_alpha
        self.top_features_for_library = top_features_for_library
        self.max_extra_terms = max_extra_terms
        self.min_resid_corr = min_resid_corr
        self.min_coef_for_display = min_coef_for_display
        self.random_state = random_state

    @staticmethod
    def _fit_ridge_with_intercept(D, y, alpha):
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(D.shape[1]) * float(alpha)
        pen[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return np.asarray(beta, dtype=float)

    @staticmethod
    def _top_correlated_features(X, y, k):
        yc = y - y.mean()
        yn = np.sqrt(np.sum(yc ** 2)) + 1e-12
        scores = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            xc = xj - xj.mean()
            xn = np.sqrt(np.sum(xc ** 2)) + 1e-12
            corr = float((xc @ yc) / (xn * yn))
            scores.append(abs(corr))
        order = np.argsort(np.asarray(scores))[::-1]
        return [int(j) for j in order[: max(1, min(int(k), X.shape[1]))]]

    @staticmethod
    def _eval_term(X, term):
        kind = term["kind"]
        if kind == "square":
            j = term["feature"]
            return X[:, j] * X[:, j]
        if kind == "hinge_pos":
            j = term["feature"]
            t = term["threshold"]
            return np.maximum(0.0, X[:, j] - t)
        if kind == "hinge_neg":
            j = term["feature"]
            t = term["threshold"]
            return np.maximum(0.0, t - X[:, j])
        if kind == "interaction":
            j1 = term["feature_1"]
            j2 = term["feature_2"]
            return X[:, j1] * X[:, j2]
        raise ValueError(f"unknown term kind: {kind}")

    def _build_library(self, X, y):
        top = self._top_correlated_features(X, y, self.top_features_for_library)
        terms = []

        for j in top:
            xj = X[:, j]
            med = float(np.median(xj))
            terms.append({"kind": "square", "feature": int(j)})
            terms.append({"kind": "hinge_pos", "feature": int(j), "threshold": med})
            terms.append({"kind": "hinge_neg", "feature": int(j), "threshold": med})

        top2 = top[: min(3, len(top))]
        for a in range(len(top2)):
            for b in range(a + 1, len(top2)):
                j1 = int(top2[a])
                j2 = int(top2[b])
                terms.append({"kind": "interaction", "feature_1": j1, "feature_2": j2})

        return terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        D0 = np.column_stack([np.ones(n), X])
        beta0 = self._fit_ridge_with_intercept(D0, y, self.ridge_alpha)
        pred0 = D0 @ beta0
        resid = y - pred0

        library = self._build_library(X, y)
        selected_terms = []
        selected_cols = []

        for _ in range(max(0, int(self.max_extra_terms))):
            best_idx = None
            best_score = -1.0
            best_col = None

            for idx, term in enumerate(library):
                if idx in selected_terms:
                    continue
                col = self._eval_term(X, term)
                col_centered = col - col.mean()
                denom = (np.sqrt(np.sum(col_centered ** 2)) * np.sqrt(np.sum(resid ** 2)) + 1e-12)
                corr = float((col_centered @ resid) / denom)
                score = abs(corr)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_col = col

            if best_idx is None or best_score < float(self.min_resid_corr):
                break

            selected_terms.append(best_idx)
            selected_cols.append(best_col)

            colc = best_col - best_col.mean()
            step = float((colc @ resid) / (np.sum(colc ** 2) + 1e-12))
            resid = resid - step * colc

        if selected_cols:
            Z = np.column_stack(selected_cols)
            D = np.column_stack([np.ones(n), X, Z])
        else:
            D = D0

        beta = self._fit_ridge_with_intercept(D, y, self.ridge_alpha)

        self.intercept_ = float(beta[0])
        self.linear_coef_ = np.asarray(beta[1 : 1 + p], dtype=float)

        self.extra_terms_ = []
        if selected_cols:
            extra_coefs = np.asarray(beta[1 + p :], dtype=float)
            for k, idx in enumerate(selected_terms):
                t = dict(library[idx])
                t["coef"] = float(extra_coefs[k])
                self.extra_terms_.append(t)

        importance = np.abs(self.linear_coef_)
        for t in self.extra_terms_:
            if t["kind"] in {"square", "hinge_pos", "hinge_neg"}:
                importance[t["feature"]] += abs(t["coef"])
            else:
                importance[t["feature_1"]] += 0.5 * abs(t["coef"])
                importance[t["feature_2"]] += 0.5 * abs(t["coef"])
        self.feature_importance_ = importance

        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "extra_terms_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        yhat += X @ self.linear_coef_

        for t in self.extra_terms_:
            c = t["coef"]
            kind = t["kind"]
            if kind == "square":
                j = t["feature"]
                yhat += c * (X[:, j] ** 2)
            elif kind == "hinge_pos":
                j = t["feature"]
                th = t["threshold"]
                yhat += c * np.maximum(0.0, X[:, j] - th)
            elif kind == "hinge_neg":
                j = t["feature"]
                th = t["threshold"]
                yhat += c * np.maximum(0.0, th - X[:, j])
            else:
                j1 = t["feature_1"]
                j2 = t["feature_2"]
                yhat += c * (X[:, j1] * X[:, j2])

        return yhat

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "extra_terms_", "feature_importance_"])

        lines = [
            "Sparse Affine Residual Regressor",
            "prediction_rule: add the terms exactly as written",
            "",
            "equation:",
        ]

        pieces = [f"{self.intercept_:+.6f}"]
        for j, c in enumerate(self.linear_coef_):
            if abs(c) >= float(self.min_coef_for_display):
                pieces.append(f"({c:+.6f})*x{j}")

        for t in self.extra_terms_:
            c = t["coef"]
            if abs(c) < float(self.min_coef_for_display):
                continue
            if t["kind"] == "square":
                pieces.append(f"({c:+.6f})*(x{t['feature']}^2)")
            elif t["kind"] == "hinge_pos":
                pieces.append(f"({c:+.6f})*max(0, x{t['feature']}-{t['threshold']:.4f})")
            elif t["kind"] == "hinge_neg":
                pieces.append(f"({c:+.6f})*max(0, {t['threshold']:.4f}-x{t['feature']})")
            else:
                pieces.append(f"({c:+.6f})*(x{t['feature_1']}*x{t['feature_2']})")

        lines.append("  y = " + " + ".join(pieces))
        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{j}: total_importance={self.feature_importance_[j]:.6f}, linear_coef={self.linear_coef_[j]:+.6f}"
            )

        max_imp = float(np.max(self.feature_importance_)) if self.feature_importance_.size > 0 else 0.0
        if max_imp > 0:
            weak = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 0.05 * max_imp]
            if weak:
                lines.append("")
                lines.append("features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("manual_prediction:")
        lines.append("  1) Start from intercept.")
        lines.append("  2) Add linear terms coef*xj.")
        lines.append("  3) Add extra nonlinear terms if present.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAffineResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAffineResid_v1"
model_description = "Ridge-stabilized sparse affine backbone plus at most two residual nonlinear correction terms (square/hinge/interaction) with explicit closed-form equation"
model_defs = [(model_shorthand_name, SparseAffineResidualRegressor())]


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
