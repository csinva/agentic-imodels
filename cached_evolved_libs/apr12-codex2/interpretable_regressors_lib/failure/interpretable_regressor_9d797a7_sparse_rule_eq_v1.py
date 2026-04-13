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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseRuleEquationRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with explicit symbolic terms.

    The model builds a library of transparent terms:
    - linear terms: xj
    - quadratic terms: xj^2
    - hinge terms: max(0, xj - median_j)
    - pairwise interactions among top linear-correlation features: xi*xj

    It then uses L1 selection (LassoCV) to keep only a small active set and
    refits least squares on those active terms for lower bias.
    """

    def __init__(
        self,
        max_active_terms=8,
        interaction_candidates=4,
        cv=3,
        random_state=42,
    ):
        self.max_active_terms = max_active_terms
        self.interaction_candidates = interaction_candidates
        self.cv = cv
        self.random_state = random_state

    def _feature_library(self, X):
        n_samples, n_features = X.shape
        cols = []
        names = []

        for j in range(n_features):
            xj = X[:, j]
            cols.append(xj)
            names.append(f"x{j}")

            cols.append(xj * xj)
            names.append(f"x{j}^2")

            m = self.medians_[j]
            cols.append(np.maximum(0.0, xj - m))
            names.append(f"max(0, x{j} - {m:.3f})")

        # Add a small set of interactions between globally important features.
        for a in range(len(self.interaction_pairs_)):
            i, j = self.interaction_pairs_[a]
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}*x{j}")

        Phi = np.column_stack(cols) if cols else np.zeros((n_samples, 0))
        return Phi, names

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.medians_ = np.median(X, axis=0)

        # Pick candidate interaction pairs using absolute correlation with y.
        corrs = []
        y_centered = y - y.mean()
        y_norm = np.linalg.norm(y_centered) + 1e-12
        for j in range(n_features):
            xj = X[:, j] - X[:, j].mean()
            corr = float(np.dot(xj, y_centered) / ((np.linalg.norm(xj) + 1e-12) * y_norm))
            corrs.append(abs(corr))
        top_k = min(max(2, self.interaction_candidates), n_features)
        top_features = np.argsort(corrs)[-top_k:]

        pairs = []
        for ii in range(len(top_features)):
            for jj in range(ii + 1, len(top_features)):
                pairs.append((int(top_features[ii]), int(top_features[jj])))
        self.interaction_pairs_ = pairs

        Phi, term_names = self._feature_library(X)

        self.term_means_ = Phi.mean(axis=0) if Phi.shape[1] else np.array([])
        self.term_stds_ = Phi.std(axis=0) if Phi.shape[1] else np.array([])
        if Phi.shape[1]:
            self.term_stds_[self.term_stds_ < 1e-8] = 1.0
            Z = (Phi - self.term_means_) / self.term_stds_
        else:
            Z = Phi

        selector = LassoCV(cv=self.cv, random_state=self.random_state, n_jobs=None)
        selector.fit(Z, y)

        coef = selector.coef_.copy()
        nonzero = np.flatnonzero(np.abs(coef) > 1e-8)
        if len(nonzero) == 0:
            nonzero = np.array([int(np.argmax(np.abs(coef)))]) if len(coef) else np.array([], dtype=int)

        # Keep the largest coefficients for compactness.
        if len(nonzero) > self.max_active_terms:
            keep = np.argsort(np.abs(coef[nonzero]))[-self.max_active_terms :]
            nonzero = nonzero[np.sort(keep)]

        self.active_idx_ = np.array(nonzero, dtype=int)
        self.term_names_ = term_names

        if len(self.active_idx_) == 0:
            self.linear_model_ = None
            self.intercept_ = float(y.mean())
            self.active_coefs_ = np.array([])
            return self

        X_active = Phi[:, self.active_idx_]
        refit = LinearRegression()
        refit.fit(X_active, y)
        self.linear_model_ = refit
        self.intercept_ = float(refit.intercept_)
        self.active_coefs_ = refit.coef_.astype(float)
        return self

    def _transform_active(self, X):
        X = np.asarray(X, dtype=float)
        Phi, _ = self._feature_library(X)
        if len(self.active_idx_) == 0:
            return np.zeros((X.shape[0], 0))
        return Phi[:, self.active_idx_]

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "active_idx_", "intercept_"])
        X = np.asarray(X, dtype=float)
        if len(self.active_idx_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        X_active = self._transform_active(X)
        return self.linear_model_.predict(X_active)

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "active_idx_", "intercept_"])

        lines = [
            "Sparse Rule-Equation Regressor:",
            "prediction = intercept + sum(active term contribution)",
            f"intercept: {self.intercept_:.6f}",
            "",
            f"active_terms: {len(self.active_idx_)}",
            "equation:",
        ]

        if len(self.active_idx_) == 0:
            lines.append(f"  y = {self.intercept_:.6f}")
        else:
            expr = [f"{self.intercept_:.6f}"]
            for k, idx in enumerate(self.active_idx_):
                name = self.term_names_[idx]
                coef = self.active_coefs_[k]
                expr.append(f"({coef:+.6f})*{name}")
                lines.append(f"  term_{k+1}: coef={coef:+.6f}, feature={name}")
            lines.append("  y = " + " + ".join(expr))

        # Explicitly list dropped raw features to help sparse-feature tests.
        active_raw = set()
        for idx in self.active_idx_:
            term = self.term_names_[idx]
            for j in range(self.n_features_in_):
                if f"x{j}" in term:
                    active_raw.add(j)
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in active_raw]
        if inactive:
            lines.append("")
            lines.append("raw_features_with_negligible_effect: " + ", ".join(inactive))

        lines.append("")
        lines.append("how_to_compute_prediction_for_one_sample:")
        lines.append("  1) Evaluate each active feature expression.")
        lines.append("  2) Multiply by its coefficient.")
        lines.append("  3) Add all contributions and the intercept.")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseRuleEquationRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseRuleEq_v1"
model_description = "Sparse additive symbolic model with linear/quadratic/hinge terms plus selected pairwise interactions, L1 term selection, and OLS refit"
model_defs = [(model_shorthand_name, SparseRuleEquationRegressor())]


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
