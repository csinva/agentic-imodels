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


class StagewiseHingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive model built from raw linear and one-knot hinge terms.

    We first build a library of candidate terms:
      - linear: x_j
      - right hinge: max(0, x_j - t)
      - left hinge:  max(0, t - x_j)
    Then we pick a small subset via forward stagewise residual fitting and
    refit a single ridge-regularized linear model on those selected terms.
    """

    def __init__(
        self,
        max_terms=12,
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        learning_rate=0.35,
        min_abs_correlation=0.02,
        ridge_alpha=0.2,
        min_term_coef=1e-4,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.knot_quantiles = knot_quantiles
        self.learning_rate = learning_rate
        self.min_abs_correlation = min_abs_correlation
        self.ridge_alpha = ridge_alpha
        self.min_term_coef = min_term_coef
        self.random_state = random_state

    @staticmethod
    def _term_values(X, term):
        kind = term["kind"]
        j = term["feature"]
        xj = X[:, j]
        if kind == "lin":
            return xj
        t = term["threshold"]
        if kind == "hinge_pos":
            return np.maximum(0.0, xj - t)
        return np.maximum(0.0, t - xj)

    def _build_candidate_library(self, X):
        n, p = X.shape
        cols = []
        terms = []
        for j in range(p):
            xj = X[:, j]
            cols.append(xj)
            terms.append({"kind": "lin", "feature": j, "threshold": None})

            thresholds = []
            for q in self.knot_quantiles:
                try:
                    thresholds.append(float(np.quantile(xj, q)))
                except Exception:
                    continue
            thresholds = sorted(set(thresholds))
            for t in thresholds:
                cols.append(np.maximum(0.0, xj - t))
                terms.append({"kind": "hinge_pos", "feature": j, "threshold": t})
                cols.append(np.maximum(0.0, t - xj))
                terms.append({"kind": "hinge_neg", "feature": j, "threshold": t})

        B = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
        return B, terms

    @staticmethod
    def _ridge_fit_with_intercept(D, y, alpha):
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(D.shape[1]) * alpha
        pen[0, 0] = 0.0  # do not penalize intercept
        try:
            return np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram + pen) @ rhs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        B, term_defs = self._build_candidate_library(X)
        if B.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.linear_coef_ = np.zeros(p, dtype=float)
            self.hinge_terms_ = []
            self.selected_terms_ = []
            self.feature_importance_ = np.zeros(p, dtype=float)
            return self

        # Normalize candidate terms for fair stagewise screening.
        B_center = B - B.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.sum(B_center ** 2, axis=0)) + 1e-12
        B_norm = B_center / norms

        residual = y - float(np.mean(y))
        chosen = []
        max_terms = min(int(self.max_terms), B.shape[1])
        for _ in range(max_terms):
            corr = (B_norm.T @ residual) / max(n, 1)
            if chosen:
                corr[np.array(chosen, dtype=int)] = 0.0
            best = int(np.argmax(np.abs(corr)))
            best_corr = float(corr[best])
            if abs(best_corr) < float(self.min_abs_correlation):
                break
            chosen.append(best)
            residual = residual - float(self.learning_rate) * best_corr * B_norm[:, best]

        if not chosen:
            corr0 = np.abs((B_norm.T @ residual) / max(n, 1))
            chosen = [int(np.argmax(corr0))]

        D_sel = B[:, chosen]
        D = np.column_stack([np.ones(n), D_sel])
        beta = self._ridge_fit_with_intercept(D, y, alpha=float(self.ridge_alpha))

        intercept = float(beta[0])
        coefs_sel = np.asarray(beta[1:], dtype=float)

        # Drop tiny coefficients for readability.
        keep = np.where(np.abs(coefs_sel) >= float(self.min_term_coef))[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(coefs_sel)))], dtype=int)

        chosen_kept = [chosen[i] for i in keep]
        coefs_kept = coefs_sel[keep]

        linear_coef = np.zeros(p, dtype=float)
        hinge_terms = []
        for idx, coef in zip(chosen_kept, coefs_kept):
            term = term_defs[idx]
            if term["kind"] == "lin":
                linear_coef[term["feature"]] += float(coef)
            else:
                hinge_terms.append(
                    {
                        "kind": term["kind"],
                        "feature": int(term["feature"]),
                        "threshold": float(term["threshold"]),
                        "coef": float(coef),
                    }
                )

        self.intercept_ = intercept
        self.linear_coef_ = linear_coef
        self.hinge_terms_ = hinge_terms
        self.selected_terms_ = [
            {**term_defs[idx], "coef": float(coef)}
            for idx, coef in zip(chosen_kept, coefs_kept)
        ]

        importance = np.abs(linear_coef)
        for t in hinge_terms:
            importance[t["feature"]] += abs(t["coef"])
        self.feature_importance_ = importance
        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "hinge_terms_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        yhat += X @ self.linear_coef_
        for t in self.hinge_terms_:
            j = t["feature"]
            if t["kind"] == "hinge_pos":
                yhat += t["coef"] * np.maximum(0.0, X[:, j] - t["threshold"])
            else:
                yhat += t["coef"] * np.maximum(0.0, t["threshold"] - X[:, j])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "hinge_terms_", "feature_importance_"])
        lines = [
            "Stagewise Hinge Additive Regressor",
            f"active_terms: {len(self.selected_terms_)}",
            "",
            "equation:",
        ]

        pieces = [f"{self.intercept_:+.6f}"]
        for j, c in enumerate(self.linear_coef_):
            if abs(c) >= float(self.min_term_coef):
                pieces.append(f"({c:+.6f})*x{j}")
        for t in self.hinge_terms_:
            c = t["coef"]
            j = t["feature"]
            thr = t["threshold"]
            if t["kind"] == "hinge_pos":
                pieces.append(f"({c:+.6f})*max(0, x{j}-{thr:.4f})")
            else:
                pieces.append(f"({c:+.6f})*max(0, {thr:.4f}-x{j})")
        lines.append("  y = " + " + ".join(pieces))

        lines.append("")
        lines.append("feature_effect_ranking:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(10, self.n_features_in_)]:
            lines.append(
                f"  x{j}: total_effect={self.feature_importance_[j]:.6f}, linear_coef={self.linear_coef_[j]:+.6f}"
            )

        weak = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < 0.03]
        if weak:
            lines.append("")
            lines.append("features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("manual_prediction:")
        lines.append("  1) Start from intercept.")
        lines.append("  2) Add each linear term coef*xj.")
        lines.append("  3) Add hinge terms using max(0, .) exactly as written.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StagewiseHingeAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StageHingeAdd_v1"
model_description = "Forward-stagewise additive regressor over raw linear and single-knot hinge terms with compact ridge-refit explicit equation"
model_defs = [(model_shorthand_name, StagewiseHingeAdditiveRegressor())]


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
