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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ResidualStumpLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone + short additive threshold-rule residual correction.

    Stage 1: sparse linear model (LassoCV) for global trend.
    Stage 2: greedily add one-feature threshold rules on residuals:
      contrib_k(x) = coeff_k * (I[x_j <= threshold_k] - coverage_k)
    """

    def __init__(
        self,
        max_linear_terms=8,
        max_rules=5,
        threshold_grid_size=5,
        min_rule_coverage=0.08,
        cv=3,
        random_state=42,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_rules = max_rules
        self.threshold_grid_size = threshold_grid_size
        self.min_rule_coverage = min_rule_coverage
        self.cv = cv
        self.random_state = random_state

    def _fit_sparse_linear(self, X, y):
        self.x_means_ = X.mean(axis=0)
        self.x_stds_ = X.std(axis=0)
        self.x_stds_[self.x_stds_ < 1e-8] = 1.0
        Z = (X - self.x_means_) / self.x_stds_

        lasso = LassoCV(cv=self.cv, random_state=self.random_state, n_jobs=None)
        lasso.fit(Z, y)

        coefs = lasso.coef_.copy()
        nz = np.flatnonzero(np.abs(coefs) > 1e-8)
        if len(nz) == 0 and len(coefs) > 0:
            nz = np.array([int(np.argmax(np.abs(coefs)))], dtype=int)
        if len(nz) > self.max_linear_terms:
            keep = np.argsort(np.abs(coefs[nz]))[-self.max_linear_terms :]
            nz = nz[np.sort(keep)]

        self.linear_idx_ = np.array(nz, dtype=int)
        self.linear_coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = float(y.mean())

        if len(self.linear_idx_) > 0:
            Z_active = Z[:, self.linear_idx_]
            Xa = np.column_stack([np.ones(Z_active.shape[0]), Z_active])
            sol, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
            self.intercept_ = float(sol[0])
            self.linear_coef_[self.linear_idx_] = sol[1:]

    def _predict_linear(self, X):
        Z = (X - self.x_means_) / self.x_stds_
        return self.intercept_ + Z @ self.linear_coef_

    def _candidate_thresholds(self, x):
        quantiles = np.linspace(0.1, 0.9, self.threshold_grid_size)
        vals = np.quantile(x, quantiles)
        return np.unique(vals)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self._fit_sparse_linear(X, y)
        residual = y - self._predict_linear(X)

        self.rules_ = []
        for _ in range(self.max_rules):
            best = None
            best_gain = 0.0
            base_sse = float(np.dot(residual, residual))
            if base_sse <= 1e-12:
                break

            for j in range(self.n_features_in_):
                xj = X[:, j]
                thresholds = self._candidate_thresholds(xj)
                for t in thresholds:
                    mask = xj <= t
                    p = float(mask.mean())
                    if p < self.min_rule_coverage or p > 1.0 - self.min_rule_coverage:
                        continue

                    centered = mask.astype(float) - p
                    denom = float(np.dot(centered, centered))
                    if denom <= 1e-12:
                        continue
                    coeff = float(np.dot(residual, centered) / denom)
                    if abs(coeff) < 1e-10:
                        continue

                    new_resid = residual - coeff * centered
                    gain = base_sse - float(np.dot(new_resid, new_resid))
                    if gain > best_gain:
                        best_gain = gain
                        best = (j, float(t), p, coeff, centered)

            if best is None or best_gain <= 1e-8:
                break

            j, t, p, coeff, centered = best
            self.rules_.append({
                "feature": int(j),
                "threshold": float(t),
                "coverage": float(p),
                "coef": float(coeff),
            })
            residual = residual - coeff * centered

        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "rules_"])
        X = np.asarray(X, dtype=float)
        pred = self._predict_linear(X)
        for rule in self.rules_:
            j = rule["feature"]
            t = rule["threshold"]
            p = rule["coverage"]
            c = rule["coef"]
            pred = pred + c * ((X[:, j] <= t).astype(float) - p)
        return pred

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "linear_coef_", "rules_"])
        lines = [
            "Residual-Stump Linear Regressor",
            "prediction = intercept + linear_terms + threshold_rule_corrections",
            f"intercept: {self.intercept_:+.6f}",
            "",
            "linear_terms (on standardized inputs z_j=(x_j-mean_j)/std_j):",
        ]

        active_linear = []
        for j in range(self.n_features_in_):
            coef = float(self.linear_coef_[j])
            if abs(coef) > 1e-8:
                active_linear.append(j)
                lines.append(
                    f"  ({coef:+.6f}) * z{j}    [mean={self.x_means_[j]:+.6f}, std={self.x_stds_[j]:.6f}]"
                )
        if not active_linear:
            lines.append("  none")

        lines.append("")
        lines.append("threshold_rule_corrections:")
        if not self.rules_:
            lines.append("  none")
        else:
            for k, rule in enumerate(self.rules_, 1):
                j = rule["feature"]
                t = rule["threshold"]
                p = rule["coverage"]
                c = rule["coef"]
                lines.append(
                    f"  rule_{k}: ({c:+.6f}) * (I[x{j} <= {t:+.6f}] - {p:.4f})"
                )

        linear_strength = np.abs(self.linear_coef_)
        rule_strength = np.zeros(self.n_features_in_, dtype=float)
        for r in self.rules_:
            rule_strength[r["feature"]] += abs(r["coef"])
        total_strength = linear_strength + rule_strength
        order = np.argsort(total_strength)[::-1]
        lines.append("")
        lines.append("feature_strength_ranking:")
        for idx in order[: min(8, self.n_features_in_)]:
            lines.append(f"  x{idx}: {total_strength[idx]:.6f}")

        weak = [f"x{j}" for j in range(self.n_features_in_) if total_strength[j] < 0.05]
        if weak:
            lines.append("")
            lines.append("raw_features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) Compute z_j for each active linear term and sum linear contributions.")
        lines.append("  2) For each rule, evaluate indicator I[x_j <= threshold], subtract coverage, multiply by rule coef.")
        lines.append("  3) Add intercept + linear sum + all rule corrections.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualStumpLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidStumpLinear_v1"
model_description = "Sparse standardized linear backbone plus greedy residual one-feature threshold rules (centered indicators) for compact nonlinear corrections"
model_defs = [(model_shorthand_name, ResidualStumpLinearRegressor())]


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
