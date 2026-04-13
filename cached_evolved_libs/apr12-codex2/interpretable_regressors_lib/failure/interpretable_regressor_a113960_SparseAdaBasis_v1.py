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
from itertools import combinations
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


class SparseAdaptiveBasisRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive basis model built from raw-feature primitives:
    - linear terms: x_j
    - hinge terms: max(0, x_j - t), max(0, t - x_j)
    - squared terms: x_j^2
    - screened pair interactions: x_i * x_j

    Terms are selected with a greedy OMP-style procedure and then refit by OLS.
    """

    def __init__(
        self,
        max_terms=8,
        threshold_grid_size=4,
        interaction_screen=8,
        min_gain=1e-4,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.threshold_grid_size = threshold_grid_size
        self.interaction_screen = interaction_screen
        self.min_gain = min_gain
        self.random_state = random_state

    def _safe_corr_rank(self, X, y):
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc) + 1e-12
        scores = []
        for j in range(X.shape[1]):
            x = X[:, j]
            xc = x - x.mean()
            score = abs(float(np.dot(xc, yc) / ((np.linalg.norm(xc) + 1e-12) * y_norm)))
            scores.append(score)
        return np.argsort(scores)[::-1]

    def _build_dictionary(self, X, y):
        n, p = X.shape
        terms = []
        cols = []

        for j in range(p):
            cols.append(X[:, j])
            terms.append(("linear", int(j), None))

        for j in range(p):
            xj = X[:, j]
            q = np.linspace(0.15, 0.85, self.threshold_grid_size)
            for t in np.unique(np.quantile(xj, q)):
                cols.append(np.maximum(0.0, xj - t))
                terms.append(("hinge_pos", int(j), float(t)))
                cols.append(np.maximum(0.0, t - xj))
                terms.append(("hinge_neg", int(j), float(t)))

        for j in range(p):
            cols.append(X[:, j] ** 2)
            terms.append(("square", int(j), None))

        ranked = self._safe_corr_rank(X, y)
        top = ranked[: min(self.interaction_screen, p)]
        for i, j in combinations(top, 2):
            cols.append(X[:, i] * X[:, j])
            terms.append(("interaction", (int(i), int(j)), None))

        Phi = np.column_stack(cols)
        Phi_means = Phi.mean(axis=0)
        Phi_centered = Phi - Phi_means
        norms = np.linalg.norm(Phi_centered, axis=0)
        valid = norms > 1e-10
        return Phi_centered[:, valid], Phi_means[valid], norms[valid], [terms[k] for k in np.where(valid)[0]]

    def _eval_term(self, X, term):
        t_type, feat, thr = term
        if t_type == "linear":
            return X[:, feat]
        if t_type == "hinge_pos":
            return np.maximum(0.0, X[:, feat] - thr)
        if t_type == "hinge_neg":
            return np.maximum(0.0, thr - X[:, feat])
        if t_type == "square":
            return X[:, feat] ** 2
        i, j = feat
        return X[:, i] * X[:, j]

    def _term_str(self, term):
        t_type, feat, thr = term
        if t_type == "linear":
            return f"x{feat}"
        if t_type == "hinge_pos":
            return f"max(0, x{feat} - {thr:+.4f})"
        if t_type == "hinge_neg":
            return f"max(0, {thr:+.4f} - x{feat})"
        if t_type == "square":
            return f"(x{feat})^2"
        i, j = feat
        return f"x{i}*x{j}"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        Phi, phi_means, norms, terms = self._build_dictionary(X, y)
        selected = []
        y_mean = float(y.mean())
        residual = y - y_mean
        prev_sse = float(np.dot(residual, residual))

        for _ in range(min(self.max_terms, Phi.shape[1])):
            corr = Phi.T @ residual
            if selected:
                corr[np.array(selected, dtype=int)] = 0.0
            scores = np.abs(corr) / (norms + 1e-12)
            k = int(np.argmax(scores))
            if scores[k] <= 1e-10:
                break
            cand = selected + [k]
            A = np.column_stack([np.ones(X.shape[0]), Phi[:, cand]])
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            new_resid = y - A @ beta
            new_sse = float(np.dot(new_resid, new_resid))
            gain = (prev_sse - new_sse) / (prev_sse + 1e-12)
            if gain < self.min_gain and len(selected) > 0:
                break
            selected = cand
            residual = new_resid
            prev_sse = new_sse

        if selected:
            A = np.column_stack([np.ones(X.shape[0]), Phi[:, selected]])
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self.intercept_ = float(beta[0])
            self.selected_terms_ = [terms[i] for i in selected]
            self.coef_ = beta[1:].astype(float)
            self.selected_phi_means_ = phi_means[np.array(selected, dtype=int)].astype(float)
        else:
            self.intercept_ = float(y.mean())
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.selected_phi_means_ = np.zeros(0, dtype=float)

        keep = np.abs(self.coef_) > 1e-8
        self.selected_terms_ = [t for t, k in zip(self.selected_terms_, keep) if k]
        self.selected_phi_means_ = self.selected_phi_means_[keep]
        self.coef_ = self.coef_[keep]

        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "selected_terms_", "coef_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, term, mu in zip(self.coef_, self.selected_terms_, self.selected_phi_means_):
            pred += c * (self._eval_term(X, term) - mu)
        return pred

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "selected_terms_", "coef_"])
        lines = [
            "Sparse Adaptive Basis Regressor",
            "prediction = intercept + sum_k coef_k * (basis_k(x) - mean_k)",
            f"intercept = {self.intercept_:+.6f}",
            "",
            "selected_basis_terms:",
        ]

        if len(self.selected_terms_) == 0:
            lines.append("  none")
        else:
            for k, (c, term, mu) in enumerate(zip(self.coef_, self.selected_terms_, self.selected_phi_means_), 1):
                lines.append(
                    f"  term_{k}: ({c:+.6f}) * ({self._term_str(term)} - {mu:+.6f})"
                )

        strength = np.zeros(self.n_features_in_, dtype=float)
        for c, term in zip(self.coef_, self.selected_terms_):
            t_type, feat, _ = term
            mag = abs(float(c))
            if t_type == "interaction":
                i, j = feat
                strength[i] += 0.6 * mag
                strength[j] += 0.6 * mag
            else:
                strength[int(feat)] += mag

        order = np.argsort(strength)[::-1]
        lines.append("")
        lines.append("feature_strength_ranking:")
        for idx in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{idx}: {strength[idx]:.6f}")

        weak = [f"x{j}" for j in range(self.n_features_in_) if strength[j] < 0.08]
        if weak:
            lines.append("")
            lines.append("raw_features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("compact_equation:")
        if len(self.selected_terms_) == 0:
            lines.append(f"  y = {self.intercept_:+.6f}")
        else:
            eq = [f"{self.intercept_:+.6f}"]
            for c, term, mu in zip(self.coef_, self.selected_terms_, self.selected_phi_means_):
                eq.append(f"({c:+.6f})*({self._term_str(term)} - {mu:+.6f})")
            lines.append("  y = " + " + ".join(eq))

        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) Evaluate each listed basis term using the raw x-values.")
        lines.append("  2) Subtract that term's listed mean, then multiply by its coefficient.")
        lines.append("  3) Sum all contributions with the intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdaptiveBasisRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdaBasis_v1"
model_description = "Greedy sparse additive basis model over raw features using linear, hinge, square, and screened interaction terms with OLS refit"
model_defs = [(model_shorthand_name, SparseAdaptiveBasisRegressor())]


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
