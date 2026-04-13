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


class HierarchicalSparseHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive model over raw features:
      y = b + sum c_k * phi_k(x)
    where phi_k are linear terms, one-sided hinges, and (optionally) one
    screened pairwise interaction.
    """

    def __init__(
        self,
        max_terms=7,
        top_features_for_hinges=6,
        hinge_quantiles=(0.25, 0.5, 0.75),
        allow_interaction=True,
        penalty_strength=0.60,
        min_gain=1e-4,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.top_features_for_hinges = top_features_for_hinges
        self.hinge_quantiles = hinge_quantiles
        self.allow_interaction = allow_interaction
        self.penalty_strength = penalty_strength
        self.min_gain = min_gain
        self.random_state = random_state

    def _safe_corr_rank(self, X, y):
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xc = X[:, j] - X[:, j].mean()
            x_norm = np.linalg.norm(xc) + 1e-12
            scores[j] = abs(float(np.dot(xc, yc) / (x_norm * y_norm)))
        return np.argsort(scores)[::-1], scores

    def _eval_term(self, X, term):
        t_type, feat, thr = term
        if t_type == "linear":
            return X[:, feat]
        if t_type == "hinge_pos":
            return np.maximum(0.0, X[:, feat] - thr)
        if t_type == "hinge_neg":
            return np.maximum(0.0, thr - X[:, feat])
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
        i, j = feat
        return f"(x{i} * x{j})"

    def _build_dictionary(self, X, y):
        p = X.shape[1]
        terms = []
        cols = []

        # Global linear backbone.
        for j in range(p):
            terms.append(("linear", int(j), None))
            cols.append(X[:, j])

        ranked, _ = self._safe_corr_rank(X, y)
        top = ranked[: min(self.top_features_for_hinges, p)]
        for j in top:
            xj = X[:, j]
            knots = np.unique(np.quantile(xj, self.hinge_quantiles))
            for t in knots:
                terms.append(("hinge_pos", int(j), float(t)))
                cols.append(np.maximum(0.0, xj - t))
                terms.append(("hinge_neg", int(j), float(t)))
                cols.append(np.maximum(0.0, t - xj))

        if self.allow_interaction and p >= 2:
            i, j = int(ranked[0]), int(ranked[1])
            terms.append(("interaction", (i, j), None))
            cols.append(X[:, i] * X[:, j])

        Phi = np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=float)
        norms = np.linalg.norm(Phi, axis=0)
        valid = norms > 1e-12
        return Phi[:, valid], [terms[k] for k in np.where(valid)[0]]

    def _objective(self, y_true, y_pred, n_terms):
        n = max(1, y_true.shape[0])
        mse = float(np.mean((y_true - y_pred) ** 2))
        y_scale = float(np.var(y_true) + 1e-12)
        penalty = self.penalty_strength * (n_terms * np.log(n + 1.0) / n) * y_scale
        return mse + penalty

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        Phi, terms = self._build_dictionary(X, y)
        selected = []
        y_hat = np.full(X.shape[0], float(y.mean()), dtype=float)
        best_obj = self._objective(y, y_hat, n_terms=0)

        for _ in range(min(self.max_terms, Phi.shape[1])):
            candidate_best_obj = np.inf
            candidate_best_idx = None
            for idx in range(Phi.shape[1]):
                if idx in selected:
                    continue
                cand = selected + [idx]
                A = np.column_stack([np.ones(X.shape[0]), Phi[:, cand]])
                beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ beta
                obj = self._objective(y, pred, n_terms=len(cand))
                if obj < candidate_best_obj:
                    candidate_best_obj = obj
                    candidate_best_idx = idx

            if candidate_best_idx is None:
                break

            rel_gain = (best_obj - candidate_best_obj) / (abs(best_obj) + 1e-12)
            if rel_gain < self.min_gain:
                break

            selected.append(candidate_best_idx)
            A = np.column_stack([np.ones(X.shape[0]), Phi[:, selected]])
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            y_hat = A @ beta
            best_obj = candidate_best_obj

        if selected:
            A = np.column_stack([np.ones(X.shape[0]), Phi[:, selected]])
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self.intercept_ = float(beta[0])
            self.selected_terms_ = [terms[i] for i in selected]
            self.coef_ = beta[1:].astype(float)
        else:
            self.intercept_ = float(y.mean())
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)

        keep = np.abs(self.coef_) > 1e-8
        self.coef_ = self.coef_[keep]
        self.selected_terms_ = [t for t, is_keep in zip(self.selected_terms_, keep) if is_keep]
        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "selected_terms_", "coef_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, term in zip(self.coef_, self.selected_terms_):
            pred += c * self._eval_term(X, term)
        return pred

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "selected_terms_", "coef_"])
        lines = [
            "Hierarchical Sparse Hinge Regressor",
            f"active_terms: {len(self.selected_terms_)}",
            "",
            "equation:",
        ]

        if len(self.selected_terms_) == 0:
            lines.append(f"  y = {self.intercept_:+.6f}")
        else:
            eq = [f"{self.intercept_:+.6f}"]
            for c, term in zip(self.coef_, self.selected_terms_):
                eq.append(f"({c:+.6f})*{self._term_str(term)}")
            lines.append("  y = " + " + ".join(eq))

        strength = np.zeros(self.n_features_in_, dtype=float)
        for c, term in zip(self.coef_, self.selected_terms_):
            t_type, feat, _ = term
            mag = abs(float(c))
            if t_type == "interaction":
                i, j = feat
                strength[i] += 0.7 * mag
                strength[j] += 0.7 * mag
            else:
                strength[int(feat)] += mag

        order = np.argsort(strength)[::-1]
        lines.append("")
        lines.append("feature_strength_ranking:")
        for idx in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{idx}: {strength[idx]:.6f}")

        weak = [f"x{j}" for j in range(self.n_features_in_) if strength[j] < 0.05]
        if weak:
            lines.append("")
            lines.append("features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("manual_prediction:")
        lines.append("  1) Compute each listed term from raw x-values.")
        lines.append("  2) Multiply by coefficient, then sum with intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HierarchicalSparseHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HierHingeSparse_v1"
model_description = "BIC-penalized sparse additive regressor over raw linear and hinge terms with one screened interaction and concise explicit equation"
model_defs = [(model_shorthand_name, HierarchicalSparseHingeRegressor())]


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
