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
from itertools import combinations
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


class SparseSymbolicResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Forward-selected sparse symbolic equation over a compact basis.

    Basis atoms:
      linear:          xj
      positive hinge:  max(0, xj)
      negative hinge:  max(0, -xj)
      square:          xj^2          (screened subset only)
      interaction:     xj * xk       (screened subset only)
    """

    def __init__(
        self,
        val_frac=0.2,
        max_terms=8,
        min_gain=1e-4,
        alpha_grid=(0.0, 1e-3, 1e-2, 1e-1, 1.0),
        max_square_features=6,
        max_interaction_features=5,
        min_abs_coef=1e-5,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.max_terms = max_terms
        self.min_gain = min_gain
        self.alpha_grid = alpha_grid
        self.max_square_features = max_square_features
        self.max_interaction_features = max_interaction_features
        self.min_abs_coef = min_abs_coef
        self.random_state = random_state

    def _make_split(self, n):
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = int(max(24, round(float(self.val_frac) * n)))
        n_val = min(max(1, n_val), max(1, n - 1))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        return tr_idx, val_idx

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

    def _fit_alpha(self, D_tr, y_tr, D_val, y_val):
        best = None
        for alpha in self.alpha_grid:
            b0, coef = self._ridge_fit(D_tr, y_tr, alpha)
            pred = b0 + D_val @ coef
            mse = float(np.mean((y_val - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, coef)
        return best

    @staticmethod
    def _screen_feature_order(X, y):
        centered_y = y - np.mean(y)
        scores = np.zeros(X.shape[1], dtype=float)
        denom_y = float(np.sqrt(np.sum(centered_y ** 2))) + 1e-12
        for j in range(X.shape[1]):
            x = X[:, j]
            cx = x - np.mean(x)
            denom_x = float(np.sqrt(np.sum(cx ** 2))) + 1e-12
            scores[j] = abs(float(np.dot(cx, centered_y) / (denom_x * denom_y)))
        return np.argsort(scores)[::-1]

    @staticmethod
    def _term_values(X, term):
        t = term["type"]
        j = term["j"]
        if t == "lin":
            return X[:, j]
        if t == "relu_pos":
            return np.maximum(0.0, X[:, j])
        if t == "relu_neg":
            return np.maximum(0.0, -X[:, j])
        if t == "sq":
            return X[:, j] ** 2
        if t == "int":
            return X[:, j] * X[:, term["k"]]
        raise ValueError(f"Unknown term type: {t}")

    def _build_candidate_terms(self, X, y):
        n, p = X.shape
        _ = n
        terms = []
        for j in range(p):
            terms.append({"type": "lin", "j": int(j)})
            terms.append({"type": "relu_pos", "j": int(j)})
            terms.append({"type": "relu_neg", "j": int(j)})

        order = self._screen_feature_order(X, y)
        square_feats = order[: min(int(self.max_square_features), p)]
        for j in square_feats:
            terms.append({"type": "sq", "j": int(j)})

        inter_feats = order[: min(int(self.max_interaction_features), p)]
        for j, k in combinations(inter_feats.tolist(), 2):
            terms.append({"type": "int", "j": int(j), "k": int(k)})
        return terms

    def _design_matrix(self, X, terms):
        if len(terms) == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._term_values(X, term) for term in terms]
        return np.column_stack(cols).astype(float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        candidate_terms = self._build_candidate_terms(X_tr, y_tr)
        remaining = list(range(len(candidate_terms)))
        selected = []
        selected_best_mse = float(np.mean((y_val - np.mean(y_tr)) ** 2))

        for _ in range(int(max(0, self.max_terms))):
            best_step = None
            for idx in remaining:
                trial_terms = selected + [idx]
                D_tr = self._design_matrix(X_tr, [candidate_terms[t] for t in trial_terms])
                D_val = self._design_matrix(X_val, [candidate_terms[t] for t in trial_terms])
                mse, alpha, b0, coef = self._fit_alpha(D_tr, y_tr, D_val, y_val)
                gain = selected_best_mse - mse
                if best_step is None or gain > best_step["gain"]:
                    best_step = {
                        "idx": idx,
                        "mse": mse,
                        "gain": gain,
                        "alpha": alpha,
                        "b0": b0,
                        "coef": coef,
                    }
            if best_step is None or best_step["gain"] < float(self.min_gain):
                break
            selected.append(best_step["idx"])
            remaining.remove(best_step["idx"])
            selected_best_mse = best_step["mse"]

        self.terms_ = [candidate_terms[t] for t in selected]
        D_all = self._design_matrix(X, self.terms_)
        D_tr = self._design_matrix(X_tr, self.terms_)
        D_val = self._design_matrix(X_val, self.terms_)
        _, self.alpha_, self.intercept_, coef = self._fit_alpha(D_tr, y_tr, D_val, y_val)
        self.coef_ = np.asarray(coef, dtype=float)
        self.coef_[np.abs(self.coef_) < float(self.min_abs_coef)] = 0.0

        if D_all.shape[1] > 0:
            self.intercept_ = float(np.mean(y - D_all @ self.coef_))
        else:
            self.intercept_ = float(np.mean(y))

        self.feature_importance_ = np.zeros(p, dtype=float)
        for c, term in zip(self.coef_, self.terms_):
            w = abs(float(c))
            if term["type"] == "int":
                self.feature_importance_[term["j"]] += 0.5 * w
                self.feature_importance_[term["k"]] += 0.5 * w
            else:
                self.feature_importance_[term["j"]] += w
        total = float(np.sum(self.feature_importance_))
        if total > 0:
            self.feature_importance_ /= total
        return self

    def predict(self, X):
        check_is_fitted(self, ["terms_", "coef_", "intercept_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        D = self._design_matrix(X, self.terms_)
        return float(self.intercept_) + D @ self.coef_

    @staticmethod
    def _term_to_text(term):
        t = term["type"]
        j = int(term["j"])
        if t == "lin":
            return f"x{j}"
        if t == "relu_pos":
            return f"max(0, x{j})"
        if t == "relu_neg":
            return f"max(0, -x{j})"
        if t == "sq":
            return f"(x{j}^2)"
        if t == "int":
            k = int(term["k"])
            return f"(x{j}*x{k})"
        return "unknown_term"

    def __str__(self):
        check_is_fitted(self, ["terms_", "coef_", "intercept_", "feature_importance_"])
        lines = ["Sparse Symbolic Residual Regressor", "equation:"]
        if len(self.terms_) == 0:
            lines.append(f"  y = {float(self.intercept_):+.6f}")
        else:
            pieces = [f"{float(self.intercept_):+.6f}"]
            for c, term in zip(self.coef_, self.terms_):
                if abs(float(c)) <= 0.0:
                    continue
                pieces.append(f"({float(c):+.6f})*{self._term_to_text(term)}")
            lines.append("  y = " + " + ".join(pieces))

        lines.append("")
        lines.append("active_terms:")
        shown = 0
        for c, term in sorted(
            zip(self.coef_, self.terms_),
            key=lambda x: abs(float(x[0])),
            reverse=True,
        ):
            if abs(float(c)) <= 0.0:
                continue
            lines.append(f"  ({float(c):+.6f}) * {self._term_to_text(term)}")
            shown += 1
        if shown == 0:
            lines.append("  none")

        lines.append("")
        lines.append("feature_importance_order:")
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{int(j)}: {float(self.feature_importance_[j]):.4f}")
        if np.sum(self.feature_importance_) == 0:
            lines.append("  all features have zero contribution")

        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) start from intercept")
        lines.append("  2) add each listed term value multiplied by its coefficient")
        lines.append("  3) all unlisted terms/features contribute 0")
        return "\n".join(lines)

# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseSymbolicResidualRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseSymbolicResidual_v1"
model_description = "Forward-selected sparse symbolic equation over linear/ReLU/square/interaction atoms with validation-gated compactness and ridge refit"
model_defs = [(model_shorthand_name, SparseSymbolicResidualRegressor())]


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
