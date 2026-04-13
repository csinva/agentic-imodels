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


class SparseResidualOneTermRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone with one optional nonlinear residual term.

    Procedure:
    1) Fit a ridge-regularized linear model with holdout-selected alpha.
    2) Hard-prune linear coefficients to a compact set of strongest features.
    3) Optionally add one nonlinear correction term (abs/relu/square/interaction)
       only if it improves holdout MSE by a meaningful margin.
    4) Refit a compact ridge model over active linear features + optional term.

    The final equation stays short and manually simulatable.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        max_linear_terms=8,
        nonlinear_top_features=3,
        min_rel_gain=0.015,
        min_abs_coef=1e-4,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.max_linear_terms = max_linear_terms
        self.nonlinear_top_features = nonlinear_top_features
        self.min_rel_gain = min_rel_gain
        self.min_abs_coef = min_abs_coef
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

    def _select_alpha(self, D_tr, y_tr, D_val, y_val):
        best = None
        for alpha in self.alpha_grid:
            b0, coef = self._ridge_fit(D_tr, y_tr, alpha)
            mse = float(np.mean((y_val - (b0 + D_val @ coef)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, coef)
        return best

    @staticmethod
    def _feature_screen(X, y):
        cy = y - np.mean(y)
        y_norm = float(np.sqrt(np.sum(cy ** 2))) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            cx = X[:, j] - np.mean(X[:, j])
            x_norm = float(np.sqrt(np.sum(cx ** 2))) + 1e-12
            scores[j] = abs(float(np.dot(cx, cy) / (x_norm * y_norm)))
        return np.argsort(scores)[::-1]

    @staticmethod
    def _term_values(X, term):
        t = term["type"]
        j = int(term["j"])
        if t == "abs":
            return np.abs(X[:, j])
        if t == "relu_pos":
            return np.maximum(0.0, X[:, j])
        if t == "relu_neg":
            return np.maximum(0.0, -X[:, j])
        if t == "sq":
            return X[:, j] ** 2
        if t == "int":
            return X[:, j] * X[:, int(term["k"])]
        raise ValueError(f"Unknown nonlinear term type: {t}")

    @staticmethod
    def _term_to_text(term):
        t = term["type"]
        j = int(term["j"])
        if t == "abs":
            return f"abs(x{j})"
        if t == "relu_pos":
            return f"max(0, x{j})"
        if t == "relu_neg":
            return f"max(0, -x{j})"
        if t == "sq":
            return f"(x{j}^2)"
        if t == "int":
            return f"(x{j}*x{int(term['k'])})"
        return "unknown_term"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Base linear model
        _, _, _, base_coef = self._select_alpha(X_tr, y_tr, X_val, y_val)

        # Prune to strongest linear terms for compactness/simulatability
        keep = np.argsort(np.abs(base_coef))[::-1][: min(int(self.max_linear_terms), p)]
        keep = np.array(sorted(keep.tolist()), dtype=int)

        Dtr_lin = X_tr[:, keep] if keep.size > 0 else np.zeros((len(X_tr), 0), dtype=float)
        Dval_lin = X_val[:, keep] if keep.size > 0 else np.zeros((len(X_val), 0), dtype=float)
        lin_best_mse, lin_alpha, _, _ = self._select_alpha(Dtr_lin, y_tr, Dval_lin, y_val)

        # Candidate nonlinear term set (very small, screened)
        screened = self._feature_screen(X_tr, y_tr)
        top = screened[: min(int(self.nonlinear_top_features), p)]

        candidates = []
        for j in top:
            jj = int(j)
            candidates.append({"type": "abs", "j": jj})
            candidates.append({"type": "relu_pos", "j": jj})
            candidates.append({"type": "relu_neg", "j": jj})
            candidates.append({"type": "sq", "j": jj})
        if len(top) >= 2:
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    candidates.append({"type": "int", "j": int(top[i]), "k": int(top[j])})

        # Evaluate one-term augmentation
        best_term = None
        best_term_mse = lin_best_mse
        best_term_alpha = lin_alpha
        for term in candidates:
            g_tr = self._term_values(X_tr, term).reshape(-1, 1)
            g_val = self._term_values(X_val, term).reshape(-1, 1)
            Dtr_aug = np.column_stack([Dtr_lin, g_tr])
            Dval_aug = np.column_stack([Dval_lin, g_val])
            mse, alpha, _, _ = self._select_alpha(Dtr_aug, y_tr, Dval_aug, y_val)
            if mse < best_term_mse:
                best_term_mse = mse
                best_term = term
                best_term_alpha = alpha

        rel_gain = (lin_best_mse - best_term_mse) / (lin_best_mse + 1e-12)
        self.nonlinear_term_ = best_term if (best_term is not None and rel_gain >= float(self.min_rel_gain)) else None

        # Final compact refit on full data
        Dall_lin = X[:, keep] if keep.size > 0 else np.zeros((n, 0), dtype=float)
        if self.nonlinear_term_ is not None:
            gall = self._term_values(X, self.nonlinear_term_).reshape(-1, 1)
            Dall = np.column_stack([Dall_lin, gall])
            final_alpha = best_term_alpha
        else:
            Dall = Dall_lin
            final_alpha = lin_alpha

        _, coef = self._ridge_fit(Dall, y, final_alpha)
        coef = np.asarray(coef, dtype=float)
        coef[np.abs(coef) < float(self.min_abs_coef)] = 0.0

        self.active_linear_idx_ = keep
        self.linear_coef_ = np.zeros(p, dtype=float)
        if keep.size > 0:
            self.linear_coef_[keep] = coef[: keep.size]

        self.nonlinear_coef_ = float(coef[-1]) if self.nonlinear_term_ is not None else 0.0
        self.intercept_ = float(np.mean(y - self._predict_no_check(X)))

        fi = np.abs(self.linear_coef_)
        if self.nonlinear_term_ is not None:
            w = abs(float(self.nonlinear_coef_))
            if self.nonlinear_term_["type"] == "int":
                fi[int(self.nonlinear_term_["j"])] += 0.5 * w
                fi[int(self.nonlinear_term_["k"])] += 0.5 * w
            else:
                fi[int(self.nonlinear_term_["j"])] += w
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def _predict_no_check(self, X):
        X = np.asarray(X, dtype=float)
        yhat = X @ self.linear_coef_
        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_coef_)) > 0:
            yhat = yhat + float(self.nonlinear_coef_) * self._term_values(X, self.nonlinear_term_)
        return yhat

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        return float(self.intercept_) + self._predict_no_check(X)

    def __str__(self):
        check_is_fitted(self, ["linear_coef_", "intercept_", "feature_importance_", "n_features_in_"])
        lines = ["Sparse Residual One-Term Regressor", "equation:"]

        terms = [f"{float(self.intercept_):+.6f}"]
        nz = np.where(np.abs(self.linear_coef_) > 0)[0]
        for j in nz:
            terms.append(f"({float(self.linear_coef_[j]):+.6f})*x{int(j)}")
        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_coef_)) > 0:
            terms.append(f"({float(self.nonlinear_coef_):+.6f})*{self._term_to_text(self.nonlinear_term_)}")

        lines.append("  y = " + " + ".join(terms))
        lines.append("")
        lines.append("active_terms:")
        shown = 0
        for j in nz:
            lines.append(f"  ({float(self.linear_coef_[j]):+.6f}) * x{int(j)}")
            shown += 1
        if self.nonlinear_term_ is not None and abs(float(self.nonlinear_coef_)) > 0:
            lines.append(f"  ({float(self.nonlinear_coef_):+.6f}) * {self._term_to_text(self.nonlinear_term_)}")
            shown += 1
        if shown == 0:
            lines.append("  none")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        count = 0
        for j in order:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{int(j)}: {float(self.feature_importance_[j]):.4f}")
            count += 1
            if count >= 12:
                break
        if count == 0:
            lines.append("  all features have zero contribution")

        lines.append("")
        lines.append("manual_prediction_steps:")
        lines.append("  1) start from intercept")
        lines.append("  2) add each listed active term value times its coefficient")
        lines.append("  3) features not listed in active terms contribute 0")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseResidualOneTermRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseResidualOneTerm_v1"
model_description = "Sparse ridge linear backbone with hard-pruned active features plus one validation-gated nonlinear residual term (abs/relu/square/interaction)"
model_defs = [(model_shorthand_name, SparseResidualOneTermRegressor())]


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
