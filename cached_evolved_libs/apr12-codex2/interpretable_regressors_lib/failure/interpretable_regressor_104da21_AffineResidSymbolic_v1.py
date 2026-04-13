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


class AffineResidualSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Affine backbone plus a tiny residual symbolic basis.

    Stage 1: fit a full linear equation y = b + sum_j w_j * x_j with validation-
    selected ridge regularization.
    Stage 2: fit up to a few residual terms (square / hinge / interaction) chosen
    by greedy validation gain and refit a single compact closed-form equation.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        val_frac=0.2,
        top_features_for_residual_library=8,
        interaction_top_features=5,
        max_residual_terms=3,
        min_gain=5e-4,
        final_ridge_alpha=1e-3,
        min_coef_for_display=1e-4,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.top_features_for_residual_library = top_features_for_residual_library
        self.interaction_top_features = interaction_top_features
        self.max_residual_terms = max_residual_terms
        self.min_gain = min_gain
        self.final_ridge_alpha = final_ridge_alpha
        self.min_coef_for_display = min_coef_for_display
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha, unpenalized_first_col=True):
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(D.shape[1], dtype=float) * float(alpha)
        if unpenalized_first_col:
            pen[0, 0] = 0.0
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return np.asarray(beta, dtype=float)

    @staticmethod
    def _top_corr_features(X, y, k):
        yc = y - y.mean()
        yn = np.sqrt(np.sum(yc ** 2)) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xc = X[:, j] - X[:, j].mean()
            xn = np.sqrt(np.sum(xc ** 2)) + 1e-12
            scores[j] = abs(float((xc @ yc) / (xn * yn)))
        order = np.argsort(scores)[::-1]
        keep = max(1, min(int(k), X.shape[1]))
        return [int(j) for j in order[:keep]]

    @staticmethod
    def _term_col(X, term):
        j = term["feature"]
        kind = term["kind"]
        if kind == "linear":
            return X[:, j]
        if kind == "hinge_pos":
            t = term["threshold"]
            return np.maximum(0.0, X[:, j] - t)
        if kind == "hinge_neg":
            t = term["threshold"]
            return np.maximum(0.0, t - X[:, j])
        if kind == "interaction":
            j2 = term["feature2"]
            return X[:, j] * X[:, j2]
        if kind == "square":
            return X[:, j] ** 2
        raise ValueError(f"Unknown term kind: {kind}")

    def _fit_linear_with_val_alpha(self, X_tr, y_tr, X_val, y_val):
        n_tr = X_tr.shape[0]
        n_val = X_val.shape[0]
        D_tr = np.column_stack([np.ones(n_tr), X_tr])
        D_val = np.column_stack([np.ones(n_val), X_val])
        best = None
        best_beta = None
        for a in self.alpha_grid:
            beta = self._ridge_fit(D_tr, y_tr, alpha=float(a))
            mse = float(np.mean((y_val - D_val @ beta) ** 2))
            if best is None or mse < best:
                best = mse
                best_beta = beta
        return best_beta

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(30, int(float(self.val_frac) * n))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size == 0:
            tr_idx = perm
            val_idx = perm[: max(1, min(20, n))]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Stage 1: full affine backbone
        beta_lin = self._fit_linear_with_val_alpha(X_tr, y_tr, X_val, y_val)
        lin_intercept = float(beta_lin[0])
        lin_coef = np.asarray(beta_lin[1:], dtype=float)

        pred_tr_lin = lin_intercept + X_tr @ lin_coef
        pred_val_lin = lin_intercept + X_val @ lin_coef
        resid_tr = y_tr - pred_tr_lin

        # Stage 2: sparse residual basis
        library = []
        top = self._top_corr_features(
            X_tr,
            resid_tr,
            k=min(int(self.top_features_for_residual_library), p),
        )
        for j in top:
            xj = X_tr[:, j]
            med = float(np.median(xj))
            sd = float(np.std(xj) + 1e-12)
            library.append({"kind": "square", "feature": int(j)})
            for t in (med, med + 0.5 * sd, med - 0.5 * sd):
                library.append({"kind": "hinge_pos", "feature": int(j), "threshold": float(t)})
                library.append({"kind": "hinge_neg", "feature": int(j), "threshold": float(t)})

        top_int = top[: max(2, min(int(self.interaction_top_features), len(top)))]
        for j1, j2 in combinations(top_int, 2):
            library.append({"kind": "interaction", "feature": int(j1), "feature2": int(j2)})

        cols_tr = [self._term_col(X_tr, t) for t in library]
        cols_val = [self._term_col(X_val, t) for t in library]

        selected = []
        current_val_mse = float(np.mean((y_val - pred_val_lin) ** 2))
        for _ in range(max(0, int(self.max_residual_terms))):
            best_j = None
            best_mse = current_val_mse
            for j in range(len(library)):
                if j in selected:
                    continue
                cand = selected + [j]
                D_tr_aug = np.column_stack([np.ones(X_tr.shape[0]), X_tr] + [cols_tr[k] for k in cand])
                D_val_aug = np.column_stack([np.ones(X_val.shape[0]), X_val] + [cols_val[k] for k in cand])
                beta_aug = self._ridge_fit(D_tr_aug, y_tr, alpha=float(self.final_ridge_alpha))
                val_mse = float(np.mean((y_val - D_val_aug @ beta_aug) ** 2))
                if val_mse + 1e-12 < best_mse:
                    best_mse = val_mse
                    best_j = j

            if best_j is None or (current_val_mse - best_mse) < float(self.min_gain):
                break
            selected.append(best_j)
            current_val_mse = best_mse

        if selected:
            cols_full = [self._term_col(X, t) for t in library]
            D_full = np.column_stack([np.ones(n), X] + [cols_full[k] for k in selected])
            beta_full = self._ridge_fit(D_full, y, alpha=float(self.final_ridge_alpha))
            self.intercept_ = float(beta_full[0])
            self.linear_coef_ = np.asarray(beta_full[1:1 + p], dtype=float)
            self.terms_ = []
            for local_i, lib_i in enumerate(selected):
                t = dict(library[lib_i])
                t["coef"] = float(beta_full[1 + p + local_i])
                self.terms_.append(t)
        else:
            D_full = np.column_stack([np.ones(n), X])
            beta_full = self._ridge_fit(D_full, y, alpha=float(self.final_ridge_alpha))
            self.intercept_ = float(beta_full[0])
            self.linear_coef_ = np.asarray(beta_full[1:], dtype=float)
            self.terms_ = []

        importance = np.zeros(p, dtype=float)
        importance += np.abs(self.linear_coef_)
        for t in self.terms_:
            importance[t["feature"]] += abs(t["coef"])
            if t["kind"] == "interaction":
                importance[t["feature2"]] += abs(t["coef"])
        self.feature_importance_ = importance
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["n_features_in_", "intercept_", "linear_coef_", "terms_", "feature_importance_"],
        )
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.linear_coef_
        for t in self.terms_:
            c = t["coef"]
            j = t["feature"]
            if t["kind"] == "hinge_pos":
                yhat += c * np.maximum(0.0, X[:, j] - t["threshold"])
            elif t["kind"] == "hinge_neg":
                yhat += c * np.maximum(0.0, t["threshold"] - X[:, j])
            elif t["kind"] == "square":
                yhat += c * (X[:, j] ** 2)
            else:
                yhat += c * (X[:, j] * X[:, t["feature2"]])
        return yhat

    def __str__(self):
        check_is_fitted(
            self,
            ["n_features_in_", "intercept_", "linear_coef_", "terms_", "feature_importance_"],
        )
        lines = [
            "Affine Residual Symbolic Regressor",
            "prediction_rule: add linear part and residual terms exactly as written",
            "",
            "equation:",
        ]
        pieces = [f"{self.intercept_:+.6f}"] + [
            f"({c:+.6f})*x{j}"
            for j, c in enumerate(self.linear_coef_)
            if abs(c) >= float(self.min_coef_for_display)
        ]
        for t in self.terms_:
            c = t["coef"]
            if abs(c) < float(self.min_coef_for_display):
                continue
            j = t["feature"]
            if t["kind"] == "hinge_pos":
                pieces.append(f"({c:+.6f})*max(0, x{j}-{t['threshold']:.6f})")
            elif t["kind"] == "hinge_neg":
                pieces.append(f"({c:+.6f})*max(0, {t['threshold']:.6f}-x{j})")
            elif t["kind"] == "square":
                pieces.append(f"({c:+.6f})*(x{j}^2)")
            else:
                pieces.append(f"({c:+.6f})*(x{j}*x{t['feature2']})")
        lines.append("  y = " + " + ".join(pieces))
        lines.append("")

        lines.append("linear_coefficients:")
        for j, c in enumerate(self.linear_coef_):
            lines.append(f"  x{j}: coef={c:+.6f}")

        lines.append("")
        lines.append("residual_terms:")
        if not self.terms_:
            lines.append("  none")
        for i, t in enumerate(self.terms_):
            if t["kind"] == "hinge_pos":
                desc = f"max(0, x{t['feature']}-{t['threshold']:.4f})"
            elif t["kind"] == "hinge_neg":
                desc = f"max(0, {t['threshold']:.4f}-x{t['feature']})"
            elif t["kind"] == "square":
                desc = f"x{t['feature']}^2"
            else:
                desc = f"x{t['feature']}*x{t['feature2']}"
            lines.append(f"  t{i+1}: coef={t['coef']:+.6f}, term={desc}")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{j}: importance={self.feature_importance_[j]:.6f}")

        max_imp = float(np.max(self.feature_importance_)) if self.feature_importance_.size else 0.0
        if max_imp > 0:
            weak = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 0.05 * max_imp]
            if weak:
                lines.append("")
                lines.append("features_with_negligible_effect: " + ", ".join(weak))

        lines.append("")
        lines.append("manual_prediction:")
        lines.append("  1) Start from intercept.")
        lines.append("  2) Add sum_j coef(xj) * xj using linear_coefficients.")
        lines.append("  3) Add each residual term value times its coef.")
        lines.append("  4) Final sum is the prediction.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AffineResidualSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AffineResidSymbolic_v1"
model_description = "Validation-tuned affine backbone with a tiny greedy residual symbolic basis (hinge/square/interaction) for compact but more expressive closed-form equations"
model_defs = [(model_shorthand_name, AffineResidualSymbolicRegressor())]


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
