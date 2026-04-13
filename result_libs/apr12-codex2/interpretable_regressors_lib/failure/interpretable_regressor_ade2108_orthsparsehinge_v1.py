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


class OrthSparseHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Orthogonal forward-selected sparse piecewise-linear regressor.

    Library terms are simple and human-simulatable:
      - linear terms: x_j
      - hinge terms: max(0, x_j - t) and max(0, t - x_j)

    A small subset of terms is selected by greedy validation-MSE improvement,
    then refit with light ridge stabilization.
    """

    def __init__(
        self,
        top_features_for_library=10,
        max_terms=7,
        min_gain=1e-4,
        ridge_alpha=0.05,
        min_coef_for_display=1e-4,
        random_state=42,
    ):
        self.top_features_for_library = top_features_for_library
        self.max_terms = max_terms
        self.min_gain = min_gain
        self.ridge_alpha = ridge_alpha
        self.min_coef_for_display = min_coef_for_display
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha):
        gram = D.T @ D
        rhs = D.T @ y
        pen = np.eye(D.shape[1], dtype=float) * float(alpha)
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
        raise ValueError(f"Unknown term kind: {kind}")

    def _build_library(self, X, y):
        top = self._top_corr_features(X, y, self.top_features_for_library)
        lib = []
        for j in top:
            xj = X[:, j]
            med = float(np.median(xj))
            sd = float(np.std(xj) + 1e-12)
            thresholds = [med, med + 0.5 * sd]

            lib.append({"kind": "linear", "feature": int(j)})
            for t in thresholds:
                lib.append({"kind": "hinge_pos", "feature": int(j), "threshold": float(t)})
                lib.append({"kind": "hinge_neg", "feature": int(j), "threshold": float(t)})
        return lib

    def _fit_with_terms(self, X, y, term_indices, term_cols):
        n = X.shape[0]
        if term_indices:
            D = np.column_stack([np.ones(n), np.column_stack([term_cols[i] for i in term_indices])])
        else:
            D = np.ones((n, 1))
        beta = self._ridge_fit(D, y, self.ridge_alpha)
        preds = D @ beta
        return beta, preds

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(30, int(0.2 * n))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size == 0:
            tr_idx = perm
            val_idx = perm[: max(1, min(20, n))]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        library = self._build_library(X_tr, y_tr)
        term_cols_tr = [self._term_col(X_tr, t) for t in library]
        term_cols_val = [self._term_col(X_val, t) for t in library]

        selected = []
        beta_base, pred_base = self._fit_with_terms(X_tr, y_tr, selected, term_cols_tr)
        del beta_base
        best_val_mse = float(np.mean((y_val - np.full_like(y_val, pred_base.mean())) ** 2))
        for _ in range(max(1, int(self.max_terms))):
            best_j = None
            best_mse = best_val_mse
            for j in range(len(library)):
                if j in selected:
                    continue
                cand = selected + [j]
                n_val_rows = X_val.shape[0]
                D_val = np.column_stack(
                    [np.ones(n_val_rows), np.column_stack([term_cols_val[k] for k in cand])]
                )
                n_tr_rows = X_tr.shape[0]
                D_tr = np.column_stack(
                    [np.ones(n_tr_rows), np.column_stack([term_cols_tr[k] for k in cand])]
                )
                beta = self._ridge_fit(D_tr, y_tr, self.ridge_alpha)
                val_mse = float(np.mean((y_val - D_val @ beta) ** 2))
                if val_mse + 1e-12 < best_mse:
                    best_mse = val_mse
                    best_j = j

            if best_j is None or (best_val_mse - best_mse) < float(self.min_gain):
                break
            selected.append(best_j)
            best_val_mse = best_mse

        # Fallback to top linear term if no term selected
        if not selected:
            for i, t in enumerate(library):
                if t["kind"] == "linear":
                    selected = [i]
                    break

        term_cols_full = [self._term_col(X, t) for t in library]
        D = np.column_stack([np.ones(n), np.column_stack([term_cols_full[k] for k in selected])])
        beta = self._ridge_fit(D, y, self.ridge_alpha)

        self.intercept_ = float(beta[0])
        self.terms_ = []
        for local_idx, lib_idx in enumerate(selected):
            t = dict(library[lib_idx])
            t["coef"] = float(beta[1 + local_idx])
            self.terms_.append(t)

        importance = np.zeros(p, dtype=float)
        for t in self.terms_:
            importance[t["feature"]] += abs(t["coef"])
        self.feature_importance_ = importance
        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "intercept_", "terms_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for t in self.terms_:
            c = t["coef"]
            j = t["feature"]
            if t["kind"] == "linear":
                yhat += c * X[:, j]
            elif t["kind"] == "hinge_pos":
                yhat += c * np.maximum(0.0, X[:, j] - t["threshold"])
            else:
                yhat += c * np.maximum(0.0, t["threshold"] - X[:, j])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "intercept_", "terms_", "feature_importance_"])
        lines = [
            "Orth Sparse Hinge Regressor",
            "prediction_rule: start at intercept and add each term below",
            "",
            "equation:",
        ]
        pieces = [f"{self.intercept_:+.6f}"]
        for t in self.terms_:
            c = t["coef"]
            if abs(c) < float(self.min_coef_for_display):
                continue
            j = t["feature"]
            if t["kind"] == "linear":
                pieces.append(f"({c:+.6f})*x{j}")
            elif t["kind"] == "hinge_pos":
                pieces.append(f"({c:+.6f})*max(0, x{j}-{t['threshold']:.6f})")
            else:
                pieces.append(f"({c:+.6f})*max(0, {t['threshold']:.6f}-x{j})")
        lines.append("  y = " + " + ".join(pieces))
        lines.append("")
        lines.append("active_terms:")
        for i, t in enumerate(self.terms_):
            if t["kind"] == "linear":
                desc = f"x{t['feature']}"
            elif t["kind"] == "hinge_pos":
                desc = f"max(0, x{t['feature']}-{t['threshold']:.4f})"
            else:
                desc = f"max(0, {t['threshold']:.4f}-x{t['feature']})"
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
        lines.append("  2) For each active term, compute term value and multiply by coef.")
        lines.append("  3) Sum everything.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
OrthSparseHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "OrthSparseHinge_v1"
model_description = "Forward-selected sparse piecewise-linear equation over linear and hinge terms with validation-gated term inclusion and explicit arithmetic form"
model_defs = [(model_shorthand_name, OrthSparseHingeRegressor())]


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
