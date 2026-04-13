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


class SparseSplineAtomRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regressor over a fixed atom library:
    - Linear, square, absolute value, and hinge terms
    - Light sinusoidal/exponential atoms on top correlated features
    - A few screened pairwise products
    Terms are selected via forward OMP-style search with ridge refits and
    holdout-based model selection.
    """

    def __init__(
        self,
        max_terms=10,
        ridge_alpha=2e-2,
        min_rel_gain=5e-4,
        coef_eps=5e-5,
        max_interactions=8,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.ridge_alpha = ridge_alpha
        self.min_rel_gain = min_rel_gain
        self.coef_eps = coef_eps
        self.max_interactions = max_interactions
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(D, y, alpha):
        n, p = D.shape
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

    @staticmethod
    def _atom_col(X, term):
        kind, j, k, t = term
        xj = X[:, int(j)]
        if kind == "lin":
            return xj
        if kind == "sq":
            return xj * xj
        if kind == "abs":
            return np.abs(xj)
        if kind == "hinge":
            return np.maximum(0.0, xj - float(t))
        if kind == "sin":
            return np.sin(xj)
        if kind == "expabs":
            return np.exp(-np.abs(np.clip(xj, -6.0, 6.0)))
        if kind == "prod":
            return xj * X[:, int(k)]
        raise ValueError(f"unknown term kind: {kind}")

    def _build_library(self, X, y):
        n, p = X.shape
        yc = y - np.mean(y)
        feats = []
        cols = []

        # Core atoms for every feature.
        for j in range(p):
            xj = X[:, j]
            feats.append(("lin", int(j), -1, 0.0)); cols.append(xj)
            feats.append(("sq", int(j), -1, 0.0)); cols.append(xj * xj)
            feats.append(("abs", int(j), -1, 0.0)); cols.append(np.abs(xj))
            feats.append(("hinge", int(j), -1, 0.0)); cols.append(np.maximum(0.0, xj))
            feats.append(("hinge", int(j), -1, 0.5)); cols.append(np.maximum(0.0, xj - 0.5))
            feats.append(("hinge", int(j), -1, 1.5)); cols.append(np.maximum(0.0, xj - 1.5))

        # Light wave/decay atoms on only top correlated features.
        corr = np.zeros(p, dtype=float)
        y_norm = np.sqrt(np.sum(yc * yc)) + 1e-12
        for j in range(p):
            xj = X[:, j]
            xj_c = xj - np.mean(xj)
            corr[j] = abs(float((xj_c @ yc) / ((np.sqrt(np.sum(xj_c * xj_c)) + 1e-12) * y_norm)))
        top_wave = np.argsort(corr)[::-1][: min(4, p)]
        for j in top_wave:
            xj = X[:, int(j)]
            feats.append(("sin", int(j), -1, 0.0)); cols.append(np.sin(xj))
            feats.append(("expabs", int(j), -1, 0.0)); cols.append(np.exp(-np.abs(np.clip(xj, -6.0, 6.0))))

        # Screened pairwise products from top correlated features.
        top_inter = np.argsort(corr)[::-1][: min(6, p)]
        max_cols = 6 * p + 2 * len(top_wave) + int(self.max_interactions)
        for a in range(len(top_inter)):
            for b in range(a + 1, len(top_inter)):
                j, k = int(top_inter[a]), int(top_inter[b])
                feats.append(("prod", j, k, 0.0))
                cols.append(X[:, j] * X[:, k])
                if len(cols) >= max_cols:
                    break
            if len(cols) >= max_cols:
                break

        D = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
        return feats, D

    def _fit_forward_sparse(self, X, y):
        feats, D = self._build_library(X, y)
        n, m = D.shape
        if m == 0:
            self.eq_intercept_ = float(np.mean(y))
            self.eq_terms_ = []
            self.feature_importance_ = np.zeros(X.shape[1], dtype=float)
            return

        # Holdout split for stable stopping/model-size selection.
        if n >= 80:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(n)
            n_tr = int(0.8 * n)
            tr_idx = perm[:n_tr]
            va_idx = perm[n_tr:]
        else:
            tr_idx = np.arange(n)
            va_idx = np.arange(n)

        D_tr, y_tr = D[tr_idx], y[tr_idx]
        D_va, y_va = D[va_idx], y[va_idx]

        norms = np.sqrt(np.sum(D_tr * D_tr, axis=0)) + 1e-12
        selected = []
        b0 = float(np.mean(y_tr))
        coef = np.zeros(0, dtype=float)
        prev_mse = float(np.mean((y_tr - b0) ** 2))

        best_val = float(np.mean((y_va - np.mean(y_tr)) ** 2))
        best_b0 = float(np.mean(y_tr))
        best_sel = []
        best_coef = np.zeros(0, dtype=float)

        for _ in range(int(self.max_terms)):
            pred_tr = b0 + (D_tr[:, selected] @ coef if selected else 0.0)
            resid = y_tr - pred_tr
            score = np.abs((D_tr.T @ resid) / norms)
            for idx in selected:
                score[idx] = -np.inf
            j_star = int(np.argmax(score))
            if not np.isfinite(score[j_star]):
                break

            cand = selected + [j_star]
            b_new, c_new = self._ridge_fit(D_tr[:, cand], y_tr, self.ridge_alpha)
            pred_new = b_new + D_tr[:, cand] @ c_new
            mse = float(np.mean((y_tr - pred_new) ** 2))
            rel_gain = (prev_mse - mse) / max(prev_mse, 1e-12)
            if rel_gain < float(self.min_rel_gain):
                break

            selected = cand
            b0, coef = b_new, c_new
            prev_mse = mse

            val_mse = float(np.mean((y_va - (b0 + D_va[:, selected] @ coef)) ** 2))
            if val_mse < best_val:
                best_val = val_mse
                best_b0 = float(b0)
                best_sel = list(selected)
                best_coef = np.asarray(coef, dtype=float).copy()

        if best_sel:
            keep = np.where(np.abs(best_coef) > float(self.coef_eps))[0]
            best_sel = [best_sel[i] for i in keep]
            best_coef = best_coef[keep]

        self.eq_intercept_ = float(best_b0)
        self.eq_terms_ = [(feats[j], float(c)) for j, c in zip(best_sel, best_coef)]

        fi = np.zeros(X.shape[1], dtype=float)
        for (kind, j, k, _thr), c in self.eq_terms_:
            fi[int(j)] += abs(float(c))
            if kind == "prod" and int(k) >= 0:
                fi[int(k)] += abs(float(c))
        tot = float(np.sum(fi))
        self.feature_importance_ = fi / tot if tot > 0 else fi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        self._fit_forward_sparse(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["n_features_in_", "eq_intercept_", "eq_terms_", "feature_importance_"])
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], float(self.eq_intercept_), dtype=float)
        for term, c in self.eq_terms_:
            yhat += float(c) * self._atom_col(X, term)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "eq_intercept_", "eq_terms_", "feature_importance_"])
        terms = [f"{float(self.eq_intercept_):+.6f}"]
        for (kind, j, k, thr), c in self.eq_terms_:
            if kind == "lin":
                terms.append(f"({c:+.6f})*x{int(j)}")
            elif kind == "sq":
                terms.append(f"({c:+.6f})*(x{int(j)}^2)")
            elif kind == "abs":
                terms.append(f"({c:+.6f})*abs(x{int(j)})")
            elif kind == "hinge":
                terms.append(f"({c:+.6f})*max(0, x{int(j)} - {float(thr):+.3f})")
            elif kind == "sin":
                terms.append(f"({c:+.6f})*sin(x{int(j)})")
            elif kind == "expabs":
                terms.append(f"({c:+.6f})*exp(-abs(x{int(j)}))")
            elif kind == "prod":
                terms.append(f"({c:+.6f})*(x{int(j)}*x{int(k)})")

        lines = [
            "Sparse Spline Atom Regressor",
            "Exact equation:",
            "  y = " + " + ".join(terms),
            "",
            f"Active terms: {len(self.eq_terms_)}",
            "Feature contributions (normalized):",
        ]
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            if self.feature_importance_[j] <= 0:
                continue
            lines.append(f"  x{int(j)}: {float(self.feature_importance_[j]):.4f}")
            shown += 1
            if shown >= 12:
                break
        if shown == 0:
            lines.append("  all zero")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseSplineAtomRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseSplineAtom_v1"
model_description = "Validation-selected sparse symbolic regressor over linear/square/abs/hinge atoms with screened sinusoidal-exponential and pairwise interaction terms"
model_defs = [(model_shorthand_name, SparseSplineAtomRegressor())]


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
