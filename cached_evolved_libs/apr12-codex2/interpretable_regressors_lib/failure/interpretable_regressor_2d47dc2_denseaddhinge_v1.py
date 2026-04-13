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


class DenseAdditiveHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense additive spline model with lightweight interactions:
    - Per feature: linear term + two learned hinge terms (piecewise linear GAM-like effect)
    - Small number of screened pairwise interactions between dominant features
    - Single ridge solve with holdout-selected regularization
    """

    def __init__(
        self,
        alpha_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
        max_interactions=6,
        coef_eps=1e-6,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_interactions = max_interactions
        self.coef_eps = coef_eps
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
    def _safe_scale(x):
        med = np.median(x, axis=0)
        q25 = np.percentile(x, 25, axis=0)
        q75 = np.percentile(x, 75, axis=0)
        iqr = q75 - q25
        std = np.std(x, axis=0)
        scale = np.where(iqr > 1e-8, iqr, np.where(std > 1e-8, std, 1.0))
        return med, scale

    def _build_design(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        Z = np.clip((X - self.center_) / self.scale_, -8.0, 8.0)
        n, p = Z.shape
        cols = []
        terms = []

        for j in range(p):
            zj = Z[:, j]
            cols.append(zj)
            terms.append(("lin", int(j), -1, 0.0))
            t1, t2 = float(self.knots_[j, 0]), float(self.knots_[j, 1])
            cols.append(np.maximum(0.0, zj - t1))
            terms.append(("hinge", int(j), -1, t1))
            cols.append(np.maximum(0.0, zj - t2))
            terms.append(("hinge", int(j), -1, t2))

        for j, k in self.interaction_pairs_:
            cols.append(Z[:, j] * Z[:, k])
            terms.append(("prod", int(j), int(k), 0.0))

        D = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
        if fit:
            self.design_terms_ = terms
        return D

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        self.center_, self.scale_ = self._safe_scale(X)
        Z = np.clip((X - self.center_) / self.scale_, -8.0, 8.0)

        self.knots_ = np.zeros((p, 2), dtype=float)
        for j in range(p):
            zj = Z[:, j]
            q1, q2 = np.percentile(zj, [33.0, 66.0])
            if abs(q2 - q1) < 1e-8:
                q1, q2 = -0.5, 0.5
            self.knots_[j, 0] = float(q1)
            self.knots_[j, 1] = float(q2)

        yc = y - float(np.mean(y))
        zc = Z - np.mean(Z, axis=0, keepdims=True)
        y_norm = float(np.linalg.norm(yc)) + 1e-12
        corr = np.zeros(p, dtype=float)
        for j in range(p):
            denom = float(np.linalg.norm(zc[:, j])) + 1e-12
            corr[j] = abs(float(zc[:, j] @ yc) / (denom * y_norm))

        top = np.argsort(corr)[::-1][: min(8, p)]
        pairs = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                pairs.append((int(top[a]), int(top[b])))
                if len(pairs) >= int(self.max_interactions):
                    break
            if len(pairs) >= int(self.max_interactions):
                break
        self.interaction_pairs_ = pairs

        if n >= 100:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(n)
            n_tr = int(0.8 * n)
            tr, va = perm[:n_tr], perm[n_tr:]
        else:
            tr = np.arange(n)
            va = np.arange(n)

        D = self._build_design(X, fit=True)
        D_tr, y_tr = D[tr], y[tr]
        D_va, y_va = D[va], y[va]

        best_alpha = float(self.alpha_grid[0])
        best_val = float("inf")
        for a in self.alpha_grid:
            b, c = self._ridge_fit(D_tr, y_tr, float(a))
            val = float(np.mean((y_va - (b + D_va @ c)) ** 2))
            if val < best_val:
                best_val = val
                best_alpha = float(a)

        self.alpha_ = best_alpha
        self.eq_intercept_, coef = self._ridge_fit(D, y, self.alpha_)

        # Keep tiny terms at zero for cleaner equations without changing structure.
        coef[np.abs(coef) < float(self.coef_eps)] = 0.0
        self.eq_terms_ = [(t, float(c)) for t, c in zip(self.design_terms_, coef)]

        fi = np.zeros(p, dtype=float)
        for (kind, j, k, _), c in self.eq_terms_:
            w = abs(float(c))
            fi[int(j)] += w
            if kind == "prod" and int(k) >= 0:
                fi[int(k)] += w
        s = float(np.sum(fi))
        self.feature_importance_ = fi / s if s > 0 else fi
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["n_features_in_", "center_", "scale_", "knots_", "interaction_pairs_", "eq_intercept_", "eq_terms_"],
        )
        X = np.asarray(X, dtype=float)
        Z = np.clip((X - self.center_) / self.scale_, -8.0, 8.0)
        yhat = np.full(X.shape[0], float(self.eq_intercept_), dtype=float)
        for (kind, j, k, thr), c in self.eq_terms_:
            if c == 0.0:
                continue
            if kind == "lin":
                yhat += c * Z[:, int(j)]
            elif kind == "hinge":
                yhat += c * np.maximum(0.0, Z[:, int(j)] - float(thr))
            elif kind == "prod":
                yhat += c * (Z[:, int(j)] * Z[:, int(k)])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["n_features_in_", "eq_intercept_", "eq_terms_", "feature_importance_"])
        eq = [f"{float(self.eq_intercept_):+.6f}"]
        for (kind, j, k, thr), c in self.eq_terms_:
            if abs(c) <= 0.0:
                continue
            if kind == "lin":
                eq.append(f"({c:+.6f})*z{int(j)}")
            elif kind == "hinge":
                eq.append(f"({c:+.6f})*max(0, z{int(j)}-{float(thr):+.4f})")
            elif kind == "prod":
                eq.append(f"({c:+.6f})*(z{int(j)}*z{int(k)})")

        lines = [
            "Dense Additive Hinge Regressor",
            "Normalization:",
            "  z_j = (x_j - center_j) / scale_j",
            f"  chosen_ridge_alpha = {self.alpha_:.4g}",
            "Exact equation in normalized features:",
            "  y = " + " + ".join(eq),
            "",
            "Top feature contributions (normalized):",
        ]
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            val = float(self.feature_importance_[j])
            if val <= 0:
                continue
            lines.append(
                f"  x{int(j)}: {val:.4f}  [center={float(self.center_[j]):+.4f}, scale={float(self.scale_[j]):.4f}, "
                f"knots={float(self.knots_[j,0]):+.3f}/{float(self.knots_[j,1]):+.3f}]"
            )
            shown += 1
            if shown >= 14:
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
DenseAdditiveHingeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "DenseAddHinge_v1"
model_description = "Dense additive piecewise-linear model with per-feature two-knot hinge basis, screened pairwise interactions, robust normalization, and holdout-selected ridge shrinkage"
model_defs = [(model_shorthand_name, DenseAdditiveHingeRegressor())]


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
