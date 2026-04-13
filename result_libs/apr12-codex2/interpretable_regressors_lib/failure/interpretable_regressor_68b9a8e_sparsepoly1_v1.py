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


class SparsePoly1Regressor(BaseEstimator, RegressorMixin):
    """
    Linear backbone plus one validated polynomial correction term.

    Candidate correction is either x_j^2 or x_i*x_j, selected on a holdout split.
    """

    def __init__(
        self,
        val_frac=0.2,
        alpha_grid=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        topk_for_poly=8,
        min_poly_gain=0.008,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.val_frac = val_frac
        self.alpha_grid = alpha_grid
        self.topk_for_poly = topk_for_poly
        self.min_poly_gain = min_poly_gain
        self.coef_eps = coef_eps
        self.random_state = random_state

    def _make_split(self, n):
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = int(round(float(self.val_frac) * n))
        n_val = min(max(1, n_val), max(1, n - 1))
        return perm[n_val:], perm[:n_val]

    @staticmethod
    def _std_params(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return mu, sigma

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

    def _best_alpha(self, D_tr, y_tr, D_val, y_val):
        best = None
        for alpha in self.alpha_grid:
            b0, coef = self._ridge_fit(D_tr, y_tr, alpha)
            pred = b0 + D_val @ coef
            mse = float(np.mean((y_val - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, coef)
        return best

    def _candidate_terms(self, Z, top_idx):
        cands = []
        for j in top_idx:
            cands.append(("square", int(j), int(j), Z[:, j] ** 2 - 1.0))
        for a in range(len(top_idx)):
            i = int(top_idx[a])
            for b in range(a + 1, len(top_idx)):
                j = int(top_idx[b])
                cands.append(("interaction", i, j, Z[:, i] * Z[:, j]))
        return cands

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, val_idx = self._make_split(n)
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        mu_tr, sc_tr = self._std_params(X_tr)
        Ztr = (X_tr - mu_tr) / sc_tr
        Zval = (X_val - mu_tr) / sc_tr

        base_val_mse, alpha_base, _, w_base = self._best_alpha(Ztr, y_tr, Zval, y_val)

        topk = int(min(max(2, self.topk_for_poly), p))
        top_idx = np.argsort(np.abs(w_base))[::-1][:topk]
        candidates = self._candidate_terms(Ztr, top_idx)

        best_poly = None
        for typ, i, j, term_tr in candidates:
            if np.std(term_tr) < 1e-10:
                continue
            term_val = Zval[:, i] ** 2 - 1.0 if typ == "square" else Zval[:, i] * Zval[:, j]
            Dtr = np.column_stack([Ztr, term_tr])
            Dval = np.column_stack([Zval, term_val])
            mse, alpha, _, _ = self._best_alpha(Dtr, y_tr, Dval, y_val)
            if best_poly is None or mse < best_poly[0]:
                best_poly = (mse, float(alpha), typ, i, j)

        use_poly = False
        if best_poly is not None:
            rel_gain = (base_val_mse - best_poly[0]) / max(base_val_mse, 1e-12)
            use_poly = rel_gain >= float(self.min_poly_gain)

        self.x_mean_, self.x_scale_ = self._std_params(X)
        Zall = (X - self.x_mean_) / self.x_scale_

        self.poly_type_ = "none"
        self.poly_i_ = -1
        self.poly_j_ = -1
        self.poly_coef_raw_ = 0.0

        if use_poly:
            _, alpha_poly, typ, i, j = best_poly
            if typ == "square":
                term_all = Zall[:, i] ** 2 - 1.0
            else:
                term_all = Zall[:, i] * Zall[:, j]
            Dall = np.column_stack([Zall, term_all])
            b_all, coef_all = self._ridge_fit(Dall, y, alpha_poly)
            wz = coef_all[:p]
            g = float(coef_all[p])
            self.poly_type_ = typ
            self.poly_i_ = int(i)
            self.poly_j_ = int(j)
            self.poly_coef_raw_ = g
        else:
            b_all, wz = self._ridge_fit(Zall, y, alpha_base)
            wz = np.asarray(wz, dtype=float)

        linear = wz / self.x_scale_
        intercept = float(b_all - np.sum(wz * self.x_mean_ / self.x_scale_))
        quad = np.zeros(p, dtype=float)
        inter = {}

        if self.poly_type_ == "square":
            i = self.poly_i_
            g = self.poly_coef_raw_
            si = self.x_scale_[i]
            mi = self.x_mean_[i]
            quad[i] += g / (si * si)
            linear[i] += -2.0 * g * mi / (si * si)
            intercept += g * (mi * mi) / (si * si) - g
        elif self.poly_type_ == "interaction":
            i, j = self.poly_i_, self.poly_j_
            g = self.poly_coef_raw_
            si, sj = self.x_scale_[i], self.x_scale_[j]
            mi, mj = self.x_mean_[i], self.x_mean_[j]
            inter[(i, j)] = g / (si * sj)
            linear[i] += -g * mj / (si * sj)
            linear[j] += -g * mi / (si * sj)
            intercept += g * mi * mj / (si * sj)

        self.intercept_ = intercept
        self.linear_coef_ = linear
        self.quad_coef_ = quad
        self.inter_coef_ = inter

        self.linear_coef_[np.abs(self.linear_coef_) < float(self.coef_eps)] = 0.0
        self.quad_coef_[np.abs(self.quad_coef_) < float(self.coef_eps)] = 0.0
        for k in list(self.inter_coef_.keys()):
            if abs(self.inter_coef_[k]) < float(self.coef_eps):
                del self.inter_coef_[k]

        fi = np.abs(self.linear_coef_) + np.abs(self.quad_coef_)
        for (i, j), c in self.inter_coef_.items():
            ac = abs(c)
            fi[i] += ac
            fi[j] += ac
        tot = float(np.sum(fi))
        self.feature_importance_ = fi / tot if tot > 0 else fi
        return self

    def _predict_no_check(self, X):
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ self.linear_coef_
        if np.any(self.quad_coef_):
            yhat = yhat + (X * X) @ self.quad_coef_
        for (i, j), c in self.inter_coef_.items():
            yhat = yhat + c * X[:, i] * X[:, j]
        return yhat

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "linear_coef_",
                "quad_coef_",
                "inter_coef_",
                "feature_importance_",
                "n_features_in_",
            ],
        )
        return self._predict_no_check(X)

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "linear_coef_",
                "quad_coef_",
                "inter_coef_",
                "feature_importance_",
                "n_features_in_",
            ],
        )
        terms = [f"{float(self.intercept_):+.6f}"]
        for j in np.where(np.abs(self.linear_coef_) > 0)[0]:
            terms.append(f"({float(self.linear_coef_[j]):+.6f})*x{int(j)}")
        for j in np.where(np.abs(self.quad_coef_) > 0)[0]:
            terms.append(f"({float(self.quad_coef_[j]):+.6f})*x{int(j)}^2")
        for (i, j), c in sorted(self.inter_coef_.items()):
            terms.append(f"({float(c):+.6f})*x{int(i)}*x{int(j)}")

        lines = [
            "Sparse-Poly1 Regressor",
            "Equation:",
            "  y = " + " + ".join(terms),
            "Only listed terms are active.",
            "",
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
SparsePoly1Regressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparsePoly1_v1"
model_description = "Dense linear backbone with one holdout-selected polynomial correction term (single square or pairwise interaction) expanded into an explicit raw-feature equation"
model_defs = [(model_shorthand_name, SparsePoly1Regressor())]


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
