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


class ResidualSparseEquationRegressor(BaseEstimator, RegressorMixin):
    """Dense linear backbone plus sparse nonlinear residual correction."""

    def __init__(
        self,
        alpha_linear_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0),
        alpha_resid_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        val_frac=0.2,
        max_resid_terms=8,
        interaction_screen=6,
        min_rel_gain=0.003,
        near_zero_rel=0.06,
        coef_eps=1e-9,
        random_state=42,
    ):
        self.alpha_linear_grid = alpha_linear_grid
        self.alpha_resid_grid = alpha_resid_grid
        self.val_frac = val_frac
        self.max_resid_terms = max_resid_terms
        self.interaction_screen = interaction_screen
        self.min_rel_gain = min_rel_gain
        self.near_zero_rel = near_zero_rel
        self.coef_eps = coef_eps
        self.random_state = random_state

    def _train_val_split(self, n, seed):
        if n < 25:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(float(self.val_frac) * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    @staticmethod
    def _fit_ridge_with_intercept(X, y, alpha):
        n, p = X.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        Z = np.column_stack([np.ones(n, dtype=float), X])
        gram = Z.T @ Z
        pen = np.eye(p + 1, dtype=float) * float(alpha)
        pen[0, 0] = 0.0
        rhs = Z.T @ y
        try:
            beta = np.linalg.solve(gram + pen, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + pen) @ rhs
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _select_alpha(self, X_tr, y_tr, X_va, y_va, alphas):
        best = None
        for alpha in alphas:
            b0, w = self._fit_ridge_with_intercept(X_tr, y_tr, float(alpha))
            mse = float(np.mean((y_va - (b0 + X_va @ w)) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha), b0, w)
        return best

    def _prune_coef(self, coef):
        coef = np.asarray(coef, dtype=float).copy()
        coef[np.abs(coef) < float(self.coef_eps)] = 0.0
        return coef

    @staticmethod
    def _corr_screen(X, y):
        yc = y - np.mean(y)
        yn = float(np.linalg.norm(yc)) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j] - np.mean(X[:, j])
            xn = float(np.linalg.norm(xj)) + 1e-12
            scores[j] = abs(float(np.dot(xj, yc) / (xn * yn)))
        return np.argsort(scores)[::-1]

    @staticmethod
    def _term_values(X, term):
        ttype = term[0]
        if ttype == "abs":
            return np.abs(X[:, int(term[1])])
        if ttype == "hinge+":
            j, thr = int(term[1]), float(term[2])
            return np.maximum(0.0, X[:, j] - thr)
        if ttype == "hinge-":
            j, thr = int(term[1]), float(term[2])
            return np.maximum(0.0, thr - X[:, j])
        if ttype == "interaction":
            i, j = int(term[1]), int(term[2])
            return X[:, i] * X[:, j]
        raise ValueError(f"Unknown term type: {ttype}")

    @staticmethod
    def _term_name(term):
        ttype = term[0]
        if ttype == "abs":
            return f"abs(x{int(term[1])})"
        if ttype == "hinge+":
            return f"max(0, x{int(term[1])} - {float(term[2]):.6f})"
        if ttype == "hinge-":
            return f"max(0, {float(term[2]):.6f} - x{int(term[1])})"
        if ttype == "interaction":
            return f"x{int(term[1])}*x{int(term[2])}"
        return str(term)

    def _build_residual_library(self, X, linear_coef):
        n, p = X.shape
        terms = []
        for j in range(p):
            med = float(np.median(X[:, j]))
            terms.append(("abs", j))
            terms.append(("hinge+", j, med))
            terms.append(("hinge-", j, med))

        top = np.argsort(np.abs(np.asarray(linear_coef, dtype=float)))[::-1]
        top = top[: min(int(self.interaction_screen), p)]
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                terms.append(("interaction", int(top[a]), int(top[b])))

        if len(terms) == 0:
            return np.zeros((n, 0), dtype=float), terms
        B = np.column_stack([self._term_values(X, t) for t in terms])
        return B, terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)
        tr_idx, va_idx = self._train_val_split(n, self.random_state)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1) Dense linear backbone
        lin_best = self._select_alpha(X_tr, y_tr, X_va, y_va, self.alpha_linear_grid)
        self.alpha_linear_ = lin_best[1]
        self.linear_intercept_, self.linear_coef_ = self._fit_ridge_with_intercept(X, y, self.alpha_linear_)
        self.linear_coef_ = self._prune_coef(self.linear_coef_)

        resid = y - (self.linear_intercept_ + X @ self.linear_coef_)
        resid_tr = resid[tr_idx]
        resid_va = resid[va_idx]

        # 2) Sparse nonlinear residual atoms
        B, all_terms = self._build_residual_library(X, self.linear_coef_)
        B_tr, B_va = B[tr_idx], B[va_idx]

        if B.shape[1] == 0 or int(self.max_resid_terms) <= 0:
            self.selected_term_idx_ = np.array([], dtype=int)
            self.resid_terms_ = []
            self.resid_coef_ = np.array([], dtype=float)
            self.resid_intercept_ = 0.0
            self.alpha_resid_ = 0.0
        else:
            mu = np.mean(B_tr, axis=0)
            sigma = np.std(B_tr, axis=0)
            sigma = np.where(sigma > 1e-12, sigma, 1.0)
            Z_tr = (B_tr - mu) / sigma
            Z_va = (B_va - mu) / sigma

            selected = []
            current_pred_va = np.zeros_like(resid_va)
            current_mse = float(np.mean((resid_va - current_pred_va) ** 2))
            best_alpha_for_size = {}

            for _ in range(min(int(self.max_resid_terms), Z_tr.shape[1])):
                r_tr = resid_tr - (
                    0.0 if len(selected) == 0
                    else self._fit_ridge_with_intercept(Z_tr[:, selected], resid_tr, best_alpha_for_size[len(selected)])[0]
                    + Z_tr[:, selected] @ self._fit_ridge_with_intercept(
                        Z_tr[:, selected], resid_tr, best_alpha_for_size[len(selected)]
                    )[1]
                )
                corrs = np.abs(Z_tr.T @ r_tr)
                if len(selected):
                    corrs[selected] = -np.inf
                j_new = int(np.argmax(corrs))
                if not np.isfinite(corrs[j_new]):
                    break
                trial = selected + [j_new]

                trial_best = self._select_alpha(
                    Z_tr[:, trial], resid_tr, Z_va[:, trial], resid_va, self.alpha_resid_grid
                )
                trial_mse, trial_alpha, b0_t, w_t = trial_best
                rel_gain = (current_mse - trial_mse) / (current_mse + 1e-12)
                if rel_gain < float(self.min_rel_gain):
                    break

                selected = trial
                best_alpha_for_size[len(selected)] = trial_alpha
                current_mse = trial_mse
                current_pred_va = b0_t + Z_va[:, selected] @ w_t

            self.selected_term_idx_ = np.array(selected, dtype=int)
            self.resid_terms_ = [all_terms[i] for i in self.selected_term_idx_]
            if len(selected) == 0:
                self.resid_coef_ = np.array([], dtype=float)
                self.resid_intercept_ = 0.0
                self.alpha_resid_ = 0.0
            else:
                Z_full = (B - mu) / sigma
                alpha_resid = best_alpha_for_size[len(selected)]
                b0, w = self._fit_ridge_with_intercept(Z_full[:, selected], resid, alpha_resid)
                self.alpha_resid_ = float(alpha_resid)
                w = np.asarray(w, dtype=float) / sigma[selected]
                b0 = float(b0 - np.dot(np.asarray(w, dtype=float), mu[selected]))
                w = self._prune_coef(w)
                self.resid_intercept_ = b0
                self.resid_coef_ = w
                self._resid_mu_ = mu
                self._resid_sigma_ = sigma
                self._all_terms_ = all_terms

        # Merge intercepts for a single explicit arithmetic expression
        self.intercept_ = float(self.linear_intercept_ + self.resid_intercept_)
        self.coef_ = np.asarray(self.linear_coef_, dtype=float)

        active_linear = np.where(np.abs(self.linear_coef_) > 0)[0]
        self.active_features_ = np.array(active_linear, dtype=int)
        fi = np.abs(self.linear_coef_).copy()
        for term, c in zip(getattr(self, "resid_terms_", []), getattr(self, "resid_coef_", [])):
            if term[0] in ("abs", "hinge+", "hinge-"):
                fi[int(term[1])] += abs(float(c))
            elif term[0] == "interaction":
                fi[int(term[1])] += 0.5 * abs(float(c))
                fi[int(term[2])] += 0.5 * abs(float(c))
        total = float(np.sum(fi))
        self.feature_importance_ = fi / total if total > 0 else fi
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "coef_",
                "intercept_",
                "n_features_in_",
            ],
        )
        X = np.asarray(X, dtype=float)
        yhat = float(self.intercept_) + X @ np.asarray(self.coef_, dtype=float)
        for t, c in zip(getattr(self, "resid_terms_", []), getattr(self, "resid_coef_", [])):
            yhat = yhat + float(c) * self._term_values(X, t)
        return yhat

    @staticmethod
    def _fmt_signed(v):
        if v >= 0:
            return f"+ {v:.6f}"
        return f"- {abs(v):.6f}"

    def _eq_from_coef(self, intercept, coef):
        eq = f"{float(intercept):.6f}"
        for j in np.where(np.asarray(coef) != 0.0)[0]:
            eq += f" {self._fmt_signed(float(coef[j]))}*x{int(j)}"
        for t, c in zip(getattr(self, "resid_terms_", []), getattr(self, "resid_coef_", [])):
            eq += f" {self._fmt_signed(float(c))}*{self._term_name(t)}"
        return eq

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_importance_", "n_features_in_"])

        lines = [
            "Residual Sparse Equation Regressor",
            f"linear_alpha: {self.alpha_linear_:.4g}",
            f"residual_alpha: {getattr(self, 'alpha_resid_', 0.0):.4g}",
            "",
            "equation:",
            "  y = " + self._eq_from_coef(self.intercept_, self.coef_),
            "",
            "active_linear_features:",
            "  " + (", ".join(f"x{int(j)}" for j in self.active_features_) if len(self.active_features_) else "none"),
        ]
        max_linear = float(np.max(np.abs(self.coef_))) if self.coef_.size else 0.0
        near_zero_thr = float(self.near_zero_rel) * max_linear
        near_zero = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if abs(float(self.coef_[j])) <= near_zero_thr
        ]
        if near_zero:
            lines.append("")
            lines.append("near_zero_linear_features:")
            lines.append("  " + ", ".join(near_zero))

        lines.append("")
        lines.append("residual_terms:")
        if len(getattr(self, "resid_terms_", [])) == 0:
            lines.append("  none")
        else:
            for k, (t, c) in enumerate(zip(self.resid_terms_, self.resid_coef_), 1):
                lines.append(f"  r{k}: ({float(c):+.6f}) * {self._term_name(t)}")

        lines.append("")
        lines.append("normalized_feature_importance:")
        shown = 0
        for j in np.argsort(self.feature_importance_)[::-1]:
            score = float(self.feature_importance_[j])
            if score <= 0:
                continue
            lines.append(f"  x{int(j)}: {score:.4f}")
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
ResidualSparseEquationRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualSparseEq_v1"
model_description = "Dense ridge backbone plus validation-gated sparse residual atoms (abs/median-hinge/interactions) with one explicit arithmetic equation"
model_defs = [(model_shorthand_name, ResidualSparseEquationRegressor())]


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
