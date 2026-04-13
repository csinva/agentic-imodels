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


class ResidualAtomRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone with one validation-selected residual atom.

    The model remains explicitly simulatable:
      y = intercept + sum_j coef_j*x_j + atom_coef * atom(x)
    where atom is one of: x_j^2, x_j*x_k, or relu(x_j - t).
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        hinge_quantiles=(0.25, 0.5, 0.75),
        top_feature_screen=8,
        atom_penalty=0.02,
        prune_rel=0.01,
        coef_eps=1e-10,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.hinge_quantiles = hinge_quantiles
        self.top_feature_screen = top_feature_screen
        self.atom_penalty = atom_penalty
        self.prune_rel = prune_rel
        self.coef_eps = coef_eps
        self.random_state = random_state

    @staticmethod
    def _ridge_fit_no_intercept(X, y, alpha):
        p = X.shape[1]
        gram = X.T @ X + float(alpha) * np.eye(p, dtype=float)
        rhs = X.T @ y
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ rhs
        return np.asarray(coef, dtype=float)

    @staticmethod
    def _train_val_split(n, seed):
        if n < 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        n_val = min(n_val, n - 1)
        return perm[n_val:], perm[:n_val]

    def _fit_ridge_raw_equation(self, X, y, alpha):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = np.where(x_std > 1e-12, x_std, 1.0)
        y_mean = float(np.mean(y))
        Xs = (X - x_mean) / x_std
        yc = y - y_mean
        w_std = self._ridge_fit_no_intercept(Xs, yc, alpha)
        raw_coef = w_std / x_std
        intercept = y_mean - float(np.dot(raw_coef, x_mean))
        return float(intercept), np.asarray(raw_coef, dtype=float)

    def _select_alpha(self, X_tr, y_tr, X_va, y_va):
        best = None
        for alpha in self.alpha_grid:
            b0, w = self._fit_ridge_raw_equation(X_tr, y_tr, float(alpha))
            pred = b0 + X_va @ w
            mse = float(np.mean((y_va - pred) ** 2))
            if best is None or mse < best[0]:
                best = (mse, float(alpha))
        return best[1]

    def _top_features(self, X, y):
        y0 = y - np.mean(y)
        scores = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj0 = xj - np.mean(xj)
            denom = np.sqrt(np.sum(xj0**2) * np.sum(y0**2)) + 1e-12
            corr = float(np.abs(np.dot(xj0, y0)) / denom)
            scores.append(corr)
        order = np.argsort(scores)[::-1]
        m = int(min(max(1, self.top_feature_screen), X.shape[1]))
        return np.asarray(order[:m], dtype=int)

    @staticmethod
    def _fit_atom_coef(phi, resid):
        denom = float(np.dot(phi, phi))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(phi, resid) / denom)

    def _iter_atom_candidates(self, X_tr, top_idx):
        for j in top_idx:
            xj = X_tr[:, j]
            yield ("square", int(j), None, xj**2, 1.0)

        for j in top_idx:
            xj = X_tr[:, j]
            for q in self.hinge_quantiles:
                t = float(np.quantile(xj, float(q)))
                yield ("relu", int(j), t, np.maximum(0.0, xj - t), 1.0)

        if len(top_idx) >= 2:
            for a in range(len(top_idx)):
                for b in range(a + 1, len(top_idx)):
                    j = int(top_idx[a])
                    k = int(top_idx[b])
                    yield ("interaction", j, k, X_tr[:, j] * X_tr[:, k], 1.2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        tr_idx, va_idx = self._train_val_split(n, self.random_state)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        self.alpha_ = self._select_alpha(X_tr, y_tr, X_va, y_va)
        self.intercept_, dense_coef = self._fit_ridge_raw_equation(X, y, self.alpha_)

        max_abs = float(np.max(np.abs(dense_coef))) if p > 0 else 0.0
        prune_thr = float(self.prune_rel) * max_abs
        dense_coef[np.abs(dense_coef) < max(prune_thr, float(self.coef_eps))] = 0.0
        self.coef_ = dense_coef
        self.active_features_ = np.where(self.coef_ != 0.0)[0]

        # Search for one residual atom using holdout objective.
        b0_tr, w_tr = self._fit_ridge_raw_equation(X_tr, y_tr, self.alpha_)
        pred_tr = b0_tr + X_tr @ w_tr
        pred_va = b0_tr + X_va @ w_tr
        resid_tr = y_tr - pred_tr
        resid_va = y_va - pred_va

        top_idx = self._top_features(X_tr, y_tr)
        best = (float(np.mean((y_va - pred_va) ** 2)), ("none", None, None), 0.0)

        for atom_type, a, b, phi_tr, complexity in self._iter_atom_candidates(X_tr, top_idx):
            atom_coef = self._fit_atom_coef(phi_tr, resid_tr)
            if abs(atom_coef) < 1e-12:
                continue

            if atom_type == "square":
                phi_va = X_va[:, a] ** 2
            elif atom_type == "relu":
                phi_va = np.maximum(0.0, X_va[:, a] - float(b))
            else:
                phi_va = X_va[:, a] * X_va[:, int(b)]

            improved = pred_va + atom_coef * phi_va
            mse = float(np.mean((y_va - improved) ** 2))
            obj = mse * (1.0 + float(self.atom_penalty) * float(complexity))
            if obj < best[0]:
                best = (obj, (atom_type, a, b), atom_coef)

        atom_type, a, b = best[1]
        self.atom_type_ = atom_type
        self.atom_args_ = (a, b)
        if atom_type == "none":
            self.atom_coef_ = 0.0
        else:
            base_full = self.intercept_ + X @ self.coef_
            resid_full = y - base_full
            if atom_type == "square":
                phi = X[:, a] ** 2
            elif atom_type == "relu":
                phi = np.maximum(0.0, X[:, a] - float(b))
            else:
                phi = X[:, a] * X[:, int(b)]
            self.atom_coef_ = self._fit_atom_coef(phi, resid_full)

        fi = np.abs(self.coef_)
        if self.atom_type_ != "none":
            atom_scale = np.std(self._atom_value(X))
            atom_score = abs(self.atom_coef_) * float(atom_scale)
        else:
            atom_score = 0.0
        total = float(np.sum(fi) + atom_score)
        self.feature_importance_ = fi / total if total > 0 else fi
        self.atom_importance_ = atom_score / total if total > 0 else 0.0
        return self

    def _atom_value(self, X):
        atom_type = self.atom_type_
        a, b = self.atom_args_
        if atom_type == "square":
            return X[:, a] ** 2
        if atom_type == "relu":
            return np.maximum(0.0, X[:, a] - float(b))
        if atom_type == "interaction":
            return X[:, a] * X[:, int(b)]
        return np.zeros(X.shape[0], dtype=float)

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "n_features_in_", "atom_type_", "atom_coef_"])
        X = np.asarray(X, dtype=float)
        pred = float(self.intercept_) + X @ self.coef_
        if self.atom_type_ != "none" and abs(self.atom_coef_) > 0.0:
            pred = pred + self.atom_coef_ * self._atom_value(X)
        return pred

    @staticmethod
    def _fmt_signed(value):
        if value >= 0:
            return f"+ {value:.6f}"
        return f"- {abs(value):.6f}"

    def _atom_to_text(self):
        atom_type = self.atom_type_
        a, b = self.atom_args_
        if atom_type == "square":
            return f"(x{a}^2)"
        if atom_type == "relu":
            return f"max(0, x{a} - {float(b):.6f})"
        if atom_type == "interaction":
            return f"(x{a}*x{int(b)})"
        return "0"

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "active_features_", "feature_importance_", "n_features_in_"])

        lines = [
            "Residual Atom Ridge Regressor",
            f"selected_ridge_alpha: {self.alpha_:.4g}",
            f"active_linear_features: {len(self.active_features_)}/{self.n_features_in_}",
            "equation:",
        ]

        eq = f"  y = {self.intercept_:.6f}"
        for j in self.active_features_:
            eq += f" {self._fmt_signed(float(self.coef_[j]))}*x{int(j)}"
        if self.atom_type_ != "none" and abs(self.atom_coef_) > 0.0:
            eq += f" {self._fmt_signed(float(self.atom_coef_))}*{self._atom_to_text()}"
        lines.append(eq)

        lines.append("")
        lines.append("linear_coefficients:")
        if len(self.active_features_) == 0:
            lines.append("  none")
        else:
            for j in self.active_features_:
                lines.append(f"  x{int(j)}: {self.coef_[j]:+.6f}")

        if self.atom_type_ == "none" or abs(self.atom_coef_) <= 0.0:
            lines.append("")
            lines.append("residual_atom:")
            lines.append("  none")
        else:
            lines.append("")
            lines.append("residual_atom:")
            lines.append(f"  term: {self._atom_to_text()}")
            lines.append(f"  coefficient: {self.atom_coef_:+.6f}")
            lines.append(f"  normalized_importance: {self.atom_importance_:.4f}")

        zero_feats = [f"x{j}" for j, c in enumerate(self.coef_) if c == 0.0]
        if zero_feats:
            lines.append("")
            lines.append("zero_or_irrelevant_features:")
            lines.append("  " + ", ".join(zero_feats))

        lines.append("")
        lines.append("normalized_linear_feature_importance:")
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
ResidualAtomRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualAtomRidge_v1"
model_description = "Dense holdout-selected ridge equation with tiny coefficient pruning plus one validation-selected residual atom (square/interaction/ReLU) for compact nonlinear correction"
model_defs = [(model_shorthand_name, ResidualAtomRidgeRegressor())]


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
