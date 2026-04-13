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


class SparseAtomHybridRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic regressor from a small dictionary of transparent atoms.

    Candidate atoms:
      - linear: x_j
      - absolute value: abs(x_j)
      - hinge at zero: max(0, x_j)
      - quadratic: x_j^2
      - screened interactions: x_i * x_j

    A validation-greedy forward step picks at most `max_terms` atoms, then a
    ridge refit produces the final arithmetic equation.
    """

    def __init__(
        self,
        alpha=1e-2,
        val_frac=0.2,
        max_terms=8,
        max_features_for_atoms=8,
        max_interaction_features=5,
        min_improvement=2e-3,
        min_coef=1e-4,
        random_state=42,
    ):
        self.alpha = alpha
        self.val_frac = val_frac
        self.max_terms = max_terms
        self.max_features_for_atoms = max_features_for_atoms
        self.max_interaction_features = max_interaction_features
        self.min_improvement = min_improvement
        self.min_coef = min_coef
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
    def _atom_values(X, atom):
        kind = atom[0]
        if kind == "lin":
            return X[:, atom[1]]
        if kind == "abs":
            return np.abs(X[:, atom[1]])
        if kind == "relu":
            return np.maximum(0.0, X[:, atom[1]])
        if kind == "sq":
            return X[:, atom[1]] ** 2
        if kind == "int":
            return X[:, atom[1]] * X[:, atom[2]]
        raise ValueError(f"Unknown atom kind: {kind}")

    def _build_candidates(self, X, y):
        _, p = X.shape
        xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = np.sqrt(np.sum(xc ** 2, axis=0) * (np.sum(yc ** 2) + 1e-12)) + 1e-12
        corr = np.abs((xc.T @ yc) / denom)

        k = min(int(max(1, self.max_features_for_atoms)), p)
        top = np.argsort(corr)[::-1][:k]

        atoms = []
        for j in top:
            j = int(j)
            atoms.append(("lin", j))
            atoms.append(("abs", j))
            atoms.append(("relu", j))
            atoms.append(("sq", j))

        ki = min(int(max(2, self.max_interaction_features)), len(top))
        top_int = [int(v) for v in top[:ki]]
        for a in range(len(top_int)):
            for b in range(a + 1, len(top_int)):
                atoms.append(("int", top_int[a], top_int[b]))

        seen = set()
        uniq = []
        for atom in atoms:
            if atom not in seen:
                uniq.append(atom)
                seen.add(atom)
        return uniq

    def _design(self, X, atoms):
        if not atoms:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._atom_values(X, a) for a in atoms]
        return np.column_stack(cols).astype(float)

    def _fit_with_atoms(self, X, y, atoms):
        D = self._design(X, atoms)
        intercept, coef = self._ridge_fit(D, y, self.alpha)
        return intercept, coef

    def _predict_with_atoms(self, X, atoms, intercept, coef):
        if not atoms:
            return np.full(X.shape[0], float(intercept), dtype=float)
        D = self._design(X, atoms)
        return float(intercept) + D @ coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p
        self.feature_names_in_ = np.array([f"x{i}" for i in range(p)], dtype=object)

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(24, int(float(self.val_frac) * n))
        if n_val >= n:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size == 0:
            tr_idx = perm
            val_idx = perm[: max(1, min(24, n))]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        candidates = self._build_candidates(X_tr, y_tr)
        selected = []
        remaining = list(range(len(candidates)))

        base_intercept = float(np.mean(y_tr))
        base_pred = np.full_like(y_val, base_intercept, dtype=float)
        best_val_mse = float(np.mean((y_val - base_pred) ** 2))

        for _ in range(int(max(1, self.max_terms))):
            cur_best_idx = None
            cur_best_tuple = None

            for idx in remaining:
                trial_atoms = selected + [candidates[idx]]
                intercept, coef = self._fit_with_atoms(X_tr, y_tr, trial_atoms)
                pred = self._predict_with_atoms(X_val, trial_atoms, intercept, coef)
                mse = float(np.mean((y_val - pred) ** 2))
                if (cur_best_tuple is None) or (mse < cur_best_tuple[0]):
                    cur_best_tuple = (mse, intercept, coef)
                    cur_best_idx = idx

            if cur_best_tuple is None:
                break

            improvement = best_val_mse - cur_best_tuple[0]
            if improvement < float(self.min_improvement):
                break

            selected.append(candidates[cur_best_idx])
            remaining.remove(cur_best_idx)
            best_val_mse = cur_best_tuple[0]

        self.selected_atoms_ = selected
        D_full = self._design(X, self.selected_atoms_)
        self.intercept_, self.coef_ = self._ridge_fit(D_full, y, self.alpha)

        coef = np.asarray(self.coef_, dtype=float).copy()
        coef[np.abs(coef) < float(self.min_coef)] = 0.0
        self.coef_ = coef
        if self.selected_atoms_:
            self.intercept_ = float(np.mean(y - self._design(X, self.selected_atoms_) @ self.coef_))
        else:
            self.intercept_ = float(np.mean(y))

        imp = np.zeros(p, dtype=float)
        for c, atom in zip(self.coef_, self.selected_atoms_):
            w = abs(float(c))
            if atom[0] == "int":
                imp[atom[1]] += 0.5 * w
                imp[atom[2]] += 0.5 * w
            else:
                imp[atom[1]] += w
        self.feature_importance_ = imp / (imp.sum() + 1e-12)
        return self

    def predict(self, X):
        check_is_fitted(self, ["selected_atoms_", "intercept_", "coef_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        return self._predict_with_atoms(X, self.selected_atoms_, self.intercept_, self.coef_)

    @staticmethod
    def _atom_to_str(atom):
        kind = atom[0]
        if kind == "lin":
            return f"x{atom[1]}"
        if kind == "abs":
            return f"abs(x{atom[1]})"
        if kind == "relu":
            return f"max(0, x{atom[1]})"
        if kind == "sq":
            return f"(x{atom[1]}^2)"
        if kind == "int":
            return f"(x{atom[1]}*x{atom[2]})"
        return "<?>"

    def __str__(self):
        check_is_fitted(self, ["selected_atoms_", "intercept_", "coef_", "feature_importance_"])
        lines = ["Sparse Atom Hybrid Regressor", "equation:"]
        terms = [f"{float(self.intercept_):+.4f}"]
        n_nonzero = 0
        for c, atom in zip(self.coef_, self.selected_atoms_):
            if abs(float(c)) < float(self.min_coef):
                continue
            terms.append(f"({float(c):+.4f})*{self._atom_to_str(atom)}")
            n_nonzero += 1
        lines.append("  y = " + " + ".join(terms))

        lines.append("")
        lines.append("active_atoms:")
        if n_nonzero == 0:
            lines.append("  none (constant model)")
        else:
            for c, atom in zip(self.coef_, self.selected_atoms_):
                if abs(float(c)) >= float(self.min_coef):
                    lines.append(f"  {self._atom_to_str(atom)} : {float(c):+.4f}")

        lines.append("")
        lines.append("feature_importance_order:")
        order = np.argsort(self.feature_importance_)[::-1]
        for j in order[: min(10, self.n_features_in_)]:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")

        lines.append("")
        lines.append(f"operation_count_estimate: {1 + 2 * n_nonzero}")
        lines.append("manual_prediction: evaluate each listed atom from x-values, multiply by coefficient, then sum with intercept.")
        return "\n".join(lines)
# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SparseAtomHybridRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseAtomHybrid_v1"
model_description = "Validation-greedy sparse symbolic regressor over linear, abs, relu, square, and screened interaction atoms with compact explicit equation"
model_defs = [(model_shorthand_name, SparseAtomHybridRegressor())]


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
