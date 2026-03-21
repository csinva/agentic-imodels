"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import os
import subprocess
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this — everything below is fair game)
# ---------------------------------------------------------------------------


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree regressor with a rich __str__ for interpretability.

    Uses max_leaf_nodes to control complexity (good tradeoff).
    The __str__ shows:
      - Full tree in text form with feature names and thresholds (export_text)
      - Feature importances sorted by importance
      - Which features are unused (zero importance)
      - Leaf value statistics for orientation
    """

    def __init__(self, max_leaf_nodes=20, min_samples_leaf=10):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = self.feature_names_in_
        n_leaves = self.tree_.get_n_leaves()
        n_nodes = self.tree_.tree_.node_count
        importances = self.tree_.feature_importances_

        lines = [
            f"DecisionTreeRegressor(max_leaf_nodes={self.max_leaf_nodes}, "
            f"min_samples_leaf={self.min_samples_leaf})",
            f"  nodes={n_nodes}, leaves={n_leaves}",
            "",
        ]

        # Full tree text — this is what allows point prediction tracing
        tree_text = export_text(self.tree_, feature_names=names, max_depth=10)
        lines.append("Tree structure (follow from root; leaf values are predictions):")
        lines.append(tree_text)

        # Feature importance ranking
        order = np.argsort(importances)[::-1]
        lines.append("Feature importances (Gini-based, higher = more important):")
        for rank, fi in enumerate(order):
            if importances[fi] > 1e-6:
                direction = self._infer_direction(fi)
                lines.append(
                    f"  {rank+1:2d}. {names[fi]:<25s}  {importances[fi]:.4f}  "
                    f"(net effect: {direction})"
                )

        # Unused features
        unused = [names[fi] for fi in range(len(names)) if importances[fi] <= 1e-6]
        if unused:
            lines.append(f"\nFeatures not used in tree (zero importance): {', '.join(unused)}")

        return "\n".join(lines)

    def _infer_direction(self, feature_idx):
        """Infer whether feature has net positive or negative effect on predictions."""
        t = self.tree_.tree_
        # Look at all splits on this feature and see if higher value -> higher leaf value
        n_nodes = t.node_count
        weighted_effect = 0.0
        for node in range(n_nodes):
            if t.children_left[node] == -1:
                continue  # leaf
            if t.feature[node] != feature_idx:
                continue
            # left child has feature <= threshold (lower), right has feature > threshold (higher)
            left_child = t.children_left[node]
            right_child = t.children_right[node]
            # Use weighted node values
            left_val = t.value[left_child][0][0]
            right_val = t.value[right_child][0][0]
            n_left = t.n_node_samples[left_child]
            n_right = t.n_node_samples[right_child]
            # higher feature (right) vs lower feature (left)
            weighted_effect += (right_val - left_val) * (n_left + n_right)
        if weighted_effect > 0:
            return "positive (higher -> higher prediction)"
        elif weighted_effect < 0:
            return "negative (higher -> lower prediction)"
        else:
            return "mixed"


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("InterpretableRegressor", InterpretableRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    std  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in ALL_TESTS})
    hard = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in HARD_TESTS})
    ins  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in INSIGHT_TESTS})

    # TabArena RMSE
    dataset_rmses = evaluate_all_regressors(model_defs)
    rmse_vals = [v["InterpretableRegressor"] for v in dataset_rmses.values()
                 if not np.isnan(v.get("InterpretableRegressor", float("nan")))]
    mean_rmse = float(np.mean(rmse_vals)) if rmse_vals else float("nan")

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rmse":                          f"{mean_rmse:.6f}" if not np.isnan(mean_rmse) else "",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}",
        "status":                             "",
        "description":                        "InterpretableRegressor",
    }], RESULTS_DIR)

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total} ({n_passed/total:.2%})  "
          f"[std {std}/8  hard {hard}/5  insight {ins}/5]")
    print(f"mean_rmse:     {mean_rmse:.4f}")
    print(f"total_seconds: {time.time() - t0:.1f}s")
