"""Wrapper that runs STreeD (``baselines/pystreed``) with the solver interface used
by ``evaluate.evaluate_solver`` (``fit``, ``tree_``, ``optimal_``, ``time_``).

STreeD's ``cost-complex-accuracy`` task minimises
``misclassifications + cost_complexity * n * (branching nodes)``; divided by
``n`` that is ``error rate + λ * (leaves - 1)``, the GOSDT objective minus the
constant λ, so ``cost_complexity = λ`` optimises the same tree.  The features
are binarized with pygosdt_v1's encoder so STreeD searches exactly the same
binary feature space as the other baselines; STreeD additionally needs a depth
cap (``max_depth``, at most 20), which is set high enough not to bind on the
suite.

Used by ``run_baselines.py --rerun`` and ``baselines/benchmarks/run_benchmark.py``.
Requires the ``baselines`` dependency group (``uv sync --group baselines``).
Do not modify.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pystreed import STreeDClassifier

from pygosdt_v1.encoder import BinaryEncoder, TargetEncoder

MAX_DEPTH = 20  # STreeD's hard limit; the cost-complexity bounds keep the search finite


class STreeD:
    """Fit STreeD on the pygosdt_v1 binarization of ``X`` and expose the reference tree schema."""

    def __init__(self, regularization: float, time_limit: float, max_depth: int = MAX_DEPTH):
        self.regularization = regularization
        self.time_limit = time_limit
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y):
        self.encoder_ = BinaryEncoder().fit(X)
        Xb = pd.DataFrame(self.encoder_.transform(X).astype(np.int64),
                          columns=[f"b{j}" for j in range(self.encoder_.n_binary_features)])
        self.target_encoder_ = TargetEncoder().fit(y)
        y_idx = self.target_encoder_.transform(y)
        self.n_binary_features_ = Xb.shape[1]
        self.model_ = STreeDClassifier(
            optimization_task="cost-complex-accuracy", cost_complexity=float(self.regularization),
            max_depth=self.max_depth, max_num_nodes=None, time_limit=float(self.time_limit),
            verbose=False)
        t0 = time.perf_counter()
        self.model_.fit(Xb, y_idx)
        self.time_ = time.perf_counter() - t0
        self.optimal_ = bool(self.model_.fit_result.is_optimal())
        self.stop_reason_ = "optimal" if self.optimal_ else "time"
        self.size_ = ""
        self.iterations_ = ""
        self.lowerbound_ = ""
        self.upperbound_ = ""
        self.tree_ = self._convert(self.model_.get_tree(), Xb, y_idx)
        return self

    def _convert(self, node, Xb: pd.DataFrame, y_idx: np.ndarray) -> dict:
        n = Xb.shape[0]
        if node.is_leaf_node():
            label = int(node.label)
            return {"prediction": self.target_encoder_.inverse(label), "name": "class",
                    "loss": float(np.sum(y_idx != label)) / n, "complexity": float(self.regularization)}
        j = int(node.feature)
        rule = self.encoder_.rules[j]
        mask = Xb.iloc[:, j].to_numpy() == 1
        return {"feature": int(rule["feature"]), "name": rule["name"], "relation": rule["relation"],
                "reference": rule["reference"], "type": rule["type"],
                "true": self._convert(node.right_child, Xb[mask], y_idx[mask]),
                "false": self._convert(node.left_child, Xb[~mask], y_idx[~mask])}
