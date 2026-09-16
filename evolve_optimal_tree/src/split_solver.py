"""Wrapper that runs SPLIT (Babbar et al., ICML 2025; ``baselines/split``) with the
solver interface used by ``evaluate.evaluate_solver``.

SPLIT optimises a shallow prefix of the tree with a GOSDT-style search in
which subtrees beyond ``lookahead_depth`` are evaluated greedily, then fills
each prefix leaf with an optimal GOSDT subtree under the remaining depth
budget.  It uses the same regularization λ (its ``reg``) and, through this
wrapper, the same binarization as every other baseline, but it needs a depth
budget and is a heuristic by construction, so it never certifies optimality:
rows are recorded with status ``heuristic`` and ``exact = approximate``.  It
only supports binary classification, so multi-class pairs (iris) are recorded
with status ``unsupported`` and no tree.

Defaults follow the paper's quick-start (lookahead depth 2, full depth 5).
Requires the ``baselines`` dependency group (``uv sync --group baselines``).
Do not modify.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from split import SPLIT
from split._tree import Leaf

from evaluate import NoModel
from pygosdt_v1.encoder import BinaryEncoder, TargetEncoder

LOOKAHEAD_DEPTH = 2
FULL_DEPTH = 5


class Split:
    def __init__(self, regularization: float, time_limit: float,
                 lookahead_depth: int = LOOKAHEAD_DEPTH, full_depth: int = FULL_DEPTH):
        self.regularization = regularization
        self.time_limit = time_limit
        self.lookahead_depth = lookahead_depth
        self.full_depth = full_depth

    def fit(self, X: pd.DataFrame, y):
        self.encoder_ = BinaryEncoder().fit(X)
        Xb = pd.DataFrame(self.encoder_.transform(X).astype(np.int64),
                          columns=[f"b{j}" for j in range(self.encoder_.n_binary_features)])
        self.target_encoder_ = TargetEncoder().fit(y)
        y_idx = self.target_encoder_.transform(y)
        self.n_binary_features_ = Xb.shape[1]
        if len(self.target_encoder_.classes_) != 2:
            raise NoModel("unsupported", "SPLIT only supports binary classification")
        self.model_ = SPLIT(lookahead_depth_budget=self.lookahead_depth, full_depth_budget=self.full_depth,
                            reg=float(self.regularization), time_limit=max(1, int(self.time_limit)),
                            verbose=False, binarize=False)
        t0 = time.perf_counter()
        self.model_.fit(Xb, pd.Series(y_idx))
        self.time_ = time.perf_counter() - t0
        self.optimal_ = False
        self.stop_reason_ = "heuristic"
        self.size_ = ""
        self.iterations_ = ""
        self.lowerbound_ = ""
        self.upperbound_ = ""
        classes = list(self.model_.clf.classes_)
        self.tree_ = self._convert(self.model_.tree, classes, Xb, y_idx)
        return self

    def _convert(self, node, classes, Xb: pd.DataFrame, y_idx: np.ndarray) -> dict:
        n = Xb.shape[0]
        if isinstance(node, Leaf):
            label = int(classes[node.prediction])
            return {"prediction": self.target_encoder_.inverse(label), "name": "class",
                    "loss": float(np.sum(y_idx != label)) / n, "complexity": float(self.regularization)}
        j = int(node.feature)
        rule = self.encoder_.rules[j]
        mask = Xb.iloc[:, j].to_numpy() == 1
        return {"feature": int(rule["feature"]), "name": rule["name"], "relation": rule["relation"],
                "reference": rule["reference"], "type": rule["type"],
                "true": self._convert(node.left_child, classes, Xb[mask], y_idx[mask]),
                "false": self._convert(node.right_child, classes, Xb[~mask], y_idx[~mask])}
