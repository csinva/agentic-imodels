"""Tree model in the reference JSON format with prediction utilities.

A tree is a nested ``dict``.  Internal nodes look like::

    {"feature": 3, "name": "age", "relation": ">=", "reference": 25,
     "type": "integral", "true": {...}, "false": {...}}

and leaves like::

    {"prediction": 1, "name": "class", "loss": 0.0123, "complexity": 0.01}

which is exactly what the reference C++ implementation emits (``Model::to_json``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _leaf_matches(value, node) -> bool:
    rel = node["relation"]
    ref = node["reference"]
    if rel == ">=":
        try:
            return bool(value >= ref)
        except TypeError:
            return False
    if rel == "<=":
        try:
            return bool(value <= ref)
        except TypeError:
            return False
    # equality (categorical or binary numeric)
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    try:
        return bool(value == ref)
    except TypeError:
        return False


class TreeClassifier:
    """Interactive wrapper around a JSON tree (mirrors ``python/model/tree_classifier.py``)."""

    def __init__(self, source: dict):
        self.source = source

    # ---------------------------------------------------------- prediction
    def _find_leaf(self, sample):
        node = self.source
        while "prediction" not in node:
            value = sample[node["feature"]]
            node = node["true"] if _leaf_matches(value, node) else node["false"]
        return node

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_numpy(dtype=object)
        else:
            rows = np.asarray(X, dtype=object)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
        return np.array([self._find_leaf(row)["prediction"] for row in rows], dtype=object)

    def predict_fast(self, X) -> np.ndarray:
        """Vectorised prediction for numeric-only / categorical feature matrices."""
        if isinstance(X, pd.DataFrame):
            frame = X
        else:
            arr = np.asarray(X)
            frame = pd.DataFrame(arr)
        n = frame.shape[0]
        out = np.empty(n, dtype=object)
        idx = np.arange(n)
        self._predict_rec(self.source, frame, idx, out)
        return out

    def _predict_rec(self, node, frame, idx, out):
        if "prediction" in node:
            out[idx] = node["prediction"]
            return
        col = frame.iloc[idx, node["feature"]]
        rel = node["relation"]
        ref = node["reference"]
        if rel == ">=":
            vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mask = vals >= ref
        elif rel == "<=":
            vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mask = vals <= ref
        else:
            if node.get("type") == "categorical":
                mask = np.array([(v == ref) if not (isinstance(v, float) and np.isnan(v)) else False
                                 for v in col.to_numpy(dtype=object)], dtype=bool)
            else:
                vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
                with np.errstate(invalid="ignore"):
                    mask = vals == ref
        self._predict_rec(node["true"], frame, idx[mask], out)
        self._predict_rec(node["false"], frame, idx[~mask], out)

    def error(self, X, y, weight=None):
        y = np.asarray(y).ravel()
        pred = self.predict_fast(X)
        miss = (pred != y).astype(np.float64)
        if weight is None:
            return float(miss.mean())
        weight = np.asarray(weight, dtype=np.float64).ravel()
        return float((miss * weight).sum() / weight.sum())

    def score(self, X, y, weight=None):
        return 1.0 - self.error(X, y, weight=weight)

    # ------------------------------------------------------------ structure
    def _all_leaves(self, node=None):
        node = self.source if node is None else node
        if "prediction" in node:
            return [node]
        return self._all_leaves(node["true"]) + self._all_leaves(node["false"])

    def leaves(self) -> int:
        return len(self._all_leaves())

    def nodes(self) -> int:
        def rec(node):
            if "prediction" in node:
                return 1
            return 1 + rec(node["true"]) + rec(node["false"])
        return rec(self.source)

    def maximum_depth(self) -> int:
        def rec(node):
            if "prediction" in node:
                return 1
            return 1 + max(rec(node["true"]), rec(node["false"]))
        return rec(self.source)

    def loss(self) -> float:
        return float(sum(leaf["loss"] for leaf in self._all_leaves()))

    def complexity(self) -> float:
        return float(sum(leaf["complexity"] for leaf in self._all_leaves()))

    def risk(self) -> float:
        return self.loss() + self.complexity()

    def __len__(self):
        return self.leaves()

    def json(self, indent: int | None = 2) -> str:
        return json.dumps(self.source, indent=indent, cls=NumpyEncoder)

    def features(self) -> list:
        feats = []

        def rec(node):
            if "prediction" in node:
                return
            feats.append(node["feature"])
            rec(node["true"])
            rec(node["false"])

        rec(self.source)
        return sorted(set(feats))

    def __str__(self):
        lines = []

        def rec(node, depth):
            pad = "    " * depth
            if "prediction" in node:
                lines.append(f"{pad}{node['name']} = {node['prediction']!r}  (loss={node['loss']:.6g})")
                return
            lines.append(f"{pad}if {node['name']} {node['relation']} {node['reference']!r} then:")
            rec(node["true"], depth + 1)
            lines.append(f"{pad}else:")
            rec(node["false"], depth + 1)

        rec(self.source, 0)
        return "\n".join(lines)

    __repr__ = __str__
