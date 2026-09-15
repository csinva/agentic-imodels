"""
Optimal sparse decision tree autoresearch script.

Defines ``OptimalTreeClassifier``, a scikit-learn compatible solver for

    minimise   misclassification rate + regularization * (number of leaves)

over all decision trees on the binarized features (the GOSDT objective), and an
evaluation loop that scores it on the fixed development suite in ``src/`` and
records the result in ``results/overall_results.csv``.

Usage: uv run optimal_tree.py [--datasets a,b] [--lams 0.1,0.05] [--time-limit 30]

This file is the ONLY file the agent edits.  Everything below the "SOLVER"
banner is fair game; the evaluation loop at the bottom must stay as is.
The starting point is pygosdt_v1 (see pygosdt_v1/ and REPORT.html) flattened
into one file: a memoised depth-first branch-and-bound over capture-set
bitsets with the reference GOSDT bounds and a numba counting kernel.
"""

import argparse
import html as _html  # noqa: F401  (kept for parity with the package modules)
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from evaluate import evaluate_solver, print_summary, record  # noqa: E402
from suite import TIME_LIMIT  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment metadata (update these every iteration)
# ---------------------------------------------------------------------------

MODEL_NAME = "pygosdt_v1_flat"
DESCRIPTION = ("pygosdt_v1 flattened into one file: memoised depth-first branch-and-bound over "
               "big-int capture sets with equivalent-points, leaf-support, look-ahead, "
               "similar-support and threshold-exchange bounds; numba popcount kernel")

# ===========================================================================
# SOLVER (edit freely below this line)
# ===========================================================================


# ------------------------------------------------------------------ fastbits

HAVE_NUMBA = True  # numba is a declared dependency; engine="python" remains selectable


@njit(cache=True, nogil=True)
def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True, nogil=True)
def child_counts(F, masks, out):
    """out[j, r] = popcount(F[j] & masks[r]) for every feature j and mask r."""
    m, W = F.shape
    R = masks.shape[0]
    for j in range(m):
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[j, r] = acc
    return out


@njit(cache=True, nogil=True)
def child_counts_subset(F, feats, masks, out):
    """Same as ``child_counts`` restricted to the rows ``feats`` of ``F``."""
    W = F.shape[1]
    R = masks.shape[0]
    for t in range(feats.shape[0]):
        j = feats[t]
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[t, r] = acc
    return out


def pack_columns(Xb: np.ndarray) -> np.ndarray:
    """(n, m) bool -> (m, W) uint64 with row i of the data in bit i."""
    n, m = Xb.shape
    W = (n + 63) // 64
    packed = np.packbits(np.ascontiguousarray(Xb.T), axis=1, bitorder="little")
    padded = np.zeros((m, W * 8), dtype=np.uint8)
    padded[:, :packed.shape[1]] = packed
    return np.ascontiguousarray(padded.view(np.uint64))


def int_to_words(value: int, W: int) -> np.ndarray:
    return np.frombuffer(value.to_bytes(W * 8, "little"), dtype=np.uint64)


def warm_up():
    """Trigger JIT compilation (cached on disk afterwards)."""
    F = np.zeros((2, 1), dtype=np.uint64)
    masks = np.zeros((1, 1), dtype=np.uint64)
    child_counts(F, masks, np.zeros((2, 1), dtype=np.uint64))
    child_counts_subset(F, np.zeros(1, dtype=np.int64), masks, np.zeros((1, 1), dtype=np.uint64))

# ------------------------------------------------------------------ encoder

_MISSING_STRINGS = {"", "NULL", "null", "Null", "NA", "na", "NaN", "nan", "N/A", "n/a"}


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])


def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    return pd.api.types.is_numeric_dtype(s)


class BinaryEncoder:
    """Fit/transform arbitrary features into a boolean split matrix.

    Attributes after ``fit``:

    rules : list of dict
        One entry per binary feature with keys ``feature`` (source column
        index), ``name`` (source column name), ``relation`` (``">="`` or
        ``"=="``), ``reference`` (threshold or category value) and ``type``
        (``"integral"``, ``"rational"`` or ``"categorical"``).
    groups : list of list of int
        Indices of binary features (in threshold order) that belong to the
        same ordinal source column with more than one threshold.
    """

    def __init__(self, drop_duplicate_columns: bool = True):
        self.drop_duplicate_columns = drop_duplicate_columns
        self.rules: list[dict] = []
        self.groups: list[list[int]] = []
        self.feature_names: list[str] = []
        self.n_source_features = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X) -> "BinaryEncoder":
        X = _to_dataframe(X)
        self.feature_names = [str(c) for c in X.columns]
        self.n_source_features = X.shape[1]
        rules: list[dict] = []
        groups: list[list[int]] = []

        for j, col in enumerate(X.columns):
            s = X[col]
            name = str(col)
            if _is_numeric_series(s):
                values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                has_missing = finite.shape[0] != values.shape[0]
                uniq = np.unique(finite)
                if uniq.shape[0] <= 1:
                    continue
                integral = bool(np.all(np.equal(np.mod(uniq, 1), 0)))
                kind = "integral" if integral else "rational"
                if uniq.shape[0] == 2 and not has_missing:
                    ref = uniq[1]
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": int(ref) if integral else float(ref), "type": kind,
                    })
                    continue
                start = len(rules)
                for a, b in zip(uniq[:-1], uniq[1:]):
                    if integral:
                        ref = int(b)
                    else:
                        ref = float(0.5 * (a + b))
                    rules.append({
                        "feature": j, "name": name, "relation": ">=",
                        "reference": ref, "type": kind,
                    })
                groups.append(list(range(start, len(rules))))
            else:
                raw = s.to_numpy(dtype=object)
                mask = np.array([not _is_missing(v) for v in raw], dtype=bool)
                present = raw[mask]
                has_missing = present.shape[0] != raw.shape[0]
                uniq = sorted(set(present.tolist()), key=lambda v: str(v))
                if len(uniq) <= 1:
                    continue
                if len(uniq) == 2 and not has_missing:
                    uniq = uniq[1:]
                for v in uniq:
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": v, "type": "categorical",
                    })

        self.rules = rules
        self.groups = groups

        if self.drop_duplicate_columns and rules:
            Xb = self._apply_rules(X, rules)
            keep = _unique_partitions(Xb)
            if keep.shape[0] != len(rules):
                remap = {old: new for new, old in enumerate(keep.tolist())}
                self.rules = [rules[i] for i in keep.tolist()]
                self.groups = [
                    g2 for g2 in ([remap[i] for i in g if i in remap] for g in groups)
                    if len(g2) > 1
                ]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X) -> np.ndarray:
        X = _to_dataframe(X)
        if X.shape[1] != self.n_source_features:
            raise ValueError(
                f"expected {self.n_source_features} feature columns, got {X.shape[1]}"
            )
        return self._apply_rules(X, self.rules)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_binary_features(self) -> int:
        return len(self.rules)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _apply_rules(X: pd.DataFrame, rules: list[dict]) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, len(rules)), dtype=bool)
        cache: dict[int, np.ndarray] = {}
        for k, rule in enumerate(rules):
            j = rule["feature"]
            if rule["relation"] == ">=":
                if j not in cache:
                    cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                col = cache[j]
                with np.errstate(invalid="ignore"):
                    out[:, k] = col >= rule["reference"]
            else:
                if rule["type"] == "categorical":
                    col = X.iloc[:, j].to_numpy(dtype=object)
                    ref = rule["reference"]
                    out[:, k] = np.array([(not _is_missing(v)) and v == ref for v in col], dtype=bool)
                else:
                    if j not in cache:
                        cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                    col = cache[j]
                    with np.errstate(invalid="ignore"):
                        out[:, k] = col == rule["reference"]
        return out


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v in _MISSING_STRINGS:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _unique_partitions(Xb: np.ndarray) -> np.ndarray:
    """Return indices of columns whose induced row partition is new.

    A column and its complement induce the same partition, so columns are
    canonicalised by flipping them when their first row is ``True``.
    """
    if Xb.shape[1] == 0:
        return np.arange(0)
    canon = Xb ^ Xb[0:1, :]
    packed = np.packbits(canon, axis=0)
    seen: dict[bytes, int] = {}
    keep = []
    for k in range(packed.shape[1]):
        key = packed[:, k].tobytes()
        if key in seen:
            continue
        seen[key] = k
        keep.append(k)
    return np.array(keep, dtype=np.int64)


class TargetEncoder:
    """Map arbitrary labels to contiguous integer class indices."""

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y) -> "TargetEncoder":
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.asarray(y).ravel()
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= self.classes_.shape[0]) or np.any(self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y):
            raise ValueError("labels contain classes not seen during fit")
        return idx.astype(np.int64)

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse(self, idx: int):
        v = self.classes_[idx]
        if isinstance(v, np.generic):
            return v.item()
        return v

# ------------------------------------------------------------------ dataset

def column_to_int(col: np.ndarray) -> int:
    """Pack a boolean vector into an int whose bit ``i`` is ``col[i]``."""
    packed = np.packbits(np.ascontiguousarray(col, dtype=bool), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def int_to_column(value: int, n: int) -> np.ndarray:
    nbytes = (n + 7) // 8
    raw = np.frombuffer(value.to_bytes(nbytes, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


class BitDataset:
    """Binary features, class targets and misclassification costs as bitsets.

    Parameters
    ----------
    Xb : (n, m) bool array of binary split features.
    y : (n,) int array of class indices in ``[0, n_classes)``.
    n_classes : number of classes.
    costs : optional (K, K) matrix; ``costs[i, j]`` is the cost of predicting
        class ``i`` when the true class is ``j``.  Defaults to ``1/n`` off the
        diagonal (unweighted misclassification rate).
    balance : if True and ``costs`` is None, use ``1 / (K * count_j)`` so every
        class carries the same total weight (the reference ``balance`` flag).
    """

    def __init__(self, Xb: np.ndarray, y: np.ndarray, n_classes: int,
                 costs: np.ndarray | None = None, balance: bool = False):
        Xb = np.ascontiguousarray(Xb, dtype=bool)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, m = Xb.shape
        if y.shape[0] != n:
            raise ValueError("X and y have different numbers of rows")
        self.n = n
        self.m = m
        self.K = int(n_classes)
        self.full = (1 << n) - 1
        self.features = [column_to_int(Xb[:, j]) for j in range(m)]
        self.targets = [column_to_int(y == k) for k in range(self.K)]
        self.class_counts = np.array([int(t.bit_count()) for t in self.targets], dtype=np.int64)

        # ---- cost matrix and its aggregations (Dataset::aggregate_cost_matrix)
        K = self.K
        if costs is not None:
            C = np.asarray(costs, dtype=np.float64)
            if C.shape != (K, K):
                raise ValueError(f"costs must have shape {(K, K)}")
            self.uniform = False
        elif balance:
            C = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    if i != j:
                        C[i, j] = 1.0 / (K * max(int(self.class_counts[j]), 1))
            self.uniform = False
        else:
            C = np.full((K, K), 1.0 / n)
            np.fill_diagonal(C, 0.0)
            self.uniform = True
        self.costs = C
        self.match_costs = np.diag(C).copy()
        self.max_costs = C.max(axis=0)
        self.min_costs = C.min(axis=0)
        self.diff_costs = self.max_costs - self.min_costs
        self._diff_list = [float(v) for v in self.diff_costs]
        mismatch = np.full(K, np.inf)
        for j in range(K):
            for i in range(K):
                if i != j:
                    mismatch[j] = min(mismatch[j], C[i, j])
        if K == 1:
            mismatch[:] = 0.0
        self.mismatch_costs = mismatch

        # ---- equivalent points: rows with identical features but different labels
        _, inverse = np.unique(Xb, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        n_groups = int(inverse.max()) + 1 if n else 0
        dist = np.zeros((n_groups, K), dtype=np.float64)
        np.add.at(dist, (inverse, y), 1.0)
        group_cost = dist @ C.T                       # [g, i] = cost of predicting i for group g
        minimizer = np.argmin(group_cost, axis=1)     # first minimal index, like the reference
        majority_rows = minimizer[inverse] == y
        self.majority = column_to_int(majority_rows)
        self.minority = self.full & ~self.majority
        self.majority_by_class = [self.majority & t for t in self.targets]
        self.minority_by_class = [self.minority & t for t in self.targets]

        # Fast paths for the equivalent-points loss.
        self.zero_diagonal = bool(np.all(self.match_costs == 0.0))
        self.equal_mismatch = bool(np.all(self.mismatch_costs == self.mismatch_costs[0]))

        # Packed 64-bit word representation used by the numba kernels.
        self.W = (n + 63) // 64
        self.F_words = pack_columns(Xb) if m else np.zeros((0, self.W), dtype=np.uint64)
        self.target_words = [int_to_words(t, self.W) for t in self.targets]
        self.minority_words = int_to_words(self.minority, self.W)
        self.minority_by_class_words = [int_to_words(v, self.W) for v in self.minority_by_class]
        self.majority_by_class_words = [int_to_words(v, self.W) for v in self.majority_by_class]

    # ------------------------------------------------------------------
    def leaf_stats(self, capture: int):
        """Return ``(count, dist, max_loss, min_loss, potential, prediction)``.

        ``max_loss`` is the loss of the best single label (the leaf loss),
        ``min_loss`` the equivalent-points lower bound on any tree's loss and
        ``potential`` the maximal loss reduction any split could achieve.
        """
        dist = np.array([int((capture & t).bit_count()) for t in self.targets], dtype=np.float64)
        count = int(dist.sum())
        pred_costs = self.costs @ dist
        prediction = int(np.argmin(pred_costs))
        max_loss = float(pred_costs[prediction])
        potential = float(self.diff_costs @ dist)
        min_loss = self.equivalent_loss(capture)
        return count, dist, max_loss, min_loss, potential, prediction

    def equivalent_loss(self, capture: int) -> float:
        if self.zero_diagonal:
            if self.equal_mismatch:
                return float(self.mismatch_costs[0]) * (capture & self.minority).bit_count()
            return float(sum(
                float(self.mismatch_costs[k]) * (capture & self.minority_by_class[k]).bit_count()
                for k in range(self.K)
            ))
        total = 0.0
        for k in range(self.K):
            total += float(self.match_costs[k]) * (capture & self.majority_by_class[k]).bit_count()
            total += float(self.mismatch_costs[k]) * (capture & self.minority_by_class[k]).bit_count()
        return total

    def distance(self, capture: int, i: int, j: int, needed: float = np.inf) -> float:
        """Similar-support distance between features ``i`` and ``j`` on ``capture``.

        Returns ``min(cost of rows where i != j, cost of rows where i == j)``.
        If the first term already exceeds ``needed`` the caller cannot prune, so
        the second term is skipped and the first is returned.
        """
        differ = capture & (self.features[i] ^ self.features[j])
        pos = 0.0
        for k in range(self.K):
            d = self._diff_list[k]
            if d != 0.0:
                pos += d * (differ & self.targets[k]).bit_count()
        if pos >= needed:
            return pos
        agree = capture & ~differ
        neg = 0.0
        for k in range(self.K):
            d = self._diff_list[k]
            if d != 0.0:
                neg += d * (agree & self.targets[k]).bit_count()
        return min(pos, neg)

# ------------------------------------------------------------------ model

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

# ------------------------------------------------------------------ optimizer

EPS = 1e-10


class TimeLimitReached(Exception):
    """Raised inside the search when the time or memory limit is hit."""


def _rss_bytes() -> int:
    """Current resident set size of this process in bytes (0 if unavailable).

    ``resource.getrusage`` only reports the lifetime peak, which would keep
    tripping the guard after one large search, so the live value is read from
    ``/proc`` on Linux and from ``ps`` elsewhere.
    """
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


class Node:
    __slots__ = ("key", "count", "leaf_risk", "prediction", "lb", "ub", "split", "solved")

    def __init__(self, key: int, count: int, leaf_risk: float, prediction: int,
                 lb: float, solved: bool):
        self.key = key
        self.count = count
        self.leaf_risk = leaf_risk
        self.prediction = prediction
        self.lb = lb
        self.ub = leaf_risk
        self.split = -1
        self.solved = solved


class Optimizer:
    def __init__(self, data: BitDataset, regularization: float, *,
                 groups: list[list[int]] | None = None,
                 time_limit: float = 0.0,
                 look_ahead: bool = True,
                 similar_support: bool = True,
                 feature_exchange: bool = True,
                 continuous_feature_exchange: bool = True,
                 greedy_init: bool = True,
                 upperbound: float = 0.0,
                 engine: str = "auto",
                 memory_limit: int = 0,
                 verbose: bool = False):
        self.data = data
        self.memory_limit = int(memory_limit)
        self.stop_reason = ""
        if engine == "auto":
            engine = "numba" if HAVE_NUMBA else "python"
        if engine == "numba" and not HAVE_NUMBA:
            raise ImportError("numba is not installed; use engine='python'")
        self.engine = engine
        if engine == "numba":
            warm_up()
        self.lam = float(regularization)
        self.time_limit = float(time_limit)
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.greedy_init = greedy_init
        self.upperbound = float(upperbound)
        self.verbose = verbose
        self.memo: dict[int, Node] = {}
        self.iterations = 0          # number of subproblem expansions
        self.start_time = 0.0
        self.elapsed = 0.0
        self.optimal = False

        # ordinal neighbour map used by the continuous feature exchange bound
        # ``feature_exchange`` is accepted for configuration compatibility only: the
        # reference's pairwise version prunes whole subtrees with parent bounds and
        # is not exact, so it is not applied (see README).
        self.next_in_group = np.full(data.m, -1, dtype=np.int64)
        for g in (groups or []):
            for a, b in zip(g[:-1], g[1:]):
                self.next_in_group[a] = b

        self._has_groups = bool(np.any(self.next_in_group >= 0))
        self._pos_buffer = np.full(data.m, -1, dtype=np.int64)

        self._costs_T = data.costs.T.copy()
        self._diff = data.diff_costs.copy()

    # ------------------------------------------------------------------ API
    def run(self) -> Node:
        self.start_time = time.perf_counter()
        root_key = self.data.full
        root = self._make_node(root_key)
        features = np.arange(self.data.m, dtype=np.int64)
        if self.upperbound > 0.0 and self.upperbound < root.ub:
            root.ub = self.upperbound  # trusted external bound (as in the reference)
        try:
            if self.greedy_init and not root.solved:
                self._greedy(root, features)
            self._solve(root, root.ub, features)
            self.optimal = True
            self.stop_reason = "optimal"
        except TimeLimitReached as exc:
            self.optimal = False
            self.stop_reason = str(exc)
        self.elapsed = time.perf_counter() - self.start_time
        return root

    # ------------------------------------------------------------- nodes
    def _make_node(self, key: int) -> Node:
        node = self.memo.get(key)
        if node is not None:
            return node
        count, dist, max_loss, min_loss, potential, prediction = self.data.leaf_stats(key)
        leaf_risk = max_loss + self.lam
        lb, solved = self._initial_bounds(count, max_loss, min_loss, potential, leaf_risk)
        node = Node(key, count, leaf_risk, prediction, lb, solved)
        self.memo[key] = node
        return node

    def _initial_bounds(self, count, max_loss, min_loss, potential, leaf_risk):
        lam = self.lam
        if (count <= 1
                or 1.0 - min_loss < lam
                or max_loss - min_loss < lam
                or potential < 2.0 * lam):
            return leaf_risk, True
        return min(leaf_risk, min_loss + 2.0 * lam), False

    # ------------------------------------------------------------ children
    def _child_statistics(self, node: Node, features: np.ndarray):
        """Vectorised statistics of the left/right child of every candidate split.

        Returns ``(feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved)``
        restricted to features that actually split the capture set.
        """
        data = self.data
        key = node.key
        K = data.K
        lam = self.lam
        total = node.count

        if self.engine == "numba":
            L, lmin, dist, min_total = self._counts_numba(key, features)
        else:
            L, lmin, dist, min_total = self._counts_python(key, features)

        lsum = L.sum(axis=1)
        valid = (lsum > 0) & (lsum < total)
        if not valid.any():
            return None
        feats = features[valid]
        L = L[valid]
        lsum = lsum[valid]
        lmin = lmin[valid]
        rmin = min_total - lmin
        R = dist[None, :] - L
        rsum = total - lsum

        l_max = (L @ self._costs_T).min(axis=1)
        r_max = (R @ self._costs_T).min(axis=1)
        l_pot = L @ self._diff
        r_pot = R @ self._diff
        l_leaf = l_max + lam
        r_leaf = r_max + lam
        l_solved = (lsum <= 1) | (1.0 - lmin < lam) | (l_max - lmin < lam) | (l_pot < 2 * lam)
        r_solved = (rsum <= 1) | (1.0 - rmin < lam) | (r_max - rmin < lam) | (r_pot < 2 * lam)
        l_lb = np.where(l_solved, l_leaf, np.minimum(l_leaf, lmin + 2 * lam))
        r_lb = np.where(r_solved, r_leaf, np.minimum(r_leaf, rmin + 2 * lam))
        return feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved

    def _counts_python(self, key: int, features: np.ndarray):
        """Per-feature left-child class counts and equivalent-points loss (big ints)."""
        data = self.data
        K = data.K
        F = data.features
        CT = [key & t for t in data.targets]
        dist = np.array([int(ct.bit_count()) for ct in CT], dtype=np.float64)
        mf = features.shape[0]
        L = np.empty((mf, K), dtype=np.float64)
        for k in range(K):
            ct = CT[k]
            L[:, k] = [(ct & F[j]).bit_count() for j in features]
        if data.zero_diagonal and data.equal_mismatch:
            w = float(data.mismatch_costs[0])
            CM = key & data.minority
            lmin = np.array([(CM & F[j]).bit_count() for j in features], dtype=np.float64) * w
            min_total = w * CM.bit_count()
        else:
            lmin = np.zeros(mf)
            min_total = 0.0
            pairs = [(data.minority_by_class[k], float(data.mismatch_costs[k])) for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class[k], float(data.match_costs[k])) for k in range(K)]
            for mask, w in pairs:
                if w == 0.0:
                    continue
                CMk = key & mask
                lmin += np.array([(CMk & F[j]).bit_count() for j in features], dtype=np.float64) * w
                min_total += w * CMk.bit_count()
        return L, lmin, dist, min_total

    def _counts_numba(self, key: int, features: np.ndarray):
        """Same as ``_counts_python`` using the packed-word numba kernel."""
        data = self.data
        K = data.K
        kw = int_to_words(key, data.W)
        masks = [kw & tw for tw in data.target_words]
        weights = []
        min_total = 0.0
        if data.zero_diagonal and data.equal_mismatch:
            w = float(data.mismatch_costs[0])
            masks.append(kw & data.minority_words)
            weights.append(w)
            min_total = w * (key & data.minority).bit_count()
        else:
            pairs = [(data.minority_by_class[k], data.minority_by_class_words[k], float(data.mismatch_costs[k]))
                     for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class[k], data.majority_by_class_words[k], float(data.match_costs[k]))
                          for k in range(K)]
            for mask_int, mask_words, w in pairs:
                if w == 0.0:
                    continue
                masks.append(kw & mask_words)
                weights.append(w)
                min_total += w * (key & mask_int).bit_count()
        M = np.stack(masks)
        out = np.empty((features.shape[0], M.shape[0]), dtype=np.uint64)
        child_counts_subset(data.F_words, features, M, out)
        counts = out.astype(np.float64)
        L = counts[:, :K]
        dist = np.array([int((key & t).bit_count()) for t in data.targets], dtype=np.float64)
        lmin = np.zeros(features.shape[0])
        for r, w in enumerate(weights):
            lmin += counts[:, K + r] * w
        return L, lmin, dist, min_total

    def _child_node(self, key: int, count: int, leaf: float, lb: float, solved: bool) -> Node:
        node = self.memo.get(key)
        if node is not None:
            return node
        # prediction must be recomputed: costs may be non-uniform
        _, dist, max_loss, _, _, prediction = self.data.leaf_stats(key)
        node = Node(key, count, leaf, prediction, lb, solved)
        self.memo[key] = node
        return node

    # --------------------------------------------------------------- greedy
    def _greedy(self, node: Node, features: np.ndarray, depth: int = 0) -> float:
        """Greedy dive that seeds ``ub``/``split`` along its path."""
        if node.solved or depth > 30:
            return node.ub
        stats = self._child_statistics(node, features)
        if stats is None:
            node.solved = True
            node.lb = node.ub = node.leaf_risk
            return node.ub
        feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved = stats
        immediate = l_leaf + r_leaf
        i = int(np.argmin(immediate))
        if immediate[i] >= node.leaf_risk - EPS:
            return node.ub
        j = int(feats[i])
        lkey = node.key & self.data.features[j]
        rkey = node.key ^ lkey
        left = self._child_node(lkey, 0, float(l_leaf[i]), float(l_lb[i]), bool(l_solved[i]))
        right = self._child_node(rkey, 0, float(r_leaf[i]), float(r_lb[i]), bool(r_solved[i]))
        left.count = lkey.bit_count()
        right.count = rkey.bit_count()
        value = self._greedy(left, feats, depth + 1) + self._greedy(right, feats, depth + 1)
        if value < node.ub:
            node.ub = value
            node.split = j
        return node.ub

    # ---------------------------------------------------------------- solve
    def _solve(self, node: Node, budget: float, features: np.ndarray) -> None:
        """Establish ``node.lb == node.ub`` if the optimum is within ``budget``,
        otherwise prove ``node.lb > budget``."""
        if node.solved or node.lb > budget + EPS:
            return
        self.iterations += 1
        if (self.iterations & 63) == 0:
            if self.time_limit > 0.0 and time.perf_counter() - self.start_time > self.time_limit:
                raise TimeLimitReached("time")
            if self.memory_limit > 0 and (self.iterations & 1023) == 0 and _rss_bytes() > self.memory_limit:
                raise TimeLimitReached("memory")

        stats = self._child_statistics(node, features)
        if stats is None:
            node.lb = node.ub = node.leaf_risk
            node.split = -1
            node.solved = True
            return
        feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved = stats
        data = self.data
        F = data.features
        key = node.key
        memo = self.memo
        split_lb = l_lb + r_lb
        split_ub = l_leaf + r_leaf

        best = node.ub
        best_split = node.split
        # immediate upper bound: both children as leaves
        i = int(np.argmin(split_ub))
        if split_ub[i] < best - EPS:
            best = float(split_ub[i])
            best_split = int(feats[i])
        bound = min(budget, best)

        # continuous feature exchange: drop thresholds dominated by a neighbour
        if self.continuous_feature_exchange and self._has_groups:
            active = self._continuous_exchange(feats, l_lb, l_leaf, r_lb, r_leaf)
        else:
            active = np.ones(feats.shape[0], dtype=bool)
        within = split_lb <= bound + EPS
        cand = np.flatnonzero(active & within)
        # splits rejected by the cheap filter still bound the node from below
        rejected = split_lb[active & ~within]
        min_pruned = float(rejected.min()) if rejected.shape[0] else np.inf

        # Candidates are visited in increasing order of their cheap lower bound, so
        # the loop can stop at the first one exceeding the budget.  Memoised bounds
        # of existing children are consulted lazily, only for visited candidates.
        order = cand[np.lexsort((split_ub[cand], split_lb[cand]))].tolist()
        sim = self.similar_support
        look_ahead = self.look_ahead
        known_lb = {}  # feature -> proven lower bound of its split at this node

        for i in order:
            raw = float(split_lb[i])
            if raw > bound + EPS:
                if raw < min_pruned:
                    min_pruned = raw
                break
            j = int(feats[i])
            lkey = key & F[j]
            rkey = key ^ lkey
            ln = memo.get(lkey)
            rn = memo.get(rkey)
            if ln is None:
                llb, lub = float(l_lb[i]), float(l_leaf[i])
            else:
                llb, lub = ln.lb, ln.ub
            if rn is None:
                rlb, rub = float(r_lb[i]), float(r_leaf[i])
            else:
                rlb, rub = rn.lb, rn.ub
            sub = lub + rub
            if sub < best - EPS:
                best = sub
                best_split = j
                bound = min(budget, best)
            slb = llb + rlb
            if sim:
                # similar support bound from neighbouring features already resolved
                # here; the distance is only computed when it could prune
                for nb in (j - 1, j + 1):
                    nlb = known_lb.get(nb)
                    if nlb is not None and nlb > bound + EPS and nlb > slb:
                        d = data.distance(key, j, nb, nlb - slb)
                        if nlb - d > slb:
                            slb = nlb - d
            if slb > bound + EPS:
                if slb < min_pruned:
                    min_pruned = slb
                known_lb[j] = slb
                continue
            if ln is None:
                ln = self._child_node(lkey, lkey.bit_count(), lub, llb, bool(l_solved[i]))
            if rn is None:
                rn = self._child_node(rkey, rkey.bit_count(), rub, rlb, bool(r_solved[i]))

            # solve the child with the larger lower bound first (more likely to prune);
            # with look-ahead the child only gets the budget its sibling leaves over
            first, second = (ln, rn) if ln.lb >= rn.lb else (rn, ln)
            self._solve(first, bound - second.lb if look_ahead else bound, feats)
            if first.lb > bound - second.lb + EPS:
                slb = first.lb + second.lb
                if slb < min_pruned:
                    min_pruned = slb
                known_lb[j] = slb
                continue
            self._solve(second, bound - first.ub if look_ahead else bound, feats)
            if second.lb > bound - first.ub + EPS:
                slb = first.ub + second.lb
                if slb < min_pruned:
                    min_pruned = slb
                known_lb[j] = slb
                continue
            value = first.ub + second.ub
            known_lb[j] = value
            if value < best - EPS:
                best = value
                best_split = j
                bound = min(budget, best)

        node.ub = best
        node.split = best_split
        if best <= budget + EPS:
            node.lb = best
            node.solved = True
        else:
            node.lb = max(node.lb, min(best, min_pruned))

    def _continuous_exchange(self, feats, l_lb, l_leaf, r_lb, r_leaf):
        """Return a mask of splits not dominated by the next threshold of their column.

        Binary feature ``j`` is ``x >= t_j``; its *left* child is the rows where
        it holds.  For consecutive thresholds ``t_i < t_k`` of one ordinal
        column, ``left_i ⊇ left_k`` and ``right_i ⊆ right_k``.  The optimal
        risk is monotone under set inclusion (restricting an optimal tree to a
        subset never increases loss or leaves), so ``R(left_i) >= R(left_k)`` and
        ``R(right_i) <= R(right_k)``.  Hence if ``lb(right_i) >= ub(right_k)``
        split ``k`` dominates split ``i``, and if ``lb(left_k) >= ub(left_i)``
        split ``i`` dominates split ``k``.  Domination chains never form cycles
        because the two rules are mutually exclusive on the same pair.
        """
        mf = feats.shape[0]
        pos = self._pos_buffer
        pos[feats] = np.arange(mf)
        nb = self.next_in_group[feats]
        has = nb >= 0
        k = np.where(has, pos[np.where(has, nb, 0)], -1)
        has &= k >= 0
        pos[feats] = -1
        active = np.ones(mf, dtype=bool)
        idx = np.flatnonzero(has)
        if idx.shape[0] == 0:
            return active
        kk = k[idx]
        dominated_i = r_lb[idx] >= r_leaf[kk] - EPS
        dominated_k = (~dominated_i) & (l_lb[kk] >= l_leaf[idx] - EPS)
        active[idx[dominated_i]] = False
        active[kk[dominated_k]] = False
        return active

    # ------------------------------------------------------------ extraction
    def extract(self, node: Node, features_hint=None) -> dict:
        """Return the memoised tree below ``node`` as nested dicts of
        ``{"feature": j, "true": ..., "false": ...}`` / ``{"prediction": k, "key": capture}``."""
        if node.split < 0:
            return {"prediction": node.prediction, "key": node.key, "count": node.count}
        j = node.split
        lkey = node.key & self.data.features[j]
        rkey = node.key ^ lkey
        left = self._make_node(lkey)
        right = self._make_node(rkey)
        return {"feature": j, "true": self.extract(left), "false": self.extract(right)}

# ------------------------------------------------------------------ gosdt

class GOSDTClassifier(BaseEstimator, ClassifierMixin):
    """Optimal sparse decision tree minimising ``loss + regularization * leaves``.

    Parameters
    ----------
    regularization : float, default 0.05
        Penalty per leaf.  The reference implementation recommends values
        above ``1 / n_samples``.
    time_limit : float, default 0
        Seconds after which the search stops and the best tree found so far is
        returned (``optimal_`` is then ``False``).  ``0`` means unlimited.
    balance : bool, default False
        Weigh every class equally (balanced accuracy).
    costs : (K, K) array-like, optional
        ``costs[i, j]`` is the cost of predicting class ``i`` when the true class
        is ``j``.  Overrides ``balance``.
    upperbound : float, default 0
        Optional trusted upper bound on the optimal risk used for pruning.
    look_ahead, similar_support, feature_exchange, continuous_feature_exchange : bool
        Enable the corresponding bounds (all exact; they only affect speed).
    greedy_init : bool, default True
        Seed the search with a greedy tree optimising the same objective.
    engine : {"auto", "numba", "python"}, default "auto"
        Bit-counting backend; ``auto`` uses numba when it is installed.  Both
        engines produce identical trees.
    memory_limit : int, default 0
        Resident memory (bytes) above which the search stops like a time limit.
    verbose : bool, default False
    """

    def __init__(self, regularization: float = 0.05, time_limit: float = 0.0,
                 balance: bool = False, costs=None, upperbound: float = 0.0,
                 look_ahead: bool = True, similar_support: bool = True,
                 feature_exchange: bool = True, continuous_feature_exchange: bool = True,
                 greedy_init: bool = True, engine: str = "auto", memory_limit: int = 0,
                 verbose: bool = False):
        self.regularization = regularization
        self.memory_limit = memory_limit
        self.time_limit = time_limit
        self.balance = balance
        self.costs = costs
        self.upperbound = upperbound
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.greedy_init = greedy_init
        self.engine = engine
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, target_name: str = "class"):
        X = _to_dataframe(X)
        if isinstance(y, pd.DataFrame):
            target_name = str(y.columns[0])
            y = y.iloc[:, 0]
        elif isinstance(y, pd.Series) and y.name is not None:
            target_name = str(y.name)
        y = np.asarray(y).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have different numbers of rows")

        t0 = time.perf_counter()
        self.encoder_ = BinaryEncoder().fit(X)
        Xb = self.encoder_.transform(X)
        self.target_encoder_ = TargetEncoder().fit(y)
        y_idx = self.target_encoder_.transform(y)
        self.classes_ = self.target_encoder_.classes_
        data = BitDataset(Xb, y_idx, len(self.classes_), costs=self.costs, balance=self.balance)
        self.encoding_time_ = time.perf_counter() - t0
        self.n_binary_features_ = data.m

        if self.verbose:
            print(f"Dataset Dimensions: {data.n} x {data.m} x {data.K}")

        opt = Optimizer(
            data, self.regularization, groups=self.encoder_.groups,
            time_limit=self.time_limit, look_ahead=self.look_ahead,
            similar_support=self.similar_support, feature_exchange=self.feature_exchange,
            continuous_feature_exchange=self.continuous_feature_exchange,
            greedy_init=self.greedy_init, upperbound=self.upperbound, engine=self.engine,
            memory_limit=self.memory_limit, verbose=self.verbose,
        )
        root = opt.run()
        self.optimal_ = opt.optimal
        self.stop_reason_ = opt.stop_reason
        self.time_ = opt.elapsed
        self.iterations_ = opt.iterations
        self.size_ = len(opt.memo)
        self.lowerbound_ = root.lb
        self.upperbound_ = root.ub
        if not self.optimal_:
            warnings.warn(f"{self.stop_reason_} limit reached before optimality was certified; "
                          "returning the best tree found", RuntimeWarning)

        raw = opt.extract(root)
        self.tree_ = self._decode(raw, data, target_name)
        self.tree = TreeClassifier(self.tree_)
        self.objective_ = self.tree.risk()
        self.n_leaves_ = self.tree.leaves()
        if self.verbose:
            print(f"Training Duration: {self.time_:.4f} seconds")
            print(f"Number of Iterations: {self.iterations_}")
            print(f"Size of Graph: {self.size_}")
            print(f"Objective Boundary: [{root.lb}, {root.ub}]")
            print(f"Loss: {self.tree.loss()}  Complexity: {self.tree.complexity()}")
        return self

    def _decode(self, node: dict, data: BitDataset, target_name: str) -> dict:
        if "prediction" in node:
            key = node["key"]
            _, _, max_loss, _, _, prediction = data.leaf_stats(key)
            return {
                "prediction": self.target_encoder_.inverse(prediction),
                "name": target_name,
                "loss": float(max_loss),
                "complexity": float(self.regularization),
            }
        rule = self.encoder_.rules[node["feature"]]
        return {
            "feature": int(rule["feature"]),
            "name": rule["name"],
            "relation": rule["relation"],
            "reference": rule["reference"],
            "type": rule["type"],
            "true": self._decode(node["true"], data, target_name),
            "false": self._decode(node["false"], data, target_name),
        }

    # -------------------------------------------------------------- predict
    def predict(self, X):
        X = _to_dataframe(X)
        pred = self.tree.predict_fast(X)
        return np.array(pred.tolist())

    def score(self, X, y, sample_weight=None):
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    # -------------------------------------------------------------- reports
    def leaves(self) -> int:
        return self.tree.leaves()

    def nodes(self) -> int:
        return self.tree.nodes()

    def max_depth(self) -> int:
        return self.tree.maximum_depth()

    def json(self) -> str:
        return self.tree.json()

    def __str__(self):
        return str(self.tree)


class GOSDT:
    """Drop-in replacement for the reference Python wrapper.

    >>> model = GOSDT({"regularization": 0.1, "time_limit": 3600})
    >>> model.fit(X, y)
    >>> model.tree, model.time, model.iterations, model.size
    """

    def __init__(self, configuration: dict | None = None):
        self.configuration = dict(configuration or {})
        self.time = 0.0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.model_ = None

    def fit(self, X, y):
        cfg = self.configuration
        if "objective" in cfg and cfg["objective"] not in (None, "acc"):
            raise NotImplementedError("only the accuracy objective is supported")
        kwargs = dict(
            regularization=cfg.get("regularization", 0.05),
            time_limit=cfg.get("time_limit", 0.0),
            balance=cfg.get("balance", False),
            upperbound=cfg.get("upperbound", 0.0),
            look_ahead=cfg.get("look_ahead", True),
            similar_support=cfg.get("similar_support", True),
            feature_exchange=cfg.get("feature_exchange", True),
            continuous_feature_exchange=cfg.get("continuous_feature_exchange", True),
            verbose=cfg.get("verbose", False),
            engine=cfg.get("engine", "auto"),
        )
        if cfg.get("costs"):
            kwargs["costs"] = _read_cost_matrix(cfg["costs"], y)
        self.model_ = GOSDTClassifier(**kwargs).fit(X, y)
        self.tree = self.model_.tree
        self.time = self.model_.time_
        self.iterations = self.model_.iterations_
        self.size = self.model_.size_
        return self

    def status(self):
        return 0 if self.model_.optimal_ else 1

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y, weight=None):
        return self.tree.score(X, y, weight=weight)

    def error(self, X, y, weight=None):
        return self.tree.error(X, y, weight=weight)

    def leaves(self):
        return self.tree.leaves()

    def nodes(self):
        return self.tree.nodes()

    def max_depth(self):
        return self.tree.maximum_depth()

    def json(self):
        return self.tree.json()

    def __len__(self):
        return self.tree.leaves()


def _read_cost_matrix(path: str, y) -> np.ndarray:
    """Parse the reference cost CSV (header of class names, then rows)."""
    table = pd.read_csv(path)
    classes = np.unique(np.asarray(y).ravel())
    names = [str(c) for c in table.columns]
    K = len(classes)
    C = np.zeros((K, K))
    col_of = {name: i for i, name in enumerate(names)}
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                continue
            if table.shape[0] == 1:
                C[i, j] = float(table.iloc[0, col_of[str(cj)]])
            else:
                C[i, j] = float(table.iloc[col_of[str(ci)], col_of[str(cj)]])
    return C


class OptimalTreeClassifier(GOSDTClassifier):
    """The solver under evolution.  ``fit(X, y)`` must set ``tree_`` (reference JSON
    schema: feature/name/relation/reference/type/true/false, leaves with
    prediction/name/loss/complexity), ``optimal_`` and ``time_``."""


def make_model(regularization: float, time_limit: float):
    return OptimalTreeClassifier(regularization=regularization, time_limit=time_limit,
                                 memory_limit=6 * (1 << 30))


# ===========================================================================
# Evaluation loop (do not edit below this line)
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="", help="comma-separated subset of the suite (default: all)")
    parser.add_argument("--lams", default="", help="comma-separated subset of the λ grid (default: all)")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--no-record", action="store_true", help="do not write results/")
    args = parser.parse_args()

    t0 = time.time()
    datasets = [d for d in args.datasets.split(",") if d] or None
    lambdas = [float(v) for v in args.lams.split(",") if v] or None
    summary = evaluate_solver(make_model, MODEL_NAME, datasets=datasets, lambdas=lambdas,
                              time_limit=args.time_limit)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""
    full_suite = datasets is None and lambdas is None and args.time_limit == TIME_LIMIT
    if full_suite and not args.no_record:
        record(MODEL_NAME, DESCRIPTION, summary, git_hash)
    elif not args.no_record:
        print("(partial suite or non-default cap: results not recorded)")
    print_summary(MODEL_NAME, summary)
    print(f"total_seconds: {time.time() - t0:.1f}s")
