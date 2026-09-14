"""Bitset representation of a binarized dataset.

Rows are indexed by bit position inside Python integers, so a "capture set"
(the rows reaching a subproblem) is a single ``int`` and splitting is a bitwise
``&``.  This mirrors the ``Bitmask``/``Dataset`` pair of the reference C++
implementation, including the cost matrix aggregation used by the bounds and
the *equivalent points* (``majority``) mask.
"""

from __future__ import annotations

import numpy as np

from .fastbits import HAVE_NUMBA, int_to_words, pack_columns


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

    def distance(self, capture: int, i: int, j: int) -> float:
        """Similar-support distance between features ``i`` and ``j`` on ``capture``."""
        fi, fj = self.features[i], self.features[j]
        differ = capture & (fi ^ fj)
        agree = capture & ~differ
        pos = 0.0
        neg = 0.0
        for k in range(self.K):
            d = float(self.diff_costs[k])
            if d == 0.0:
                continue
            t = self.targets[k]
            pos += d * (differ & t).bit_count()
            neg += d * (agree & t).bit_count()
        return min(pos, neg)
