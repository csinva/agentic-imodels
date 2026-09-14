"""Branch-and-bound search for the optimal sparse decision tree.

The objective minimised is

    risk(tree) = loss(tree) + regularization * (#leaves)

where ``loss`` is the (cost-weighted) misclassification rate.  The search is a
depth-first dynamic program over subproblems identified by their *capture set*
(the bitset of rows that reach a node), memoised in ``self.memo`` exactly like
the dependency graph of the reference implementation.  Every subproblem stores
a certified interval ``[lb, ub]`` on its optimal risk together with the split
that achieves ``ub``.

Bounds implemented (all are exact, so the returned tree is provably optimal):

* leaf risk upper bound and the *equivalent points* lower bound
  ``min_loss + 2 * regularization`` for any split;
* leaf-support and incremental-accuracy conditions that prove a subproblem is
  best left as a leaf (``Task::Task`` in the reference);
* one-step look-ahead: children only get the budget left after subtracting
  the sibling's lower bound (``send_explorers`` scopes in the reference);
* similar-support bound between neighbouring binary features
  (``Dataset::distance`` in the reference);
* feature exchange / continuous feature exchange between nested features,
  applied per subproblem (``Task::feature_exchange`` in the reference);
* incumbent from a greedy dive with the same objective.
"""

from __future__ import annotations

import time

import numpy as np

from .dataset import BitDataset
from .fastbits import HAVE_NUMBA, child_counts_subset, int_to_words, warm_up

EPS = 1e-10


class TimeLimitReached(Exception):
    pass


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
                 verbose: bool = False):
        self.data = data
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
        self.next_in_group = np.full(data.m, -1, dtype=np.int64)
        self.prev_in_group = np.full(data.m, -1, dtype=np.int64)
        for g in (groups or []):
            for a, b in zip(g[:-1], g[1:]):
                self.next_in_group[a] = b
                self.prev_in_group[b] = a

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
        except TimeLimitReached:
            self.optimal = False
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

    def _child_node(self, key: int, count: int, leaf: float, lb: float, solved: bool,
                    l_or_r_pred_key: int | None = None) -> Node:
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
        if self.time_limit > 0.0 and (self.iterations & 63) == 0:
            if time.perf_counter() - self.start_time > self.time_limit:
                raise TimeLimitReached()

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

        # incorporate memoised bounds of children that already exist, only for
        # candidates that survive the cheap filter (memo bounds can only tighten)
        lkeys = {}
        for i in cand.tolist():
            lkey = key & F[feats[i]]
            lkeys[i] = lkey
            ln = memo.get(lkey)
            if ln is not None:
                l_lb[i] = ln.lb
                l_leaf[i] = ln.ub  # ub is achievable, not necessarily a leaf
                l_solved[i] = ln.solved
            rn = memo.get(key ^ lkey)
            if rn is not None:
                r_lb[i] = rn.lb
                r_leaf[i] = rn.ub
                r_solved[i] = rn.solved
            if ln is not None or rn is not None:
                split_lb[i] = l_lb[i] + r_lb[i]
                split_ub[i] = l_leaf[i] + r_leaf[i]
                if split_ub[i] < best - EPS:
                    best = float(split_ub[i])
                    best_split = int(feats[i])
                    bound = min(budget, best)

        order = cand[np.lexsort((split_ub[cand], split_lb[cand]))]
        sim = self.similar_support
        known_lb = {}  # feature -> proven lower bound of its split at this node

        for oi in order.tolist():
            i = oi
            j = int(feats[i])
            slb = float(split_lb[i])
            if sim:
                # similar support bound from neighbouring features already resolved
                # here; the distance is only computed when it could prune
                for nb in (j - 1, j + 1):
                    nlb = known_lb.get(nb)
                    if nlb is not None and nlb > bound + EPS and nlb > slb:
                        d = data.distance(key, j, nb)
                        if nlb - d > slb:
                            slb = nlb - d
            if slb > bound + EPS:
                if slb < min_pruned:
                    min_pruned = slb
                known_lb[j] = slb
                continue
            lkey = lkeys[i]
            rkey = key ^ lkey
            left = self._child_node(lkey, 0, float(l_leaf[i]), float(l_lb[i]), bool(l_solved[i]))
            right = self._child_node(rkey, 0, float(r_leaf[i]), float(r_lb[i]), bool(r_solved[i]))
            if left.count == 0:
                left.count = lkey.bit_count()
            if right.count == 0:
                right.count = rkey.bit_count()

            # solve the child with the larger lower bound first (more likely to prune)
            first, second = (left, right) if left.lb >= right.lb else (right, left)
            self._solve(first, bound - second.lb, feats)
            if first.lb > bound - second.lb + EPS:
                slb = first.lb + second.lb
                if slb < min_pruned:
                    min_pruned = slb
                known_lb[j] = slb
                continue
            self._solve(second, bound - first.ub, feats)
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
