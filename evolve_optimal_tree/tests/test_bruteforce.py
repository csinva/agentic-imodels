"""Exactness tests: compare the bounded search against an unbounded exhaustive DP.

The exhaustive DP evaluates every decision tree over the binary features (each
feature used at most once per path) with plain Python sets, so it shares no code
with the optimizer beyond the objective definition.
"""

import functools
import itertools

import numpy as np
import pandas as pd
import pytest

from pygosdt import BitDataset, GOSDTClassifier, Optimizer


def exhaustive_optimum(Xb, y, K, lam, costs=None):
    n = len(y)
    if costs is None:
        C = np.full((K, K), 1.0 / n)
        np.fill_diagonal(C, 0.0)
    else:
        C = np.asarray(costs, dtype=float)
    m = Xb.shape[1]
    cols = [frozenset(np.flatnonzero(Xb[:, j])) for j in range(m)]
    all_rows = frozenset(range(n))

    def leaf(rows):
        dist = np.zeros(K)
        for i in rows:
            dist[y[i]] += 1
        return float((C @ dist).min()) + lam

    @functools.lru_cache(maxsize=None)
    def solve(rows, feats):
        best = leaf(rows)
        for j in feats:
            left = rows & cols[j]
            if not left or left == rows:
                continue
            right = rows - left
            rest = feats - {j}
            best = min(best, solve(left, rest) + solve(right, rest))
        return best

    return solve(all_rows, frozenset(range(m)))


def random_problem(rng, n, m, K):
    Xb = rng.random((n, m)) < 0.5
    y = rng.integers(0, K, size=n)
    return Xb, y


@pytest.mark.parametrize("seed", range(40))
def test_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 24))
    m = int(rng.integers(2, 6))
    K = int(rng.choice([2, 2, 3]))
    Xb, y = random_problem(rng, n, m, K)
    lam = float(rng.choice([0.2, 0.1, 0.05, 0.02, 0.01, 1.0 / n]))
    expected = exhaustive_optimum(Xb, y, K, lam)

    data = BitDataset(Xb, y, K)
    for flags in ({}, {"engine": "python"},
                  {"similar_support": False, "continuous_feature_exchange": False, "greedy_init": False},
                  {"engine": "python", "similar_support": False, "continuous_feature_exchange": False}):
        opt = Optimizer(data, lam, **flags)
        root = opt.run()
        assert opt.optimal
        assert root.lb == pytest.approx(root.ub)
        assert root.ub == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_matches_exhaustive_search_with_costs(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(6, 20))
    m = int(rng.integers(2, 5))
    K = int(rng.choice([2, 3]))
    Xb, y = random_problem(rng, n, m, K)
    lam = float(rng.choice([0.2, 0.1, 0.05, 0.02]))
    costs = rng.random((K, K)) / n
    np.fill_diagonal(costs, 0.0)
    expected = exhaustive_optimum(Xb, y, K, lam, costs=costs)
    data = BitDataset(Xb, y, K, costs=costs)
    for engine in ("numba", "python"):
        opt = Optimizer(data, lam, engine=engine)
        root = opt.run()
        assert opt.optimal
        assert root.ub == pytest.approx(expected, abs=1e-9)


def test_estimator_objective_is_consistent_with_tree():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "num": np.round(rng.normal(size=n), 0),   # few distinct values keeps the DP tractable
        "int": rng.integers(0, 3, size=n),
        "cat": rng.choice(["a", "b", "c"], size=n),
        "bin": rng.integers(0, 2, size=n),
    })
    y = ((X["num"] > 0) ^ (X["cat"] == "a")).astype(int).to_numpy().copy()
    y[rng.random(n) < 0.05] ^= 1
    lam = 0.02
    model = GOSDTClassifier(regularization=lam).fit(X, y)
    assert model.optimal_
    pred = model.predict(X)
    loss = float(np.mean(pred != y))
    assert loss == pytest.approx(model.tree.loss(), abs=1e-12)
    assert model.objective_ == pytest.approx(loss + lam * model.n_leaves_, abs=1e-12)
    # exhaustive check in the binarized space
    Xb = model.encoder_.transform(X)
    expected = exhaustive_optimum(Xb, model.target_encoder_.transform(y), 2, lam)
    assert model.objective_ == pytest.approx(expected, abs=1e-9)


def test_time_limit_returns_incumbent():
    rng = np.random.default_rng(1)
    Xb = rng.random((400, 40)) < 0.5
    y = rng.integers(0, 2, size=400)
    model = GOSDTClassifier(regularization=0.002, time_limit=0.2)
    with pytest.warns(RuntimeWarning):
        model.fit(Xb.astype(int), y)
    assert not model.optimal_
    assert model.tree.leaves() >= 1
    assert model.time_ < 5.0
