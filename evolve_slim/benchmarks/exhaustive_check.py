"""How often does a solver reach the true optimum? Small problems where every integer point
vector can be enumerated: 50 random problems, 300 rows, 7 binary features, k = 3, points in
[-3, 3] (each has 7*6 + 21*36 + 35*216 = 8,358 candidate point vectors).

    uv run benchmarks/exhaustive_check.py                 # the shipped v35_scratch
    uv run benchmarks/exhaustive_check.py fasterrisk      # FasterRisk 0.1.10
    uv run benchmarks/exhaustive_check.py path/to/slim.py

Prints how many of the 50 the solver solves exactly and its largest gap to the optimum.
"""

import importlib.util
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
from evaluate import calibrate  # noqa: E402

BOUND, K = 3, 3
arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "sep26-run2", "slim_lib", "v35_scratch.py")
if arg == "fasterrisk":
    from fasterrisk.fasterrisk import RiskScoreOptimizer

    def fit(X, y):
        opt = RiskScoreOptimizer(X=X, y=np.where(y > 0, 1.0, -1.0), k=K, lb=-BOUND, ub=BOUND)
        opt.optimize()
        return np.round(opt.get_models(0)[2])
else:
    spec = importlib.util.spec_from_file_location("solver_under_test", arg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.COEF_BOUND = BOUND  # the snapshots read the bound from this module constant at call time

    def fit(X, y):
        return mod.make_model(K, 30.0).fit(X, y).coef_


def brute_force(X, y):
    best, d = np.inf, X.shape[1]
    for size in range(1, K + 1):
        for S in itertools.combinations(range(d), size):
            for vals in itertools.product([v for v in range(-BOUND, BOUND + 1) if v], repeat=size):
                w = np.zeros(d)
                w[list(S)] = vals
                best = min(best, calibrate(X @ w, y)[2])
    return best


gaps = []
for seed in range(50):
    rng = np.random.default_rng(1000 + seed)
    n, d = 300, 7
    X = (rng.random((n, d)) < rng.uniform(0.2, 0.7, d)).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ rng.normal(0, 1.5, d) - 1)))).astype(int)
    w = fit(X, y)
    assert np.count_nonzero(w) <= K and np.abs(w).max() <= BOUND
    gaps.append(calibrate(X @ w, y)[2] - brute_force(X, y))
g = np.array(gaps)
print(f"{os.path.basename(arg)}: exact on {np.sum(g < 1e-9)}/50; max gap {g.max():.2e}; mean gap {g.mean():.2e}")
