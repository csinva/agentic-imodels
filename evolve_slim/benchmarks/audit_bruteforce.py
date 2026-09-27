"""Audit certifying solvers against brute force on problems harder than the tiny suite.

Generates enumerable problems (k = 4 with 10 to 12 binary features, k = 5 with 8 features;
synthetic ones with interactions and real ones: the top-MI binary columns of visible datasets, at
most 1,500 rows), finds each true optimum by enumerating every support and integer point vector
(cached in results/audit_optima.csv), then runs each solver file and checks:

* a certified answer (lower_bound_ >= loss - 1e-7) must reach the optimum (within 1e-7);
* lower_bound_ may never exceed the optimum (+1e-7).

    uv run benchmarks/audit_bruteforce.py runs/sep27-exact2/slim_lib/x4_family.py [more.py ...]
"""

import importlib.util
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "data"))
from build_tiny import batch_calibrate  # noqa: E402
from evaluate import calibrate  # noqa: E402

TOL = 1e-7
CACHE = os.path.join(ROOT, "results", "audit_optima.csv")
REALS = ["adult", "bank", "mammo", "mushroom", "fico", "compas", "australian", "heart", "ilpd", "magic"]


def problems():
    out = []
    for i in range(10):
        rng = np.random.default_rng(100 + i)
        d, k = (12, 4) if i < 5 else (8, 5)
        n = int(rng.choice([400, 1000, 1500]))
        X = (rng.random((n, d)) < rng.uniform(0.15, 0.6, d)).astype(float)
        beta = rng.normal(0, 1.3, d) * (rng.random(d) < 0.6)
        logit = X @ beta + 1.5 * X[:, 0] * X[:, 1] - 1.2 * X[:, 2] * (1 - X[:, 3]) + 0.8 * X[:, 4] * X[:, 5] - 0.4
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
        out.append((f"syn{i}_d{d}k{k}", X, y, k))
    from sklearn.feature_selection import mutual_info_classif
    for j, name in enumerate(REALS):
        z = np.load(os.path.join(ROOT, "data", "visible", f"{name}.npz"))
        X, y = z["Xtr"].astype(float), z["ytr"].astype(np.int64)
        binary = [c for c in range(X.shape[1]) if set(np.unique(X[:, c])) <= {0.0, 1.0}]
        X = X[:, binary]
        rng = np.random.default_rng(200 + j)
        if len(y) > 1500:
            idx = rng.choice(len(y), 1500, replace=False)
            X, y = X[idx], y[idx]
        mi = mutual_info_classif(X, y, discrete_features=True, random_state=0)
        d, k = (11, 4) if j % 2 == 0 else (8, 5)
        cols = np.argsort(-mi, kind="stable")[:d]
        if len(cols) == d:
            out.append((f"{name}_d{d}k{k}", X[:, cols], y, k))
    return out


def optimum(args):
    name, X, y, k = args
    vals = [v for v in range(-5, 6) if v]
    best = np.inf
    for s in range(1, k + 1):
        W = np.array(list(itertools.product(vals, repeat=s)), float)
        for S in itertools.combinations(range(X.shape[1]), s):
            cells, inv = np.unique(X[:, S], axis=0, return_inverse=True)
            inv = inv.ravel()
            pos = np.bincount(inv, weights=y, minlength=len(cells))
            neg = np.bincount(inv, minlength=len(cells)) - pos
            for lo in range(0, len(W), 20000):
                best = min(best, float(batch_calibrate(W[lo:lo + 20000] @ cells.T, pos, neg).min()))
    return name, best


def load_solver(path):
    spec = importlib.util.spec_from_file_location("audited", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_model


if __name__ == "__main__":
    probs = problems()
    known = pd.read_csv(CACHE).set_index("name")["loss"].to_dict() if os.path.exists(CACHE) else {}
    todo = [p for p in probs if p[0] not in known]
    if todo:
        with mp.get_context("spawn").Pool(min(len(todo), 20)) as pool:
            for name, loss in pool.imap_unordered(optimum, todo):
                known[name] = loss
                print(f"optimum {name}: {loss:.10f}", flush=True)
        pd.DataFrame({"name": list(known), "loss": [f"{v:.12f}" for v in known.values()]}).to_csv(CACHE, index=False)
    for path in sys.argv[1:]:
        make_model = load_solver(path)
        n_cert = n_bad = 0
        for name, X, y, k in probs:
            m = make_model(k, 60.0).fit(X, y)
            loss = calibrate(X @ np.asarray(m.coef_, float), y)[2]
            lb = float(getattr(m, "lower_bound_", np.nan))
            opt = known[name]
            cert = not np.isnan(lb) and lb >= loss - TOL
            bad = (cert and loss > opt + TOL) or (not np.isnan(lb) and lb > opt + TOL)
            n_cert += cert
            n_bad += bad
            print(f"  {os.path.basename(path)} {name:18s} k={k} loss={loss:.9f} opt={opt:.9f} lb={lb:.9f} "
                  f"{'CERT' if cert else '    '} {'WRONG' if bad else ''}", flush=True)
        print(f"{os.path.basename(path)}: certified {n_cert}/{len(probs)}, WRONG {n_bad}")
