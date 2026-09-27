"""Build the tiny suite: small problems whose true optimum is known by enumeration.

Used by the exact track as its correctness gate (a solver that certifies a loss other than the
true optimum, or claims a lower bound above it, is wrong). 30 problems with 10 features each:

* 12 synthetic: binary features with random frequencies, labels from a sparse logistic model
  with interactions (so the best score is not the planted one), 300 to 2,000 rows;
* 18 real: 10 columns of a visible dataset (the 10 with the highest mutual information with the
  label, among the binary ones), subsampled to at most 2,000 rows.

For k in {2, 3} the optimum is found by enumerating every support of size <= k and every integer
point vector in {-5..5} on it, and calibrating each exactly (a batch of damped Newton steps on the
two numbers a, b, run to a gradient below 1e-11). Writes data/tiny/<name>.npz, data/tiny/manifest.csv
and src/tiny_optima.csv.

    uv run data/build_tiny.py
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D, BOUND, KS = 10, 5, (2, 3)


def batch_calibrate(S, pos, neg):
    """Min over (a, b) of the mean log loss for each row of the cell-score matrix S (m x c), with
    pos/neg the label counts per cell. Returns the losses (m,)."""
    n = pos.sum() + neg.sum()
    m = S.shape[0]
    p0 = pos.sum() / n
    a = np.zeros(m)
    b = np.full(m, np.log(p0 / (1 - p0)))

    def f(a_, b_):
        z = a_[:, None] * S + b_[:, None]
        return (pos * np.logaddexp(0, -z) + neg * np.logaddexp(0, z)).sum(1) / n

    cur = f(a, b)
    for _ in range(100):
        z = a[:, None] * S + b[:, None]
        q = 1 / (1 + np.exp(-z))
        r = (pos + neg) * q - pos
        w = (pos + neg) * q * (1 - q)
        ga, gb = (r * S).sum(1) / n, r.sum(1) / n
        haa, hab, hbb = (w * S * S).sum(1) / n + 1e-12, (w * S).sum(1) / n, w.sum(1) / n + 1e-12
        det = haa * hbb - hab * hab
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        t = np.ones(m)
        new = f(a - da, b - db)
        for _ in range(40):
            bad = new > cur - 1e-4 * t * (ga * da + gb * db)
            if not bad.any():
                break
            t = np.where(bad, t * 0.5, t)
            new = np.where(bad, f(a - t * da, b - t * db), new)
        a, b = a - t * da, b - t * db
        done = np.abs(cur - new) < 1e-15
        cur = np.minimum(cur, new)
        if np.all(np.sqrt(ga ** 2 + gb ** 2) < 1e-11) or done.all():
            break
    return cur


def optimum(X, y, k):
    """Exact minimum of the calibrated loss over integer points (<= k nonzero, each in [-5, 5])."""
    best = np.inf
    vals = [v for v in range(-BOUND, BOUND + 1) if v]
    for s in range(1, k + 1):
        W = np.array(list(itertools.product(vals, repeat=s)), float)
        for S in itertools.combinations(range(X.shape[1]), s):
            cells, inv = np.unique(X[:, S], axis=0, return_inverse=True)
            inv = inv.ravel()
            pos = np.bincount(inv, weights=y, minlength=len(cells))
            neg = np.bincount(inv, minlength=len(cells)) - pos
            best = min(best, float(batch_calibrate(W @ cells.T, pos, neg).min()))
    return best


def synthetic(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([300, 800, 2000]))
    X = (rng.random((n, D)) < rng.uniform(0.15, 0.6, D)).astype(float)
    beta = rng.normal(0, 1.2, D) * (rng.random(D) < 0.5)
    logit = X @ beta + 1.5 * X[:, 0] * X[:, 1] - 1.5 * X[:, 2] * (1 - X[:, 3]) - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return X, y


def real(name, seed):
    z = np.load(os.path.join(ROOT, "data", "visible", f"{name}.npz"))
    X, y = z["Xtr"].astype(float), z["ytr"].astype(np.int64)
    binary = [j for j in range(X.shape[1]) if set(np.unique(X[:, j])) <= {0.0, 1.0}]
    Xb = X[:, binary]
    mi = mutual_info_classif(Xb, y, discrete_features=True, random_state=0)
    cols = np.argsort(-mi, kind="stable")[:D]
    X = Xb[:, cols]
    if len(y) > 2000:
        idx, _ = train_test_split(np.arange(len(y)), train_size=2000, stratify=y, random_state=seed)
        X, y = X[idx], y[idx]
    return X, y


if __name__ == "__main__":
    out = os.path.join(HERE, "tiny")
    os.makedirs(out, exist_ok=True)
    problems = [(f"syn{i}", *synthetic(i)) for i in range(12)]
    reals = ["adult", "bank", "mammo", "mushroom", "fico", "compas", "australian", "heart", "ilpd"]
    problems += [(f"{r}{j}", *real(r, j)) for r in reals for j in range(2)]
    man, opt = [], []
    for name, X, y in problems:
        if len(np.unique(y)) < 2 or X.shape[1] < D:
            continue
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)
        np.savez_compressed(os.path.join(out, f"{name}.npz"), Xtr=X[itr], ytr=y[itr], Xte=X[ite], yte=y[ite],
                            feature_names=np.array([f"x{j}" for j in range(D)]))
        man.append({"name": name, "n_train": len(itr), "features": D, "positive_rate": round(float(y.mean()), 3)})
        for k in KS:
            opt.append({"suite": "tiny", "dataset": name, "k": k, "loss": f"{optimum(X[itr], y[itr], k):.12f}"})
            print(opt[-1], flush=True)
    pd.DataFrame(man).to_csv(os.path.join(out, "manifest.csv"), index=False)
    pd.DataFrame(opt).to_csv(os.path.join(ROOT, "src", "tiny_optima.csv"), index=False)
    print(f"{len(man)} problems, {len(opt)} optima", file=sys.stderr)
