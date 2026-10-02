"""Offline: strength of K_r (free intercept per cell of the first s-r columns + r shared shifts) vs LR,
on random good supports (columns from the top 25 by single-column loss)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from suite import load_problem
def lr(U, y, iters=60):
    n, p = U.shape
    th = np.zeros(p)
    for _ in range(iters):
        z = U @ th
        q = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        g = U.T @ (q - y)
        H = (U * (q * (1 - q))[:, None]).T @ U + 1e-9 * np.eye(p)
        d = np.linalg.lstsq(H, g, rcond=None)[0]
        f0 = np.sum(np.logaddexp(0, z) - y * z)
        t = 1.0
        while True:
            z2 = U @ (th - t * d)
            f2 = np.sum(np.logaddexp(0, z2) - y * z2)
            if f2 <= f0 + 1e-12 or t < 1e-10:
                break
            t /= 2
        th = th - t * d
        if abs(f0 - f2) < 1e-10:
            break
    z = U @ th
    return np.sum(np.logaddexp(0, z) - y * z)
ds, k, ubm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X, y, _, _, names = load_problem(ds)
X = X[:, np.ptp(X, 0) > 0]; X = (X == X.max(0)).astype(float)
n, d = X.shape; ub = ubm * n
rng = np.random.default_rng(0)
F1 = np.array([lr(np.c_[X[:, [j]], np.ones(n)], y) for j in range(d)])
top = np.argsort(F1)[:30]
res = []
for trial in range(40):
    S = list(rng.choice(top, size=k, replace=False))
    row = [lr(np.c_[X[:, S], np.ones(n)], y) - ub]
    for r in (2, 3, 4, 5, 6):
        T = S[:k - r]
        key = np.unique(X[:, T], axis=0, return_inverse=True)[1].ravel()
        Ind = np.eye(key.max() + 1)[key]
        row.append(lr(np.c_[Ind, X[:, S[k - r:]]], y) - ub)
    res.append(row)
    print(" ".join(f"{v:7.2f}" for v in row), flush=True)
res = np.array(res)
m = res[:, 0] >= 0
print("LR>=ub:", m.sum(), " K_r>=ub among those, r=2..6:", [(res[m, i] >= 0).sum() for i in range(1, 6)])
