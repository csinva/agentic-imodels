"""Offline: how much room is there for a cardinality-aware family bound at depth s-1 / s-2?
For random nodes T near good supports: LR(T), min over leaves LR(T+A), chi2 bound LR(T) - r^T S^-1 r."""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from suite import load_problem

def lr(X, y, iters=50):
    n, p = X.shape
    U = np.c_[X, np.ones(n)]
    th = np.zeros(p + 1)
    for _ in range(iters):
        z = U @ th
        q = 1 / (1 + np.exp(-z))
        g = U.T @ (q - y)
        H = (U * (q * (1 - q))[:, None]).T @ U + 1e-10 * np.eye(p + 1)
        d = np.linalg.solve(H, g)
        # backtracking
        f0 = np.sum(np.logaddexp(0, z) - y * z)
        t = 1.0
        while True:
            z2 = U @ (th - t * d)
            f2 = np.sum(np.logaddexp(0, z2) - y * z2)
            if f2 <= f0 + 1e-12 or t < 1e-8:
                break
            t /= 2
        th = th - t * d
        if abs(f0 - f2) < 1e-11:
            break
    z = U @ th
    return np.sum(np.logaddexp(0, z) - y * z), th

ds, k, ubm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X, y, _, _, names = load_problem(ds)
X = X[:, np.ptp(X, 0) > 0]
X = (X == X.max(0)).astype(float)
n, d = X.shape
ub = ubm * n
rng = np.random.default_rng(0)
# ranking of columns by single LR gain
base = lr(np.zeros((n, 0)), y)[0]
g1 = np.array([base - lr(X[:, [j]], y)[0] for j in range(d)])
top = np.argsort(-g1)[:25]
for lev in (1, 2):
    print(f"--- depth s-{lev} (m={lev})")
    cnt = dict(perfect=0, chi2=0, chi2box=0, fam=0, total=0)
    for trial in range(40):
        T = sorted(rng.choice(top, size=k - lev, replace=False))
        L = [j for j in range(d) if j not in T]
        FT, thT = lr(X[:, T], y)
        U = np.c_[X[:, T], np.ones(n)]
        q = 1 / (1 + np.exp(-(U @ thT)))
        w = q * (1 - q)
        r = X.T @ (y - q)                        # residuals of all columns
        HT = (U * w[:, None]).T @ U
        B = np.linalg.solve(HT, (U * w[:, None]).T @ X)   # weighted LS of each column on T
        Xt = X - U @ B                           # residualized columns
        S = (Xt * w[:, None]).T @ Xt             # Schur complement over all columns
        best = np.inf
        bestchi = 0.0
        boxok = True
        for A in itertools.combinations(L, lev):
            A = list(A)
            FA = lr(X[:, T + A], y)[0]
            best = min(best, FA)
            g = np.linalg.solve(S[np.ix_(A, A)] + 1e-12 * np.eye(lev), r[A])
            chi = r[A] @ g
            u = Xt[:, A] @ g
            if np.any(u > 1 / q + 1e-12) or np.any(u < -1 / (1 - q) - 1e-12):
                boxok = False
            bestchi = max(bestchi, chi)
        Ffam = lr(X[:, T + L], y)[0]
        cnt["total"] += 1
        cnt["perfect"] += best >= ub
        cnt["chi2"] += FT - bestchi >= ub
        cnt["chi2box"] += (FT - bestchi >= ub) and boxok
        cnt["fam"] += Ffam >= ub
        print(f"T={T} LR(T)-ub={FT-ub:7.3f} minleaf-ub={best-ub:7.3f} chi2max={bestchi:7.3f} fam-ub={Ffam-ub:8.3f} box={boxok}")
    print(cnt)
