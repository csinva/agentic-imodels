"""Offline test of the pivot-cone family bound (P26 candidate).
R_p = min F s.t. |theta_j| <= 5 theta_p (j in T u L'), sum_{L'} |theta_j| <= 5 m theta_p.
Compared with ub, LR(T u L') and (for small m) the min over leaves."""
import sys, os, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from suite import load_problem

ds, k, ubm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
levs = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2, 3, 4]
X, y, _, _, names = load_problem(ds)
X = X[:, np.ptp(X, 0) > 0]
X = (X == X.max(0)).astype(float)
n, d = X.shape
ub = ubm * n

def Ff(U, th):
    z = U @ th
    F = np.sum(np.logaddexp(0, z) - y * z)
    q = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    return F, U.T @ (q - y)

def cone(T, L, p, m, box=5.0, sg=1.0):
    """variables: theta_T (t), L+ (l), L- (l), b."""
    t, l = len(T), len(L)
    U = np.c_[X[:, T], X[:, L], np.ones(n)]
    ip = T.index(p)
    def unpack(v):
        return np.r_[v[:t], v[t:t + l] - v[t + l:t + 2 * l], v[-1]]
    def f(v):
        F, g = Ff(U, unpack(v))
        return F, np.r_[g[:t], g[t:t + l], -g[t:t + l], g[-1]]
    cons = []
    A = []
    for q in range(t):
        if q == ip:
            continue
        r = np.zeros(t + 2 * l + 1); r[ip] = sg * box; r[q] = -1; A.append(r)
        r = np.zeros(t + 2 * l + 1); r[ip] = sg * box; r[q] = 1; A.append(r)
    for q in range(l):
        r = np.zeros(t + 2 * l + 1); r[ip] = sg * box; r[t + q] = -1; r[t + l + q] = -1; A.append(r)
    r = np.zeros(t + 2 * l + 1); r[ip] = sg * box * m; r[t:t + 2 * l] = -1; A.append(r)
    A = np.array(A)
    cons = [{"type": "ineq", "fun": lambda v: A @ v, "jac": lambda v: A}]
    bnds = [(None, None)] * t + [(0, None)] * (2 * l) + [(None, None)]
    v0 = np.zeros(t + 2 * l + 1); v0[ip] = 0.1 * sg
    r = minimize(f, v0, jac=True, bounds=bnds, constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
    return r.fun

def lrfree(cols):
    U = np.c_[X[:, cols], np.ones(n)]
    r = minimize(lambda v: Ff(U, v), np.zeros(len(cols) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return r.fun

rng = np.random.default_rng(1)
F1 = np.array([lrfree([j]) for j in range(d)])
top = list(np.argsort(F1)[:25])
for lev in levs:
    cnt = dict(cone=0, perfect=0, fam=0, tot=0)
    for trial in range(12):
        T = [int(v) for v in sorted(rng.choice(top, size=k - lev, replace=False))]
        L = [j for j in range(d) if j not in T]
        Rs = [min(cone(T, L, p, lev), cone(T, L, p, lev, sg=-1.0)) for p in T]
        R = max(Rs)
        fam = lrfree(T + L)
        best = min(lrfree(T + list(A)) for A in itertools.combinations(L, lev)) if lev <= 2 else np.nan
        cnt["tot"] += 1; cnt["cone"] += R >= ub; cnt["fam"] += fam >= ub; cnt["perfect"] += best >= ub
        print(f"m={lev} LR(T)-ub={lrfree(T)-ub:7.2f} cone_max-ub={R-ub:7.2f} cone_min-ub={min(Rs)-ub:7.2f} fam-ub={fam-ub:7.2f} minleaf-ub={best-ub:7.2f}", flush=True)
    print(lev, cnt, flush=True)
