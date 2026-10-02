"""Offline: cone with all T signs fixed (natural signs of LR(T)) as an indication for sign branching.
constraints (a = scale var): sg_q theta_q >= a (q in T), |theta_j| <= 5a (all), sum_L |theta_j| <= 5 m a."""
import sys, os, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from suite import load_problem
ds, k, ubm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
levs = [int(v) for v in sys.argv[4].split(",")]
X, y, _, _, names = load_problem(ds)
X = X[:, np.ptp(X, 0) > 0]; X = (X == X.max(0)).astype(float)
n, d = X.shape; ub = ubm * n
def Ff(U, th):
    z = U @ th; F = np.sum(np.logaddexp(0, z) - y * z)
    q = 1 / (1 + np.exp(-np.clip(z, -700, 700))); return F, U.T @ (q - y)
def lrfree(cols):
    U = np.c_[X[:, cols], np.ones(n)]
    r = minimize(lambda v: Ff(U, v), np.zeros(len(cols) + 1), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return r.fun, r.x
def cone_signs(T, L, sg, m, fixed):
    """vars: theta_T (t), L+ (l), L- (l), b, a. sign constraints only for q in fixed."""
    t, l = len(T), len(L); nv = t + 2 * l + 2
    U = np.c_[X[:, T], X[:, L], np.ones(n)]
    def unpack(v): return np.r_[v[:t], v[t:t + l] - v[t + l:t + 2 * l], v[t + 2 * l]]
    def f(v):
        F, g = Ff(U, unpack(v)); return F, np.r_[g[:t], g[t:t + l], -g[t:t + l], g[-1], 0.0]
    A = []
    ia = nv - 1
    for q in range(t):
        r = np.zeros(nv); r[ia] = 5; r[q] = -1; A.append(r)
        r = np.zeros(nv); r[ia] = 5; r[q] = 1; A.append(r)
        if q in fixed:
            r = np.zeros(nv); r[q] = sg[q]; r[ia] = -1; A.append(r)
    for q in range(l):
        r = np.zeros(nv); r[ia] = 5; r[t + q] = -1; r[t + l + q] = -1; A.append(r)
    r = np.zeros(nv); r[ia] = 5 * m; r[t:t + 2 * l] = -1; A.append(r)
    A = np.array(A)
    bnds = [(None, None)] * t + [(0, None)] * (2 * l) + [(None, None), (0, None)]
    v0 = np.zeros(nv); v0[ia] = 0.01
    for q in range(t): v0[q] = 0.01 * sg[q]
    r = minimize(f, v0, jac=True, bounds=bnds, constraints=[{"type": "ineq", "fun": lambda v: A @ v, "jac": lambda v: A}],
                 method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
    return r.fun
rng = np.random.default_rng(1)
F1 = np.array([lrfree([j])[0] for j in range(d)]); top = list(np.argsort(F1)[:25])
for lev in levs:
    c1 = c2 = 0
    for trial in range(12):
        T = [int(v) for v in sorted(rng.choice(top, size=k - lev, replace=False))]
        L = [j for j in range(d) if j not in T]
        FT, th = lrfree(T); sg = np.sign(th[:len(T)]); sg[sg == 0] = 1
        Rall = cone_signs(T, L, sg, lev, set(range(len(T))))
        # flip the sign of the weakest coefficient: the second-best pattern indication
        q0 = int(np.argmin(np.abs(th[:len(T)]))); sg2 = sg.copy(); sg2[q0] *= -1
        Rflip = cone_signs(T, L, sg2, lev, set(range(len(T))))
        c1 += Rall >= ub; c2 += min(Rall, Rflip) >= ub
        print(f"m={lev} LR(T)-ub={FT-ub:6.2f} natural-signs-ub={Rall-ub:7.2f} weakest-flipped-ub={Rflip-ub:7.2f}", flush=True)
    print(lev, c1, c2, flush=True)
