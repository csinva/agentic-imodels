"""Offline test of the scale-split family bound.
E_min: LP lower bound on the perceptron loss sum_i n_i max(0, -yt_i (x_i.w - tau)) over real w with
||w||_inf <= 5, ||w||_1 <= 5k, and some |w_j| >= 1 (2d LPs). a* = ub / E_min.
R(a*) = min F(theta, b) s.t. |theta_j| <= 5 a*, sum_{j in L} |theta_j| <= 5 a* m (T coords only boxed)."""
import sys, os
import numpy as np
from scipy.optimize import linprog, minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from suite import load_problem

ds, k, ubm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X, y, _, _, names = load_problem(ds)
X = X[:, np.ptp(X, 0) > 0]
n, d = X.shape
ub = ubm * n
yt = 2.0 * y - 1
# LP variables: w+ (d), w- (d), tau+ , tau-, e (n)
best = np.inf
cE = np.r_[np.zeros(2 * d + 2), np.ones(n)]
# e_i >= -yt_i (x_i.(w+ - w-) - tau)  ->  -yt x w+ + yt x w- + yt tau+ - yt tau- - e <= 0
A1 = np.c_[-yt[:, None] * X, yt[:, None] * X, yt, -yt, -np.eye(n)]
b1 = np.zeros(n)
A2 = np.r_[np.ones(2 * d), 0, 0, np.zeros(n)][None, :]      # ||w||_1 <= 5k
b2 = [5.0 * k]
bounds = [(0, 5)] * (2 * d) + [(0, None), (0, None)] + [(0, None)] * n
res_all = []
for j in range(d):
    for sg in (0, 1):
        A3 = np.zeros((1, 2 * d + 2 + n))
        A3[0, j + sg * d] = -1.0                               # w_j^{+/-} >= 1
        r = linprog(cE, A_ub=np.r_[A1, A2, A3], b_ub=np.r_[b1, b2, [-1.0]], bounds=bounds, method="highs")
        res_all.append(r.fun)
        best = min(best, r.fun)
res_all = np.array(res_all)
print(f"E_min={best:.4f}  ub={ub:.3f}  a*={ub / best:.4f}  median E_j={np.median(res_all):.3f}")
astar = ub / best

def lr_box(cols_T, cols_L, box, l1):
    """min F over theta_T in [-box, box], theta_L in box with sum |theta_L| <= l1, b free (SLSQP on split vars)."""
    T, L = list(cols_T), list(cols_L)
    U = np.c_[X[:, T], X[:, L], np.ones(n)]
    t, l = len(T), len(L)
    def unpack(v):
        return np.r_[v[:t], v[t:t + l] - v[t + l:t + 2 * l], v[-1]]
    def f(v):
        th = unpack(v)
        z = U @ th
        F = np.sum(np.logaddexp(0, z) - y * z)
        q = 1 / (1 + np.exp(-z))
        g = U.T @ (q - y)
        gv = np.r_[g[:t], g[t:t + l], -g[t:t + l], g[-1]]
        return F, gv
    v0 = np.zeros(t + 2 * l + 1)
    bnds = [(-box, box)] * t + [(0, box)] * (2 * l) + [(None, None)]
    cons = [{"type": "ineq", "fun": lambda v: l1 - np.sum(v[t:t + 2 * l]),
             "jac": lambda v: np.r_[np.zeros(t), -np.ones(2 * l), 0]}]
    r = minimize(f, v0, jac=True, bounds=bnds, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return r.fun

def lr_free(cols):
    return lr_box(cols, [], 1e6, 1e9)

rng = np.random.default_rng(0)
F1 = [lr_free([j]) for j in range(d)]
top = np.argsort(F1)[:25]
for lev in (1, 2, 3, 4):
    ok = 0
    vals = []
    for trial in range(15):
        T = sorted(rng.choice(top, size=k - lev, replace=False))
        L = [j for j in range(d) if j not in T]
        R = lr_box(T, L, 5 * astar, 5 * astar * lev)
        Rfree = lr_box(T, L, 1e6, 5 * astar * lev)
        vals.append((R - ub, Rfree - ub, lr_free(T) - ub))
        ok += R >= ub
    print(f"m={lev}: R(a*)>=ub on {ok}/15;  (R-ub, R_noTbox-ub, LR(T)-ub) samples:",
          " ".join(f"({a:.1f},{b:.1f},{c:.1f})" for a, b, c in vals[:8]))
