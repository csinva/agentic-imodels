"""Probe: at a solver's final point, the best exact pair value change (two support columns change value at once,
all value combinations) and the best rescale (round(w * f), f in a grid). usage: SOLVER=f.py pairprobe.py ds"""
import os, sys, time
import numpy as np, numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbhd
M = nbhd.M
from suite import load_problem, K_VALUES

@nb.njit(cache=False)
def best_pair(s, w, S, y, c, a, b, QT, var_of, tptr, tabv):
    n = s.shape[0]; m = S.shape[0]
    x1 = np.empty(n); x2 = np.empty(n); s2 = np.empty(n)
    bL = 1e300; bi = -1; bj = -1; bv1 = 0.0; bv2 = 0.0
    for p in range(m):
        for i in range(n):
            x1[i] = M.xval(QT, var_of, tptr, tabv, i, S[p])
        for q in range(p + 1, m):
            for i in range(n):
                x2[i] = M.xval(QT, var_of, tptr, tabv, i, S[q])
            for v1 in range(-5, 6):
                for v2 in range(-5, 6):
                    d1 = v1 - w[S[p]]; d2 = v2 - w[S[q]]
                    if d1 == 0 or d2 == 0:
                        continue
                    for i in range(n):
                        s2[i] = s[i] + d1 * x1[i] + d2 * x2[i]
                    L = nbhd.lossof(s2, y, c, a, b)
                    if L < bL:
                        bL = L; bi = S[p]; bj = S[q]; bv1 = v1; bv2 = v2
    return bL, bi, bj, bv1, bv2

name = sys.argv[1]
X, y, *_ = load_problem(name, "visible_fine")
for k in K_VALUES:
    m = M.make_model(k, 60.0).fit(X, y)
    D = M.Data(X, y); w = m.coef_.astype(np.float64)
    st = M.start_state(w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N); L0 = st[1] / D.N
    S = np.flatnonzero(w).astype(np.int64)
    t0 = time.time()
    bL, bi, bj, v1, v2 = best_pair(st[0], w, S, D.y, D.c, st[2], st[3], D.QT, D.var_of, D.tptr, D.tabv)
    resc = []
    for f in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.1, 1.2, 1.25, 1.33, 1.5, 2.0):
        w2 = np.clip(np.round(w * f), -5, 5)
        if np.count_nonzero(w2) == 0: continue
        resc.append(M.start_state(w2, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)[1] / D.N - L0)
    print(f"{name} k{k} pair {bL / D.N - L0:+.6f} rescale {min(resc):+.6f} {time.time()-t0:.1f}s", flush=True)
