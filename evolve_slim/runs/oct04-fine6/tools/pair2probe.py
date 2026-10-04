"""Probe: at a solver's final point, all single moves (value changes, additions, swaps: every valid column and value)
by exact calibrated loss; then every compatible pair of the M best single moves (including worsening ones) by exact
loss. Prints per problem the best single and best pair gain, and the mean pair gain.
usage: SOLVER=f.py uv run tools/pair2probe.py [datasets] [ks] [M]"""
import os, sys, time
import numpy as np, numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbhd
M = nbhd.M
from suite import load_problem, suite_datasets, K_VALUES

@nb.njit(cache=False)
def all_singles(s, w, S, valid, y, c, a, b, QT, var_of, tptr, tabv, k):
    d = w.shape[0]; n = s.shape[0]; nS = S.shape[0]
    xr = np.empty(n); xa = np.empty(n); s2 = np.empty(n)
    out_r = []; out_a = []; out_v = []; out_L = []
    for ri in range(-1, nS):
        j = -1
        for i in range(n):
            xr[i] = s[i]
        if ri >= 0:
            j = S[ri]
            for i in range(n):
                xr[i] = s[i] - w[j] * M.xval(QT, var_of, tptr, tabv, i, j)
        for l in range(d):
            if not valid[l] or l == j:
                continue
            if ri < 0 and w[l] == 0 and nS >= k:
                continue
            if ri >= 0 and w[l] != 0:
                continue
            for i in range(n):
                xa[i] = M.xval(QT, var_of, tptr, tabv, i, l)
            for vv in range(-5, 6):
                v = float(vv)
                if ri < 0 and w[l] == v:
                    continue
                if ri >= 0 and v == 0.0:
                    continue
                if ri < 0 and v == 0.0 and nS <= 1:
                    continue
                for i in range(n):
                    s2[i] = xr[i] + (v - (w[l] if ri < 0 else 0.0)) * xa[i]
                out_r.append(j); out_a.append(l); out_v.append(v); out_L.append(nbhd.lossof(s2, y, c, a, b))
    return np.array(out_r), np.array(out_a), np.array(out_v), np.array(out_L)

def apply(w, r, a_, v):
    w = w.copy()
    if r >= 0: w[r] = 0.0
    w[a_] = v
    return w

def main():
    ds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else suite_datasets("visible_fine")
    ks = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else K_VALUES
    Mtop = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    tot1 = tot2 = 0.0; nimp = 0
    for name in ds:
        X, y, *_ = load_problem(name, "visible_fine")
        for k in ks:
            m = M.make_model(k, 60.0).fit(X, y)
            D = M.Data(X, y); w = m.coef_.astype(np.float64)
            st = M.start_state(w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N); L0 = st[1] / D.N
            S = np.flatnonzero(w).astype(np.int64)
            t0 = time.time()
            R, A, Vv, L = all_singles(st[0], w, S, D.valid, D.y, D.c, st[2], st[3], D.QT, D.var_of, D.tptr, D.tabv, k)
            L = L / D.N
            b1 = L.min() - L0
            o = np.argsort(L)[:Mtop]
            best2 = 0.0
            for p in range(len(o)):
                for q in range(p + 1, len(o)):
                    i1, i2 = o[p], o[q]
                    cols1 = {A[i1], R[i1]}; cols2 = {A[i2], R[i2]}
                    if (cols1 & cols2) - {-1}:
                        continue
                    w2 = apply(apply(w, R[i1], A[i1], Vv[i1]), R[i2], A[i2], Vv[i2])
                    nz = np.count_nonzero(w2)
                    if nz > k or nz == 0:
                        continue
                    l2 = M.start_state(w2, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)[1] / D.N - L0
                    best2 = min(best2, l2)
            tot1 += min(b1, 0); tot2 += min(best2, min(b1, 0)); nimp += best2 < -1e-7
            print(f"{name:12s} k{k:2d} single {b1:+.6f} pair {best2:+.6f}  {time.time()-t0:.1f}s", flush=True)
    n = len(ds) * len(ks)
    print(f"mean single gain {tot1/n:.6f}  mean pair-or-single gain {tot2/n:.6f}  pair improves {nimp}/{n}")

if __name__ == "__main__":
    main()
