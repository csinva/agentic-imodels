"""Exhaustive exact neighbourhood of a solver's final point: best single value change / addition / swap (all
free valid columns, all values) and uniform shifts / rescales of the support, by exact calibrated loss.
usage: uv run tools/nbhd.py solver.py [datasets] [ks]"""
import importlib.util, os, sys, time
import numpy as np, numba as nb
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(os.environ.get("SOLVER", sys.argv[1] if len(sys.argv) > 1 else "slim.py")))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

@nb.njit(cache=False)
def lossof(s, y, c, a, b):
    inv, sv, Wp, Wn = M.bin_scores(s, y, c)
    if sv.shape[0] < 2:
        return 1e300
    L, a2, b2 = M.calibrate_bins(sv, Wp, Wn, a, b)
    return L

@nb.njit(cache=False)
def best_swaps(s, w, S, valid, y, c, a, b, QT, var_of, tptr, tabv, k):
    d = w.shape[0]; n = s.shape[0]
    xr = np.empty(n); xa = np.empty(n); s2 = np.empty(n)
    bestL = 1e300; bj = -1; bl = -1; bv = 0.0
    nS = S.shape[0]
    for ri in range(-1, nS):  # -1: no removal (addition / value change)
        if ri >= 0:
            j = S[ri]
            for i in range(n):
                xr[i] = s[i] - w[j] * M.xval(QT, var_of, tptr, tabv, i, j)
        else:
            j = -1
            for i in range(n):
                xr[i] = s[i]
        for l in range(d):
            if not valid[l]:
                continue
            if ri < 0 and w[l] == 0 and nS >= k:
                continue
            if ri >= 0 and w[l] != 0 and l != j:
                continue
            for i in range(n):
                xa[i] = M.xval(QT, var_of, tptr, tabv, i, l)
            base = xr
            for vv in range(-5, 6):
                v = float(vv)
                if v == 0.0:
                    continue
                if ri < 0 and w[l] == v:
                    continue
                if ri >= 0 and l == j:
                    continue
                for i in range(n):
                    s2[i] = base[i] + (v - (w[l] if ri < 0 else 0.0)) * xa[i]
                L = lossof(s2, y, c, a, b)
                if L < bestL:
                    bestL = L; bj = j; bl = l; bv = v
    return bestL, bj, bl, bv

def main():
    global tot
    ds = sys.argv[2].split(",") if len(sys.argv) > 2 else suite_datasets("visible_fine")
    ks = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else K_VALUES
    tot = 0.0
    for name in ds:
        X, y, *_ = load_problem(name, "visible_fine")
        for k in ks:
            m = M.make_model(k, 60.0).fit(X, y)
            D = M.Data(X, y); w = m.coef_.astype(np.float64)
            st = M.start_state(w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
            L0 = st[1] / D.N
            S = np.flatnonzero(w).astype(np.int64)
            t0 = time.time()
            bL, bj, bl, bv = best_swaps(st[0], w, S, D.valid, D.y, D.c, st[2], st[3], D.QT, D.var_of, D.tptr, D.tabv, k)
            bL /= D.N
            # shifts / rescales
            sh = {}
            for name2, f in [("+1", lambda w: np.where(w != 0, np.clip(w + 1, -5, 5), 0)), ("-1", lambda w: np.where(w != 0, np.clip(w - 1, -5, 5), 0)),
                             ("toward0", lambda w: w - np.sign(w)), ("away0", lambda w: np.clip(w + np.sign(w), -5, 5))]:
                w2 = f(w)
                if np.count_nonzero(w2) < 1: continue
                sh[name2] = M.start_state(w2.astype(np.float64), D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)[1] / D.N - L0
            gain = min(0.0, bL - L0); tot += gain
            print(f"{name:12s} k{k:2d} L {L0:.5f} best1 {bL - L0:+.6f} (rm {bj} add {bl} v {bv:+.0f}) shifts " +
                  " ".join(f"{a}:{v:+.5f}" for a, v in sh.items()) + f"  {time.time()-t0:.1f}s", flush=True)
    print("mean single-move gain", tot / (len(ds) * len(ks)))


if __name__ == "__main__":
    main()
