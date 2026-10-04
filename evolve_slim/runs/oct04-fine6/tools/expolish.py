"""Probe: after a solver's fit, repeat {exhaustive exact best single move; ILS from it} until no exact single move
improves. usage: SOLVER=file.py uv run tools/expolish.py dataset"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbhd
M = nbhd.M
from suite import load_problem, K_VALUES
name = sys.argv[1]
X, y, *_ = load_problem(name, "visible_fine")
for k in K_VALUES:
    m = M.make_model(k, 60.0); m.fit(X, y)
    D = M.Data(X, y); w = m.coef_.astype(np.float64)
    L0 = m.train_loss_; L = L0; rounds = 0; t0 = time.time()
    ils = M.ILS(D, k, n_exact=M.NEXACT, n_screen=M.NSCREEN)
    while True:
        st = M.start_state(w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
        S = np.flatnonzero(w).astype(np.int64)
        bL, bj, bl, bv = nbhd.best_swaps(st[0], w, S, D.valid, D.y, D.c, st[2], st[3], D.QT, D.var_of, D.tptr, D.tabv, k)
        if bL / D.N >= st[1] / D.N - 1e-9:
            break
        w1 = w.copy()
        if bj >= 0: w1[bj] = 0.0
        w1[bl] = bv
        l2, w2 = ils.run(w1)
        if l2 > bL / D.N: l2, w2 = bL / D.N, w1
        w, L = w2, l2; rounds += 1
    print(f"{name} k{k} before {L0:.6f} after {L:.6f} diff {L-L0:+.6f} rounds {rounds} {time.time()-t0:.1f}s", flush=True)
