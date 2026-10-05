"""Beam / start / ILS-end probe of a fine-lineage solver (m*) and of v35 on one problem.
usage: uv run tools/probe.py solver.py dataset k [v35]"""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem
from evaluate import score_points
f, ds, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(f))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
Xtr, ytr, Xte, yte, names = load_problem(ds)
def show(w):
    return {names[j]: int(w[j]) for j in np.flatnonzero(w)}
def L(w):
    return score_points(np.asarray(w, float), Xtr, ytr, Xte, yte)
t = time.perf_counter(); m = M.make_model(k, 60.0).fit(Xtr, ytr); t = time.perf_counter() - t
print(f"final {m.train_loss_:.5f} t={t:.3f}", show(m.coef_))
D = M.Data(Xtr, ytr)
if len(sys.argv) > 4:  # v35
    P = M.beam_search(D, k, 10, 10)
    for nd in P:
        w, l = M.calib_round(D, nd)
        print(f"  beam cont {nd.loss/D.N:.5f} round {l:.5f}", [names[j] for j in nd.S], show(w))
else:
    st = M.beam_search(D, k, M.PARENT, M.CHILD)
    RR, LL, KK = M.round_all(st[2], st[1], st[3], st[4], st[5], st[7], st[8], st[9], D.QT, D.var_of, D.tptr, D.tabv,
                             20, 5.0, M._normals(7, D.n), D.N)
    for q in range(len(LL)):
        S = st[7][q, :st[8][q]]
        w = np.zeros(D.d); w[S] = RR[q, :st[8][q]]
        print(f"  beam cont {st[10][q]/D.N:.5f} round {LL[q]:.5f}", [names[j] for j in S], show(w))
    print("  start losses", [(round(a, 5), round(b, 5)) for a, b in m.start_losses_])
