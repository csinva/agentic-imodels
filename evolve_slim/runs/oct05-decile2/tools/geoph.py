"""Serial per-problem phase times of a solver over the suite; geo-mean total and the mean share of each phase
(data, beam, rounding, ILS starts, refit swaps, polish+rest). usage: uv run tools/geoph.py solver.py"""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
acc = {}
def wrap(name):
    f = getattr(M.SparseIntegerClassifier, name)
    def g(self, *a, **kw):
        t = time.perf_counter(); r = f(self, *a, **kw); acc[name] = acc.get(name, 0) + time.perf_counter() - t; return r
    setattr(M.SparseIntegerClassifier, name, g)
for nm in ("_rswap", "_polish", "_upscale"):
    if hasattr(M.SparseIntegerClassifier, nm): wrap(nm)
rows = []
for name in suite_datasets("visible"):
    X, y, *_ = load_problem(name)
    for k in K_VALUES:
        best = None
        for _ in range(2):
            acc.clear()
            t = time.perf_counter(); m = M.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t
            ph = list(m.timing_[:3]) + [m.timing_[3] - sum(acc.values()), acc.get("_rswap", 0), acc.get("_polish", 0) + acc.get("_upscale", 0)]
            if best is None or t < best[0]: best = (t, ph)
        rows.append(best)
T = np.array([r[0] for r in rows]); P = np.array([r[1] for r in rows])
print(f"geo {np.exp(np.mean(np.log(T)))*1e3:.2f} ms; mean share data {np.mean(P[:,0]/T):.2f} beam {np.mean(P[:,1]/T):.2f} round {np.mean(P[:,2]/T):.2f} starts {np.mean(P[:,3]/T):.2f} rswap {np.mean(P[:,4]/T):.2f} polish+ups {np.mean(P[:,5]/T):.2f}")
