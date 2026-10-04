"""Serial fits with the ILS split: time in start runs, refit swaps, extra phase, polish (fractions of fit time),
number of start runs. usage: uv run tools/ilsprof.py solver.py"""
import sys, os, time, importlib.util, numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("m", os.path.abspath(sys.argv[1])); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
acc = {}
cur = ["start"]
def wrap(cls, name, tag):
    f = getattr(cls, name)
    def g(self, *a, **kw):
        prev = cur[0]; cur[0] = tag; t = time.perf_counter()
        try: return f(self, *a, **kw)
        finally:
            acc[tag] = acc.get(tag, 0) + time.perf_counter() - t; cur[0] = prev
    setattr(cls, name, g)
runf = m.ILS.run
def run(self, *a, **kw):
    t = time.perf_counter(); r = runf(self, *a, **kw); dt = time.perf_counter() - t
    acc["run@" + cur[0]] = acc.get("run@" + cur[0], 0) + dt; acc["nrun@" + cur[0]] = acc.get("nrun@" + cur[0], 0) + 1
    return r
m.ILS.run = run
for nm, tag in [("_rswap", "rswap"), ("_polish", "polish"), ("_extra", "extra")]:
    if hasattr(m.SparseIntegerClassifier, nm): wrap(m.SparseIntegerClassifier, nm, tag)
tot = 0; ph = np.zeros(4)
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    for k in K_VALUES:
        t = time.perf_counter(); mod = m.make_model(k, 60.0).fit(X, y); tot += time.perf_counter() - t; ph += mod.timing_[:4]
print("total %.3fs  data %.3f beam %.3f round %.3f ils-all %.3f" % (tot, *ph))
for kk in sorted(acc): print(kk, round(acc[kk], 4) if not kk.startswith("nrun") else acc[kk])
