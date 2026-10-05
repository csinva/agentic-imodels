"""Serial interleaved A/B timing: for every problem fit A and B alternately R times (min time each); prints the
geometric-mean time ratio B/A, per-dataset ratios, and how many problems give different points / their mean loss diff.
usage: uv run tools/ab.py A.py B.py [R] [datasets]"""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
def load(p, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.abspath(p)); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
A, B = load(sys.argv[1], "ma"), load(sys.argv[2], "mb")
R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ds = sys.argv[4].split(",") if len(sys.argv) > 4 else suite_datasets("visible")
rat, ndiff, dl = [], 0, []
per = {}
for name in ds:
    X, y, *_ = load_problem(name, "visible")
    for k in K_VALUES:
        ta, tb = np.inf, np.inf
        for _ in range(R):
            t0 = time.perf_counter(); ma = A.make_model(k, 60.0).fit(X, y); ta = min(ta, time.perf_counter() - t0)
            t0 = time.perf_counter(); mb = B.make_model(k, 60.0).fit(X, y); tb = min(tb, time.perf_counter() - t0)
        rat.append(tb / ta); per.setdefault(name, []).append(tb / ta)
        if not np.array_equal(ma.coef_, mb.coef_):
            ndiff += 1; dl.append(mb.train_loss_ - ma.train_loss_)
print(f"geo ratio B/A {np.exp(np.mean(np.log(rat))):.3f}  differing points {ndiff}/70  mean dloss {np.sum(dl)/70:.6f}  load {os.getloadavg()[0]:.0f}")
print({k: round(float(np.exp(np.mean(np.log(v)))), 2) for k, v in per.items()})
