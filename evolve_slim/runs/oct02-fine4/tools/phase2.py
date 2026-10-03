"""Per-dataset phase times (ms per fit, min of R reps) for a solver file. usage: uv run tools/phase2.py file.py [R]"""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
print("dataset         total   data   beam  round    ils  kicks (ms per fit, mean over k)")
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    acc = np.zeros(6)
    for k in K_VALUES:
        best = None
        for _ in range(R):
            t0 = time.perf_counter(); m = mod.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t0
            v = np.append(t, m.timing_)
            if best is None or v[0] < best[0]: best = v
        acc += best[:6]
    acc /= len(K_VALUES)
    print(f"{name:14s}" + "".join(f"{x*1e3:7.2f}" for x in acc))
