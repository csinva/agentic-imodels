import sys, time, importlib.util, numpy as np, numba
sys.path.insert(0, "src")
from suite import load_problem, DATASETS
spec = importlib.util.spec_from_file_location("s", "slim.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
before = {n: len(getattr(m, n).signatures) for n in dir(m) if isinstance(getattr(m, n), numba.core.registry.CPUDispatcher)}
for d in DATASETS:
    X, y, *_ = load_problem(d)
    for k in [3, 10]:
        m.make_model(k, 60).fit(X, y)
new = [n for n, c in before.items() if len(getattr(m, n).signatures) != c]
print("functions compiled after import:", new)
