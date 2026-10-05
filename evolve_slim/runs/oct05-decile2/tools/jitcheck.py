"""Check that no numba kernel compiles a new signature during fits on the suite (after import warm-up)."""
import importlib.util, os, sys
import numba
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1])); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def sigs():
    return {k: len(v.signatures) for k, v in vars(M).items() if isinstance(v, numba.core.registry.CPUDispatcher)}
s0 = sigs()
import time
for name in suite_datasets("visible"):
    X, y, *_ = load_problem(name, "visible")
    for k in (3, 4, 5, 7, 10):
        a = sigs(); t = time.perf_counter()
        M.make_model(k, 60).fit(X, y)
        b = sigs()
        new = {q: b[q] - a.get(q, 0) for q in b if b[q] != a.get(q, 0)}
        if new:
            print(name, k, f"{time.perf_counter() - t:.2f}s", new, flush=True)
s1 = sigs()
print("new compilations:", {k: s1[k] - s0.get(k, 0) for k in s1 if s1[k] != s0.get(k, 0)} or "none")
