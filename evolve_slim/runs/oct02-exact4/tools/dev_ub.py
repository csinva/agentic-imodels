"""certify() alone with a given ub (analysis only: how fast would the search be with a better incumbent)."""
import importlib.util, os, re, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem
ds, k, ub, lim = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
sets = sys.argv[5:]
src = open(os.path.join(here, "slim.py")).read().split("# ===========================================================================\n# Evaluation loop")[0]
for kv in sets:
    a, v = kv.split("=", 1)
    src, n = re.subn(rf"^{a} = [^#\n]*", f"{a} = {v}  ", src, count=1, flags=re.M); assert n == 1
tmp = os.path.join(here, "tools", f"_mod_{os.getpid()}.py"); open(tmp, "w").write(src)
spec = importlib.util.spec_from_file_location("slimmod", tmp); M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M); os.remove(tmp)
X, y, _, _, _ = load_problem(ds)
t0 = time.perf_counter()
lb, w, l = M.certify(np.asarray(X, float), y, k, np.zeros(X.shape[1]), ub, t0 + lim)
print(f"{ds} k={k} ub={ub} t={time.perf_counter()-t0:.1f}s lb={lb:.8f} newub={l:.8f} stats={list(M.certify.stats)}", flush=True)
