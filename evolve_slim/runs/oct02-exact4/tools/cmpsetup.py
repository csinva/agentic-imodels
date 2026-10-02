"""check that the fast setup gives identical arrays: run certify's setup part from both versions"""
import importlib.util, os, sys
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, DATASETS
def load(f, name):
    src = open(os.path.join(here, f)).read().split("# ===========================================================================\n# Evaluation loop")[0]
    src = src.replace("    order = np.argsort(Hone, kind=\"stable\").astype(np.int64)\n", "    order = np.argsort(Hone, kind=\"stable\").astype(np.int64)\n    certify.dbg = (crow.copy(), ccode.copy(), lval.copy(), cptr.copy(), lptr.copy(), Hone.copy(), order.copy())\n    return 0.0, w_inc, ub\n", 1)
    tmp = os.path.join(here, "tools", f"_cs_{name}_{os.getpid()}.py"); open(tmp, "w").write(src)
    spec = importlib.util.spec_from_file_location(name, tmp); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); os.remove(tmp)
    return M
A = load("slim.py", "a"); B = load("scratch/slim_dev.py", "b")
for ds in DATASETS:
    X, y, _, _, _ = load_problem(ds)
    A.certify(X, y, 5, np.zeros(X.shape[1]), 0.5, 1e12); B.certify(X, y, 5, np.zeros(X.shape[1]), 0.5, 1e12)
    da, db = A.certify.dbg, B.certify.dbg
    ok = all(np.array_equal(u, v) for u, v in zip(da[:5], db[:5])) and np.allclose(da[5], db[5], rtol=1e-12, atol=1e-9) and np.array_equal(da[6], db[6])
    print(ds, ok, np.max(np.abs(da[5] - db[5])))
