"""Stage timing of slim.py fits on selected problems: uv run python prof.py [file] [datasets] [ks]"""
import os, sys, time, importlib.util
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem
from evaluate import calibrate
f = sys.argv[1] if len(sys.argv) > 1 else 'slim.py'
spec = importlib.util.spec_from_file_location('m', f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ds = (sys.argv[2] if len(sys.argv) > 2 else 'magic,ionosphere,spambase,fico').split(',')
ks = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else '3,10').split(',')]
for dname in ds:
    X, y, *_ = load_problem(dname, 'visible_fine')
    for k in ks:
        t = time.perf_counter(); mod = m.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t
        l = calibrate(X @ mod.coef_, y)[2]
        print(f"{dname:12s} k={k:2d} n={X.shape[0]} d={X.shape[1]} t={t:.3f} loss={l:.6f} stages={np.round(getattr(mod,'timing_',[]),3)}", flush=True)
