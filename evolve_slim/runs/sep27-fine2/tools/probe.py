"""Loss of slim.py variants with custom kwargs on a set of problems, vs a reference file.
uv run python tools/probe.py file 'kw_json' datasets ks"""
import os, sys, time, json, importlib.util
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem, DATASETS
from evaluate import calibrate
f = sys.argv[1]; kw = json.loads(sys.argv[2])
spec = importlib.util.spec_from_file_location('m', f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ds = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else DATASETS
ks = [int(k) for k in (sys.argv[4] if len(sys.argv) > 4 else '3,4,5,7,10').split(',')]
import csv
best = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
regs, ts = [], []
for dname in ds:
    X, y, *_ = load_problem(dname, 'visible_fine')
    for k in ks:
        t = time.perf_counter(); mod = m.SparseIntegerClassifier(k=k, time_limit=60.0, **kw).fit(X, y); t = time.perf_counter() - t
        l = calibrate(X @ mod.coef_, y)[2]
        regs.append(l - best[(dname, k)]); ts.append(t)
        print(f"{dname:12s} k={k:2d} t={t:.3f} reg={1e3*regs[-1]:+.3f}e-3", flush=True)
print(f"MEAN reg={1e3*np.mean(regs):+.4f}e-3 geo_t={np.exp(np.mean(np.log(ts))):.3f}")
