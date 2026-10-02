"""Interleaved A/B timing and loss of two solver files over the suite (serial, single thread).
uv run python tools/ab.py A.py B.py [ks] [datasets]"""
import os, sys, time, importlib.util, csv
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem, suite_datasets
DATASETS = suite_datasets('visible_fine')
from evaluate import calibrate, auc
mods = []
for i, f in enumerate(sys.argv[1:3]):
    spec = importlib.util.spec_from_file_location(f'm{i}', f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods.append(m)
ks = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else '3,4,5,7,10').split(',')]
ds = sys.argv[4].split(',') if len(sys.argv) > 4 else DATASETS
best = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
T = [[], []]; R = [[], []]; A = [[], []]
for d in ds:
    X, y, Xte, yte, _ = load_problem(d, 'visible_fine')
    for k in ks:
        line = f"{d:12s} k={k:2d}"
        for i, m in enumerate(mods):
            ts = []
            for rep in range(int(os.environ.get("REPS", "2"))):
                t = time.perf_counter(); mod = m.make_model(k, 60.0).fit(X, y); ts.append(time.perf_counter() - t)
            T[i].append(min(ts)); l = calibrate(X @ mod.coef_, y)[2]; R[i].append(l - best[(d, k)]); A[i].append(auc(Xte @ mod.coef_, yte))
            line += f" | t={T[i][-1]:.3f} reg={1e3*R[i][-1]:+.3f} auc={A[i][-1]:.4f}"
        print(line, flush=True)
for i in range(2):
    print(f"{sys.argv[1+i]:40s} geo_t={np.exp(np.mean(np.log(T[i]))):.4f} reg={1e3*np.mean(R[i]):+.4f}e-3 auc={np.mean(A[i]):.4f}")
print(f"time ratio B/A = {np.exp(np.mean(np.log(np.array(T[1]) / np.array(T[0])))):.3f}")
