"""Sum of stage times (data, beam, round, ILS) over the suite, and geo mean: uv run python tools/stages.py file"""
import os, sys, time, importlib.util
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem, suite_datasets
DATASETS = suite_datasets('visible_fine')
f = sys.argv[1]
spec = importlib.util.spec_from_file_location('m', f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ks = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else '3,5,10').split(',')]
tot = []; ts = []
for dname in DATASETS:
    X, y, *_ = load_problem(dname, 'visible_fine')
    row = []
    for k in ks:
        t = time.perf_counter(); mod = m.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t
        ts.append(t); tot.append(mod.timing_); row.append(np.round(mod.timing_ * 1e3, 1))
    print(dname, row)
tot = np.array(tot)
print('geo', np.exp(np.mean(np.log(ts))), 'geo per stage', np.exp(np.mean(np.log(tot + 1e-5), 0)))
