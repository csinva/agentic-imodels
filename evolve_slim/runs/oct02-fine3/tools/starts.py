"""Per problem: rounded loss and ILS end loss of each start (regret x1e4 vs best known). uv run python tools/starts.py file [ks]"""
import os, sys, importlib.util, csv
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem, suite_datasets
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ks = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else '3,4,5,7,10').split(',')]
best = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
win = np.zeros(20)
for d in suite_datasets('visible_fine'):
    X, y, *_ = load_problem(d, 'visible_fine')
    for k in ks:
        mod = m.make_model(k, 60.0).fit(X, y)
        sl = np.array(mod.start_losses_); bk = best[(d, k)]
        ends = (sl[:, 1] - bk) * 1e4
        win[np.argmin(ends)] += 1
        print(f"{d:12s} k={k:2d} round " + " ".join(f"{v:7.1f}" for v in (sl[:, 0] - bk) * 1e4) + " | ils " + " ".join(f"{v:7.1f}" for v in ends))
print("argmin start index counts", win[:10])
