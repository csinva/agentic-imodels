"""Dev harness: fit selected problems with a solver file, print exact loss (harness calibrate) and time."""
import sys, time, importlib.util, os
os.environ.setdefault("NUMBA_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem
from evaluate import calibrate, auc
import csv
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}

def load(path):
    spec = importlib.util.spec_from_file_location("m" + str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

if __name__ == '__main__':
    files = sys.argv[1].split(',')
    ds = sys.argv[2].split(',')
    ks = [int(v) for v in sys.argv[3].split(',')]
    mods = [load(f) for f in files]
    tot = np.zeros(len(mods))
    for name in ds:
        Xtr, ytr, Xte, yte, _ = load_problem(name, 'visible_fine')
        for k in ks:
            row = f"{name:12s} k={k:2d}"
            for q, m in enumerate(mods):
                t = time.perf_counter(); mdl = m.make_model(k, 60.0).fit(Xtr, ytr); t = time.perf_counter() - t
                _, _, L = calibrate(Xtr @ mdl.coef_, ytr)
                r = L - bk[(name, k)]; tot[q] += r
                row += f" | {r*1e4:8.2f} {t:6.2f}s auc {auc(np.sign(1)*(Xte@mdl.coef_), yte):.4f}"
            print(row, flush=True)
    print("sum regret e-4:", (tot * 1e4).round(2))
