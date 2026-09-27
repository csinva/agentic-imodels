import os, sys, time, importlib.util, collections
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
from suite import load_problem
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
acc = collections.Counter()
def wrap(name, obj=m):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); acc[name] += time.perf_counter() - t; return r
    setattr(obj, name, g)
for nm in sys.argv[4].split(','):
    if '.' in nm:
        c, f = nm.split('.'); wrap(f, getattr(m, c))
    else:
        wrap(nm)
for d in sys.argv[2].split(','):
    X, y = load_problem(d, 'visible_fine')[:2]
    for k in sys.argv[3].split(','):
        t = time.perf_counter(); m.make_model(int(k), 60.0).fit(X, y); acc['TOTAL'] += time.perf_counter() - t
for k, v in acc.most_common():
    print(f"{k:24s} {v*1e3:8.1f} ms")
