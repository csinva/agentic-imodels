import os, sys, time, importlib.util, collections
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
from suite import load_problem
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cnt = collections.Counter()
def wrap(obj, name):
    f = getattr(obj, name)
    def g(*a, **k):
        cnt[name] += 1; return f(*a, **k)
    setattr(obj, name, g)
wrap(m.ILS, 'run'); wrap(m.ILS, 'check'); wrap(m.ScoreState, 'eval'); wrap(m.ScoreState, '__init__')
for d in sys.argv[2].split(','):
    X, y = load_problem(d, 'visible_fine')[:2]
    for k in sys.argv[3].split(','):
        cnt.clear(); mod = m.make_model(int(k), 60.0).fit(X, y)
        print(d, k, dict(cnt), (mod.timing_*1e3).round(1))
