import sys, time, importlib.util, numpy as np, collections
sys.path.insert(0, "src")
from suite import load_problem, DATASETS, K_VALUES
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
acc = collections.defaultdict(float); cnt = collections.Counter()
import numba
for name in dir(m):
    f = getattr(m, name)
    if isinstance(f, numba.core.registry.CPUDispatcher):
        def wrap(f, name):
            def g(*a, **k):
                t = time.perf_counter(); r = f(*a, **k); acc[name] += time.perf_counter() - t; cnt[name] += 1; return r
            return g
        setattr(m, name, wrap(f, name))
for cls in ["ScoreState", "ILS"]:
    pass
tot = 0; per = collections.defaultdict(float)
for d in DATASETS:
    X, y, *_ = load_problem(d)
    for k in K_VALUES:
        before = dict(acc)
        t = time.perf_counter(); m.make_model(k, 60).fit(X, y); t = time.perf_counter() - t
        tot += np.log(t)
        for kk in acc:  # share of this problem's time
            per[kk] += (acc[kk] - before.get(kk, 0)) / t / 70
print("geo", np.exp(tot / 70))
for kk, v in sorted(per.items(), key=lambda x: -x[1]): print(f"{kk:24s} share {v:.3f} calls {cnt[kk]}")
