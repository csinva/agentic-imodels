import sys, os, time, importlib.util, numpy as np
sys.path.insert(0, "src")
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("m", "slim.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tot = {}; cnt = {}
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    for k in K_VALUES:
        mod = m.make_model(k, 60.0).fit(X, y)
        D = m.Data(X, y); w = mod.coef_.astype(float); L0 = mod.train_loss_
        cl = lambda v: m.start_state(v, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)[1] / D.N
        T = {}
        a = np.abs(w); sg = np.sign(w)
        T['shrink1'] = sg * np.maximum(a - 1, 1) * (a > 0)
        T['grow1'] = sg * np.minimum(a + 1, 5)
        for f in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0]:
            T[f'x{f}'] = np.clip(np.round(w * f), -5, 5)
        # shrink only the large ones / grow only small ones
        T['shrinkbig'] = np.where(a >= 3, sg * (a - 1), w)
        T['growsmall'] = np.where((a > 0) & (a <= 2), sg * (a + 1), w)
        best = None
        for key, v in T.items():
            if np.count_nonzero(v) == 0 or np.array_equal(v, w): continue
            l = cl(v); d = l - L0
            if d < -1e-9:
                cnt[key] = cnt.get(key, 0) + 1; tot[key] = tot.get(key, 0) + d
                print(name, k, key, round(d, 6), flush=True)
print(cnt); print({k: v / 70 for k, v in tot.items()})
