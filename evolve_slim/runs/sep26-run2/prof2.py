import sys, time, cProfile, pstats, importlib.util
import numpy as np
sys.path.insert(0, "src")
from suite import load_problem
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng = np.random.default_rng(0); Xw = (rng.random((300, 12)) < 0.4).astype(float); yw = (Xw[:, 0] + Xw[:, 1] + rng.random(300) > 1.2).astype(np.int64)
m.make_model(3, 5.0).fit(Xw, yw)
for arg in sys.argv[2:]:
    d, k = arg.split(":"); X, y, *_ = load_problem(d)
    pr = cProfile.Profile(); t = time.time(); pr.enable(); mod = m.make_model(int(k), 60).fit(X, y); pr.disable()
    print(d, k, round(time.time() - t, 3), getattr(mod, "train_loss_", None)); pstats.Stats(pr).sort_stats("tottime").print_stats(8)
