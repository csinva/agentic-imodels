import sys, time, importlib.util, numpy as np
sys.path.insert(0, "src")
from suite import load_problem, DATASETS, K_VALUES
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ks = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else K_VALUES
rows = []; losses = []
for d in DATASETS:
    X, y, *_ = load_problem(d)
    for k in ks:
        t = time.perf_counter(); mod = m.make_model(k, 60).fit(X, y); t = time.perf_counter() - t
        rows.append([t] + list(mod.timing_)); losses.append(mod.train_loss_)
        print(f"{d:12s} k={k:2d} total={t:.3f} " + " ".join(f"{v:.3f}" for v in mod.timing_), flush=True)
R = np.array(rows)
np.save("phase_losses_%s.npy" % sys.argv[3], np.array(losses)) if len(sys.argv) > 3 else None
print("mean loss", np.mean(losses))
print("geo mean total", np.exp(np.mean(np.log(R[:, 0]))))
print("mean share (data, beam, round, ils):", np.round((R[:, 1:] / R[:, :1]).mean(0), 3))
