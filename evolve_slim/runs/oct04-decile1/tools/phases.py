"""Per-phase fit time (data, beam, rounding, ILS+rswap+polish) of a solver on datasets, all k; serial.
usage: uv run tools/phases.py solver.py ds1,ds2 [reps]"""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for ds in sys.argv[2].split(","):
    X, y, *_ = load_problem(ds)
    for k in K_VALUES:
        best = None
        for _ in range(reps):
            t = time.perf_counter(); m = M.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t
            if best is None or t < best[0]:
                best = (t, m.timing_, m.train_loss_)
        t, tm, L = best
        print(f"{ds:12s} k{k:2d} t {t*1e3:7.1f} ms  phases " + " ".join(f"{x*1e3:6.1f}" for x in tm) + f"  L {L:.5f}", flush=True)
