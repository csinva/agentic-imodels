"""Serial in-process fits of every problem; prints time per phase (data, beam, rounding, ILS) summed per dataset
and the train loss. usage: uv run tools/prof.py <file.py> [datasets] [ks]"""
import importlib.util, os, sys, time
import numpy as np
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else suite_datasets("visible_fine")
ks = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else K_VALUES
tot = np.zeros(4); lt = []
for name in ds:
    Xtr, ytr, *_ = load_problem(name, "visible_fine")
    T = np.zeros(4)
    for k in ks:
        t0 = time.perf_counter(); m = mod.make_model(k, 60.0).fit(Xtr, ytr); dt = time.perf_counter() - t0
        T += m.timing_[:4]; lt.append(dt)
    tot += T
    print(f"{name:14s} n={Xtr.shape[0]:6d} d={Xtr.shape[1]:5d} data {T[0]:.3f} beam {T[1]:.3f} round {T[2]:.3f} ils {T[3]:.3f}", flush=True)
print(f"TOTAL data {tot[0]:.3f} beam {tot[1]:.3f} round {tot[2]:.3f} ils {tot[3]:.3f}  geo {np.exp(np.mean(np.log(lt))):.4f}")
