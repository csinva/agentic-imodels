import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    mod.CNT[:] = 0; T = np.zeros(4)
    for k in K_VALUES:
        T += mod.make_model(k, 60.0).fit(X, y).timing_[:4]
    print(f"{name:14s} runs {mod.CNT[3]:4d} iters {mod.CNT[0]:5d} swaps {mod.CNT[1]:5d} succ {mod.CNT[2]:4d} slides {mod.CNT[4]:4d}  ils {T[3]:.3f}")
