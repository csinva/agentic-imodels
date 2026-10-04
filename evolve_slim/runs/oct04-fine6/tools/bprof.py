import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    mod.BT[:] = 0; T = np.zeros(4)
    for k in K_VALUES:
        T += mod.make_model(k, 60.0).fit(X, y).timing_[:4]
    print(f"{name:14s} beam {T[1]*1e3:6.1f}ms: rows {mod.BT[0]*1e3:5.1f} colsum {mod.BT[1]*1e3:5.1f} props {mod.BT[2]*1e3:5.1f} fits {mod.BT[3]*1e3:5.1f} select {mod.BT[4]*1e3:5.1f}")
