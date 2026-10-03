"""Per-problem phase fractions (data, beam, round, ils-starts, kicks) of serial fits; geometric means."""
import importlib.util, os, sys, time
import numpy as np
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
F = []; tt = []
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    for k in K_VALUES:
        t0 = time.perf_counter(); m = mod.make_model(k, 60.0).fit(X, y); t = time.perf_counter() - t0
        f = np.append(m.timing_, t - m.timing_.sum()); F.append(f / t); tt.append(t)
F = np.array(F)
print("mean fractions data/beam/round/ils/kicks/other:", F.mean(0).round(3), " geo time", np.exp(np.mean(np.log(tt))).round(4))
