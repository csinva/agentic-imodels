"""cProfile of serial fits. usage: uv run tools/cprof.py <file.py> [datasets] [ks] [n]"""
import cProfile, pstats, importlib.util, os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1]))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else suite_datasets("visible_fine")
ks = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else K_VALUES
P = [(load_problem(n, "visible_fine")[:2], k) for n in ds for k in ks]
pr = cProfile.Profile(); pr.enable()
for (X, y), k in P:
    mod.make_model(k, 60.0).fit(X, y)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(int(sys.argv[4]) if len(sys.argv) > 4 else 30)
