"""cProfile of a solver over the visible suite (one fit per problem after import warm-up), excluding datasets
given in argv[2] (default: spambase, whose time is a small share of the geo mean). usage: uv run tools/prof.py f.py [excl]"""
import cProfile, pstats, importlib.util, os, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("cand", os.path.abspath(sys.argv[1])); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
excl = set((sys.argv[2] if len(sys.argv) > 2 else "spambase").split(","))
data = [(load_problem(n)[:2]) for n in suite_datasets("visible") if n not in excl]
pr = cProfile.Profile(); pr.enable()
for X, y in data:
    for k in K_VALUES:
        M.make_model(k, 60.0).fit(X, y)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(25)
