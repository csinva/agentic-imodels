import os, sys, cProfile, pstats, importlib.util
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
from suite import load_problem
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
probs = [(d, int(k)) for d in sys.argv[2].split(',') for k in sys.argv[3].split(',')]
data = {d: load_problem(d, 'visible_fine')[:2] for d, _ in probs}
pr = cProfile.Profile(); pr.enable()
for d, k in probs:
    m.make_model(k, 60.0).fit(*data[d])
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(18)
