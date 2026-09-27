"""Dev: time numba kernels called from Python by wrapping module globals."""
import sys, time, importlib.util, collections
sys.path.insert(0, 'src')
from suite import load_problem
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
T = collections.defaultdict(float); C = collections.Counter()
for name in ['chain_dp', 'chain_shift', 'bin_scores', 'chain_counts', 'move_tables', 'newton_estimate', 'newton_point_loss',
             'bin_stats', 'exact_move', 'calibrate_bins', 'beam_search', 'calib_round', 'Data', 'ScoreState', 'sparse_score']:
    f = getattr(m, name)
    def wrap(*a, _f=f, _n=name, **k):
        t = time.perf_counter(); r = _f(*a, **k); T[_n] += time.perf_counter() - t; C[_n] += 1; return r
    setattr(m, name, wrap)
for p in sys.argv[2].split(';'):
    ds, k = p.split(); T.clear(); C.clear()
    Xtr, ytr, _, _, _ = load_problem(ds, 'visible_fine')
    t = time.perf_counter(); m.make_model(int(k), 60.).fit(Xtr, ytr); tot = time.perf_counter() - t
    print(f"{ds} {k} total {tot:.3f}: " + ", ".join(f"{n} {T[n]:.3f}/{C[n]}" for n in sorted(T, key=lambda n: -T[n])))
