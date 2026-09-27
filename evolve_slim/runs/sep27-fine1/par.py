"""Dev: run solver files on chosen problems x kick seeds in parallel; print mean regret (e-4) and mean time per file.
usage: uv run python par.py file1.py,file2.py "ds:k,ds:k" seeds"""
import sys, os, time, importlib.util, csv
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, 'src')
import numpy as np
import multiprocessing as mp
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
RUGGED = "heart:7,heart:10,ilpd:7,ilpd:10,ionosphere:4,ionosphere:5,ionosphere:7,ionosphere:10,mushroom:5,mushroom:7,mushroom:10,magic:4,magic:7,breastcancer:10,australian:10"

def job(args):
    f, name, k, seed = args
    os.environ["SLIM_SEED"] = str(seed)
    from suite import load_problem
    from evaluate import calibrate
    spec = importlib.util.spec_from_file_location("m", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    Xtr, ytr, _, _, _ = load_problem(name, 'visible_fine')
    t = time.perf_counter(); mdl = m.make_model(k, 60.0).fit(Xtr, ytr); t = time.perf_counter() - t
    _, _, L = calibrate(Xtr @ mdl.coef_, ytr)
    return f, name, k, seed, (L - bk[(name, k)]) * 1e4, t

if __name__ == '__main__':
    files = sys.argv[1].split(',')
    probs = [(p.split(':')[0], int(p.split(':')[1])) for p in (RUGGED if sys.argv[2] == 'rugged' else sys.argv[2]).split(',')]
    seeds = [int(s) for s in sys.argv[3].split(',')]
    tasks = [(f, n, k, s) for f in files for (n, k) in probs for s in seeds]
    with mp.get_context('spawn').Pool(min(len(tasks), 60)) as pool:
        res = pool.map(job, tasks, chunksize=1)
    for f in files:
        R = np.array([[r[4] for r in res if r[0] == f and (r[1], r[2]) == p] for p in probs])
        T = np.array([[r[5] for r in res if r[0] == f and (r[1], r[2]) == p] for p in probs])
        print(f"{os.path.basename(f):28s} mean regret {R.mean():8.2f} (per-seed {np.round(R.mean(0),1)})  geo time {np.exp(np.log(T).mean()):.3f}s")
        if len(sys.argv) > 4:
            for p, r in zip(probs, R): print('   ', p, np.round(r, 1))
