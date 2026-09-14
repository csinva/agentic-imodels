import os
"""Multi-seed evaluator on the cached imodels-65 suite.

  uv run run.py NAME [--model gpmodel.BinGP] [--seeds 0,1,2] [--jobs 4] [k=v ...]

Seed 0 is the official harness split. Seed s>0 pools train+test, redraws a
split of the same train size with RandomState(s), and re-standardises y by the
new train statistics. Rows are checkpointed to results/NAME.csv so a killed run
resumes where it stopped.
"""
import csv, importlib, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

J = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, J)
DATA = f"{J}/data" + os.environ.get("GEN_DATA_SUFFIX", ""); RES = f"{J}/results" + os.environ.get("GEN_DATA_SUFFIX", ""); os.makedirs(RES, exist_ok=True)

name = sys.argv[1]
opts = dict(model="gpmodel.BinGP", seeds="0", jobs="4", ntrain="0", tag="", maxn="0", minn="0", only="")
kwargs = {}
for a in sys.argv[2:]:
    if a.startswith("--"):
        k, v = a[2:].split("=", 1); opts[k] = v
    else:
        k, v = a.split("=", 1)
        try: kwargs[k] = eval(v)
        except Exception: kwargs[k] = v
seeds = [int(x) for x in opts["seeds"].split(",")]
mod, cls = (opts["model"].rsplit(".", 1) if "." in opts["model"] else (None, None))

def make():
    if opts["model"] == "EBM":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000, **kwargs)
    return getattr(importlib.import_module(mod), cls)(**kwargs)

def split(ds, seed):
    z = np.load(f"{DATA}/{ds.replace('/', '__')}.npz")
    if seed == 0:
        return z["Xtr"], z["Xte"], z["ytr"], z["yte"]
    X = np.vstack([z["Xtr"], z["Xte"]]); y = np.concatenate([z["ytr_raw"], z["yte_raw"]])
    idx = np.random.RandomState(seed).permutation(len(y)); ntr = len(z["ytr"])
    tr, te = idx[:ntr], idx[ntr:]
    ym, ys = y[tr].mean(), y[tr].std()
    ys = ys if ys > 0 else 1.0
    return X[tr], X[te], (y[tr] - ym) / ys, (y[te] - ym) / ys

def one(ds, seed):
    Xtr, Xte, ytr, yte = split(ds, seed)
    nt = int(opts["ntrain"])
    if nt and len(ytr) > nt:
        keep = np.random.RandomState(1000 + seed).choice(len(ytr), nt, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
        ym, ys = ytr.mean(), ytr.std(); ys = ys if ys > 0 else 1.0
        ytr, yte = (ytr - ym) / ys, (yte - ym) / ys
    t0 = time.time()
    try:
        m = make().fit(Xtr, ytr)
        r = float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))
    except Exception as e:
        r = float("nan"); print(f"  {ds} seed{seed} FAILED {type(e).__name__}: {e}", flush=True)
    return ds, seed, r, time.time() - t0

names = open(f"{DATA}/names.txt").read().split("\n")
path = f"{RES}/{name}{opts['tag']}.csv"
done = set()
if os.path.exists(path):
    done = {(r["dataset"], int(r["seed"])) for r in csv.DictReader(open(path))}
else:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "seed", "rmse", "seconds"])
if int(opts["maxn"]):
    names = [ds for ds in names if len(np.load(f"{DATA}/{ds.replace('/', '__')}.npz")["ytr"]) <= int(opts["maxn"])]
if opts["only"]:
    import re as _re
    names = [ds for ds in names if _re.search(opts["only"], ds)]
if int(opts["minn"]):
    names = [ds for ds in names if len(np.load(f"{DATA}/{ds.replace('/', '__')}.npz")["ytr"]) >= int(opts["minn"])]
todo = [(ds, s) for s in seeds for ds in names if (ds, s) not in done]
print(f"{name}: {len(todo)} fits to run ({len(done)} cached)  kwargs={kwargs}", flush=True)

from joblib import Parallel, delayed
t0 = time.time()
# largest datasets first so the tail of the run is short fits
def size(ds): return -np.load(f"{DATA}/{ds.replace('/', '__')}.npz")["Xtr"].size
todo.sort(key=lambda t: size(t[0]))
with Parallel(n_jobs=int(opts["jobs"]), return_as="generator") as par:
    for ds, seed, r, sec in par(delayed(one)(ds, s) for ds, s in todo):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([ds, seed, f"{r:.6f}", f"{sec:.1f}"])
        print(f"  {ds:<34} seed{seed} {r:.4f} ({sec:.0f}s)", flush=True)
print(f"done in {time.time()-t0:.0f}s", flush=True)
