"""Refit historical versions (from git blobs) on the official split; sibling-free mean rank."""
import csv, os, sys, subprocess, types, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
E=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'evolve'); J=os.environ.get('GEN_DIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, E); sys.path.insert(0, f'{E}/src'); sys.path.insert(0, f'{J}/histmods')
VERSIONS = [("v1","fbee076"),("v4","f5068f5"),("v8","a7c5ecf"),("v10","2670bfb"),("v18","ac24544"),("v24","5078a51"),
            ("v31","16e8a4d"),("v33","90e2552"),("v34","7c57f3d"),("v35","83b4cec"),
            ("v41","6cc658d"),("v44","729b16b")]
if len(sys.argv) > 1: VERSIONS = [tuple(a.split(":")) for a in sys.argv[1:]]
names = open(f'{J}/data/names.txt').read().split('\n')
out = f'{J}/results/hist_ranks.csv'
done = set()
if os.path.exists(out): done = {r['version'] for r in csv.DictReader(open(out))}
else: open(out,'w').write('version,commit,model,dataset,rmse,seconds\n')
def load_class(commit):
    """Write the version's class section to a real module so worker processes can import it."""
    import importlib
    src = subprocess.check_output(['git','show',f'{commit}:evolve/interpretable_regressor.py'], cwd=os.path.dirname(E), text=True)
    cut = src.index('# Evaluation (do not edit anything below this line)')
    head = src[:cut]
    head = head.replace('sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))', f'sys.path.insert(0, "{E}/src")')
    head = head.replace('os.path.dirname(__file__)', f'"{E}"')
    head = head.replace('_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])', '')
    head = head.replace('.__module__ = "interpretable_regressor"', f'.__module__ = "hist_{commit}"')
    path = f'{J}/histmods/hist_{commit}.py'
    open(path, 'w').write(head)
    mod = importlib.import_module(f'hist_{commit}')
    return mod.model_shorthand_name, mod.model_defs[0][1]
from joblib import Parallel, delayed
def one(ds, est):
    from copy import deepcopy
    z = np.load(f"{J}/data/{ds.replace('/','__')}.npz"); t=time.time()
    try:
        m = deepcopy(est).fit(z['Xtr'], z['ytr']); r = float(np.sqrt(np.mean((m.predict(z['Xte'])-z['yte'])**2)))
    except Exception as e:
        r = float('nan'); print(f"    {ds} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    return ds, r, time.time()-t
for ver, commit in VERSIONS:
    if ver in done: continue
    try:
        name, est = load_class(commit)
    except Exception as e:
        print(f"{ver} {commit}: cannot load ({type(e).__name__}: {str(e)[:100]})", flush=True); continue
    print(f"=== {ver} {commit} {name}", flush=True); t0=time.time()
    with Parallel(n_jobs=3, return_as="generator") as par:
        for ds, r, sec in par(delayed(one)(ds, est) for ds in names):
            open(out,'a').write(f"{ver},{commit},{name},{ds},{r:.6f},{sec:.0f}\n")
    print(f"    done in {time.time()-t0:.0f}s", flush=True)
print("HIST DONE", flush=True)
