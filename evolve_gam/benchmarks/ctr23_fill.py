"""Fill every missing (dataset, model) pair for the CTR23 column.

CTR23 excludes only the datasets the imodels suite also uses; overlap with
TabArena is allowed, so the suite is 30 datasets. Earlier runs covered parts of
it: ctr23_eval.py did five models on the 28 that excluded TabArena's, and
ctr23_glassbox.py did five more on a smaller 21. This fills the gaps using the
same protocols, so every cell in the column comes out of one procedure.
"""
import csv, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from imodels import FIGSRegressor, RuleFitRegressor, HSTreeRegressorCV

J = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(J, '..', '..'))
sys.path.insert(0, os.path.join(J, '..', 'model'))
from addgp import BinGP

C = os.path.expanduser('~/.cache/imodels-evolve/ctr23')
RES = f'{J}/ctr23_results.csv'
rows = json.load(open(f'{J}/ctr23.json'))
todo = [r for r in rows if not r[6]]

done = set()
for p in (RES, f'{J}/../results/ctr23_results.csv'):
    if os.path.exists(p):
        done |= {(r['dataset'], r['model']) for r in csv.DictReader(open(p))}
if not os.path.exists(RES):
    with open(RES, 'w', newline='') as f:
        csv.writer(f).writerow(['dataset', 'model', 'rmse', 'seconds'])


def prep(did, target):
    df = pd.read_parquet(f'{C}/{did}.parquet')
    y = pd.to_numeric(df[target], errors='coerce').values.astype(float)
    Xdf = df.drop(columns=[target]); cols = []
    for c in Xdf.columns:
        sr = Xdf[c]
        if sr.dtype.name in ('category', 'object', 'string', 'bool'):
            cols.append(sr.astype('category').cat.codes.values.astype(float))
        else:
            v = pd.to_numeric(sr, errors='coerce').values.astype(float)
            md = np.nanmedian(v)
            cols.append(np.where(np.isfinite(v), v, md if np.isfinite(md) else 0.0))
    X = np.column_stack(cols) if cols else np.zeros((len(y), 1))
    ok = np.isfinite(y)
    return train_test_split(X[ok], y[ok], test_size=0.2, random_state=42)


for nm, tid, did, tgt, n, d, dup in sorted(todo, key=lambda r: int(r[4])):
    # the five glass-box models run from ctr23_glassbox.py instead: the imodels
    # build in this venv fails HSTreeRegressorCV on real data
    need = [m for m in ('AddGP','EBM','RF','GBM','Ridge','TabPFN') if (nm, m) not in done]
    if not need:
        continue
    try:
        Xtr, Xte, ytr, yte = prep(did, tgt)
    except Exception as e:
        print(f'{nm}: LOAD FAILED {type(e).__name__}', flush=True); continue
    ym, ys = ytr.mean(), ytr.std()
    if not np.isfinite(ys) or ys == 0:
        print(f'{nm}: degenerate target, skipped', flush=True); continue
    ytr_n, yte_n = (ytr - ym) / ys, (yte - ym) / ys

    # AddGP is fit on the raw target and rescaled, as in ctr23_eval.py
    fits = {
        'AddGP':        lambda: (BinGP().fit(Xtr, ytr).predict(Xte) - ym) / ys,
        'EBM':          lambda: ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000).fit(Xtr, ytr_n).predict(Xte),
        'RF':           lambda: RandomForestRegressor(random_state=42, n_jobs=4).fit(Xtr, ytr_n).predict(Xte),
        'GBM':          lambda: GradientBoostingRegressor(random_state=42).fit(Xtr, ytr_n).predict(Xte),
        'Ridge':        lambda: RidgeCV().fit(Xtr, ytr_n).predict(Xte),
        'FIGS_large':   lambda: FIGSRegressor(max_rules=20, random_state=42).fit(Xtr, ytr_n).predict(Xte),
        'RuleFit':      lambda: RuleFitRegressor(max_rules=20, random_state=42).fit(Xtr, ytr_n).predict(Xte),
        'HSTree_large': lambda: HSTreeRegressorCV(max_leaf_nodes=20, random_state=42).fit(Xtr, ytr_n).predict(Xte),
        'DT':           lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=2, random_state=42).fit(Xtr, ytr_n).predict(Xte),
        'MLP':          lambda: MLPRegressor(random_state=42).fit(Xtr, ytr_n).predict(Xte),
    }

    for mn in need:
        t0 = time.time()
        try:
            if mn == 'TabPFN':
                # separate process, capped at 2500 training rows and 500 features
                np.save(f'{J}/_X.npy', Xtr); np.save(f'{J}/_y.npy', ytr_n)
                np.save(f'{J}/_Xt.npy', Xte); np.save(f'{J}/_yt.npy', yte_n)
                r = subprocess.run(['uv', 'run', f'{J}/tabpfn_one.py'], capture_output=True,
                                   text=True, timeout=5400, cwd=ROOT)
                rmse = next((float(l.split()[1]) for l in r.stdout.splitlines()
                             if l.startswith('RMSE')), float('nan'))
                if not np.isfinite(rmse):
                    print(f'{nm:<30} TabPFN FAILED {r.stderr[-160:]}', flush=True); continue
            else:
                rmse = float(np.sqrt(mean_squared_error(yte_n, fits[mn]())))
        except Exception as e:
            print(f'{nm:<30} {mn:<13} FAILED {type(e).__name__}: {e}', flush=True); continue
        with open(RES, 'a', newline='') as f:
            csv.writer(f).writerow([nm, mn, f'{rmse:.6f}', f'{time.time()-t0:.0f}'])
        print(f'{nm:<30} {mn:<13} {rmse:.4f} ({time.time()-t0:.0f}s)', flush=True)

print('ALL DONE', flush=True)
