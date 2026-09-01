"""CTR23 (non-TabArena subset): AddGP vs EBM/RF/GBM/Ridge. Resumable per (dataset, model)."""
import csv, json, os, sys, time
import numpy as np
import pandas as pd
J = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, J)
sys.path.insert(0, _os.path.join(_HERE, "..", "model"))
sys.path.insert(0, _os.path.join(_ROOT, "evolve", "src"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from interpret.glassbox import ExplainableBoostingRegressor
from addgp import BinGP

C = os.path.expanduser("~/.cache/imodels-evolve/ctr23")
RES = f"{J}/ctr23_results.csv"
rows = json.load(open(f"{J}/ctr23.json"))
todo = [r for r in rows if not r[6]]

def prep(did, target):
    df = pd.read_parquet(f"{C}/{did}.parquet")
    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)
    Xdf = df.drop(columns=[target]); cols = []
    for c in Xdf.columns:
        sr = Xdf[c]
        if sr.dtype.name in ("category", "object", "string", "bool"):
            cols.append(sr.astype("category").cat.codes.values.astype(float))
        else:
            v = pd.to_numeric(sr, errors="coerce").values.astype(float)
            md = np.nanmedian(v)
            cols.append(np.where(np.isfinite(v), v, md if np.isfinite(md) else 0.0))
    X = np.column_stack(cols) if cols else np.zeros((len(y), 1))
    ok = np.isfinite(y)
    return train_test_split(X[ok], y[ok], test_size=0.2, random_state=42)

done = set()
if os.path.exists(RES):
    done = {(r["dataset"], r["model"]) for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "model", "rmse", "seconds"])

for nm, tid, did, tgt, n, d, dup in todo:
    try:
        Xtr, Xte, ytr, yte = prep(did, tgt)
    except Exception as e:
        print(f"{nm}: LOAD FAILED {type(e).__name__}", flush=True); continue
    ym, ys = ytr.mean(), ytr.std()
    if not np.isfinite(ys) or ys == 0:
        print(f"{nm}: degenerate target, skipped", flush=True); continue
    yte_n = (yte - ym) / ys
    models = {
        "AddGP":  lambda: (BinGP().fit(Xtr, ytr).predict(Xte) - ym) / ys,
        "EBM":    lambda: (ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000).fit(Xtr, (ytr-ym)/ys).predict(Xte)),
        "RF":     lambda: (RandomForestRegressor(random_state=42, n_jobs=4).fit(Xtr, (ytr-ym)/ys).predict(Xte)),
        "GBM":    lambda: (GradientBoostingRegressor(random_state=42).fit(Xtr, (ytr-ym)/ys).predict(Xte)),
        "Ridge":  lambda: (RidgeCV().fit(Xtr, (ytr-ym)/ys).predict(Xte)),
    }
    for mn, fn in models.items():
        if (nm, mn) in done:
            continue
        t0 = time.time()
        try:
            r = float(np.sqrt(mean_squared_error(yte_n, fn())))
        except Exception as e:
            print(f"{nm} {mn} FAILED: {type(e).__name__} {str(e)[:80]}", flush=True)
            r = float("nan")
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([nm, mn, f"{r:.6f}", f"{time.time()-t0:.0f}"])
        print(f"{nm:<32} {mn:<7} {r:.4f} ({time.time()-t0:.0f}s)", flush=True)
print("ALL DONE", flush=True)
