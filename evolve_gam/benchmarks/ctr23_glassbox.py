"""The five models CTR23 was missing, on the non-overlapping 21.

ctr23_eval.py covers AddGP, EBM, RF, GBM and Ridge; ctr23_tabpfn.py covers
TabPFN. The report's table also carries FIGS, RuleFit, hierarchical shrinkage,
a decision tree and a neural net, which had no per-dataset CTR23 results. Same
protocol as ctr23_eval.py -- one 80/20 split at seed 42, target standardized on
the training half -- and the same configurations those five use in
evolve/run_baselines.py, so the column is comparable within itself.

Resumable per (dataset, model). Appends to ctr23_results.csv.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from imodels import FIGSRegressor, RuleFitRegressor, HSTreeRegressorCV

J = os.path.dirname(os.path.abspath(__file__))
C = os.path.expanduser("~/.cache/imodels-evolve/ctr23")
RES = f"{J}/ctr23_results.csv"

rows = json.load(open(f"{J}/ctr23.json"))
todo = [r for r in rows if not r[6]]          # the 21 with no overlap


def prep(did, target):
    df = pd.read_parquet(f"{C}/{did}.parquet")
    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)
    Xdf = df.drop(columns=[target])
    cols = []
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

MODELS = {
    "FIGS_large":   lambda: FIGSRegressor(max_rules=20, random_state=42),
    "RuleFit":      lambda: RuleFitRegressor(max_rules=20, random_state=42),
    "HSTree_large": lambda: HSTreeRegressorCV(max_leaf_nodes=20, random_state=42),
    "DT":           lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=2, random_state=42),
    "MLP":          lambda: MLPRegressor(random_state=42),
}

for nm, tid, did, tgt, n, d, dup in sorted(todo, key=lambda r: int(r[4])):
    try:
        Xtr, Xte, ytr, yte = prep(did, tgt)
    except Exception as e:
        print(f"{nm}: LOAD FAILED {type(e).__name__}", flush=True)
        continue
    ym, ys = ytr.mean(), ytr.std()
    if not np.isfinite(ys) or ys == 0:
        print(f"{nm}: degenerate target, skipped", flush=True)
        continue
    yte_n = (yte - ym) / ys
    ytr_n = (ytr - ym) / ys
    for mn, make in MODELS.items():
        if (nm, mn) in done:
            continue
        t0 = time.time()
        try:
            pred = make().fit(Xtr, ytr_n).predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yte_n, pred)))
        except Exception as e:
            print(f"{nm:<28} {mn:<13} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([nm, mn, f"{rmse:.6f}", f"{time.time() - t0:.0f}"])
        print(f"{nm:<28} {mn:<13} {rmse:.4f} ({time.time() - t0:.0f}s)", flush=True)

print("ALL DONE", flush=True)
