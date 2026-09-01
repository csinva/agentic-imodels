import csv, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
J = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, J)
from sklearn.model_selection import train_test_split
C = os.path.expanduser("~/.cache/imodels-evolve/ctr23")
RES = f"{J}/ctr23_results.csv"
rows = json.load(open(f"{J}/ctr23.json"))
done = {(r["dataset"], r["model"]) for r in csv.DictReader(open(RES))}
for nm, tid, did, tgt, n, d, dup in [r for r in rows if not r[6]]:
    if (nm, "TabPFN") in done:
        continue
    df = pd.read_parquet(f"{C}/{did}.parquet")
    y = pd.to_numeric(df[tgt], errors="coerce").values.astype(float)
    Xdf = df.drop(columns=[tgt]); cols = []
    for c in Xdf.columns:
        sr = Xdf[c]
        if sr.dtype.name in ("category","object","string","bool"):
            cols.append(sr.astype("category").cat.codes.values.astype(float))
        else:
            v = pd.to_numeric(sr, errors="coerce").values.astype(float)
            md = np.nanmedian(v); cols.append(np.where(np.isfinite(v), v, md if np.isfinite(md) else 0.0))
    X = np.column_stack(cols); ok = np.isfinite(y)
    Xtr, Xte, ytr, yte = train_test_split(X[ok], y[ok], test_size=0.2, random_state=42)
    ym, ys = ytr.mean(), ytr.std()
    np.save(f"{J}/_X.npy", Xtr); np.save(f"{J}/_y.npy", (ytr-ym)/ys)
    np.save(f"{J}/_Xt.npy", Xte); np.save(f"{J}/_yt.npy", (yte-ym)/ys)
    t0 = time.time()
    try:
        r = subprocess.run(["uv","run",f"{J}/tabpfn_one.py"], capture_output=True, text=True,
                           timeout=5400, cwd=_ROOT)
        val = next((float(l.split()[1]) for l in r.stdout.splitlines() if l.startswith("RMSE")), float("nan"))
        if not np.isfinite(val):
            print(f"{nm} TabPFN failed: {r.stderr[-150:]}", flush=True)
    except Exception as e:
        val = float("nan"); print(f"{nm} TabPFN {type(e).__name__}", flush=True)
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([nm, "TabPFN", f"{val:.6f}", f"{time.time()-t0:.0f}"])
    print(f"{nm:<32} TabPFN {val:.4f} ({time.time()-t0:.0f}s)", flush=True)
print("ALL DONE", flush=True)
