import os
"""Held-out evaluation after de-duplication: TabArena-13 minus `houses` (California
housing, in the dev suite) and CTR23 minus its imodels-65 overlaps and TabArena
overlaps. EBM, v48 and v47 are fit on identical preprocessing and the same 80/20
split (rs=42); RMSE on the train-standardised target. Resumable."""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'evolve')
sys.path.insert(0, E); sys.path.insert(0, f'{E}/src'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model'))
CACHE = os.path.expanduser("~/.cache/imodels-evolve")
J = os.path.dirname(os.environ.get('GEN_DIR', os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{J}/gen")
RES = f"{J}/gen/results_heldout.csv"

TAB = [("airfoil",46904,"scaled-sound-pressure"),("Fiat",46907,"price"),("concrete",46917,"ConcreteCompressiveStrength"),
       ("diamonds",46923,"price"),("Food",46928,"Time_taken(min)"),("healthcare",46931,"charges"),
       ("miami",46942,"SALE_PRC"),("protein",46949,"ResidualSize"),("QSAR",46953,"MEDIAN_PXC50"),
       ("fish",46954,"LC50"),("supercon",46961,"critical_temp"),("wine",46964,"median_wine_quality")]   # houses removed
TAB_DUP_CTR23 = {"airfoil_self_noise","concrete_compressive_strength","physiochemical_protein","superconductivity",
                 "diamonds","miami_housing","QSAR_fish_toxicity"}
ctr = [r for r in json.load(open('ctr23.json'))
       if not r[6] and r[0] not in TAB_DUP_CTR23]
tasks = [("tabarena", n, f"{CACHE}/tabarena/{d}.parquet", t) for n, d, t in TAB] + \
        [("ctr23", r[0], f"{CACHE}/ctr23/{r[2]}.parquet", r[3]) for r in ctr]

def frame(path, target):
    df = pd.read_parquet(path)
    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)
    Xd = df.drop(columns=[target]); cols = []
    for c in Xd.columns:
        sr = Xd[c]
        if sr.dtype.name in ("category", "object", "string", "bool"):
            cols.append(sr.astype("category").cat.codes.values.astype(float))
        else:
            v = pd.to_numeric(sr, errors="coerce").values.astype(float); md = np.nanmedian(v)
            cols.append(np.where(np.isfinite(v), v, md if np.isfinite(md) else 0.0))
    X = np.column_stack(cols); ok = np.isfinite(y)
    Xtr, Xte, ytr, yte = train_test_split(X[ok], y[ok], test_size=0.2, random_state=42)
    ym, ys = ytr.mean(), ytr.std(); return Xtr, Xte, ytr, yte, ym, ys

def make(name):
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000, n_jobs=2)
    if name == "v48":
        import interpretable_regressor as ir; return ir.BinGP()
    if name == "v47":
        from addgp import BinGP as B47; return B47()
    if name == "v49h":
        from gpm import BinGP as BW; return BW(schedule='v49h', tau=1.0, learn_scales=True, scale_prior=0.5)
    if name == "v49":
        from gpm import BinGP as BW; return BW(schedule='v49', tau=1.0, learn_scales=True, scale_prior=0.5)
    if name == "v49n":
        from gpm import BinGP as BW; return BW(schedule='v48', tau=1.0, learn_scales=True, scale_prior=0.5, nugget=True)

def run(suite, name, path, target, model):
    Xtr, Xte, ytr, yte, ym, ys = frame(path, target)
    t0 = time.time()
    try:
        m = make(model).fit(Xtr, ytr)                       # raw y: the log rule may fire
        r = float(np.sqrt(np.mean(((m.predict(Xte) - ym) / ys - (yte - ym) / ys) ** 2)))
    except Exception as e:
        r = float("nan"); print(f"  {name} {model} FAILED {type(e).__name__}: {e}", flush=True)
    return suite, name, model, r, time.time() - t0

done = set()
if os.path.exists(RES):
    done = {(r["dataset"], r["model"]) for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f: csv.writer(f).writerow(["suite","dataset","model","rmse","seconds"])
models = sys.argv[1].split(",") if len(sys.argv) > 1 else ("EBM", "v48", "v47")
skip = set(os.environ.get("SKIP", "").split(",")) - {""}
tasks = [t for t in tasks if t[1] not in skip]
todo = [(s, n, p, t, m) for m in models for s, n, p, t in tasks if (n, m) not in done]
print(f"{len(tasks)} held-out datasets; {len(todo)} fits to run", flush=True)
from joblib import Parallel, delayed
with Parallel(n_jobs=int(os.environ.get('WORKERS', '3')), return_as="generator") as par:
    for suite, name, model, r, sec in par(delayed(run)(*a) for a in todo):
        with open(RES, "a", newline="") as f: csv.writer(f).writerow([suite, name, model, f"{r:.6f}", f"{sec:.0f}"])
        print(f"  {suite:<9} {name:<32} {model:<4} {r:.4f} ({sec:.0f}s)", flush=True)
print("HELDOUT DONE", flush=True)
