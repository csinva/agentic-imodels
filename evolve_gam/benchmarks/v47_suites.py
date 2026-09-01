import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
import csv, json, os, sys, time
import numpy as np
import pandas as pd
J = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_HERE, "..", "model"))
sys.path.insert(0, _os.path.join(_ROOT, "evolve", "src"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from addgp import BinGP
from performance_eval import _load_openml_dataset
RES=f"{J}/v47_suites.csv"
done=set()
if os.path.exists(RES): done={(r["suite"],r["dataset"]) for r in csv.DictReader(open(RES))}
else:
    with open(RES,"w",newline="") as f: csv.writer(f).writerow(["suite","dataset","rmse","seconds"])
def rec(su,ds,v,t):
    with open(RES,"a",newline="") as f: csv.writer(f).writerow([su,ds,f"{v:.6f}",f"{t:.0f}"])
    print(f"{su:<9} {ds:<12} {v:.4f} ({t:.0f}s)",flush=True)
for name in ["abalone","cpu_act","kin8nm","pol","elevators","california","house_16H"]:
    if ("classic7",name) in done: continue
    Xtr,Xte,ytr,yte=_load_openml_dataset(name); ym,ys=ytr.mean(),ytr.std(); t0=time.time()
    m=BinGP().fit(Xtr,ytr)
    rec("classic7",name,float(np.sqrt(mean_squared_error((yte-ym)/ys,(m.predict(Xte)-ym)/ys))),time.time()-t0)
TA=[("airfoil",46904,"scaled-sound-pressure"),("Fiat",46907,"price"),("concrete",46917,"ConcreteCompressiveStrength"),
    ("diamonds",46923,"price"),("Food",46928,"Time_taken(min)"),("healthcare",46931,"charges"),
    ("houses",46934,"LnMedianHouseValue"),("miami",46942,"SALE_PRC"),("protein",46949,"ResidualSize"),
    ("QSAR",46953,"MEDIAN_PXC50"),("fish",46954,"LC50"),("supercon",46961,"critical_temp"),("wine",46964,"median_wine_quality")]
for nm,did,tgt in TA:
    if ("tabarena",nm) in done: continue
    df=pd.read_parquet(os.path.expanduser(f"~/.cache/imodels-evolve/tabarena/{did}.parquet"))
    y=pd.to_numeric(df[tgt],errors="coerce").values.astype(float); Xdf=df.drop(columns=[tgt]); cols=[]
    for c in Xdf.columns:
        sr=Xdf[c]
        if sr.dtype.name in ("category","object","string","bool"): cols.append(sr.astype("category").cat.codes.values.astype(float))
        else:
            v=pd.to_numeric(sr,errors="coerce").values.astype(float); md=np.nanmedian(v)
            cols.append(np.where(np.isfinite(v),v,md if np.isfinite(md) else 0.0))
    X=np.column_stack(cols); ok=np.isfinite(y)
    Xtr,Xte,ytr,yte=train_test_split(X[ok],y[ok],test_size=0.2,random_state=42)
    ym,ys=ytr.mean(),ytr.std(); t0=time.time()
    m=BinGP().fit(Xtr,ytr)
    rec("tabarena",nm,float(np.sqrt(mean_squared_error((yte-ym)/ys,(m.predict(Xte)-ym)/ys))),time.time()-t0)
print("ALL DONE",flush=True)
