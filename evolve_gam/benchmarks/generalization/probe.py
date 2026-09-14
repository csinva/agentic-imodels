import os
"""Probe variants on one (dataset, seed): uv run probe.py DATASET SEED 'kw1' 'kw2' ..."""
import sys, time, numpy as np, warnings; warnings.filterwarnings("ignore")
J=os.environ.get('GEN_DIR', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,J)
from gpm import BinGP
ds, seed = sys.argv[1], int(sys.argv[2])
z=np.load(f"{J}/data/{ds.replace('/','__')}.npz")
if seed==0: Xtr,Xte,ytr,yte=z['Xtr'],z['Xte'],z['ytr'],z['yte']
else:
    X=np.vstack([z["Xtr"],z["Xte"]]); y=np.concatenate([z["ytr_raw"],z["yte_raw"]])
    idx=np.random.RandomState(seed).permutation(len(y)); ntr=len(z["ytr"]); tr,te=idx[:ntr],idx[ntr:]
    ym,ys=y[tr].mean(),y[tr].std(); Xtr,Xte,ytr,yte=X[tr],X[te],(y[tr]-ym)/ys,(y[te]-ym)/ys
for kws in sys.argv[3:]:
    kw=eval(f"dict({kws})"); t=time.time()
    m=BinGP(**kw).fit(Xtr,ytr); r=np.sqrt(np.mean((m.predict(Xte)-yte)**2))
    print(f"  {kws:<50} rmse {r:.4f}  ({time.time()-t:.0f}s)", flush=True)
