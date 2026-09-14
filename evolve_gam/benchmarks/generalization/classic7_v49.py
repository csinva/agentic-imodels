import os
"""AddGP_v49 on the classic-7 at full size (same loader/split as the blog-post baselines)."""
import csv, os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
E=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'evolve'); sys.path.insert(0,E); sys.path.insert(0,f'{E}/src')
from performance_eval import _load_openml_dataset
import interpretable_regressor as ir
OUT='/Users/chandansingh/.claude/jobs/6a490400/tmp/gen/classic7_v49.csv'
done=set()
if os.path.exists(OUT): done={r['dataset'] for r in csv.DictReader(open(OUT))}
else: open(OUT,'w').write('dataset,model,rmse,seconds\n')
for name in ["abalone","cpu_act","kin8nm","pol","elevators","california","house_16H"]:
    if name in done: continue
    Xtr,Xte,ytr,yte=map(np.asarray,_load_openml_dataset(name)); ym,ys=ytr.mean(),ytr.std()
    t=time.time(); m=ir.model_defs[0][1].__class__().fit(Xtr,ytr)
    r=float(np.sqrt(np.mean(((m.predict(Xte)-ym)/ys-(yte-ym)/ys)**2)))
    open(OUT,'a').write(f"{name},AddGP_v49,{r:.6f},{time.time()-t:.0f}\n"); print(f"  {name:<12} {r:.4f} ({time.time()-t:.0f}s) n={len(ytr)}", flush=True)
print("CLASSIC7 DONE", flush=True)
