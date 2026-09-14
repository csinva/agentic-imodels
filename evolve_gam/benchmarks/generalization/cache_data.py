"""Cache the imodels-65 suite exactly as the official harness presents it.

Stores, per dataset, the post-subsample train/test arrays with y standardised
by train statistics (what the harness hands to a model), plus the raw pooled
arrays so alternative splits of the same data can be drawn later.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'evolve', 'src'))
from performance_eval import get_all_datasets, subsample_dataset, MIN_SAMPLES, MIN_FEATURES

OUT = '/Users/chandansingh/.claude/jobs/6a490400/tmp/gen/data'
os.makedirs(OUT, exist_ok=True)
names = []
for name, Xtr, Xte, ytr, yte in get_all_datasets():
    Xtr, Xte, ytr, yte = subsample_dataset(Xtr, Xte, ytr, yte)
    if len(Xtr) < MIN_SAMPLES or Xtr.shape[1] < MIN_FEATURES:
        continue
    ym, ys = float(ytr.mean()), float(ytr.std())
    ytr_n, yte_n = ((ytr - ym) / ys, (yte - ym) / ys) if ys > 0 else (ytr, yte)
    key = name.replace('/', '__')
    np.savez(f"{OUT}/{key}.npz", Xtr=np.asarray(Xtr, float), Xte=np.asarray(Xte, float),
             ytr=ytr_n, yte=yte_n, ytr_raw=ytr, yte_raw=yte)
    names.append(name)
    print(f"{name:<34} n={len(Xtr):<5} d={Xtr.shape[1]:<3} test={len(Xte)}", flush=True)
open(f"{OUT}/names.txt", "w").write("\n".join(names))
print(len(names), "datasets cached")
