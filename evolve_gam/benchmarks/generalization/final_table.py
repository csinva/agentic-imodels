import os
"""Final summary: v47 vs v48 vs EBM at three scales (n=200 subsamples, the official
n<=1000 suite, and the 7 OpenML suite datasets at 4000 rows)."""
import csv, os
from collections import defaultdict
import numpy as np
J = os.path.dirname(os.path.abspath(__file__))
def load(path):
    d = defaultdict(dict)
    if not os.path.exists(path): return d
    for r in csv.DictReader(open(path)):
        v = float(r["rmse"])
        if np.isfinite(v): d[int(r["seed"])][r["dataset"]] = v
    return d
def summarise(model, ebm):
    out = []
    for s in sorted(model):
        c = [d for d in model[s] if d in ebm.get(s, {})]
        if not c: continue
        r = np.array([model[s][d] / ebm[s][d] for d in c])
        out.append((s, len(c), int((r < 1).sum()), float(np.exp(np.log(r).mean())), float(np.percentile(r, 90))))
    return out
scales = [("n=200 subsample", "results", "_n200"), ("official suite (n<=1000)", "results", ""), ("n=4000 (7 OpenML)", "results_large", "")]
print(f"{'scale':<26}{'model':<6}{'seed':>5}{'n':>4}{'wins':>8}{'GM vs EBM':>11}{'p90':>7}")
for label, res, tag in scales:
    ebm = load(f"{J}/{res}/EBM{tag}.csv")
    for name in ("v47", "v48"):
        for s, n, w, gm, p90 in summarise(load(f"{J}/{res}/{name}{tag}.csv"), ebm):
            print(f"{label:<26}{name:<6}{s:>5}{n:>4}{w:>4}/{n:<3}{gm:>11.3f}{p90:>7.3f}")
# per-dataset large-scale detail
ebm = load(f"{J}/results_large/EBM.csv"); v47 = load(f"{J}/results_large/v47.csv"); v48 = load(f"{J}/results_large/v48.csv")
if 0 in v48:
    print("\nn=4000 detail (seed 0):  dataset  v47  v48  EBM")
    for d in sorted(v48[0]):
        print(f"  {d:<12} {v47[0].get(d, float('nan')):.4f}  {v48[0][d]:.4f}  {ebm[0][d]:.4f}")
