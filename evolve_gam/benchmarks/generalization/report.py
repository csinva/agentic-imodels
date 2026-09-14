import os
"""Compare variants: uv run report.py NAME [NAME2 ...]  (EBM.csv must exist)

Prints, per seed: official sibling-free mean rank (seed 0 only), wins vs EBM,
geometric-mean RMSE ratio vs EBM, and the 90th percentile of that ratio
(the blow-up tail). Then paired per-dataset deltas between the first two names.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
J = os.path.dirname(os.path.abspath(__file__)); RES = f"{J}/results"
OFFICIAL = '/Users/chandansingh/Downloads/agentic-imodels/evolve/results/performance_results.csv'
SIB = ("SegGAM","GA2M","SB_","GCV","Simple","AddGP","BinGP","TVG","PFN","SPGP","EMGP")

def load(n):
    d = defaultdict(dict)
    for r in csv.DictReader(open(f"{RES}/{n}.csv")):
        v = float(r["rmse"])
        if np.isfinite(v): d[int(r["seed"])][r["dataset"]] = v
    return d

pool = defaultdict(dict)
for r in csv.DictReader(open(OFFICIAL)):
    if r["rmse"] and not r["model"].startswith(SIB):
        pool[r["dataset"]][r["model"]] = float(r["rmse"])

def official_rank(res0):
    ranks = defaultdict(list)
    for ds, mv in pool.items():
        if ds not in res0: continue
        mv = dict(mv); mv["_this_"] = res0[ds]
        for i, m in enumerate(sorted(mv, key=lambda m: mv[m])): ranks[m].append(i + 1)
    return {m: np.mean(v) for m, v in ranks.items()}

names = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {a[2:].split("=")[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
RES = RES + opt.get("suffix", "")
ebm = load(opt.get("ebm", "EBM"))
res = {n: load(n) for n in names}
print(f"{'variant':<22}{'seed':>5}{'n':>4}{'rank':>7}{'EBMrank':>9}{'wins':>7}{'GMratio':>9}{'p90':>7}{'medNRMSE':>10}")
for n in names:
    for s in sorted(res[n]):
        common = [ds for ds in res[n][s] if ds in ebm.get(s, {})]
        if not common: continue
        ratio = np.array([res[n][s][ds] / ebm[s][ds] for ds in common])
        wins = int((ratio < 1).sum())
        rk = official_rank(res[n][0]) if s == 0 else {}
        print(f"{n:<22}{s:>5}{len(common):>4}"
              f"{rk.get('_this_', float('nan')):>7.2f}{rk.get('EBM', float('nan')):>9.2f}"
              f"{wins:>4}/{len(common):<3}{np.exp(np.mean(np.log(ratio))):>8.3f}"
              f"{np.percentile(ratio, 90):>7.3f}{np.median([res[n][s][ds] for ds in common]):>10.4f}")
if len(names) >= 2:
    a, b = names[0], names[1]
    print(f"\npaired {b} vs {a} (ratio<1 means {b} better):")
    allr = []
    for s in sorted(set(res[a]) & set(res[b])):
        common = [ds for ds in res[a][s] if ds in res[b][s]]
        r = np.array([res[b][s][ds] / res[a][s][ds] for ds in common]); allr += list(r)
        print(f"  seed{s}: better on {int((r<1).sum())}/{len(r)}, GM {np.exp(np.mean(np.log(r))):.4f}, "
              f"worst +{100*(r.max()-1):.1f}% ({common[int(r.argmax())]}), best {100*(r.min()-1):.1f}% ({common[int(r.argmin())]})")
    # collapse exact-duplicate datasets (the suite repeats several) before testing
    distinct = set(open(f"{J}/distinct.txt").read().split("\n")) if os.path.exists(f"{J}/distinct.txt") else None
    if distinct:
        allr = []
        for s in sorted(set(res[a]) & set(res[b])):
            allr += [res[b][s][ds] / res[a][s][ds] for ds in res[a][s] if ds in res[b][s] and ds in distinct]
    allr = np.array(allr)
    from scipy.stats import binomtest
    k, m = int((allr < 1).sum()), int((allr != 1).sum())
    pv = binomtest(k, m, 0.5).pvalue if m else float('nan')
    p90a = np.percentile([res[a][s][ds] / ebm[s][ds] for s in res[a] for ds in res[a][s] if ds in ebm.get(s, {})], 90)
    p90b = np.percentile([res[b][s][ds] / ebm[s][ds] for s in res[b] for ds in res[b][s] if ds in ebm.get(s, {})], 90)
    print(f"  all seeds (distinct datasets): better on {k}/{m}, GM {np.exp(np.mean(np.log(allr))):.4f}, sign-test p={pv:.3g}; "
          f"p90 vs EBM {p90a:.3f} -> {p90b:.3f}")
    ok = (np.exp(np.mean(np.log(allr))) < 1) and (pv < 0.05) and (p90b <= p90a * 1.01)
    print(f"  VERDICT: {'ACCEPT candidate' if ok else 'reject'} (pre-registered rule: GM<1, p<0.05, tail not worse)")
