"""Sibling-free official-split mean rank for every refit historical version, against the fixed baseline pool."""
import csv, json, os
from collections import defaultdict
import numpy as np
J=os.path.dirname(os.path.abspath(__file__)); E='/Users/chandansingh/Downloads/agentic-imodels/evolve'
SIB=("SegGAM","GA2M","SB_","GCV","Simple","AddGP","BinGP","TVG","PFN","SPGP","EMGP")
pool=defaultdict(dict)
for r in csv.DictReader(open(f'{E}/results/performance_results.csv')):
    if r['rmse'] and not r['model'].startswith(SIB): pool[r['dataset']][r['model']]=float(r['rmse'])
res=defaultdict(dict)
for r in csv.DictReader(open(f'{J}/results/hist_ranks.csv')):
    v=float(r['rmse']);
    if np.isfinite(v): res[r['version']][r['dataset']]=v
def rank(m):
    rk=[]
    for ds,mv in pool.items():
        if ds not in m: continue
        vals=dict(mv); vals['_']=m[ds]; rk.append(sorted(vals,key=vals.get).index('_')+1)
    return float(np.mean(rk)), len(rk)
out={}
for v in sorted(res, key=lambda s:int(s[1:])):
    r,n=rank(res[v])
    if n < 65: print(f"  {v:<4} incomplete ({n}/65), skipped"); continue
    out[v]=round(r,2); print(f"  {v:<4} mean rank {r:.2f} over {n} datasets")
# GP-era points measured in this same harness (seed 0, same pool)
for name,v in [('v47','v47'),('v48','v48'),('v49','v48')]:
    d={}
    for r in csv.DictReader(open(f'{J}/results/{v}.csv')):
        if r['seed']=='0' and np.isfinite(float(r['rmse'])): d[r['dataset']]=float(r['rmse'])
    r,n=rank(d); out[name]=round(r,2); print(f"  {name:<4} mean rank {r:.2f} (this harness)")
json.dump(out, open(f'{J}/hist_ranks_summary.json','w')); print("saved hist_ranks_summary.json")
