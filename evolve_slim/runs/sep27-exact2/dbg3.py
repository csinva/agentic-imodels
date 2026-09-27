import sys, itertools, numpy as np
sys.path.insert(0,'src')
import importlib.util
spec = importlib.util.spec_from_file_location('slim','slim.py'); slim = importlib.util.module_from_spec(spec); spec.loader.exec_module(slim)
from suite import load_problem
name, k, ubl = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X,y,*_ = load_problem(name)
N=len(y); ub = ubl*N; thr=2e-8*N
cols = np.flatnonzero(np.ptp(X,0)>0)
rows, inv = np.unique(X[:,cols],axis=0,return_inverse=True); inv=inv.ravel()
pos = np.bincount(inv,weights=y); neg=np.bincount(inv,weights=1-y)
d=len(cols)
# quick H filter using per-support unique
cnt=0
for S in itertools.combinations(range(d),k):
    key = rows[:,S] @ (np.arange(k)*7.31+1.0)**2 if False else None
    cells, ci = np.unique(rows[:,S],axis=0,return_inverse=True); ci=ci.ravel()
    p = np.bincount(ci,weights=pos); n=np.bincount(ci,weights=neg)
    H = sum(slim._h2(a,b) for a,b in zip(p,n))
    if H >= ub-thr: continue
    C=len(cells)
    U = np.hstack([cells, np.ones((C,1))]); th=np.zeros(k+1)
    lb,F = slim.robust_lr(U,C,k+1,p,n,th,60,-np.inf); lb=max(lb,H)
    if lb >= ub-thr: continue
    perm = np.argsort(-np.abs(th[:k])*np.ptp(cells,0))
    wS=np.zeros(k,np.int64)
    lbs, ub2, imp, comp, nodes = slim.int_bb(np.ascontiguousarray(cells.astype(float)),C,k,p,n,perm,ub,thr,lb,H,wS,1e18)
    if lbs < ub-thr:
        print(S, 'H',H/N,'LR',lb/N,'bb lb',lbs/N,'ub2',ub2/N, imp, nodes, flush=True)
        cnt+=1
        if cnt>5: break
