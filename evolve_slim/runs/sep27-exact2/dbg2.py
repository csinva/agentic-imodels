import sys, itertools, numpy as np
sys.path.insert(0,'src')
import importlib.util
spec = importlib.util.spec_from_file_location('slim','slim.py'); slim = importlib.util.module_from_spec(spec); spec.loader.exec_module(slim)
from suite import load_problem
X,y,*_ = load_problem('mammo')
N=len(y); ub = 0.467206716*N
cols = np.flatnonzero(np.ptp(X,0)>0)
rows, inv = np.unique(X[:,cols],axis=0,return_inverse=True); inv=inv.ravel()
pos = np.bincount(inv,weights=y); neg=np.bincount(inv,weights=1-y)
S=(3,4,10,13)
cells, ci = np.unique(rows[:,S],axis=0,return_inverse=True); ci=ci.ravel()
p = np.bincount(ci,weights=pos); n=np.bincount(ci,weights=neg)
print(np.column_stack([cells,p,n]))
U = np.hstack([cells, np.ones((len(cells),1))])
th=np.zeros(5)
print(slim.lr_bound(U,len(cells),5,p,n,th,60,-np.inf), th)
th=np.zeros(5)
print([v/N for v in slim.robust_lr(U,len(cells),5,p,n,th,60,-np.inf)], th)
