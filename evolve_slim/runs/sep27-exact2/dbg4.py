import sys, itertools, numpy as np
sys.path.insert(0,'src')
import importlib.util
spec = importlib.util.spec_from_file_location('slim','slim.py'); slim = importlib.util.module_from_spec(spec); spec.loader.exec_module(slim)
from suite import load_problem
X,y,*_ = load_problem(sys.argv[1])
S=tuple(int(v) for v in sys.argv[2].split(','))
N=len(y)
cols = np.flatnonzero(np.ptp(X,0)>0)
rows, inv = np.unique(X[:,cols],axis=0,return_inverse=True); inv=inv.ravel()
pos = np.bincount(inv,weights=y); neg=np.bincount(inv,weights=1-y)
cells, ci = np.unique(rows[:,S],axis=0,return_inverse=True); ci=ci.ravel()
p = np.bincount(ci,weights=pos); n=np.bincount(ci,weights=neg)
print(np.column_stack([cells,p,n]))
k=len(S); C=len(cells)
U = np.hstack([cells, np.ones((C,1))]); th=np.zeros(k+1)
print(slim.lr_bound(U,C,k+1,p,n,th,60,-np.inf), th)
th=np.zeros(k+1)
print(slim.robust_lr(U,C,k+1,p,n,th,60,-np.inf), th)
