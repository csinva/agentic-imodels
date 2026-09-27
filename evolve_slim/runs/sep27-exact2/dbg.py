import sys, itertools, numpy as np
sys.path.insert(0,'src')
import importlib.util
spec = importlib.util.spec_from_file_location('slim','slim.py'); slim = importlib.util.module_from_spec(spec); spec.loader.exec_module(slim)
from suite import load_problem
X,y,*_ = load_problem('mammo')
ub = 0.467206716*len(y)
cols = np.flatnonzero(np.ptp(X,0)>0)
rows, inv = np.unique(X[:,cols],axis=0,return_inverse=True); inv=inv.ravel()
pos = np.bincount(inv,weights=y); neg=np.bincount(inv,weights=1-y)
for S in itertools.combinations(range(len(cols)),4):
    cells, ci = np.unique(rows[:,S],axis=0,return_inverse=True); ci=ci.ravel()
    p = np.bincount(ci,weights=pos); n=np.bincount(ci,weights=neg)
    H = sum(slim._h2(a,b) for a,b in zip(p,n))
    if H >= ub: continue
    U = np.hstack([cells, np.ones((len(cells),1))])
    th = np.zeros(5)
    lb,F = slim.lr_bound(U,len(cells),5,p,n,th,60,-np.inf)
    if lb >= ub: continue
    # brute force all integer vectors
    best=np.inf; worst_lb=np.inf
    for w in itertools.product(range(-5,6),repeat=4):
        w=np.array(w,float)
        s=cells@w
        if np.all(s==s[0]): continue
        U2=np.column_stack([s,np.ones(len(s))]); th2=np.zeros(2)
        l2,F2=slim.lr_bound(U2,len(s),2,p,n,th2,60,-np.inf)
        if F2<best: best=F2
        if l2<ub and F2>=ub: print('leaf lb<ub but F>=ub',S,w,l2/len(y),F2/len(y))
        worst_lb=min(worst_lb,l2)
    print(S,'H',H/len(y),'LR',lb/len(y),F/len(y),'best int',best/len(y),'minlb',worst_lb/len(y))
