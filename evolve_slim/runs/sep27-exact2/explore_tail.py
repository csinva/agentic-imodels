import sys, itertools, numpy as np, math, time
sys.path.insert(0,'src')
from suite import load_problem
from sklearn.linear_model import LogisticRegression
name, k, ub = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
X,y,*_ = load_problem(name)
cols = np.flatnonzero(np.ptp(X,0)>0); X=X[:,cols]; d=X.shape[1]
# order by single feature entropy
def lrloss(S):
    if len(S)==0: p=y.mean(); return -(p*np.log(p)+(1-p)*np.log(1-p))
    m=LogisticRegression(C=1e6,max_iter=2000).fit(X[:,S],y)
    pr=np.clip(m.predict_proba(X[:,S])[:,1],1e-15,1-1e-15)
    return -np.mean(y*np.log(pr)+(1-y)*np.log(1-pr))
single=[lrloss([j]) for j in range(d)]
order=list(np.argsort(single))
tot=math.comb(d,k); pruned=0; nlr=0
t0=time.time()
def rec(T, start):
    global pruned, nlr
    m = k-len(T)
    C = order[start:]
    nleaves = math.comb(len(C), m)
    if nleaves==0: return
    if nleaves >= 200 and len(T)>0:
        nlr+=1
        if lrloss(list(T)+C) >= ub:
            pruned += nleaves; return
    if m==0: return
    if m==1: return
    for p in range(start, d):
        rec(T+[order[p]], p+1)
    if time.time()-t0>120: raise SystemExit(f'timeout pruned {pruned/tot:.3f} nlr {nlr}')
rec([],0)
print(name,k,'total',tot,'pruned frac',pruned/tot,'LR calls',nlr)
