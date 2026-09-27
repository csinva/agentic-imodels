import sys, os; sys.path.insert(0,'src')
import numpy as np, importlib.util, csv, time
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m','slim.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
name=sys.argv[1]; k=int(sys.argv[2]); budget=float(sys.argv[3])
Xtr,ytr,_,_,_=load_problem(name,'visible_fine'); D=m.Data(Xtr.astype(float),ytr)
w=m.make_model(k,60.).fit(Xtr,ytr).coef_.astype(float)
ils=m.ChainILS(D,k)
def errs(w):
    st=ils._counts(D.score(w)); return m.hist_errors(st[1],st[2],st[3])
Lb,wb=ils.run(w); eb=errs(wb)
print('start regret',(Lb-bk[(name,k)])*1e4,'errors',eb)
we,_=ils.err_descent(wb)[1],0
we=ils.err_descent(wb)[1]; ee=errs(we); print('err descent errors',ee)
rng=np.random.default_rng(0); t0=time.perf_counter(); it=0
while time.perf_counter()-t0<budget:
    it+=1
    w2=we.copy(); S=np.flatnonzero(w2); w2[rng.choice(S,size=rng.integers(1,3),replace=False)]=0
    _,w2=ils.err_descent(w2, max_iter=100); e2=errs(w2)
    L2,w3=ils.run(w2)
    if L2<Lb-1e-12: Lb,wb=L2,w3; print(it, 'new loss best', round((Lb-bk[(name,k)])*1e4,1), 'errs', errs(w3), flush=True)
    if e2<=ee:
        if e2<ee: print(it,'errors',e2, flush=True)
        ee,we=e2,w2
print('final', (Lb-bk[(name,k)])*1e4, 'min errors', ee, 'iters', it)
