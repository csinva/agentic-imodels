import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, time, csv
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
name=sys.argv[2]; k=int(sys.argv[3]); budget=float(sys.argv[4])
Xtr,ytr,_,_,_=load_problem(name,'visible_fine')
D=m.Data(Xtr.astype(float),ytr)
parents=m.beam_search(D,k)
w,l=m.calib_round(D,parents[0])
ils=m.ChainILS(D,k)
L,w=ils.run(w)
print('start', (L-bk[(name,k)])*1e4)
rng=np.random.default_rng(0)
t0=time.perf_counter(); it=0
while time.perf_counter()-t0<budget:
    it+=1
    w2=w.copy(); S=np.flatnonzero(w2)
    for j in rng.choice(S, size=min(len(S), rng.integers(1,3)), replace=False): w2[j]=0
    ils.visited=set()
    L2,w2=ils.run(w2)
    if L2<L-1e-12:
        L,w=L2,w2; print(it, round(time.perf_counter()-t0,1), (L-bk[(name,k)])*1e4, flush=True)
print('final', (L-bk[(name,k)])*1e4, 'iters', it)
