import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, time, csv
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
name=sys.argv[2]; k=int(sys.argv[3]); nk=int(sys.argv[4]); size=int(sys.argv[5]) if len(sys.argv)>5 else 2
Xtr,ytr,_,_,_=load_problem(name,'visible_fine')
D=m.Data(Xtr.astype(float),ytr)
parents=m.beam_search(D,k)
ils=m.ChainILS(D,k)
for nd in parents:
    w,l=m.calib_round(D,nd)
    L,w=ils.run(w); L0=L
    rng=np.random.default_rng(0)
    for it in range(nk):
        w2=w.copy(); S=np.flatnonzero(w2)
        w2[rng.choice(S, size=min(len(S), rng.integers(1,size+1)), replace=False)]=0
        L2,w2=ils.run(w2)
        if L2<L-1e-12: L,w=L2,w2
    print(f"round {(l-bk[(name,k)])*1e4:8.1f} ils {(L0-bk[(name,k)])*1e4:8.1f} kicks {(L-bk[(name,k)])*1e4:8.1f}")
