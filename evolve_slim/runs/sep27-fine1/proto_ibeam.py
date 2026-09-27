import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, csv, time
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m','slim.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def state(ils, w):
    D=ils.D; s=D.X@w; st=ils._counts(s)
    if np.ptp(s)==0:
        npos=D.c[D.y>0].sum(); L,a,b=m.calibrate_bins(st[1],st[2],st[3],0.,np.log(npos/(D.N-npos))); a=0.5
    else:
        sd=s.std(); L,a,b=m.calibrate_bins(st[1]/sd,st[2],st[3],0.,0.); a/=sd
    return s,st,L,a,b

def int_beam(D,k,B,E):
    ils=m.ChainILS(D,k); ils.n_exact=max(ils.n_exact,E)
    beam=[np.zeros(D.d)]
    for t in range(k):
        ch={}
        for w in beam:
            s,st,L,a,b=state(ils,w)
            cands=[c for c in ils._basic(w,s,st,L,a,b,np.inf) if (c[1]==-1 and w[c[2]]==0) or (c[1]==-2 and np.count_nonzero(c[3])==np.count_nonzero(w[D.ccol[D.cptr[c[2]]:D.cptr[c[2]+1]]])+1)]
            cands.sort(key=lambda c:c[0])
            for c in cands[:2*E]:
                r=ils._check([c],w,s,L,a,b,Lbar=np.inf)
                if r is None: continue
                w2=w.copy(); ils._apply(w2,r)
                ch[w2.tobytes()]=(r[0],w2)
        beam=[w for _,w in sorted(ch.values(),key=lambda t:t[0])[:B]]
    return beam

for p in sys.argv[1].split(','):
    name,k=p.split(':'); k=int(k)
    Xtr,ytr,_,_,_=load_problem(name,'visible_fine'); D=m.Data(Xtr.astype(float),ytr)
    t=time.perf_counter(); beam=int_beam(D,k,int(sys.argv[2]),int(sys.argv[3])); tb=time.perf_counter()-t
    ils=m.ChainILS(D,k); res=[]
    t=time.perf_counter()
    for w in beam:
        L,w2=ils.run(w); res.append((L-bk[(name,k)])*1e4)
    ti=time.perf_counter()-t
    parents=m.beam_search(D,k); r0=[]
    ils=m.ChainILS(D,k)
    for nd in parents[:5]:
        w,l=m.calib_round(D,nd); L,_=ils.run(w); r0.append((L-bk[(name,k)])*1e4)
    print(f"{name} {k}: ibeam+ils {np.round(sorted(res),1)} ({tb:.2f}+{ti:.2f}s) | beam+ils {np.round(sorted(r0),1)}", flush=True)
