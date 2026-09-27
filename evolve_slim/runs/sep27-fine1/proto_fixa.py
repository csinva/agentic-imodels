import sys, os; sys.path.insert(0,'src')
import numpy as np, importlib.util, csv, time
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m','slim.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def fixed_descent(ils, w, A, iters=60):
    """descent on min_b loss(A s + b): candidates by exact fixed-(A,b) loss F, b re-fit after each move"""
    D=ils.D; k=ils.k; cols=D.chain_cols; dl=ils.deltas_all; nzv=ils.nzv
    w=w.copy(); s=D.score(w)
    def fitb(s,b):
        st=ils._counts(s); sv,Wp,Wn=st[1],st[2],st[3]
        for _ in range(30):
            z=A*sv+b; p=1/(1+np.exp(-z)); g=(Wn*p-Wp*(1-p)).sum(); h=((Wp+Wn)*p*(1-p)).sum()+1e-12; b-=g/h
        z=A*sv+b; L=(Wp*np.logaddexp(0,-z)+Wn*np.logaddexp(0,z)).sum()
        return st,b,L
    st,b,L=fitb(s,0.0)
    for _ in range(iters):
        inv,sv,Wp,Wn,Cp,Cn=st; S=np.flatnonzero(w); full=len(S)>=k
        TP,TN=m.move_tables(sv,A,b,dl); F=L+(Cp[cols]@TP[:,:20]+Cn[cols]@TN[:,:20])
        wc=w[cols]; bad=(np.abs(wc[:,None]+dl[None,:])>5)|((wc==0)[:,None]&full); F[bad]=np.inf
        f=int(np.argmin(F)); best=(F.flat[f],-1,cols[f//20],wc[f//20]+dl[f%20])
        for rj in S:
            s2=s-w[rj]*D.XT[rj]; st2=ils._counts(s2)
            z=A*st2[1]+b; L2=(st2[2]*np.logaddexp(0,-z)+st2[3]*np.logaddexp(0,z)).sum()
            TP,TN=m.move_tables(st2[1],A,b,nzv); F2=L2+(st2[4][cols]@TP[:,:10]+st2[5][cols]@TN[:,:10]); F2[w[cols]!=0]=np.inf
            f=int(np.argmin(F2))
            if F2.flat[f]<best[0]: best=(F2.flat[f],rj,cols[f//10],nzv[f%10])
        if best[0]>=L*(1-1e-9): break
        _,rj,aj,v=best
        if rj>=0: w[rj]=0
        w[aj]=v; s=D.score(w); st,b,L=fitb(s,b)
    return w

for p in sys.argv[1].split(','):
    name,k=p.split(':'); k=int(k)
    Xtr,ytr,_,_,_=load_problem(name,'visible_fine'); D=m.Data(Xtr.astype(float),ytr)
    w0=m.make_model(k,60.).fit(Xtr,ytr).coef_.astype(float)
    ils=m.ChainILS(D,k); s,st,L0,a,b=ils.state(w0)
    row=f"{name} {k} start {(L0/D.N-bk[(name,k)])*1e4:.1f} a={a:.2f}:"
    for g in [0.25,0.5,2,4]:
        best=np.inf; rng=np.random.default_rng(0); wf=fixed_descent(ils,w0,g*a)
        L1,_=ils.run(wf); best=min(best,L1)
        for t in range(10):
            w2=wf.copy(); S=np.flatnonzero(w2); w2[rng.choice(S,size=rng.integers(1,3),replace=False)]=0
            w2=fixed_descent(ils,w2,g*a); L1,_=ils.run(w2); best=min(best,L1)
        row+=f" g{g}: {(best-bk[(name,k)])*1e4:.1f}"
    print(row, flush=True)
