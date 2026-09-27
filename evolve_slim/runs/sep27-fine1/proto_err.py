import sys, os; sys.path.insert(0,'src')
import numpy as np, importlib.util, csv, time, numba as nb
from suite import load_problem
bk = {(r['dataset'], int(r['k'])): float(r['loss']) for r in csv.DictReader(open('src/best_known.csv')) if r['suite'] == 'visible_fine'}
spec=importlib.util.spec_from_file_location('m','slim.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@nb.njit(cache=False)
def errs_all(Cp, Cn, cols, sv, Wp, Wn, deltas, allowed):
    """weighted errors of the best threshold (either orientation) after w_j += delta, for all (col, delta)."""
    nb_ = sv.shape[0]; nd = deltas.shape[0]
    lo = sv.min() - 10.0; R = int(sv.max() - lo) + 11
    Hp = np.zeros(R); Hn = np.zeros(R)
    for q in range(nb_):
        Hp[int(sv[q] - lo)] += Wp[q]; Hn[int(sv[q] - lo)] += Wn[q]
    Tp = Wp.sum(); Tn = Wn.sum()
    out = np.full((cols.shape[0], nd), np.inf)
    hp = np.empty(R); hn = np.empty(R)
    for u in range(cols.shape[0]):
        j = cols[u]
        for r in range(nd):
            if not allowed[u, r]:
                continue
            dl = int(deltas[r])
            hp[:] = Hp; hn[:] = Hn
            for q in range(nb_):
                cp = Cp[j, q]; cn = Cn[j, q]
                if cp + cn > 0:
                    i0 = int(sv[q] - lo)
                    hp[i0] -= cp; hn[i0] -= cn; hp[i0 + dl] += cp; hn[i0 + dl] += cn
            # threshold between cells: predict + above. errors = pos at/below + neg above
            best = min(Tp, Tn)
            cp_ = 0.0; cn_ = 0.0
            for i in range(R):
                cp_ += hp[i]; cn_ += hn[i]
                e1 = cp_ + (Tn - cn_)   # + above
                e2 = cn_ + (Tp - cp_)   # + below
                if e1 < best: best = e1
                if e2 < best: best = e2
            out[u, r] = best
    return out

def err_of(D, s):
    inv, sv, Wp, Wn = m.bin_scores(s, D.y, D.c)
    o = np.argsort(sv); cp = np.cumsum(Wp[o]); cn = np.cumsum(Wn[o]); Tp, Tn = cp[-1], cn[-1]
    return min(Tp, Tn, (cp + Tn - cn).min(), (cn + Tp - cp).min())

def err_ils(D, k, w, iters=200):
    ils = m.ChainILS(D, k); cols = D.chain_cols; dl = ils.deltas_all; nzv = ils.nzv
    w = w.copy(); s = D.score(w)
    cur = err_of(D, s); _, _, Lc, _, _ = state(ils, w)
    for it in range(iters):
        S = np.flatnonzero(w); full = len(S) >= k
        inv, sv, Wp, Wn, Cp, Cn = ils._counts(s)
        _, _, L, a, b = state(ils, w)
        best = None
        wc = w[cols]; allowed = (np.abs(wc[:, None] + dl[None, :]) <= 5) & ~((wc == 0)[:, None] & full)
        E = errs_all(Cp, Cn, cols, sv, Wp, Wn, dl, allowed)
        # candidates: all with minimal errors; tie-break by estimated calibrated loss
        est = ils._estimates(sv, Wp, Wn, Cp, Cn, a, b, dl, cols, np.inf, ~allowed)
        key = E * 1e6 + est / D.N
        f = int(np.argmin(key)); q, r = divmod(f, dl.shape[0])
        cand = (E.flat[f], est.flat[f], -1, cols[q], w[cols[q]] + dl[r])
        for rj in S:
            s2 = s - w[rj] * D.XT[rj]
            inv2, sv2, Wp2, Wn2, Cp2, Cn2 = ils._counts(s2)
            allowed2 = np.broadcast_to((w[cols] == 0)[:, None], (len(cols), len(nzv))).copy()
            E2 = errs_all(Cp2, Cn2, cols, sv2, Wp2, Wn2, nzv, allowed2)
            est2 = ils._estimates(sv2, Wp2, Wn2, Cp2, Cn2, a, b, nzv, cols, np.inf, ~allowed2)
            key2 = E2 * 1e6 + est2 / D.N
            f = int(np.argmin(key2)); q, r = divmod(f, nzv.shape[0])
            if key2.flat[f] < cand[0] * 1e6 + cand[1] / D.N:
                cand = (E2.flat[f], est2.flat[f], rj, cols[q], nzv[r])
        e, es, rj, aj, v = cand
        if e > cur or (e == cur and es >= L * (1 - 1e-9)):
            break
        if rj >= 0: w[rj] = 0
        w[aj] = v; s = D.score(w); cur = err_of(D, s)
    return w, cur

def state(ils, w):
    D=ils.D; s=D.score(w); st=ils._counts(s); sd=s.std()
    L,a,b=m.calibrate_bins(st[1]/sd,st[2],st[3],0.,0.); a/=sd
    return s,st,L,a,b

for p in sys.argv[1].split(','):
    name,k=p.split(':'); k=int(k)
    Xtr,ytr,_,_,_=load_problem(name,'visible_fine'); D=m.Data(Xtr.astype(float),ytr)
    for seed in ['0','1']:
        os.environ['SLIM_SEED']=seed
        w0=m.make_model(k,60.).fit(Xtr,ytr).coef_.astype(float)
        L0=m.ScoreState(D,D.score(w0)).L/D.N
        t=time.perf_counter(); w1,e1=err_ils(D,k,w0); t1=time.perf_counter()-t
        L1=m.ScoreState(D,D.score(w1)).L/D.N
        L2,w2=m.ChainILS(D,k).run(w1)
        print(name,k,seed,'start',round((L0-bk[(name,k)])*1e4,1),'errs',err_of(D,D.score(w0)),'-> err-ils errs',e1,'loss',round((L1-bk[(name,k)])*1e4,1),'polished',round((L2-bk[(name,k)])*1e4,1),f'{t1:.2f}s',flush=True)
