import sys, importlib.util, numpy as np, csv
sys.path.insert(0,'src')
from suite import load_problem
from evaluate import calibrate
spec=importlib.util.spec_from_file_location('m', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bk={(r['dataset'],int(r['k'])):float(r['loss']) for r in csv.DictReader(open('src/best_known.csv'))}
for pk in sys.argv[2].split(';'):
    ds,k=pk.split(','); k=int(k)
    X,y,_,_,f=load_problem(ds)
    for kw in sys.argv[3:] or ['']:
        kwargs=eval('dict('+kw+')')
        mod=m.SparseIntegerClassifier(k=k, **kwargs).fit(X,y)
        w=mod.coef_; L=calibrate(X@w,y)[2]
        print(ds,k,kw,'regret %.5f'%(L-bk[(ds,k)]), {f[j]:int(w[j]) for j in np.flatnonzero(w)})
