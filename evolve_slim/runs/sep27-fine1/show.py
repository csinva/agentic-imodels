import sys, os; sys.path.insert(0,'src')
import numpy as np, importlib.util, csv
from suite import load_problem
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
name=sys.argv[2]; k=int(sys.argv[3])
Xtr,ytr,Xte,yte,fn=load_problem(name,'visible_fine')
D=m.Data(Xtr.astype(float),ytr)
for seed in sys.argv[4].split(','):
    os.environ['SLIM_SEED']=seed
    mdl=m.make_model(k,60.).fit(Xtr,ytr); w=mdl.coef_
    st=m.ScoreState(D,D.X@w)
    S=np.flatnonzero(w)
    print(seed, round(st.L/D.N,5),'a',round(st.a,2),'b',round(st.b,2), sorted([(fn[j],int(w[j])) for j in S]))
