import sys; sys.path.insert(0,'src')
import numpy as np
from suite import load_problem, DATASETS
for d in DATASETS:
    X,y,*_=load_problem(d,'visible_fine')
    isb=np.all((X==0)|(X==1),axis=0)
    print(d, X.shape, 'nonbin', (~isb).sum(), 'const', (np.ptp(X,0)==0).sum(), 'uniq rows', len(np.unique(X,axis=0)), 'dens %.2f'%(X!=0).mean())
