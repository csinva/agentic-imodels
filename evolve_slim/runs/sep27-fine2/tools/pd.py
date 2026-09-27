import os,sys,time
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"]: os.environ[v]="1"
sys.path.insert(0,'src')
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from suite import load_problem
X,y,*_=load_problem(sys.argv[2],'visible_fine')
T=time.perf_counter
for rep in range(2):
    t=T(); isbin,cnt,B=m.scan_columns(X); const=(cnt==0)|(cnt==X.shape[0]); t1=T()-t
    cand=np.flatnonzero(isbin&~const); order=cand[np.lexsort((cand,-cnt[cand]))]
    t=T(); chain,lev,nch=m.build_chains(B,cnt,order,X.shape[1]); t2=T()-t
    t=T(); Q=m.chain_codes(B,X.shape[0],chain,lev,nch); t3=T()-t
    t=T(); D=m.Data(X,y); t4=T()-t
print('scan %.1f chains %.1f codes %.1f total %.1f ms'%(t1*1e3,t2*1e3,t3*1e3,t4*1e3))
