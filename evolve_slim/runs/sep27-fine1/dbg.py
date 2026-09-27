import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, time
from suite import load_problem
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Xtr,ytr,_,_,_=load_problem(sys.argv[2],'visible_fine')
k=int(sys.argv[3])
D=m.Data(Xtr.astype(float),ytr)
parents=m.beam_search(D,k)
for nd in parents:
    w,l=m.calib_round(D,nd)
    t=time.perf_counter(); ils=m.ChainILS(D,k); l2,w2=ils.run(w); t2=time.perf_counter()-t
    t=time.perf_counter(); ils0=m.ILS(D,k); l3,w3=ils0.run(w); t3=time.perf_counter()-t
    l4,w4=m.ILS(D,k).run(w2)
    print(f"{nd.loss/D.N:.5f} round {l:.5f} chain {l2:.5f} ({t2:.2f}s) old {l3:.5f} ({t3:.2f}s) chain+old {l4:.5f}")
