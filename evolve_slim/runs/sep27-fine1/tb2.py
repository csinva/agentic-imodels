import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, time
from suite import load_problem
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for p in sys.argv[2].split(';'):
    name,k=p.split(); k=int(k)
    Xtr,ytr,_,_,_=load_problem(name,'visible_fine')
    for key in m.TM: m.TM[key]=0
    t=time.perf_counter(); mdl=m.make_model(k,60.).fit(Xtr,ytr); t=time.perf_counter()-t
    print(f"{name} {k} total {t:.2f}", {a:round(b,3) for a,b in m.TM.items()})
