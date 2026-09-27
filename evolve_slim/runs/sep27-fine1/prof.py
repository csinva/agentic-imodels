import sys; sys.path.insert(0,'src')
import numpy as np, importlib.util, time, cProfile, pstats
from suite import load_problem
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Xtr,ytr,_,_,_=load_problem(sys.argv[2],'visible_fine'); k=int(sys.argv[3])
mdl=m.make_model(k,60.0)
cProfile.run('mdl.fit(Xtr,ytr)','/tmp/prof.out')
print(getattr(mdl,'timing_',None))
pstats.Stats('/tmp/prof.out').sort_stats('cumtime').print_stats(22)
