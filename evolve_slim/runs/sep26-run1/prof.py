import sys, time, cProfile, pstats, importlib.util
sys.path.insert(0,'src')
from suite import load_problem
spec=importlib.util.spec_from_file_location('m', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ds=sys.argv[2]; k=int(sys.argv[3])
X,y,_,_,_=load_problem(ds)
m.make_model(3,5).fit(X[:300,:12],y[:300])
t=time.time()
if len(sys.argv)>4:
    cProfile.run('m.make_model(k,60).fit(X,y)','/tmp/prof_sep26'); pstats.Stats('/tmp/prof_sep26').sort_stats('cumtime').print_stats(18)
else:
    mod=m.make_model(k,60).fit(X,y)
print('time',time.time()-t)
