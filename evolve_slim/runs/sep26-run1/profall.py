import sys, time, cProfile, pstats, importlib.util, math
sys.path.insert(0,'src')
from suite import load_problem, DATASETS, K_VALUES
spec=importlib.util.spec_from_file_location('m', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
rng=np.random.default_rng(0); Xw=(rng.random((300,12))<.4).astype(float); yw=(Xw[:,0]+Xw[:,1]+rng.random(300)>1.2).astype(int)
m.make_model(3,5).fit(Xw,yw)
data={d:load_problem(d) for d in DATASETS}
pr=cProfile.Profile()
logs=[]
for d in (sys.argv[2].split(",") if len(sys.argv)>2 else DATASETS):
    X,y,_,_,_=data[d]
    for k in K_VALUES:
        t=time.perf_counter(); pr.enable(); m.make_model(k,60).fit(X,y); pr.disable(); logs.append(math.log(time.perf_counter()-t))
print('geo', math.exp(np.mean(logs)))
st=pstats.Stats(pr); st.sort_stats('tottime').print_stats(15)
