import sys, time, os
[os.environ.__setitem__(v, "1") for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"]]
import numpy as np
sys.path.insert(0, 'src')
[os.environ.__setitem__(v, '1') for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']]
import importlib.util
spec = importlib.util.spec_from_file_location('slim', sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.py') else 'slim.py')
slim = importlib.util.module_from_spec(spec); spec.loader.exec_module(slim)
from suite import load_problem
from evaluate import calibrate
args = [a for a in sys.argv[1:] if not a.endswith('.py')]
probs = [(a.split(':')[0], int(a.split(':')[1])) for a in args]
for name, k in probs:
    X, y, *_ = load_problem(name)
    t0 = time.perf_counter()
    m = slim.make_model(k, 60.0).fit(X, y)
    t = time.perf_counter() - t0
    loss = calibrate(X @ m.coef_, y)[2]
    st = getattr(slim.certify, 'stats', None)
    print(f"{name} k={k} t={t:.2f}s loss={loss:.9f} lb={m.lower_bound_:.9f} cert={m.lower_bound_ >= loss - 1e-7} "
          f"stats={None if st is None else st.tolist()}", flush=True)
