import os, sys, importlib.util
sys.path.insert(0, 'src')
import numpy as np
from suite import load_problem, DATASETS
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for d in DATASETS:
    X, y = load_problem(d, 'visible_fine')[:2]
    D = m.Data(X, y)
    nz = (D.QT > 0).mean(1)
    print(f"{d:12s} n={D.n} d={D.d} V={D.V} chainlen>1={np.sum(D.ncode>2)} nnzfrac_mean={nz.mean():.2f} sum_nnz/n={nz.sum():.1f} maxncode={D.ncode.max()}")
