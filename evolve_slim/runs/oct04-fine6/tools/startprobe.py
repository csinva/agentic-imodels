import sys, os, time, importlib.util, numpy as np, pickle
os.environ["SPAT"] = "0"; os.environ["NSTARTS"] = "10"; os.environ["FINAL_POOL"] = "10"; os.environ["RSWAP"]="0"; os.environ["POLISH"]="0"
sys.path.insert(0, "src")
from suite import load_problem, suite_datasets, K_VALUES
spec = importlib.util.spec_from_file_location("m", "slim.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = {}
for name in suite_datasets("visible_fine"):
    X, y, *_ = load_problem(name, "visible_fine")
    for k in K_VALUES:
        mod = m.make_model(k, 60.0).fit(X, y)
        out[(name, k)] = mod.start_losses_
os.makedirs("scratch", exist_ok=True); pickle.dump(out, open("scratch/sprobe.pkl", "wb"))  # run from the run folder
