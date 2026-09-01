import os, sys
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error
J = _os.path.dirname(_os.path.abspath(__file__))
Xtr = np.load(f"{J}/_X.npy"); ytr = np.load(f"{J}/_y.npy")
Xte = np.load(f"{J}/_Xt.npy"); yte = np.load(f"{J}/_yt.npy")
cap = int(os.environ.get("CAP", "2500"))
rng = np.random.RandomState(0)
if len(Xtr) > cap:
    idx = rng.choice(len(Xtr), cap, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
if Xtr.shape[1] > 500:
    Xtr, Xte = Xtr[:, :500], Xte[:, :500]
m = TabPFNRegressor(device="mps", random_state=42, ignore_pretraining_limits=True)
m.fit(Xtr, ytr)
p = np.concatenate([m.predict(Xte[i:i+100]) for i in range(0, len(Xte), 100)])
print("RMSE", float(np.sqrt(mean_squared_error(yte, p))))
