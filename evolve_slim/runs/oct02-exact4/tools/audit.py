"""Brute-force audit of the certificate: random small binary problems, the optimum by enumerating every integer
vector with at most k nonzeros in [-5, 5] (exact 2-parameter calibration), then the solver's fit.
Checks lower_bound_ <= optimum + 1e-9 and (if certified) loss == optimum. usage: audit.py n_problems [k] [d]"""
import importlib.util, itertools, os, sys, time
import numpy as np

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import re
src = open(os.path.join(here, "slim.py")).read().split(
    "# ===========================================================================\n# Evaluation loop")[0]
for kv in sys.argv[5:]:
    k_, v_ = kv.split("=", 1)
    src, n_ = re.subn(rf"^{k_} = [^#\n]*", f"{k_} = {v_}  ", src, count=1, flags=re.M)
    assert n_ == 1, k_
tmp = os.path.join(here, "tools", f"_aud_{os.getpid()}.py")
open(tmp, "w").write(src)
spec = importlib.util.spec_from_file_location("slimaud", tmp)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
os.remove(tmp)


def calib_loss(S, y):
    """min over a, b of mean log loss of a * S[:, q] + b, for every column q (vectorised Newton)."""
    n, Q = S.shape
    a = np.zeros(Q)
    p = y.mean()
    b = np.full(Q, np.log(p / (1 - p)))
    for it in range(100):
        z = a[None, :] * S + b[None, :]
        q = 1 / (1 + np.exp(-z))
        r = q - y[:, None]
        w = q * (1 - q)
        ga = (r * S).sum(0)
        gb = r.sum(0)
        haa = (w * S * S).sum(0) + 1e-12
        hab = (w * S).sum(0)
        hbb = w.sum(0) + 1e-12
        det = haa * hbb - hab * hab
        det = np.where(det > 1e-14, det, 1e-14)
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        F0 = np.logaddexp(0, z).sum(0) - (y[:, None] * z).sum(0)
        t = np.ones(Q)
        for ls in range(30):
            z2 = (a - t * da)[None, :] * S + (b - t * db)[None, :]
            F2 = np.logaddexp(0, z2).sum(0) - (y[:, None] * z2).sum(0)
            bad = F2 > F0 + 1e-12
            if not bad.any():
                break
            t = np.where(bad, t / 2, t)
        a -= t * da
        b -= t * db
        if np.max(np.abs(t * da)) < 1e-12 and np.max(np.abs(t * db)) < 1e-12:
            break
    z = a[None, :] * S + b[None, :]
    return (np.logaddexp(0, z).sum(0) - (y[:, None] * z).sum(0)) / n


def brute(X, y, k):
    n, d = X.shape
    best = np.inf
    vals = np.array([v for v in range(-5, 6) if v != 0])
    for kk in range(1, k + 1):
        for S in itertools.combinations(range(d), kk):
            W = np.array(list(itertools.product(vals, repeat=kk)), float)
            W = W[W[:, 0] > 0]                       # w and -w give the same criterion
            for ch in range(0, len(W), 20000):
                sc = X[:, S] @ W[ch:ch + 20000].T
                best = min(best, calib_loss(sc, y).min())
    p = y.mean()
    best = min(best, -(p * np.log(p) + (1 - p) * np.log(1 - p)))
    return best


nprob = int(sys.argv[1])
kk = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dd = int(sys.argv[3]) if len(sys.argv) > 3 else 6
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
bad = 0
ncert = 0
for it in range(nprob):
    n = int(rng.integers(60, 220))
    d = dd
    X = (rng.random((n, d)) < rng.uniform(0.2, 0.6, size=d)).astype(float)
    if rng.random() < 0.5:
        X[:, d - 1] = X[:, 0]                          # planted duplicate
    if rng.random() < 0.3:
        X[:, d - 2] = 1 - X[:, 1]                      # planted complement
    beta = rng.normal(0, 1.2, size=d) * (rng.random(d) < 0.7)
    z = X @ beta - beta.sum() / 2 + rng.normal(0, 0.8, size=n)
    y = (z > 0).astype(np.int64)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    opt = brute(X, y, kk)
    m = M.make_model(kk, 60.0)
    t0 = time.time()
    m.fit(X, y)
    lb, loss = m.lower_bound_, m.train_loss_
    cert = lb >= loss - 1e-7
    ok = lb <= opt + 1e-9 and (not cert or abs(loss - opt) <= 1e-7)
    ncert += cert
    bad += not ok
    print(f"{it}: n={n} opt={opt:.9f} loss={loss:.9f} lb={lb:.9f} cert={cert} ok={ok} t={time.time()-t0:.2f}s "
          f"stats={list(M.certify.stats[:8])}", flush=True)
print(f"AUDIT k={kk} d={dd}: {nprob} problems, {ncert} certified, {bad} wrong")
