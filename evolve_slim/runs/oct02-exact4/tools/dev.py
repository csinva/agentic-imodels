"""Dev driver: fit one or more (dataset, k) problems with a modified copy of slim.py and print stats.
usage: python scratch/dev.py heart:10 australian:7 --limit 600 --set GIVEUP=1e9 --file slim.py"""
import argparse, importlib.util, os, re, sys, time
import numpy as np

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem

ap = argparse.ArgumentParser()
ap.add_argument("probs", nargs="+")
ap.add_argument("--limit", type=float, default=60.0)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--file", default=os.path.join(here, "slim.py"))
a = ap.parse_args()
src = open(a.file).read()
src = src.split("# ===========================================================================\n# Evaluation loop")[0]
for kv in a.set:
    k, v = kv.split("=", 1)
    src, n = re.subn(rf"^{k} = [^#\n]*", f"{k} = {v}  ", src, count=1, flags=re.M)
    assert n == 1, k
tmp = os.path.join(here, "tools", f"_mod_{os.getpid()}.py")
open(tmp, "w").write(src)
try:
    spec = importlib.util.spec_from_file_location("slimmod", tmp)
    M = importlib.util.module_from_spec(spec)
    t0 = time.time()
    spec.loader.exec_module(M)
    print(f"import+warmup {time.time() - t0:.1f}s", flush=True)
finally:
    os.remove(tmp)
for p in a.probs:
    ds, k = p.split(":")
    X, y, _, _, _ = load_problem(ds)
    m = M.make_model(int(k), a.limit)
    t0 = time.perf_counter()
    m.fit(X, y)
    el = time.perf_counter() - t0
    st = getattr(M.certify, "stats", None)
    cert = m.lower_bound_ >= m.train_loss_ - 1e-7
    print(f"{ds} k={k} t={el:.2f}s loss={m.train_loss_:.8f} lb={m.lower_bound_:.8f} cert={cert} "
          f"stats={list(st) if st is not None else None}", flush=True)
