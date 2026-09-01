import json, os, time
import pandas as pd
from sklearn.datasets import fetch_openml
J = _os.path.dirname(_os.path.abspath(__file__))
C = os.path.expanduser("~/.cache/imodels-evolve/ctr23")
rows = json.load(open(f"{J}/ctr23.json"))
TAB = {r[0] for r in rows if r[6]}
todo = [r for r in rows if not r[6]]
print("fetching", len(todo), "datasets", flush=True)
for nm, tid, did, tgt, n, d, dup in todo:
    fp = f"{C}/{did}.parquet"
    if os.path.exists(fp):
        print(f"  cached {nm}", flush=True); continue
    for attempt in range(3):
        try:
            t0 = time.time()
            b = fetch_openml(data_id=int(did), as_frame=True, parser="auto")
            b.frame.to_parquet(fp)
            print(f"  got {nm:<34} n={n} d={d} ({time.time()-t0:.0f}s)", flush=True)
            break
        except Exception as e:
            print(f"  retry {nm}: {type(e).__name__}", flush=True); time.sleep(10)
print("DONE", flush=True)
