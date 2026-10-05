"""Compare per-problem regret/time/auc of two scratch tags. usage: uv run tools/cmp.py A B [thresh]"""
import sys, numpy as np, pandas as pd
here = __file__.rsplit("/tools/", 1)[0]
import os
def rd(t):
    f = f"{here}/scratch/{t}.csv"
    if os.path.exists(f): return pd.read_csv(f)
    P = pd.read_csv(f"{here}/results/problem_results.csv"); return P[P.model == t].reset_index(drop=True)
A = rd(sys.argv[1]); B = rd(sys.argv[2])
th = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
m = A.merge(B, on=["dataset", "k"], suffixes=("_a", "_b"))
m["dl"] = m.loss_b - m.loss_a; m["dauc"] = m.auc_test_b - m.auc_test_a; m["tr"] = m.seconds_b / m.seconds_a
print(m[m.dl.abs() > th][["dataset", "k", "dl", "dauc", "seconds_a", "seconds_b"]].to_string())
print(f"mean dloss {m.dl.mean():.6f}  mean dauc {m.dauc.mean():.5f}  geo time ratio {np.exp(np.log(m.tr).mean()):.3f}")
g = m.groupby("dataset")[["seconds_a", "seconds_b"]].sum(); print((g.seconds_b / g.seconds_a).round(2).to_dict())
