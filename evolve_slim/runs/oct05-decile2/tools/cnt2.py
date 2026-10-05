"""Per-problem comparison of model B against A (rows of results/problem_results.csv or scratch/<tag>.csv):
problems improved / worsened (|dloss| > th), mean dloss, geo time ratio, mean dAUC; and the gap of each to the
per-problem best of all models in problem_results.csv. usage: uv run tools/cnt2.py A B [th]"""
import os, sys, numpy as np, pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = pd.read_csv(f"{here}/results/problem_results.csv")
def rd(t):
    f = f"{here}/scratch/{t}.csv"
    if os.path.exists(f): return pd.read_csv(f)
    return P[P.model == t].reset_index(drop=True)
A, B = rd(sys.argv[1]), rd(sys.argv[2]); th = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
m = A.merge(B, on=["dataset", "k"], suffixes=("_a", "_b"))
m["dl"] = m.loss_b - m.loss_a
best = P.groupby(["dataset", "k"]).loss.min().rename("best").reset_index(); m = m.merge(best, on=["dataset", "k"])
imp, wor = (m.dl < -th).sum(), (m.dl > th).sum()
tr = np.exp(np.mean(np.log(m.seconds_b / m.seconds_a)))
print(f"improve {imp} worsen {wor} mean dloss {m.dl.mean():+.6f} geo time ratio {tr:.3f} dauc {(m.auc_test_b - m.auc_test_a).mean():+.5f} "
      f"gap-to-best A {(m.loss_a - m.best).mean():.6f} B {(m.loss_b - m.best).mean():.6f}")
ch = m[m.dl.abs() > th].sort_values("dl")
print(ch[["dataset", "k", "dl"]].to_string(index=False) if len(ch) else "(no change)")
