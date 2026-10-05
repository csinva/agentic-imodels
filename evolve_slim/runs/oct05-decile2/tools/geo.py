"""Exact geo-mean time, regret, loss, AUC of recorded models. usage: uv run tools/geo.py m1 m2 ..."""
import sys, numpy as np, pandas as pd
P = pd.read_csv(__file__.rsplit("/tools/", 1)[0] + "/results/problem_results.csv")
for m in sys.argv[1:]:
    r = P[P.model == m]
    print(f"{m:22s} geo {np.exp(np.log(r.seconds).mean()):.5f} regret {r.regret.mean():.6f} loss {r.loss.mean():.6f} auc {r.auc_test.mean():.5f}")
