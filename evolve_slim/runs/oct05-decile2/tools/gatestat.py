"""Gate statistics at a model's final points (results/problem_results.csv): full logit range a*(max-min score),
robust range a*(q99-q01), the fraction of rows with |a s + b| > 4, loss / base entropy.
usage: uv run tools/gatestat.py MODEL"""
import os, sys, json, numpy as np, pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(here, "src"))
from suite import load_problem
P = pd.read_csv(f"{here}/results/problem_results.csv"); P = P[P.model == sys.argv[1]]
for ds, g in P.groupby("dataset", sort=False):
    X, y, *_, names = load_problem(ds)
    idx = {n: i for i, n in enumerate(names)}
    p = y.mean(); H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    for _, r in g.sort_values("k").iterrows():
        w = np.zeros(X.shape[1])
        for n, v in json.loads(r.points).items(): w[idx[n]] = v
        s = X @ w; z = r.a * s + r.b
        q1, q99 = np.quantile(s, [0.01, 0.99])
        print(f"{ds:12s} k{r.k:2d} L {r.loss:.4f} L/H {r.loss/H:.3f} full {abs(r.a)*(s.max()-s.min()):6.1f} q {abs(r.a)*(q99-q1):6.1f} conf {np.mean(np.abs(z) > 4):.2f}")
