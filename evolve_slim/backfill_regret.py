"""Rewrite the regret columns of every leaderboard against the current src/best_known.csv.

    uv run backfill_regret.py            # results/ and every runs/*/results/

regret = loss - best known loss of the problem; mean_regret is its mean over a
model's problems. Only these two columns change.
"""

import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
best = pd.read_csv(os.path.join(ROOT, "src", "best_known.csv"))
key = {(r.suite, r.dataset, int(r.k)): float(r.loss) for r in best.itertuples()}
for d in [os.path.join(ROOT, "results")] + glob.glob(os.path.join(ROOT, "runs", "*", "results")):
    for pf in glob.glob(os.path.join(d, "*problem_results.csv")):
        prefix = os.path.basename(pf)[: -len("problem_results.csv")]
        prob = pd.read_csv(pf)
        prob["best_known"] = [key.get((s, ds, int(k)), float("nan")) for s, ds, k in
                              zip(prob["suite"], prob["dataset"], prob["k"])]
        prob["regret"] = prob["loss"] - prob["best_known"]
        prob.to_csv(pf, index=False)
        of = os.path.join(d, f"{prefix}overall_results.csv")
        if os.path.exists(of):
            ov = pd.read_csv(of, dtype=str, keep_default_na=False)
            means = prob.groupby("model")["regret"].mean()
            ov["mean_regret"] = [f"{means[m]:.5f}" if m in means.index else r
                                 for m, r in zip(ov["model_name"], ov["mean_regret"])]
            ov.to_csv(of, index=False)
        print(f"updated {os.path.relpath(pf, ROOT)}")
