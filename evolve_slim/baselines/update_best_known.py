"""Rebuild src/best_known.csv: for every (suite, dataset, k), the lowest criterion any
integer solver has reached in results/*problem_results.csv or runs/*/results/.

    uv run baselines/update_best_known.py

Run by the human between loop sessions, never by the agent; mean_regret is measured
against this file, so a refresh shifts every row of a leaderboard by the same amount
per problem (re-run `uv run backfill_regret.py` afterwards to rewrite the regrets).
"""

import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = glob.glob(os.path.join(ROOT, "results", "*problem_results.csv")) + \
    glob.glob(os.path.join(ROOT, "runs", "*", "results", "problem_results.csv"))
frames = []
for f in files:
    df = pd.read_csv(f)
    df = df[~df["status"].isin(["invalid", "crash"])]
    df = df[df["model"] != "continuous_beam"]  # real-valued reference, not a scoring system
    df["source_file"] = os.path.relpath(f, ROOT)
    frames.append(df)
table = pd.concat(frames)
idx = table.groupby(["suite", "dataset", "k"])["loss"].idxmin()
best = table.reset_index(drop=True).loc[table.reset_index(drop=True).groupby(["suite", "dataset", "k"])["loss"].idxmin()]
best = best[["suite", "dataset", "k", "loss", "model", "source_file"]].sort_values(["suite", "dataset", "k"])
best["loss"] = best["loss"].map(lambda v: f"{v:.8f}")
out = os.path.join(ROOT, "src", "best_known.csv")
best.to_csv(out, index=False)
print(f"{len(best)} problems -> {out}")
print(best.groupby("suite")["model"].value_counts().to_string())
