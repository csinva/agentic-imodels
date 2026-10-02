"""Per-dataset AUC / loss of runs (scratch/<tag>.csv or a model in results/problem_results.csv).
uv run python tools/cmp.py TAG1 TAG2 ..."""
import os, sys
import pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = pd.read_csv(os.path.join(here, "results", "problem_results.csv"))
fr = []
for t in sys.argv[1:]:
    p = os.path.join(here, "scratch", t + ".csv")
    d = pd.read_csv(p) if os.path.exists(p) else res[res.model == t]
    d = d.assign(model=t); fr.append(d)
df = pd.concat(fr)
pd.set_option("display.width", 250)
print(df.pivot_table(index="dataset", columns="model", values="auc_test", sort=False)[sys.argv[1:]].round(4).to_string())
print(df.pivot_table(index="k", columns="model", values="auc_test")[sys.argv[1:]].round(4).to_string())
print(df.groupby("model")[["regret", "auc_test", "loss"]].mean().loc[sys.argv[1:]].round(5).to_string())
