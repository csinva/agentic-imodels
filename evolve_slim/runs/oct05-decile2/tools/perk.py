"""Per-k and per-problem summary of model B (rows of problem_results.csv or scratch/<tag>.csv) vs references:
gap to best known (visible suite), dAUC vs v35, slowest problems. usage: uv run tools/perk.py B [ref=v35_scratch_cmp]"""
import os, sys, numpy as np, pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = pd.read_csv(f"{here}/results/problem_results.csv")
def rd(t):
    f = f"{here}/scratch/{t}.csv"
    d = pd.read_csv(f) if os.path.exists(f) else P[P.model == t]
    return d.set_index(["dataset", "k"])
b, r = rd(sys.argv[1]), rd(sys.argv[2] if len(sys.argv) > 2 else "v35_scratch_cmp")
bk = pd.read_csv(f"{here}/src/best_known.csv"); bk = bk[bk.suite == bk.suite.iloc[0] if "visible" not in set(bk.suite) else bk.suite == "visible"]
bk = bk.set_index(["dataset", "k"]).loss
d = pd.DataFrame({"loss": b.loss, "ref": r.loss, "bk": bk.reindex(b.index), "t": b.seconds, "t_ref": r.seconds,
                  "auc": b.auc_test, "auc_ref": r.auc_test})
d["gap"] = d.loss - d.bk; d["dl"] = d.loss - d.ref; d["dauc"] = d.auc - d.auc_ref
g = d.groupby(level=1)
print(pd.DataFrame({"gap": g.gap.mean(), "dl_ref": g.dl.mean(), "dauc_ref": g.dauc.mean(),
                    "geo_t": g.t.apply(lambda s: np.exp(np.log(s).mean())), "geo_tref": g.t_ref.apply(lambda s: np.exp(np.log(s).mean()))}).round(6))
print(d.sort_values("gap", ascending=False).head(10)[["loss", "ref", "gap", "dl", "dauc"]].round(5))
print(d.sort_values("t", ascending=False).head(8)[["t", "t_ref"]].round(4))
print("dauc by dataset", d.groupby(level=0).dauc.mean().round(4).to_dict())
