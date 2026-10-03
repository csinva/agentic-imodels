"""Set the status of a row in results/overall_results.csv. usage: uv run tools/status.py NAME keep|discard|crash"""
import sys, pandas as pd
p = __file__.rsplit("/tools/", 1)[0] + "/results/overall_results.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
assert (df.model_name == sys.argv[1]).sum() == 1
df.loc[df.model_name == sys.argv[1], "status"] = sys.argv[2]
df.to_csv(p, index=False)
print(df[df.model_name == sys.argv[1]].iloc[0, :10].to_dict())
