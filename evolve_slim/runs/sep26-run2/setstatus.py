import sys, pandas as pd
p = "results/overall_results.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
df.loc[df.model_name == sys.argv[1], "status"] = sys.argv[2]
df.to_csv(p, index=False)
print(df[["mean_regret","geo_mean_time","mean_test_auc","n_invalid","status","model_name"]].to_string())
