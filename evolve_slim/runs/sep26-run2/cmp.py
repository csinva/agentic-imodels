import sys, pandas as pd
d = pd.read_csv("results/problem_results.csv")
a, b = sys.argv[1], sys.argv[2]
A = d[d.model == a].set_index(["dataset", "k"]); B = d[d.model == b].set_index(["dataset", "k"])
print("regret", a, "->", b)
print((B.regret.unstack() * 1e4).round(1).to_string())
print("time", b); print(B.seconds.unstack().round(2).to_string())
print("regret sum w/o spambase", A.drop('spambase').regret.sum()/70, B.drop('spambase').regret.sum()/70)
print("spambase", A.loc['spambase'].regret.mean(), B.loc['spambase'].regret.mean())
