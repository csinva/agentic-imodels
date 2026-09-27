"""Per-problem comparison of models in results/problem_results.csv: uv run python cmp.py A B"""
import sys, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
d = pd.read_csv('results/problem_results.csv')
ms = sys.argv[1:]
cols = []
base = None
for m in ms:
    f = d[d.model == m].drop_duplicates(['dataset', 'k'], keep='last').set_index(['dataset', 'k'])
    if base is None:
        base = f[['n_train', 'd']].copy()
    base[m[:10] + '_reg'] = f.regret * 1e3
    base[m[:10] + '_t'] = f.seconds
    base[m[:10] + '_auc'] = f.auc_test
print(base.round(3).to_string())
print(base.mean(numeric_only=True).round(4).to_string())
