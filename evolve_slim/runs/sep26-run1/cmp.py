import sys, pandas as pd
pd.set_option('display.width', 200)
a = pd.read_csv(sys.argv[1]).set_index(['dataset', 'k']); b = pd.read_csv(sys.argv[2]).set_index(['dataset', 'k'])
d = pd.DataFrame({'sec_a': a.seconds, 'sec_b': b.seconds, 'reg_a': a.regret, 'reg_b': b.regret})
d['dr'] = d.reg_b - d.reg_a
print(d.sort_values('dr').to_string(float_format=lambda x: f'{x:.5f}'))
