"""Helpers: set status of a row, compare per-problem results of models."""
import sys, csv
import pandas as pd

def status(model, st):
    rows = list(csv.reader(open('results/overall_results.csv')))
    h = rows[0]
    for r in rows[1:]:
        if r[h.index('model_name')] == model:
            r[h.index('status')] = st
    csv.writer(open('results/overall_results.csv', 'w', newline='')).writerows(rows)

def cmp(a, b):
    d = pd.read_csv('results/problem_results.csv')
    A = d[d.model == a].set_index(['dataset', 'k'])
    B = d[d.model == b].set_index(['dataset', 'k'])
    m = A[['d', 'seconds', 'loss', 'auc_test']].join(B[['seconds', 'loss', 'auc_test']], rsuffix='_b')
    m['dloss'] = (m.loss_b - m.loss) * 1e4
    m['dauc'] = (m.auc_test_b - m.auc_test) * 1e2
    pd.set_option('display.width', 200)
    print(m[['d', 'seconds', 'seconds_b', 'dloss', 'dauc']].round(3).to_string())
    print('mean dloss(e-4)', m.dloss.mean(), 'wins', (m.dloss < -0.1).sum(), 'losses', (m.dloss > 0.1).sum(),
          'mean dauc(%)', m.dauc.mean())


RUG = {('heart', 7), ('heart', 10), ('ilpd', 7), ('ilpd', 10), ('ionosphere', 4), ('ionosphere', 5), ('ionosphere', 7),
       ('ionosphere', 10), ('mushroom', 5), ('mushroom', 7), ('mushroom', 10), ('magic', 4), ('magic', 7),
       ('breastcancer', 10), ('australian', 10)}

def split(models):
    d = pd.read_csv('results/problem_results.csv')
    P = d[d.model.isin(models)].pivot_table(index=['dataset', 'k'], columns='model', values='regret') * 1e4
    rug = P.index.isin(list(RUG))
    print('rugged (15) mean e-4:', P[rug].mean().round(2).to_dict())
    print('stable (55) mean e-4:', P[~rug].mean().round(2).to_dict())

if __name__ == '__main__':
    if sys.argv[1] == 'status':
        status(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'split':
        split(sys.argv[2:])
    else:
        cmp(sys.argv[2], sys.argv[3])
