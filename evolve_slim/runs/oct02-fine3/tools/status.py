"""Set the status of a model row: uv run python tools/status.py MODEL keep|discard|crash"""
import sys, csv
m, s = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open('results/overall_results.csv')))
for r in rows:
    if r['model_name'] == m:
        r['status'] = s
with open('results/overall_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print([ (r['model_name'], r['mean_regret'], r['geo_mean_time'], r['mean_test_auc'], r['status']) for r in rows if r['model_name']==m])
