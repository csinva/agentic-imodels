import csv, sys
p = 'results/overall_results.csv'
rows = list(csv.DictReader(open(p)))
for r in rows:
    if r['model_name'] == sys.argv[1]:
        r['status'] = sys.argv[2]
w = csv.DictWriter(open(p, 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
