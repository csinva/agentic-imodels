import csv, sys
m = sys.argv[1]
r=[x for x in csv.DictReader(open('results/problem_results.csv')) if x['suite']=='visible' and x['model']==m]
ks=['3','4','5','7','10']
ds=sorted(set(x['dataset'] for x in r), key=lambda d: [x['dataset'] for x in r].index(d))
print('dataset'.ljust(13)+''.join(k.rjust(9) for k in ks))
for d in ds:
    row=d.ljust(13)
    for k in ks:
        x=[z for z in r if z['dataset']==d and z['k']==k]
        if not x: row+=' '*9; continue
        x=x[0]; row += (('C' if x['certified']=='True' else '.')+f"{float(x['seconds']):.1f}").rjust(9)
    print(row)
print('certified by k:', {k: sum(1 for x in r if x['k']==k and x['certified']=='True') for k in ks})
