import sys, pandas as pd
pd.set_option('display.width',200)
d=pd.read_csv('results/problem_results.csv')
m=sys.argv[1]; d2=d[d.model==m].set_index(['dataset','k'])
if len(sys.argv)>2:
    b=d[d.model==sys.argv[2]].set_index(['dataset','k'])
    d2['base_regret']=b.regret; d2['base_sec']=b.seconds
    d2['dr']=d2.regret-d2.base_regret
    print(d2[['n_train','d','seconds','base_sec','regret','base_regret','dr']].sort_values('dr').to_string())
else:
    print(d2[['n_train','d','seconds','regret']].sort_values('regret').to_string())
