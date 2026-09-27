#!/bin/bash
# usage: tools/envab.sh "VAR=val ..." : A/B of slim.py with env defaults changed vs slim.py
cd /home/chandan.singh/tabular/agentic-imodels/evolve_slim/runs/sep27-fine2
f=scratch/env_$(echo "$1" | tr ' =' '__').py
cp slim.py $f
for kv in $1; do k=${kv%%=*}; v=${kv#*=}; sed -i "s/os.environ.get(\"$k\", \"[^\"]*\")/os.environ.get(\"$k\", \"$v\")/" $f; done
uv run python tools/ab.py slim.py $f ${2:-3,4,5,7,10} > $f.txt 2>&1; echo "$1"; tail -3 $f.txt
