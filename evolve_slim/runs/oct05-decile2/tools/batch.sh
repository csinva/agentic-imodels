#!/bin/bash
# usage: tools/batch.sh REF "TAG ENV=.. ENV=.." ...   (dev runs of slim.py, each compared with REF by cnt2.py)
cd /data15/chandan/tabular/agentic-imodels/evolve_slim/runs/oct05-decile2
export UV_CACHE_DIR=/data15/chandan/tabular/.uv-cache
ref=$1; shift
for spec in "$@"; do
  set -- $spec; tag=$1; shift
  env "$@" uv run tools/dev.py ${FILE:-slim.py} $tag 2>&1 | tail -1
  uv run tools/cnt2.py $ref $tag | head -12
  uv run tools/cnt2.py v35_scratch_cmp $tag | head -1 | sed 's/^/  vs v35: /'
done
