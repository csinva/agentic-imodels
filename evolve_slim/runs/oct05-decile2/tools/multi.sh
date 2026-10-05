#!/bin/bash
# usage: tools/multi.sh REF FILE "TAG ENV=.. ENV=.." ...   dev runs of FILE, two at a time, each compared with REF and v35
cd /data15/chandan/tabular/agentic-imodels/evolve_slim/runs/oct05-decile2
export UV_CACHE_DIR=/data15/chandan/tabular/.uv-cache
ref=$1; file=$2; shift 2
one() { spec="$1"; set -- $spec; tag=$1; shift
  out=$(env "$@" uv run tools/dev.py $file $tag 2>&1 | tail -1; uv run tools/cnt2.py $ref $tag | head -14; uv run tools/cnt2.py v35_scratch_cmp $tag | head -1 | sed 's/^/  vs v35: /')
  echo "### $spec"; echo "$out"; }
while [ $# -gt 0 ]; do
  a="$1"; b="$2"; shift; [ $# -gt 0 ] && shift
  one "$a" > scratch/_m1.txt & p1=$!
  if [ -n "$b" ]; then one "$b" > scratch/_m2.txt & p2=$!; wait $p2; fi
  wait $p1; cat scratch/_m1.txt; [ -n "$b" ] && cat scratch/_m2.txt; rm -f scratch/_m1.txt scratch/_m2.txt
done
