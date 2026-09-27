#!/bin/sh
# usage: ./seeds.sh file.py [seeds...] -- extra unrecorded suite runs with other kick seeds, run in parallel
cd "$(dirname "$0")"
f=$1; shift
tag=$(basename $f .py)
for s in "$@"; do
  ( cp $f _st_${tag}_$s.py
    SLIM_SEED=$s uv run _st_${tag}_$s.py --no-record > _st_${tag}_$s.log 2>&1
    echo "$tag seed $s: $(grep -E 'mean_regret|geo_mean_time|mean_test_auc' _st_${tag}_$s.log | awk '{print $2}' | tr '\n' ' ')"
    rm -f _st_${tag}_$s.py _st_${tag}_$s.log ) &
done
wait
