#!/bin/bash
# usage: tools/sweep.sh FILE "ENV=.. tag" "ENV=.. tag" ...  (runs in parallel, waits, prints summaries)
export UV_CACHE_DIR=/data15/chandan/tabular/.uv-cache
cd "$(dirname "$0")/.."
f=$1; shift
for cfg in "$@"; do
  tag=${cfg##* }; envs=${cfg% *}; [ "$envs" = "$tag" ] && envs=""
  env $envs uv run python tools/dev.py $f $tag > scratch/$tag.log 2>&1 &
done
wait
for cfg in "$@"; do tag=${cfg##* }; tail -n 1 scratch/$tag.log; done
