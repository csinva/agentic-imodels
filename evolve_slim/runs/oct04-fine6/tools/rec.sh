#!/bin/bash
# usage: tools/rec.sh file.py NAME "DESC" status   -- record a variant file as an attempt, set its status, restore slim.py
cd "$(dirname "$0")/.." && export UV_CACHE_DIR=/data15/chandan/tabular/.uv-cache
cp slim.py scratch/_slim_backup.py && cp "$1" slim.py
uv run tools/attempt.py "$2" "$3" 2>&1 | grep -E "^(model|mean_regret|geo_mean|mean_test|mean_loss|n_invalid)"
uv run tools/status.py "$2" "$4" > /dev/null
cp scratch/_slim_backup.py slim.py
