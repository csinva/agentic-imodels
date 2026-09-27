#!/usr/bin/env bash
# Held-out suite (27 TabArena datasets x 5 values of k, 60 s a problem): every kept version of both
# loop sessions, the two end-of-run safety versions, and the baselines. One solver at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
for r in sep26-run1 sep26-run2; do
  tag="_${r#sep26-}"
  files=$(awk -F, -v r="runs/$r/slim_lib/" 'NR>1 && $10=="keep"{print r $11 ".py"}' runs/$r/results/overall_results.csv)
  [ "$r" = sep26-run1 ] && files="$files runs/$r/slim_lib/v36_robust.py"
  [ "$r" = sep26-run2 ] && files="$(echo $files | sed 's#runs/sep26-run2/slim_lib/pyfasterrisk_v1.py##') runs/$r/slim_lib/v35_scratch.py"
  uv run benchmarks/eval_solver.py $files --suite hidden --tag "$tag" --jobs 14
done
uv run run_baselines.py --suite hidden --jobs 14
