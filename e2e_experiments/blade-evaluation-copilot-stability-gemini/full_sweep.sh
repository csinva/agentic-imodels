#!/bin/bash
set -uo pipefail
cd /home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-copilot-stability-gemini

MODE="$1"
LOG_PREFIX="$2"
DATASETS=(affairs amtl boxes caschools crofoot fertility fish hurricane mortgage panda_nuts reading soccer teachingratings)

for run in run1 run2 run3; do
  for ds in "${DATASETS[@]}"; do
    out_dir="outputs_${MODE}_${run}"
    target="${out_dir}/${ds}/conclusion.txt"
    if [ -f "$target" ]; then
      echo "[$(date +%H:%M:%S)] SKIP  $MODE/$run/$ds (have conclusion)"
      continue
    fi
    echo "[$(date +%H:%M:%S)] start $MODE/$run/$ds"
    bash run_all.sh --mode "$MODE" --output-dir "$out_dir" "$ds" \
      > "logs_full/${LOG_PREFIX}_${run}_${ds}.log" 2>&1
    if [ -f "$target" ]; then
      echo "[$(date +%H:%M:%S)] DONE  $MODE/$run/$ds"
    else
      echo "[$(date +%H:%M:%S)] FAIL  $MODE/$run/$ds (timeout or other)"
    fi
  done
done
echo "[$(date +%H:%M:%S)] driver-$MODE finished"
