#!/usr/bin/env bash
# After benchmarks/run_hidden.sh: the slow reference for the hidden best known losses, the full-size
# hidden panel (every row, k = 5, 10 minutes a problem, one dataset per worker), and repeats.
set -uo pipefail
cd "$(dirname "$0")/.."
SHIP=runs/sep26-run2/slim_lib/v35_scratch.py
uv run run_baselines.py --models fasterrisk_wide --suite hidden --time-limit 1200 --jobs 35
uv run benchmarks/eval_solver.py $SHIP --suite hidden_full --ks 5 --time-limit 600 --jobs 27
for m in fasterrisk rounded_lr imodels_slim continuous_beam riskslim slim_milp; do
  uv run run_baselines.py --models $m --suite hidden_full --ks 5 --time-limit 600 --jobs 27
done
for rep in 2 3; do
  uv run benchmarks/eval_solver.py $SHIP --suite visible --tag _rep$rep
  uv run benchmarks/eval_solver.py $SHIP --suite hidden --tag _run2_rep$rep
  uv run run_baselines.py --models fasterrisk --tag _rep$rep
  uv run run_baselines.py --models fasterrisk --suite hidden --tag _rep$rep
done
