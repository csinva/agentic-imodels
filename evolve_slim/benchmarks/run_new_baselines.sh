#!/usr/bin/env bash
# The baselines added on 2026-10-03 (riskscores in R, skscope, L0Learn in R, OKGLM), on all three panels:
# visible and hidden (5 values of k, 60 s a problem) and the full-size hidden set (k = 5, 10 minutes).
# riskscores, riskscores_cd and l0learn_seqround need the R environment of baselines/rpkgs/setup_r.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
M=${1:-riskscores,riskscores_cd,skscope_seqround,l0learn_seqround,okglm_seqround}
uv run run_baselines.py --models $M --jobs 14
uv run run_baselines.py --models $M --suite hidden --jobs 14
uv run run_baselines.py --models $M --suite hidden_full --ks 5 --time-limit 600 --jobs 27
