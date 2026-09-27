#!/usr/bin/env bash
# The baselines added after the first release of the post, on all three panels:
# visible and hidden (5 values of k, 60 s a problem) and the full-size hidden set (k = 5, 10 minutes).
set -uo pipefail
cd "$(dirname "$0")/.."
M=${1:-unit_weighting,autoscore,l1path_seqround,cpa_highs,abess_seqround,fastsparse_seqround,okridge_seqround,psl}
uv run run_baselines.py --models $M --jobs 14
uv run run_baselines.py --models $M --suite hidden --jobs 14
uv run run_baselines.py --models $M --suite hidden_full --ks 5 --time-limit 600 --jobs 27
