#!/usr/bin/env bash
# Fetch RiskSLIM (github.com/ustunb/risk-slim) into baselines/risk-slim (untracked),
# make it run on numpy 2, and compile its Cython loss functions in place.
# Needs the `baselines` dependency group (cplex Community Edition, cython, prettytable).
set -euo pipefail
cd "$(dirname "$0")"
[ -d risk-slim ] || git clone --depth 1 https://github.com/ustunb/risk-slim risk-slim
grep -rl "np\.float_\b" risk-slim/riskslim --include=*.py | xargs -r sed -i 's/np\.float_\b/np.float64/g'
cd risk-slim/riskslim/loss_functions
sed -i 's/scipy.get_include()/numpy.get_include()/g' build_cython_loss_functions.py
uv run --project ../../../.. python build_cython_loss_functions.py build_ext --inplace
