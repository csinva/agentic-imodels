#!/usr/bin/env bash
# Builds the R environment of the R-package baselines (riskscores, riskscores_cd, l0learn_seqround):
# micromamba, R 4.4 from conda-forge, then riskscores 1.3.0 and L0Learn 2.1.0 from CRAN.
# Paths can be changed with R_ENV / TOOLS; src/baselines.py reads EVOLVE_SLIM_R_ENV (default below).
set -euo pipefail
TOOLS=${TOOLS:-/data15/chandan/tabular/tools}
R_ENV=${R_ENV:-/data15/chandan/tabular/.r-env}
mkdir -p "$TOOLS"
if [ ! -x "$TOOLS/bin/micromamba" ]; then
  (cd "$TOOLS" && curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba)
fi
export MAMBA_ROOT_PREFIX="$TOOLS/mamba-root"
"$TOOLS/bin/micromamba" create -y -p "$R_ENV" -c conda-forge --no-rc \
  r-base=4.4 r-dplyr r-foreach r-ggplot2 r-magrittr r-proc r-prroc
"$R_ENV/bin/Rscript" -e 'install.packages(c("riskscores", "L0Learn"), repos = "https://cloud.r-project.org", lib = .Library, Ncpus = 8)'
"$R_ENV/bin/Rscript" -e 'cat(format(packageVersion("riskscores")), format(packageVersion("L0Learn")), "\n")'
