# vendored from github.com/jiachangliu/OKGLM (src/okglm) at commit 461e78a (2026-02-04), BSD 3-Clause, see LICENSE.
# The package is not on PyPI; its pyproject pins gurobipy, mosek, cvxpy and cupy, none of which the CPU
# branch and bound for logistic regression uses. evolve_slim patches (lines marked evolve_slim):
# - __init__.py no longer reads the installed version;
# - gurobipy, mosek and cvxpy are optional imports (prox_operators.py, baselines/baselines_solve_relaxation.py,
#   with `from __future__ import annotations` for the gurobipy type hints);
# - the GRB_LICENSE_FILE / MOSEKLM_LICENSE_FILE lines are set only when the license path variables exist;
# - BnBTree/unused_code/ removed.
