"""Adapter: the solver shipped in imodels (imodels/algebraic/risk_score/solver.py, a port of
runs/oct05-decile2/slim_lib/n17_lean.py) as a harness solver file.

    RS_PROFILE=decile uv run benchmarks/eval_solver.py benchmarks/imodels_solver.py --suite hidden --tag _pkg
    RS_PROFILE=fine uv run benchmarks/eval_solver.py benchmarks/imodels_solver.py --suite hidden_fine --tag _pkg

The module is loaded from its file (it needs only numpy and numba), so the imodels package itself need not be
installed in this environment. Its kernels are compiled (or loaded from numba's disk cache) at the first solve(),
which the harness's untimed warm-up fit triggers in every worker.
"""

import importlib.util
import os
import sys

import numpy as np

IMODELS_SOLVER = os.environ.get(
    "IMODELS_SOLVER", "/data15/chandan/tabular/imodels/imodels/algebraic/risk_score/solver.py")
PROFILE = os.environ.get("RS_PROFILE", "decile")
COEF_BOUND = 5  # largest point; benchmarks/exhaustive_check.py sets it to 3, as for the run snapshots

_spec = importlib.util.spec_from_file_location("imodels_risk_score_solver", IMODELS_SOLVER)
solver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = solver  # numba's disk cache re-imports the module by name
_spec.loader.exec_module(solver)


class PackageSolver:
    def __init__(self, k, time_limit):
        self.k, self.time_limit = k, time_limit

    def fit(self, X, y):
        points, loss, seconds, stopped = solver.solve(X, y, self.k, COEF_BOUND, self.time_limit, profile=PROFILE)
        self.coef_ = np.asarray(points, dtype=float)
        self.train_loss_, self.timing_, self.stopped_early_ = loss, seconds, stopped
        return self


def make_model(k, time_limit):
    return PackageSolver(k, time_limit)
