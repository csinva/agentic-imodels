"""Baseline solvers, each as ``make_baseline(name) -> make_model(k, time_limit)``.

Every model's ``fit(X, y)`` takes y in {0, 1} and leaves integer points in
``coef_`` (except the real-valued reference). The harness ignores intercepts and
multipliers and fits the score-to-risk map itself.

fasterrisk        FasterRisk (Liu et al., NeurIPS 2022), the published package
                  (vendored in baselines/fasterrisk), default settings.
riskslim          RiskSLIM, lattice cutting planes (Ustun & Rudin, JMLR 2019), with
                  the CPLEX Community Edition and the solver's time limit.
slim_milp         SLIM (Ustun & Rudin, 2016): 0-1 loss with integer coefficients,
                  solved as a MILP by HiGHS (scipy.optimize.milp) within the time limit.
rounded_lr        The common practice: L1 logistic regression tuned to k features,
                  refit, scaled so the largest point is 5, and rounded.
imodels_slim      SLIMClassifier of imodels 3.0.2 without a MIP solver, which rounds
                  an L2 logistic regression; the penalty is searched so at most k
                  points are nonzero, and points are clipped to [-5, 5].
fasterrisk_wide   FasterRisk with every search width raised (beam 50 x 50, pool 200,
                  100 swaps, 100 multipliers); a slow reference for the best known losses.
continuous_beam   Reference, not integer: FasterRisk's beam search before rounding,
                  the k-sparse real-valued model it rounds from.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
BASELINES_DIR = os.path.join(os.path.dirname(HERE), "baselines")
for p in (BASELINES_DIR, os.path.join(BASELINES_DIR, "risk-slim")):
    if p not in sys.path:
        sys.path.insert(0, p)

COEF_BOUND = 5


class _Base:
    def __init__(self, k, time_limit):
        self.k, self.time_limit = k, time_limit
        self.stop_reason_ = ""


class FasterRisk(_Base):
    def fit(self, X, y):
        from fasterrisk.fasterrisk import RiskScoreOptimizer
        opt = RiskScoreOptimizer(X=X, y=np.where(y > 0, 1.0, -1.0), k=self.k, lb=-COEF_BOUND, ub=COEF_BOUND)
        opt.optimize()
        mult, b0, betas = opt.get_models(0)
        self.coef_, self.intercept_, self.multiplier_ = np.round(betas), float(b0), float(mult)
        return self


class FasterRiskWide(_Base):
    """FasterRisk with every search width raised: a slow reference for the best known losses."""

    def fit(self, X, y):
        from fasterrisk.fasterrisk import RiskScoreOptimizer
        opt = RiskScoreOptimizer(X=X, y=np.where(y > 0, 1.0, -1.0), k=self.k, lb=-COEF_BOUND, ub=COEF_BOUND,
                                 parent_size=50, child_size=50, select_top_m=200, maxAttempts=100,
                                 num_ray_search=100, gap_tolerance=0.1)
        opt.optimize()
        mult, b0, betas = opt.get_models(0)
        self.coef_, self.intercept_, self.multiplier_ = np.round(betas), float(b0), float(mult)
        return self


class ContinuousBeam(_Base):
    def fit(self, X, y):
        from fasterrisk.sparseBeamSearch import sparseLogRegModel
        m = sparseLogRegModel(X, np.where(y > 0, 1.0, -1.0), intercept=True, original_lb=-COEF_BOUND,
                              original_ub=COEF_BOUND)
        m.get_sparse_sol_via_OMP(k=self.k, parent_size=10, child_size=10)
        b0, betas = m.get_original_beta0_betas()
        self.coef_, self.intercept_ = np.asarray(betas, float), float(b0)
        return self


class RiskSLIM(_Base):
    def fit(self, X, y):
        import riskslim
        names = ["(Intercept)"] + [f"x{j}" for j in range(X.shape[1])]
        data = {"X": np.hstack([np.ones((X.shape[0], 1)), X]), "Y": np.where(y > 0, 1, -1).reshape(-1, 1),
                "variable_names": names, "outcome_name": "y", "sample_weights": np.ones(X.shape[0])}
        coef_set = riskslim.CoefficientSet(variable_names=names, lb=-COEF_BOUND, ub=COEF_BOUND, sign=0)
        coef_set.update_intercept_bounds(X=data["X"], y=data["Y"], max_offset=50)
        settings = {"c0_value": 1e-6, "max_runtime": float(self.time_limit), "max_tolerance": np.finfo(float).eps,
                    "display_cplex_progress": False, "loss_computation": "fast", "round_flag": True,
                    "polish_flag": True, "chained_updates_flag": True, "add_cuts_at_heuristic_solutions": True,
                    "initialization_flag": True, "init_max_runtime": 0.25 * float(self.time_limit),
                    "init_max_coefficient_gap": 0.49, "cplex_randomseed": 0, "cplex_mipemphasis": 0}
        constraints = {"L0_min": 0, "L0_max": min(self.k, X.shape[1]), "coef_set": coef_set}
        try:
            model_info, _, _ = riskslim.run_lattice_cpa(data, constraints, settings)
        except Exception as exc:  # noqa: BLE001
            if "1016" not in str(exc):
                raise
            self.no_model_, self.stop_reason_ = True, "CPLEX Community Edition size limit"
            self.coef_ = np.zeros(X.shape[1])
            return self
        rho = np.asarray(model_info["solution"], float)
        self.coef_, self.intercept_ = np.round(rho[1:]), float(rho[0])
        self.optimality_gap_ = float(model_info.get("optimality_gap", np.nan))
        self.stop_reason_ = f"gap={self.optimality_gap_:.3g}"
        return self


class SlimMILP(_Base):
    """min (1/n) sum_i err_i + c0 * |support| + eps * sum_j |w_j|  s.t. |support| <= k,
    err_i >= 1 whenever y_i (w0 + x_i w) < margin (big-M), w_j integer in [-5, 5].
    Duplicate rows are merged with weights, which leaves the problem unchanged."""

    def fit(self, X, y):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix, hstack as sphstack, identity, vstack as spvstack
        t0 = time.perf_counter()
        s = np.where(y > 0, 1.0, -1.0)
        rows, weights = np.unique(np.hstack([X, s[:, None]]), axis=0, return_counts=True)
        Xu, su = rows[:, :-1], rows[:, -1]
        m, d = Xu.shape
        n = float(len(y))
        margin, w0max = 0.5, COEF_BOUND * max(1, self.k) + 1.0
        # variables: w0 (1), w (d), a (d, support), p (d, |w|), z (m, errors)
        nv = 1 + 3 * d + m
        big_m = margin + w0max + COEF_BOUND * np.sort(np.abs(Xu), axis=1)[:, ::-1][:, :self.k].sum(1)
        c = np.zeros(nv)
        c[1 + d:1 + 2 * d] = 1e-6
        c[1 + 2 * d:1 + 3 * d] = 1e-8
        c[1 + 3 * d:] = weights / n
        yx = csr_matrix(su[:, None] * Xu)
        # y_i (w0 + x_i w) + M_i z_i >= margin
        A1 = sphstack([csr_matrix(su[:, None]), yx, csr_matrix((m, 2 * d)), csr_matrix(np.diag(big_m))])
        I = identity(d, format="csr")
        Z = csr_matrix((d, d))
        zc = csr_matrix((d, m))
        one = csr_matrix((d, 1))
        A2 = sphstack([one, I, -COEF_BOUND * I, Z, zc])     # w - 5a <= 0
        A3 = sphstack([one, I, COEF_BOUND * I, Z, zc])      # w + 5a >= 0
        A4 = sphstack([one, I, Z, -I, zc])                  # w - p <= 0
        A5 = sphstack([one, I, Z, I, zc])                   # w + p >= 0
        A6 = csr_matrix(np.concatenate([[0.0], np.zeros(d), np.ones(d), np.zeros(d + m)])[None, :])
        A = spvstack([A1, A2, A3, A4, A5, A6]).tocsr()
        lo = np.concatenate([np.full(m, margin), np.full(d, -np.inf), np.zeros(d), np.full(d, -np.inf), np.zeros(d),
                             [0.0]])
        hi = np.concatenate([np.full(m, np.inf), np.zeros(d), np.full(d, np.inf), np.zeros(d), np.full(d, np.inf),
                             [float(self.k)]])
        lb = np.concatenate([[-w0max], np.full(d, -COEF_BOUND), np.zeros(d), np.zeros(d), np.zeros(m)])
        ub = np.concatenate([[w0max], np.full(d, COEF_BOUND), np.ones(d), np.full(d, COEF_BOUND), np.ones(m)])
        integrality = np.concatenate([[1], np.ones(d), np.ones(d), np.zeros(d), np.ones(m)])
        left = max(1.0, self.time_limit - (time.perf_counter() - t0))
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=Bounds(lb, ub),
                   options={"time_limit": left, "disp": False})
        if res.x is None:
            self.coef_, self.intercept_ = np.zeros(d), 0.0
            self.stop_reason_ = f"no incumbent ({res.message[:40]})"
            return self
        self.coef_ = np.round(res.x[1:1 + d])
        self.intercept_ = float(np.round(res.x[0]))
        self.stop_reason_ = "optimal" if res.status == 0 else f"status {res.status}"
        return self


def _l1_support(X, y, k):
    """Largest L1 logistic penalty path point with at most k nonzero coefficients."""
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in np.logspace(-4, 2, 40):
        lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit(X, y)
        nz = np.flatnonzero(np.abs(lr.coef_[0]) > 1e-8)
        if len(nz) > k:
            break
        best = nz
    return best if best is not None else np.array([], dtype=int)


class RoundedLR(_Base):
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        support = _l1_support(X, y, self.k)
        coef = np.zeros(X.shape[1])
        if len(support):
            lr = LogisticRegression(C=1e4, max_iter=5000).fit(X[:, support], y)
            w = lr.coef_[0]
            coef[support] = np.round(w * COEF_BOUND / np.max(np.abs(w)))
        self.coef_ = coef
        return self


class ImodelsSLIM(_Base):
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        def rounded(alpha):
            return np.round(LogisticRegression(C=1 / alpha, max_iter=5000).fit(X, y).coef_[0])

        lo, hi = -4.0, 4.0  # log10 alpha; larger alpha shrinks more points to zero
        best = np.zeros(X.shape[1])
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            c = rounded(10 ** mid)
            if np.count_nonzero(c) <= self.k:
                best, hi = c, mid
            else:
                lo = mid
        self.coef_ = np.clip(best, -COEF_BOUND, COEF_BOUND)
        return self


BASELINES = {"fasterrisk": FasterRisk, "fasterrisk_wide": FasterRiskWide, "riskslim": RiskSLIM, "slim_milp": SlimMILP, "rounded_lr": RoundedLR,
             "imodels_slim": ImodelsSLIM, "continuous_beam": ContinuousBeam}
INTEGER = {name: name != "continuous_beam" for name in BASELINES}


def make_baseline(name):
    cls = BASELINES[name]
    return lambda k, time_limit: cls(k, time_limit)
