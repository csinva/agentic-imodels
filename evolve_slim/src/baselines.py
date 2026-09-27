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
unit_weighting    Unit weighting: the L1-selected features, each worth +1 or -1.
autoscore         AutoScore (Xie et al. 2020) adapted to binary features: random-forest
                  ranking, logistic regression, points = coefficients / smallest, rounded.
l1path_seqround   FasterRisk's star-ray sequential rounding applied to every support on
                  the L1 logistic path (its rounding stage without its beam search).
cpa_highs         RiskSLIM's problem by its cutting-plane algorithm with the open-source
                  HiGHS MILP solver (re-solved each round); proves optimality on small problems.
abess_seqround    abess best-subset logistic regression (sizes 1..k), rounded as above.
fastsparse_seqround  fastSparse / L0Learn L0L2 logistic path (<= k), rounded as above.
okridge_seqround  OKRidge's optimal k-sparse ridge support (squared-loss proxy), logistic
                  refit, rounded as above.
psl               Probabilistic scoring lists (scikit-psl, vendored), k greedy stages.
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


def _calibrated_loss(X, y, w):
    """The harness's criterion for points w (imported lazily to keep this module standalone)."""
    from evaluate import calibrate
    return calibrate(X @ w, y)[2]


def _l1_path_supports(X, y, k, n_c=40):
    """Distinct supports of size 1..k along the L1 logistic regularization path (liblinear)."""
    from sklearn.linear_model import LogisticRegression
    out, seen = [], set()
    for C in np.logspace(-4, 2, n_c):
        lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000).fit(X, y)
        nz = np.flatnonzero(np.abs(lr.coef_[0]) > 1e-8)
        if len(nz) > k:
            break
        if len(nz) and tuple(nz) not in seen:
            seen.add(tuple(nz))
            out.append(nz)
    return out


def _star_ray_round(X, y, b0, beta, ray=None):
    """Round a real-valued solution (intercept b0, coefficients beta) to integer points with
    FasterRisk's star-ray search and sequential rounding, after scaling it into the point box as
    FasterRisk's own bounded fits guarantee. Returns the points (intercept dropped)."""
    from fasterrisk.rounding import starRaySearchModel
    if ray is None:
        ray = starRaySearchModel(X=X, y=np.where(y > 0, 1.0, -1.0), lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20)
    full = np.concatenate([[b0], beta]).astype(float)
    top = np.max(np.abs(full[1:]))
    if top == 0:
        return np.zeros(len(beta))
    if top > COEF_BOUND:
        full *= COEF_BOUND / top
    _, sol = ray.line_search_scale_and_round(full)
    return np.clip(np.round(sol[1:]), -COEF_BOUND, COEF_BOUND)


def _best_of_pool(X, y, candidates):
    """The candidate points with the lowest calibrated loss (zeros if none has a nonzero point)."""
    best, best_l = np.zeros(X.shape[1]), np.inf
    for w in candidates:
        if not np.any(w):
            continue
        l = _calibrated_loss(X, y, w)
        if l < best_l:
            best, best_l = w, l
    return best


def _logistic_refit(X, y, S):
    """Unpenalized-ish logistic fit on the columns S: (intercept, full coefficient vector)."""
    from sklearn.linear_model import LogisticRegression
    beta = np.zeros(X.shape[1])
    if len(S) == 0:
        return 0.0, beta
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(X[:, S], y)
    beta[S] = lr.coef_[0]
    return float(lr.intercept_[0]), beta


class UnitWeighting(_Base):
    """Unit weighting (Burgess 1928; a baseline in the FasterRisk paper): the features of the
    L1 path with at most k nonzero coefficients, each worth +1 or -1 point by its sign."""

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        support = _l1_support(X, y, self.k)
        coef = np.zeros(X.shape[1])
        if len(support):
            w = LogisticRegression(C=1e4, max_iter=5000).fit(X[:, support], y).coef_[0]
            coef[support] = np.sign(w)
        self.coef_ = coef
        return self


class AutoScore(_Base):
    """AutoScore (Xie et al., JMIR Med Inform 2020), adapted to binary features: rank features by
    random-forest importance, keep the top k, fit a logistic regression, divide the coefficients by
    the smallest in absolute value and round (AutoScore's points rule), then shrink to the point
    bound if the largest exceeds it. AutoScore's own quantile binning is replaced by the suite's
    binarization, and its parsimony plot by the fixed k."""

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1).fit(X, y)
        top = np.argsort(-rf.feature_importances_, kind="stable")[:self.k]
        top = top[rf.feature_importances_[top] > 0]
        coef = np.zeros(X.shape[1])
        if len(top):
            w = LogisticRegression(C=1e4, max_iter=5000).fit(X[:, top], y).coef_[0]
            nz = np.abs(w) > 1e-8
            if nz.any():
                pts = w / np.min(np.abs(w[nz]))
                if np.max(np.abs(pts)) > COEF_BOUND:
                    pts = pts * COEF_BOUND / np.max(np.abs(pts))
                coef[top] = np.round(pts)
        self.coef_ = coef
        return self


class L1PathSeqRound(_Base):
    """FasterRisk's rounding stage without its beam search: every support of size <= k on the L1
    logistic path is refit (bounded coefficients, intercept) and rounded by FasterRisk's star-ray
    search with sequential rounding; the rounding with the lowest calibrated loss is kept. The
    ElasticNet-pool-plus-rounding baselines of the FasterRisk paper are of this kind."""

    def fit(self, X, y):
        from fasterrisk.rounding import starRaySearchModel
        ray = starRaySearchModel(X=X, y=np.where(y > 0, 1.0, -1.0), lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20)
        pool = [_star_ray_round(X, y, *_logistic_refit(X, y, S), ray=ray) for S in _l1_path_supports(X, y, self.k)]
        self.coef_ = _best_of_pool(X, y, pool)
        return self

class CuttingPlaneHiGHS(_Base):
    """RiskSLIM's problem solved by its cutting-plane algorithm (CPA) with the open-source HiGHS
    MILP solver: minimise the mean logistic loss of w0 + x.w over integer w in [-5, 5]^d with at
    most k nonzero and an integer intercept in [-50, 50]. Each iteration solves a MILP whose
    objective is the maximum of the linear cuts collected so far (a lower bound on the loss),
    evaluates the true loss at its solution (an upper bound, and a new cut there), and stops when
    the two meet or time runs out. Unlike RiskSLIM's lattice CPA the MILP is re-solved from scratch
    each round (HiGHS has no lazy-constraint callbacks), so it proves optimality only on small
    problems. Rows are merged into unique (x, y) with counts. As in RiskSLIM's initialization, the
    search starts from rounded heuristic solutions (here the L1 path rounded by FasterRisk's star-ray
    search), which give the first incumbent and cuts."""

    def fit(self, X, y):
        from scipy.optimize import Bounds, LinearConstraint, milp
        t0 = time.perf_counter()
        s = np.where(y > 0, 1.0, -1.0)
        rows, cnt = np.unique(np.hstack([X, s[:, None]]), axis=0, return_counts=True)
        Z = rows[:, -1:] * np.hstack([np.ones((len(rows), 1)), rows[:, :-1]])  # y * [1, x]
        c = cnt / cnt.sum()
        d = X.shape[1]
        p = d + 1  # rho = (w0, w)

        def loss_grad(rho):
            m = Z @ rho
            loss = float(c @ np.logaddexp(0, -m))
            g = -(Z.T @ (c / (1 + np.exp(m))))
            return loss, g

        # variables: rho (p, integer), alpha (d, binary support), theta (1, continuous)
        nv = p + d + 1
        lb = np.concatenate([[-50.0], np.full(d, -COEF_BOUND), np.zeros(d), [0.0]])
        ub = np.concatenate([[50.0], np.full(d, COEF_BOUND), np.ones(d), [np.inf]])
        integ = np.concatenate([np.ones(p), np.ones(d), [0]])
        cost = np.zeros(nv); cost[-1] = 1.0
        link_rows, link_lo, link_hi = [], [], []
        for j in range(d):  # |w_j| <= 5 alpha_j
            r = np.zeros(nv); r[1 + j] = 1; r[p + j] = -COEF_BOUND
            link_rows.append(r); link_lo.append(-np.inf); link_hi.append(0.0)
            r = np.zeros(nv); r[1 + j] = 1; r[p + j] = COEF_BOUND
            link_rows.append(r); link_lo.append(0.0); link_hi.append(np.inf)
        r = np.zeros(nv); r[p:p + d] = 1
        link_rows.append(r); link_lo.append(0.0); link_hi.append(float(min(self.k, d)))
        cuts, cut_lo = [], []
        rho = np.zeros(p)
        pos = c @ (Z[:, 0] > 0)
        rho[0] = float(np.clip(np.round(np.log(max(pos, 1e-9) / max(1 - pos, 1e-9))), -50, 50))
        best_rho, ub_loss, lb_loss = rho.copy(), np.inf, 0.0
        # warm start, as RiskSLIM's initialization does with rounding heuristics: the rounded solutions of
        # the L1 path (FasterRisk's star-ray rounding), each with its best integer intercept, give the
        # first incumbent and a cut each
        from fasterrisk.rounding import starRaySearchModel
        ray = starRaySearchModel(X=X, y=s, lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20)
        ints = np.arange(-50, 51, dtype=float)
        for S in _l1_path_supports(X, y, self.k, n_c=20):
            w = _star_ray_round(X, y, *_logistic_refit(X, y, S), ray=ray)
            if not np.any(w):
                continue
            sw = Z[:, 1:] @ w
            losses = [float(c @ np.logaddexp(0, -(sw + Z[:, 0] * b))) for b in ints]
            cand = np.concatenate([[ints[int(np.argmin(losses))]], w])
            l_, g_ = loss_grad(cand)
            r = np.zeros(nv); r[:p] = -g_; r[-1] = 1.0
            cuts.append(r); cut_lo.append(l_ - g_ @ cand)
            if l_ < ub_loss:
                ub_loss, best_rho = l_, cand.copy()
        it = 0
        while time.perf_counter() - t0 < self.time_limit:
            loss, g = loss_grad(rho)
            if loss < ub_loss:
                ub_loss, best_rho = loss, rho.copy()
            r = np.zeros(nv); r[:p] = -g; r[-1] = 1.0   # theta >= loss + g.(rho' - rho)
            cuts.append(r); cut_lo.append(loss - g @ rho)
            if ub_loss - lb_loss <= 1e-6 * max(ub_loss, 1e-12):
                break
            A = np.vstack(link_rows + cuts)
            lo = np.array(link_lo + cut_lo); hi = np.array(link_hi + [np.inf] * len(cuts))
            left = self.time_limit - (time.perf_counter() - t0)
            if left <= 0.5:
                break
            res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
                       options={"time_limit": left, "disp": False, "mip_rel_gap": 1e-9})
            if res.x is None:
                break
            lb_loss = max(lb_loss, float(getattr(res, "mip_dual_bound", res.fun) or res.fun)) if res.status == 0 else lb_loss
            rho = np.round(res.x[:p])
            it += 1
        self.coef_ = best_rho[1:]
        self.intercept_ = float(best_rho[0])
        self.optimal_ = bool(ub_loss - lb_loss <= 1e-6 * max(ub_loss, 1e-12))
        self.stop_reason_ = f"{'optimal' if self.optimal_ else 'time'} it={it} gap={ub_loss - lb_loss:.2e}"
        return self


class AbessSeqRound(_Base):
    """abess (Zhu et al., JMLR 2022) best-subset logistic regression by splicing, fit at every
    support size 1..k; each solution is rounded by FasterRisk's star-ray sequential rounding and
    the rounding with the lowest calibrated loss is kept."""

    def fit(self, X, y):
        from abess import LogisticRegression as AbessLR
        from fasterrisk.rounding import starRaySearchModel
        ray = starRaySearchModel(X=X, y=np.where(y > 0, 1.0, -1.0), lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20)
        pool = []
        for s_ in range(1, min(self.k, X.shape[1]) + 1):
            m = AbessLR(support_size=s_, thread=1).fit(X, y)
            pool.append(_star_ray_round(X, y, float(m.intercept_), np.asarray(m.coef_, float).ravel(), ray=ray))
        self.coef_ = _best_of_pool(X, y, pool)
        return self


class FastSparseSeqRound(_Base):
    """fastSparse (Liu et al., AISTATS 2022; the L0Learn line, package fastsparsegams): the L0L2
    logistic path with coefficients boxed to [-5, 5]; every path solution with at most k nonzero
    coefficients is rounded by FasterRisk's star-ray sequential rounding, and the rounding with the
    lowest calibrated loss is kept."""

    def fit(self, X, y):
        import fastsparsegams
        from fasterrisk.rounding import starRaySearchModel
        d = X.shape[1]
        ray = starRaySearchModel(X=X, y=np.where(y > 0, 1.0, -1.0), lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20)
        m = fastsparsegams.fit(X, np.where(y > 0, 1.0, -1.0), loss="Logistic", penalty="L0L2",
                               max_support_size=min(self.k, d), num_gamma=1, gamma_max=1e-3, gamma_min=1e-3,
                               lows=-COEF_BOUND * np.ones(d), highs=COEF_BOUND * np.ones(d))
        ch = m.characteristics()
        pool = []
        for lam, size in zip(ch["l0"], ch["support_size"]):
            if 0 < size <= self.k:
                c = np.asarray(m.coeff(lambda_0=lam, gamma=1e-3).todense()).ravel()
                pool.append(_star_ray_round(X, y, c[0], c[1:], ray=ray))
        self.coef_ = _best_of_pool(X, y, pool)
        return self


class OKRidgeSeqRound(_Base):
    """OKRidge (Liu et al., NeurIPS 2023): the certifiably optimal k-sparse support for ridge
    regression of the +-1 labels (a squared-loss proxy for the log loss), found by branch and bound
    within half the time limit; a logistic model is refit on that support and rounded by FasterRisk's
    star-ray sequential rounding."""

    def fit(self, X, y):
        from okridge.tree import BNBTree
        d = X.shape[1]
        k = min(self.k, d)
        Xc = X - X.mean(0)
        ok = np.std(Xc, axis=0) > 0
        tree = BNBTree(Xc[:, ok], np.where(y > 0, 1.0, -1.0), lambda2=1e-3)
        _, beta, gap, _, _ = tree.solve(min(k, int(ok.sum())), gap_tol=1e-4, time_limit=0.5 * self.time_limit)
        S = np.flatnonzero(ok)[np.flatnonzero(np.abs(np.asarray(beta).ravel()) > 1e-10)]
        self.coef_ = _best_of_pool(X, y, [_star_ray_round(X, y, *_logistic_refit(X, y, S))])
        self.stop_reason_ = f"okridge gap={float(gap):.2g}"
        return self


class PSL(_Base):
    """Probabilistic scoring lists (Hanselle et al., Machine Learning 2025; package scikit-psl,
    vendored): features and integer scores in {-5..5} \\ {0} are added greedily, one stage at a
    time, choosing at each stage the (feature, score) whose calibrated list has the lowest expected
    entropy; stopped after k stages. The harness refits its own logistic risk curve on the total
    score (PSL itself calibrates each total by isotonic regression)."""

    def fit(self, X, y):
        from skpsl import ProbabilisticScoringList
        m = ProbabilisticScoringList(score_set={-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}, lookahead=1, n_jobs=1,
                                     max_stages=min(self.k, X.shape[1])).fit(X, y)
        coef = np.zeros(X.shape[1])
        for f, s_ in zip(m.features[:self.k], m.scores[:self.k]):
            coef[int(f)] = s_
        self.coef_ = coef
        return self


BASELINES = {"fasterrisk": FasterRisk, "fasterrisk_wide": FasterRiskWide, "riskslim": RiskSLIM, "slim_milp": SlimMILP,
             "rounded_lr": RoundedLR, "imodels_slim": ImodelsSLIM, "unit_weighting": UnitWeighting,
             "autoscore": AutoScore, "l1path_seqround": L1PathSeqRound, "cpa_highs": CuttingPlaneHiGHS,
             "abess_seqround": AbessSeqRound, "fastsparse_seqround": FastSparseSeqRound,
             "okridge_seqround": OKRidgeSeqRound, "psl": PSL, "continuous_beam": ContinuousBeam}
INTEGER = {name: name != "continuous_beam" for name in BASELINES}


def make_baseline(name):
    cls = BASELINES[name]
    return lambda k, time_limit: cls(k, time_limit)
