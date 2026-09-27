"""Sparse integer linear model (risk score) solver: the file the agent edits.

FasterRisk's algorithm (Liu et al., NeurIPS 2022: sparse beam search, diverse pool,
star-ray search with sequential rounding) re-implemented on compressed data (unique
(x, y) rows with counts) with numba kernels, and a Newton fine-tuning step on the
support in place of coordinate descent.

Interface (fixed): ``make_model(k, time_limit)`` returns an estimator whose
``fit(X, y)`` (y in {0, 1}) leaves integer points in ``coef_`` with at most ``k``
nonzero entries in [-5, 5]. The harness scores ``coef_`` alone.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from numba import njit

MODEL_NAME = "v4_fast_lattice"
DESCRIPTION = "v3 + calibrated-loss lattice local search (exact coord moves, 2nd-order screened swaps/adds) from the 3 best rounded pool solutions"
INTEGER = True
COEF_BOUND = 5


# ---------------------------------------------------------------- kernels
@njit
def _softplus(z):
    if z > 0:
        return z + np.log1p(np.exp(-z))
    return np.log1p(np.exp(z))


@njit
def _sig(z):
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


@njit
def margin_loss(M, c):
    tot = 0.0
    for i in range(M.shape[0]):
        tot += c[i] * _softplus(-M[i])
    return tot


@njit
def newton_support(ZT, y, c, S, beta0, betas, M, lam2, lbs, ubs, maxit):
    """Box-constrained damped Newton on (intercept, betas[S]); M = margins, updated in place.

    Returns (beta0, objective). betas modified in place."""
    k = S.shape[0]
    m = M.shape[0]
    q = k + 1
    obj = margin_loss(M, c)
    for j in S:
        obj += lam2 * betas[j] * betas[j]
    newM = np.empty(m)
    for it in range(maxit):
        g = np.zeros(q)
        H = np.zeros((q, q))
        for i in range(m):
            p = _sig(-M[i])
            r = -c[i] * p
            w = c[i] * p * (1.0 - p)
            g[0] += r * y[i]
            H[0, 0] += w
            for a in range(k):
                za = ZT[S[a], i]
                g[a + 1] += r * za
                H[0, a + 1] += w * y[i] * za
                for b in range(a + 1):
                    H[a + 1, b + 1] += w * za * ZT[S[b], i]
        for a in range(k):
            g[a + 1] += 2 * lam2 * betas[S[a]]
            H[a + 1, a + 1] += 2 * lam2 + 1e-10
            H[a + 1, 0] = H[0, a + 1]
            for b in range(a):
                H[b + 1, a + 1] = H[a + 1, b + 1]
        H[0, 0] += 1e-10
        d = -np.linalg.solve(H, g)
        t = 1.0
        accepted = False
        while t > 1e-6:
            nb0 = beta0 + t * d[0]
            for i in range(m):
                newM[i] = M[i] + y[i] * (nb0 - beta0)
            pen = 0.0
            for a in range(k):
                j = S[a]
                nb = min(ubs[j], max(lbs[j], betas[j] + t * d[a + 1]))
                dj = nb - betas[j]
                if dj != 0.0:
                    for i in range(m):
                        newM[i] += ZT[j, i] * dj
                pen += lam2 * nb * nb
            nobj = margin_loss(newM, c) + pen
            if nobj <= obj + 1e-4 * t * (g @ d):
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        beta0 = beta0 + t * d[0]
        for a in range(k):
            j = S[a]
            betas[j] = min(ubs[j], max(lbs[j], betas[j] + t * d[a + 1]))
        M[:] = newM
        rel = (obj - nobj) / max(nobj, 1e-12)
        obj = nobj
        if rel < 1e-9:
            break
    return beta0, obj


@njit
def grad_cols(ZT, c, M, cols):
    out = np.zeros(cols.shape[0])
    r = np.empty(M.shape[0])
    for i in range(M.shape[0]):
        r[i] = c[i] * _sig(-M[i])
    for a in range(cols.shape[0]):
        j = cols[a]
        s = 0.0
        for i in range(M.shape[0]):
            s += r[i] * ZT[j, i]
        out[a] = -s
    return out


@njit
def newton_1d(ZT, c, M, j, lb, ub, lam2, iters):
    """Fit one coefficient (starting at 0) with everything else fixed; returns (beta, loss)."""
    m = M.shape[0]
    b = 0.0
    for it in range(iters):
        g = 0.0
        h = 1e-10 + 2 * lam2
        for i in range(m):
            z = M[i] + ZT[j, i] * b
            p = _sig(-z)
            g -= c[i] * p * ZT[j, i]
            h += c[i] * p * (1 - p) * ZT[j, i] * ZT[j, i]
        g += 2 * lam2 * b
        nb = min(ub, max(lb, b - g / h))
        if abs(nb - b) < 1e-9:
            b = nb
            break
        b = nb
    tot = 0.0
    for i in range(m):
        tot += c[i] * _softplus(-(M[i] + ZT[j, i] * b))
    return b, tot + lam2 * b * b


@njit
def seq_round(betas, lyx):
    """FasterRisk sequential rounding (auxiliary loss) on rows lyx (already weighted)."""
    m, q = lyx.shape
    fl = np.floor(betas)
    ce = np.ceil(betas)
    dfl = fl - betas
    dce = ce - betas
    todo = np.zeros(q, np.bool_)
    ntodo = 0
    for j in range(q):
        if fl[j] != ce[j]:
            todo[j] = True
            ntodo += 1
    lsq = np.zeros(q)
    for j in range(q):
        for i in range(m):
            lsq[j] += lyx[i, j] * lyx[i, j]
    diff = np.zeros(m)
    cur = 0.0
    out = betas.copy()
    while ntodo > 0:
        best = 1e300
        bj = -1
        bc = False
        for j in range(q):
            if not todo[j]:
                continue
            expect = cur - lsq[j] * dfl[j] * dce[j]
            u = 0.0
            for i in range(m):
                t = diff[i] + dce[j] * lyx[i, j]
                u += t * t
            if u < best:
                best, bj, bc = u, j, True
            if u > expect:
                u2 = 0.0
                for i in range(m):
                    t = diff[i] + dfl[j] * lyx[i, j]
                    u2 += t * t
                if u2 < best:
                    best, bj, bc = u2, j, False
        cur = best
        step = dce[bj] if bc else dfl[bj]
        out[bj] += step
        for i in range(m):
            diff[i] += step * lyx[i, bj]
        todo[bj] = False
        ntodo -= 1
    return out


# ---------------------------------------------------------------- the model
class Compressed:
    def __init__(self, X, ys, lam2=1e-8, bound=COEF_BOUND):
        XY = np.hstack([X, ys[:, None]])
        U, cnt = np.unique(XY, axis=0, return_counts=True)
        self.Xo = np.ascontiguousarray(U[:, :-1])
        self.y = np.ascontiguousarray(U[:, -1])
        self.c = cnt.astype(float)
        n = self.c.sum()
        self.n = n
        self.mean = (self.c @ self.Xo) / n
        Xc = self.Xo - self.mean
        self.norm = np.sqrt(self.c @ (Xc * Xc))
        self.scaled = self.norm >= 1e-9
        Xc[:, self.scaled] /= self.norm[self.scaled]
        self.m, self.p = Xc.shape
        self.ZT = np.ascontiguousarray((self.y[:, None] * Xc).T)
        self.lam2 = lam2
        self.lbs = -bound * np.ones(self.p)
        self.ubs = bound * np.ones(self.p)
        self.lbs[self.scaled] *= self.norm[self.scaled]
        self.ubs[self.scaled] *= self.norm[self.scaled]
        npos = self.c @ (self.y > 0)
        self.b0_init = np.log(npos / (n - npos))

    def margins(self, b0, betas):
        S = np.flatnonzero(betas)
        return self.y * b0 + betas[S] @ self.ZT[S]

    def finetune(self, b0, betas, M, maxit=30):
        S = np.flatnonzero(betas != 0).astype(np.int64)
        return newton_support(self.ZT, self.y, self.c, S, b0, betas, M, self.lam2, self.lbs, self.ubs, maxit)

    def to_original(self, b0, betas):
        out = np.zeros(self.p)
        out[self.scaled] = betas[self.scaled] / self.norm[self.scaled]
        return b0 - self.mean @ out, out


def beam_search(cm, k, parent_size=10, child_size=10):
    """Returns list of (obj, b0, betas, M) of the final beam, best first."""
    b0 = cm.b0_init
    betas = np.zeros(cm.p)
    M = cm.y * b0
    parents = [(margin_loss(M, cm.c), b0, betas, M)]
    seen = set()
    for _ in range(min(k, cm.p)):
        children = []
        for obj, pb0, pbetas, pM in parents:
            ns = np.flatnonzero(pbetas == 0).astype(np.int64)
            if len(ns) == 0:
                continue
            g = grad_cols(cm.ZT, cm.c, pM, ns)
            new_js = ns[np.argsort(-np.abs(g), kind="stable")[:child_size]]
            for j in new_js:
                key = tuple(sorted(np.flatnonzero(pbetas).tolist() + [int(j)]))
                if key in seen:
                    continue
                seen.add(key)
                b = pbetas.copy()
                cb, _ = newton_1d(cm.ZT, cm.c, pM, j, cm.lbs[j], cm.ubs[j], cm.lam2, 10)
                if cb == 0.0:
                    cb = 1e-6
                b[j] = cb
                Mc = pM + cb * cm.ZT[j]
                nb0, nobj = cm.finetune(pb0, b, Mc)
                children.append((nobj, nb0, b, Mc))
        if not children:
            break
        children.sort(key=lambda t: t[0])
        parents = children[:parent_size]
    return parents


def diverse_pool(cm, best, gap_tolerance=0.05, select_top_m=50, max_attempts=50):
    obj0, b0, betas, M = best
    nz = np.flatnonzero(betas)
    z = np.flatnonzero(betas == 0).astype(np.int64)
    attempts = min(max_attempts, len(z))
    pool = [(obj0, b0, betas)]
    for j in nz:
        Mj = M - betas[j] * cm.ZT[j]
        g = grad_cols(cm.ZT, cm.c, Mj, z)
        new_js = z[np.argsort(-np.abs(g), kind="stable")[:attempts]]
        for l in new_js:
            cb, loss = newton_1d(cm.ZT, cm.c, Mj, l, cm.lbs[l], cm.ubs[l], cm.lam2, 10)
            loss += cm.lam2 * (betas[nz] @ betas[nz] - betas[j] ** 2)
            if (loss - obj0) / obj0 < gap_tolerance and cb != 0.0:
                b = betas.copy(); b[j] = 0.0; b[l] = cb
                Ml = Mj + cb * cm.ZT[l]
                nb0, nobj = cm.finetune(b0, b, Ml)
                pool.append((nobj, nb0, b))
    pool.sort(key=lambda t: t[0])
    return pool[:select_top_m]


class StarRay:
    def __init__(self, cm, bound=COEF_BOUND, num_ray_search=20, early_stop_tolerance=0.001):
        X1 = np.hstack([np.ones((cm.m, 1)), cm.Xo])
        self.yX = cm.y[:, None] * X1
        self.c = cm.c
        self.sqc = np.sqrt(cm.c)
        self.p = X1.shape[1]
        self.ub_arr = bound * np.ones(self.p); self.ub_arr[0] = 100.0
        self.lb_arr = -bound * np.ones(self.p); self.lb_arr[0] = -100.0
        self.num_ray_search = num_ray_search
        self.early_stop_tolerance = early_stop_tolerance

    def multipliers(self, b, idx):
        pos, neg = b > 1e-8, b < -1e-8
        largest = 1e8
        if pos.any():
            largest = min(largest, np.min(self.ub_arr[idx][pos] / b[pos]))
        if neg.any():
            largest = min(largest, np.min(self.lb_arr[idx][neg] / b[neg]))
        if largest > 1:
            return np.linspace(1, largest, self.num_ray_search)
        return np.linspace(1, 0.5, self.num_ray_search)

    def wloss(self, yx, b):
        return float(self.c @ np.logaddexp(0, -(yx @ b)))

    def line_search(self, betas):
        idx = np.flatnonzero(np.abs(betas) > 1e-9)
        yx = self.yX[:, idx]
        b = betas[idx]
        loss_cont = self.wloss(yx, b)
        best_loss, best_b, best_m = 1e300, np.zeros(len(idx)), 1.0
        for mult in self.multipliers(b, idx):
            yxm = yx / mult
            bm = b * mult
            fl = np.floor(bm)
            gamma = fl[None, :] + 1.0 * (yxm <= 0)
            l_fac = 1.0 / (1.0 + np.exp(np.sum(yxm * gamma, axis=1)))
            lyx = np.ascontiguousarray((self.sqc * l_fac)[:, None] * yxm)
            scaled = seq_round(bm, lyx)
            loss = self.wloss(yx, scaled / mult)
            if loss < best_loss:
                best_loss, best_b, best_m = loss, scaled.copy(), mult
            if (loss - loss_cont) / loss_cont < self.early_stop_tolerance:
                break
        out = np.zeros(self.p)
        out[idx] = best_b
        return best_m, out, best_loss


# ------------------------------------------------------ lattice local search
# The harness scores points w by  min_{a,b} sum_i loss(a * x_i.w + b).  Everything below
# works on that calibrated loss directly, on unique rows with positive / negative counts.


@njit
def _loss_ab(s, wp, wn, a, b):
    tot = 0.0
    for i in range(s.shape[0]):
        z = a * s[i] + b
        tot += wp[i] * _softplus(-z) + wn[i] * _softplus(z)
    return tot


@njit
def calib(s, wp, wn, a0, b0):
    """min over (a, b) of the weighted logistic loss of a*s+b; damped Newton, warm started."""
    m = s.shape[0]
    smin, smax = s[0], s[0]
    for i in range(m):
        smin = min(smin, s[i]); smax = max(smax, s[i])
    a, b = a0, b0
    if smax - smin < 1e-12:
        a = 0.0
    L = _loss_ab(s, wp, wn, a, b)
    for it in range(100):
        ga = 0.0; gb = 0.0; haa = 0.0; hab = 0.0; hbb = 0.0
        for i in range(m):
            z = a * s[i] + b
            q = 1.0 / (1.0 + np.exp(-z))
            c = wp[i] + wn[i]
            r = c * q - wp[i]
            h = c * q * (1.0 - q)
            ga += r * s[i]; gb += r
            haa += h * s[i] * s[i]; hab += h * s[i]; hbb += h
        if smax - smin < 1e-12:
            ga = 0.0; haa = 1.0; hab = 0.0
        det = haa * hbb - hab * hab
        if det <= 1e-300:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        dec = ga * da + gb * db
        if dec < 1e-12 * (1.0 + L):
            break
        t = 1.0
        newL = L
        ok = False
        while t > 1e-10:
            newL = _loss_ab(s, wp, wn, a - t * da, b - t * db)
            if newL <= L - 1e-4 * t * dec:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        a -= t * da; b -= t * db
        impr = L - newL
        L = newL
        if impr < 1e-13 * (1.0 + L):
            break
    return a, b, L


@njit
def calib_move(s, xj, delta, wp, wn, a0, b0, buf):
    for i in range(s.shape[0]):
        buf[i] = s[i] + delta * xj[i]
    return calib(buf, wp, wn, a0, b0)


class Lattice:
    """Calibrated-loss local search over integer points on unique rows."""

    def __init__(self, X, y01, bound=COEF_BOUND):
        Xu, inv = np.unique(X, axis=0, return_inverse=True)
        inv = inv.ravel()
        m = Xu.shape[0]
        self.wp = np.bincount(inv, weights=y01, minlength=m).astype(float)
        self.wn = np.bincount(inv, minlength=m).astype(float) - self.wp
        self.X = np.ascontiguousarray(Xu)
        self.XT = np.ascontiguousarray(Xu.T)
        self.X2T = self.XT * self.XT
        self.m, self.d = Xu.shape
        self.bound = bound
        self.buf = np.zeros(m)
        p = min(max(self.wp.sum() / (self.wp.sum() + self.wn.sum()), 1e-12), 1 - 1e-12)
        self.b_init = np.log(p / (1 - p))

    def evaluate(self, w, a=None, b=None):
        s = self.X @ w
        if a is None:
            a, b = 1.0, self.b_init
        return calib(s, self.wp, self.wn, a, b)

    def coord_pass(self, w, s, a, b, L):
        """Best single-coordinate change (value in [-B, B]) over the support."""
        best = (L, -1, 0, a, b)
        for j in np.flatnonzero(w):
            xj = self.XT[j]
            for v in range(-self.bound, self.bound + 1):
                if v == w[j]:
                    continue
                aa, bb, LL = calib_move(s, xj, float(v - w[j]), self.wp, self.wn, a, b, self.buf)
                if LL < best[0] - 1e-10:
                    best = (LL, j, v, aa, bb)
        return best

    def swap_screen(self, w, s, a, b, k, top):
        """Second-order screen of (remove j or nothing) + (add l with value v), intercept re-fit."""
        supp = np.flatnonzero(w)
        bases = list(supp) if len(supp) >= k else [-1] + list(supp)
        vs = np.array([v for v in range(-self.bound, self.bound + 1) if v != 0], float)
        cands = []
        for j in bases:
            s2 = s if j < 0 else s - w[j] * self.XT[j]
            z = a * s2 + b
            q = 1.0 / (1.0 + np.exp(-z))
            c = self.wp + self.wn
            r = c * q - self.wp
            h = c * q * (1 - q)
            L0 = float(np.sum(self.wp * np.logaddexp(0, -z) + self.wn * np.logaddexp(0, z)))
            G = self.XT @ r
            H1 = self.XT @ h
            H2 = self.X2T @ h
            R, Ht = r.sum(), h.sum() + 1e-12
            av = a * vs[:, None]
            q_ = av * G[None, :] + 0.5 * av * av * H2[None, :] - (R + av * H1[None, :]) ** 2 / (2 * Ht)
            q_ += R * R / (2 * Ht)
            q_[:, supp] = np.inf
            best_v = np.argmin(q_, axis=0)
            pred = L0 + q_[best_v, np.arange(self.d)]
            order = np.argsort(pred)[:top]
            for l in order:
                if np.isfinite(pred[l]):
                    cands.append((pred[l], j, l, int(vs[best_v[l]])))
        cands.sort()
        return cands[:top]

    def search(self, w, k, max_iter=200, top=10):
        w = np.asarray(np.round(w), dtype=np.int64).copy()
        s = self.X @ w.astype(float)
        a, b, L = calib(s, self.wp, self.wn, 1.0, self.b_init)
        for _ in range(max_iter):
            LL, j, v, aa, bb = self.coord_pass(w, s, a, b, L)
            if j >= 0:
                s = s + (v - w[j]) * self.XT[j]
                w[j] = v; a, b, L = aa, bb, LL
                continue
            improved = False
            for pred, jr, l, v in self.swap_screen(w, s, a, b, k, top):
                s2 = s if jr < 0 else s - w[jr] * self.XT[jr]
                for vv in (v - 1, v, v + 1):
                    if vv == 0 or abs(vv) > self.bound:
                        continue
                    aa, bb, LL = calib_move(s2, self.XT[l], float(vv), self.wp, self.wn, a, b, self.buf)
                    if LL < L - 1e-10:
                        best = (LL, jr, l, vv, aa, bb)
                        L = LL; improved = True
                if improved:
                    break
            if not improved:
                break
            LL, jr, l, vv, a, b = best
            if jr >= 0:
                s = s - w[jr] * self.XT[jr]; w[jr] = 0
            s = s + vv * self.XT[l]; w[l] = vv
        return w, L


class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, select_top_m=50,
                 gap_tolerance=0.05, max_attempts=50, num_ray_search=20, n_starts=3):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.select_top_m, self.gap_tolerance = select_top_m, gap_tolerance
        self.max_attempts, self.num_ray_search = max_attempts, num_ray_search
        self.n_starts = n_starts

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        ys = np.where(np.asarray(y) > 0, 1.0, -1.0)
        cm = Compressed(X, ys)
        beam = beam_search(cm, self.k, self.parent_size, self.child_size)
        pool = diverse_pool(cm, beam[0], self.gap_tolerance, self.select_top_m, self.max_attempts)
        ray = StarRay(cm, num_ray_search=self.num_ray_search)
        cands = []
        seen = set()
        for _, b0, b in pool:
            ob0, ob = cm.to_original(b0, b)
            mult, sol, loss = ray.line_search(np.concatenate([[ob0], ob]))
            key = np.flatnonzero(sol).tobytes()
            if key in seen:
                continue
            seen.add(key)
            cands.append((loss, np.round(sol[1:])))
        lat = Lattice(X, (ys > 0).astype(float))
        cands.sort(key=lambda c: c[0])
        best_w, best_L = None, np.inf
        for _, w0 in cands[:self.n_starts]:
            w, L = lat.search(w0, self.k)
            if L < best_L - 1e-12:
                best_w, best_L = w, L
        self.coef_ = best_w.astype(float)
        self.intercept_, self.multiplier_ = 0.0, 1.0
        return self


def make_model(k, time_limit):
    return SparseIntegerClassifier(k=k, time_limit=time_limit)



# ===========================================================================
# Evaluation loop (do not edit below this line)
# ===========================================================================

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "src"))
    from evaluate import evaluate_solver, print_summary, record
    from suite import TIME_LIMIT

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="", help="comma-separated subset of the suite (default: all)")
    ap.add_argument("--ks", default="", help="comma-separated subset of the k grid (default: all)")
    ap.add_argument("--jobs", type=int, default=14, help="parallel worker processes")
    ap.add_argument("--no-record", action="store_true", help="do not write results/")
    args = ap.parse_args()
    t0 = time.time()
    datasets = [d for d in args.datasets.split(",") if d] or None
    ks = [int(v) for v in args.ks.split(",") if v] or None
    summary = evaluate_solver(("file", os.path.abspath(__file__)), MODEL_NAME, datasets=datasets, ks=ks,
                              time_limit=TIME_LIMIT, jobs=args.jobs, integer=INTEGER)
    if datasets is None and ks is None and not args.no_record:
        record(MODEL_NAME, DESCRIPTION, summary, results_dir=os.path.join(here, "results"), integer=INTEGER)
    elif not args.no_record:
        print("(partial suite: results not recorded)")
    print_summary(MODEL_NAME, summary)
    print(f"total_seconds: {time.time() - t0:.1f}s")
