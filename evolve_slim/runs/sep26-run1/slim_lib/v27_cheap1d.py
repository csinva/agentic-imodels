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

MODEL_NAME = "v27_cheap1d"
DESCRIPTION = "v24 + 1-D Newton ranking fits capped at 3 steps, 20 pool attempts per support feature + lattice searches stop on reaching a state visited by an earlier search in the same fit; v23 with beam child_size 5; v22 + pair moves (joint +-1 on two support points) in the polish stage after rescale fails; v19 + Newton fine-tune stops at 1e-6 relative + non-integer data: value scans ranked by 2 warm Newton steps, best one calibrated fully; v18 + calibration Newton with one fused pass (loss, grad, Hessian) per trial point; v17 + rescale polish of the best lattice result (round t*w, t in 0.5..2, coordinate descent, accept if better); v11 + lattice value scans on per-group score histograms (rows grouped by the feature value, shift and merge per value); v9 + diverse pool fine-tunes only the 2*select_top_m swaps with the best one-coordinate fit; v7 with the star-ray pool cut from 50 to 10 solutions and the lattice reusing the compressed rows; v6 + calibrated-loss lattice local search (score histograms when X is integer) from the 3 best rounded candidates by calibrated loss; v5 + star-ray search and sequential rounding in one numba kernel; v3 + beam search fine-tunes only the 2*parent_size children with the best one-coordinate loss; contiguous support block in Newton"
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
    Zs = np.empty((m, k))
    for a in range(k):
        for i in range(m):
            Zs[i, a] = ZT[S[a], i]
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
            wy = w * y[i]
            for a in range(k):
                za = Zs[i, a]
                g[a + 1] += r * za
                H[0, a + 1] += wy * za
                wz = w * za
                for b in range(a + 1):
                    H[a + 1, b + 1] += wz * Zs[i, b]
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
                        newM[i] += Zs[i, a] * dj
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
        if rel < 1e-6:
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



@njit
def star_ray_kernel(yx, c, sqc, b, mults, tol):
    m, q = yx.shape
    loss_cont = 0.0
    for i in range(m):
        z = 0.0
        for j in range(q):
            z += yx[i, j] * b[j]
        loss_cont += c[i] * _softplus(-z)
    best_loss = 1e300
    best_b = np.zeros(q)
    best_m = 1.0
    lyx = np.empty((m, q))
    for mult in mults:
        bm = b * mult
        fl = np.floor(bm)
        for i in range(m):
            s = 0.0
            for j in range(q):
                v = yx[i, j] / mult
                s += v * (fl[j] + (1.0 if v <= 0 else 0.0))
            lf = sqc[i] / (1.0 + np.exp(s))
            for j in range(q):
                lyx[i, j] = lf * yx[i, j] / mult
        scaled = seq_round(bm, lyx)
        loss = 0.0
        for i in range(m):
            z = 0.0
            for j in range(q):
                z += yx[i, j] * scaled[j]
            loss += c[i] * _softplus(-z / mult)
        if loss < best_loss:
            best_loss = loss
            best_b[:] = scaled
            best_m = mult
        if (loss - loss_cont) / loss_cont < tol:
            break
    return best_m, best_b, best_loss


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


def beam_search(cm, k, parent_size=10, child_size=10, tune_factor=2):
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
                cb, l1 = newton_1d(cm.ZT, cm.c, pM, j, cm.lbs[j], cm.ubs[j], cm.lam2, 3)
                if cb == 0.0:
                    cb = 1e-6
                children.append((l1, pb0, pbetas, pM, j, cb))
        if not children:
            break
        # fine-tune only the children whose one-coordinate fit is among the most promising
        children.sort(key=lambda t: t[0])
        tuned = []
        for l1, pb0, pbetas, pM, j, cb in children[:tune_factor * parent_size]:
            b = pbetas.copy(); b[j] = cb
            Mc = pM + cb * cm.ZT[j]
            nb0, nobj = cm.finetune(pb0, b, Mc)
            tuned.append((nobj, nb0, b, Mc))
        tuned.sort(key=lambda t: t[0])
        parents = tuned[:parent_size]
    return parents


def diverse_pool(cm, best, gap_tolerance=0.05, select_top_m=50, max_attempts=50, tune_factor=2):
    obj0, b0, betas, M = best
    nz = np.flatnonzero(betas)
    z = np.flatnonzero(betas == 0).astype(np.int64)
    attempts = min(max_attempts, len(z))
    pool = [(obj0, b0, betas)]
    cand = []
    for j in nz:
        Mj = M - betas[j] * cm.ZT[j]
        g = grad_cols(cm.ZT, cm.c, Mj, z)
        new_js = z[np.argsort(-np.abs(g), kind="stable")[:attempts]]
        for l in new_js:
            cb, loss = newton_1d(cm.ZT, cm.c, Mj, l, cm.lbs[l], cm.ubs[l], cm.lam2, 3)
            loss += cm.lam2 * (betas[nz] @ betas[nz] - betas[j] ** 2)
            if (loss - obj0) / obj0 < gap_tolerance and cb != 0.0:
                cand.append((loss, j, l, cb))
    # fine-tune only the swaps whose one-coordinate fit is most promising
    cand.sort(key=lambda t: t[0])
    for loss, j, l, cb in cand[:tune_factor * select_top_m]:
        b = betas.copy(); b[j] = 0.0; b[l] = cb
        Ml = M - betas[j] * cm.ZT[j] + cb * cm.ZT[l]
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
        yx = np.ascontiguousarray(self.yX[:, idx])
        b = betas[idx]
        mults = self.multipliers(b, idx).astype(float)
        best_m, best_b, best_loss = star_ray_kernel(yx, self.c, self.sqc, b, mults, self.early_stop_tolerance)
        out = np.zeros(self.p)
        out[idx] = best_b
        return best_m, out, best_loss


# ------------------------------------------------------ lattice local search
# The harness scores points w by  min_{a,b} sum_i loss(a * x_i.w + b).  The search below
# works on that calibrated loss directly, on unique rows with positive / negative counts.
# When X is integer-valued the score is an integer, and a calibration only needs the
# positive / negative counts per score value (a histogram), not every row.
@njit
def _loss_ab(s, wp, wn, a, b):
    tot = 0.0
    for i in range(s.shape[0]):
        z = a * s[i] + b
        tot += wp[i] * _softplus(-z) + wn[i] * _softplus(z)
    return tot


@njit
def _pass(s, wp, wn, a, b, out):
    """Loss, gradient and Hessian in (a, b) of sum wp*softplus(-z) + wn*softplus(z), z = a*s+b."""
    L = 0.0; ga = 0.0; gb = 0.0; haa = 0.0; hab = 0.0; hbb = 0.0
    for i in range(s.shape[0]):
        z = a * s[i] + b
        e = np.exp(-abs(z))
        l1 = np.log1p(e)
        if z >= 0:
            q = 1.0 / (1.0 + e)
            L += wp[i] * l1 + wn[i] * (z + l1)
        else:
            q = e / (1.0 + e)
            L += wp[i] * (l1 - z) + wn[i] * l1
        c = wp[i] + wn[i]
        r = c * q - wp[i]
        h = c * q * (1.0 - q)
        ga += r * s[i]; gb += r
        haa += h * s[i] * s[i]; hab += h * s[i]; hbb += h
    out[0] = ga; out[1] = gb; out[2] = haa; out[3] = hab; out[4] = hbb
    return L


@njit
def calib(s, wp, wn, a0, b0, maxit=100):
    """min over (a, b) of the weighted logistic loss of a*s+b; damped Newton, warm started,
    one pass over the data per trial point."""
    m = s.shape[0]
    smin, smax = s[0], s[0]
    for i in range(m):
        smin = min(smin, s[i]); smax = max(smax, s[i])
    const = smax - smin < 1e-12
    a, b = a0, b0
    if const:
        a = 0.0
    g = np.empty(5); g2 = np.empty(5)
    L = _pass(s, wp, wn, a, b, g)
    for it in range(maxit):
        ga, gb, haa, hab, hbb = g[0], g[1], g[2], g[3], g[4]
        if const:
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
        ok = False
        newL = L
        while t > 1e-10:
            newL = _pass(s, wp, wn, a - t * da, b - t * db, g2)
            if newL <= L - 1e-4 * t * dec:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        a -= t * da; b -= t * db
        impr = L - newL
        L = newL
        g[:] = g2
        if impr < 1e-13 * (1.0 + L):
            break
    return a, b, L


@njit
def calib_move(s, xj, delta, wp, wn, a0, b0, buf, is_int, hp, hn, us, up, un, maxit=100):
    """Calibrated loss of the score s + delta * xj (histogram of score values if integer)."""
    m = s.shape[0]
    lo = 1e300; hi = -1e300
    for i in range(m):
        v = s[i] + delta * xj[i]
        buf[i] = v
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    R = int(hi - lo) + 1 if is_int else 0
    if is_int and R <= hp.shape[0]:
        for r in range(R):
            hp[r] = 0.0; hn[r] = 0.0
        for i in range(m):
            r = int(buf[i] - lo + 0.5)
            hp[r] += wp[i]; hn[r] += wn[i]
        u = 0
        for r in range(R):
            if hp[r] + hn[r] > 0:
                us[u] = lo + r; up[u] = hp[r]; un[u] = hn[r]
                u += 1
        # shift by lo for conditioning: a*(s) + b = a*(s - lo) + (b + a*lo)
        for r in range(u):
            us[r] -= lo
        aa, bb, L = calib(us[:u], up[:u], un[:u], a0, b0 + a0 * lo)
        return aa, bb - aa * lo, L
    return calib(buf, wp, wn, a0, b0, maxit)



@njit
def coord_scan(s, xj, deltas, wp, wn, a0, b0, is_int, buf, hp, hn, us, up, un, maxg):
    """Calibrated loss of s + d * xj for every d in deltas.

    Integer data: rows are grouped by the value of xj, one score histogram per group is built
    once, and each d only shifts and merges those histograms (cost per d ~ range of scores)."""
    nd = deltas.shape[0]
    A = np.empty(nd); B = np.empty(nd); L = np.empty(nd)
    m = s.shape[0]
    if is_int:
        vals = np.unique(xj)
        G = vals.shape[0]
        lo = s[0]; hi = s[0]
        for i in range(m):
            lo = min(lo, s[i]); hi = max(hi, s[i])
        R = int(hi - lo) + 1
        dmax = 0.0
        for d in deltas:
            dmax = max(dmax, abs(d))
        vmax = 0.0
        for g in range(G):
            vmax = max(vmax, abs(vals[g]))
        R2max = R + int(2 * dmax * vmax) + 1
        if G <= maxg and R2max <= hp.shape[0] and G * R <= 4 * hp.shape[0]:
            HP = np.zeros((G, R)); HN = np.zeros((G, R))
            for i in range(m):
                g = np.searchsorted(vals, xj[i])
                r = int(s[i] - lo + 0.5)
                HP[g, r] += wp[i]; HN[g, r] += wn[i]
            for t in range(nd):
                d = deltas[t]
                smin = 1e300
                for g in range(G):
                    smin = min(smin, d * vals[g])
                base = lo + smin
                top = 0
                for g in range(G):
                    off = int(d * vals[g] - smin + 0.5)
                    top = max(top, off + R)
                for r in range(top):
                    hp[r] = 0.0; hn[r] = 0.0
                for g in range(G):
                    off = int(d * vals[g] - smin + 0.5)
                    for r in range(R):
                        hp[off + r] += HP[g, r]; hn[off + r] += HN[g, r]
                u = 0
                for r in range(top):
                    if hp[r] + hn[r] > 0:
                        us[u] = r; up[u] = hp[r]; un[u] = hn[r]
                        u += 1
                aa, bb, LL = calib(us[:u], up[:u], un[:u], a0, b0 + a0 * base)
                A[t] = aa; B[t] = bb - aa * base; L[t] = LL
            return A, B, L
    # per-row path: rank the deltas by 2 warm Newton steps (an achievable loss), then
    # calibrate the best one fully
    best = 0
    for t in range(nd):
        aa, bb, LL = calib_move(s, xj, deltas[t], wp, wn, a0, b0, buf, is_int, hp, hn, us, up, un, 2)
        A[t] = aa; B[t] = bb; L[t] = LL
        if LL < L[best]:
            best = t
    aa, bb, LL = calib_move(s, xj, deltas[best], wp, wn, A[best], B[best], buf, is_int, hp, hn, us, up, un)
    A[best] = aa; B[best] = bb; L[best] = LL
    return A, B, L


class Lattice:
    """Calibrated-loss local search over integer points on unique rows."""

    HIST = 8192
    scales = (0.5, 0.67, 0.8, 1.25, 1.5, 2.0)
    MAXG = 16

    def __init__(self, Xu, wp, wn, bound=COEF_BOUND):
        m = Xu.shape[0]
        self.wp, self.wn = wp, wn
        self.X = np.ascontiguousarray(Xu)
        self.XT = np.ascontiguousarray(Xu.T)
        self.X2T = self.XT * self.XT
        self.m, self.d = Xu.shape
        self.bound = bound
        self.is_int = bool(np.all(Xu == np.round(Xu)))
        self.buf = np.zeros(m)
        H = self.HIST
        self.hp, self.hn = np.zeros(H), np.zeros(H)
        self.us, self.up, self.un = np.zeros(H), np.zeros(H), np.zeros(H)
        self.zero = np.zeros(m)
        p = min(max(self.wp.sum() / (self.wp.sum() + self.wn.sum()), 1e-12), 1 - 1e-12)
        self.b_init = np.log(p / (1 - p))
        self.visited = set()

    def cm(self, s, xj, delta, a, b):
        return calib_move(s, xj, delta, self.wp, self.wn, a, b, self.buf, self.is_int,
                          self.hp, self.hn, self.us, self.up, self.un)

    def evaluate(self, w, a=1.0, b=None):
        s = self.X @ np.asarray(w, float)
        return self.cm(s, self.zero, 0.0, a, self.b_init if b is None else b)

    def coord_pass(self, w, s, a, b, L):
        """Best single-coordinate change (value in [-B, B]) over the support."""
        best = (L, -1, 0, a, b)
        vals = np.arange(-self.bound, self.bound + 1)
        for j in np.flatnonzero(w):
            vs = vals[vals != w[j]]
            A, B, Ls = self.scan(s, self.XT[j], (vs - w[j]).astype(float), a, b)
            i = int(np.argmin(Ls))
            if Ls[i] < best[0] - 1e-10:
                best = (Ls[i], j, int(vs[i]), A[i], B[i])
        return best

    def scan(self, s, xj, deltas, a, b):
        return coord_scan(s, xj, deltas, self.wp, self.wn, a, b, self.is_int, self.buf,
                          self.hp, self.hn, self.us, self.up, self.un, self.MAXG)

    def swap_screen(self, w, s, a, b, k, top):
        """Second-order screen of (remove j or nothing) + (add l with value v), intercept re-fit."""
        supp = np.flatnonzero(w)
        bases = list(supp) if len(supp) >= k else [-1] + list(supp)
        vs = np.array([v for v in range(-self.bound, self.bound + 1) if v != 0], float)
        cands = []
        c = self.wp + self.wn
        for j in bases:
            s2 = s if j < 0 else s - w[j] * self.XT[j]
            z = a * s2 + b
            q = 1.0 / (1.0 + np.exp(-z))
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
            order = np.argsort(pred, kind="stable")[:top]
            for l in order:
                if np.isfinite(pred[l]):
                    cands.append((pred[l], j, l, int(vs[best_v[l]])))
        cands.sort()
        return cands[:top]

    def coord_descent(self, w, s, a, b, L, max_iter=50):
        for _ in range(max_iter):
            LL, j, v, aa, bb = self.coord_pass(w, s, a, b, L)
            if j < 0:
                break
            s = s + (v - w[j]) * self.XT[j]
            w[j] = v; a, b, L = aa, bb, LL
        return w, s, a, b, L

    def rescale(self, w, s, a, b, L):
        """Scale the points by t, round (keeping the support), re-optimise values; accept an improvement."""
        for t in self.scales:
            w2 = np.clip(np.round(w * t), -self.bound, self.bound).astype(np.int64)
            w2[(w2 == 0) & (w != 0)] = np.sign(w[(w2 == 0) & (w != 0)])
            if np.array_equal(w2, w):
                continue
            s2 = self.X @ w2.astype(float)
            a2, b2, L2 = self.cm(s2, self.zero, 0.0, a / t, b)
            w2, s2, a2, b2, L2 = self.coord_descent(w2, s2, a2, b2, L2)
            if L2 < L - 1e-10:
                return w2, s2, a2, b2, L2
        return None

    def pair_pass(self, w, s, a, b, L):
        """Best joint +-1 change of two support coordinates."""
        supp = np.flatnonzero(w)
        best = None
        for x1 in range(len(supp)):
            j1 = supp[x1]
            for d1 in (-1, 1):
                if abs(w[j1] + d1) > self.bound:
                    continue
                s1 = s + d1 * self.XT[j1]
                for x2 in range(x1 + 1, len(supp)):
                    j2 = supp[x2]
                    d2s = np.array([d for d in (-1.0, 1.0) if abs(w[j2] + d) <= self.bound])
                    A, B, Ls = self.scan(s1, self.XT[j2], d2s, a, b)
                    i = int(np.argmin(Ls))
                    if Ls[i] < L - 1e-10 and (best is None or Ls[i] < best[0]):
                        best = (Ls[i], j1, d1, j2, int(d2s[i]), A[i], B[i])
        if best is None:
            return None
        LL, j1, d1, j2, d2, aa, bb = best
        w = w.copy(); w[j1] += d1; w[j2] += d2
        s = s + d1 * self.XT[j1] + d2 * self.XT[j2]
        return w, s, aa, bb, LL

    def search(self, w, k, max_iter=200, top=10, polish=False):
        w = np.asarray(np.round(w), dtype=np.int64).copy()
        s = self.X @ w.astype(float)
        a, b, L = self.cm(s, self.zero, 0.0, 1.0, self.b_init)
        for _ in range(max_iter):
            key = (polish, w.tobytes())
            if key in self.visited:  # an earlier search passed here: the rest of the path is known
                break
            self.visited.add(key)
            LL, j, v, aa, bb = self.coord_pass(w, s, a, b, L)
            if j >= 0:
                s = s + (v - w[j]) * self.XT[j]
                w[j] = v; a, b, L = aa, bb, LL
                continue
            improved = False
            for pred, jr, l, v in self.swap_screen(w, s, a, b, k, top):
                s2 = s if jr < 0 else s - w[jr] * self.XT[jr]
                vvs = np.array([vv for vv in (v - 1, v, v + 1) if vv != 0 and abs(vv) <= self.bound])
                A, B, Ls = self.scan(s2, self.XT[l], vvs.astype(float), a, b)
                for i in range(len(vvs)):
                    if Ls[i] < L - 1e-10:
                        best = (Ls[i], jr, l, int(vvs[i]), A[i], B[i])
                        L = Ls[i]; improved = True
                if improved:
                    break
            if not improved:
                if not polish:
                    break
                res = self.rescale(w, s, a, b, L)
                if res is None:
                    res = self.pair_pass(w, s, a, b, L)
                if res is None:
                    break
                w, s, a, b, L = res
                continue
            LL, jr, l, vv, a, b = best
            if jr >= 0:
                s = s - w[jr] * self.XT[jr]; w[jr] = 0
            s = s + vv * self.XT[l]; w[l] = vv
        return w, L


class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=5, select_top_m=10,
                 gap_tolerance=0.05, max_attempts=20, num_ray_search=20, n_starts=3, n_beam_starts=0):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.select_top_m, self.gap_tolerance = select_top_m, gap_tolerance
        self.max_attempts, self.num_ray_search = max_attempts, num_ray_search
        self.n_starts, self.n_beam_starts = n_starts, n_beam_starts

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        ys = np.where(np.asarray(y) > 0, 1.0, -1.0)
        cm = Compressed(X, ys)
        beam = beam_search(cm, self.k, self.parent_size, self.child_size)
        pool = diverse_pool(cm, beam[0], self.gap_tolerance, self.select_top_m, self.max_attempts)
        ray = StarRay(cm, num_ray_search=self.num_ray_search)
        lat = Lattice(cm.Xo, cm.c * (cm.y > 0), cm.c * (cm.y < 0))
        cands = []
        seen = set()
        for _, b0, b in pool:
            ob0, ob = cm.to_original(b0, b)
            mult, sol, loss = ray.line_search(np.concatenate([[ob0], ob]))
            key = np.flatnonzero(sol).tobytes()
            if key in seen:
                continue
            seen.add(key)
            w0 = np.round(sol[1:])
            cands.append((lat.evaluate(w0)[2], w0))
        cands.sort(key=lambda c: c[0])
        starts = [w0 for _, w0 in cands[:self.n_starts]]
        best_w, best_L = None, np.inf
        for w0 in starts:
            w, L = lat.search(w0, self.k)
            if L < best_L - 1e-12:
                best_w, best_L = w, L
        best_w, best_L = lat.search(best_w, self.k, polish=True)
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
