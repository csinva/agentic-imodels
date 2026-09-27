"""Sparse integer linear model (risk score) solver: the file the agent edits.

See notes.md in the run folder for the history of versions.

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
import numba as nb

MODEL_NAME = "v6_screen"
DESCRIPTION = ("FasterRisk ported to numba on unique (x, y) rows with sparse columns: beam search, diverse pool, "
               "star-ray sequential rounding; final pick by calibrated loss")
INTEGER = True
COEF_BOUND = 5


# ---------------------------------------------------------------- kernels
@nb.njit(cache=False, inline="always")
def _lrow(m):
    # log(1 + exp(-m)), stable
    if m > 0:
        return np.log1p(np.exp(-m))
    return -m + np.log1p(np.exp(m))


@nb.njit(cache=False, inline="always")
def _sig(z):
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


@nb.njit(cache=False)
def total_loss(ym, c):
    s = 0.0
    for i in range(ym.shape[0]):
        s += c[i] * _lrow(ym[i])
    return s


@nb.njit(cache=False)
def coord_step(j, ym, beta, ptr, idx, yval, c, lb, ub, nsteps):
    """Newton steps (with backtracking) on coordinate j. ym = y * margin per row.
    yval holds y_i * x_ij for the nonzeros of column j. Returns total loss decrease."""
    dec_total = 0.0
    for _ in range(nsteps):
        g = 0.0
        h = 0.0
        for t in range(ptr[j], ptr[j + 1]):
            i = idx[t]
            v = yval[t]
            p = _sig(-ym[i])
            g -= c[i] * v * p
            h += c[i] * v * v * p * (1.0 - p)
        if h < 1e-12:
            h = 1e-12
        step = -g / h
        new = beta[j] + step
        if new > ub:
            new = ub
        if new < lb:
            new = lb
        d = new - beta[j]
        if abs(d) < 1e-10:
            break
        for _bt in range(30):
            dl = 0.0
            for t in range(ptr[j], ptr[j + 1]):
                i = idx[t]
                dl += c[i] * (_lrow(ym[i] + d * yval[t]) - _lrow(ym[i]))
            if dl <= 0.0:
                break
            d *= 0.5
        if dl > 0.0:
            break
        for t in range(ptr[j], ptr[j + 1]):
            ym[idx[t]] += d * yval[t]
        beta[j] += d
        dec_total -= dl
        if abs(d) < 1e-7:
            break
    return dec_total


@nb.njit(cache=False)
def intercept_step(ym, b0, y, c, nsteps):
    n = ym.shape[0]
    dec_total = 0.0
    for _ in range(nsteps):
        g = 0.0
        h = 0.0
        for i in range(n):
            p = _sig(-ym[i])
            g -= c[i] * y[i] * p
            h += c[i] * p * (1.0 - p)
        if h < 1e-12:
            h = 1e-12
        d = -g / h
        if abs(d) < 1e-10:
            break
        for _bt in range(30):
            dl = 0.0
            for i in range(n):
                dl += c[i] * (_lrow(ym[i] + d * y[i]) - _lrow(ym[i]))
            if dl <= 0.0:
                break
            d *= 0.5
        if dl > 0.0:
            break
        for i in range(n):
            ym[i] += d * y[i]
        b0[0] += d
        dec_total -= dl
        if abs(d) < 1e-7:
            break
    return dec_total


@nb.njit(cache=False)
def finetune(ym, b0, beta, support, X, y, c, lb, ub, maxit, tol):
    """Projected Newton on (intercept, beta[support]) with box bounds; updates ym, b0, beta in place."""
    n = ym.shape[0]
    p = support.shape[0] + 1
    w = np.empty(p)
    lo = np.empty(p)
    hi = np.empty(p)
    w[0] = b0[0]
    lo[0] = -1e300
    hi[0] = 1e300
    for q in range(1, p):
        w[q] = beta[support[q - 1]]
        lo[q] = lb[support[q - 1]]
        hi[q] = ub[support[q - 1]]
    Z = np.empty((n, p))
    for i in range(n):
        Z[i, 0] = y[i]
        for q in range(1, p):
            Z[i, q] = y[i] * X[i, support[q - 1]]
    m = np.empty(n)
    for i in range(n):
        s = 0.0
        for q in range(p):
            s += Z[i, q] * w[q]
        m[i] = s
    cur = total_loss(m, c)
    g = np.empty(p)
    H = np.empty((p, p))
    wn = np.empty(p)
    mn = np.empty(n)
    for _ in range(maxit):
        g[:] = 0.0
        H[:, :] = 0.0
        for i in range(n):
            pr = _sig(-m[i])
            gi = -c[i] * pr
            hi_ = c[i] * pr * (1.0 - pr)
            for q in range(p):
                zq = Z[i, q]
                g[q] += gi * zq
                hz = hi_ * zq
                for r in range(q, p):
                    H[q, r] += hz * Z[i, r]
        free = np.ones(p, np.bool_)
        for q in range(p):
            if (w[q] <= lo[q] + 1e-12 and g[q] > 0) or (w[q] >= hi[q] - 1e-12 and g[q] < 0):
                free[q] = False
        fi = np.flatnonzero(free)
        nf = fi.shape[0]
        d = np.zeros(p)
        if nf > 0:
            A = np.empty((nf, nf))
            bb = np.empty(nf)
            for a in range(nf):
                bb[a] = -g[fi[a]]
                for b in range(nf):
                    qa, qb = fi[a], fi[b]
                    A[a, b] = H[min(qa, qb), max(qa, qb)]
                A[a, a] += 1e-10 * (1.0 + A[a, a])
            sol = np.linalg.solve(A, bb)
            for a in range(nf):
                d[fi[a]] = sol[a]
        t = 1.0
        new = cur
        ok = False
        while t > 1e-8:
            for q in range(p):
                v = w[q] + t * d[q]
                wn[q] = min(max(v, lo[q]), hi[q])
            for i in range(n):
                s = 0.0
                for q in range(p):
                    s += Z[i, q] * wn[q]
                mn[i] = s
            new = total_loss(mn, c)
            if new <= cur:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        dec = cur - new
        w[:] = wn
        m[:] = mn
        cur = new
        if dec <= tol * cur:
            break
    b0[0] = w[0]
    for q in range(1, p):
        beta[support[q - 1]] = w[q]
    ym[:] = m
    return cur


@nb.njit(cache=False)
def residual_grads(ym, c, ptr, idx, yval, d, scale):
    """|d loss / d beta_j| * scale_j for every column j."""
    r = np.empty(ym.shape[0])
    for i in range(ym.shape[0]):
        r[i] = c[i] * _sig(-ym[i])
    g = np.empty(d)
    for j in range(d):
        s = 0.0
        for t in range(ptr[j], ptr[j + 1]):
            s += yval[t] * r[idx[t]]
        g[j] = abs(s) * scale[j]
    return g


# ------------------------------------------------------------- data
class Data:
    def __init__(self, X, y01):
        ys = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        Z = np.hstack([X, ys[:, None]])
        U, cnt = np.unique(Z, axis=0, return_counts=True)
        self.X = np.ascontiguousarray(U[:, :-1])
        self.y = np.ascontiguousarray(U[:, -1])
        self.c = cnt.astype(np.float64)
        self.n, self.d = self.X.shape
        self.N = float(self.c.sum())
        ptr = [0]
        idx, yval = [], []
        for j in range(self.d):
            nz = np.flatnonzero(self.X[:, j] != 0)
            idx.append(nz)
            yval.append(self.X[nz, j] * self.y[nz])
            ptr.append(ptr[-1] + len(nz))
        self.ptr = np.array(ptr, dtype=np.int64)
        self.idx = np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, np.int64)
        self.yval = np.concatenate(yval).astype(np.float64) if yval else np.zeros(0)
        self.xval = self.yval * self.y[self.idx]
        self.XT = np.ascontiguousarray(self.X.T)
        # per column: distinct nonzero values (codes) when there are few, else the column is 'raw'
        vptr, vals, xcode, raw = [0], [], np.zeros(len(self.idx), np.int64), np.zeros(self.d, np.bool_)
        for j in range(self.d):
            xs = self.xval[self.ptr[j]:self.ptr[j + 1]]
            u, code = np.unique(xs, return_inverse=True)
            if len(u) > 32:
                raw[j] = True
                u = u[:0]
            else:
                xcode[self.ptr[j]:self.ptr[j + 1]] = code
            vals.append(u)
            vptr.append(vptr[-1] + len(u))
        self.vptr = np.array(vptr, np.int64)
        self.vals = np.concatenate(vals).astype(np.float64)
        self.xcode, self.raw = xcode, raw
        self.maxnv = int(max(1, np.max(np.diff(self.vptr))))
        w = self.c / self.N
        mean = w @ self.X
        var = w @ (self.X ** 2) - mean ** 2
        self.norm = np.sqrt(np.maximum(var, 0.0) * self.N)  # centred column norm, as in FasterRisk
        self.valid = self.norm > 1e-9
        self.scale = np.where(self.valid, 1.0 / np.maximum(self.norm, 1e-12), 0.0)
        self.lb = -COEF_BOUND * np.ones(self.d)
        self.ub = COEF_BOUND * np.ones(self.d)


def base_state(D):
    npos = D.c[D.y > 0].sum()
    b = np.log(npos / (D.N - npos))
    return D.y * b, b


# ------------------------------------------------------------- beam search
def beam_search(D, k, parent_size=10, child_size=10, ft_mult=1.0, ft_tol=1e-10):
    ym0, b00 = base_state(D)
    parents = [(total_loss(ym0, D.c), ym0, np.array([b00]), np.zeros(D.d))]
    seen = set()
    for _ in range(min(k, int(D.valid.sum()))):
        children = []
        for loss_p, ym_p, b0_p, beta_p in parents:
            g = residual_grads(ym_p, D.c, D.ptr, D.idx, D.yval, D.d, D.scale)
            g[beta_p != 0] = -1
            cand = np.argsort(-g)[:child_size]
            cand = cand[g[cand] > 0]
            sup_p = np.flatnonzero(beta_p)
            for j in cand:
                key = tuple(sorted(list(sup_p) + [int(j)]))
                if key in seen:
                    continue
                seen.add(key)
                ym, b0, beta = ym_p.copy(), b0_p.copy(), beta_p.copy()
                loss = loss_p - coord_step(j, ym, beta, D.ptr, D.idx, D.yval, D.c, D.lb[j], D.ub[j], 10)
                loss -= intercept_step(ym, b0, D.y, D.c, 1)
                children.append([loss, ym, b0, beta, key])
        if not children:
            break
        # screen children by the loss after fitting only the new coordinate; finetune the best few
        children.sort(key=lambda t: t[0])
        children = children[:max(parent_size, int(ft_mult * parent_size))]
        for ch in children:
            support = np.array(ch[4], dtype=np.int64)
            ch[0] = finetune(ch[1], ch[2], ch[3], support, D.X, D.y, D.c, D.lb, D.ub, 50, ft_tol)
        children.sort(key=lambda t: t[0])
        parents = [tuple(ch[:4]) for ch in children[:parent_size]]
    return parents


# ------------------------------------------------------------ diverse pool
def diverse_pool(D, best, gap_tolerance=0.05, select_top_m=50, max_attempts=50):
    loss_b, ym_b, b0_b, beta_b = best
    sup = np.flatnonzero(beta_b)
    pool = [(loss_b, b0_b[0], beta_b.copy())]
    for j in sup:
        ym_r = ym_b.copy()
        beta_r = beta_b.copy()
        for t in range(D.ptr[j], D.ptr[j + 1]):
            ym_r[D.idx[t]] -= beta_r[j] * D.yval[t]
        beta_r[j] = 0.0
        loss_r = total_loss(ym_r, D.c)
        g = residual_grads(ym_r, D.c, D.ptr, D.idx, D.yval, D.d, D.scale)
        g[beta_r != 0] = -1
        g[j] = -1
        cand = np.argsort(-g)[:max_attempts]
        cand = cand[g[cand] > 0]
        for jn in cand:
            ym, beta, b0 = ym_r.copy(), beta_r.copy(), b0_b.copy()
            loss = loss_r - coord_step(jn, ym, beta, D.ptr, D.idx, D.yval, D.c, D.lb[jn], D.ub[jn], 10)
            if (loss - loss_b) / loss_b < gap_tolerance:
                sup_new = np.sort(np.where(sup == j, jn, sup))
                loss = finetune(ym, b0, beta, sup_new, D.X, D.y, D.c, D.lb, D.ub, 50, 1e-10)
                pool.append((loss, b0[0], beta))
    pool.sort(key=lambda t: t[0])
    return pool[:select_top_m]


# ---------------------------------------------------------- star-ray search
@nb.njit(cache=False)
def sequential_round(betas, yx, c):
    """FasterRisk's sequential rounding (auxiliary loss), rows weighted by counts. yx = y * [1, X_S] / mult."""
    n, p = yx.shape
    floor = np.floor(betas)
    ceil = np.ceil(betas)
    d_floor = floor - betas
    d_ceil = ceil - betas
    todo = np.zeros(p, np.bool_)
    for j in range(p):
        todo[j] = floor[j] != ceil[j]
    lyx = np.empty((n, p))
    for i in range(n):
        s = 0.0
        for j in range(p):
            gam = floor[j] + (1.0 if yx[i, j] <= 0 else 0.0)
            s += yx[i, j] * gam
        l = 1.0 / (1.0 + np.exp(s))
        for j in range(p):
            lyx[i, j] = l * yx[i, j]
    lyx_sq = np.zeros(p)
    for j in range(p):
        for i in range(n):
            lyx_sq[j] += c[i] * lyx[i, j] * lyx[i, j]
    diff = np.zeros(n)
    cur = 0.0
    out = betas.copy()
    while True:
        best_v = 1e300
        best_j = -1
        best_up = False
        for j in range(p):
            if not todo[j]:
                continue
            expect = cur - lyx_sq[j] * d_floor[j] * d_ceil[j]
            u = 0.0
            for i in range(n):
                t = diff[i] + d_ceil[j] * lyx[i, j]
                u += c[i] * t * t
            if u < best_v:
                best_v, best_j, best_up = u, j, True
            if u > expect:
                u2 = 0.0
                for i in range(n):
                    t = diff[i] + d_floor[j] * lyx[i, j]
                    u2 += c[i] * t * t
                if u2 < best_v:
                    best_v, best_j, best_up = u2, j, False
        if best_j < 0:
            break
        cur = best_v
        stp = d_ceil[best_j] if best_up else d_floor[best_j]
        out[best_j] += stp
        for i in range(n):
            diff[i] += stp * lyx[i, best_j]
        todo[best_j] = False
    return out


@nb.njit(cache=False)
def wloss_lin(yx, w, c):
    s = 0.0
    for i in range(yx.shape[0]):
        m = 0.0
        for j in range(yx.shape[1]):
            m += yx[i, j] * w[j]
        s += c[i] * _lrow(m)
    return s


def line_search(D, b0, beta, num_ray_search=20, early_stop_tolerance=0.001):
    idx = np.flatnonzero(np.abs(beta) > 1e-9)
    yx = D.y[:, None] * np.hstack([np.ones((D.n, 1)), D.X[:, idx]])
    b = np.concatenate([[b0], beta[idx]])
    ub = np.concatenate([[100.0], D.ub[idx]])
    lb = np.concatenate([[-100.0], D.lb[idx]])
    loss_cont = wloss_lin(yx, b, D.c)
    pos, neg = b > 1e-8, b < -1e-8
    largest = 1e8
    if pos.any():
        largest = min(largest, np.min(ub[pos] / b[pos]))
    if neg.any():
        largest = min(largest, np.min(lb[neg] / b[neg]))
    mults = np.linspace(1, largest, num_ray_search) if largest > 1 else np.linspace(1, 0.5, num_ray_search)
    best = (1e300, None)
    for mult in mults:
        r = sequential_round(b * mult, yx / mult, D.c)
        loss = wloss_lin(yx, r / mult, D.c)
        if loss < best[0]:
            best = (loss, r)
        if (loss - loss_cont) / loss_cont < early_stop_tolerance:
            break
    w = np.zeros(D.d)
    w[idx] = best[1][1:]
    return w


# ---------------------------------------------------- calibrated integer loss
@nb.njit(cache=False)
def calibrate(s, y, c, a, b):
    """min over (a, b) of sum c log(1 + exp(-y (a s + b))), damped Newton from (a, b)."""
    n = s.shape[0]
    cur = 0.0
    for i in range(n):
        cur += c[i] * _lrow(y[i] * (a * s[i] + b))
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for i in range(n):
            z = y[i] * (a * s[i] + b)
            p = _sig(-z)
            w = c[i] * p * (1.0 - p)
            gi = -c[i] * y[i] * p
            ga += gi * s[i]
            gb += gi
            haa += w * s[i] * s[i]
            hab += w * s[i]
            hbb += w
        haa += 1e-12
        hbb += 1e-12
        det = haa * hbb - hab * hab
        if det <= 1e-18 * (haa * hbb + 1e-300):
            da = 0.0
            db = gb / hbb
        else:
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
        dec = ga * da + gb * db
        t = 1.0
        new = cur
        while t > 1e-10:
            na, nb_ = a - t * da, b - t * db
            new = 0.0
            for i in range(n):
                new += c[i] * _lrow(y[i] * (na * s[i] + nb_))
            if new <= cur - 1e-4 * t * dec:
                break
            t *= 0.5
        if t <= 1e-10:
            break
        a, b = na, nb_
        improv = cur - new
        cur = new
        if improv < 1e-12 * (1.0 + cur):
            break
    return cur, a, b


def calibrated_loss(D, w):
    s = D.X @ w
    if np.ptp(s) == 0:
        npos = D.c[D.y > 0].sum()
        p = npos / D.N
        return -(npos * np.log(p) + (D.N - npos) * np.log(1 - p)) / D.N
    sd = s.std() + 1e-12
    loss, _, _ = calibrate(s / sd, D.y, D.c, 1.0, 0.0)
    return loss / D.N


# ------------------------------------------------ integer local search (ILS)
@nb.njit(cache=False)
def row_stats(s, y, c, a, b, l, p, w):
    """Per-row loss l, p = sigmoid(-z), curvature w at z = y (a s + b); returns totals
    (L, Ga, Gb, Haa, Hab, Hbb) of the loss and its gradient / Hessian in (a, b)."""
    T = np.zeros(6)
    for i in range(s.shape[0]):
        z = y[i] * (a * s[i] + b)
        e = np.exp(-abs(z))
        if z > 0:
            li = np.log1p(e)
            pi = e / (1.0 + e)
        else:
            li = -z + np.log1p(e)
            pi = 1.0 / (1.0 + e)
        l[i] = c[i] * li
        p[i] = pi
        w[i] = c[i] * pi * (1.0 - pi)
        gi = -c[i] * y[i] * pi
        T[0] += l[i]
        T[1] += gi * s[i]
        T[2] += gi
        T[3] += w[i] * s[i] * s[i]
        T[4] += w[i] * s[i]
        T[5] += w[i]
    return T


@nb.njit(cache=False)
def eval_moves(cols, deltas, s, l, p, w, T, y, c, a, b, ptr, idx, xv, out):
    """out[q, r] = estimated calibrated loss after s += deltas[r] * x_cols[q]: exact loss at the
    current (a, b) minus one Newton step in (a, b), all updated from the rows the move touches."""
    nd = deltas.shape[0]
    dL = np.empty(nd)
    dGa = np.empty(nd)
    dGb = np.empty(nd)
    dHaa = np.empty(nd)
    dHab = np.empty(nd)
    dHbb = np.empty(nd)
    for q in range(cols.shape[0]):
        j = cols[q]
        dL[:] = 0.0
        dGa[:] = 0.0
        dGb[:] = 0.0
        dHaa[:] = 0.0
        dHab[:] = 0.0
        dHbb[:] = 0.0
        for t in range(ptr[j], ptr[j + 1]):
            i = idx[t]
            x = xv[t]
            si = s[i]
            yi = y[i]
            ci = c[i]
            g0 = -ci * yi * p[i]
            w0 = w[i]
            for r in range(nd):
                sn = si + deltas[r] * x
                z = yi * (a * sn + b)
                e = np.exp(-abs(z))
                if z > 0:
                    ln = np.log1p(e)
                    pn = e / (1.0 + e)
                else:
                    ln = -z + np.log1p(e)
                    pn = 1.0 / (1.0 + e)
                wn = ci * pn * (1.0 - pn)
                gn = -ci * yi * pn
                dL[r] += ci * ln - l[i]
                dGa[r] += gn * sn - g0 * si
                dGb[r] += gn - g0
                dHaa[r] += wn * sn * sn - w0 * si * si
                dHab[r] += wn * sn - w0 * si
                dHbb[r] += wn - w0
        for r in range(nd):
            ga = T[1] + dGa[r]
            gb = T[2] + dGb[r]
            haa = T[3] + dHaa[r] + 1e-12
            hab = T[4] + dHab[r]
            hbb = T[5] + dHbb[r] + 1e-12
            det = haa * hbb - hab * hab
            if det > 1e-14 * haa * hbb:
                da = (hbb * ga - hab * gb) / det
                db = (haa * gb - hab * ga) / det
            else:
                da = 0.0
                db = gb / hbb
            # trust region on the Newton step: |da| <= |a| / 2, |db| <= 1
            t = 1.0
            if abs(da) * t > 0.5 * abs(a) + 1e-300:
                t = 0.5 * abs(a) / abs(da)
            if abs(db) * t > 1.0:
                t = 1.0 / abs(db)
            gd = ga * da + gb * db
            out[q, r] = T[0] + dL[r] - (t - 0.5 * t * t) * gd
    return out


def calib_from_scratch(D, s):
    npos = D.c[D.y > 0].sum()
    base = -(npos * np.log(npos / D.N) + (D.N - npos) * np.log(1 - npos / D.N))
    if np.ptp(s) == 0:
        return base, 0.0, np.log(npos / (D.N - npos))
    sd = s.std()
    loss, a, b = calibrate(s / sd, D.y, D.c, 0.0, np.log(npos / (D.N - npos)))
    return loss, a / sd, b


@nb.njit(cache=False)
def bin_scores(s, y, c):
    """Group rows by distinct score: inv (row -> bin), bin scores sv, weights of y=+1 (Wp) and y=-1 (Wn)."""
    n = s.shape[0]
    order = np.argsort(s)
    inv = np.empty(n, np.int64)
    sv = np.empty(n)
    Wp = np.zeros(n)
    Wn = np.zeros(n)
    nb_ = -1
    last = 0.0
    for t in range(n):
        i = order[t]
        if t == 0 or s[i] != last:
            nb_ += 1
            sv[nb_] = s[i]
            last = s[i]
        inv[i] = nb_
        if y[i] > 0:
            Wp[nb_] += c[i]
        else:
            Wn[nb_] += c[i]
    nb_ += 1
    return inv, sv[:nb_].copy(), Wp[:nb_].copy(), Wn[:nb_].copy()


@nb.njit(cache=False)
def calibrate_bins(sv, Wp, Wn, a, b):
    m = sv.shape[0]
    s2 = np.empty(2 * m)
    y2 = np.empty(2 * m)
    c2 = np.empty(2 * m)
    for i in range(m):
        s2[i] = sv[i]
        y2[i] = 1.0
        c2[i] = Wp[i]
        s2[m + i] = sv[i]
        y2[m + i] = -1.0
        c2[m + i] = Wn[i]
    return calibrate(s2, y2, c2, a, b)


@nb.njit(cache=False)
def bin_stats(sv, Wp, Wn, a, b, lp, ln_, pp, pn, wq):
    T = np.zeros(6)
    for q in range(sv.shape[0]):
        z = a * sv[q] + b
        e = np.exp(-abs(z))
        l1 = np.log1p(e)
        if z > 0:
            lp[q] = l1
            ln_[q] = z + l1
            pp[q] = e / (1.0 + e)
            pn[q] = 1.0 / (1.0 + e)
        else:
            lp[q] = -z + l1
            ln_[q] = l1
            pp[q] = 1.0 / (1.0 + e)
            pn[q] = e / (1.0 + e)
        wq[q] = pp[q] * pn[q]
        g = -Wp[q] * pp[q] + Wn[q] * pn[q]
        wt = (Wp[q] + Wn[q]) * wq[q]
        T[0] += Wp[q] * lp[q] + Wn[q] * ln_[q]
        T[1] += g * sv[q]
        T[2] += g
        T[3] += wt * sv[q] * sv[q]
        T[4] += wt * sv[q]
        T[5] += wt
    return T


@nb.njit(cache=False)
def eval_binned(cols, deltas, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, ptr, idx, xv, xcode, vptr, vals,
                raw, out, sp, sn, touched):
    """out[q, r] = estimated calibrated loss after s += deltas[r] * x_cols[q]. Rows are grouped into cells
    (score bin, value of x_j); each cell costs one exp per delta. The estimate is the exact loss at the
    current (a, b) minus a trust-region Newton step in (a, b)."""
    nd = deltas.shape[0]
    dL = np.empty(nd)
    dGa = np.empty(nd)
    dGb = np.empty(nd)
    dHaa = np.empty(nd)
    dHab = np.empty(nd)
    dHbb = np.empty(nd)
    for q in range(cols.shape[0]):
        j = cols[q]
        dL[:] = 0.0
        dGa[:] = 0.0
        dGb[:] = 0.0
        dHaa[:] = 0.0
        dHab[:] = 0.0
        dHbb[:] = 0.0
        nt = 0
        if raw[j]:
            for t in range(ptr[j], ptr[j + 1]):
                touched[nt] = t
                nt += 1
        else:
            nv = vptr[j + 1] - vptr[j]
            for t in range(ptr[j], ptr[j + 1]):
                i = idx[t]
                key = inv[i] * nv + xcode[t]
                if sp[key] == 0.0 and sn[key] == 0.0:
                    touched[nt] = key
                    nt += 1
                if y[i] > 0:
                    sp[key] += c[i]
                else:
                    sn[key] += c[i]
        for u in range(nt):
            if raw[j]:
                t = touched[u]
                i = idx[t]
                bq = inv[i]
                x = xv[t]
                wpc = c[i] if y[i] > 0 else 0.0
                wnc = c[i] - wpc
            else:
                key = touched[u]
                nv = vptr[j + 1] - vptr[j]
                bq = key // nv
                x = vals[vptr[j] + key - bq * nv]
                wpc = sp[key]
                wnc = sn[key]
                sp[key] = 0.0
                sn[key] = 0.0
            so = sv[bq]
            lo = wpc * lp[bq] + wnc * ln_[bq]
            go = -wpc * pp[bq] + wnc * pn[bq]
            wo = (wpc + wnc) * wq[bq]
            for r in range(nd):
                sn_ = so + deltas[r] * x
                z = a * sn_ + b
                e = np.exp(-abs(z))
                l1 = np.log1p(e)
                if z > 0:
                    lpn = l1
                    lnn = z + l1
                    ppn = e / (1.0 + e)
                    pnn = 1.0 / (1.0 + e)
                else:
                    lpn = -z + l1
                    lnn = l1
                    ppn = 1.0 / (1.0 + e)
                    pnn = e / (1.0 + e)
                gn = -wpc * ppn + wnc * pnn
                wn = (wpc + wnc) * ppn * pnn
                dL[r] += wpc * lpn + wnc * lnn - lo
                dGa[r] += gn * sn_ - go * so
                dGb[r] += gn - go
                dHaa[r] += wn * sn_ * sn_ - wo * so * so
                dHab[r] += wn * sn_ - wo * so
                dHbb[r] += wn - wo
        for r in range(nd):
            ga = T[1] + dGa[r]
            gb = T[2] + dGb[r]
            haa = T[3] + dHaa[r] + 1e-12
            hab = T[4] + dHab[r]
            hbb = T[5] + dHbb[r] + 1e-12
            det = haa * hbb - hab * hab
            if det > 1e-14 * haa * hbb:
                da = (hbb * ga - hab * gb) / det
                db = (haa * gb - hab * ga) / det
            else:
                da = 0.0
                db = gb / hbb
            t = 1.0
            if abs(da) * t > 0.5 * abs(a) + 1e-300:
                t = 0.5 * abs(a) / abs(da)
            if abs(db) * t > 1.0:
                t = 1.0 / abs(db)
            gd = ga * da + gb * db
            out[q, r] = T[0] + dL[r] - (t - 0.5 * t * t) * gd
    return out


class ScoreState:
    """Scores s = X w grouped into bins, with the calibrated (a, b) and per-bin statistics."""

    def __init__(self, D, s, a=None, b=None):
        self.s = s
        self.inv, self.sv, self.Wp, self.Wn = bin_scores(s, D.y, D.c)
        if a is None:
            if np.ptp(s) == 0:
                npos = D.c[D.y > 0].sum()
                self.L, self.a, self.b = calibrate_bins(self.sv, self.Wp, self.Wn, 0.0, np.log(npos / (D.N - npos)))
            else:
                sd = s.std()
                L, a, b = calibrate_bins(self.sv / sd, self.Wp, self.Wn, 0.0, 0.0)
                self.L, self.a, self.b = L, a / sd, b
        elif a == "keep":
            pass
        else:
            self.L, self.a, self.b = calibrate_bins(self.sv, self.Wp, self.Wn, a, b)

    def stats(self, a, b):
        m = self.sv.shape[0]
        self.lp, self.ln, self.pp, self.pn, self.wq = (np.empty(m) for _ in range(5))
        self.T = bin_stats(self.sv, self.Wp, self.Wn, a, b, self.lp, self.ln, self.pp, self.pn, self.wq)

    def eval(self, D, cols, deltas, a, b, maxnv):
        out = np.empty((len(cols), len(deltas)))
        size = max(self.sv.shape[0] * maxnv, 1)
        sp, sn = np.zeros(size), np.zeros(size)
        touched = np.empty(max(size, D.n), np.int64)
        eval_binned(cols, deltas, self.inv, self.sv, self.lp, self.ln, self.pp, self.pn, self.wq, self.T, a, b,
                    D.y, D.c, D.ptr, D.idx, D.xval, D.xcode, D.vptr, D.vals, D.raw, out, sp, sn, touched)
        return out


class ILS:
    """Best-improvement local search over integer points, scored by the calibrated loss."""

    def __init__(self, D, k, n_exact=3):
        self.D, self.k, self.n_exact = D, k, n_exact
        self.nevals = 0

    def run(self, w, max_iter=100):
        D, k = self.D, self.k
        w = w.astype(np.float64).copy()
        st = ScoreState(D, D.X @ w)
        allv = np.arange(-COEF_BOUND, COEF_BOUND + 1, dtype=np.float64)
        nzv = allv[allv != 0]
        for _ in range(max_iter):
            a, b = st.a, st.b
            st.stats(a, b)
            S = np.flatnonzero(w)
            cands = []  # (est, remove_j, add_j, new_value_of_add_j)
            for j in S:
                dl = allv - w[j]
                out = st.eval(D, np.array([j]), dl, a, b, D.maxnv)
                for r, v in enumerate(allv):
                    if v != w[j]:
                        cands.append((out[0, r], -1, j, v))
            nonS = np.flatnonzero((w == 0) & D.valid)
            if len(S) < k and len(nonS):
                out = st.eval(D, nonS, nzv, a, b, D.maxnv)
                for q, j in enumerate(nonS):
                    for r, v in enumerate(nzv):
                        cands.append((out[q, r], -1, j, v))
            if len(nonS):
                for j in S:
                    st2 = ScoreState(D, st.s - w[j] * D.XT[j], "keep")
                    st2.stats(a, b)
                    out = st2.eval(D, nonS, nzv, a, b, D.maxnv)
                    order = np.argsort(out, axis=None)[:self.n_exact]
                    for f in order:
                        q, r = divmod(int(f), len(nzv))
                        cands.append((out[q, r], j, nonS[q], nzv[r]))
            cands.sort(key=lambda t: t[0])
            best = None
            for est, rj, aj, v in cands[:self.n_exact * 2]:
                s2 = st.s.copy()
                if rj >= 0:
                    s2 -= w[rj] * D.XT[rj]
                s2 += (v - w[aj]) * D.XT[aj]
                if np.ptp(s2) == 0:
                    continue
                self.nevals += 1
                st2 = ScoreState(D, s2, a, b)
                if st2.L < st.L - 1e-9 * st.L and (best is None or st2.L < best[0].L):
                    best = (st2, rj, aj, v)
            if best is None:
                break
            st, rj, aj, v = best
            if rj >= 0:
                w[rj] = 0
            w[aj] = v
        return st.L / D.N, w


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, select_top_m=50,
                 gap_tolerance=0.05, max_attempts=50, num_ray_search=20, n_starts=5, use_pool=False):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.select_top_m, self.gap_tolerance = select_top_m, gap_tolerance
        self.max_attempts, self.num_ray_search = max_attempts, num_ray_search
        self.n_starts, self.use_pool = n_starts, use_pool

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        D = Data(X, y)
        parents = beam_search(D, self.k, self.parent_size, self.child_size)
        if self.use_pool:
            pool = diverse_pool(D, parents[0], self.gap_tolerance, self.select_top_m, self.max_attempts)
        else:
            pool = [(l, b0[0], beta) for l, _, b0, beta in parents]
        seen = set()
        starts = []
        best_w, best_l = np.zeros(D.d), np.inf
        for _, b0, beta in pool:
            w = np.round(line_search(D, b0, beta, self.num_ray_search))
            key = w.tobytes()
            if key in seen:
                continue
            seen.add(key)
            l = calibrated_loss(D, w)
            starts.append((l, w))
            if l < best_l:
                best_l, best_w = l, w
        # integer local search from the best few distinct rounded solutions
        ils = ILS(D, self.k)
        starts.sort(key=lambda t: t[0])
        for l0, w0 in starts[:self.n_starts]:
            l, w = ils.run(w0)
            if l < best_l:
                best_l, best_w = l, w
        self.coef_ = np.clip(np.round(best_w), -COEF_BOUND, COEF_BOUND)
        self.intercept_, self.multiplier_ = 0.0, 1.0
        self.train_loss_ = best_l
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
