"""Exact track: sparse integer risk scores with a proof of optimality. The file the agent edits.

Starting point exact_v1 = the shipped heuristic v35_scratch (run sep26-run2), unchanged, for the
incumbent, plus the certification stage at the end of this file (see ``certify``). ``fit`` sets
``lower_bound_``: a PROVEN lower bound on the minimum calibrated loss over every feasible point
vector. The harness counts a problem as certified when lower_bound_ meets the returned loss.

The rest of this docstring describes the heuristic:

Pipeline (see notes.md in the run folder for the history and the measured effect of each part):

1. data: unique (x, y) rows with counts; per column the CSC nonzeros and codes of its distinct values;
2. continuous beam search (FasterRisk's: grow the support one column at a time from the 10 best parents by
   the 10 largest gradients), where every node keeps its rows grouped by (y, x_support): a child's groups are
   the parent's groups split by x_j, so its projected-Newton fit runs on a few hundred groups instead of all
   rows and only needs the nonzeros of column j;
3. rounding: for each of the 10 final supports, round m * beta over a grid of scales and keep the rounding
   with the smallest calibrated loss (the harness's criterion: min over a, b of the log loss of a * score + b);
4. integer local search from the 5 best roundings: value changes and additions, then swaps when those fail.
   Moves are scored on cells (score bin, x_j value) by an estimate of the calibrated loss (exact at the
   current (a, b), then one Newton step in (a, b) with the touched cells exact); the best two are checked
   exactly. Swap-in columns are screened by a second-order model (two BLAS mat-vecs).

Interface (fixed): ``make_model(k, time_limit)`` returns an estimator whose
``fit(X, y)`` (y in {0, 1}) leaves integer points in ``coef_`` with at most ``k``
nonzero entries in [-5, 5]. The harness scores ``coef_`` alone.
"""

from __future__ import annotations

import argparse
import os
import sys
import itertools
from math import comb
import time

import numpy as np
import numba as nb

MODEL_NAME = "x6_wald"
DESCRIPTION = ("x5 + Wald-statistic candidate order at internal nodes, no recomputation of a node's own family bound "
               "for its first child, family LR gives up after 2 Newton steps when the Newton model predicts an optimum below UB")
INTEGER = True
COEF_BOUND = 5
TR_A, TR_B = 1.0, 2.0  # trust region of the (a, b) Newton step in the move estimate


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
def column_codes(XT, y, max_vals):
    """Per column: rank codes of all values, CSC nonzeros, codes among distinct nonzero values."""
    d, n = XT.shape
    ncode = np.ones(d, np.int64)
    rank = np.empty(n, np.int64)
    isbin = np.zeros(d, np.bool_)
    cnz = np.zeros(d, np.int64)
    has0 = np.zeros(d, np.bool_)
    for j in range(d):
        col = XT[j]
        b = True
        nz = 0
        for i in range(n):
            v = col[i]
            if v != 0.0:
                nz += 1
                if v != 1.0:
                    b = False
        isbin[j] = b
        cnz[j] = nz
        has0[j] = nz < n
    nnz = cnz.sum()
    ptr = np.zeros(d + 1, np.int64)
    for j in range(d):
        ptr[j + 1] = ptr[j] + cnz[j]
    idx = np.empty(nnz, np.int64)
    xval = np.empty(nnz)
    xcode = np.zeros(nnz, np.int64)
    raw = np.zeros(d, np.bool_)
    vals_all = np.empty(nnz + d)
    vptr = np.zeros(d + 1, np.int64)
    uvals = np.empty(n)
    for j in range(d):
        col = XT[j]
        t = ptr[j]
        if isbin[j]:
            h0 = has0[j]
            h1 = cnz[j] > 0
            ncode[j] = (1 if h0 else 0) + (1 if h1 else 0)
            if h1:
                vals_all[vptr[j]] = 1.0
                vptr[j + 1] = vptr[j] + 1
            else:
                vptr[j + 1] = vptr[j]
            for i in range(n):
                if col[i] != 0.0:
                    idx[t] = i
                    xval[t] = 1.0
                    t += 1
            continue
        order = np.argsort(col)
        nu = 0
        for r in range(n):
            i = order[r]
            if r == 0 or col[i] != uvals[nu - 1]:
                uvals[nu] = col[i]
                nu += 1
            rank[i] = nu - 1
        ncode[j] = nu
        zpos = -1
        for q in range(nu):
            if uvals[q] == 0.0:
                zpos = q
        nnzv = nu - (1 if zpos >= 0 else 0)
        if nnzv > max_vals:
            raw[j] = True
            vptr[j + 1] = vptr[j]
        else:
            for q in range(nu):
                if q != zpos:
                    vals_all[vptr[j] + (q if (zpos < 0 or q < zpos) else q - 1)] = uvals[q]
            vptr[j + 1] = vptr[j] + nnzv
        for i in range(n):
            if col[i] != 0.0:
                idx[t] = i
                xval[t] = col[i]
                if not raw[j]:
                    cq = rank[i]
                    xcode[t] = cq if (zpos < 0 or cq < zpos) else cq - 1
                t += 1
    return ncode, isbin, ptr, idx, xval, xcode, raw, vals_all[:vptr[d]].copy(), vptr


@nb.njit(cache=False)
def col_moments(XT, w):
    d, n = XT.shape
    mean = np.zeros(d)
    var = np.zeros(d)
    for j in range(d):
        m1 = 0.0
        m2 = 0.0
        for i in range(n):
            v = XT[j, i]
            m1 += w[i] * v
            m2 += w[i] * v * v
        mean[j] = m1
        var[j] = m2 - m1 * m1
    return mean, var


class Data:
    def code(self, j):
        """Rank code of every row's value in column j (computed on first use)."""
        cj = self._codes.get(j)
        if cj is None:
            if self.isbin[j]:
                cj = self.XT[j].astype(np.int64) if self.ncode[j] > 1 else np.zeros(self.n, np.int64)
            else:
                cj = np.unique(self.XT[j], return_inverse=True)[1].astype(np.int64)
            self._codes[j] = cj
        return cj

    @property
    def XT2(self):
        if self._XT2 is None:
            self._XT2 = self.XT if self.isbin.all() else self.XT * self.XT
        return self._XT2

    def __init__(self, X, y01):
        ys = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        # unique (x, y) rows via a random projection hash (deterministic seed)
        r = np.random.default_rng(12345).standard_normal(X.shape[1] + 1)
        h = X @ r[:-1] + ys * r[-1]
        _, first, inv = np.unique(h, return_index=True, return_inverse=True)
        cnt = np.bincount(inv.ravel())
        self.X = np.ascontiguousarray(X[first])
        self.y = np.ascontiguousarray(ys[first])
        self.c = cnt.astype(np.float64)
        self.n, self.d = self.X.shape
        self.N = float(self.c.sum())
        self.XT = np.ascontiguousarray(self.X.T)
        self._XT2 = None
        self.yc = self.y * self.c
        (self.ncode, self.isbin, self.ptr, self.idx, self.xval, self.xcode, self.raw, self.vals,
         self.vptr) = column_codes(self.XT, self.y, 32)
        self._codes = {}
        self.scratch = None
        self.maxnv = int(max(1, np.max(np.diff(self.vptr))))
        mean, var = col_moments(self.XT, self.c / self.N)
        self.norm = np.sqrt(np.maximum(var, 0.0) * self.N)  # centred column norm, as in FasterRisk
        self.valid = self.norm > 1e-9
        self.scale = np.where(self.valid, 1.0 / np.maximum(self.norm, 1e-12), 0.0)
        self.lb = -COEF_BOUND * np.ones(self.d)
        self.ub = COEF_BOUND * np.ones(self.d)


# ------------------------------------------------------------- beam search
@nb.njit(cache=False)
def newton_fit(Z, c, w, lo, hi, maxit, tol):
    """Projected Newton for min sum_g c_g log(1 + exp(-Z_g . w)) with box bounds; w updated in place.
    Returns (loss, margins)."""
    n, p = Z.shape
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
    return cur, m


@nb.njit(cache=False)
def regroup(ginv, ng, code, ncode, y, c):
    """Refine row groups by a column's value code. Returns (inv, ng2, group y, group weight, representative row)."""
    n = ginv.shape[0]
    inv = np.empty(n, np.int64)
    M = ng * ncode
    cnt = 0
    if M <= 4 * n + 4096:
        table = np.full(M, -1, np.int64)
        for i in range(n):
            key = ginv[i] * ncode + code[i]
            t = table[key]
            if t < 0:
                t = cnt
                table[key] = t
                cnt += 1
            inv[i] = t
    else:
        keys = np.empty(n, np.int64)
        for i in range(n):
            keys[i] = ginv[i] * ncode + code[i]
        order = np.argsort(keys)
        last = -1
        for t in range(n):
            i = order[t]
            if t == 0 or keys[i] != last:
                cnt += 1
                last = keys[i]
            inv[i] = cnt - 1
    gy = np.empty(cnt)
    gc = np.zeros(cnt)
    rep = np.full(cnt, -1, np.int64)
    for i in range(n):
        g = inv[i]
        gc[g] += c[i]
        if rep[g] < 0:
            rep[g] = i
            gy[g] = y[i]
    return inv, cnt, gy, gc, rep


class Node:
    """A beam state: support, continuous (b0, beta) and the rows grouped by (y, x_support)."""
    __slots__ = ("loss", "w", "S", "inv", "ng", "gy", "gc", "rep", "mg", "par", "j")


@nb.njit(cache=False)
def group_design(X, gy, rep, S):
    ng = rep.shape[0]
    p = S.shape[0] + 1
    Z = np.empty((ng, p))
    for g in range(ng):
        Z[g, 0] = gy[g]
        for q in range(1, p):
            Z[g, q] = gy[g] * X[rep[g], S[q - 1]]
    return Z


@nb.njit(cache=False)
def child_fit_sparse(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, j, ptr, idx, xcode, vptr, vals, c, X,
                     bound, tol):
    """Fit a child (parent support + column j) without touching every row: the child's groups are the parent's
    groups split by the value of x_j, and their weights come from the nonzeros of column j only."""
    nv = vptr[j + 1] - vptr[j] + 1  # slot 0: x_j = 0
    W = np.zeros(par_ng * nv)
    for t in range(ptr[j], ptr[j + 1]):
        i = idx[t]
        W[par_inv[i] * nv + 1 + xcode[t]] += c[i]
    for g in range(par_ng):
        rest = par_gc[g]
        for v in range(1, nv):
            rest -= W[g * nv + v]
        W[g * nv] = rest if rest > 0.5 else 0.0
    ncell = 0
    for g in range(par_ng):
        for v in range(nv):
            if W[g * nv + v] > 0.0:
                ncell += 1
    ps = par_S.shape[0]
    S = np.empty(ps + 1, np.int64)
    w = np.zeros(ps + 2)
    w[0] = par_w[0]
    pos = 0
    q = 0
    ins = False
    for r in range(ps + 1):
        if not ins and (q >= ps or j < par_S[q]):
            S[r] = j
            pos = r
            ins = True
        else:
            S[r] = par_S[q]
            w[r + 1] = par_w[q + 1]
            q += 1
    Z = np.empty((ncell, ps + 2))
    cw = np.empty(ncell)
    u = 0
    for g in range(par_ng):
        for v in range(nv):
            wt = W[g * nv + v]
            if wt <= 0.0:
                continue
            xj = 0.0 if v == 0 else vals[vptr[j] + v - 1]
            yg = par_gy[g]
            Z[u, 0] = yg
            for r in range(ps + 1):
                if r == pos:
                    Z[u, r + 1] = yg * xj
                else:
                    Z[u, r + 1] = yg * X[par_rep[g], S[r]]
            cw[u] = wt
            u += 1
    lo = np.full(ps + 2, -bound)
    hi = np.full(ps + 2, bound)
    lo[0] = -1e300
    hi[0] = 1e300
    loss, _ = newton_fit(Z, cw, w, lo, hi, 50, tol)
    return S, loss, w


@nb.njit(cache=False)
def child_fit_batch(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, js, ptr, idx, xcode, vptr, vals, c, X,
                    bound, tol):
    """child_fit_sparse for several columns of one parent: losses and fitted (b0, beta_S) per child."""
    m = js.shape[0]
    losses = np.empty(m)
    W = np.empty((m, par_S.shape[0] + 2))
    for u in range(m):
        _, l, w = child_fit_sparse(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, js[u], ptr, idx, xcode,
                                   vptr, vals, c, X, bound, tol)
        losses[u] = l
        W[u] = w
    return losses, W


@nb.njit(cache=False)
def make_child(par_inv, par_ng, par_S, par_w, j, colcode_j, ncode_j, y, c, X, bound, tol):
    """Add column j to a parent: refine its row groups and refit (b0, beta_S) by projected Newton."""
    inv, ng, gy, gc, rep = regroup(par_inv, par_ng, colcode_j, ncode_j, y, c)
    ps = par_S.shape[0]
    S = np.empty(ps + 1, np.int64)
    w = np.zeros(ps + 2)
    w[0] = par_w[0]
    q = 0
    ins = False
    for r in range(ps + 1):
        if not ins and (q >= ps or j < par_S[q]):
            S[r] = j
            w[r + 1] = 0.0
            ins = True
        else:
            S[r] = par_S[q]
            w[r + 1] = par_w[q + 1]
            q += 1
    lo = np.full(ps + 2, -bound)
    hi = np.full(ps + 2, bound)
    lo[0] = -1e300
    hi[0] = 1e300
    Z = group_design(X, gy, rep, S)
    loss, mg = newton_fit(Z, gc, w, lo, hi, 50, tol)
    return S, inv, ng, gy, gc, rep, loss, mg, w


def beam_search(D, k, parent_size=10, child_size=10, deadline=np.inf):
    root = Node()
    root.S = ()
    root.inv, root.ng, root.gy, root.gc, root.rep = regroup(np.zeros(D.n, np.int64), 1, (D.y > 0).astype(np.int64), 2, D.y, D.c)
    npos = D.c[D.y > 0].sum()
    root.w = np.array([np.log(npos / (D.N - npos))])
    root.mg = root.gy * root.w[0]
    root.loss = total_loss(root.mg, root.gc)
    parents = [root]
    seen = set()
    bound = float(COEF_BOUND)
    for _ in range(min(k, int(D.valid.sum()))):
        if time.perf_counter() > deadline:
            parents = parents[:1]  # out of time: finish the support greedily from the best parent
        children = []
        # |gradient| of every column for all parents at once (one BLAS mat-mat), scaled by the centred norm
        R = np.empty((D.n, len(parents)))
        for q, par in enumerate(parents):
            R[:, q] = D.yc / (1.0 + np.exp(par.mg[par.inv]))
        Gall = np.abs(D.XT @ R) * D.scale[:, None]
        for q, par in enumerate(parents):
            g = Gall[:, q]
            if par.S:
                g[list(par.S)] = -1
            cand = np.argsort(-g)[:child_size]
            cand = cand[g[cand] > 0]
            new_js, keys = [], []
            for j in cand:
                key = tuple(sorted(par.S + (int(j),)))
                if key not in seen:
                    seen.add(key)
                    new_js.append(int(j))
                    keys.append(key)
            if not new_js:
                continue
            par_S = np.array(par.S, dtype=np.int64)
            js = np.array(new_js, dtype=np.int64)
            sparse = ~D.raw[js]
            if sparse.any():
                # children fitted from the parent's groups split by x_j (no pass over all rows)
                losses, W = child_fit_batch(par.inv, par.ng, par.gy, par.gc, par.rep, par_S, par.w, js[sparse],
                                            D.ptr, D.idx, D.xcode, D.vptr, D.vals, D.c, D.X, bound, 1e-7)
                for u, q in enumerate(np.flatnonzero(sparse)):
                    ch = Node()
                    ch.loss, ch.w, ch.S, ch.inv, ch.par, ch.j = losses[u], W[u], keys[q], None, par, new_js[q]
                    children.append(ch)
            for q in np.flatnonzero(~sparse):
                j = new_js[q]
                ch = Node()
                (S, ch.inv, ch.ng, ch.gy, ch.gc, ch.rep, ch.loss, ch.mg, ch.w) = make_child(
                    par.inv, par.ng, par_S, par.w, j, D.code(j), D.ncode[j], D.y, D.c, D.X, bound, 1e-7)
                ch.S = keys[q]
                children.append(ch)
        if not children:
            break
        children.sort(key=lambda t: t.loss)
        parents = children[:parent_size]
        for ch in parents:
            if ch.inv is None:  # materialise the row groups of the selected children only
                par, j = ch.par, ch.j
                ch.inv, ch.ng, ch.gy, ch.gc, ch.rep = regroup(par.inv, par.ng, D.code(j), D.ncode[j], D.y, D.c)
                ch.mg = group_design(D.X, ch.gy, ch.rep, np.array(ch.S, dtype=np.int64)) @ ch.w
                ch.par = None
    return parents


# ------------------------------------------------------- calibrated rounding
@nb.njit(cache=False)
def calib_round_kernel(XS, gy, gc, beta, n_mult, bound):
    """Round m * beta for a grid of scales (largest point 0.5 .. bound + 0.49); return the rounding with the
    smallest calibrated loss on the row groups (XS: group rows of the support columns)."""
    ng, p = XS.shape
    top = 0.0
    for q in range(p):
        top = max(top, abs(beta[q]))
    best_l = np.inf
    best_r = np.zeros(p)
    prev = np.full(p, np.nan)
    r = np.empty(p)
    sg = np.empty(ng)
    if top < 1e-12:
        return best_r, best_l
    for t in range(n_mult):
        L = 0.5 + (bound - 0.01) * t / max(n_mult - 1, 1)
        same = True
        anynz = False
        for q in range(p):
            v = np.round(beta[q] * L / top)
            v = min(max(v, -bound), bound)
            r[q] = v
            if v != prev[q]:
                same = False
            if v != 0:
                anynz = True
        if same or not anynz:
            continue
        prev[:] = r
        mn = np.inf
        mx = -np.inf
        m1 = 0.0
        m2 = 0.0
        for g in range(ng):
            v = 0.0
            for q in range(p):
                v += XS[g, q] * r[q]
            sg[g] = v
            mn = min(mn, v)
            mx = max(mx, v)
            m1 += gc[g] * v
            m2 += gc[g] * v * v
        if mx == mn:
            continue
        tot = gc.sum()
        sd = np.sqrt(max(m2 / tot - (m1 / tot) ** 2, 1e-300))
        for g in range(ng):
            sg[g] /= sd
        loss, _, _ = calibrate(sg, gy, gc, 0.0, 0.0)
        if loss < best_l:
            best_l = loss
            best_r[:] = r
    return best_r, best_l


def calib_round(D, nd, n_mult=20):
    """Best rounding of the node's beta by the calibrated loss over a grid of scales."""
    S = np.array(nd.S, dtype=np.int64)
    XS = group_design(D.X, np.ones(nd.ng), nd.rep, S)[:, 1:]
    r, l = calib_round_kernel(np.ascontiguousarray(XS), nd.gy, nd.gc, nd.w[1:].copy(), n_mult, float(COEF_BOUND))
    w = np.zeros(D.d)
    w[S] = r
    return w, l / D.N


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


# ------------------------------------------------ integer local search (ILS)
@nb.njit(cache=False)
def bin_scores(s, y, c):
    """Group rows by distinct score: inv (row -> bin), bin scores sv, weights of y=+1 (Wp) and y=-1 (Wn)."""
    n = s.shape[0]
    lo = s[0]
    hi = s[0]
    integral = True
    for i in range(n):
        v = s[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        if integral and v != np.floor(v):
            integral = False
    if integral and hi - lo <= 4 * n + 1024:
        # integer scores in a small range: counting instead of sorting
        R = int(hi - lo) + 1
        cid = np.full(R, -1, np.int64)
        for i in range(n):
            cid[int(s[i] - lo)] = 0
        nb_ = 0
        for r in range(R):
            if cid[r] >= 0:
                cid[r] = nb_
                nb_ += 1
        inv = np.empty(n, np.int64)
        sv = np.empty(nb_)
        Wp = np.zeros(nb_)
        Wn = np.zeros(nb_)
        for r in range(R):
            if cid[r] >= 0:
                sv[cid[r]] = lo + r
        for i in range(n):
            q = cid[int(s[i] - lo)]
            inv[i] = q
            if y[i] > 0:
                Wp[q] += c[i]
            else:
                Wn[q] += c[i]
        return inv, sv, Wp, Wn
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
                raw, out, sp, sn, touched, tr_a, tr_b):
    """out[q, r] = estimated calibrated loss after s += deltas[q, r] * x_cols[q]. Rows are grouped into cells
    (score bin, value of x_j); each cell costs one exp per delta. The estimate is the exact loss at the
    current (a, b) minus a trust-region Newton step in (a, b)."""
    nd = deltas.shape[1]
    cb = np.empty(touched.shape[0], np.int64)
    cx = np.empty(touched.shape[0])
    cwp = np.empty(touched.shape[0])
    cwn = np.empty(touched.shape[0])
    stepa = np.empty(nd)
    stepb = np.empty(nd)
    hyb = np.empty(nd)
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
        # gather the touched cells: bin, x value, weights of y = +1 / -1
        for u in range(nt):
            if raw[j]:
                t = touched[u]
                i = idx[t]
                cb[u] = inv[i]
                cx[u] = xv[t]
                cwp[u] = c[i] if y[i] > 0 else 0.0
                cwn[u] = c[i] - cwp[u]
            else:
                key = touched[u]
                nv = vptr[j + 1] - vptr[j]
                bq = key // nv
                cb[u] = bq
                cx[u] = vals[vptr[j] + key - bq * nv]
                cwp[u] = sp[key]
                cwn[u] = sn[key]
                sp[key] = 0.0
                sn[key] = 0.0
        oL = oGa = oGb = oHaa = oHab = oHbb = 0.0
        for u in range(nt):
            bq = cb[u]
            x = cx[u]
            wpc = cwp[u]
            wnc = cwn[u]
            so = sv[bq]
            lo = wpc * lp[bq] + wnc * ln_[bq]
            go = -wpc * pp[bq] + wnc * pn[bq]
            wo = (wpc + wnc) * wq[bq]
            oL += lo
            oGa += go * so
            oGb += go
            oHaa += wo * so * so
            oHab += wo * so
            oHbb += wo
            for r in range(nd):
                sn_ = so + deltas[q, r] * x
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
            if abs(da) * t > tr_a * abs(a) + 1e-300:
                t = tr_a * abs(a) / abs(da)
            if abs(db) * t > tr_b:
                t = tr_b / abs(db)
            stepa[r] = -t * da
            stepb[r] = -t * db
            out[1, q, r] = T[0] + dL[r]
            hyb[r] = 0.0
        # the untouched rows at the Newton point, by their quadratic model around (a, b)
        UGa = T[1] - oGa
        UGb = T[2] - oGb
        UHaa = T[3] - oHaa
        UHab = T[4] - oHab
        UHbb = T[5] - oHbb
        for r in range(nd):
            hyb[r] = (T[0] - oL + UGa * stepa[r] + UGb * stepb[r]
                      + 0.5 * (UHaa * stepa[r] * stepa[r] + 2 * UHab * stepa[r] * stepb[r] + UHbb * stepb[r] * stepb[r]))
        # the touched cells exactly at the Newton point
        for u in range(nt):
            so = sv[cb[u]]
            x = cx[u]
            wpc = cwp[u]
            wnc = cwn[u]
            for r in range(nd):
                z = (a + stepa[r]) * (so + deltas[q, r] * x) + b + stepb[r]
                l1 = np.log1p(np.exp(-abs(z)))
                if z > 0:
                    hyb[r] += wpc * l1 + wnc * (z + l1)
                else:
                    hyb[r] += wpc * (l1 - z) + wnc * l1
        for r in range(nd):
            out[0, q, r] = min(out[1, q, r], hyb[r])
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

    def screen_rows(self, D):
        """Per-row first and second derivative of the loss in the score at the current (a, b)."""
        return np.where(D.y > 0, -self.pp[self.inv], self.pn[self.inv]) * D.c, self.wq[self.inv] * D.c

    def eval(self, D, cols, deltas, a, b, maxnv, n_screen=None, GH=None):
        """deltas: one row of score changes per column, or one row shared by all columns."""
        if deltas.ndim == 1:
            deltas = np.ascontiguousarray(np.broadcast_to(deltas, (len(cols), len(deltas))))
        if n_screen is not None and len(cols) > n_screen:
            # rank columns by a second-order model at fixed (a, b) (no exp per row), evaluate the best
            if GH is None:
                r_row, h_row = self.screen_rows(D)
                GH = ((D.XT @ r_row)[cols], (D.XT2 @ h_row)[cols])
            G, H = GH
            u = a * deltas
            sc = np.min(np.minimum(0.0, u * G[:, None] + 0.5 * (u * u) * H[:, None]), axis=1)
            pick = np.sort(np.argsort(sc)[:n_screen])
            out = np.full((2,) + deltas.shape, np.inf)
            out[:, pick] = self.eval(D, cols[pick], deltas[pick], a, b, maxnv)
            return out
        out = np.empty((2,) + deltas.shape)  # [estimated calibrated loss, loss at fixed (a, b)]
        size = max(self.sv.shape[0] * maxnv, 1)
        if D.scratch is None or D.scratch[0].shape[0] < size:
            # zeroed scratch shared by all evaluations (the kernel resets every cell it touches)
            m = max(size, D.n)
            D.scratch = (np.zeros(m), np.zeros(m), np.empty(m, np.int64))
        sp, sn, touched = D.scratch
        eval_binned(cols, deltas, self.inv, self.sv, self.lp, self.ln, self.pp, self.pn, self.wq, self.T, a, b,
                    D.y, D.c, D.ptr, D.idx, D.xval, D.xcode, D.vptr, D.vals, D.raw, out, sp, sn, touched, TR_A, TR_B)
        return out


class ILS:
    """Best-improvement local search over integer points, scored by the calibrated loss."""

    def __init__(self, D, k, n_exact=2, n_screen=4, n_rank=1):
        self.D, self.k, self.n_exact, self.n_screen = D, k, n_exact, n_screen
        self.visited = set()
        self.n_rank = n_rank
        self.est_margin = 1e-4
        self.nevals = 0

    def check(self, st, w, cands, a, b):
        """Exact calibrated loss of the best estimated candidates; returns the best improving one or None."""
        D = self.D
        m = self.n_exact * 2
        by_est = sorted(cands, key=lambda t: t[0])[:m]
        by_bound = sorted(cands, key=lambda t: t[4])[:m] if self.n_rank > 1 else []
        todo, seen = [], set()
        for t in by_est + by_bound:
            if (t[1], t[2], t[3]) not in seen:
                seen.add((t[1], t[2], t[3]))
                todo.append(t)
        best = None
        for est, rj, aj, v, _ in todo:
            if est > st.L * (1.0 + self.est_margin):
                continue  # the estimate says the move does not help
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
        return best

    def run(self, w, max_iter=100, deadline=np.inf):
        D, k = self.D, self.k
        w = w.astype(np.float64).copy()
        st = ScoreState(D, D.X @ w)
        allv = np.arange(-COEF_BOUND, COEF_BOUND + 1, dtype=np.float64)
        nzv = allv[allv != 0]
        shifts = np.arange(-2 * COEF_BOUND, 2 * COEF_BOUND + 1, dtype=np.float64)
        for _ in range(max_iter):
            key = w.tobytes()
            if key in self.visited or time.perf_counter() > deadline:
                break  # the search from here is deterministic and was already done
            self.visited.add(key)
            a, b = st.a, st.b
            st.stats(a, b)
            S = np.flatnonzero(w)
            cands = []  # (est, remove_j, add_j, new_value_of_add_j)
            ne = self.n_exact

            def top(out, rj, cols, vals):
                # best moves by the estimate and by the upper bound (loss at the current (a, b))
                picked = set()
                for o in out[:self.n_rank]:
                    flat = o.ravel()
                    m = min(ne, flat.size)
                    for f in np.argpartition(flat, m - 1)[:m]:
                        f = int(f)
                        if f in picked or not np.isfinite(flat[f]):
                            continue
                        picked.add(f)
                        q, r = divmod(f, o.shape[1])
                        cands.append((out[0].flat[f], rj, cols[q], vals[r], out[1].flat[f]))

            if len(S):
                # value changes of support features: every other value in [-5, 5] (0 removes the feature)
                newv = np.array([allv[allv != w[j]] for j in S])
                out = st.eval(D, S, newv - w[S][:, None], a, b, D.maxnv)
                for q, j in enumerate(S):
                    top(out[:, q:q + 1], -1, np.array([j]), newv[q])
            nonS = np.flatnonzero((w == 0) & D.valid)
            if len(S) < k and len(nonS):
                top(st.eval(D, nonS, nzv, a, b, D.maxnv), -1, nonS, nzv)
            best = self.check(st, w, cands, a, b)
            if best is None and len(nonS):
                # only when no value change / addition helps: swaps (remove j, add j2 with a value)
                cands = []
                sts = []
                R = np.empty((D.n, 2 * len(S)))
                for q, j in enumerate(S):
                    st2 = ScoreState(D, st.s - w[j] * D.XT[j], "keep")
                    st2.stats(a, b)
                    sts.append(st2)
                    R[:, 2 * q], R[:, 2 * q + 1] = st2.screen_rows(D)
                # second-order screen of the swap-in columns for all removals at once (one BLAS mat-mat)
                GH = D.XT @ R if D.XT2 is D.XT else None
                for q, j in enumerate(S):
                    if GH is not None:
                        G, H = GH[nonS, 2 * q], GH[nonS, 2 * q + 1]
                    else:
                        G, H = D.XT[nonS] @ R[:, 2 * q], D.XT2[nonS] @ R[:, 2 * q + 1]
                    top(sts[q].eval(D, nonS, nzv, a, b, D.maxnv, self.n_screen, GH=(G, H)), j, nonS, nzv)
                best = self.check(st, w, cands, a, b)
            if best is None:
                break
            st, rj, aj, v = best
            if rj >= 0:
                w[rj] = 0
            w[aj] = v
        return st.L / D.N, w


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, n_starts=5):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.n_starts = n_starts

    def fit(self, X, y):
        tm = [time.perf_counter()]
        t0 = tm[0]
        X = np.asarray(X, dtype=float)
        D = Data(X, y)
        tm.append(time.perf_counter())
        parents = beam_search(D, self.k, self.parent_size, self.child_size, t0 + 0.4 * self.time_limit)
        tm.append(time.perf_counter())
        seen = set()
        starts = []
        best_w, best_l = np.zeros(D.d), np.inf
        for nd in parents:
            w, l = calib_round(D, nd)
            key = w.tobytes()
            if key in seen or not np.isfinite(l):
                continue
            seen.add(key)
            starts.append((l, w))
            if l < best_l:
                best_l, best_w = l, w
        tm.append(time.perf_counter())
        # integer local search from the best few distinct rounded solutions
        ils = ILS(D, self.k)
        starts.sort(key=lambda t: t[0])
        for l0, w0 in starts[:self.n_starts]:
            if time.perf_counter() > t0 + 0.8 * self.time_limit:
                break
            l, w = ils.run(w0, deadline=t0 + 0.9 * self.time_limit)
            if l < best_l:
                best_l, best_w = l, w
        tm.append(time.perf_counter())
        self.timing_ = np.diff(tm)  # data, beam, rounding, ILS
        self.coef_ = np.clip(np.round(best_w), -COEF_BOUND, COEF_BOUND)
        self.intercept_, self.multiplier_ = 0.0, 1.0
        self.train_loss_ = best_l
        return self




# ------------------------------------------------------------------ certification
# All bounds below are proven in notes.md (B1-B6). Totals (sums over rows) are used throughout; the mean
# criterion is total / N.
TOL = 1e-9          # global slack (mean loss) for floating-point error in the proven bounds


@nb.njit(cache=False, inline="always")
def _ent(a):
    if a <= 0.0 or a >= 1.0:
        return 0.0
    return -(a * np.log(a) + (1.0 - a) * np.log1p(-a))


@nb.njit(cache=False)
def _obj(Z, P, N, v):
    m, p = Z.shape
    f = 0.0
    for i in range(m):
        z = v[p]
        for j in range(p):
            x = Z[i, j]
            if x != 0.0:
                z += x * v[j]
        if P[i] > 0:
            f += P[i] * _lrow(z)
        if N[i] > 0:
            f += N[i] * _lrow(-z)
    return f


@nb.njit(cache=False)
def _chol(A):
    """Cholesky factor of the symmetric positive definite A (ridge already added); the ridge is raised if a
    pivot is not positive (the solution only feeds the Newton direction and the dual construction, whose
    feasibility is re-checked from the residual, so any nonsingular matrix is acceptable)."""
    q = A.shape[0]
    L = np.zeros((q, q))
    extra = 0.0
    for attempt in range(60):
        ok = True
        for i in range(q):
            for j in range(i + 1):
                sm = A[i, j] + (extra if i == j else 0.0)
                for l in range(j):
                    sm -= L[i, l] * L[j, l]
                if i == j:
                    if sm <= 0.0:
                        ok = False
                        break
                    L[i, i] = np.sqrt(sm)
                else:
                    L[i, j] = sm / L[j, j]
            if not ok:
                break
        if ok:
            return L
        tr = 0.0
        for i in range(q):
            tr += abs(A[i, i])
        extra = max(extra * 10.0, 1e-12 * tr / q + 1e-300)
    return L


@nb.njit(cache=False)
def _chol_sol(L, b):
    q = L.shape[0]
    x = np.empty(q)
    for i in range(q):
        sm = b[i]
        for l in range(i):
            sm -= L[i, l] * x[l]
        x[i] = sm / L[i, i]
    for i in range(q - 1, -1, -1):
        sm = x[i]
        for l in range(i + 1, q):
            sm -= L[l, i] * x[l]
        x[i] = sm / L[i, i]
    return x


@nb.njit(cache=False)
def _solve_ref(L, Hs, g):
    """Solve Hs d = g with the Cholesky factor L of the ridge-regularised Hs and two refinement steps."""
    d = _chol_sol(L, g)
    for _ in range(2):
        d += _chol_sol(L, g - Hs @ d)
    return d


@nb.njit(cache=False)
def _dual_value(Z, P, N, z, d, L, Hs):
    """D(alpha') of B3: alpha' = Newton-linearised duals at logits z along the Newton direction d, minus the
    margin E for the floating-point residual. -inf when alpha' leaves [0, 1]."""
    m, p = Z.shape
    q = p + 1
    r = np.zeros(q)
    ap = np.empty(m)
    am = np.empty(m)
    wv = np.empty(m)
    D = 0.0
    tot = 0.0
    for i in range(m):
        s = _sig(z[i])
        w = s * (1.0 - s)
        xd = d[p]
        for j in range(p):
            x = Z[i, j]
            if x != 0.0:
                xd += x * d[j]
        a1 = (1.0 - s) + w * xd
        a0 = s - w * xd
        if P[i] > 0 and (a1 < 0.0 or a1 > 1.0):
            return -np.inf
        if N[i] > 0 and (a0 < 0.0 or a0 > 1.0):
            return -np.inf
        a1 = min(max(a1, 0.0), 1.0)
        a0 = min(max(a0, 0.0), 1.0)
        ap[i], am[i], wv[i] = a1, a0, w
        D += P[i] * _ent(a1) + N[i] * _ent(a0)
        tot += P[i] + N[i]
        c = P[i] * a1 - N[i] * a0
        for j in range(p):
            x = Z[i, j]
            if x != 0.0:
                r[j] += c * x
        r[p] += c
    d2 = _solve_ref(L, Hs, r)
    E = 0.0
    for i in range(m):
        w = wv[i]
        if w == 0.0:
            continue          # the exact correction leaves this row's alpha unchanged (it is proportional to w)
        xd = d2[p]
        for j in range(p):
            x = Z[i, j]
            if x != 0.0:
                xd += x * d2[j]
        dl = w * abs(xd)
        if dl == 0.0:
            continue
        a1, a0 = ap[i], am[i]
        if P[i] > 0:
            if a1 <= 0.0 or a1 >= 1.0:
                return -np.inf
            E += P[i] * abs(np.log((1.0 - a1) / a1)) * dl
        if N[i] > 0:
            if a0 <= 0.0 or a0 >= 1.0:
                return -np.inf
            E += N[i] * abs(np.log((1.0 - a0) / a0)) * dl
    return D - 2.0 * E - 1e-12 * tot


@nb.njit(cache=False)
def lr_bound(Z, P, N, ub, maxit, stop_primal, v0, work):
    """Logistic regression on the columns of Z (+ intercept) by damped Newton. Returns (lb, f, v): lb the best
    valid dual bound (B1/B3) seen, f the primal value at v. Stops early once lb >= ub (prunable) or, when
    stop_primal, once f < ub (a continuous point already beats ub, so the bound cannot reach it)."""
    m, p = Z.shape
    q = p + 1
    v = v0.copy()
    f = _obj(Z, P, N, v)
    work[0] += m * (p + 1)
    z = np.empty(m)
    best = -np.inf
    Hs = np.empty((q, q))
    g = np.empty(q)
    nzi = np.empty(q, np.int64)
    for it in range(maxit):
        if stop_primal > 0 and f < ub:
            return best, f, v
        g[:] = 0.0
        Hs[:, :] = 0.0
        for i in range(m):
            zi = v[p]
            nn = 0
            for j in range(p):
                x = Z[i, j]
                if x != 0.0:
                    zi += x * v[j]
                    nzi[nn] = j
                    nn += 1
            z[i] = zi
            s = _sig(zi)
            gi = -P[i] * (1.0 - s) + N[i] * s
            wi = (P[i] + N[i]) * s * (1.0 - s)
            for aa in range(nn):
                ja = nzi[aa]
                xa = Z[i, ja]
                g[ja] += gi * xa
                Hs[p, ja] += wi * xa
                for bb in range(aa + 1):
                    jb = nzi[bb]
                    Hs[ja, jb] += wi * xa * Z[i, jb]
            g[p] += gi
            Hs[p, p] += wi
            work[0] += p + 3 * (nn + 1) * (nn + 2)
        for a_ in range(q):
            for b_ in range(a_):
                Hs[b_, a_] = Hs[a_, b_]
        tr = 0.0
        for a_ in range(q):
            tr += Hs[a_, a_]
        A = Hs.copy()
        eps = 1e-11 * tr / q + 1e-300
        for a_ in range(q):
            A[a_, a_] += eps
        L = _chol(A)
        d = _solve_ref(L, Hs, g)
        lb = _dual_value(Z, P, N, z, d, L, Hs)
        if lb > best:
            best = lb
        if best >= ub:
            return best, f, v
        gd = 0.0
        for a_ in range(q):
            gd += g[a_] * d[a_]
        if not (gd > 1e-13 * (1.0 + f)):
            break
        if stop_primal == 2 and it >= 2 and f - 0.5 * gd < ub:
            return best, f, v        # search heuristic: the Newton model predicts the optimum below ub
        t = 1.0
        ok = False
        for _ in range(40):
            vn = v - t * d
            fn = _obj(Z, P, N, vn)
            work[0] += m * (p + 1)
            if fn <= f - 1e-4 * t * gd:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        v = vn
        f = fn
    return best, f, v


@nb.njit(cache=False)
def _compact(ids, P, N, ncell_max):
    """Cells from integer ids in [0, ncell_max): (rep row, P, N) per non-empty cell and the compact id per row."""
    cp = np.zeros(ncell_max)
    cn = np.zeros(ncell_max)
    rep = -np.ones(ncell_max, np.int64)
    for i in range(ids.shape[0]):
        c = ids[i]
        cp[c] += P[i]
        cn[c] += N[i]
        if rep[c] < 0:
            rep[c] = i
    mapc = -np.ones(ncell_max, np.int64)
    nc = 0
    for c in range(ncell_max):
        if rep[c] >= 0:
            mapc[c] = nc
            nc += 1
    R = np.empty(nc, np.int64)
    Pc = np.empty(nc)
    Nc = np.empty(nc)
    for c in range(ncell_max):
        if rep[c] >= 0:
            R[mapc[c]] = rep[c]
            Pc[mapc[c]] = cp[c]
            Nc[mapc[c]] = cn[c]
    cid = np.empty(ids.shape[0], np.int64)
    for i in range(ids.shape[0]):
        cid[i] = mapc[ids[i]]
    return R, Pc, Nc, cid


@nb.njit(cache=False)
def _sat_loss(Pc, Nc):
    """B4: sum over cells of the Bayes-optimal log loss of the cell."""
    h = 0.0
    for c in range(Pc.shape[0]):
        t = Pc[c] + Nc[c]
        if Pc[c] > 0 and Nc[c] > 0:
            h += t * _ent(Pc[c] / t)
    return h


@nb.njit(cache=False)
def cells_of(codes, nv, S, P, N, limit):
    """Cells of the columns S (by value codes). Returns (rep, Pc, Nc, cid, ok); ok False when the number of
    value combinations exceeds limit (then every unique row is its own cell: a finer partition, still valid)."""
    u = codes.shape[0]
    radix = 1
    for j in S:
        radix *= nv[j]
        if radix > limit:
            ids = np.arange(u)
            R, Pc, Nc, cid = _compact(ids, P, N, u)
            return R, Pc, Nc, cid, False
    ids = np.zeros(u, np.int64)
    for i in range(u):
        c = 0
        for j in S:
            c = c * nv[j] + codes[i, j]
        ids[i] = c
    R, Pc, Nc, cid = _compact(ids, P, N, radix)
    return R, Pc, Nc, cid, True


@nb.njit(cache=False)
def leaf_scan(codes, nv, Xs, P, N, T, cid_T, nc_T, cands, ub, limit, maxit, ptr1, idx1, val0, val1, work, xlx):
    """Leaves T u {j} for j in cands (B4 then B1/B3 on the cells). Returns the j that survive both bounds.
    Two-valued columns take a fast path: the cells of T u {j} are the cells of T split by the rows where
    column j has its second value (the CSC list idx1), so a leaf costs O(nnz_j + cells) instead of O(rows)."""
    u = codes.shape[0]
    t = T.shape[0]
    out = np.empty(cands.shape[0], np.int64)
    no = 0
    S = np.empty(t + 1, np.int64)
    S[:t] = T
    # cells of T: counts, a representative row, their LR solution (warm start of every leaf)
    PT = np.zeros(nc_T)
    NT = np.zeros(nc_T)
    repT = -np.ones(nc_T, np.int64)
    for r in range(u):
        c = cid_T[r]
        PT[c] += P[r]
        NT[c] += N[r]
        if repT[c] < 0:
            repT[c] = r
    ZT = np.empty((nc_T, t))
    for c in range(nc_T):
        for a_ in range(t):
            ZT[c, a_] = Xs[repT[c], T[a_]]
    vT0 = np.zeros(t + 1)
    vT0[t] = np.log(max(P.sum(), 1e-300) / max(N.sum(), 1e-300))
    work[0] += u
    _, _, vT = lr_bound(ZT, PT, NT, np.inf, 6, False, vT0, work)
    v0 = np.zeros(t + 2)
    v0[:t] = vT[:t]
    v0[t + 1] = vT[t]
    P1 = np.zeros(nc_T)
    N1 = np.zeros(nc_T)
    Zb = np.empty((2 * nc_T, t + 1))
    Pb = np.empty(2 * nc_T)
    Nb = np.empty(2 * nc_T)
    for jj in range(cands.shape[0]):
        j = cands[jj]
        S[t] = j
        if nv[j] == 2:
            P1[:] = 0.0
            N1[:] = 0.0
            for q in range(ptr1[j], ptr1[j + 1]):
                r = idx1[q]
                c = cid_T[r]
                P1[c] += P[r]
                N1[c] += N[r]
            m = 0
            h = 0.0
            for c in range(nc_T):
                for side in range(2):
                    if side == 0:
                        pp = PT[c] - P1[c]
                        nn = NT[c] - N1[c]
                    else:
                        pp = P1[c]
                        nn = N1[c]
                    if pp + nn <= 0.5:
                        continue
                    if pp > 0.5 and nn > 0.5:
                        h += xlx[int(pp + nn + 0.5)] - xlx[int(pp + 0.5)] - xlx[int(nn + 0.5)]
                    for a_ in range(t):
                        Zb[m, a_] = ZT[c, a_]
                    Zb[m, t] = val1[j] if side == 1 else val0[j]
                    Pb[m] = pp if pp > 0.5 else 0.0
                    Nb[m] = nn if nn > 0.5 else 0.0
                    m += 1
            work[0] += ptr1[j + 1] - ptr1[j] + 2 * nc_T * (t + 2)
            if h >= ub:
                continue
            lb, f, v = lr_bound(Zb[:m], Pb[:m], Nb[:m], ub, maxit, 1, v0, work)
            if lb >= ub:
                continue
            out[no] = j
            no += 1
            continue
        radix = nc_T * nv[j]
        work[0] += 3 * u
        if radix <= limit:
            ids = np.empty(u, np.int64)
            for i in range(u):
                ids[i] = cid_T[i] * nv[j] + codes[i, j]
            R, Pc, Nc, cid = _compact(ids, P, N, radix)
            if _sat_loss(Pc, Nc) >= ub:
                continue
        else:
            R = np.arange(u)
            Pc, Nc = P, N
        Z = np.empty((R.shape[0], t + 1))
        for c in range(R.shape[0]):
            for a_ in range(t + 1):
                Z[c, a_] = Xs[R[c], S[a_]]
        lb, f, v = lr_bound(Z, Pc, Nc, ub, maxit, 1, v0, work)
        if lb >= ub:
            continue
        out[no] = j
        no += 1
    return out[:no]


@nb.njit(cache=False)
def pair_scan(codes, nv, Xs, P, N, T, cid_T, nc_T, C, i0, i1, ub, limit, maxit, ptr1, idx1, val0, val1, work, xlx):
    """Leaves T u {C[i], C[j]} for i0 <= i < i1, j > i (same bounds as leaf_scan). Returns surviving pairs."""
    u = codes.shape[0]
    t = T.shape[0]
    outa = np.empty(0, np.int64)
    outb = np.empty(0, np.int64)
    T2 = np.empty(t + 1, np.int64)
    T2[:t] = T
    for i in range(i0, i1):
        c = C[i]
        T2[t] = c
        radix = nc_T * nv[c]
        if radix <= limit:
            ids = np.empty(u, np.int64)
            for r in range(u):
                ids[r] = cid_T[r] * nv[c] + codes[r, c]
            R, Pc, Nc, cid = _compact(ids, P, N, radix)
            nc = R.shape[0]
        else:
            cid = np.arange(u)
            nc = u
        work[0] += u
        surv = leaf_scan(codes, nv, Xs, P, N, T2, cid, nc, C[i + 1:], ub, limit, maxit, ptr1, idx1, val0, val1, work, xlx)
        if surv.shape[0] > 0:
            outa = np.concatenate((outa, np.full(surv.shape[0], c, np.int64)))
            outb = np.concatenate((outb, surv))
    return outa, outb


@nb.njit(cache=False)
def pair_gram(P, N, cid_T, nc_T, C, rowptr, rowidx, D):
    """Per cell of T: weighted counts of rows with the second value of candidate a (G1) and of both a and b
    (G2, a < b), from the row lists of second values: O(sum over rows of nnz^2) for all pairs at once."""
    u = P.shape[0]
    nC = C.shape[0]
    pos = -np.ones(D, np.int64)
    for a in range(nC):
        pos[C[a]] = a
    TP = np.zeros(nc_T)
    TN = np.zeros(nc_T)
    G1P = np.zeros((nc_T, nC))
    G1N = np.zeros((nc_T, nC))
    G2P = np.zeros((nc_T, nC, nC))
    G2N = np.zeros((nc_T, nC, nC))
    buf = np.empty(nC, np.int64)
    order = np.argsort(cid_T, kind="mergesort")      # rows grouped by cell: the cell's count block stays in cache
    for rr in range(u):
        r = order[rr]
        c = cid_T[r]
        pr, nr = P[r], N[r]
        TP[c] += pr
        TN[c] += nr
        kk = 0
        for q in range(rowptr[r], rowptr[r + 1]):
            a = pos[rowidx[q]]
            if a >= 0:
                buf[kk] = a
                kk += 1
        for x in range(kk):
            a = buf[x]
            G1P[c, a] += pr
            G1N[c, a] += nr
            for y in range(x + 1, kk):
                b = buf[y]
                if a < b:
                    G2P[c, a, b] += pr
                    G2N[c, a, b] += nr
                else:
                    G2P[c, b, a] += pr
                    G2N[c, b, a] += nr
    return TP, TN, G1P, G1N, G2P, G2N


@nb.njit(cache=False)
def pair_scan_gram(Xs, P, N, T, cid_T, nc_T, C, i0, i1, ub, maxit, val0, val1, TP, TN, G1P, G1N, G2P, G2N, work,
                   xlx):
    """Leaves T u {C[i], C[j]} (i0 <= i < i1 < j, all two-valued) from the pair counts: B4, then B1/B3 on the
    (at most 4 nc_T) cells. Returns the surviving pairs."""
    u = P.shape[0]
    t = T.shape[0]
    nC = C.shape[0]
    repT = -np.ones(nc_T, np.int64)
    for r in range(u):
        if repT[cid_T[r]] < 0:
            repT[cid_T[r]] = r
    ZT = np.empty((nc_T, t))
    for c in range(nc_T):
        for a_ in range(t):
            ZT[c, a_] = Xs[repT[c], T[a_]]
    vT0 = np.zeros(t + 1)
    vT0[t] = np.log(max(P.sum(), 1e-300) / max(N.sum(), 1e-300))
    _, _, vT = lr_bound(ZT, TP, TN, np.inf, 6, False, vT0, work)
    v0 = np.zeros(t + 3)
    v0[:t] = vT[:t]
    v0[t + 2] = vT[t]
    Zb = np.empty((4 * nc_T, t + 2))
    Pb = np.empty(4 * nc_T)
    Nb = np.empty(4 * nc_T)
    outa = np.empty(0, np.int64)
    outb = np.empty(0, np.int64)
    for i in range(i0, i1):
        ci = C[i]
        for j in range(i + 1, nC):
            cj = C[j]
            work[0] += 8 * nc_T
            m = 0
            h = 0.0
            for c in range(nc_T):
                p11 = G2P[c, i, j]
                n11 = G2N[c, i, j]
                p10 = G1P[c, i] - p11
                n10 = G1N[c, i] - n11
                p01 = G1P[c, j] - p11
                n01 = G1N[c, j] - n11
                p00 = TP[c] - p10 - p01 - p11
                n00 = TN[c] - n10 - n01 - n11
                for sd in range(4):
                    if sd == 0:
                        pp, nn, xa, xb = p00, n00, val0[ci], val0[cj]
                    elif sd == 1:
                        pp, nn, xa, xb = p10, n10, val1[ci], val0[cj]
                    elif sd == 2:
                        pp, nn, xa, xb = p01, n01, val0[ci], val1[cj]
                    else:
                        pp, nn, xa, xb = p11, n11, val1[ci], val1[cj]
                    if pp + nn <= 0.5:
                        continue          # counts are integers: an empty cell
                    if pp > 0.5 and nn > 0.5:
                        # (p + n) * entropy(p / (p + n)) from the table x log x of integer counts
                        h += xlx[int(pp + nn + 0.5)] - xlx[int(pp + 0.5)] - xlx[int(nn + 0.5)]
                    for a_ in range(t):
                        Zb[m, a_] = ZT[c, a_]
                    Zb[m, t] = xa
                    Zb[m, t + 1] = xb
                    Pb[m] = pp if pp > 0.5 else 0.0
                    Nb[m] = nn if nn > 0.5 else 0.0
                    m += 1
            if h >= ub:
                continue
            lb, f, v = lr_bound(Zb[:m], Pb[:m], Nb[:m], ub, maxit, 1, v0, work)
            if lb >= ub:
                continue
            outa = np.concatenate((outa, np.full(1, ci, np.int64)))
            outb = np.concatenate((outb, np.full(1, cj, np.int64)))
    return outa, outb


class _Cert:
    """State of one certification run."""

    def __init__(self, X, y01, k, w_inc, deadline):
        self.deadline = deadline
        self.work = np.zeros(1)     # deterministic operation counts (drive the cost model; no timing)
        y = np.asarray(y01, float)
        self.Nrows = float(len(y))
        self.d_all = X.shape[1]
        cols = np.flatnonzero(np.ptp(X, axis=0) > 0)
        self.cols = cols
        Xc = X[:, cols]
        ux, inv = np.unique(Xc, axis=0, return_inverse=True)
        inv = inv.ravel()
        self.P = np.bincount(inv, weights=y, minlength=len(ux)).astype(float)
        self.N = np.bincount(inv, weights=1 - y, minlength=len(ux)).astype(float)
        self.Xr = np.ascontiguousarray(ux)                        # raw values (integer scores)
        sc = np.abs(ux).max(axis=0)
        self.Xs = np.ascontiguousarray(ux / np.where(sc > 0, sc, 1.0))  # LR columns (scaling leaves LR unchanged)
        D = len(cols)
        self.D = D
        codes = np.empty(ux.shape, np.int64)
        nv = np.empty(D, np.int64)
        for j in range(D):
            _, c = np.unique(ux[:, j], return_inverse=True)
            codes[:, j] = c.ravel()
            nv[j] = c.max() + 1
        self.codes, self.nv = codes, nv
        # rows with the second value of each two-valued column, and the two (scaled) values
        ptr1 = np.zeros(D + 1, np.int64)
        lists = []
        val0 = np.zeros(D)
        val1 = np.zeros(D)
        side = np.zeros(D, np.int64)       # the listed value of a two-valued column: its rarer one (fewer rows)
        for j in range(D):
            if nv[j] == 2:
                side[j] = 1 if (codes[:, j] == 1).sum() <= (codes[:, j] == 0).sum() else 0
                r1 = np.flatnonzero(codes[:, j] == side[j])
                val0[j] = self.Xs[np.flatnonzero(codes[:, j] != side[j])[0], j]
                val1[j] = self.Xs[r1[0], j]
            else:
                r1 = np.zeros(0, np.int64)
            lists.append(r1)
            ptr1[j + 1] = ptr1[j] + len(r1)
        idx1 = np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, np.int64)
        self.csc = (ptr1, idx1, val0, val1)
        xs_ = np.arange(int(self.Nrows) + 2, dtype=float)
        self.xlx = xs_ * np.log(np.maximum(xs_, 1.0))      # x log x of integer counts (0 log 0 = 0)
        # row lists of the columns at their second value (two-valued columns only)
        two = (nv == 2)
        M1 = (codes == side[None, :]) & two[None, :]
        self.rowptr = np.concatenate([[0], np.cumsum(M1.sum(axis=1))]).astype(np.int64)
        self.rowidx = np.nonzero(M1)[1].astype(np.int64)
        self.rownnz2 = float((M1.sum(axis=1) ** 2).sum())
        self.k = k
        self.tol = TOL * self.Nrows
        self.limit = 4 * len(ux) + 64
        p0 = self.P.sum() / self.Nrows
        self.b0 = np.log(max(p0, 1e-300) / max(1 - p0, 1e-300))
        # incumbent: its exact loss (total), computed like every other integer candidate
        self.best_w = np.asarray(w_inc, float).copy()
        s = self.Xr @ self.best_w[cols]
        self.UB = self.exact_score(s)
        self.timeout = False
        self.n_nodes = 0
        self.n_leaves = 0
        self.n_int = 0
        self.t_fam = self.t_leaf = self.t_int = 0.0
        self.n_fam = 0
        self.n_famp = 0
        self.w_famp = 0.0
        self.wl = np.zeros(1)
        self.w_fam = 0.0

    def exact_score(self, s):
        """Total calibrated loss of the integer score s (min over a, b) and nothing else."""
        Z = np.ascontiguousarray(s.reshape(-1, 1))
        if np.ptp(s) == 0:
            Z = np.zeros((len(s), 0))
        v0 = np.zeros(Z.shape[1] + 1)
        v0[-1] = self.b0
        lb, f, v = lr_bound(Z, self.P, self.N, np.inf, 200, 0, v0, self.work)
        return f

    def check_time(self):
        if time.perf_counter() > self.deadline:
            self.timeout = True
        return self.timeout

    # ---------------------------------------------------------------- support tree (B5)
    def order_features(self):
        D = self.D
        v0 = np.zeros(D + 1)
        v0[-1] = self.b0
        lb, f, v = lr_bound(self.Xs, self.P, self.N, np.inf, 50, 0, v0, self.work)
        # Wald statistic v_j^2 / (H^-1)_jj ~ the loss increase when column j is dropped (a heuristic order only)
        Z = self.Xs
        zz = Z @ v[:D] + v[D]
        s = 1 / (1 + np.exp(-zz))
        w = (self.P + self.N) * s * (1 - s)
        Zt = np.column_stack([Z, np.ones(len(Z))])
        H = (Zt * w[:, None]).T @ Zt
        H += (1e-9 * np.trace(H) / (D + 1) + 1e-12) * np.eye(D + 1)
        try:
            Hi = np.linalg.inv(H)
            imp = v[:D] ** 2 / np.maximum(np.diag(Hi)[:D], 1e-300)
        except np.linalg.LinAlgError:
            imp = np.abs(v[:D])
        imp = np.where(np.isfinite(imp), imp, 0.0)
        return np.argsort(-imp, kind="stable"), v

    def family_bound(self, U, v0):
        Z = np.ascontiguousarray(self.Xs[:, U])
        t_ = time.perf_counter()
        w_ = self.work[0]
        lb, f, v = lr_bound(Z, self.P, self.N, self.UB - self.tol, 30, 2, v0, self.work)
        self.w_fam += self.work[0] - w_
        self.t_fam += time.perf_counter() - t_
        self.n_fam += 1
        if lb >= self.UB - self.tol:
            self.w_famp += self.work[0] - w_
            self.n_famp += 1
        self.vf[U] = v[:-1]
        return lb, v

    def wald_imp(self, U, rest):
        """Search order only: Wald statistic v_j^2 / (H^-1)_jj of the columns rest inside the LR on U."""
        Z = self.Xs[:, U]
        v = self.vf[U]
        z = Z @ v + self.vf[self.D]
        q = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        w = (self.P + self.N) * q * (1 - q)
        Zt = np.column_stack([Z, np.ones(len(Z))])
        H = (Zt * w[:, None]).T @ Zt
        H += (1e-9 * np.trace(H) / len(H) + 1e-12) * np.eye(len(H))
        try:
            hi = np.diag(np.linalg.inv(H))[:len(U)]
        except np.linalg.LinAlgError:
            return self.vf[rest] ** 2 * self.cvar[rest]
        pos = {c: i for i, c in enumerate(U)}
        idx = np.array([pos[c] for c in rest])
        return v[idx] ** 2 / np.maximum(hi[idx], 1e-300)

    def run(self):
        D, k = self.D, self.k
        if D == 0:
            return True
        # B4 over all columns (the unique rows): if the incumbent already reaches it, it is optimal
        if _sat_loss(self.P, self.N) >= self.UB - self.tol:
            return True
        self.order, vfull = self.order_features()
        m = min(k, D)
        self.kk = m
        vf = np.zeros(D + 1)          # warm start: last LR solution per original column
        vf[:D] = vfull[:D]
        vf[D] = vfull[D]
        self.vf = vf
        self.survivors = []
        P_, N_ = self.P, self.N
        wt = (P_ + N_) / self.Nrows
        mu = wt @ self.Xs
        self.cvar = wt @ (self.Xs - mu) ** 2
        self.dyn = True
        self.wald = True
        self.wald_min = 20000
        self.fam0 = 5.0 * len(self.P) * (D + (self.Xs != 0).sum(axis=1).mean() ** 2)
        self.dfs([], list(self.order))
        if self.timeout:
            return False
        for S in self.survivors:
            if self.check_time():
                return False
            t_ = time.perf_counter()
            self.int_bb(S)
            self.t_int += time.perf_counter() - t_
            if self.timeout:
                return False
        return True

    def _v0(self, U):
        v0 = np.empty(len(U) + 1)
        v0[:-1] = self.vf[U]
        v0[-1] = self.vf[self.D]
        return v0

    def dfs(self, T, C):
        """Node (T, C): all k-subsets T u A with A inside C (B5). The children of C = (c_0, c_1, ...) are
        (T + c_i, (c_{i+1}, ...)); any order of C is valid, so C is ordered per node (most important first)."""
        if self.check_time():
            return
        mleft = self.kk - len(T)
        if mleft == 1:
            Tc = np.array(T, np.int64)
            R, Pc, Nc, cid, ok = cells_of(self.codes, self.nv, Tc, self.P, self.N, self.limit)
            nc = len(R) if ok else len(self.P)
            if not ok:
                cid = np.arange(len(self.P))
            cands = np.asarray(C, np.int64)
            # leaves are bounded in chunks so the deadline is checked regularly
            for c0 in range(0, len(cands), 64):
                if self.check_time():
                    return
                t_ = time.perf_counter()
                surv = leaf_scan(self.codes, self.nv, self.Xs, self.P, self.N, Tc, cid, nc,
                                 cands[c0:c0 + 64], self.UB - self.tol, self.limit, 30, *self.csc, self.wl, self.xlx)
                self.t_leaf += time.perf_counter() - t_
                self.n_leaves += min(64, len(cands) - c0)
                for j in surv:
                    self.survivors.append(list(Tc) + [int(j)])
            return
        C = list(C)
        for i in range(len(C) - mleft + 1):
            if self.check_time():
                return
            # cost model (speed only): when child i's subtree has so few leaves that bounding them one by one
            # is cheaper than one family bound, enumerate the leaves of this child and all later siblings
            if not T:
                self.root_i = i
            nsub = comb(len(C) - i - 1, mleft - 1)
            if nsub * (200.0 + 5.0 * (self.wl[0] + 2000.0) / (self.n_leaves + 10)) <= (self.w_fam + self.fam0) / (self.n_fam + 1):
                self.enum_block(T, C[i:], mleft)
                return
            U = np.array(T + C[i:], np.int64)
            self.n_nodes += 1
            # the first child's column set T u C is this node's own set, whose bound was already computed by the
            # parent's loop and did not prune (only the root has not been bounded)
            if i > 0 or not T:
                lb, v = self.family_bound(U, self._v0(U))
                if lb >= self.UB - self.tol:
                    break               # B5: every later sibling's column set is a subset of U
            if self.dyn:
                # greedy backward order (search order only): the next child takes the column that is most important
                # in the LR on U = T u C[i:] just solved, so the later siblings' sets lose it first
                rem = np.array(C[i:], np.int64)
                if self.wald and comb(len(rem) - 1, mleft - 1) > self.wald_min:
                    imp = self.wald_imp(U, rem)
                else:
                    imp = self.vf[rem] ** 2 * self.cvar[rem]
                o = np.argsort(-imp, kind="stable")
                C[i:] = [int(x) for x in rem[o]]
            self.dfs(T + [C[i]], list(C[i + 1:]))
            if self.timeout:
                return

    def _cells_T(self, T):
        Tc = np.array(T, np.int64)
        R, Pc, Nc, cid, ok = cells_of(self.codes, self.nv, Tc, self.P, self.N, self.limit)
        if not ok:
            return Tc, np.arange(len(self.P)), len(self.P)
        return Tc, cid, len(R)

    def enum_block(self, T, C, m):
        """Every leaf T u A, A inside C, |A| = m, bounded one by one (no family bounds)."""
        if self.check_time():
            return
        C = list(C)
        if m == 1:
            self.dfs(T, C)
            return
        if m == 2:
            Tc, cid, nc = self._cells_T(T)
            Ca = np.array(C, np.int64)
            if len(C) > 1 and np.all(self.nv[Ca] == 2) and nc * len(C) ** 2 <= 4_000_000:
                t_ = time.perf_counter()
                gram = pair_gram(self.P, self.N, cid, nc, Ca, self.rowptr, self.rowidx, self.D)
                self.wl[0] += len(self.P) + self.rownnz2
                step = max(1, 20000 // max(len(C), 1))
                for i0 in range(0, len(C) - 1, step):
                    if self.check_time():
                        return
                    i1 = min(i0 + step, len(C) - 1)
                    a, b = pair_scan_gram(self.Xs, self.P, self.N, Tc, cid, nc, Ca, i0, i1, self.UB - self.tol, 30,
                                          self.csc[2], self.csc[3], *gram, self.wl, self.xlx)
                    self.n_leaves += sum(len(C) - 1 - i for i in range(i0, i1))
                    for x, y_ in zip(a, b):
                        self.survivors.append(list(Tc) + [int(x), int(y_)])
                self.t_leaf += time.perf_counter() - t_
                return
            step = max(1, 4096 // max(len(C), 1))
            for i0 in range(0, len(C) - 1, step):
                if self.check_time():
                    return
                i1 = min(i0 + step, len(C) - 1)
                t_ = time.perf_counter()
                a, b = pair_scan(self.codes, self.nv, self.Xs, self.P, self.N, Tc, cid, nc, Ca, i0, i1,
                                 self.UB - self.tol, self.limit, 30, *self.csc, self.wl, self.xlx)
                self.t_leaf += time.perf_counter() - t_
                self.n_leaves += sum(len(C) - 1 - i for i in range(i0, i1))
                for x, y_ in zip(a, b):
                    self.survivors.append(list(Tc) + [int(x), int(y_)])
            return
        for i in range(len(C) - m + 1):
            self.enum_block(T + [C[i]], C[i + 1:], m - 1)
            if self.timeout:
                return

    # ---------------------------------------------------------------- integer branch and bound (B6)
    def int_bb(self, S):
        S = np.array(S, np.int64)
        R, Pc, Nc, cid, ok = cells_of(self.codes, self.nv, S, self.P, self.N, self.limit)
        Xr = np.ascontiguousarray(self.Xr[R][:, S])
        Xs = np.ascontiguousarray(self.Xs[R][:, S])
        s_ = len(S)
        v0 = np.zeros(s_ + 1)
        v0[-1] = self.b0
        lb, f, v = lr_bound(Xs, Pc, Nc, self.UB - self.tol, 60, 0, v0, self.work)
        if lb >= self.UB - self.tol:
            return
        # fix coordinates in decreasing order of their continuous contribution
        scale = np.abs(Xs).max(axis=0)
        imp = np.abs(v[:s_]) * scale
        ordc = np.argsort(-imp, kind="stable")
        self.ib = dict(S=S, Pc=Pc, Nc=Nc, Xr=Xr, Xs=Xs, ordc=ordc)
        w = np.zeros(s_)
        self._ibb(w, 0, False, v)

    def _ibb(self, w, depth, nonzero, vrel):
        if self.check_time():
            return
        ib = self.ib
        ordc, Xr, Xs, Pc, Nc = ib["ordc"], ib["Xr"], ib["Xs"], ib["Pc"], ib["Nc"]
        s_ = len(ordc)
        c = ordc[depth]
        # candidate values: symmetry (B6) makes the first nonzero coordinate positive
        vals = list(range(0, COEF_BOUND + 1)) if not nonzero else list(range(-COEF_BOUND, COEF_BOUND + 1))
        # order values by closeness to the continuous ratio (search order only)
        if nonzero and vrel is not None and abs(vrel[0]) > 0:
            pred = vrel[1] / vrel[0]
            vals.sort(key=lambda x: abs(x - pred))
        for val in vals:
            if self.timeout:
                return
            w[c] = val
            fixed = ordc[:depth + 1]
            free = ordc[depth + 1:]
            nz = nonzero or val != 0
            last = depth + 1 == s_
            if last:
                if not nz:
                    continue
                wi = np.round(w[fixed]).astype(np.int64)
                g = 0
                for x in wi:
                    g = np.gcd(g, abs(int(x)))
                if g > 1:
                    continue           # B6: the primitive vector w / g is enumerated elsewhere
                sP = Xr[:, fixed] @ w[fixed]
                if np.ptp(sP) == 0:
                    continue           # constant score: the intercept-only loss, >= UB since UB <= base loss
                Z = np.ascontiguousarray(sP.reshape(-1, 1))
                v0 = np.array([0.0, self.b0])
                lb, f, v = lr_bound(Z, Pc, Nc, self.UB - self.tol, 200, 0, v0, self.work)
                self.n_int += 1
                if f < self.UB:
                    self.UB = f
                    bw = np.zeros(self.d_all)
                    bw[self.cols[ib["S"]]] = w
                    self.best_w = bw
                continue
            # relaxation: columns (s_P, X_free) (B6)
            if nz:
                sP = Xr[:, fixed] @ w[fixed]
                Z = np.ascontiguousarray(np.column_stack([sP, Xs[:, free]]))
            else:
                Z = np.ascontiguousarray(Xs[:, free])
            v0 = np.zeros(Z.shape[1] + 1)
            v0[-1] = self.b0
            lb, f, v = lr_bound(Z, Pc, Nc, self.UB - self.tol, 60, 0, v0, self.work)
            self.n_int += 1
            if lb >= self.UB - self.tol:
                continue
            nxt = None
            if nz:
                # continuous ratio of the next free coordinate to the fixed combination (raw scale)
                sc_next = np.abs(Xr[:, free[0]]).max() / max(np.abs(Xs[:, free[0]]).max(), 1e-300)
                nxt = (v[0], v[1] / max(sc_next, 1e-300))
            self._ibb(w, depth + 1, nz, nxt)
        w[c] = 0


def certify(X, y01, k, w_inc, ub, deadline):
    """Try to prove that no feasible point vector has a calibrated loss below the incumbent's (see notes.md).
    Returns (lower_bound, w_best, loss_best) in mean units; lower_bound = loss_best - TOL when every support
    was pruned by a valid bound or solved, else the saturated loss of all columns (valid, weak)."""
    C = _Cert(np.asarray(X, float), y01, k, w_inc, deadline)
    done = C.run()
    loss = C.UB / C.Nrows
    if done:
        lb = loss - TOL
    else:
        # B4 over all columns: valid for every model
        # (the unique rows are the cells of all columns)
        lb = _sat_loss(C.P, C.N) / C.Nrows
        lb = min(lb, loss - TOL)
    C.stats = (C.n_nodes, C.n_leaves, len(getattr(C, "survivors", [])), C.n_int, round(C.t_fam, 2), round(C.t_leaf, 2), round(C.t_int, 2), 'fam ns/work', round(1e9*C.t_fam/max(C.w_fam,1),2), 'leaf ns/work', round(1e9*C.t_leaf/max(C.wl[0],1),2), 'root_i', getattr(C, 'root_i', -1), C.D, 'fam n/pruned/work/prunedwork', C.n_fam, C.n_famp, int(C.w_fam), int(C.w_famp))
    return lb, C.best_w, loss, C


def make_model(k, time_limit):
    return CertifyingClassifier(k=k, time_limit=time_limit)


class CertifyingClassifier(SparseIntegerClassifier):
    """The heuristic for the incumbent, then ``certify`` with the time that is left."""

    def fit(self, X, y):
        t0 = time.perf_counter()
        super().fit(X, y)
        X = np.asarray(X, dtype=float)
        lb, w, l, C = certify(X, y, self.k, self.coef_.astype(float), float(self.train_loss_),
                              t0 + 0.92 * self.time_limit)
        self.coef_ = np.clip(np.round(w), -COEF_BOUND, COEF_BOUND)
        self.train_loss_ = l
        self.lower_bound_ = lb
        self.cert_stats_ = C.stats
        return self


def _jit_warmup():
    """Compile every numba kernel at import (binary, small-integer and continuous columns), so that no
    compilation happens inside a timed fit. No data or state is kept."""
    rng = np.random.default_rng(0)
    n = 200
    Xb = (rng.random((n, 20)) < 0.4).astype(float)
    Xi = rng.integers(1, 6, size=(n, 2)).astype(float)
    Xc = rng.random((n, 12)) * 10
    X = np.hstack([Xb, Xi, Xc])
    y = (X[:, 0] + X[:, 4] / 5 + X[:, 6] / 10 + rng.random(n) > 1.5).astype(np.int64)
    SparseIntegerClassifier(k=3, time_limit=1e9).fit(X, y)
    CertifyingClassifier(k=2, time_limit=2.0).fit(X[:, [0, 4, 6, 20, 22]], y)
    CertifyingClassifier(k=2, time_limit=2.0).fit(X[:, :6], y)
    # call every certification kernel directly once (some paths are data dependent)
    for cols in ([0, 4, 6, 1, 2, 3], [0, 4, 20, 22, 25, 26]):
        C = _Cert(X[:, cols], y, 3, np.zeros(len(cols)), 1e18)
        Tc, cid, nc = C._cells_T([0])
        cands = np.arange(1, C.D).astype(np.int64)
        pair_scan(C.codes, C.nv, C.Xs, C.P, C.N, Tc, cid, nc, cands, 0, len(cands) - 1, 0.0, C.limit, 3, *C.csc,
                  np.zeros(1), C.xlx)
        leaf_scan(C.codes, C.nv, C.Xs, C.P, C.N, Tc, cid, nc, cands, 0.0, C.limit, 3, *C.csc, np.zeros(1), C.xlx)
        gram = pair_gram(C.P, C.N, cid, nc, cands, C.rowptr, C.rowidx, C.D)
        pair_scan_gram(C.Xs, C.P, C.N, Tc, cid, nc, cands, 0, len(cands) - 1, 0.0, 3, C.csc[2], C.csc[3], *gram,
                       np.zeros(1), C.xlx)


_jit_warmup()


# ===========================================================================
# Evaluation loop (do not edit below this line)
# ===========================================================================

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "src"))
    from exact import evaluate_exact, print_exact, record_exact

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="", help="comma-separated subset of the visible suite (default: all)")
    ap.add_argument("--ks", default="", help="comma-separated subset of the k grid (default: all)")
    ap.add_argument("--skip-tiny", action="store_true", help="skip the tiny correctness suite (not recorded)")
    ap.add_argument("--jobs", type=int, default=14, help="parallel worker processes")
    ap.add_argument("--no-record", action="store_true", help="do not write results/")
    args = ap.parse_args()
    t0 = time.time()
    datasets = [d for d in args.datasets.split(",") if d] or None
    ks = [int(v) for v in args.ks.split(",") if v] or None
    s = evaluate_exact(("file", os.path.abspath(__file__)), MODEL_NAME, datasets=datasets, ks=ks, jobs=args.jobs,
                       tiny=not args.skip_tiny)
    if datasets is None and ks is None and not args.skip_tiny and not args.no_record:
        record_exact(MODEL_NAME, DESCRIPTION, s, results_dir=os.path.join(here, "results"))
    elif not args.no_record:
        print("(partial suite: results not recorded)")
    print_exact(MODEL_NAME, s)
    print(f"total_seconds: {time.time() - t0:.1f}s")
