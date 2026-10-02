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
import ctypes
import itertools
import time

import numpy as np
import numba as nb
from numba import types as nbt
from numba.typed import Dict
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir as llir

MODEL_NAME = "y7_own"
DESCRIPTION = ("y6 + each smaller support owned by one s-subset (zeros in the B&B only below the first column "
               "outside the support); bits DFS for all-binary data up to s = 12")
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
# Proofs of every bound below are in notes.md (P1..P6); each use cites them.
EPS = 2.220446049250313e-16
PRUNE_TOL = 2e-8    # a node is discarded when its PROVEN bound >= ub - PRUNE_TOL (mean-loss units)
MAXB = 5            # COEF_BOUND
FAMC = 0.5          # family-bound call rule (cost model), 0 disables
GIVEUP = 12.0       # give up when the projected finish exceeds this multiple of the budget
GIVEUP_MIN = 3.0    # ... but only after this many seconds
BBSKIP = True       # P4b: no re-solve at the first nonzero of a B&B prefix
BBMEMO = True       # P4c: reuse the bound of the reduced prefix
PAIRLEAF = True     # P12: paired child cells at the leaf level of the bits DFS
VERT = True         # cell-major leaf pass (same bound, better locality)
ONEREP = True       # P17: one Hessian in the gap bound
COMPACT = True      # P16: rows of each depth s-2 cell packed densely (PEXT) for the leaf level
LRORDER = True      # P13: reorder a node's remaining candidates after a failed family solve

_libc = ctypes.CDLL(None)
_cgt = _libc.clock_gettime
_cgt.restype = ctypes.c_int
_cgt.argtypes = [ctypes.c_int, ctypes.c_void_p]


@nb.njit(cache=False)
def _cpu():
    """Wall clock (CLOCK_MONOTONIC, the clock of time.monotonic) inside numba."""
    ts = np.zeros(2, np.int64)
    _cgt(1, ts.ctypes.data)
    return ts[0] + ts[1] * 1e-9


@nb.njit(cache=False, inline="always")
def _sp(z):
    # log(1 + e^z), stable
    if z > 0:
        return z + np.log1p(np.exp(-z))
    return np.log1p(np.exp(z))


@nb.njit(cache=False, inline="always")
def _dsig(t):
    # sigma(t) (1 - sigma(t)): even, decreasing in |t|
    e = np.exp(-abs(t))
    return e / ((1.0 + e) * (1.0 + e))


@nb.njit(cache=False, inline="always")
def _h2(p, q):
    # saturated loss of a cell with p positives, q negatives (P1)
    n = p + q
    r = 0.0
    if p > 0:
        r += p * np.log(n / p)
    if q > 0:
        r += q * np.log(n / q)
    return r


@nb.njit(cache=False)
def _chol_solve(A, g, p, d, L):
    """d = A^{-1} g by Cholesky (A p x p); damping added until it succeeds. Only a search direction."""
    tr = 0.0
    for i in range(p):
        tr += A[i, i]
    lam = 0.0
    for attempt in range(30):
        ok = True
        for i in range(p):
            for j in range(i + 1):
                s = A[i, j]
                if i == j:
                    s += lam
                for q in range(j):
                    s -= L[i, q] * L[j, q]
                if i == j:
                    if s <= 0.0:
                        ok = False
                        break
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
            if not ok:
                break
        if ok:
            for i in range(p):
                s = g[i]
                for q in range(i):
                    s -= L[i, q] * d[q]
                d[i] = s / L[i, i]
            for i in range(p - 1, -1, -1):
                s = d[i]
                for q in range(i + 1, p):
                    s -= L[q, i] * d[q]
                d[i] = s / L[i, i]
            return True
        lam = max(lam * 10.0, 1e-12 * (tr + 1e-300))
    return False


@nb.njit(cache=False)
def _lmin(A, p):
    if p == 1:
        return A[0, 0]
    if p == 2:
        a, b, c = A[0, 0], A[0, 1], A[1, 1]
        m = 0.5 * (a + c)
        r = np.sqrt(0.25 * (a - c) * (a - c) + b * b)
        # stable smaller root: det / larger
        big = m + r
        if big <= 0.0:
            return 0.0
        return (a * c - b * b) / big
    return np.linalg.eigvalsh(A)[0]


@nb.njit(cache=False)
def _chol_ok(A, p, shift, L):
    """True iff the floating-point Cholesky of A - shift I runs to completion."""
    for i in range(p):
        for j in range(i + 1):
            s = A[i, j]
            if i == j:
                s -= shift
            for q in range(j):
                s -= L[i, q] * L[j, q]
            if i == j:
                if not (s > 0.0):
                    return False
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return True


@nb.njit(cache=False)
def _lmin_rig(A, p, marg):
    """PROVEN lower bound on lambda_min of the symmetric matrix A (or <= 0 when none is found), for p > 2.
    An estimate lam from inverse iteration (not trusted) proposes mu; mu is accepted only if the
    floating-point Cholesky of A - (mu + marg) I completes. Completion implies that A - (mu + marg) I + E
    is positive semidefinite with ||E||_2 <= gamma_{p+1} tr(|R|^T |R|) <= 2 (p + 1) eps tr(A) (Higham,
    Thm 10.3); marg covers that and the rounding of A itself, so lambda_min(A) >= mu."""
    L = np.empty((p, p))
    if not _chol_ok(A, p, 0.0, L):
        return -1.0
    v = np.empty(p)
    x = np.empty(p)
    for i in range(p):
        v[i] = 1.0 / np.sqrt(p) * (1.0 + 0.01 * i)
    lam = 0.0
    for it in range(10):
        for i in range(p):
            s = v[i]
            for q in range(i):
                s -= L[i, q] * x[q]
            x[i] = s / L[i, i]
        for i in range(p - 1, -1, -1):
            s = x[i]
            for q in range(i + 1, p):
                s -= L[q, i] * x[q]
            x[i] = s / L[i, i]
        nr = 0.0
        for i in range(p):
            nr += x[i] * x[i]
        nr = np.sqrt(nr)
        if not (nr > 0.0):
            return -1.0
        lam = 1.0 / nr
        for i in range(p):
            v[i] = x[i] / nr
    for fac in (0.98, 0.85, 0.6, 0.3):
        mu = fac * lam
        if mu <= marg:
            return -1.0
        if _chol_ok(A, p, mu + marg, L):
            return mu
    return -1.0


@nb.njit(cache=False)
def lr_bound(U, C, p, pos, neg, th, maxit, stop_below):
    """Minimise F(theta) = sum_c pos_c sp(-u_c.theta) + neg_c sp(u_c.theta) from theta (in place).
    Returns (lb, F): lb a PROVEN lower bound on inf F (P3), or -inf when P3 cannot be closed.
    If F drops below stop_below the solve stops early and returns (-inf, F) (the caller only needed
    to know the node cannot be discarded)."""
    z = np.empty(C)
    g = np.empty(p)
    H = np.empty((p, p))
    L = np.empty((p, p))
    dd = np.empty(p)
    thn = np.empty(p)
    F = 0.0
    for c in range(C):
        s = 0.0
        for j in range(p):
            s += U[c, j] * th[j]
        z[c] = s
        F += pos[c] * _sp(-s) + neg[c] * _sp(s)
    for it in range(maxit):
        if F < stop_below:
            return -np.inf, F
        for j in range(p):
            g[j] = 0.0
            for q in range(p):
                H[j, q] = 0.0
        for c in range(C):
            e = np.exp(-abs(z[c]))
            sg = 1.0 / (1.0 + e) if z[c] >= 0 else e / (1.0 + e)
            n = pos[c] + neg[c]
            r = n * sg - pos[c]
            w = n * e / ((1.0 + e) * (1.0 + e))
            for j in range(p):
                uj = U[c, j]
                g[j] += r * uj
                wu = w * uj
                for q in range(j + 1):
                    H[j, q] += wu * U[c, q]
        for j in range(p):
            for q in range(j):
                H[q, j] = H[j, q]
        if not _chol_solve(H, g, p, dd, L):
            break
        dec = 0.0
        for j in range(p):
            dec += g[j] * dd[j]
        if dec < 1e-13 * (F + 1.0):
            break
        t = 1.0
        acc = False
        Fn = F
        for ls in range(40):
            for j in range(p):
                thn[j] = th[j] - t * dd[j]
            Fn = 0.0
            for c in range(C):
                s = 0.0
                for j in range(p):
                    s += U[c, j] * thn[j]
                Fn += pos[c] * _sp(-s) + neg[c] * _sp(s)
            if Fn <= F - 1e-4 * t * dec:
                acc = True
                break
            t *= 0.5
        if not acc:
            break
        for j in range(p):
            th[j] = thn[j]
        for c in range(C):
            s = 0.0
            for j in range(p):
                s += U[c, j] * th[j]
            z[c] = s
        F = Fn
    return gap_bound(U, C, p, pos, neg, th, z, F), F


@nb.njit(cache=False)
def gap_bound(U, C, p, pos, neg, th, z, F):
    """P3: proven lower bound on inf F from the point th (z = U th, F = F(th)), or -inf."""
    g = np.zeros(p)
    gerr = 0.0
    unorm = np.empty(C)
    zerr = np.empty(C)
    Fp = 0.0
    for c in range(C):
        n = pos[c] + neg[c]
        e = np.exp(-abs(z[c]))
        sg = 1.0 / (1.0 + e) if z[c] >= 0 else e / (1.0 + e)
        r = n * sg - pos[c]
        s2 = 0.0
        sa = 0.0
        for j in range(p):
            g[j] += r * U[c, j]
            s2 += U[c, j] * U[c, j]
            sa += abs(U[c, j] * th[j])
        unorm[c] = np.sqrt(s2)
        zerr[c] = 4.0 * p * EPS * sa
        gerr += n * unorm[c]
        Fp += pos[c] * _sp(-z[c]) + neg[c] * _sp(z[c])
    Fp = min(Fp, F)
    G = 0.0
    for j in range(p):
        G += g[j] * g[j]
    G = np.sqrt(G) * (1.0 + 1e-12) + 4.0 * (C + p) * EPS * gerr
    Flow = Fp * (1.0 - 8.0 * (C + 4) * EPS)
    if G == 0.0:
        return Flow
    H = np.empty((p, p))
    R = 0.0
    for rep in range(8):
        for j in range(p):
            for q in range(p):
                H[j, q] = 0.0
        tr = 0.0
        for c in range(C):
            w = (pos[c] + neg[c]) * _dsig(abs(z[c]) + R * unorm[c] + zerr[c]) * (1.0 - 1e-12)
            for j in range(p):
                wu = w * U[c, j]
                for q in range(j + 1):
                    H[j, q] += wu * U[c, q]
        for j in range(p):
            tr += H[j, j]
            for q in range(j):
                H[q, j] = H[j, q]
        if p <= 2:
            mu = _lmin(H, p) - (10.0 * p + p * C + 10.0) * EPS * tr
        else:
            mu = _lmin_rig(H, p, (12.0 * p + p * C + 12.0) * EPS * tr)
        if not (mu > 0.0):
            return -np.inf
        need = 2.0 * G / mu
        if R >= need:
            return Flow - G * G / (2.0 * mu)
        if rep == 0 and ONEREP:
            # P17: on the ball of radius R the Hessian is >= exp(-R umax) H_0 (H_0 = the matrix just built,
            # with |z| + zerr), so mu(R) >= exp(-R umax) mu; accept R if R >= 2 G / mu(R)
            umax = 0.0
            for c in range(C):
                if unorm[c] > umax:
                    umax = unorm[c]
            R1 = need
            for it2 in range(30):
                R1 = need * np.exp(min(R1 * umax, 700.0))
            R1 = 1.25 * R1
            muR = mu * np.exp(-R1 * umax) * (1.0 - 1e-12)
            if muR > 0.0 and R1 * muR >= 2.0 * G:
                return Flow - G * G / (2.0 * muR)
        R = 1.25 * need
    return -np.inf


@nb.njit(cache=False)
def part_sat(U, C, p, pos, neg):
    """P1 on the partition of the cells by their row of U: any score u_c.theta is a function of that row,
    so its loss is >= the saturated loss of the partition. Rows are grouped when exactly equal and
    adjacent after sorting by a projection; a finer grouping only lowers the value, so it stays valid."""
    key = np.empty(C)
    for c in range(C):
        s = 0.0
        for j in range(p):
            s += U[c, j] * (1.0 + 0.6180339887 * j + 0.1 * j * j)
        key[c] = s
    o = np.argsort(key)
    tot = 0.0
    gp = pos[o[0]]
    gn = neg[o[0]]
    for r in range(1, C):
        a = o[r - 1]
        b = o[r]
        same = True
        for j in range(p):
            if U[a, j] != U[b, j]:
                same = False
                break
        if same:
            gp += pos[b]
            gn += neg[b]
        else:
            tot += _h2(gp, gn)
            gp = pos[b]
            gn = neg[b]
    tot += _h2(gp, gn)
    return tot * (1.0 - 1e-10)


@nb.njit(cache=False)
def reduce_cols(U, C, p, keep):
    """P8: columns of U that are EXACTLY linear combinations of earlier kept columns on the kept cells
    are dropped (the set of reachable score vectors on those cells is unchanged, so the minimum is the
    same). A dependency found numerically is used only after it is verified exactly: small-integer
    coefficients a (common denominator q <= 64) with sum_i a_i U[c, J_i] == q U[c, j] on every kept
    cell, all entries integers below 2^20 so that the float arithmetic is exact. Returns the mask."""
    colk = np.zeros(p, np.bool_)
    J = np.zeros(p, np.int64)
    nJ = 0
    intok = True
    for c in range(C):
        if keep[c]:
            for j in range(p):
                v = U[c, j]
                if v != np.floor(v) or abs(v) > 1048576.0:
                    intok = False
    for j in range(p):
        allz = True
        for c in range(C):
            if keep[c] and U[c, j] != 0.0:
                allz = False
                break
        if allz:
            continue           # an exactly zero column
        if nJ == 0:
            colk[j] = True
            J[0] = j
            nJ = 1
            continue
        # least squares of column j on the kept columns J (normal equations, tiny system)
        G = np.zeros((nJ, nJ))
        r = np.zeros(nJ)
        uu = 0.0
        for c in range(C):
            if not keep[c]:
                continue
            uj = U[c, j]
            uu += uj * uj
            for a_ in range(nJ):
                va = U[c, J[a_]]
                r[a_] += va * uj
                for b_ in range(a_ + 1):
                    G[a_, b_] += va * U[c, J[b_]]
        for a_ in range(nJ):
            for b_ in range(a_):
                G[b_, a_] = G[a_, b_]
        lam = np.zeros(nJ)
        Lw = np.empty((nJ, nJ))
        _chol_solve(G, r, nJ, lam, Lw)
        res = 0.0
        for c in range(C):
            if not keep[c]:
                continue
            e = U[c, j]
            for a_ in range(nJ):
                e -= lam[a_] * U[c, J[a_]]
            res += e * e
        dep = False
        if intok and res <= 1e-16 * (uu + 1.0):
            for q in range(1, 65):
                ok = True
                for a_ in range(nJ):
                    t = q * lam[a_]
                    if abs(t - np.round(t)) > 1e-6 or abs(t) > 1048576.0:
                        ok = False
                        break
                if not ok:
                    continue
                # exact verification
                for c in range(C):
                    if not keep[c]:
                        continue
                    acc = 0.0
                    for a_ in range(nJ):
                        acc += np.round(q * lam[a_]) * U[c, J[a_]]
                    if acc != q * U[c, j]:
                        ok = False
                        break
                if ok:
                    dep = True
                break
        if not dep:
            colk[j] = True
            J[nJ] = j
            nJ += 1
    return colk


@nb.njit(cache=False)
def _sub_lr(U, C, p, pos, neg, keep, maxit):
    """P3 on the kept cells and the kept columns (reduce_cols); -inf when not closed."""
    colk = reduce_cols(U, C, p, keep)
    C2 = 0
    for c in range(C):
        if keep[c]:
            C2 += 1
    p2 = 0
    for j in range(p):
        if colk[j]:
            p2 += 1
    if C2 == 0 or p2 == 0:
        return -np.inf
    U2 = np.empty((C2, p2))
    ps = np.empty(C2)
    ns = np.empty(C2)
    r = 0
    for c in range(C):
        if keep[c]:
            q = 0
            for j in range(p):
                if colk[j]:
                    U2[r, q] = U[c, j]
                    q += 1
            ps[r] = pos[c]
            ns[r] = neg[c]
            r += 1
    th2 = np.zeros(p2)
    lb2, F2 = lr_bound(U2, C2, p2, ps, ns, th2, maxit, -np.inf)
    return lb2


@nb.njit(cache=False)
def robust_lr(U, C, p, pos, neg, th, maxit, stop_below):
    """lr_bound, and when P3 cannot be closed: (a) exact column reduction P8 on all cells; (b) P7: drop
    the pure cells whose loss is already negligible (a separated direction) and reduce columns again;
    (c) P1 on the partition of the cells by their row of U. The best valid value is returned."""
    lb, F = lr_bound(U, C, p, pos, neg, th, maxit, stop_below)
    if lb > -np.inf or F < stop_below:
        return lb, F
    best = part_sat(U, C, p, pos, neg)
    keep = np.ones(C, np.bool_)
    lb1 = _sub_lr(U, C, p, pos, neg, keep, maxit)
    if lb1 > best:
        best = lb1
    C2 = 0
    for c in range(C):
        z = 0.0
        for j in range(p):
            z += U[c, j] * th[j]
        lc = pos[c] * _sp(-z) + neg[c] * _sp(z)
        keep[c] = (pos[c] > 0 and neg[c] > 0) or lc > 1e-7
        if keep[c]:
            C2 += 1
    if 0 < C2 < C and lb1 == -np.inf:
        lb2 = _sub_lr(U, C, p, pos, neg, keep, maxit)
        if lb2 > best:
            best = lb2
    return best, F


@nb.njit(cache=False)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@nb.njit(cache=False)
def int_bb(Z, C, k, pos, neg, perm, ub, thr, blev0, Hsat, best_w, t_end, zok):
    """Branch and bound over integer w in [-5, 5]^k on one support (cells Z: C x k), P4/P5.
    Returns (lbmin, ub, improved, complete, nodes). lbmin: min over every bound used to discard a
    node and every leaf lower bound, i.e. a proven lower bound on every w of this support when
    complete; best_w (length k, columns of Z) receives an improving vector when one is found.
    Coefficients per level are kept in a canonical layout [alpha, beta_0..beta_{k-1}, b] (position
    in perm order) for warm starts only."""
    N = 0.0
    P = 0.0
    for c in range(C):
        N += pos[c] + neg[c]
        P += pos[c]
    base = _h2(P, N - P)          # exact minimum when every score is equal (P5)
    w = np.zeros(k, np.int64)
    val = np.zeros(k, np.int64)
    sF = np.zeros((k + 1, C))     # prefix score per level
    thc = np.zeros((k + 1, k + 2))
    blev = np.empty(k + 1)        # proven bound valid for the node at each level (P4 monotone)
    U = np.empty((C, k + 2))
    th = np.empty(k + 2)
    lbmin = np.inf
    improved = False
    b0 = np.log(max(P, 0.5) / max(N - P, 0.5))
    thc[0, k + 1] = b0
    blev[0] = blev0
    nodes = 0
    m = 0
    val[0] = 0
    memo = Dict.empty(key_type=nbt.int64, value_type=nbt.float64)
    while m >= 0:
        nzp = False
        for i in range(m):
            if w[i] != 0:
                nzp = True
                break
        lo = -MAXB if nzp else 0
        v = val[m]
        if v > MAXB:
            m -= 1
            if m >= 0:
                val[m] += 1
            continue
        if v == 0 and not zok[perm[m]] and lo <= 0:
            # P18: this support does not own vectors with a zero at this coordinate
            val[m] += 1
            continue
        if v < lo:
            v = lo
            val[m] = v
        w[m] = v
        col = perm[m]
        for c in range(C):
            sF[m + 1, c] = sF[m, c] + v * Z[c, col]
        nodes += 1
        if (nodes & 255) == 0 and _cpu() > t_end:
            return lbmin, ub, improved, False, nodes
        has_pref = nzp or v != 0
        # warm start in canonical layout
        alpha = thc[m, 0] if nzp else (thc[m, 1 + m] / v if v != 0 else 0.0)
        if m == k - 1:
            # leaf: sign symmetry holds by construction; skip non-primitive vectors (P4)
            gg = 0
            for i in range(k):
                gg = _gcd(gg, abs(w[i]))
            if gg <= 1:
                allsame = True
                for c in range(1, C):
                    if sF[k, c] != sF[k, 0]:
                        allsame = False
                        break
                if allsame:
                    lb = base * (1.0 - 1e-12)
                    Fv = base
                else:
                    for c in range(C):
                        U[c, 0] = sF[k, c]
                        U[c, 1] = 1.0
                    th[0] = alpha
                    th[1] = thc[m, k + 1]
                    lb, Fv = robust_lr(U, C, 2, pos, neg, th, 60, -np.inf)
                    lb = max(lb, blev[m])   # P4: the parent's bound is valid for the leaf
                if lb < lbmin:
                    lbmin = lb
                if Fv < ub:
                    ub = Fv
                    improved = True
                    for i in range(k):
                        best_w[perm[i]] = w[i]
            val[m] += 1
            continue
        if BBSKIP and (not nzp) and v != 0:
            # first nonzero of the prefix: the columns [v z_m, z_{m+1}.., 1] span the same set as the
            # parent's [z_m, z_{m+1}.., 1], so the relaxation (and its bound) is the parent's (P4b):
            # descend without solving it again.
            thc[m + 1, :] = thc[m, :]
            thc[m + 1, 0] = thc[m, 1 + m] / v
            thc[m + 1, 1 + m] = 0.0
            blev[m + 1] = blev[m]
            m += 1
            val[m] = -MAXB
            continue
        # P4c: prefixes w_P and w_P / g (g = gcd > 1) have the same relaxation (alpha absorbs g); the
        # reduced prefix has a smaller first nonzero, so it was visited before: reuse its bound.
        key = -1
        if BBMEMO and nzp:
            gg = 0
            for i in range(m + 1):
                gg = _gcd(gg, abs(w[i]))
            key = m + 1
            for i in range(m + 1):
                key = key * 11 + (w[i] // gg + MAXB)
            if gg > 1:
                if key in memo:
                    lbr = max(memo[key], blev[m])
                    if lbr >= ub - thr:
                        if lbr < lbmin:
                            lbmin = lbr
                        val[m] += 1
                        continue
                    thc[m + 1, :] = thc[m, :]
                    thc[m + 1, 1 + m] = 0.0
                    blev[m + 1] = lbr
                    m += 1
                    val[m] = -MAXB
                    continue
                key = -1
        # internal node: relaxation with columns [prefix score (if nonzero), remaining z's, 1] (P4)
        p = 0
        if has_pref:
            for c in range(C):
                U[c, 0] = sF[m + 1, c]
            th[0] = alpha
            p = 1
        for i in range(m + 1, k):
            ci = perm[i]
            for c in range(C):
                U[c, p] = Z[c, ci]
            th[p] = thc[m, 1 + i]
            p += 1
        for c in range(C):
            U[c, p] = 1.0
        th[p] = thc[m, k + 1]
        p += 1
        lb, Fv = robust_lr(U, C, p, pos, neg, th, 60, ub - thr)
        if key >= 0:
            memo[key] = lb
        lb = max(lb, blev[m])   # the parent's bound stays valid for the child (P4)
        if lb >= ub - thr:
            if lb < lbmin:
                lbmin = lb
            val[m] += 1
            continue
        blev[m + 1] = max(lb, blev[m])
        q = 0
        thc[m + 1, :] = 0.0
        if has_pref:
            thc[m + 1, 0] = th[0]
            q = 1
        for i in range(m + 1, k):
            thc[m + 1, 1 + i] = th[q]
            q += 1
        thc[m + 1, k + 1] = th[q]
        m += 1
        val[m] = -MAXB
    return lbmin, ub, improved, True, nodes


@nb.njit(cache=False)
def _wgram32(U32, w):
    """U^T diag(w) U in single precision (a Newton search direction only), returned in float64."""
    n, p = U32.shape
    A = np.empty((n, p), np.float32)
    for i in range(n):
        sw = np.float32(np.sqrt(w[i]))
        for j in range(p):
            A[i, j] = U32[i, j] * sw
    AT = np.ascontiguousarray(A.T)
    return np.dot(AT, A).astype(np.float64)


@nb.njit(cache=False)
def _wgram(U, w):
    """U^T diag(w) U by BLAS (w >= 0)."""
    n, p = U.shape
    A = np.empty((n, p))
    for i in range(n):
        sw = np.sqrt(w[i])
        for j in range(p):
            A[i, j] = U[i, j] * sw
    AT = np.ascontiguousarray(A.T)
    return np.dot(AT, A)


@nb.njit(cache=False)
def lr_bound_big(U, n, p, pos, neg, th, maxit, stop_below):
    """lr_bound for many columns (Hessians by BLAS); same guarantees (P3)."""
    z = np.dot(U, th)
    F = 0.0
    for i in range(n):
        F += pos[i] * _sp(-z[i]) + neg[i] * _sp(z[i])
    r = np.empty(n)
    w = np.empty(n)
    dd = np.empty(p)
    L = np.empty((p, p))
    UT = np.ascontiguousarray(U.T)
    U32 = U.astype(np.float32)
    for it in range(maxit):
        if F < stop_below:
            return -np.inf, F
        for i in range(n):
            e = np.exp(-abs(z[i]))
            sg = 1.0 / (1.0 + e) if z[i] >= 0 else e / (1.0 + e)
            nn = pos[i] + neg[i]
            r[i] = nn * sg - pos[i]
            w[i] = nn * e / ((1.0 + e) * (1.0 + e))
        g = np.dot(UT, r)
        # search direction only (never part of a bound): single-precision Gram matrix
        H = _wgram32(U32, w)
        if not _chol_solve(H, g, p, dd, L):
            break
        dec = 0.0
        for j in range(p):
            dec += g[j] * dd[j]
        if dec < 1e-13 * (F + 1.0):
            break
        t = 1.0
        acc = False
        Fn = F
        zd = np.dot(U, dd)
        for ls in range(40):
            Fn = 0.0
            for i in range(n):
                zz = z[i] - t * zd[i]
                Fn += pos[i] * _sp(-zz) + neg[i] * _sp(zz)
            if Fn <= F - 1e-4 * t * dec:
                acc = True
                break
            t *= 0.5
        if not acc:
            break
        for j in range(p):
            th[j] -= t * dd[j]
        z = np.dot(U, th)
        F = Fn
    return gap_bound_big(U, UT, n, p, pos, neg, th, z, F), F


@nb.njit(cache=False)
def gap_bound_big(U, UT, n, p, pos, neg, th, z, F):
    """P3 with BLAS Hessians; error terms as in gap_bound (n + p terms per sum)."""
    r = np.empty(n)
    unorm = np.empty(n)
    zerr = np.empty(n)
    gerr = 0.0
    Fp = 0.0
    for i in range(n):
        nn = pos[i] + neg[i]
        e = np.exp(-abs(z[i]))
        sg = 1.0 / (1.0 + e) if z[i] >= 0 else e / (1.0 + e)
        r[i] = nn * sg - pos[i]
        s2 = 0.0
        sa = 0.0
        for j in range(p):
            s2 += U[i, j] * U[i, j]
            sa += abs(U[i, j] * th[j])
        unorm[i] = np.sqrt(s2)
        zerr[i] = 4.0 * p * EPS * sa
        gerr += nn * unorm[i]
        Fp += pos[i] * _sp(-z[i]) + neg[i] * _sp(z[i])
    Fp = min(Fp, F)
    g = np.dot(UT, r)
    G = 0.0
    for j in range(p):
        G += g[j] * g[j]
    G = np.sqrt(G) * (1.0 + 1e-12) + 4.0 * (n + p) * EPS * gerr
    Flow = Fp * (1.0 - 8.0 * (n + 4) * EPS)
    if G == 0.0:
        return Flow
    w = np.empty(n)
    R = 0.0
    for rep in range(8):
        for i in range(n):
            w[i] = (pos[i] + neg[i]) * _dsig(abs(z[i]) + R * unorm[i] + zerr[i]) * (1.0 - 1e-12)
        H = _wgram(U, w)
        tr = 0.0
        for j in range(p):
            tr += H[j, j]
        mu = _lmin(H, p) - (10.0 * p + p * n + 10.0) * EPS * tr
        if not (mu > 0.0):
            return -np.inf
        need = 2.0 * G / mu
        if R >= need:
            return Flow - G * G / (2.0 * mu)
        if rep == 0 and ONEREP:
            # P17: on the ball of radius R the Hessian is >= exp(-R umax) H_0 (H_0 = the matrix just built,
            # with |z| + zerr), so mu(R) >= exp(-R umax) mu; accept R if R >= 2 G / mu(R)
            umax = 0.0
            for c in range(n):
                if unorm[c] > umax:
                    umax = unorm[c]
            R1 = need
            for it2 in range(30):
                R1 = need * np.exp(min(R1 * umax, 700.0))
            R1 = 1.25 * R1
            muR = mu * np.exp(-R1 * umax) * (1.0 - 1e-12)
            if muR > 0.0 and R1 * muR >= 2.0 * G:
                return Flow - G * G / (2.0 * muR)
        R = 1.25 * need
    return -np.inf


@nb.njit(cache=False)
def reduce_cols_big(U, n, p):
    """P8 for many columns: greedy Gram-Schmidt; a column whose residual vanishes numerically is
    dropped only after its exact integer dependency on the kept columns is verified (as reduce_cols)."""
    colk = np.zeros(p, np.bool_)
    intok = True
    for i in range(n):
        for j in range(p):
            v = U[i, j]
            if v != np.floor(v) or abs(v) > 1048576.0:
                intok = False
    Q = np.zeros((n, p))
    Rm = np.zeros((p, p))
    J = np.zeros(p, np.int64)
    nJ = 0
    v = np.empty(n)
    for j in range(p):
        nrm0 = 0.0
        for i in range(n):
            v[i] = U[i, j]
            nrm0 += v[i] * v[i]
        if nrm0 == 0.0:
            continue
        coef = np.zeros(nJ)
        for rep in range(2):
            for a_ in range(nJ):
                s = 0.0
                for i in range(n):
                    s += Q[i, a_] * v[i]
                coef[a_] += s
                for i in range(n):
                    v[i] -= s * Q[i, a_]
        nrm = 0.0
        for i in range(n):
            nrm += v[i] * v[i]
        dep = False
        if intok and nJ > 0 and nrm <= 1e-18 * nrm0:
            # coefficients on the kept columns: solve R lam = coef (R upper triangular)
            lam = np.zeros(nJ)
            for a_ in range(nJ - 1, -1, -1):
                s = coef[a_]
                for b_ in range(a_ + 1, nJ):
                    s -= Rm[a_, b_] * lam[b_]
                lam[a_] = s / Rm[a_, a_]
            for q in range(1, 65):
                ok = True
                for a_ in range(nJ):
                    tq = q * lam[a_]
                    if abs(tq - np.round(tq)) > 1e-6 or abs(tq) > 1048576.0:
                        ok = False
                        break
                if not ok:
                    continue
                for i in range(n):
                    acc = 0.0
                    for a_ in range(nJ):
                        acc += np.round(q * lam[a_]) * U[i, J[a_]]
                    if acc != q * U[i, j]:
                        ok = False
                        break
                dep = ok
                break
        if dep:
            continue
        if nrm <= 1e-24 * nrm0:
            # numerically dependent but not provably: keep it (the bound may then fail, never wrong)
            pass
        nv = np.sqrt(nrm)
        if nv == 0.0:
            colk[j] = True
            continue
        for a_ in range(nJ):
            Rm[a_, nJ] = coef[a_]
        Rm[nJ, nJ] = nv
        for i in range(n):
            Q[i, nJ] = v[i] / nv
        J[nJ] = j
        nJ += 1
        colk[j] = True
    return colk


@nb.njit(cache=False)
def fam_bound(Xu, pos_u, neg_u, cols, ncols, thf, maxit, stop_below):
    """Family bound (P9): LR over the columns cols[:ncols] + intercept on all unique rows; thf (d + 1,
    full layout, intercept last) is the warm start and receives the solution. -inf when not closed."""
    n0 = Xu.shape[0]
    d = Xu.shape[1]
    p = ncols + 1
    # rows that are equal on these columns are merged (P14: the objective is a sum over rows, so summing
    # the counts of identical rows leaves it unchanged); grouping by a hash, then exact comparison with
    # the group's first row (rows with equal hash but different values just stay separate groups)
    hv = np.zeros(n0)
    for i in range(n0):
        h = 0.0
        for q in range(ncols):
            h += Xu[i, cols[q]] * (0.6180339887498949 + 0.7548776662466927 * q + 0.5698402909980532 * q * q)
        hv[i] = h
    o = np.argsort(hv)
    U = np.empty((n0, p))
    pos_g = np.empty(n0)
    neg_g = np.empty(n0)
    n = 0
    lead = -1
    for r in range(n0):
        i = o[r]
        same = False
        if lead >= 0 and hv[i] == hv[lead]:
            same = True
            for q in range(ncols):
                if Xu[i, cols[q]] != Xu[lead, cols[q]]:
                    same = False
                    break
        if same:
            pos_g[n - 1] += pos_u[i]
            neg_g[n - 1] += neg_u[i]
        else:
            for q in range(ncols):
                U[n, q] = Xu[i, cols[q]]
            U[n, ncols] = 1.0
            pos_g[n] = pos_u[i]
            neg_g[n] = neg_u[i]
            lead = i
            n += 1
    U = np.ascontiguousarray(U[:n])
    pos_u = pos_g[:n]
    neg_u = neg_g[:n]
    th = np.empty(p)
    for q in range(ncols):
        th[q] = thf[cols[q]]
    th[ncols] = thf[d]
    lb, F = lr_bound_big(U, n, p, pos_u, neg_u, th, maxit, stop_below)
    for q in range(ncols):
        thf[cols[q]] = th[q]
    thf[d] = th[ncols]
    if lb > -np.inf or F < stop_below:
        return lb
    colk = reduce_cols_big(U, n, p)
    p2 = 0
    for q in range(p):
        if colk[q]:
            p2 += 1
    if p2 == p:
        return -np.inf
    U2 = np.empty((n, p2))
    th2 = np.empty(p2)
    r = 0
    for q in range(p):
        if colk[q]:
            for i in range(n):
                U2[i, r] = U[i, q]
            th2[r] = th[q]
            r += 1
    lb2, F2 = lr_bound_big(U2, n, p2, pos_u, neg_u, th2, maxit, stop_below)
    return lb2


@nb.njit(cache=False)
def _comb(n, r):
    if r < 0 or r > n:
        return 0.0
    v = 1.0
    for i in range(r):
        v = v * (n - i) / (i + 1)
    return v


@nb.njit(cache=False, inline="always")
def _ht(Lt, p, q):
    # saturated loss of a cell (P1) from the table Lt[m] = m log m (counts are integers)
    ip = int(p + 0.5)
    iq = int(q + 0.5)
    return Lt[ip + iq] - Lt[ip] - Lt[iq]


@nb.njit(cache=False)
def _push(t, j, cid, cp, cn, hc, par, cdep, lvl, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr, lval, curq,
          curid, Lt):
    """Refine the current cells by feature j (the feature at depth t): rows with a non-base code move to
    new cells (children of their cell); in place, undone by _pop."""
    nc0 = ncell[t]
    nc = nc0
    for idx in range(cptr[j], cptr[j + 1]):
        i = crow[idx]
        q = ccode[idx]
        c = cid[i]
        if curq[c] != q:
            curq[c] = q
            curid[c] = nc
            par[nc] = c
            cdep[nc] = t
            lvl[nc] = lval[lptr[j] + q]
            cp[nc] = 0.0
            cn[nc] = 0.0
            nc += 1
        nid = curid[c]
        cid[i] = nid
        cp[nid] += pos_u[i]
        cn[nid] += neg_u[i]
        cp[c] -= pos_u[i]
        cn[c] -= neg_u[i]
    H = Hs[t]
    for c in range(nc0, nc):
        pc = par[c]
        if curq[pc] != -1:
            curq[pc] = -1
            h = _ht(Lt, cp[pc], cn[pc])
            H += h - hc[pc]
            hc[pc] = h
        h = _ht(Lt, cp[c], cn[c])
        hc[c] = h
        H += h
        curq[c] = -1
    ncell[t + 1] = nc
    Hs[t + 1] = H


@nb.njit(cache=False)
def _pop(t, j, cid, cp, cn, hc, par, ncell, cptr, crow, Lt):
    nc0 = ncell[t]
    nc = ncell[t + 1]
    for idx in range(cptr[j], cptr[j + 1]):
        i = crow[idx]
        cid[i] = par[cid[i]]
    for c in range(nc0, nc):
        pc = par[c]
        cp[pc] += cp[c]
        cn[pc] += cn[c]
    for c in range(nc0, nc):
        pc = par[c]
        hc[pc] = _ht(Lt, cp[pc], cn[pc])


@nb.njit(cache=False)
def _leafH(t, j, binj, cid, cp, cn, hc, Hs, pos_u, neg_u, cptr, crow, ccode, stc, stamp, tix, curq, curid, touched,
           mp, mn, npos, nneg, Lt):
    """Saturated loss (P1) of the current cells refined by feature j, without moving rows."""
    nt = 0
    nn = 0
    for idx in range(cptr[j], cptr[j + 1]):
        i = crow[idx]
        c = cid[i]
        if stc[c] != stamp:
            stc[c] = stamp
            tix[c] = nt
            touched[nt] = c
            mp[nt] = 0.0
            mn[nt] = 0.0
            nt += 1
            curq[c] = -1
        r = tix[c]
        mp[r] += pos_u[i]
        mn[r] += neg_u[i]
        if not binj:
            q = ccode[idx]
            if curq[c] != q:
                curq[c] = q
                curid[c] = nn
                npos[nn] = 0.0
                nneg[nn] = 0.0
                nn += 1
            npos[curid[c]] += pos_u[i]
            nneg[curid[c]] += neg_u[i]
    H = Hs[t]
    for r in range(nt):
        c = touched[r]
        curq[c] = -1
        H += _ht(Lt, cp[c] - mp[r], cn[c] - mn[r]) - hc[c]
        if binj:
            H += _ht(Lt, mp[r], mn[r])
    if not binj:
        for r in range(nn):
            H += _ht(Lt, npos[r], nneg[r])
    return H


@nb.njit(cache=False)
def _cells_now(t, ncell, cp, cn, par, cdep, lvl, featd, lptr, lval, Z, zp, zn):
    """Compact the nonempty cells at depth t (features featd[0..t-1]) into Z (C x t), zp, zn."""
    C = 0
    for c in range(ncell[t]):
        if cp[c] + cn[c] > 0:
            cc = c
            for dd in range(t - 1, -1, -1):
                if cdep[cc] == dd:
                    Z[C, dd] = lvl[cc]
                    cc = par[cc]
                else:
                    Z[C, dd] = lval[lptr[featd[dd]]]
            zp[C] = cp[c]
            zn[C] = cn[c]
            C += 1
    return C


@nb.njit(cache=False)
def _stop(stats, fst, t_end, total):
    """Stop at the deadline, or give up early when the measured progress (visited + family-discarded
    leaves out of C(d, s)) projects a finish beyond GIVEUP x the whole budget (no certificate is lost
    that could have been obtained in time unless the rate later rises by that factor)."""
    now = _cpu()
    if now > t_end:
        return True
    el = now - fst[0]
    if el > GIVEUP_MIN and total >= 1e6:
        done = stats[0] + stats[6] + 1.0
        if el * total / done > GIVEUP * (t_end - fst[0]):
            return True
    return False


@nb.njit(cache=False)
def _fam_try(t, jj, d, s, nu, featd, order, fcols, Xu, pos_u, neg_u, thf, ub, thr, fst, stats, depm, isdep, inF):
    """Family bound P9 for the remaining children of a DFS node, called only when its expected saving
    (success rate at this depth x leaves discarded x measured cost per leaf) exceeds its expected cost
    (measured seconds per n_u p^2). Returns the bound, or -inf when not called or not closed.
    fst: [t_start, family seconds, family units, prior leaf cost, calls per depth.., successes per depth..]"""
    pf = t + d - jj + 1
    units = nu * pf * pf
    kappa = (fst[1] + 2e-3) / (fst[2] + 1e6)
    now = _cpu()
    if stats[0] >= 2000:
        cleaf = max(now - fst[0] - fst[1], 0.0) / stats[0]
    else:
        cleaf = fst[3]
    calls = fst[4 + t]
    succ = fst[4 + s + 1 + t]
    pi = (succ + 1.0) / (calls + 2.0)
    nl = _comb(d - jj, s - t)
    if pi * nl * cleaf < kappa * units:
        return -np.inf
    ncol = 0
    for q in range(t):
        fcols[ncol] = featd[q]
        ncol += 1
    for q in range(jj, d):
        fcols[ncol] = order[q]
        ncol += 1
    # P8 with global dependencies (P15): a column that is an exact (verified) combination of the intercept
    # and independent columns all present in this family adds nothing to the reachable scores: drop it
    for q in range(ncol):
        inF[fcols[q]] = True
    m = 0
    for q in range(ncol):
        jq = fcols[q]
        drop = False
        if isdep[jq]:
            drop = True
            for a in range(depm.shape[1]):
                if depm[jq, a] and not inF[a]:
                    drop = False
                    break
        if not drop:
            fcols[m] = jq
            m += 1
    for q in range(ncol):
        inF[fcols[q]] = False
    for q in range(inF.shape[0]):
        inF[q] = False
    ncol = m
    stats[5] += 1
    lbf = fam_bound(Xu, pos_u, neg_u, fcols, ncol, thf, 40, ub - thr)
    fst[1] += _cpu() - now
    fst[2] += units
    fst[4 + t] += 1.0
    if lbf >= ub - thr:
        fst[4 + s + 1 + t] += 1.0
        stats[6] += nl
    return lbf


try:
    import llvmlite.binding as _llb
    HAS_BMI2 = bool(_llb.get_host_cpu_features().get("bmi2", False))
except Exception:  # pragma: no cover
    HAS_BMI2 = False


@intrinsic
def _pext_hw(typingctx, x, m):
    """Parallel bit extract (BMI2 PEXT), only called when the host has BMI2."""
    sig = nbt.uint64(nbt.uint64, nbt.uint64)

    def codegen(context, builder, signature, args):
        fnty = llir.FunctionType(llir.IntType(64), [llir.IntType(64), llir.IntType(64)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.x86.bmi.pext.64")
        return builder.call(fn, [args[0], args[1]])
    return sig, codegen


@nb.njit(cache=False, inline="always")
def _pext(x, m):
    """Bits of x at the set positions of m, packed to the low end (software fallback without BMI2)."""
    if HAS_BMI2:
        return _pext_hw(x, m)
    r = np.uint64(0)
    k = np.uint64(0)
    mm = m
    while mm != np.uint64(0):
        low = mm & (~mm + np.uint64(1))
        if (x & low) != np.uint64(0):
            r |= np.uint64(1) << k
        k += np.uint64(1)
        mm &= mm - np.uint64(1)
    return r


@intrinsic
def _popc(typingctx, x):
    """Population count of a uint64 (LLVM ctpop)."""
    sig = nbt.uint64(nbt.uint64)

    def codegen(context, builder, signature, args):
        fnty = llir.FunctionType(llir.IntType(64), [llir.IntType(64)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.ctpop.i64")
        return builder.call(fn, [args[0]])
    return sig, codegen


@nb.njit(cache=False, inline="always")
def _cmp_part(cB, t, c, w0, w1, Xb, jq, CX, o):
    """P16: append the bits of column jq at the rows of cell c (words w0..w1) to CX[jq, o..], packed."""
    acc = np.uint64(0)
    fill = 0
    for w in range(w0, w1):
        m = cB[t, c, w]
        if m == np.uint64(0):
            continue
        bits = _pext(Xb[jq, w], m)
        k = int(_popc(m))
        if fill == 0:
            acc = bits
        else:
            acc |= bits << np.uint64(fill)
        if fill + k >= 64:
            CX[jq, o] = acc
            o += 1
            if fill == 0:
                acc = np.uint64(0)
            else:
                acc = bits >> np.uint64(64 - fill)
            fill = fill + k - 64
        else:
            fill += k
    if fill > 0:
        CX[jq, o] = acc


@nb.njit(cache=False, inline="always")
def _phi(Lt, a, b, P, N):
    # post-split saturated loss (P1) of a cell with P positives, N negatives whose split part holds (a, b)
    return (Lt[a + b] - Lt[a] - Lt[b]) + (Lt[P - a + N - b] - Lt[P - a] - Lt[N - b])


@nb.njit(cache=False)
def support_dfs_bits(s, Xb, Wp, v0, v1, order, ub, thr, best_w, t_end, stats, Lt, hmarg, Xu, pos_u, neg_u, famc,
                     leaf0, sdc, depm, isdep):
    """support_dfs for all-binary columns with bit-packed raw rows (positives in words [0, Wp), negatives
    after): the cells of a depth are bitsets, a leaf's saturated loss (P1) is 2 popcounts per cell and
    word. Same bounds and the same family rule (P9) as support_dfs."""
    d = order.shape[0]
    W = Xb.shape[1]
    nu = Xu.shape[0]
    MC = 1
    for q in range(s):
        MC *= 2
    cB = np.zeros((s + 1, MC, W), np.uint64)
    ccp = np.zeros((s + 1, MC))
    ccn = np.zeros((s + 1, MC))
    chc = np.zeros((s + 1, MC))
    cval = np.zeros((s + 1, MC, s))
    ncell = np.zeros(s + 2, np.int64)
    Hs = np.zeros(s + 2)
    P = 0.0
    Nn = 0.0
    for i in range(nu):
        P += pos_u[i]
        Nn += neg_u[i]
    iP = int(P + 0.5)
    iN = int(Nn + 0.5)
    for w in range(W):
        cB[0, 0, w] = np.uint64(0)
    for i in range(iP):
        cB[0, 0, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    for i in range(iN):
        cB[0, 0, Wp + (i >> 6)] |= np.uint64(1) << np.uint64(i & 63)
    ncell[0] = 1
    corder = np.zeros((s + 1, MC), np.int64)
    wr = np.zeros((s + 1, MC, 4), np.int64)
    wr[0, 0, 0] = 0
    wr[0, 0, 1] = Wp
    wr[0, 0, 2] = Wp
    wr[0, 0, 3] = W
    ccp[0, 0] = P
    ccn[0, 0] = Nn
    chc[0, 0] = _ht(Lt, P, Nn)
    Hs[0] = chc[0, 0]
    featd = np.zeros(s, np.int64)
    chosen = np.zeros(s, np.int64)
    nxt = np.zeros(s + 1, np.int64)
    wS = np.zeros(s, np.int64)
    zokb = np.zeros(s, np.bool_)
    thT = np.zeros(s + 1)
    Z = np.zeros((2 * MC, s))
    zp = np.zeros(2 * MC)
    zn = np.zeros(2 * MC)
    pa_ = np.zeros(MC)
    na_ = np.zeros(MC)
    lbmin = np.inf
    b0 = np.log(max(P, 0.5) / max(Nn, 0.5))
    fcols = np.zeros(d + 1, np.int64)
    thfam = np.zeros((s + 1, d + 1))
    thfam[0, d] = b0
    fst = np.zeros(4 + 2 * (s + 1))
    fst[0] = _cpu()
    fst[3] = leaf0
    total = _comb(d, s)
    PJa = np.zeros((MC, d), np.int64)
    PJb = np.zeros((MC, d), np.int64)
    cpar = np.zeros((s + 1, MC), np.int64)
    chalf = np.zeros((s + 1, MC), np.int64)
    pk0 = np.zeros(MC, np.int64)
    pk1 = np.zeros(MC, np.int64)
    ploss = np.zeros(MC)
    psm = np.zeros(MC, np.int64)
    porder = np.zeros(MC, np.int64)
    vpost = np.zeros(d)
    cofp = np.zeros(MC + 1, np.int64)
    cofn = np.zeros(MC + 1, np.int64)
    CX = np.zeros((d, W + MC + 2), np.uint64)
    alive = np.zeros(d, np.int64)
    vdead = np.zeros(d, np.bool_)
    fresh = np.zeros(s + 1, np.bool_)
    fresh[0] = True
    # per-depth candidate lists: any order of a node's remaining candidates partitions its remaining
    # supports (P6), so the list may be permuted between children (P13)
    ordl = np.zeros((s + 1, d), np.int64)
    nord = np.zeros(s + 1, np.int64)
    for q in range(d):
        ordl[0, q] = order[q]
    nord[0] = d
    okey = np.zeros(d)
    lst_ = np.zeros(d, np.int64)
    inF = np.zeros(d, np.bool_)
    t = 0
    while True:
        if t == s - 1:
            haveT = False
            L = ncell[t]
            pairm = PAIRLEAF and t >= 1
            npc = 0
            if pairm:
                npc = ncell[t - 1]
                for pc in range(npc):
                    pk0[pc] = -1
                    pk1[pc] = -1
                for r in range(L):
                    if chalf[t, r] == 0:
                        pk0[cpar[t, r]] = r
                    else:
                        pk1[cpar[t, r]] = r
                for pc in range(npc):
                    lsum = 0.0
                    wl0 = 1 << 40
                    wl1 = 1 << 40
                    if pk0[pc] >= 0:
                        r = pk0[pc]
                        lsum += chc[t, r]
                        wl0 = wr[t, r, 1] - wr[t, r, 0] + wr[t, r, 3] - wr[t, r, 2]
                    if pk1[pc] >= 0:
                        r = pk1[pc]
                        lsum += chc[t, r]
                        wl1 = wr[t, r, 1] - wr[t, r, 0] + wr[t, r, 3] - wr[t, r, 2]
                    ploss[pc] = lsum
                    psm[pc] = 0 if wl0 <= wl1 else 1
                po = np.argsort(-ploss[:npc])
                for r in range(npc):
                    porder[r] = po[r]
            vmode = pairm and VERT
            if vmode:
                # cell-major pass (same P1 partial sums as below, P12 counts): every candidate's post-split
                # loss is accumulated parent cell by parent cell; a candidate whose partial sum minus the
                # margin reaches ub - thr is discarded (the other cells add >= 0).
                j0 = nxt[t]
                ncand = nord[t] - j0
                for q in range(ncand):
                    vpost[q] = 0.0
                    alive[q] = q
                    vdead[q] = True
                na = ncand
                for rr in range(npc):
                    pc = porder[rr]
                    if ploss[pc] <= 0.0 or na == 0:
                        break
                    k0 = pk0[pc]
                    k1 = pk1[pc]
                    m = 0
                    if COMPACT and k0 >= 0 and k1 >= 0:
                        fpar = featd[t - 1]
                        p0 = cofp[pc]
                        p1 = cofn[pc]
                        p2 = cofp[pc + 1] if pc + 1 < npc else cofp[npc]
                        Ps = int(ccp[t, k0] + 0.5)
                        Ns = int(ccn[t, k0] + 0.5)
                        Po = int(ccp[t, k1] + 0.5)
                        No = int(ccn[t, k1] + 0.5)
                        for u in range(na):
                            q = alive[u]
                            j = ordl[t, j0 + q]
                            a = 0
                            for w in range(p0, p1):
                                a += _popc(CX[fpar, w] & CX[j, w])
                            b = 0
                            for w in range(p1, p2):
                                b += _popc(CX[fpar, w] & CX[j, w])
                            A = PJa[pc, j]
                            B = PJb[pc, j]
                            v_ = vpost[q] + _phi(Lt, a, b, Ps, Ns) + _phi(Lt, A - a, B - b, Po, No)
                            vpost[q] = v_
                            if v_ - hmarg >= ub - thr:
                                if v_ - hmarg < lbmin:
                                    lbmin = v_ - hmarg
                            else:
                                alive[m] = q
                                m += 1
                    elif k0 >= 0 and k1 >= 0:
                        cs = k0 if psm[pc] == 0 else k1
                        co = k1 if psm[pc] == 0 else k0
                        w0 = wr[t, cs, 0]
                        w1 = wr[t, cs, 1]
                        w2 = wr[t, cs, 2]
                        w3 = wr[t, cs, 3]
                        Ps = int(ccp[t, cs] + 0.5)
                        Ns = int(ccn[t, cs] + 0.5)
                        Po = int(ccp[t, co] + 0.5)
                        No = int(ccn[t, co] + 0.5)
                        for u in range(na):
                            q = alive[u]
                            j = ordl[t, j0 + q]
                            a = 0
                            for w in range(w0, w1):
                                a += _popc(cB[t, cs, w] & Xb[j, w])
                            b = 0
                            for w in range(w2, w3):
                                b += _popc(cB[t, cs, w] & Xb[j, w])
                            A = PJa[pc, j]
                            B = PJb[pc, j]
                            v_ = vpost[q] + _phi(Lt, a, b, Ps, Ns) + _phi(Lt, A - a, B - b, Po, No)
                            vpost[q] = v_
                            if v_ - hmarg >= ub - thr:
                                if v_ - hmarg < lbmin:
                                    lbmin = v_ - hmarg
                            else:
                                alive[m] = q
                                m += 1
                    else:
                        cs = k0 if k0 >= 0 else k1
                        Ps = int(ccp[t, cs] + 0.5)
                        Ns = int(ccn[t, cs] + 0.5)
                        for u in range(na):
                            q = alive[u]
                            j = ordl[t, j0 + q]
                            v_ = vpost[q] + _phi(Lt, PJa[pc, j], PJb[pc, j], Ps, Ns)
                            vpost[q] = v_
                            if v_ - hmarg >= ub - thr:
                                if v_ - hmarg < lbmin:
                                    lbmin = v_ - hmarg
                            else:
                                alive[m] = q
                                m += 1
                    na = m
                for u in range(na):
                    vdead[alive[u]] = False
            for jj in range(nxt[t], nord[t]):
                j = ordl[t, jj]
                if (stats[0] & 63) == 0 and _stop(stats, fst, t_end, total):
                    return lbmin, ub, False
                stats[0] += 1
                if vmode and vdead[jj - nxt[t]]:
                    continue
                # cells in decreasing saturated loss; pure cells (zero loss) stay pure under any split.
                # post = sum over the cells seen so far of their loss after the split; the other cells
                # add >= 0, so post - hmarg is already a valid bound on H(S) (P1): stop once it prunes.
                post = 0.0
                early = False
                if pairm:
                    # paired cells (P12): the two children of a parent cell pc share the parent's counts
                    # with column j (PJ, exact, computed once at the parent node), so one popcount over the
                    # smaller child gives both; a parent with one nonempty child needs no popcount.
                    rr = 0
                    while rr < npc:
                        pc = porder[rr]
                        if ploss[pc] <= 0.0:
                            break
                        k0 = pk0[pc]
                        k1 = pk1[pc]
                        A = PJa[pc, j]
                        B = PJb[pc, j]
                        if COMPACT and k0 >= 0 and k1 >= 0:
                            fpar = featd[t - 1]
                            p0 = cofp[pc]
                            p1 = cofn[pc]
                            p2 = cofp[pc + 1] if pc + 1 < npc else cofp[npc]
                            a = 0
                            for w in range(p0, p1):
                                a += _popc(CX[fpar, w] & CX[j, w])
                            b = 0
                            for w in range(p1, p2):
                                b += _popc(CX[fpar, w] & CX[j, w])
                            pa_[k0] = a
                            na_[k0] = b
                            pa_[k1] = A - a
                            na_[k1] = B - b
                            post += _phi(Lt, a, b, int(ccp[t, k0] + 0.5), int(ccn[t, k0] + 0.5))
                            post += _phi(Lt, A - a, B - b, int(ccp[t, k1] + 0.5), int(ccn[t, k1] + 0.5))
                        elif k0 >= 0 and k1 >= 0:
                            cs = k0 if psm[pc] == 0 else k1
                            co = k1 if psm[pc] == 0 else k0
                            a = 0
                            for w in range(wr[t, cs, 0], wr[t, cs, 1]):
                                a += _popc(cB[t, cs, w] & Xb[j, w])
                            b = 0
                            for w in range(wr[t, cs, 2], wr[t, cs, 3]):
                                b += _popc(cB[t, cs, w] & Xb[j, w])
                            pa_[cs] = a
                            na_[cs] = b
                            pa_[co] = A - a
                            na_[co] = B - b
                            post += _phi(Lt, a, b, int(ccp[t, cs] + 0.5), int(ccn[t, cs] + 0.5))
                            post += _phi(Lt, A - a, B - b, int(ccp[t, co] + 0.5), int(ccn[t, co] + 0.5))
                        else:
                            cs = k0 if k0 >= 0 else k1
                            pa_[cs] = A
                            na_[cs] = B
                            post += _phi(Lt, A, B, int(ccp[t, cs] + 0.5), int(ccn[t, cs] + 0.5))
                        rr += 1
                        if post - hmarg >= ub - thr:
                            early = True
                            break
                    if not early:
                        for r2 in range(rr, npc):
                            pc = porder[r2]
                            if COMPACT:
                                # parents skipped above (zero loss): exact counts from the packed columns
                                k0 = pk0[pc]
                                k1 = pk1[pc]
                                A = PJa[pc, j]
                                B = PJb[pc, j]
                                if k0 >= 0 and k1 >= 0:
                                    fpar = featd[t - 1]
                                    p2 = cofp[pc + 1] if pc + 1 < npc else cofp[npc]
                                    a = 0
                                    for w in range(cofp[pc], cofn[pc]):
                                        a += _popc(CX[fpar, w] & CX[j, w])
                                    b = 0
                                    for w in range(cofn[pc], p2):
                                        b += _popc(CX[fpar, w] & CX[j, w])
                                    pa_[k0] = a
                                    na_[k0] = b
                                    pa_[k1] = A - a
                                    na_[k1] = B - b
                                else:
                                    cs = k0 if k0 >= 0 else k1
                                    pa_[cs] = A
                                    na_[cs] = B
                                continue
                            if pk0[pc] >= 0:
                                pa_[pk0[pc]] = -1.0
                            if pk1[pc] >= 0:
                                pa_[pk1[pc]] = -1.0
                for r in range(L):
                    if pairm:
                        break
                    c = corder[t, r]
                    if chc[t, c] <= 0.0:
                        for r2 in range(r, L):
                            c2 = corder[t, r2]
                            pa_[c2] = -1.0
                        break
                    a = 0
                    for w in range(wr[t, c, 0], wr[t, c, 1]):
                        a += _popc(cB[t, c, w] & Xb[j, w])
                    b = 0
                    for w in range(wr[t, c, 2], wr[t, c, 3]):
                        b += _popc(cB[t, c, w] & Xb[j, w])
                    pa_[c] = a
                    na_[c] = b
                    post += (Lt[a + b] - Lt[a] - Lt[b]) + _ht(Lt, ccp[t, c] - a, ccn[t, c] - b)
                    if post - hmarg >= ub - thr:
                        early = True
                        break
                Hl = post - hmarg
                if early or Hl >= ub - thr:
                    if Hl < lbmin:
                        lbmin = Hl
                    continue
                for c in range(L):
                    if pa_[c] < 0:
                        # a pure cell skipped above: count its rows exactly for the child cells
                        a = 0
                        for w in range(wr[t, c, 0], wr[t, c, 1]):
                            a += _popc(cB[t, c, w] & Xb[j, w])
                        b = 0
                        for w in range(wr[t, c, 2], wr[t, c, 3]):
                            b += _popc(cB[t, c, w] & Xb[j, w])
                        pa_[c] = a
                        na_[c] = b
                stats[1] += 1
                if not haveT:
                    for c in range(L):
                        for q in range(t):
                            Z[c, q] = cval[t, c, q]
                        zp[c] = ccp[t, c]
                        zn[c] = ccn[t, c]
                    _warmT(Z, L, t, zp, zn, thT, b0)
                    haveT = True
                C = 0
                for c in range(L):
                    for half in range(2):
                        pp = pa_[c] if half == 0 else ccp[t, c] - pa_[c]
                        qq = na_[c] if half == 0 else ccn[t, c] - na_[c]
                        if pp + qq > 0:
                            for q in range(t):
                                Z[C, q] = cval[t, c, q]
                            Z[C, t] = v1[j] if half == 0 else v0[j]
                            zp[C] = pp
                            zn[C] = qq
                            C += 1
                featd[t] = j
                chosen[t] = jj
                lbs, ub2, improved, complete = _leaf_solve(Z, C, s, zp, zn, thT, t, Hl, ub, thr, t_end, wS, stats,
                                                           _zok(featd, s, zokb))
                if lbs < lbmin:
                    lbmin = lbs
                if improved:
                    ub = ub2
                    best_w[:] = 0
                    for q in range(s):
                        best_w[featd[q]] = wS[q]
                if not complete:
                    return lbmin, ub, False
            t -= 1
            if t < 0:
                break
            continue
        jj = nxt[t]
        if jj > nord[t] - (s - t):
            t -= 1
            if t < 0:
                break
            continue
        if famc > 0:
            ncall = stats[5]
            lbf = _fam_try(t, jj, nord[t], s, nu, featd, ordl[t], fcols, Xu, pos_u, neg_u, thfam[t], ub, thr, fst,
                           stats, depm, isdep, inF)
            if lbf >= ub - thr:
                if lbf < lbmin:
                    lbmin = lbf
                nxt[t] = nord[t]
                continue
            if LRORDER and stats[5] > ncall:
                # P13: the family solve failed; put the remaining candidates in decreasing |coef| x sd of its
                # (partial) solution, so the next children remove the columns that keep the family below ub
                nr = nord[t] - jj
                for q in range(nr):
                    g_ = ordl[t, jj + q]
                    okey[q] = -abs(thfam[t, g_]) * sdc[g_]
                oq = np.argsort(okey[:nr], kind="mergesort")
                for q in range(nr):
                    lst_[q] = ordl[t, jj + oq[q]]
                for q in range(nr):
                    ordl[t, jj + q] = lst_[q]
        nxt[t] = jj + 1
        chosen[t] = jj
        j = ordl[t, jj]
        featd[t] = j
        if PAIRLEAF and COMPACT and t == s - 2 and fresh[t]:
            # P12 + P16: per cell of this node, the rows are renumbered densely (positives, then negatives)
            # and every column its leaves can add is packed into that numbering (PEXT); counts are exact
            fresh[t] = False
            off = 0
            for c in range(ncell[t]):
                cofp[c] = off
                off += (int(ccp[t, c] + 0.5) + 63) // 64
                cofn[c] = off
                off += (int(ccn[t, c] + 0.5) + 63) // 64
            cofp[ncell[t]] = off
            for q in range(jj, nord[t]):
                jq = ordl[t, q]
                for w in range(off):
                    CX[jq, w] = np.uint64(0)
                for c in range(ncell[t]):
                    _cmp_part(cB, t, c, wr[t, c, 0], wr[t, c, 1], Xb, jq, CX, cofp[c])
                    _cmp_part(cB, t, c, wr[t, c, 2], wr[t, c, 3], Xb, jq, CX, cofn[c])
                    a = 0
                    for w in range(cofp[c], cofn[c]):
                        a += _popc(CX[jq, w])
                    b = 0
                    for w in range(cofn[c], cofp[c + 1] if c + 1 < ncell[t] else off):
                        b += _popc(CX[jq, w])
                    PJa[c, jq] = a
                    PJb[c, jq] = b
        elif PAIRLEAF and t == s - 2 and fresh[t]:
            # P12: exact counts of every cell of this node with every column its leaves can add
            fresh[t] = False
            for q in range(jj, nord[t]):
                jq = ordl[t, q]
                for c in range(ncell[t]):
                    a = 0
                    for w in range(wr[t, c, 0], wr[t, c, 1]):
                        a += _popc(cB[t, c, w] & Xb[jq, w])
                    b = 0
                    for w in range(wr[t, c, 2], wr[t, c, 3]):
                        b += _popc(cB[t, c, w] & Xb[jq, w])
                    PJa[c, jq] = a
                    PJb[c, jq] = b
        # push: split every cell of depth t by column j
        nc = 0
        H = 0.0
        cmode = PAIRLEAF and COMPACT and t == s - 2
        for c in range(ncell[t]):
            if not cmode:
                break
            # P16: child counts from the packed column (half 0 = rows with column j set)
            a = 0
            for w in range(cofp[c], cofn[c]):
                a += _popc(CX[j, w])
            b = 0
            for w in range(cofn[c], cofp[c + 1] if c + 1 < ncell[t] else cofp[ncell[t]]):
                b += _popc(CX[j, w])
            for half in range(2):
                aa = a if half == 0 else int(ccp[t, c] + 0.5) - a
                bb = b if half == 0 else int(ccn[t, c] + 0.5) - b
                if aa + bb > 0:
                    cpar[t + 1, nc] = c
                    chalf[t + 1, nc] = half
                    ccp[t + 1, nc] = aa
                    ccn[t + 1, nc] = bb
                    chc[t + 1, nc] = Lt[aa + bb] - Lt[aa] - Lt[bb]
                    H += chc[t + 1, nc]
                    for q in range(t):
                        cval[t + 1, nc, q] = cval[t, c, q]
                    cval[t + 1, nc, t] = v1[j] if half == 0 else v0[j]
                    nc += 1
        for c in range(ncell[t]):
            if cmode:
                break
            for half in range(2):
                a = 0
                b = 0
                lo0 = -1
                hi0 = -1
                lo1 = -1
                hi1 = -1
                for w in range(wr[t, c, 0], wr[t, c, 1]):
                    x = Xb[j, w] if half == 0 else ~Xb[j, w]
                    v = cB[t, c, w] & x
                    cB[t + 1, nc, w] = v
                    if v != 0:
                        a += _popc(v)
                        if lo0 < 0:
                            lo0 = w
                        hi0 = w
                for w in range(wr[t, c, 2], wr[t, c, 3]):
                    x = Xb[j, w] if half == 0 else ~Xb[j, w]
                    v = cB[t, c, w] & x
                    cB[t + 1, nc, w] = v
                    if v != 0:
                        b += _popc(v)
                        if lo1 < 0:
                            lo1 = w
                        hi1 = w
                # word ranges holding the cell's rows (only words inside the parent's ranges are written)
                wr[t + 1, nc, 0] = lo0 if lo0 >= 0 else 0
                wr[t + 1, nc, 1] = hi0 + 1 if lo0 >= 0 else 0
                wr[t + 1, nc, 2] = lo1 if lo1 >= 0 else 0
                wr[t + 1, nc, 3] = hi1 + 1 if lo1 >= 0 else 0
                if a + b > 0:
                    cpar[t + 1, nc] = c
                    chalf[t + 1, nc] = half
                    ccp[t + 1, nc] = a
                    ccn[t + 1, nc] = b
                    chc[t + 1, nc] = Lt[a + b] - Lt[a] - Lt[b]
                    H += chc[t + 1, nc]
                    for q in range(t):
                        cval[t + 1, nc, q] = cval[t, c, q]
                    cval[t + 1, nc, t] = v1[j] if half == 0 else v0[j]
                    nc += 1
        ncell[t + 1] = nc
        Hs[t + 1] = H
        if not cmode:
            oo = np.argsort(-chc[t + 1, :nc])
            for r in range(nc):
                corder[t + 1, r] = oo[r]
        nxt[t + 1] = 0
        nord[t + 1] = nord[t] - jj - 1
        for q in range(nord[t + 1]):
            ordl[t + 1, q] = ordl[t, jj + 1 + q]
        fresh[t + 1] = True
        for q in range(d + 1):
            thfam[t + 1, q] = thfam[t, q]
        t += 1
    return lbmin, ub, True


@nb.njit(cache=False)
def _zok(featd, s, zokb):
    """P18: coordinate q of the support featd[:s] may be zero only if its column index is below every
    column index outside the support (each smaller support is owned by exactly one s-subset)."""
    m0 = 0
    while True:
        found = False
        for q in range(s):
            if featd[q] == m0:
                found = True
                break
        if not found:
            break
        m0 += 1
    for q in range(s):
        zokb[q] = featd[q] < m0
    return zokb


@nb.njit(cache=False)
def _warmT(Z, CT, t, zp, zn, thT, b0):
    """LR on the cells of T (t columns): only a warm start for its children's LR (not a bound)."""
    UT = np.empty((CT, t + 1))
    for c in range(CT):
        for q in range(t):
            UT[c, q] = Z[c, q]
        UT[c, t] = 1.0
    thT[:] = 0.0
    thT[t] = b0
    lr_bound(UT, CT, t + 1, zp, zn, thT, 30, -np.inf)


@nb.njit(cache=False)
def _leaf_solve(Z, C, s, zp, zn, thT, t, Hl, ub, thr, t_end, wS, stats, zok):
    """A support S that survived P1: LR bound (P2/P3, P7/P8), else integer B&B (P4/P5).
    Returns (bound used for S, ub, improved, complete); wS receives an improving vector."""
    U = np.empty((C, s + 1))
    for c in range(C):
        for q in range(s):
            U[c, q] = Z[c, q]
        U[c, s] = 1.0
    th = np.zeros(s + 1)
    for q in range(t):
        th[q] = thT[q]
    th[s] = thT[t]
    lb, Fv = robust_lr(U, C, s + 1, zp, zn, th, 60, ub - thr)
    lb = max(lb, Hl)
    if lb >= ub - thr:
        return lb, ub, False, True
    stats[2] += 1
    if Fv < ub - thr:
        # the solve stopped early: finish it for the coordinate order (not a bound)
        lr_bound(U, C, s + 1, zp, zn, th, 60, -np.inf)
    key = np.empty(s)
    for q in range(s):
        lo_ = np.inf
        hi_ = -np.inf
        for c in range(C):
            lo_ = min(lo_, Z[c, q])
            hi_ = max(hi_, Z[c, q])
        key[q] = -abs(th[q]) * (hi_ - lo_)
    perm = np.argsort(key)
    wS[:] = 0
    lbs, ub2, improved, complete, nodes = int_bb(Z[:C], C, s, zp[:C], zn[:C], perm, ub, thr, lb, Hl, wS, t_end, zok)
    stats[3] += nodes
    return lbs, ub2, improved, complete


@nb.njit(cache=False)
def support_dfs(s, pos_u, neg_u, cptr, crow, ccode, lptr, lval, order, ub, thr, best_w, t_end, stats, Lt, hmarg, Xu,
                famc, leaf0, sdc, depm, isdep):
    """Visit every support of size s (combinations of `order`) (P6). Leaf S: saturated bound (P1),
    then LR bound (P2/P3), then integer branch and bound (P4/P5). Returns (lbmin, ub, complete).
    best_w (length d, columns of the reduced matrix) receives improving vectors."""
    nu = pos_u.shape[0]
    d = order.shape[0]
    M = (s + 1) * nu + 2
    cid = np.zeros(nu, np.int64)
    cp = np.zeros(M)
    cn = np.zeros(M)
    hc = np.zeros(M)
    par = np.zeros(M, np.int64)
    cdep = -np.ones(M, np.int64)
    lvl = np.zeros(M)
    curq = -np.ones(M, np.int64)
    curid = np.zeros(M, np.int64)
    stc = np.zeros(M, np.int64)
    tix = np.zeros(M, np.int64)
    touched = np.zeros(nu + 1, np.int64)
    mp = np.zeros(nu + 1)
    mn = np.zeros(nu + 1)
    npos = np.zeros(nu + 1)
    nneg = np.zeros(nu + 1)
    ncell = np.zeros(s + 2, np.int64)
    Hs = np.zeros(s + 2)
    Z = np.zeros((nu, s))
    zp = np.zeros(nu)
    zn = np.zeros(nu)
    binf = np.zeros(d, np.bool_)
    for j in range(d):
        binf[j] = (lptr[j + 1] - lptr[j]) == 2
    P = 0.0
    Nn = 0.0
    for i in range(nu):
        P += pos_u[i]
        Nn += neg_u[i]
    ncell[0] = 1
    cp[0] = P
    cn[0] = Nn
    hc[0] = _ht(Lt, P, Nn)
    Hs[0] = hc[0]
    featd = np.zeros(s, np.int64)
    chosen = np.zeros(s, np.int64)
    nxt = np.zeros(s + 1, np.int64)
    wS = np.zeros(s, np.int64)
    zokb = np.zeros(s, np.bool_)
    thT = np.zeros(s + 1)
    lbmin = np.inf
    stamp = 0
    b0 = np.log(max(P, 0.5) / max(Nn, 0.5))
    fcols = np.zeros(d + 1, np.int64)
    thfam = np.zeros((s + 1, d + 1))
    thfam[0, d] = b0
    fst = np.zeros(4 + 2 * (s + 1))
    fst[0] = _cpu()
    fst[3] = leaf0
    total = _comb(d, s)
    avgnnz = cptr[d] / max(d, 1)
    ordl = np.zeros((s + 1, d), np.int64)
    nord = np.zeros(s + 1, np.int64)
    for q in range(d):
        ordl[0, q] = order[q]
    nord[0] = d
    okey = np.zeros(d)
    lst_ = np.zeros(d, np.int64)
    inF = np.zeros(d, np.bool_)
    t = 0
    while True:
        if t == s - 1:
            haveT = False
            for jj in range(nxt[t], nord[t]):
                j = ordl[t, jj]
                if (stats[0] & 15) == 0 and _stop(stats, fst, t_end, total):
                    return lbmin, ub, False
                stats[0] += 1
                stamp += 1
                H = _leafH(t, j, binf[j], cid, cp, cn, hc, Hs, pos_u, neg_u, cptr, crow, ccode, stc, stamp, tix,
                           curq, curid, touched, mp, mn, npos, nneg, Lt)
                Hl = H - hmarg      # proven floating-point margin on the table-based sum (notes, x3)
                if Hl >= ub - thr:
                    if Hl < lbmin:
                        lbmin = Hl
                    continue
                stats[1] += 1
                if not haveT:
                    CT = _cells_now(t, ncell, cp, cn, par, cdep, lvl, featd, lptr, lval, Z, zp, zn)
                    _warmT(Z, CT, t, zp, zn, thT, b0)
                    haveT = True
                featd[t] = j
                chosen[t] = jj
                _push(t, j, cid, cp, cn, hc, par, cdep, lvl, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr,
                      lval, curq, curid, Lt)
                C = _cells_now(s, ncell, cp, cn, par, cdep, lvl, featd, lptr, lval, Z, zp, zn)
                _pop(t, j, cid, cp, cn, hc, par, ncell, cptr, crow, Lt)
                lbs, ub2, improved, complete = _leaf_solve(Z, C, s, zp, zn, thT, t, Hl, ub, thr, t_end, wS, stats,
                                                           _zok(featd, s, zokb))
                if lbs < lbmin:
                    lbmin = lbs
                if improved:
                    ub = ub2
                    best_w[:] = 0
                    for q in range(s):
                        best_w[featd[q]] = wS[q]
                if not complete:
                    return lbmin, ub, False
            t -= 1
            if t < 0:
                break
            _pop(t, featd[t], cid, cp, cn, hc, par, ncell, cptr, crow, Lt)
            continue
        jj = nxt[t]
        if jj > nord[t] - (s - t):
            t -= 1
            if t < 0:
                break
            _pop(t, featd[t], cid, cp, cn, hc, par, ncell, cptr, crow, Lt)
            continue
        # family bound (P9): every leaf below the remaining children of this node uses only the columns
        # featd[:t] + order[jj:], so LR over those columns bounds all of them; the sets shrink with jj,
        # so a bound >= ub - thr discards every remaining child of the node.
        if famc > 0:
            ncall = stats[5]
            lbf = _fam_try(t, jj, nord[t], s, nu, featd, ordl[t], fcols, Xu, pos_u, neg_u, thfam[t], ub, thr, fst,
                           stats, depm, isdep, inF)
            if lbf >= ub - thr:
                if lbf < lbmin:
                    lbmin = lbf
                nxt[t] = nord[t]
                continue
            if LRORDER and stats[5] > ncall:
                # P13 (see support_dfs_bits)
                nr = nord[t] - jj
                for q in range(nr):
                    g_ = ordl[t, jj + q]
                    okey[q] = -abs(thfam[t, g_]) * sdc[g_]
                oq = np.argsort(okey[:nr], kind="mergesort")
                for q in range(nr):
                    lst_[q] = ordl[t, jj + oq[q]]
                for q in range(nr):
                    ordl[t, jj + q] = lst_[q]
        nxt[t] = jj + 1
        chosen[t] = jj
        featd[t] = ordl[t, jj]
        _push(t, ordl[t, jj], cid, cp, cn, hc, par, cdep, lvl, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr, lval,
              curq, curid, Lt)
        nxt[t + 1] = 0
        nord[t + 1] = nord[t] - jj - 1
        for q in range(nord[t + 1]):
            ordl[t + 1, q] = ordl[t, jj + 1 + q]
        for q in range(d + 1):
            thfam[t + 1, q] = thfam[t, q]
        t += 1
    return lbmin, ub, True


@nb.njit(cache=False)
def _global_deps(Xu, order):
    """Exact linear dependencies among [1, columns in `order`] on all unique rows (Gram-Schmidt in that
    order, as reduce_cols_big; every dependency verified exactly with small integer coefficients, P8).
    isdep[j]: column j is an exact combination of the intercept and earlier independent columns;
    depm[j, a]: column a has a nonzero coefficient in that combination."""
    n, d = Xu.shape
    p = d + 1
    U = np.empty((n, p))
    for i in range(n):
        U[i, 0] = 1.0
        for q in range(d):
            U[i, q + 1] = Xu[i, order[q]]
    depm = np.zeros((d, d), np.bool_)
    isdep = np.zeros(d, np.bool_)
    for i in range(n):
        for j in range(p):
            v = U[i, j]
            if v != np.floor(v) or abs(v) > 1048576.0:
                return depm, isdep
    Q = np.zeros((n, p))
    Rm = np.zeros((p, p))
    J = np.zeros(p, np.int64)
    nJ = 0
    v = np.empty(n)
    for j in range(p):
        nrm0 = 0.0
        for i in range(n):
            v[i] = U[i, j]
            nrm0 += v[i] * v[i]
        if nrm0 == 0.0:
            continue
        coef = np.zeros(nJ)
        for rep in range(2):
            for a_ in range(nJ):
                sm = 0.0
                for i in range(n):
                    sm += Q[i, a_] * v[i]
                coef[a_] += sm
                for i in range(n):
                    v[i] -= sm * Q[i, a_]
        nrm = 0.0
        for i in range(n):
            nrm += v[i] * v[i]
        dep = False
        if nJ > 0 and nrm <= 1e-18 * nrm0:
            lam = np.zeros(nJ)
            for a_ in range(nJ - 1, -1, -1):
                sm = coef[a_]
                for b_ in range(a_ + 1, nJ):
                    sm -= Rm[a_, b_] * lam[b_]
                lam[a_] = sm / Rm[a_, a_]
            for qq in range(1, 65):
                ok = True
                for a_ in range(nJ):
                    tq = qq * lam[a_]
                    if abs(tq - np.round(tq)) > 1e-6 or abs(tq) > 1048576.0:
                        ok = False
                        break
                if not ok:
                    continue
                for i in range(n):
                    acc = 0.0
                    for a_ in range(nJ):
                        acc += np.round(qq * lam[a_]) * U[i, J[a_]]
                    if acc != qq * U[i, j]:
                        ok = False
                        break
                if ok and j > 0:
                    dep = True
                    jc = order[j - 1]
                    isdep[jc] = True
                    for a_ in range(nJ):
                        if np.round(qq * lam[a_]) != 0.0 and J[a_] > 0:
                            depm[jc, order[J[a_] - 1]] = True
                break
        if dep:
            continue
        nv = np.sqrt(nrm)
        if nv == 0.0:
            continue
        for a_ in range(nJ):
            Rm[a_, nJ] = coef[a_]
        Rm[nJ, nJ] = nv
        for i in range(n):
            Q[i, nJ] = v[i] / nv
        J[nJ] = j
        nJ += 1
    return depm, isdep


def certify(X, y01, k, w_inc, ub, deadline):
    """Prove a lower bound on the minimum calibrated loss over every feasible point vector (P6).
    Returns (lower_bound, w_best, ub_best) in mean-loss units."""
    y = np.asarray(y01, float)
    N = float(len(y))
    t_start = time.perf_counter()
    cols = np.flatnonzero(np.ptp(X, axis=0) > 0)
    d = len(cols)
    if d == 0:
        return ub - PRUNE_TOL, w_inc, ub
    if ub <= 5e-8:
        return 0.0, w_inc, ub     # a loss is never negative, and 0 is within the tolerance of ub
    Xc = X[:, cols]
    rows, inv = np.unique(Xc, axis=0, return_inverse=True)
    inv = inv.ravel()
    pos_u = np.bincount(inv, weights=y, minlength=len(rows)).astype(np.float64)
    neg_u = np.bincount(inv, weights=1 - y, minlength=len(rows)).astype(np.float64)
    nu = len(rows)
    cptr = np.zeros(d + 1, np.int64)
    crow_l, ccode_l, lval_l = [], [], []
    lptr = np.zeros(d + 1, np.int64)
    Hone = np.zeros(d)
    for j in range(d):
        u, code = np.unique(rows[:, j], return_inverse=True)
        code = code.ravel()
        wcnt = np.bincount(code, weights=pos_u + neg_u, minlength=len(u))
        base = int(np.argmax(wcnt))
        # relabel: base level -> 0, others 1.. in increasing value order
        rel = np.empty(len(u), np.int64)
        rel[base] = 0
        others = [q for q in range(len(u)) if q != base]
        for r_, q in enumerate(others):
            rel[q] = r_ + 1
        code2 = rel[code]
        lv = np.empty(len(u))
        lv[0] = u[base]
        for r_, q in enumerate(others):
            lv[r_ + 1] = u[q]
        nzr = np.flatnonzero(code2 != 0)
        o = np.argsort(code2[nzr], kind="stable")
        crow_l.append(nzr[o])
        ccode_l.append(code2[nzr][o])
        lval_l.append(lv)
        cptr[j + 1] = cptr[j] + len(nzr)
        lptr[j + 1] = lptr[j] + len(u)
        pp = np.bincount(code2, weights=pos_u, minlength=len(u))
        nn_ = np.bincount(code2, weights=neg_u, minlength=len(u))
        Hone[j] = sum(_h2(pp[q], nn_[q]) for q in range(len(u)))
    crow = np.concatenate(crow_l).astype(np.int64) if cptr[-1] else np.zeros(0, np.int64)
    ccode = np.concatenate(ccode_l).astype(np.int64) if cptr[-1] else np.zeros(0, np.int64)
    lval = np.concatenate(lval_l)
    order = np.argsort(Hone, kind="stable").astype(np.int64)
    s = min(k, d)
    best_w = np.zeros(d, np.int64)
    stats = np.zeros(16, np.int64)
    t_end = time.monotonic() + max(0.0, deadline - time.perf_counter())
    Nint = int(round(N))
    mm = np.arange(Nint + 1, dtype=np.float64)
    Lt = mm * np.log(np.maximum(mm, 1.0))
    # table-based saturated sums: each entry within 1 ulp, each cell term within 4 eps Lt[N], at most
    # 2 (s + 1) (n_u + 1) + 1 terms and additions along a DFS path (notes, x3)
    hmarg = 12.0 * (s + 1) * (nu + 1) * EPS * Lt[-1] + 1e-12 * Lt[-1]
    Xu = np.ascontiguousarray(rows.astype(np.float64))
    allbin = bool(np.all(np.diff(lptr) == 2))
    avgnnz = cptr[-1] / max(d, 1)
    Wp = (int(round(y.sum())) + 63) // 64
    Wn = (int(round(N - y.sum())) + 63) // 64
    use_bits = allbin and s <= 12 and (s + 1) * (2 ** s) * (Wp + Wn) * 8 <= 2e8
    wts = pos_u + neg_u
    mu_c = (wts @ Xu) / N
    sdc = np.sqrt(np.maximum((wts @ (Xu * Xu)) / N - mu_c * mu_c, 0.0))
    depm, isdep = _global_deps(Xu, order)
    if use_bits:
        # raw rows bit-packed per column: bit set iff the value is the larger of the two (v1)
        v0 = np.array([lval_l[j].min() for j in range(d)])
        v1 = np.array([lval_l[j].max() for j in range(d)])
        # rows sorted lexicographically by the columns in `order` (strongest first) so that cells of
        # those columns occupy few words; any row order gives the same counts
        Bo = (Xc[:, order] == v1[order][None, :])
        ro = np.lexsort(Bo.T[::-1])
        Xs, ys = Xc[ro], y[ro]
        Xp = Xs[ys > 0]
        Xn = Xs[ys <= 0]
        Xb = np.zeros((d, Wp + Wn), np.uint64)
        for part, off in ((Xp, 0), (Xn, Wp)):
            m = len(part)
            B = (part == v1[None, :])
            pad = (-m) % 64
            Bp = np.concatenate([B, np.zeros((pad, d), bool)]) if pad else B
            packed = np.ascontiguousarray(np.packbits(np.ascontiguousarray(Bp.T), axis=1, bitorder="little"))
            Xb[:, off:off + (m + pad) // 64] = packed.view("<u8")
        lbmin, ub_u, complete = support_dfs_bits(s, Xb, Wp, v0, v1, order, ub * N, PRUNE_TOL * N, best_w, t_end,
                                                 stats, Lt, hmarg, Xu, pos_u, neg_u, FAMC,
                                                 2 ** (s - 1) * (Wp + Wn) * 0.3e-9 + 1e-7, sdc, depm, isdep)
    else:
        lbmin, ub_u, complete = support_dfs(s, pos_u, neg_u, cptr, crow, ccode, lptr, lval, order, ub * N,
                                            PRUNE_TOL * N, best_w, t_end, stats, Lt, hmarg, Xu, FAMC,
                                            avgnnz * 5e-9 + 2e-7, sdc, depm, isdep)
    stats[7] = int(use_bits)
    w_out = w_inc
    if ub_u < ub * N:
        w_out = np.zeros(X.shape[1])
        w_out[cols] = best_w
    ub_new = min(ub, ub_u / N)
    certify.stats = stats
    if not complete:
        return 0.0, w_out, ub_new   # a loss is never negative
    return min(lbmin / N, ub_new), w_out, ub_new


def make_model(k, time_limit):
    return CertifyingClassifier(k=k, time_limit=time_limit)


class CertifyingClassifier(SparseIntegerClassifier):
    """The heuristic for the incumbent, then ``certify`` with the time that is left."""

    def fit(self, X, y):
        t0 = time.perf_counter()
        super().fit(X, y)
        X = np.asarray(X, dtype=float)
        lb, w, l = certify(X, y, self.k, self.coef_.astype(float), float(self.train_loss_),
                           t0 + 0.9 * self.time_limit)
        self.coef_ = np.clip(np.round(w), -COEF_BOUND, COEF_BOUND)
        self.train_loss_ = l
        self.lower_bound_ = lb
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
    CertifyingClassifier(k=3, time_limit=5.0).fit(X[:, [0, 4, 6, 20, 21, 22, 1, 2]], y)


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
