"""Fine track: sparse integer risk scores when every numeric column is split at its 99 percentiles
(up to ~4,000 nested threshold features). The file the agent edits; starts as v35_scratch.

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
import time

import numpy as np
import numba as nb

MODEL_NAME = "f20_dpmargin"
DESCRIPTION = ('f19 + DP moves within 1% above the current loss at fixed (a, b) are also checked exactly (best 4 by the fixed-(a, b) loss, own quota), since re-calibration can make them improve')
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


@nb.njit(cache=False)
def sparse_score(w, n, ptr, idx, xval):
    """s = X w from the CSC nonzeros of the support columns only."""
    s = np.zeros(n)
    for j in range(w.shape[0]):
        if w[j] != 0.0:
            for t in range(ptr[j], ptr[j + 1]):
                s[idx[t]] += w[j] * xval[t]
    return s


@nb.njit(cache=False)
def transpose(X):
    """Blocked transpose (about twice as fast as numpy's strided copy for large matrices)."""
    n, d = X.shape
    XT = np.empty((d, n))
    B = 64
    for i0 in range(0, n, B):
        i1 = min(n, i0 + B)
        for j0 in range(0, d, B):
            j1 = min(d, j0 + B)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    XT[j, i] = X[i, j]
    return XT


class Data:
    def score(self, w):
        return sparse_score(np.asarray(w, dtype=np.float64), self.n, self.ptr, self.idx, self.xval)

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
        if len(first) == X.shape[0]:  # no duplicate rows: keep the matrix as it is
            self.X = np.ascontiguousarray(X)
            self.y = np.ascontiguousarray(ys)
            self.c = np.ones(X.shape[0])
        else:
            self.X = np.ascontiguousarray(X[first])
            self.y = np.ascontiguousarray(ys[first])
            self.c = cnt.astype(np.float64)
        self.n, self.d = self.X.shape
        self.N = float(self.c.sum())
        self.XT = transpose(self.X)
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
        (self.chain, self.cpos, self.dup, self.cptr, self.ccol, self.rptr, self.rrows,
         self.rrank) = build_chains(self.XT, self.isbin, self.valid, self.ptr, self.idx)
        self.inchain = self.chain >= 0
        self.chain_cols = np.flatnonzero(self.inchain)
        self.other_cols = np.flatnonzero(self.valid & ~self.inchain & ~self.dup)


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
        st = ScoreState(D, D.score(w))
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


# ------------------------------------------ nested threshold chains (fine track)
@nb.njit(cache=False)
def build_chains(XT, isbin, valid, ptr, idx):
    """Group binary columns into chains of nested sets (col_1 ⊇ col_2 ⊇ ...), found from the data: columns are
    taken by decreasing count and appended to the chain whose last column is the tightest superset. Identical
    columns are marked as duplicates. Returns per column: chain id (-1: not in a chain), position (1-based),
    dup flag; per chain: the column list (cptr, ccol) and the rows of its first column with their rank (the
    deepest chain column containing the row): (rptr, rrows, rrank)."""
    d, n = XT.shape
    W = (n + 63) // 64
    cnt = np.zeros(d, np.int64)
    first = np.full(d, -1, np.int64)
    B = np.zeros((d, W), np.uint64)
    use = np.zeros(d, np.bool_)
    for j in range(d):
        if isbin[j] and valid[j]:
            use[j] = True
            for t in range(ptr[j], ptr[j + 1]):
                i = idx[t]
                B[j, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
                if first[j] < 0:
                    first[j] = i
            cnt[j] = ptr[j + 1] - ptr[j]
    order = np.argsort(-cnt, kind="mergesort")
    end = np.empty(d, np.int64)
    head = np.empty(d, np.int64)
    nxt = np.full(d, -1, np.int64)
    clen = np.zeros(d, np.int64)
    chain = np.full(d, -1, np.int64)
    pos = np.zeros(d, np.int64)
    dup = np.zeros(d, np.bool_)
    nch = 0
    for oj in range(d):
        j = order[oj]
        if not use[j]:
            continue
        f = first[j]
        fw = f >> 6
        fb = np.uint64(1) << np.uint64(f & 63)
        best = -1
        bestc = n + 1
        for h in range(nch):
            e = end[h]
            if cnt[e] < cnt[j] or cnt[e] >= bestc:
                continue
            if (B[e, fw] & fb) == 0:
                continue
            ok = True
            for w in range(W):
                if (B[j, w] & ~B[e, w]) != 0:
                    ok = False
                    break
            if ok:
                best = h
                bestc = cnt[e]
        if best >= 0 and bestc == cnt[j]:
            dup[j] = True
            continue
        if best < 0:
            best = nch
            head[nch] = j
            nch += 1
        else:
            nxt[end[best]] = j
        end[best] = j
        clen[best] += 1
        chain[j] = best
        pos[j] = clen[best]
    cptr = np.zeros(nch + 1, np.int64)
    for h in range(nch):
        cptr[h + 1] = cptr[h] + clen[h]
    ccol = np.empty(cptr[nch], np.int64)
    rptr = np.zeros(nch + 1, np.int64)
    for h in range(nch):
        j = head[h]
        u = cptr[h]
        while j >= 0:
            ccol[u] = j
            u += 1
            j = nxt[j]
        rptr[h + 1] = rptr[h] + cnt[head[h]]
    rrows = np.empty(rptr[nch], np.int64)
    rrank = np.empty(rptr[nch], np.int64)
    tmp = np.zeros(n, np.int64)
    for h in range(nch):
        for u in range(cptr[h], cptr[h + 1]):
            j = ccol[u]
            for t in range(ptr[j], ptr[j + 1]):
                tmp[idx[t]] = u - cptr[h] + 1
        j = head[h]
        v = rptr[h]
        for t in range(ptr[j], ptr[j + 1]):
            i = idx[t]
            rrows[v] = i
            rrank[v] = tmp[i]
            tmp[i] = 0
            v += 1
    return chain, pos, dup, cptr, ccol, rptr, rrows, rrank


@nb.njit(cache=False)
def chain_counts(inv, nbin, y, c, cptr, ccol, rptr, rrows, rrank, d):
    """Weight of y = +1 (Cp) and y = -1 (Cn) rows per (column, score bin) for every chain column, by one pass
    over each chain's rows (bucketed by rank) and a suffix sum over the ranks."""
    Cp = np.zeros((d, nbin), np.float32)
    Cn = np.zeros((d, nbin), np.float32)
    nch = cptr.shape[0] - 1
    for h in range(nch):
        m = cptr[h + 1] - cptr[h]
        ap = np.zeros((m + 1, nbin))
        an = np.zeros((m + 1, nbin))
        for t in range(rptr[h], rptr[h + 1]):
            i = rrows[t]
            if y[i] > 0:
                ap[rrank[t], inv[i]] += c[i]
            else:
                an[rrank[t], inv[i]] += c[i]
        for r in range(m - 1, 0, -1):
            for q in range(nbin):
                ap[r, q] += ap[r + 1, q]
                an[r, q] += an[r + 1, q]
        for r in range(1, m + 1):
            j = ccol[cptr[h] + r - 1]
            for q in range(nbin):
                Cp[j, q] = ap[r, q]
                Cn[j, q] = an[r, q]
    return Cp, Cn


@nb.njit(cache=False)
def move_tables(sv, a, b, deltas):
    """Per (bin, delta) change of loss / gradient / Hessian terms per unit weight of y = +1 and y = -1 rows
    when a row's score moves from sv[bin] to sv[bin] + delta, at fixed (a, b)."""
    nb_ = sv.shape[0]
    nd = deltas.shape[0]
    TP = np.empty((nb_, 6 * nd), np.float32)
    TN = np.empty((nb_, 6 * nd), np.float32)
    for q in range(nb_):
        so = sv[q]
        z = a * so + b
        lpo = _lrow(z)
        lno = _lrow(-z)
        ppo = _sig(-z)
        pno = 1.0 - ppo
        wo = ppo * pno
        for r in range(nd):
            s1 = so + deltas[r]
            z1 = a * s1 + b
            lp1 = _lrow(z1)
            ln1 = _lrow(-z1)
            pp1 = _sig(-z1)
            pn1 = 1.0 - pp1
            w1 = pp1 * pn1
            TP[q, r] = lp1 - lpo
            TN[q, r] = ln1 - lno
            TP[q, nd + r] = -pp1 * s1 + ppo * so
            TN[q, nd + r] = pn1 * s1 - pno * so
            TP[q, 2 * nd + r] = -pp1 + ppo
            TN[q, 2 * nd + r] = pn1 - pno
            hA = w1 * s1 * s1 - wo * so * so
            hB = w1 * s1 - wo * so
            hC = w1 - wo
            TP[q, 3 * nd + r] = hA
            TN[q, 3 * nd + r] = hA
            TP[q, 4 * nd + r] = hB
            TN[q, 4 * nd + r] = hB
            TP[q, 5 * nd + r] = hC
            TN[q, 5 * nd + r] = hC
    return TP, TN


@nb.njit(cache=False)
def newton_estimate(E, T, a, b, nd, tr_a, tr_b):
    """E: (m, 6 nd) summed changes; T: totals at the current (a, b). Per move: the exact loss at fixed (a, b)
    (F), the quadratic-model loss after a trust-region Newton step in (a, b) (Q) and that step (SA, SB)."""
    m = E.shape[0]
    F = np.empty((m, nd))
    Q = np.empty((m, nd))
    SA = np.empty((m, nd))
    SB = np.empty((m, nd))
    Q2 = np.empty((m, nd))
    for j in range(m):
        for r in range(nd):
            f = T[0] + E[j, r]
            ga = T[1] + E[j, nd + r]
            gb = T[2] + E[j, 2 * nd + r]
            haa = T[3] + E[j, 3 * nd + r] + 1e-12
            hab = T[4] + E[j, 4 * nd + r]
            hbb = T[5] + E[j, 5 * nd + r] + 1e-12
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
            sa = -t * da
            sb = -t * db
            F[j, r] = f
            Q[j, r] = f + ga * sa + gb * sb + 0.5 * (haa * sa * sa + 2 * hab * sa * sb + hbb * sb * sb)
            SA[j, r] = sa
            SB[j, r] = sb
            # the same model in a small trust region (fewer false positives among large moves)
            t = 1.0
            if abs(da) * t > 0.25 * tr_a * abs(a) + 1e-300:
                t = 0.25 * tr_a * abs(a) / abs(da)
            if abs(db) * t > 0.25 * tr_b:
                t = 0.25 * tr_b / abs(db)
            sa = -t * da
            sb = -t * db
            Q2[j, r] = f + ga * sa + gb * sb + 0.5 * (haa * sa * sa + 2 * hab * sa * sb + hbb * sb * sb)
    return F, Q, SA, SB, Q2


@nb.njit(cache=False)
def newton_point_loss(sel, nd, F, SA, SB, a, b, Cp, Cn, cols, sv, Wp, Wn, deltas, out):
    """out[f] = min(F, exact histogram loss at the Newton point) for the selected flat moves f."""
    nbn = sv.shape[0]
    for u in range(sel.shape[0]):
        f = sel[u]
        j = f // nd
        r = f - j * nd
        cj = cols[j]
        a2 = a + SA[j, r]
        b2 = b + SB[j, r]
        dl = deltas[r]
        tot = 0.0
        for q in range(nbn):
            cp = Cp[cj, q]
            cn = Cn[cj, q]
            z = a2 * sv[q] + b2
            tot += (Wp[q] - cp) * _lrow(z) + (Wn[q] - cn) * _lrow(-z)
            if cp + cn > 0.0:
                z = z + a2 * dl
                tot += cp * _lrow(z) + cn * _lrow(-z)
        out[j, r] = min(F[j, r], tot)


@nb.njit(cache=False)
def exact_move(sv, Wp, Wn, cp, cn, delta, a, b):
    """Exact calibrated loss after moving the weights (cp, cn) of every bin from sv to sv + delta."""
    m = sv.shape[0]
    s2 = np.empty(4 * m)
    y2 = np.empty(4 * m)
    c2 = np.empty(4 * m)
    for q in range(m):
        s2[q] = sv[q]
        y2[q] = 1.0
        c2[q] = Wp[q] - cp[q]
        s2[m + q] = sv[q]
        y2[m + q] = -1.0
        c2[m + q] = Wn[q] - cn[q]
        s2[2 * m + q] = sv[q] + delta
        y2[2 * m + q] = 1.0
        c2[2 * m + q] = cp[q]
        s2[3 * m + q] = sv[q] + delta
        y2[3 * m + q] = -1.0
        c2[3 * m + q] = cn[q]
    return calibrate(s2, y2, c2, a, b)


@nb.njit(cache=False)
def chain_dp(h, inv, sv, y, c, cptr, rptr, rrows, rrank, a, b, Lmax, C, bound):
    """Best step function on chain h at fixed (a, b), given the rest of the score (row bins inv, bin scores sv):
    the level of a row of rank r is the sum of the points of the chosen chain columns at positions <= r.
    Returns for every number of steps c = 0..C the loss of the chain's rows (inf if none) and the points of
    the chain columns (C + 1, m)."""
    m = cptr[h + 1] - cptr[h]
    nbn = sv.shape[0]
    nlev = 2 * Lmax + 1
    Bp = np.zeros((m + 1, nbn))
    Bn = np.zeros((m + 1, nbn))
    for t in range(rptr[h], rptr[h + 1]):
        i = rrows[t]
        if y[i] > 0:
            Bp[rrank[t], inv[i]] += c[i]
        else:
            Bn[rrank[t], inv[i]] += c[i]
    LP = np.empty((nbn, nlev))
    LN = np.empty((nbn, nlev))
    for q in range(nbn):
        for l in range(nlev):
            z = a * (sv[q] + l - Lmax) + b
            LP[q, l] = _lrow(z)
            LN[q, l] = _lrow(-z)
    cost = Bp @ LP + Bn @ LN  # (m + 1, nlev)
    INF = np.inf
    dp = np.full((nlev, C + 1), INF)
    dp[Lmax, 0] = 0.0
    # choice[r, l, c]: value of the step placed at position r (0: none)
    choice = np.zeros((m + 1, nlev, C + 1), np.int8)
    nd = np.empty((nlev, C + 1))
    for r in range(1, m + 1):
        for l in range(nlev):
            for cc in range(C + 1):
                best = dp[l, cc]
                bv = 0
                if cc > 0:
                    for v in range(-bound, bound + 1):
                        if v == 0:
                            continue
                        l0 = l - v
                        if l0 < 0 or l0 >= nlev:
                            continue
                        val = dp[l0, cc - 1]
                        if val < best:
                            best = val
                            bv = v
                nd[l, cc] = best + cost[r, l] if best < INF else INF
                choice[r, l, cc] = bv
        for l in range(nlev):
            for cc in range(C + 1):
                dp[l, cc] = nd[l, cc]
    out = np.full(C + 1, INF)
    pts = np.zeros((C + 1, m))
    for cc in range(C + 1):
        bl = -1
        bestv = INF
        for l in range(nlev):
            if dp[l, cc] < bestv:
                bestv = dp[l, cc]
                bl = l
        if bl < 0:
            continue
        out[cc] = bestv
        l = bl
        k2 = cc
        for r in range(m, 0, -1):
            v = choice[r, l, k2]
            if v != 0:
                pts[cc, r - 1] = v
                l -= v
                k2 -= 1
    base = 0.0
    for r in range(1, m + 1):
        base += cost[r, Lmax]
    return out, pts, base


@nb.njit(cache=False)
def chain_shift(s, h, pts, cptr, rptr, rrows, rrank):
    """s + the chain's step function with points pts (in chain order): rows of rank r get the sum of pts[:r]."""
    m = cptr[h + 1] - cptr[h]
    lev = np.zeros(m + 1)
    for r in range(1, m + 1):
        lev[r] = lev[r - 1] + pts[r - 1]
    out = s.copy()
    for t in range(rptr[h], rptr[h + 1]):
        out[rrows[t]] += lev[rrank[t]]
    return out


class ChainILS:
    """Best-improvement local search over integer points on binary chain columns. Every move 'w_j += delta'
    (value change, addition, removal) of every column, and every swap (remove one support column, add any
    column with any value; threshold slides included) is scored at once from per-(column, score bin) weights
    computed by cumulative passes over the chains; the best estimated moves are then checked exactly on the
    score histogram."""

    def __init__(self, D, k, n_exact=8, dp_cmax=4, lmax=15, n_newton=64):
        self.D, self.k, self.n_exact, self.n_newton = D, k, n_exact, n_newton
        self.dp_cmax, self.lmax = dp_cmax, lmax
        self.lazy_swaps = True
        self.dp_margin, self.n_dp_check = 0.01, 4
        self.deltas_all = np.array([v for v in range(-2 * COEF_BOUND, 2 * COEF_BOUND + 1) if v != 0], dtype=np.float64)
        self.nzv = np.array([v for v in range(-COEF_BOUND, COEF_BOUND + 1) if v != 0], dtype=np.float64)
        self.visited = set()
        self.nevals = 0

    def _counts(self, s):
        D = self.D
        inv, sv, Wp, Wn = bin_scores(s, D.y, D.c)
        Cp, Cn = chain_counts(inv, sv.shape[0], D.y, D.c, D.cptr, D.ccol, D.rptr, D.rrows, D.rrank, D.d)
        return inv, sv, Wp, Wn, Cp, Cn

    def _estimates(self, sv, Wp, Wn, Cp, Cn, a, b, deltas, cols, Lcut, bad=None):
        TP, TN = move_tables(sv, a, b, deltas)
        nd3 = 3 * deltas.shape[0]
        # float32 BLAS (weights are integer counts; the estimate only ranks moves, the best are checked exactly);
        # the Hessian terms do not depend on y, so they use the total weight
        P, N = Cp[cols], Cn[cols]
        E = np.empty((len(cols), 2 * nd3))
        E[:, :nd3] = P @ TP[:, :nd3] + N @ TN[:, :nd3]
        E[:, nd3:] = (P + N) @ TP[:, nd3:]
        T = bin_stats(sv, Wp, Wn, a, b, *(np.empty(sv.shape[0]) for _ in range(5)))
        nd = deltas.shape[0]
        F, Q, SA, SB, Q2 = newton_estimate(E, T, a, b, nd, TR_A, TR_B)
        if bad is not None:
            F[bad] = np.inf
            Q[bad] = np.inf
            Q2[bad] = np.inf
        # the exact loss at the Newton point only for the most promising moves by the quadratic model (half by
        # the large trust region, half by the small one)
        est = F.copy()
        sel = []
        for qf in (Q.ravel(), Q2.ravel()):
            m = min(self.n_newton // 2, qf.size)
            sq = np.argpartition(qf, m - 1)[:m]
            sel.append(sq[qf[sq] < Lcut])
        sel = np.unique(np.concatenate(sel))
        newton_point_loss(sel.astype(np.int64), nd, F, SA, SB, a, b, Cp, Cn, cols, sv, Wp, Wn, deltas, est)
        return est

    def _check(self, cands, w, s, L, a, b, Lbar=None, tabu=None):
        """Exact calibrated loss of the best estimated candidates. Returns the best one below Lbar (default: an
        improvement on L) or None; candidates touching a tabu column are skipped unless they beat L."""
        D = self.D
        if Lbar is None:
            Lbar = L - 1e-9 * L
        cands.sort(key=lambda t: t[0])
        best = None
        n = 0
        ndp = 0
        for e, rj, aj, v, ctx in cands:
            if rj == -2:
                if ndp >= self.n_dp_check:
                    continue
                ndp += 1
            elif n >= 2 * self.n_exact:
                continue
            if tabu and any(c in tabu for c in self._touched(w, rj, aj, v)):
                if e >= L:
                    continue
                aspir = True
            else:
                aspir = False
            if rj != -2:
                n += 1
            self.nevals += 1
            if rj == -2:
                cols_h = D.ccol[D.cptr[aj]:D.cptr[aj + 1]]
                s2 = chain_shift(s, aj, v - w[cols_h], D.cptr, D.rptr, D.rrows, D.rrank)
                st2 = ScoreState(D, s2, a, b)
                L2, a2, b2 = st2.L, st2.a, st2.b
            else:
                inv0, sv0, Wp0, Wn0, Cp0, Cn0, s2 = ctx
                delta = v if rj >= 0 else v - w[aj]  # w[aj] == 0 in a swap
                L2, a2, b2 = exact_move(sv0, Wp0, Wn0, Cp0[aj], Cn0[aj], delta, a, b)
            if aspir and L2 >= L - 1e-9 * L:
                continue
            if L2 < Lbar and (best is None or L2 < best[0]):
                best = (L2, a2, b2, rj, aj, v, s2)
        return best

    def _touched(self, w, rj, aj, v):
        D = self.D
        if rj == -2:
            cols_h = D.ccol[D.cptr[aj]:D.cptr[aj + 1]]
            return [int(c) for c in cols_h[w[cols_h] != v]]
        return [int(aj)] if rj < 0 else [int(rj), int(aj)]

    def _basic(self, w, s, st, L, a, b, Lcut):
        """Candidates: w_j += delta for every chain column, and the best step function of every chain (DP)."""
        D, k = self.D, self.k
        inv, sv, Wp, Wn, Cp, Cn = st
        S = np.flatnonzero(w)
        full = len(S) >= k
        cand_cols = D.chain_cols
        dl = self.deltas_all
        cands = []
        wc = w[cand_cols]
        newv = wc[:, None] + dl[None, :]
        bad = (np.abs(newv) > COEF_BOUND) | ((wc == 0)[:, None] & full)
        est = self._estimates(sv, Wp, Wn, Cp, Cn, a, b, dl, cand_cols, L, bad)
        flat = est.ravel()
        m = min(self.n_exact, flat.size)
        for f in np.argpartition(flat, m - 1)[:m]:
            if np.isfinite(flat[f]) and flat[f] < Lcut:
                q, r = divmod(int(f), dl.shape[0])
                j = cand_cols[q]
                cands.append((flat[f], -1, j, w[j] + dl[r], (inv, sv, Wp, Wn, Cp, Cn, s)))
        # block moves: the best step function of each chain given the rest (thresholds slid, split, merged)
        for h in range(len(D.cptr) - 1):
            cols_h = D.ccol[D.cptr[h]:D.cptr[h + 1]]
            wh = w[cols_h]
            nh = int(np.count_nonzero(wh))
            budget = min(k - (len(S) - nh), self.dp_cmax)
            if budget <= 0:
                continue
            s_rest = chain_shift(s, h, -wh, D.cptr, D.rptr, D.rrows, D.rrank) if nh else s
            invr, svr, Wpr, Wnr = bin_scores(s_rest, D.y, D.c)
            out, pts, base = chain_dp(h, invr, svr, D.y, D.c, D.cptr, D.rptr, D.rrows, D.rrank, a, b,
                                      self.lmax, budget, COEF_BOUND)
            T = bin_stats(svr, Wpr, Wnr, a, b, *(np.empty(svr.shape[0]) for _ in range(5)))
            for cc in range(budget + 1):
                if (cc == 0 and nh == 0) or not np.isfinite(out[cc]):
                    continue
                tot = T[0] - base + out[cc]
                # DP moves are scored at fixed (a, b): keep near misses, re-calibration may still make them improve
                if tot < Lcut + self.dp_margin * L and not np.array_equal(pts[cc], wh):
                    cands.append((tot, -2, h, pts[cc].copy(), None))
        return cands

    def _swaps(self, w, s, L, a, b, Lcut):
        """Candidates: remove a support column, add any other column with any value."""
        D = self.D
        cand_cols = D.chain_cols
        nzv = self.nzv
        cands = []
        for rj in np.flatnonzero(w):
            if not D.inchain[rj]:
                continue
            s2 = s - w[rj] * D.XT[rj]
            if np.ptp(s2) == 0:
                continue
            inv2, sv2, Wp2, Wn2, Cp2, Cn2 = self._counts(s2)
            bad2 = np.broadcast_to((w[cand_cols] != 0)[:, None], (len(cand_cols), nzv.shape[0]))
            est2 = self._estimates(sv2, Wp2, Wn2, Cp2, Cn2, a, b, nzv, cand_cols, L, bad2)
            flat = est2.ravel()
            m = min(self.n_exact, flat.size)
            for f in np.argpartition(flat, m - 1)[:m]:
                if np.isfinite(flat[f]) and flat[f] < Lcut:
                    q, r = divmod(int(f), nzv.shape[0])
                    cands.append((flat[f], rj, cand_cols[q], nzv[r], (inv2, sv2, Wp2, Wn2, Cp2, Cn2, s2)))
        return cands

    def _apply(self, w, best):
        D = self.D
        L, a, b, rj, aj, v, s0 = best
        if rj == -2:
            w[D.ccol[D.cptr[aj]:D.cptr[aj + 1]]] = v
            s = s0
        else:
            s = s0 + (v - (0.0 if rj >= 0 else w[aj])) * D.XT[aj]
            if rj >= 0:
                w[rj] = 0
            w[aj] = v
        st = self._counts(s)
        L, a, b = calibrate_bins(st[1], st[2], st[3], a, b)  # exact re-calibration on the new histogram
        return s, st, L, a, b

    def run(self, w, max_iter=200, deadline=np.inf, depth=0, tenure=3):
        """Best-improvement descent; at a local optimum, when depth > 0, a tabu walk of up to `depth` steps
        (best non-tabu move even if worse; changed columns are tabu for `tenure` steps) that resumes the
        descent as soon as it beats the best point of this run."""
        D = self.D
        w = w.astype(np.float64).copy()
        s = D.score(w)
        st = self._counts(s)
        inv, sv, Wp, Wn, Cp, Cn = st
        if np.ptp(s) == 0:
            npos = D.c[D.y > 0].sum()
            L, a, b = calibrate_bins(sv, Wp, Wn, 0.0, np.log(npos / (D.N - npos)))
            a = 0.5  # a reference scale for the first moves (any a fits a constant score)
        else:
            sd = s.std()
            L, a, b = calibrate_bins(sv / sd, Wp, Wn, 0.0, 0.0)
            a /= sd
        best_L, best_w = L, w.copy()
        walk = 0
        tabu = {}
        for it in range(max_iter):
            if time.perf_counter() > deadline:
                break
            if walk == 0:
                key = w.tobytes()
                if key in self.visited:
                    break
                self.visited.add(key)
                best = self._check(self._basic(w, s, st, L, a, b, L * (1 - 1e-9)), w, s, L, a, b)
                if best is None:
                    best = self._check(self._swaps(w, s, L, a, b, L * (1 - 1e-9)), w, s, L, a, b)
                if best is None:
                    if depth <= 0:
                        break
                    walk = 1
            if walk > 0:
                if walk > depth:
                    break
                tb = {c for c, t in tabu.items() if t > it}
                cands = self._basic(w, s, st, L, a, b, np.inf) + self._swaps(w, s, L, a, b, np.inf)
                best = self._check(cands, w, s, L, a, b, Lbar=np.inf, tabu=tb)
                if best is None:
                    break
                walk += 1
            for c in self._touched(w, best[3], best[4], best[5]):
                tabu[c] = it + 1 + tenure
            s, st, L, a, b = self._apply(w, best)
            if L < best_L - 1e-12 * best_L:
                best_L, best_w = L, w.copy()
                walk = 0
        return best_L / D.N, best_w


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, n_starts=5, n_kicks=200, patience=25):
        self.k, self.time_limit = k, time_limit
        self.n_kicks, self.patience = n_kicks, patience
        self.depth_start = 0
        self.depth_kick = 0
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
        ils = ChainILS(D, self.k) if len(D.chain_cols) else None
        ils_old = ILS(D, self.k) if len(D.other_cols) or ils is None else None
        starts.sort(key=lambda t: t[0])
        for l0, w0 in starts[:self.n_starts]:
            if time.perf_counter() > t0 + 0.8 * self.time_limit:
                break
            w = w0
            if ils is not None:
                l, w = ils.run(w, deadline=t0 + 0.9 * self.time_limit, depth=self.depth_start)
            if ils_old is not None:
                l, w = ils_old.run(w, deadline=t0 + 0.9 * self.time_limit)
            if l < best_l:
                best_l, best_w = l, w
        # perturbation: drop 1-2 random support columns of the incumbent and descend again
        if ils is not None:
            rng = np.random.default_rng(int(os.environ.get("SLIM_SEED", "0")))  # dev-only seed override
            fails = 0
            for _ in range(self.n_kicks):
                if fails >= self.patience or time.perf_counter() > t0 + 0.8 * self.time_limit:
                    break
                S = np.flatnonzero(best_w)
                if len(S) == 0:
                    break
                w = best_w.copy()
                w[rng.choice(S, size=min(len(S), int(rng.integers(1, 3))), replace=False)] = 0
                l, w = ils.run(w, deadline=t0 + 0.9 * self.time_limit, depth=self.depth_kick)
                if l < best_l - 1e-12:
                    best_l, best_w = l, w
                    fails = 0
                else:
                    fails += 1
        tm.append(time.perf_counter())
        self.timing_ = np.diff(tm)  # data, beam, rounding, ILS
        self.coef_ = np.clip(np.round(best_w), -COEF_BOUND, COEF_BOUND)
        self.intercept_, self.multiplier_ = 0.0, 1.0
        self.train_loss_ = best_l
        return self


def make_model(k, time_limit):
    return SparseIntegerClassifier(k=k, time_limit=time_limit)


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


_jit_warmup()


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
    summary = evaluate_solver(("file", os.path.abspath(__file__)), MODEL_NAME, suite="visible_fine",
                              datasets=datasets, ks=ks, time_limit=TIME_LIMIT, jobs=args.jobs, integer=INTEGER)
    if datasets is None and ks is None and not args.no_record:
        record(MODEL_NAME, DESCRIPTION, summary, results_dir=os.path.join(here, "results"), integer=INTEGER)
    elif not args.no_record:
        print("(partial suite: results not recorded)")
    print_summary(MODEL_NAME, summary)
    print(f"total_seconds: {time.time() - t0:.1f}s")
