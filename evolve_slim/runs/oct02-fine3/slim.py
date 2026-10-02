"""Fine track: sparse integer risk scores when every numeric column is split at its 99 percentiles
(up to ~2,400 nested threshold features). The file the agent edits.

Pipeline (see notes.md in the run folder for the history and the measured effect of each part):

0. data as variables: binary columns are grouped into chains of nested sets (found from the data by
   subset tests on bitsets), and each chain becomes one ordinal code per row (the number of chain columns
   that contain the row), so column j of the chain is x_j = [code >= level_j]. Every other column is its own
   variable with the rank codes of its values. All per-column statistics are computed from per-code
   histograms (suffix sums for chains), which costs O(n) per variable instead of O(n) per column;
1. unique (codes, y) rows with counts;
2. continuous beam search (FasterRisk's: grow the support one column at a time from the 10 best parents by
   the 10 largest gradients), where every node keeps its rows grouped by (y, x_support);
3. rounding: for each final support, round m * beta over a grid of scales and keep the rounding with the
   smallest calibrated loss (the harness's criterion: min over a, b of the log loss of a * score + b);
4. integer local search from the best roundings: value changes and additions, then swaps when those fail,
   scored on cells (score bin, x_j value) by an estimate of the calibrated loss; the best are checked exactly.

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
from numba.typed import Dict, List

MODEL_NAME = "h8_fastbeam"
DESCRIPTION = ("h7 with the beam proposals (per-variable picks, support hashes, signature cut) and the row regrouping "
               "of kept children in numba, the kick pair ranking in numba, and the rounding calibrated on distinct "
               "score values (same search up to rounding ties)")
INTEGER = True
COEF_BOUND = 5
TR_A, TR_B = 1.0, 2.0  # trust region of the (a, b) Newton step in the move estimate
FINAL_POOL = int(os.environ.get("FINAL_POOL", "10"))
PARENT = int(os.environ.get("PARENT", "10"))
CHILD = int(os.environ.get("CHILD", "10"))
SIGMAX = int(os.environ.get("SIGMAX", "1"))
NEXACT = int(os.environ.get("NEXACT", "2"))
NSCREEN = int(os.environ.get("NSCREEN", "4"))
NSTARTS = int(os.environ.get("NSTARTS", "5"))
SCREEN = float(os.environ.get("SCREEN", "1.5"))
MS_FRAC = float(os.environ.get("MS_FRAC", "0"))
MS_SQRT = float(os.environ.get("MS_SQRT", "1"))
MS_ABS = float(os.environ.get("MS_ABS", "0"))
MS_BAND = float(os.environ.get("MS_BAND", "1"))
MS_ENDM = float(os.environ.get("MS_ENDM", "1"))
MS_CHAIN = int(os.environ.get("MS_CHAIN", "0"))
KICKS = int(os.environ.get("KICKS", "2"))
BTOL = float(os.environ.get("BTOL", "1e-4"))
PAIRS = int(os.environ.get("PAIRS", "0"))
QK = float(os.environ.get("QK", "0.001"))
ADDSCREEN = int(os.environ.get("ADDSCREEN", "16"))
LASTSCREEN = float(os.environ.get("LASTSCREEN", "1.5"))
RAW_DENSE = 32  # a non-binary column with at most this many codes is histogrammed, otherwise handled per row


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


# ------------------------------------------------------------ data layout
@nb.njit(cache=False)
def scan_columns(X):
    """Per column: count of nonzeros, whether a value other than 0 / 1 occurs; bitsets (word, column) of the
    nonzeros."""
    n, d = X.shape
    nw = (n + 63) // 64
    B = np.zeros((nw, d), np.uint64)
    cnt = np.zeros(d, np.int64)
    bad = np.zeros(d, np.bool_)
    for i in range(n):
        sh = np.uint64(i & 63)
        row = X[i]
        Bw = B[i >> 6]
        for j in range(d):
            v = row[j]
            nz = v != 0.0
            Bw[j] |= np.uint64(nz) << sh
            cnt[j] += nz
            bad[j] |= nz & (v != 1.0)
    return ~bad, cnt, B


@nb.njit(cache=False)
def build_chains(B, cnt, order, d):
    """Greedy chain cover of the columns in `order` (by count descending): each column joins the chain whose
    last column contains it with the smallest count, else starts a new chain. Returns chain id and level
    (1-based position) of every column (-1 when not in `order`), and the number of chains."""
    nw = B.shape[0]
    m = order.shape[0]
    last = np.empty(m, np.int64)
    length = np.zeros(m, np.int64)
    chain = np.full(d, -1, np.int64)
    lev = np.zeros(d, np.int64)
    nch = 0
    for t in range(m):
        j = order[t]
        best = -1
        for ch in range(nch):
            l = last[ch]
            if cnt[l] < cnt[j]:
                continue
            if best >= 0 and cnt[l] >= cnt[last[best]]:
                continue
            ok = True
            for w in range(nw):
                if B[w, j] & ~B[w, l]:
                    ok = False
                    break
            if ok:
                best = ch
        if best < 0:
            best = nch
            nch += 1
        last[best] = j
        length[best] += 1
        chain[j] = best
        lev[j] = length[best]
    return chain, lev, nch


@nb.njit(cache=False)
def chain_codes(B, n, chain, lev, nch):
    """Code of every row in every chain: the largest level whose column contains the row (binary search, the
    columns of a chain are nested)."""
    d = chain.shape[0]
    length = np.zeros(nch, np.int64)
    for j in range(d):
        if chain[j] >= 0:
            length[chain[j]] = max(length[chain[j]], lev[j])
    cp = np.zeros(nch + 1, np.int64)
    for ch in range(nch):
        cp[ch + 1] = cp[ch] + length[ch]
    cols = np.empty(cp[nch], np.int64)
    for j in range(d):
        if chain[j] >= 0:
            cols[cp[chain[j]] + lev[j] - 1] = j
    Q = np.zeros((nch, n), np.int32)
    nw = B.shape[0]
    for ch in range(nch):
        m = length[ch]
        top = 1
        while top <= m:
            top <<= 1
        loc = np.zeros(top, np.uint64)  # loc[l - 1]: bits of level l (zero past the chain: never taken)
        for w in range(nw):
            for l in range(m):
                loc[l] = B[w, cols[cp[ch] + l]]
            for i in range(w * 64, min(n, w * 64 + 64)):
                sh = np.uint64(i & 63)
                lo = 0  # levels 1..lo contain the row (branchless binary search)
                step = top >> 1
                while step > 0:
                    lo += step * np.int64((loc[lo + step - 1] >> sh) & np.uint64(1))
                    step >>= 1
                Q[ch, i] = lo
    return Q


@nb.njit(cache=False)
def colsum(QT, R, kind, ncode, vcp, vcols, lev_of, tptr, tabv, pow2, vrp, vrows):
    """out[j, p] = sum_i x_ij^(1 + pow2[p]) R[i, p] for every column, from per-code histograms of each variable."""
    V, n = QT.shape
    P = R.shape[1]
    d = lev_of.shape[0]
    out = np.zeros((d, P))
    for v in range(V):
        nc = ncode[v]
        Bq = np.zeros((nc, P))
        for t_ in range(vrp[v], vrp[v + 1]):
            i = vrows[t_]
            q = QT[v, i]
            for p in range(P):
                Bq[q, p] += R[i, p]
        if kind[v] == 0:
            for q in range(nc - 2, -1, -1):  # suffix sums: column of level l sums codes >= l
                for p in range(P):
                    Bq[q, p] += Bq[q + 1, p]
            for t in range(vcp[v], vcp[v + 1]):
                j = vcols[t]
                for p in range(P):
                    out[j, p] = Bq[lev_of[j], p]
        else:
            for t in range(vcp[v], vcp[v + 1]):
                j = vcols[t]
                base = tptr[j]
                for p in range(P):
                    s = 0.0
                    for q in range(nc):
                        x = tabv[base + q]
                        s += (x * x if pow2[p] else x) * Bq[q, p]
                    out[j, p] = s
    return out


@nb.njit(cache=False)
def xval(QT, var_of, tptr, tabv, i, j):
    return tabv[tptr[j] + QT[var_of[j], i]]


@nb.njit(cache=False)
def score_kernel(QT, var_of, tptr, tabv, S, wS):
    n = QT.shape[1]
    s = np.zeros(n)
    for t in range(S.shape[0]):
        j = S[t]
        v = var_of[j]
        base = tptr[j]
        for i in range(n):
            s[i] += wS[t] * tabv[base + QT[v, i]]
    return s


@nb.njit(cache=False)
def forbid_kernel(S, skip, var_of, kind, vcols, vcp, cntw, mb, d):
    mask = np.zeros(d, np.bool_)
    if mb <= 0:
        return mask
    for t in range(S.shape[0]):
        u = S[t]
        if u == skip:
            continue
        v = var_of[u]
        if kind[v] != 0:
            continue
        cu = cntw[u]
        for r in range(vcp[v], vcp[v + 1]):
            j = vcols[r]
            if abs(cntw[j] - cu) < mb:
                mask[j] = True
    return mask


@nb.njit(cache=False)
def chain_tables(tptr, lev_of, var_of, nch):
    """Value tables of the chain columns: x = 1 for codes >= level (other columns filled by the caller)."""
    tabv = np.zeros(tptr[-1])
    for j in range(lev_of.shape[0]):
        if var_of[j] < nch:
            for t in range(tptr[j] + lev_of[j], tptr[j + 1]):
                tabv[t] = 1.0
    return tabv


@nb.njit(cache=False)
def nz_rows(QT, kind):
    """Rows that can have a nonzero value per variable (code > 0 for a chain, every row otherwise), CSR."""
    V, n = QT.shape
    vrp = np.zeros(V + 1, np.int64)
    for v in range(V):
        cnt = 0
        if kind[v] > 0:
            cnt = n
        else:
            for i in range(n):
                cnt += QT[v, i] > 0
        vrp[v + 1] = vrp[v] + cnt
    vrows = np.empty(vrp[V], np.int64)
    for v in range(V):
        t = vrp[v]
        for i in range(n):
            if kind[v] > 0 or QT[v, i] > 0:
                vrows[t] = i
                t += 1
    return vrp, vrows


@nb.njit(cache=False)
def gather_cols(Q, idx):
    V = Q.shape[0]
    m = idx.shape[0]
    out = np.empty((V, m), Q.dtype)
    for v in range(V):
        for t in range(m):
            out[v, t] = Q[v, idx[t]]
    return out


class Data:
    def forbid(self, S, skip=-1):
        """Columns that would make a piece narrower than msup together with a support column of the same chain."""
        return forbid_kernel(np.asarray(S, np.int64), skip, self.var_of, self.kind, self.vcols, self.vcp, self.cntw,
                             MS_BAND * self.msup, self.d)

    def col(self, j):
        v = self.var_of[j]
        if self.kind[v] == 0:
            return (self.QT[v] >= self.lev_of[j]).astype(np.float64)
        return self.tabv[self.tptr[j] + self.QT[v]]

    def score(self, w):
        S = np.flatnonzero(w)
        return score_kernel(self.QT, self.var_of, self.tptr, self.tabv, S, w[S].astype(np.float64))

    def code(self, j):
        """Code of every row's value in column j and the number of codes."""
        v = self.var_of[j]
        if self.kind[v] == 0:
            return (self.QT[v] >= self.lev_of[j]).astype(np.int64), 2
        return self.QT[v].astype(np.int64), int(self.ncode[v])

    def colsum(self, R, pow2=False):
        if np.ndim(pow2) == 0:
            pow2 = np.full(R.shape[1], bool(pow2))
        return colsum(self.QT, np.ascontiguousarray(R), self.kind, self.ncode, self.vcp, self.vcols, self.lev_of,
                      self.tptr, self.tabv, np.asarray(pow2, np.bool_), self.vrp, self.vrows)

    def __init__(self, X, y01):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n0, d = X.shape
        self.d = d
        ys0 = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        isbin, cnt, B = scan_columns(X)
        const = np.where(isbin, (cnt == 0) | (cnt == n0), False)
        nb_cols = np.flatnonzero(~isbin)
        if len(nb_cols):
            const[nb_cols] = np.ptp(X[:, nb_cols], axis=0) == 0
        chainable = isbin & ~const
        cand = np.flatnonzero(chainable)
        order = cand[np.lexsort((cand, -cnt[cand]))]
        chain, lev, nch = build_chains(B, cnt, order, d)
        others = np.flatnonzero(~chainable)
        V = nch + len(others)
        kind = np.zeros(V, np.int64)
        ncode = np.zeros(V, np.int64)
        var_of = chain.copy()
        lev_of = lev.copy()
        Q0 = np.empty((V, n0), np.int32)
        Q0[:nch] = chain_codes(B, n0, chain, lev, nch)
        for ch in range(nch):
            ncode[ch] = 0
        np.maximum.at(ncode, chain[cand], lev[cand])
        ncode[:nch] += 1
        tabs = [None] * d
        for t, j in enumerate(others):
            v = nch + t
            u, inv = np.unique(X[:, j], return_inverse=True)
            Q0[v] = inv.ravel()
            ncode[v] = len(u)
            kind[v] = 1 if len(u) <= RAW_DENSE else 2
            var_of[j] = v
            lev_of[j] = 0
            tabs[j] = u.astype(np.float64)
        tptr = np.zeros(d + 1, np.int64)
        np.cumsum(ncode[var_of], out=tptr[1:])
        tabv = chain_tables(tptr, lev_of, var_of, nch)
        for j in others:
            tabv[tptr[j]:tptr[j + 1]] = tabs[j]
        # unique (codes, y) rows with counts, via a random projection hash (deterministic seed)
        r = np.random.default_rng(12345).standard_normal(V + 1)
        h = r[:V] @ Q0 + ys0 * r[V]
        _, first, inv = np.unique(h, return_index=True, return_inverse=True)
        self.c = np.bincount(inv.ravel()).astype(np.float64)
        self.QT = gather_cols(Q0, first.astype(np.int64))
        self.y = np.ascontiguousarray(ys0[first])
        self.n = self.QT.shape[1]
        self.N = float(self.c.sum())
        self.yc = self.y * self.c
        self.V, self.kind, self.ncode, self.var_of, self.lev_of = V, kind, ncode, var_of, lev_of
        self.tptr, self.tabv = tptr, tabv
        vorder = np.lexsort((lev_of, var_of))
        self.vcols = vorder.astype(np.int64)
        self.vcp = np.zeros(V + 1, np.int64)
        np.add.at(self.vcp, var_of + 1, 1)
        self.vcp = np.cumsum(self.vcp)
        self.nchains = nch
        self.allbin = bool(np.all((tabv == 0.0) | (tabv == 1.0)))
        # rows that can have a nonzero value per variable (code > 0 for a chain, every row otherwise)
        self.vrp, self.vrows = nz_rows(self.QT, kind)
        cn = self.c / self.N
        MM = self.colsum(np.stack([cn, cn], 1), np.array([False, True]))
        M, M2 = MM[:, 0].copy(), MM[:, 1].copy()
        var = M2 - M * M
        self.norm = np.sqrt(np.maximum(var, 0.0) * self.N)  # centred column norm, as in FasterRisk
        self.valid = self.norm > 1e-9
        # minimum support: every piece of a variable's step function must hold at least msup (weighted) rows
        self.cntw = M * self.N
        self.msup = max(MS_FRAC * self.N, MS_SQRT * np.sqrt(self.N), MS_ABS)
        colbin = np.ones(d, np.bool_)
        if not self.allbin:
            colbin[others] = [np.all((tabs[j] == 0) | (tabs[j] == 1)) for j in others]
        if MS_CHAIN:  # only thresholds of a numeric variable (chains of >= 2 columns)
            colbin &= (kind[var_of] == 0) & (ncode[var_of] > 2)
        self.valid &= ~colbin | ((self.cntw >= MS_ENDM * self.msup) & (self.N - self.cntw >= MS_ENDM * self.msup))
        self.scale = np.where(self.valid, 1.0 / np.maximum(self.norm, 1e-12), 0.0)
        self.bvalid = self.valid.copy()  # columns the beam may add


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
    __slots__ = ("loss", "w", "S", "inv", "ng", "gy", "gc", "rep", "mg", "par", "j", "sig", "h")


@nb.njit(cache=False)
def group_design(QT, var_of, tptr, tabv, gy, rep, S):
    ng = rep.shape[0]
    p = S.shape[0] + 1
    Z = np.empty((ng, p))
    for g in range(ng):
        Z[g, 0] = gy[g]
        for q in range(1, p):
            Z[g, q] = gy[g] * xval(QT, var_of, tptr, tabv, rep[g], S[q - 1])
    return Z


@nb.njit(cache=False)
def child_fit_batch(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, js, QT, var_of, lev_of, kind, ncode,
                    tptr, tabv, c, bound, tol, screen, vrp, vrows, init):
    """Fit the children (parent support + column j) for the columns js (sorted by variable) of one parent
    without regrouping rows: the child's cells are the parent's groups split by the value of x_j, with weights
    from a (group, code) histogram of j's variable (suffix sums over codes for a chain)."""
    m = js.shape[0]
    ps = par_S.shape[0]
    losses = np.empty(m)
    Wout = np.empty((m, ps + 2))
    n = par_inv.shape[0]
    Zp = np.empty((par_ng, ps))  # y_g * x_g of the parent's support columns
    for g in range(par_ng):
        for r in range(ps):
            Zp[g, r] = par_gy[g] * xval(QT, var_of, tptr, tabv, par_rep[g], par_S[r])
    offg = np.zeros(par_ng)  # margin of the parent's support part per group (fixed in the screen)
    for g in range(par_ng):
        for r in range(ps):
            offg[g] += Zp[g, r] * par_w[r + 1]
    u = 0
    while u < m:
        v = var_of[js[u]]
        u2 = u
        while u2 < m and var_of[js[u2]] == v:
            u2 += 1
        nc = ncode[v]
        H = np.zeros((par_ng, nc))
        if kind[v] == 0 and (u2 - u) * n < n + par_ng * nc:
            # few thresholds of this chain: accumulate the weight of x_j = 1 per group directly
            for t_ in range(vrp[v], vrp[v + 1]):
                i = vrows[t_]
                qi = QT[v, i]
                g = par_inv[i]
                for t in range(u, u2):
                    l = lev_of[js[t]]
                    if qi >= l:
                        H[g, l] += c[i]
        else:
            for t_ in range(vrp[v], vrp[v + 1]):
                i = vrows[t_]
                H[par_inv[i], QT[v, i]] += c[i]
        if kind[v] == 0 and not ((u2 - u) * n < n + par_ng * nc):
            for g in range(par_ng):
                for q in range(nc - 2, -1, -1):
                    H[g, q] += H[g, q + 1]
        for t in range(u, u2):
            j = js[t]
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
                    w[r + 1] = init[t]
                    ins = True
                else:
                    S[r] = par_S[q]
                    w[r + 1] = par_w[q + 1]
                    q += 1
            maxcell = par_ng * (2 if kind[v] == 0 else nc)
            ncol = 3 if screen else ps + 2
            Z = np.empty((maxcell, ncol))
            cw = np.empty(maxcell)
            nce = 0
            for g in range(par_ng):
                yg = par_gy[g]
                ncg = 2 if kind[v] == 0 else nc
                for xq in range(ncg):
                    if kind[v] == 0:
                        w1 = H[g, lev_of[j]]
                        if xq == 1:
                            wt = w1
                        else:
                            w0 = par_gc[g] - w1
                            wt = w0 if w0 > 0.5 else 0.0
                        xv = float(xq)
                    else:
                        wt = H[g, xq]
                        xv = tabv[tptr[j] + xq]
                    if wt <= 0.0:
                        continue
                    Z[nce, 0] = yg
                    if screen:
                        Z[nce, 1] = yg * xv
                        Z[nce, 2] = offg[g]
                    else:
                        for r in range(ps + 1):
                            if r == pos:
                                Z[nce, r + 1] = yg * xv
                            else:
                                Z[nce, r + 1] = Zp[g, r if r < pos else r - 1]
                    cw[nce] = wt
                    nce += 1
            if screen:
                # only the intercept and the new coefficient move (an upper bound on the child's loss)
                w2 = np.array([w[0], 0.0, 1.0])
                lo2 = np.array([-1e300, -bound, 1.0])
                hi2 = np.array([1e300, bound, 1.0])
                loss, _ = newton_fit(Z[:nce], cw[:nce], w2, lo2, hi2, 20, tol)
                w[0] = w2[0]
                w[pos + 1] = w2[1]
            else:
                lo = np.full(ps + 2, -bound)
                hi = np.full(ps + 2, bound)
                lo[0] = -1e300
                hi[0] = 1e300
                loss, _ = newton_fit(Z[:nce], cw[:nce], w, lo, hi, 50, tol)
            losses[t] = loss
            Wout[t] = w
        u = u2
    return losses, Wout


@nb.njit(cache=False)
def make_child(par_inv, par_ng, par_S, par_w, j, colcode_j, ncode_j, y, c, QT, var_of, tptr, tabv, bound, tol):
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
    Z = group_design(QT, var_of, tptr, tabv, gy, rep, S)
    loss, mg = newton_fit(Z, gc, w, lo, hi, 50, tol)
    return S, inv, ng, gy, gc, rep, loss, mg, w


@nb.njit(cache=False)
def materialise_kernel(par_inv, par_ng, j, S, w, QT, var_of, kind, lev_of, ncode, tptr, tabv, y, c):
    v = var_of[j]
    n = par_inv.shape[0]
    code = np.empty(n, np.int64)
    if kind[v] == 0:
        l = lev_of[j]
        for i in range(n):
            code[i] = 1 if QT[v, i] >= l else 0
        nc = 2
    else:
        for i in range(n):
            code[i] = QT[v, i]
        nc = ncode[v]
    inv, ng, gy, gc, rep = regroup(par_inv, par_ng, code, nc, y, c)
    Z = group_design(QT, var_of, tptr, tabv, gy, rep, S)
    mg = np.zeros(ng)
    for g in range(ng):
        for q in range(Z.shape[1]):
            mg[g] += Z[g, q] * w[q]
    return inv, ng, gy, gc, rep, mg


def materialise(D, ch):
    par, j = ch.par, ch.j
    ch.inv, ch.ng, ch.gy, ch.gc, ch.rep, ch.mg = materialise_kernel(
        par.inv, par.ng, j, np.array(ch.S, dtype=np.int64), ch.w, D.QT, D.var_of, D.kind, D.lev_of, D.ncode, D.tptr,
        D.tabv, D.y, D.c)
    ch.par = None


@nb.njit(cache=False)
def pick_per_var(g, var_of, V, m):
    """Columns with the largest g > 0, at most one per variable (its best), best first."""
    bestj = np.full(V, -1, np.int64)
    for j in range(g.shape[0]):
        if g[j] > 0:
            v = var_of[j]
            if bestj[v] < 0 or g[j] > g[bestj[v]]:
                bestj[v] = j
    nv = 0
    for v in range(V):
        if bestj[v] >= 0:
            nv += 1
    cand = np.empty(nv, np.int64)
    vals = np.empty(nv)
    u = 0
    for v in range(V):
        if bestj[v] >= 0:
            cand[u] = bestj[v]
            vals[u] = -g[bestj[v]]
            u += 1
    o = np.argsort(vals, kind="mergesort")
    return cand[o[:m]]


@nb.njit(cache=False)
def sig_groups(mg):
    out = np.empty(mg.shape[0])
    for g in range(mg.shape[0]):
        out[g] = _sig(-mg[g])
    return out


@nb.njit(cache=False)
def beam_rows_pr(Pr, yc, c):
    """beam_rows from sigma(-m) per row."""
    n, P = Pr.shape
    R = np.empty((n, 3 * P))
    for i in range(n):
        for q in range(P):
            pr = Pr[i, q]
            R[i, q] = yc[i] * pr
            h = c[i] * pr * (1.0 - pr)
            R[i, P + q] = h
            R[i, 2 * P + q] = h
    return R


@nb.njit(cache=False)
def beam_rows(Mg, yc, c):
    """Per row and parent: gradient weight y c sigma(-m), curvature c sigma(m) sigma(-m) (twice: x and x^2)."""
    n, P = Mg.shape
    R = np.empty((n, 3 * P))
    for i in range(n):
        for q in range(P):
            pr = _sig(-Mg[i, q])
            R[i, q] = yc[i] * pr
            h = c[i] * pr * (1.0 - pr)
            R[i, P + q] = h
            R[i, 2 * P + q] = h
    return R


@nb.njit(cache=False)
def newton_gains(Gm, h00, valid, bound):
    """gain[q, j]: loss decrease of one Newton step on a new coefficient for column j (box-clipped), with the
    intercept's curvature projected out."""
    d = Gm.shape[0]
    P = h00.shape[0]
    GA = np.full((P, d), -1.0)
    BA = np.zeros((P, d))
    for j in range(d):
        if not valid[j]:
            continue
        for q in range(P):
            g = abs(Gm[j, q])
            h0 = Gm[j, P + q]
            heff = max(Gm[j, 2 * P + q] - h0 * h0 / max(h00[q], 1e-300), 1e-12)
            if g <= bound * heff:
                GA[q, j] = 0.5 * g * g / heff
                BA[q, j] = Gm[j, q] / heff
            else:
                GA[q, j] = bound * g - 0.5 * heff * bound * bound
                BA[q, j] = bound if Gm[j, q] > 0 else -bound
    return GA, BA


@nb.njit(cache=False)
def beam_proposals(GA, BA, ploss, PS, plen, phash, psig, colh, varh, var_of, V, kind, vcols, vcp, cntw, mb, seen,
                   child_size, nfit, sigmax):
    """Per parent its child_size best variables (each at its best column by the Newton gain), new supports only
    (seen: hashes of supports already proposed); sorted by estimated loss, at most sigmax per multiset of
    variables, the nfit best. Returns (parent, column, estimated loss, warm start, support hash, signature)."""
    P, d = GA.shape
    maxp = P * child_size
    e_est = np.empty(maxp)
    e_q = np.empty(maxp, np.int64)
    e_j = np.empty(maxp, np.int64)
    e_b = np.empty(maxp)
    e_h = np.empty(maxp, np.uint64)
    e_s = np.empty(maxp, np.uint64)
    u = 0
    for q in range(P):
        gain = GA[q].copy()
        Sq = PS[q, :plen[q]]
        for t in range(plen[q]):
            gain[Sq[t]] = -1.0
        if plen[q] > 0 and mb > 0:
            mask = forbid_kernel(Sq, -1, var_of, kind, vcols, vcp, cntw, mb, d)
            for j in range(d):
                if mask[j]:
                    gain[j] = -1.0
        pick = pick_per_var(gain, var_of, V, child_size)
        for t in range(pick.shape[0]):
            j = pick[t]
            key = phash[q] + colh[j]
            if key in seen:
                continue
            seen[key] = True
            e_est[u] = ploss[q] - gain[j]
            e_q[u] = q
            e_j[u] = j
            e_b[u] = BA[q, j]
            e_h[u] = key
            e_s[u] = psig[q] + varh[var_of[j]]
            u += 1
    o = np.argsort(e_est[:u], kind="mergesort")
    cnt = Dict.empty(key_type=nb.types.uint64, value_type=nb.types.int64)
    sel = np.empty(min(u, nfit), np.int64)
    m = 0
    for t in range(u):
        if m >= nfit:
            break
        i = o[t]
        c0 = cnt.get(e_s[i], 0)
        if c0 < sigmax:
            cnt[e_s[i]] = c0 + 1
            sel[m] = i
            m += 1
    sel = sel[:m]
    return e_q[sel], e_j[sel], e_est[sel], e_b[sel], e_h[sel], e_s[sel]


def beam_search(D, k, parent_size=10, child_size=10, deadline=np.inf):
    """Beam search over supports. Every column's gain for every parent is bounded by one Newton step on the new
    coefficient (from per-code histograms of the gradient and curvature rows); each parent proposes its
    child_size best variables (each at its best column), and only the SCREEN * beam best proposals overall are
    fitted exactly."""
    root = Node()
    root.S = ()
    root.sig = 0
    root.h = 0
    hr = np.random.default_rng(5)
    colh = hr.integers(0, 2 ** 63, D.d, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    varh = hr.integers(0, 2 ** 63, D.V, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    root.inv, root.ng, root.gy, root.gc, root.rep = regroup(np.zeros(D.n, np.int64), 1, (D.y > 0).astype(np.int64), 2, D.y, D.c)
    npos = D.c[D.y > 0].sum()
    root.w = np.array([np.log(npos / (D.N - npos))])
    root.mg = root.gy * root.w[0]
    root.loss = total_loss(root.mg, root.gc)
    parents = [root]
    seen = Dict.empty(key_type=nb.types.uint64, value_type=nb.types.boolean)
    bound = float(COEF_BOUND)
    nlev = min(k, int(D.valid.sum()))
    for lev in range(nlev):
        last = lev == nlev - 1
        if time.perf_counter() > deadline:
            parents = parents[:1]  # out of time: finish the support greedily from the best parent
        P = len(parents)
        Pr = np.empty((D.n, P))
        for q, par in enumerate(parents):
            Pr[:, q] = sig_groups(par.mg)[par.inv]  # sigma(-margin) per row group, gathered to rows
        R = beam_rows_pr(Pr, D.yc, D.c)
        if D.allbin:  # x^2 = x: the third block equals the second
            Gm = D.colsum(R[:, :2 * P])
            Gm = np.hstack([Gm, Gm[:, P:]])
        else:
            pw = np.zeros(3 * P, np.bool_)
            pw[2 * P:] = True
            Gm = D.colsum(R, pw)
        GA, BA = newton_gains(Gm, R[:, P:2 * P].sum(0), D.bvalid, bound)
        keep = FINAL_POOL if last else parent_size
        nfit = int((LASTSCREEN if last else SCREEN) * keep)
        lmax = max(1, max(len(p.S) for p in parents))
        PS = np.zeros((P, lmax), np.int64)
        plen = np.zeros(P, np.int64)
        for q, par in enumerate(parents):
            plen[q] = len(par.S)
            PS[q, :len(par.S)] = par.S
        eq, ej, eest, eb, eh, es = beam_proposals(
            GA, BA, np.array([p.loss for p in parents]), PS, plen, np.array([p.h for p in parents], np.uint64),
            np.array([p.sig for p in parents], np.uint64), colh, varh, D.var_of, D.V, D.kind, D.vcols, D.vcp, D.cntw,
            MS_BAND * D.msup, seen, child_size, nfit, SIGMAX)
        if len(eq) == 0:
            break
        props = [(eest[t], int(eq[t]), int(ej[t]), int(eh[t]), int(es[t]), eb[t]) for t in range(len(eq))]
        children = []
        byp = {}
        for t in props:
            byp.setdefault(t[1], []).append(t)
        for q in sorted(byp):
            par = parents[q]
            par_S = np.array(par.S, dtype=np.int64)
            lst = byp[q]
            js = np.array([t[2] for t in lst], dtype=np.int64)
            b0s = np.array([t[5] for t in lst])
            sparse = D.kind[D.var_of[js]] < 2
            if sparse.any():
                idx = np.flatnonzero(sparse)
                idx = idx[np.argsort(D.var_of[js[idx]], kind="stable")]
                losses, W = child_fit_batch(par.inv, par.ng, par.gy, par.gc, par.rep, par_S, par.w, js[idx],
                                            D.QT, D.var_of, D.lev_of, D.kind, D.ncode, D.tptr, D.tabv, D.c,
                                            bound, BTOL, False, D.vrp, D.vrows, b0s[idx])
                for u, qq in enumerate(idx):
                    ch = Node()
                    ch.loss, ch.w, ch.inv, ch.par, ch.j = losses[u], W[u], None, par, int(js[qq])
                    ch.S = tuple(sorted(par.S + (ch.j,)))
                    ch.h, ch.sig = lst[qq][3], lst[qq][4]
                    children.append(ch)
            for qq in np.flatnonzero(~sparse):
                j = int(js[qq])
                ch = Node()
                code, nc = D.code(j)
                (S, ch.inv, ch.ng, ch.gy, ch.gc, ch.rep, ch.loss, ch.mg, ch.w) = make_child(
                    par.inv, par.ng, par_S, par.w, j, code, nc, D.y, D.c, D.QT, D.var_of, D.tptr, D.tabv,
                    bound, BTOL)
                ch.S = tuple(sorted(par.S + (j,)))
                ch.h, ch.sig = lst[qq][3], lst[qq][4]
                ch.par, ch.j = par, j
                children.append(ch)
        children.sort(key=lambda t: t.loss)
        if SIGMAX > 0:
            # at most SIGMAX children per multiset of variables (supports that differ only in thresholds)
            cnt, sel = {}, []
            for ch in children:
                if cnt.get(ch.sig, 0) < SIGMAX:
                    cnt[ch.sig] = cnt.get(ch.sig, 0) + 1
                    sel.append(ch)
            children = sel
        parents = children[:keep]
        for ch in parents:
            if ch.inv is None:  # materialise the row groups of the selected children only
                materialise(D, ch)
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
        # integer scores: calibrate on the distinct score values
        inv_, sv, Wp, Wn = bin_scores(sg, gy, gc)
        loss, _, _ = calibrate_bins(sv / sd, Wp, Wn, 0.0, 0.0)
        if loss < best_l:
            best_l = loss
            best_r[:] = r
    return best_r, best_l


def calib_round(D, nd, n_mult=20):
    """Best rounding of the node's beta by the calibrated loss over a grid of scales."""
    S = np.array(nd.S, dtype=np.int64)
    XS = group_design(D.QT, D.var_of, D.tptr, D.tabv, np.ones(nd.ng), nd.rep, S)[:, 1:]
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
def eval_cells(q, deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b):
    """out[:, q, r]: estimated calibrated loss and loss at fixed (a, b) after s += deltas[q, r] * x, from the
    touched cells (bin cb, value cx, weights cwp / cwn of y = +1 / -1). The estimate is the exact loss at the
    current (a, b) minus a trust-region Newton step in (a, b)."""
    nd = deltas.shape[1]
    stepa = np.empty(nd)
    stepb = np.empty(nd)
    hyb = np.empty(nd)
    dL = np.zeros(nd)
    dGa = np.zeros(nd)
    dGb = np.zeros(nd)
    dHaa = np.zeros(nd)
    dHab = np.zeros(nd)
    dHbb = np.zeros(nd)
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


@nb.njit(cache=False)
def eval_vars(cols, qidx, deltas, inv, nbins, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of, kind,
              ncode, tptr, tabv, out, tr_a, tr_b, vrp, vrows):
    """Moves s += deltas[q] * x_cols[q] for columns sorted by variable (qidx: their rows in deltas / out).
    Rows are histogrammed once per variable by (score bin, code); a chain's columns read suffix sums."""
    m = cols.shape[0]
    n = inv.shape[0]
    u = 0
    while u < m:
        v = var_of[cols[u]]
        u2 = u
        while u2 < m and var_of[cols[u2]] == v:
            u2 += 1
        nc = ncode[v]
        kv = kind[v]
        maxc = nbins if kv == 0 else (nbins * nc if kv == 1 else n)
        cb = np.empty(maxc, np.int64)
        cx = np.empty(maxc)
        cwp = np.empty(maxc)
        cwn = np.empty(maxc)
        if kv < 2:
            HP = np.zeros((nbins, nc))
            HN = np.zeros((nbins, nc))
            for t_ in range(vrp[v], vrp[v + 1]):
                i = vrows[t_]
                if y[i] > 0:
                    HP[inv[i], QT[v, i]] += c[i]
                else:
                    HN[inv[i], QT[v, i]] += c[i]
            if kv == 0:
                for bq in range(nbins):
                    for qq in range(nc - 2, -1, -1):
                        HP[bq, qq] += HP[bq, qq + 1]
                        HN[bq, qq] += HN[bq, qq + 1]
        for t in range(u, u2):
            j = cols[t]
            nt = 0
            if kv == 0:
                l = lev_of[j]
                for bq in range(nbins):
                    if HP[bq, l] > 0.0 or HN[bq, l] > 0.0:
                        cb[nt] = bq
                        cx[nt] = 1.0
                        cwp[nt] = HP[bq, l]
                        cwn[nt] = HN[bq, l]
                        nt += 1
            elif kv == 1:
                for bq in range(nbins):
                    for qq in range(nc):
                        x = tabv[tptr[j] + qq]
                        if x != 0.0 and (HP[bq, qq] > 0.0 or HN[bq, qq] > 0.0):
                            cb[nt] = bq
                            cx[nt] = x
                            cwp[nt] = HP[bq, qq]
                            cwn[nt] = HN[bq, qq]
                            nt += 1
            else:
                for i in range(n):
                    x = tabv[tptr[j] + QT[v, i]]
                    if x != 0.0:
                        cb[nt] = inv[i]
                        cx[nt] = x
                        cwp[nt] = c[i] if y[i] > 0 else 0.0
                        cwn[nt] = c[i] - cwp[nt]
                        nt += 1
            eval_cells(qidx[t], deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b)
        u = u2
    return out


@nb.njit(cache=False)
def removal_state(s, dj, y, c, a, b):
    """Score bins and statistics at fixed (a, b) after s -= dj, with the per-row screening derivatives."""
    n = s.shape[0]
    s2 = s - dj
    inv, sv, Wp, Wn = bin_scores(s2, y, c)
    m = sv.shape[0]
    lp = np.empty(m)
    ln_ = np.empty(m)
    pp = np.empty(m)
    pn = np.empty(m)
    wq = np.empty(m)
    T = bin_stats(sv, Wp, Wn, a, b, lp, ln_, pp, pn, wq)
    r = np.empty(n)
    h = np.empty(n)
    for i in range(n):
        q = inv[i]
        r[i] = (-pp[q] if y[i] > 0 else pn[q]) * c[i]
        h[i] = wq[q] * c[i]
    return s2, inv, sv, Wp, Wn, lp, ln_, pp, pn, wq, T, r, h


@nb.njit(cache=False)
def screen_pick(G, H, a, dl, m):
    """Indices (ascending) of the m columns with the best second-order score min_delta a d G + (a d)^2 H / 2."""
    nc = G.shape[0]
    sc = np.empty(nc)
    for q in range(nc):
        best = 0.0
        for r in range(dl.shape[0]):
            u = a * dl[r]
            v = u * G[q] + 0.5 * u * u * H[q]
            if v < best:
                best = v
        sc[q] = best
    idx = np.argsort(sc, kind="mergesort")[:m]
    return np.sort(idx)


@nb.njit(cache=False)
def screen_cols(G12, p, cols, a, dl, m):
    """The m columns (ascending) of `cols` with the best second-order score min_delta u G + u^2 H / 2 (u = a d),
    G and H in columns p and p + 1 of G12."""
    nc = cols.shape[0]
    sc = np.empty(nc)
    for t in range(nc):
        G = G12[cols[t], p]
        H = G12[cols[t], p + 1]
        best = 0.0
        for r in range(dl.shape[0]):
            u = a * dl[r]
            v = u * G + 0.5 * u * u * H
            if v < best:
                best = v
        sc[t] = best
    # partial selection of the m smallest (ties: lower index first)
    m = min(m, nc)
    sel = np.empty(m, np.int64)
    cnt = 0
    for t in range(nc):
        v = sc[t]
        if cnt < m:
            u = cnt
            cnt += 1
        elif v < sc[sel[m - 1]]:
            u = m - 1
        else:
            continue
        while u > 0 and sc[sel[u - 1]] > v:
            sel[u] = sel[u - 1]
            u -= 1
        sel[u] = t
    return cols[np.sort(sel)]


@nb.njit(cache=False)
def swap_phase(s, w, S, free, y, c, a, b, QT, var_of, lev_of, kind, ncode, tptr, tabv, vrp, vrows, vcols, vcp,
               cntw, mb, nzv, n_screen, ne, tr_a, tr_b):
    """Swap candidates (remove S[q], add a column with a value): per removal, the n_screen best swap-in columns by
    a second-order screen at the removal state, evaluated by the calibrated-loss estimate; returns the ne best per
    removal as arrays (estimate, removed column, added column, value, loss at fixed (a, b))."""
    n = s.shape[0]
    k = S.shape[0]
    d = w.shape[0]
    nv = nzv.shape[0]
    R = np.empty((n, 2 * k))
    INV = np.empty((k, n), np.int64)
    BIN = List()  # per removal: (sv, lp, ln, pp, pn, wq) stacked, and T
    TT = np.empty((k, 6))
    for q in range(k):
        j = S[q]
        v = var_of[j]
        base = tptr[j]
        dj = np.empty(n)
        for i in range(n):
            dj[i] = w[j] * tabv[base + QT[v, i]]
        s2, inv, sv, Wp, Wn, lp, ln_, pp, pn, wq, T, r_, h_ = removal_state(s, dj, y, c, a, b)
        R[:, 2 * q] = r_
        R[:, 2 * q + 1] = h_
        INV[q] = inv
        st6 = np.empty((6, sv.shape[0]))
        st6[0] = sv
        st6[1] = lp
        st6[2] = ln_
        st6[3] = pp
        st6[4] = pn
        st6[5] = wq
        BIN.append(st6)
        TT[q] = T
    pow2 = np.zeros(2 * k, np.bool_)
    for q in range(k):
        pow2[2 * q + 1] = True
    G12 = colsum(QT, R, kind, ncode, vcp, vcols, lev_of, tptr, tabv, pow2, vrp, vrows)
    o_est = np.full(k * ne, np.inf)
    o_fx = np.full(k * ne, np.inf)
    o_rj = np.full(k * ne, -1, np.int64)
    o_aj = np.full(k * ne, -1, np.int64)
    o_v = np.zeros(k * ne)
    for q in range(k):
        j = S[q]
        mask = forbid_kernel(S, j, var_of, kind, vcols, vcp, cntw, mb, d)
        cnt = 0
        for t in range(d):
            if free[t] and not mask[t]:
                cnt += 1
        if cnt == 0:
            continue
        nonSq = np.empty(cnt, np.int64)
        cnt = 0
        for t in range(d):
            if free[t] and not mask[t]:
                nonSq[cnt] = t
                cnt += 1
        cols = screen_cols(G12, 2 * q, nonSq, a, nzv, n_screen)
        m = cols.shape[0]
        st6 = BIN[q]
        inv = INV[q]
        sv = st6[0]
        lp = st6[1]
        ln_ = st6[2]
        pp = st6[3]
        pn = st6[4]
        wq = st6[5]
        T = TT[q]
        vo = np.empty(m, np.int64)
        for t in range(m):
            vo[t] = var_of[cols[t]]
        o = np.argsort(vo, kind="mergesort")
        deltas = np.empty((m, nv))
        for t in range(m):
            deltas[t, :] = nzv
        out = np.empty((2, m, nv))
        eval_vars(cols[o], o, deltas, inv, sv.shape[0], sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of,
                  lev_of, kind, ncode, tptr, tabv, out, tr_a, tr_b, vrp, vrows)
        flat = out[0].ravel()
        fo = np.argsort(flat, kind="mergesort")
        u = 0
        for t in range(fo.shape[0]):
            if u >= ne:
                break
            f = fo[t]
            if not np.isfinite(flat[f]):
                continue
            r = f % nv
            qq = f // nv
            o_est[q * ne + u] = flat[f]
            o_fx[q * ne + u] = out[1].ravel()[f]
            o_rj[q * ne + u] = j
            o_aj[q * ne + u] = cols[qq]
            o_v[q * ne + u] = nzv[r]
            u += 1
    return o_est, o_rj, o_aj, o_v, o_fx


@nb.njit(cache=False)
def main_phase(w, S, free, k, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of, kind, ncode, tptr,
               tabv, vrp, vrows, vcols, vcp, cntw, mb, allv, nzv, ne, tr_a, tr_b, add_screen):
    """Value changes of the support columns (every other value; 0 removes) and additions (when |S| < k): the ne
    best by the estimate per support column, and the ne best additions overall."""
    m = S.shape[0]
    d = w.shape[0]
    nv = nzv.shape[0]
    nb_ = sv.shape[0]
    o_est = np.full(m * ne + ne, np.inf)
    o_fx = np.full(m * ne + ne, np.inf)
    o_aj = np.full(m * ne + ne, -1, np.int64)
    o_v = np.zeros(m * ne + ne)
    if m > 0:
        newv = np.empty((m, allv.shape[0] - 1))
        deltas = np.empty((m, allv.shape[0] - 1))
        vo = np.empty(m, np.int64)
        for q in range(m):
            u = 0
            wj = w[S[q]]
            for r in range(allv.shape[0]):
                if allv[r] != wj:
                    newv[q, u] = allv[r]
                    deltas[q, u] = allv[r] - wj
                    u += 1
            vo[q] = var_of[S[q]]
        o = np.argsort(vo, kind="mergesort")
        out = np.empty((2, m, newv.shape[1]))
        eval_vars(S[o], o, deltas, inv, nb_, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of, kind,
                  ncode, tptr, tabv, out, tr_a, tr_b, vrp, vrows)
        for q in range(m):
            fo = np.argsort(out[0, q], kind="mergesort")
            u = 0
            for t in range(fo.shape[0]):
                if u >= ne:
                    break
                r = fo[t]
                if not np.isfinite(out[0, q, r]):
                    continue
                o_est[q * ne + u] = out[0, q, r]
                o_fx[q * ne + u] = out[1, q, r]
                o_aj[q * ne + u] = S[q]
                o_v[q * ne + u] = newv[q, r]
                u += 1
    if m < k:
        mask = forbid_kernel(S, -1, var_of, kind, vcols, vcp, cntw, mb, d)
        cnt = 0
        for t in range(d):
            if free[t] and not mask[t]:
                cnt += 1
        if cnt > 0:
            cols = np.empty(cnt, np.int64)
            vo = np.empty(cnt, np.int64)
            cnt = 0
            for t in range(d):
                if free[t] and not mask[t]:
                    cols[cnt] = t
                    vo[cnt] = var_of[t]
                    cnt += 1
            if add_screen > 0 and cnt > add_screen:
                # second-order screen of the additions at the current (a, b); only the best are estimated
                R2 = np.empty((y.shape[0], 2))
                for i in range(y.shape[0]):
                    q = inv[i]
                    R2[i, 0] = (-pp[q] if y[i] > 0 else pn[q]) * c[i]
                    R2[i, 1] = wq[q] * c[i]
                pw2 = np.zeros(2, np.bool_)
                pw2[1] = True
                G12 = colsum(QT, R2, kind, ncode, vcp, vcols, lev_of, tptr, tabv, pw2, vrp, vrows)
                cols = screen_cols(G12, 0, cols, a, nzv, add_screen)
                cnt = cols.shape[0]
                vo = np.empty(cnt, np.int64)
                for t in range(cnt):
                    vo[t] = var_of[cols[t]]
            o = np.argsort(vo, kind="mergesort")
            deltas = np.empty((cnt, nv))
            for t in range(cnt):
                deltas[t, :] = nzv
            out = np.empty((2, cnt, nv))
            eval_vars(cols[o], o, deltas, inv, nb_, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of,
                      kind, ncode, tptr, tabv, out, tr_a, tr_b, vrp, vrows)
            flat = out[0].ravel()
            fl1 = out[1].ravel()
            # the ne smallest finite estimates
            u = 0
            for _ in range(ne):
                bi = -1
                bv = np.inf
                for f in range(flat.shape[0]):
                    if flat[f] < bv:
                        dup = False
                        for z in range(u):
                            if o_aj[m * ne + z] == cols[f // nv] and o_v[m * ne + z] == nzv[f % nv]:
                                dup = True
                        if not dup:
                            bv = flat[f]
                            bi = f
                if bi < 0:
                    break
                o_est[m * ne + u] = flat[bi]
                o_fx[m * ne + u] = fl1[bi]
                o_aj[m * ne + u] = cols[bi // nv]
                o_v[m * ne + u] = nzv[bi % nv]
                u += 1
    return o_est, o_aj, o_v, o_fx


@nb.njit(cache=False)
def check_kernel(s, w, L, est, rjs, ajs, vs, a, b, margin, y, c, QT, var_of, tptr, tabv):
    """Exact calibrated loss of the candidate moves (in order); returns the index of the best improving one (-1 if
    none) and its score vector, bins and calibration."""
    n = s.shape[0]
    bi = -1
    bL = np.inf
    bs = s
    binv = np.zeros(0, np.int64)
    bsv = np.zeros(0)
    bWp = np.zeros(0)
    bWn = np.zeros(0)
    ba = a
    bb = b
    for t in range(est.shape[0]):
        if est[t] > L * (1.0 + margin):
            continue  # the estimate says the move does not help
        s2 = s.copy()
        rj = rjs[t]
        if rj >= 0:
            v = var_of[rj]
            base = tptr[rj]
            for i in range(n):
                s2[i] -= w[rj] * tabv[base + QT[v, i]]
        aj = ajs[t]
        dv = vs[t] - w[aj]
        v = var_of[aj]
        base = tptr[aj]
        for i in range(n):
            s2[i] += dv * tabv[base + QT[v, i]]
        lo = s2[0]
        hi = s2[0]
        for i in range(n):
            lo = min(lo, s2[i])
            hi = max(hi, s2[i])
        if hi == lo:
            continue
        inv, sv, Wp, Wn = bin_scores(s2, y, c)
        L2, a2, b2 = calibrate_bins(sv, Wp, Wn, a, b)
        if L2 < L - 1e-9 * L and L2 < bL:
            bi, bL, bs, binv, bsv, bWp, bWn, ba, bb = t, L2, s2, inv, sv, Wp, Wn, a2, b2
    return bi, bL, bs, binv, bsv, bWp, bWn, ba, bb


@nb.njit(cache=False)
def pair_moves(s, XS, wS, y, c, a, b, L0, bound, coherent):
    """Exact calibrated loss of every pair move w_p += d1, w_q += d2 (d in {-1, +1}) of the support; returns
    the best (loss, p, q, d1, d2) (p = -1 when none improves on L0)."""
    n, k = XS.shape
    best = L0
    bp = -1
    bq = -1
    bd1 = 0.0
    bd2 = 0.0
    s2 = np.empty(n)
    for p in range(k):
        for q in range(p + 1, k):
            for d1 in (-1.0, 1.0):
                v1 = wS[p] + d1
                if v1 == 0.0 or abs(v1) > bound:
                    continue
                for d2 in (-1.0, 1.0):
                    if coherent and (d1 * wS[p] > 0) != (d2 * wS[q] > 0):
                        continue  # only both magnitudes up or both down
                    v2 = wS[q] + d2
                    if v2 == 0.0 or abs(v2) > bound:
                        continue
                    for i in range(n):
                        s2[i] = s[i] + d1 * XS[i, p] + d2 * XS[i, q]
                    inv, sv, Wp, Wn = bin_scores(s2, y, c)
                    if sv.shape[0] < 2:
                        continue
                    L, _, _ = calibrate_bins(sv, Wp, Wn, a, b)
                    if L < best - 1e-9 * best:
                        best = L
                        bp = p
                        bq = q
                        bd1 = d1
                        bd2 = d2
    return best, bp, bq, bd1, bd2


@nb.njit(cache=False)
def top_order(est, m):
    """Indices of the m smallest finite entries of est, ties in index order."""
    o = np.argsort(est, kind="mergesort")
    cnt = 0
    for t in range(o.shape[0]):
        if np.isfinite(est[o[t]]):
            cnt += 1
    cnt = min(cnt, m)
    return o[:cnt]


@nb.njit(cache=False)
def ils_kernel(w, s, L, a, b, inv, sv, Wp, Wn, visited, hw, k, max_iter, swaps, gate, N, ne, n_screen, add_screen,
               margin, mb, y, c, QT, var_of, lev_of, kind, ncode, tptr, tabv, vrp, vrows, vcols, vcp, cntw, valid,
               allv, nzv, tr_a, tr_b, pairs, hr):
    """Best-improvement local search on the integer lattice (value changes and additions; swaps when those fail),
    candidates ranked by the calibrated-loss estimate and the best 2 ne checked exactly. visited: keys of points
    already expanded in this fit (the search from them is deterministic). Returns (loss, w)."""
    d = w.shape[0]
    m_chk = 2 * ne
    for it in range(max_iter):
        key = 0.0
        for j in range(d):
            if w[j] != 0.0:
                key += w[j] * hw[j]
        if swaps:
            if key in visited:
                break
            visited[key] = True
        nb_ = sv.shape[0]
        lp = np.empty(nb_)
        ln_ = np.empty(nb_)
        pp = np.empty(nb_)
        pn = np.empty(nb_)
        wq = np.empty(nb_)
        T = bin_stats(sv, Wp, Wn, a, b, lp, ln_, pp, pn, wq)
        cntS = 0
        for j in range(d):
            if w[j] != 0.0:
                cntS += 1
        S = np.empty(cntS, np.int64)
        free = np.empty(d, np.bool_)
        u = 0
        nfree = 0
        for j in range(d):
            if w[j] != 0.0:
                S[u] = j
                u += 1
            free[j] = w[j] == 0.0 and valid[j]
            nfree += free[j]
        oe, oaj, ov, ofx = main_phase(w, S, free, k, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of,
                                      lev_of, kind, ncode, tptr, tabv, vrp, vrows, vcols, vcp, cntw, mb, allv, nzv,
                                      ne, tr_a, tr_b, add_screen)
        sel = top_order(oe, m_chk)
        rjs = np.full(sel.shape[0], -1, np.int64)
        bi, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, oe[sel], rjs, oaj[sel], ov[sel], a, b, margin,
                                                               y, c, QT, var_of, tptr, tabv)
        rj = -1
        aj = -1
        v = 0.0
        if bi >= 0:
            aj = oaj[sel[bi]]
            v = ov[sel[bi]]
        elif nfree > 0 and swaps and L / N <= gate:
            oe, orj, oaj, ov, ofx = swap_phase(s, w, S, free, y, c, a, b, QT, var_of, lev_of, kind, ncode, tptr, tabv,
                                               vrp, vrows, vcols, vcp, cntw, mb, nzv, n_screen, ne, tr_a, tr_b)
            sel = top_order(oe, m_chk)
            bi, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, oe[sel], orj[sel], oaj[sel], ov[sel], a,
                                                                   b, margin, y, c, QT, var_of, tptr, tabv)
            if bi >= 0:
                rj = orj[sel[bi]]
                aj = oaj[sel[bi]]
                v = ov[sel[bi]]
        if bi < 0 and pairs and swaps and cntS >= 2 and L / N <= gate:
            # pair moves: two support points change by one each, exact on rows grouped by (x_S, y)
            n = s.shape[0]
            XS = np.empty((n, cntS))
            hh = np.empty(n)
            for i in range(n):
                hh[i] = y[i] * hr[cntS]
            for q in range(cntS):
                j = S[q]
                vv = var_of[j]
                base = tptr[j]
                for i in range(n):
                    XS[i, q] = tabv[base + QT[vv, i]]
                    hh[i] += XS[i, q] * hr[q]
            o = np.argsort(hh)
            first = np.empty(n, np.int64)
            gcw = np.zeros(n)
            ng = 0
            for t in range(n):
                i = o[t]
                if t == 0 or hh[i] != hh[o[t - 1]]:
                    first[ng] = i
                    ng += 1
                gcw[ng - 1] += c[i]
            first = first[:ng]
            wS = np.empty(cntS)
            for q in range(cntS):
                wS[q] = w[S[q]]
            Lp, p1, p2, d1, d2 = pair_moves(s[first], np.ascontiguousarray(XS[first]), wS, y[first], gcw[:ng], a, b,
                                            L, float(abs(allv[0])), False)
            if p1 >= 0:
                w[S[p1]] += d1
                w[S[p2]] += d2
                s = s + d1 * XS[:, p1] + d2 * XS[:, p2]
                inv, sv, Wp, Wn = bin_scores(s, y, c)
                L, a, b = calibrate_bins(sv, Wp, Wn, a, b)
                continue
        if bi < 0:
            break
        if rj >= 0:
            w[rj] = 0.0
        w[aj] = v
        s, L, a, b, inv, sv, Wp, Wn = s2, L2, a2, b2, inv2, sv2, Wp2, Wn2
    return L / N, w


@nb.njit(cache=False)
def pair_removal_losses(s, w, S, y, c, a, b, QT, var_of, tptr, tabv):
    """Calibrated loss after removing each pair of support columns (Newton from the incumbent's (a, b))."""
    n = s.shape[0]
    k = S.shape[0]
    X = np.empty((k, n))
    for q in range(k):
        j = S[q]
        v = var_of[j]
        base = tptr[j]
        for i in range(n):
            X[q, i] = w[j] * tabv[base + QT[v, i]]
    out = np.full((k, k), np.inf)
    s2 = np.empty(n)
    for p in range(k):
        for q in range(p + 1, k):
            lo = np.inf
            hi = -np.inf
            for i in range(n):
                s2[i] = s[i] - X[p, i] - X[q, i]
                lo = min(lo, s2[i])
                hi = max(hi, s2[i])
            if hi > lo:
                inv, sv, Wp, Wn = bin_scores(s2, y, c)
                out[p, q] = calibrate_bins(sv, Wp, Wn, a, b)[0]
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

    def eval(self, D, cols, deltas, a, b, n_screen=None, GH=None):
        """deltas: one row of score changes per column, or one row shared by all columns."""
        shared = deltas.ndim == 1
        if shared:
            deltas = np.ascontiguousarray(np.broadcast_to(deltas, (len(cols), len(deltas))))
        if n_screen is not None and len(cols) > n_screen:
            # rank columns by a second-order model at fixed (a, b) (no exp per row), evaluate the best
            if GH is None:
                r_row, h_row = self.screen_rows(D)
                G12 = D.colsum(np.stack([r_row, h_row], 1), np.array([False, True]))
                GH = (G12[cols, 0], G12[cols, 1])
            G, H = GH
            if shared:
                pick = screen_pick(np.ascontiguousarray(G), np.ascontiguousarray(H), a, deltas[0], n_screen)
            else:
                u = a * deltas
                sc = np.min(np.minimum(0.0, u * G[:, None] + 0.5 * (u * u) * H[:, None]), axis=1)
                pick = np.sort(np.argsort(sc)[:n_screen])
            out = np.full((2,) + deltas.shape, np.inf)
            out[:, pick] = self.eval(D, cols[pick], deltas[pick], a, b)
            return out
        out = np.empty((2,) + deltas.shape)  # [estimated calibrated loss, loss at fixed (a, b)]
        cols = np.asarray(cols, dtype=np.int64)
        o = np.argsort(D.var_of[cols], kind="stable")
        eval_vars(cols[o], o.astype(np.int64), deltas, self.inv, self.sv.shape[0], self.sv, self.lp, self.ln,
                  self.pp, self.pn, self.wq, self.T, a, b, D.y, D.c, D.QT, D.var_of, D.lev_of, D.kind, D.ncode,
                  D.tptr, D.tabv, out, TR_A, TR_B, D.vrp, D.vrows)
        return out


class ILS:
    """Best-improvement local search over integer points, scored by the calibrated loss."""

    def __init__(self, D, k, n_exact=2, n_screen=4, n_rank=1):
        self.D, self.k, self.n_exact, self.n_screen = D, k, n_exact, n_screen
        self.visited = set()
        self.n_rank = n_rank
        self.est_margin = 1e-4
        self.nevals = 0
        self.hr = np.random.default_rng(3).standard_normal(k + 1)
        self.hw = np.random.default_rng(11).standard_normal(D.d)  # hash of a point: its random projection
        self.vis = Dict.empty(key_type=nb.types.float64, value_type=nb.types.boolean)

    def run(self, w, max_iter=100, deadline=np.inf, swaps=True, gate=np.inf):
        D, k = self.D, self.k
        w = w.astype(np.float64).copy()
        st = ScoreState(D, D.score(w))
        allv = np.arange(-COEF_BOUND, COEF_BOUND + 1, dtype=np.float64)
        nzv = allv[allv != 0]
        return ils_kernel(w, st.s, st.L, st.a, st.b, st.inv, st.sv, st.Wp, st.Wn, self.vis, self.hw, k, max_iter,
                          swaps, float(gate), D.N, self.n_exact, self.n_screen, ADDSCREEN, self.est_margin,
                          MS_BAND * D.msup, D.y, D.c, D.QT, D.var_of, D.lev_of, D.kind, D.ncode, D.tptr, D.tabv,
                          D.vrp, D.vrows, D.vcols, D.vcp, D.cntw, D.valid, allv, nzv, TR_A, TR_B, PAIRS > 0,
                          self.hr)


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, n_starts=5):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.n_starts = n_starts

    def fit(self, X, y):
        tm = [time.perf_counter()]
        t0 = tm[0]
        D = Data(X, y)
        tm.append(time.perf_counter())
        parents = beam_search(D, self.k, self.parent_size, self.child_size, t0 + 0.4 * self.time_limit)
        tm.append(time.perf_counter())
        seen = set()
        starts = []
        best_w, best_l = np.zeros(D.d), np.inf
        hv = np.random.default_rng(7).standard_normal(D.n)
        for nd in parents:
            w, l = calib_round(D, nd)
            key = round(float(hv @ D.score(w)), 6)  # equivalent points (duplicate columns) give one start
            if key in seen or not np.isfinite(l):
                continue
            seen.add(key)
            starts.append((l, w))
            if l < best_l:
                best_l, best_w = l, w
        tm.append(time.perf_counter())
        # integer local search from the best few distinct rounded solutions
        ils = ILS(D, self.k, n_exact=NEXACT, n_screen=NSCREEN)
        starts.sort(key=lambda t: t[0])
        self.start_losses_ = []
        for l0, w0 in starts[:self.n_starts]:
            if time.perf_counter() > t0 + 0.8 * self.time_limit:
                break
            l, w = ils.run(w0, deadline=t0 + 0.9 * self.time_limit)
            self.start_losses_.append((l0, l))
            if l < best_l:
                best_l, best_w = l, w
        if KICKS > 0 and np.isfinite(best_l):
            fails = 0
            tried = {}
            while fails < KICKS and time.perf_counter() < t0 + 0.8 * self.time_limit:
                S = np.flatnonzero(best_w)
                if len(S) == 0:
                    break
                w = best_w.copy()
                # pairs of support columns, cheapest joint removal first, each once per incumbent
                key0 = best_w.tobytes()
                if key0 not in tried:
                    st0 = ScoreState(D, D.score(best_w))
                    PL = pair_removal_losses(st0.s, best_w, S.astype(np.int64), D.y, D.c, st0.a, st0.b, D.QT,
                                             D.var_of, D.tptr, D.tabv)
                    prs = [(PL[p, q], p, q) for p in range(len(S)) for q in range(p + 1, len(S))]
                    prs.sort()
                    tried[key0] = [(S[p], S[q]) for _, p, q in prs]
                if not tried[key0]:
                    break
                p, q = tried[key0].pop(0)
                w[p] = 0
                w[q] = 0
                # swap phases only while the descent is within QK of the best loss
                l, w = ils.run(w, deadline=t0 + 0.9 * self.time_limit, gate=best_l * (1 + QK) if QK >= 0 else np.inf)
                if l < best_l - 1e-12:
                    best_l, best_w, fails = l, w, 0
                else:
                    fails += 1
        tm.append(time.perf_counter())
        self.timing_ = np.diff(tm)  # data, beam, rounding, ILS
        self.coef_ = np.clip(np.round(best_w), -COEF_BOUND, COEF_BOUND)
        self.intercept_, self.multiplier_ = 0.0, 1.0
        self.train_loss_ = best_l
        return self


def make_model(k, time_limit):
    return SparseIntegerClassifier(k=k, time_limit=time_limit, parent_size=PARENT, child_size=CHILD, n_starts=NSTARTS)


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
