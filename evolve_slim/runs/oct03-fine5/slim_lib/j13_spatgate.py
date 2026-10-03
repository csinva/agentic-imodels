"""Fine track: sparse integer risk scores when every numeric column is split at its 99 percentiles
(up to ~2,400 nested threshold features). The file the agent edits.

Pipeline (see notes.md in the run folder for the history and the measured effect of each part):

0. data as variables: binary columns are grouped into chains of nested sets (found from the data by
   subset tests on bitsets), and each chain becomes one ordinal code per row (the number of chain columns
   that contain the row; a byte per code when every chain is short), so column j of the chain is
   x_j = [code >= level_j]. Every other column is its own variable with the rank codes of its values. All
   per-column statistics come from per-code histograms (suffix sums for chains);
1. unique (codes, y) rows with counts (radix sort of a row hash);
2. continuous beam search (FasterRisk's: grow the support one column at a time from the 10 best parents;
   every column's gain bounded by one Newton step, 15 exact child fits per level, at most one child per
   multiset of variables), one numba kernel per beam; minimum support: every indicator keeps >= sqrt(N)
   weighted rows on each side;
3. rounding: each of the 5 best final supports rounded at 20 scales, the rounding with the smallest
   calibrated loss (the harness's criterion: min over a, b of the log loss of a * score + b) is a start;
4. integer local search from the starts (in order of rounded loss; stop when two starts in a row do not
   improve the best): value changes and additions, then threshold slides (a support threshold
   moves up to 8 levels along its chain, same points), then swaps; moves ranked by an estimate of the
   calibrated loss on (score bin, code) cells, the best checked exactly;
5. refit swaps: the 4 best swaps at the best point, each followed by a continuous refit of its support and a
   calibrated rounding; a local search from the best rounding.

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

MODEL_NAME = "j13_spatgate"
DESCRIPTION = ('j9 + start patience 1 in gated problems (logit range >= 10)')
INTEGER = True
COEF_BOUND = 5
TR_A, TR_B = 1.0, 2.0  # trust region of the (a, b) Newton step in the move estimate
FINAL_POOL = int(os.environ.get("FINAL_POOL", "5"))
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
MS_BAND = float(os.environ.get("MS_BAND", "0"))
MS_ENDM = float(os.environ.get("MS_ENDM", "1"))
MS_CHAIN = int(os.environ.get("MS_CHAIN", "0"))
BTOL = float(os.environ.get("BTOL", "1e-4"))
ADDSCREEN = int(os.environ.get("ADDSCREEN", "6"))
LASTSCREEN = float(os.environ.get("LASTSCREEN", "4"))
DEBRUIJN = np.array([0, 1, 56, 2, 57, 49, 28, 3, 61, 58, 42, 50, 38, 29, 17, 4, 62, 47, 59, 36, 45, 43, 51, 22, 53, 39, 33, 30, 24, 18, 12, 5, 63, 55, 48, 27, 60, 41, 37, 16, 46, 35, 44, 21, 52, 32, 23, 11, 54, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], np.int64)  # bit index of 2^i by a de Bruijn product
SLIDES = int(os.environ.get("SLIDES", "2"))
SLIDER = int(os.environ.get("SLIDER", "6"))
RSWAP = int(os.environ.get("RSWAP", "4"))
RSF = int(os.environ.get("RSF", "1"))
RAW_DENSE = 32  # a non-binary column with at most this many codes is histogrammed, otherwise handled per row

# Fixed pseudo-random streams (data independent constants, drawn once at import): prefixes of
# default_rng(seed).standard_normal / .integers, so that a fit does not construct generators.
RSQ = float(os.environ.get("RSQ", "0"))
RECAL = int(os.environ.get("RECAL", "0"))
POLISH = int(os.environ.get("POLISH", "3"))
POLM = int(os.environ.get("POLM", "8"))
POLVAL = int(os.environ.get("POLVAL", "1"))
EXV = int(os.environ.get("EXV", "0"))
RCFB = int(os.environ.get("RCFB", "0"))
VARDP = int(os.environ.get("VARDP", "0"))
POLNS = int(os.environ.get("POLNS", "32"))
POLNE = int(os.environ.get("POLNE", "2"))
POLX = int(os.environ.get("POLX", "1"))
POLV = int(os.environ.get("POLV", "2"))
POLVV = int(os.environ.get("POLVV", "3"))
POLOLD = int(os.environ.get("POLOLD", "0"))
POLG = float(os.environ.get("POLG", "10"))
POLALL = int(os.environ.get("POLALL", "1"))
RG = float(os.environ.get("RG", "0"))
REJ = float(os.environ.get("REJ", "2"))
RSFG = int(os.environ.get("RSFG", "0"))
SPATG = int(os.environ.get("SPATG", "1"))
CHKFIRST = int(os.environ.get("CHKFIRST", "1"))
MULTI = int(os.environ.get("MULTI", "1"))
SPAT = int(os.environ.get("SPAT", "2"))
SPEPS = float(os.environ.get("SPEPS", "0.001"))
_RCAP = 1 << 16
_RNORM = {sd: np.random.default_rng(sd).standard_normal(_RCAP) for sd in (3, 7, 11, 12345)}
_RINT5 = np.random.default_rng(5).integers(0, 2 ** 63, _RCAP, dtype=np.int64)


def _normals(seed, m):
    return _RNORM[seed][:m] if m <= _RCAP else np.random.default_rng(seed).standard_normal(m)


def _ints5(m):
    return _RINT5[:m] if m <= _RCAP else np.random.default_rng(5).integers(0, 2 ** 63, m, dtype=np.int64)


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


@nb.njit(cache=False, inline="always")
def _lrow_e(m):
    """(log(1 + exp(-m)), exp(-|m|)): the second gives sigma(-m) without another exp (see _sig_e)."""
    if m > 0:
        e = np.exp(-m)
        return np.log1p(e), e
    e = np.exp(m)
    return -m + np.log1p(e), e


@nb.njit(cache=False, inline="always")
def _sig_e(m, e):
    """sigma(-m) from e = exp(-|m|), bitwise equal to _sig(-m)."""
    if m <= 0:
        return 1.0 / (1.0 + e)
    return e / (1.0 + e)


@nb.njit(cache=False)
def _new_dict_u64():
    """An empty typed dict made inside numba (much cheaper than Dict.empty from Python)."""
    return Dict.empty(key_type=nb.types.uint64, value_type=nb.types.boolean)


@nb.njit(cache=False)
def _new_dict_f64():
    return Dict.empty(key_type=nb.types.float64, value_type=nb.types.boolean)


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
    nonzeros (counts by popcount of the bitsets)."""
    n, d = X.shape
    nw = (n + 63) // 64
    B = np.zeros((nw, d), np.uint64)
    bad = np.zeros(d, np.bool_)
    for w in range(nw):
        Bw = B[w]
        for i in range(w * 64, min(n, w * 64 + 64)):
            sh = np.uint64(i & 63)
            row = X[i]
            for j in range(d):
                v = row[j]
                Bw[j] |= np.uint64(v != 0.0) << sh
                bad[j] |= (v != 0.0) & (v != 1.0)
    cnt = np.zeros(d, np.int64)
    for w in range(nw):
        for j in range(d):
            x = B[w, j]
            x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
            x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
            x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            cnt[j] += np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))
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
def chain_codes(B, n, chain, lev, nch, Q):
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
    nw = B.shape[0]
    for ch in range(nch):
        # rows in level l but not in level l + 1 have code l (levels are nested): visit the set bits of the
        # differences, so each row is written once
        m = length[ch]
        base = cp[ch]
        for w in range(nw):
            nxt = np.uint64(0)
            for l in range(m - 1, -1, -1):
                cur = B[w, cols[base + l]]
                x = cur & ~nxt
                nxt = cur
                while x != np.uint64(0):
                    low = x & (~x + np.uint64(1))
                    i = w * 64 + DEBRUIJN[(low * np.uint64(0x03F79D71B4CA8B09)) >> np.uint64(58)]
                    Q[ch, i] = l + 1
                    x ^= low
    return Q
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
    vrows = np.empty(vrp[V], np.int32)
    for v in range(V):
        t = vrp[v]
        for i in range(n):
            if kind[v] > 0 or QT[v, i] > 0:
                vrows[t] = i
                t += 1
    return vrp, vrows


@nb.njit(cache=False)
def row_hash(Q0, r, ys0):
    V, n = Q0.shape
    h = ys0 * r[V]
    for v in range(V):
        rv = r[v]
        for i in range(n):
            h[i] += rv * Q0[v, i]
    return h


@nb.njit(cache=False)
def unique_rows(h):
    """np.unique(h, return_index=True) with the counts: first occurrence of every distinct value, in value order,
    and the number of rows with that value. A stable LSD radix sort on the order-preserving bit pattern of h
    gives the same order as numpy's stable sort."""
    n = h.shape[0]
    key = np.empty(n, np.uint64)
    hb = h.view(np.uint64)
    top = np.uint64(1) << np.uint64(63)
    for i in range(n):
        u = hb[i]
        key[i] = ~u if (u & top) else (u | top)  # IEEE order -> unsigned order
    o = np.arange(n)
    o2 = np.empty(n, np.int64)
    cnt = np.empty(257, np.int64)
    for sh in range(0, 64, 8):
        cnt[:] = 0
        for t in range(n):
            cnt[((key[o[t]] >> np.uint64(sh)) & np.uint64(255)) + 1] += 1
        if cnt[1:].max() == n:
            continue  # all keys share this byte
        for b in range(256):
            cnt[b + 1] += cnt[b]
        for t in range(n):
            bb = (key[o[t]] >> np.uint64(sh)) & np.uint64(255)
            o2[cnt[bb]] = o[t]
            cnt[bb] += 1
        o, o2 = o2, o
    first = np.empty(n, np.int64)
    cw = np.zeros(n)
    m = -1
    for t in range(n):
        i = o[t]
        if t == 0 or h[i] != h[o[t - 1]]:
            m += 1
            first[m] = i
        cw[m] += 1.0
    m += 1
    return first[:m].copy(), cw[:m].copy()


@nb.njit(cache=False)
def _max_at(a, idx, v):
    """np.maximum.at(a, idx, v) (ufunc.at is slow)."""
    for t in range(idx.shape[0]):
        if v[t] > a[idx[t]]:
            a[idx[t]] = v[t]


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
        # codes in a byte when every variable is a chain of at most 255 columns (less memory in every row pass)
        small = len(others) == 0 and (lev.max() if len(lev) else 0) <= 255
        Q0 = np.zeros((V, n0), np.uint8 if small else np.int32)
        chain_codes(B, n0, chain, lev, nch, Q0)
        for ch in range(nch):
            ncode[ch] = 0
        _max_at(ncode, chain[cand], lev[cand])
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
        r = _normals(12345, V + 1)
        h = row_hash(Q0, r, ys0)
        first, self.c = unique_rows(h)
        self.QT = gather_cols(Q0, first)
        if self.QT.dtype != np.uint8 and ncode.max() <= 256:
            self.QT = self.QT.astype(np.uint8)  # codes in a byte: less memory traffic in every row pass
        self.y = np.ascontiguousarray(ys0[first])
        self.n = self.QT.shape[1]
        self.N = float(self.c.sum())
        self.yc = self.y * self.c
        self.V, self.kind, self.ncode, self.var_of, self.lev_of = V, kind, ncode, var_of, lev_of
        self.tptr, self.tabv = tptr, tabv
        vorder = np.lexsort((lev_of, var_of))
        self.vcols = vorder.astype(np.int64)
        self.vcp = np.zeros(V + 1, np.int64)
        self.vcp[1:] = np.cumsum(np.bincount(var_of, minlength=V))
        self.nchains = nch
        self.allbin = len(others) == 0 or bool(np.all((tabv == 0.0) | (tabv == 1.0)))  # chain tables are 0 / 1
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
def chol_solve_buf(A, bb, nf, Lm, x):
    """chol_solve on the leading nf x nf block of A, with caller buffers Lm, x (x returned in x[:nf])."""
    for i in range(nf):
        for j in range(i + 1):
            sm = A[i, j]
            for t in range(j):
                sm -= Lm[i, t] * Lm[j, t]
            if i == j:
                if sm <= 0.0:
                    sol = np.linalg.solve(np.ascontiguousarray(A[:nf, :nf]), bb[:nf].copy())
                    for u in range(nf):
                        x[u] = sol[u]
                    return
                Lm[i, i] = np.sqrt(sm)
            else:
                Lm[i, j] = sm / Lm[j, j]
    for i in range(nf):
        x[i] = bb[i]
    for i in range(nf):
        sm = x[i]
        for t in range(i):
            sm -= Lm[i, t] * x[t]
        x[i] = sm / Lm[i, i]
    for i in range(nf - 1, -1, -1):
        sm = x[i]
        for t in range(i + 1, nf):
            sm -= Lm[t, i] * x[t]
        x[i] = sm / Lm[i, i]


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
    E = np.empty(n)
    En = np.empty(n)
    cur = 0.0
    for i in range(n):
        lv, E[i] = _lrow_e(m[i])
        cur += c[i] * lv
    g = np.empty(p)
    H = np.empty((p, p))
    wn = np.empty(p)
    mn = np.empty(n)
    fi = np.empty(p, np.int64)
    d = np.empty(p)
    A = np.empty((p, p))
    bb = np.empty(p)
    Lm = np.empty((p, p))
    sol = np.empty(p)
    for _ in range(maxit):
        g[:] = 0.0
        H[:, :] = 0.0
        for i in range(n):
            pr = _sig_e(m[i], E[i])
            gi = -c[i] * pr
            hi_ = c[i] * pr * (1.0 - pr)
            for q in range(p):
                zq = Z[i, q]
                g[q] += gi * zq
                hz = hi_ * zq
                for r in range(q, p):
                    H[q, r] += hz * Z[i, r]
        nf = 0
        for q in range(p):
            d[q] = 0.0
            if not ((w[q] <= lo[q] + 1e-12 and g[q] > 0) or (w[q] >= hi[q] - 1e-12 and g[q] < 0)):
                fi[nf] = q
                nf += 1
        if nf > 0:
            for a in range(nf):
                bb[a] = -g[fi[a]]
                for b in range(nf):
                    qa, qb = fi[a], fi[b]
                    A[a, b] = H[min(qa, qb), max(qa, qb)]
                A[a, a] += 1e-10 * (1.0 + A[a, a])
            chol_solve_buf(A, bb, nf, Lm, sol)
            for a in range(nf):
                d[fi[a]] = sol[a]
        t = 1.0
        new = cur
        ok = False
        while t > 1e-8:
            for q in range(p):
                v = w[q] + t * d[q]
                wn[q] = min(max(v, lo[q]), hi[q])
            new = 0.0
            for i in range(n):
                s = 0.0
                for q in range(p):
                    s += Z[i, q] * wn[q]
                mn[i] = s
                lv, En[i] = _lrow_e(s)
                new += c[i] * lv
            if new <= cur:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        dec = cur - new
        w[:] = wn
        m, mn = mn, m
        E, En = En, E
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
    S_buf = np.empty(ps + 1, np.int64)
    w_buf = np.zeros(ps + 2)
    Z_buf = np.empty((0, 1))
    cw_buf = np.empty(0)
    lo_b = np.full(ps + 2, -bound)
    hi_b = np.full(ps + 2, bound)
    lo_b[0] = -1e300
    hi_b[0] = 1e300
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
            S = S_buf
            w = w_buf
            w[:] = 0.0
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
            if Z_buf.shape[0] < maxcell or Z_buf.shape[1] != ncol:
                Z_buf = np.empty((maxcell, ncol))
                cw_buf = np.empty(maxcell)
            Z = Z_buf
            cw = cw_buf
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
                loss, _ = newton_fit(Z[:nce], cw[:nce], w, lo_b, hi_b, 50, tol)
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
        # binary column: the parent's groups split by x_j in one pass (groups numbered by first appearance)
        l = lev_of[j]
        table = np.full(2 * par_ng, -1, np.int64)
        inv = np.empty(n, np.int64)
        gy = np.empty(n)
        gc = np.zeros(n)
        rep = np.empty(n, np.int64)
        ng = 0
        for i in range(n):
            key = 2 * par_inv[i] + (1 if QT[v, i] >= l else 0)
            g = table[key]
            if g < 0:
                g = ng
                table[key] = g
                rep[g] = i
                gy[g] = y[i]
                ng += 1
            inv[i] = g
            gc[g] += c[i]
        gy = gy[:ng].copy()
        gc = gc[:ng].copy()
        rep = rep[:ng].copy()
        Z = group_design(QT, var_of, tptr, tabv, gy, rep, S)
        mg = np.zeros(ng)
        for g in range(ng):
            for q in range(Z.shape[1]):
                mg[g] += Z[g, q] * w[q]
        return inv, ng, gy, gc, rep, mg
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
def beam_rows_pr2(Pr, yc, c, third):
    """beam_rows from sigma(-m) per row; the x^2 block only when `third`; also the column sums of the curvature."""
    n, P = Pr.shape
    R = np.empty((n, (3 if third else 2) * P))
    h00 = np.zeros(P)
    for i in range(n):
        for q in range(P):
            pr = Pr[i, q]
            R[i, q] = yc[i] * pr
            h = c[i] * pr * (1.0 - pr)
            R[i, P + q] = h
            h00[q] += h
            if third:
                R[i, 2 * P + q] = h
    return R, h00


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


@nb.njit(cache=False)
def beam_level(PINV, PNG, GOFF, GY, GC, GREP, GMG, PS, plen, PW, ploss, phash, psig, QT, var_of, lev_of, kind,
               ncode, vcp, vcols, tptr, tabv, vrp, vrows, cntw, bvalid, y, c, yc, allbin, colh, varh, seen, V,
               child_size, nfit, keep, sigmax, bound, tol, mb):
    """One beam level on parents stored as arrays (row groups concatenated, offsets GOFF): Newton-gain screen of
    every column for every parent, proposals, exact child fits, sort by loss, at most sigmax children per
    signature, the keep best materialised as the next parents. Returns ok = False when nothing is proposed."""
    P = PNG.shape[0]
    n = PINV.shape[1]
    d = var_of.shape[0]
    Pr = np.empty((n, P))
    for q in range(P):
        off = GOFF[q]
        sg = np.empty(PNG[q])
        for g in range(PNG[q]):
            sg[g] = _sig(-GMG[off + g])
        for i in range(n):
            Pr[i, q] = sg[PINV[q, i]]
    R, h00 = beam_rows_pr2(Pr, yc, c, not allbin)
    if allbin:  # x^2 = x: the third block equals the second
        G2 = colsum(QT, R, kind, ncode, vcp, vcols, lev_of, tptr, tabv, np.zeros(2 * P, np.bool_), vrp, vrows)
        Gm = np.empty((d, 3 * P))
        Gm[:, :2 * P] = G2
        Gm[:, 2 * P:] = G2[:, P:]
    else:
        pw = np.zeros(3 * P, np.bool_)
        pw[2 * P:] = True
        Gm = colsum(QT, R, kind, ncode, vcp, vcols, lev_of, tptr, tabv, pw, vrp, vrows)
    GA, BA = newton_gains(Gm, h00, bvalid, bound)
    eq, ej, eest, eb, eh, es = beam_proposals(GA, BA, ploss, PS, plen, phash, psig, colh, varh, var_of, V, kind,
                                              vcols, vcp, cntw, mb, seen, child_size, nfit, sigmax)
    m = eq.shape[0]
    lmax = PS.shape[1]
    if m == 0:
        return (False, PINV, PNG, GOFF, GY, GC, GREP, GMG, PS, plen, PW, ploss, phash, psig)
    # exact child fits, in the order: parents ascending; per parent its binary / small-code columns sorted by
    # variable (stable), then the others
    closs = np.empty(m)
    cq = np.empty(m, np.int64)
    cj = np.empty(m, np.int64)
    ch_ = np.empty(m, np.uint64)
    cs_ = np.empty(m, np.uint64)
    CW = np.zeros((m, lmax + 2))
    CS = np.zeros((m, lmax + 1), np.int64)
    u = 0
    for q in range(P):
        cnt = 0
        for t in range(m):
            if eq[t] == q:
                cnt += 1
        if cnt == 0:
            continue
        lst = np.empty(cnt, np.int64)
        cnt = 0
        for t in range(m):
            if eq[t] == q:
                lst[cnt] = t
                cnt += 1
        off = GOFF[q]
        ng = PNG[q]
        par_S = PS[q, :plen[q]].copy()
        par_w = PW[q, :plen[q] + 1].copy()
        gy = GY[off:off + ng]
        gc = GC[off:off + ng]
        rep = GREP[off:off + ng]
        nsp = 0
        for t in range(cnt):
            if kind[var_of[ej[lst[t]]]] < 2:
                nsp += 1
        if nsp > 0:
            idx = np.empty(nsp, np.int64)
            vv = np.empty(nsp, np.int64)
            nsp = 0
            for t in range(cnt):
                if kind[var_of[ej[lst[t]]]] < 2:
                    idx[nsp] = lst[t]
                    vv[nsp] = var_of[ej[lst[t]]]
                    nsp += 1
            idx = idx[np.argsort(vv, kind="mergesort")]
            js = ej[idx].copy()
            b0s = eb[idx].copy()
            losses, W = child_fit_batch(PINV[q], ng, gy, gc, rep, par_S, par_w, js, QT, var_of, lev_of, kind, ncode,
                                        tptr, tabv, c, bound, tol, False, vrp, vrows, b0s)
            for t in range(nsp):
                closs[u] = losses[t]
                cq[u] = q
                cj[u] = js[t]
                ch_[u] = eh[idx[t]]
                cs_[u] = es[idx[t]]
                CW[u, :plen[q] + 2] = W[t]
                u += 1
        for t in range(cnt):
            tt = lst[t]
            j = ej[tt]
            if kind[var_of[j]] < 2:
                continue
            v = var_of[j]
            code = np.empty(n, np.int64)
            for i in range(n):
                code[i] = QT[v, i]
            S, inv_, ng_, gy_, gc_, rep_, loss, mg_, w = make_child(PINV[q], ng, par_S, par_w, j, code, ncode[v], y,
                                                                    c, QT, var_of, tptr, tabv, bound, tol)
            closs[u] = loss
            cq[u] = q
            cj[u] = j
            ch_[u] = eh[tt]
            cs_[u] = es[tt]
            CW[u, :plen[q] + 2] = w
            u += 1
    # supports of the children (sorted)
    for t in range(m):
        q = cq[t]
        ps = plen[q]
        j = cj[t]
        r = 0
        ins = False
        for z in range(ps + 1):
            if not ins and (r >= ps or j < PS[q, r]):
                CS[t, z] = j
                ins = True
            else:
                CS[t, z] = PS[q, r]
                r += 1
    o = np.argsort(closs, kind="mergesort")
    sel = np.empty(min(m, keep), np.int64)
    ns = 0
    if sigmax > 0:
        sc = Dict.empty(key_type=nb.types.uint64, value_type=nb.types.int64)
        for t in range(m):
            if ns >= keep:
                break
            i = o[t]
            c0 = sc.get(cs_[i], 0)
            if c0 < sigmax:
                sc[cs_[i]] = c0 + 1
                sel[ns] = i
                ns += 1
    else:
        for t in range(min(m, keep)):
            sel[t] = o[t]
        ns = min(m, keep)
    sel = sel[:ns]
    # materialise the kept children as the next parents
    P2 = ns
    PINV2 = np.empty((P2, n), np.int64)
    PNG2 = np.empty(P2, np.int64)
    GOFF2 = np.zeros(P2 + 1, np.int64)
    PS2 = np.zeros((P2, lmax + 1), np.int64)
    plen2 = np.empty(P2, np.int64)
    PW2 = np.zeros((P2, lmax + 2))
    ploss2 = np.empty(P2)
    phash2 = np.empty(P2, np.uint64)
    psig2 = np.empty(P2, np.uint64)
    parts = List()
    for t in range(P2):
        i = sel[t]
        q = cq[i]
        ps = plen[q] + 1
        S = CS[i, :ps].copy()
        w = CW[i, :ps + 1].copy()
        inv_, ng_, gy_, gc_, rep_, mg_ = materialise_kernel(PINV[q], PNG[q], cj[i], S, w, QT, var_of, kind, lev_of,
                                                            ncode, tptr, tabv, y, c)
        PINV2[t] = inv_
        PNG2[t] = ng_
        GOFF2[t + 1] = GOFF2[t] + ng_
        PS2[t, :ps] = S
        plen2[t] = ps
        PW2[t, :ps + 1] = w
        ploss2[t] = closs[i]
        phash2[t] = ch_[i]
        psig2[t] = cs_[i]
        parts.append((gy_, gc_, rep_, mg_))
    tot = GOFF2[P2]
    GY2 = np.empty(tot)
    GC2 = np.empty(tot)
    GREP2 = np.empty(tot, np.int64)
    GMG2 = np.empty(tot)
    for t in range(P2):
        gy_, gc_, rep_, mg_ = parts[t]
        a0 = GOFF2[t]
        GY2[a0:a0 + PNG2[t]] = gy_
        GC2[a0:a0 + PNG2[t]] = gc_
        GREP2[a0:a0 + PNG2[t]] = rep_
        GMG2[a0:a0 + PNG2[t]] = mg_
    return (True, PINV2, PNG2, GOFF2, GY2, GC2, GREP2, GMG2, PS2, plen2, PW2, ploss2, phash2, psig2)


@nb.njit(cache=False)
def beam_levels(st, lev0, nlev, QT, var_of, lev_of, kind, ncode, vcp, vcols, tptr, tabv, vrp, vrows, cntw, bvalid, y,
                c, yc, allbin, colh, varh, seen, V, child_size, parent_size, final_pool, screen, lastscreen, sigmax,
                bound, tol, mb):
    """Beam levels lev0 .. nlev - 1 in one kernel (same steps as the per-level loop of beam_search)."""
    for lev in range(lev0, nlev):
        last = lev == nlev - 1
        keep = final_pool if last else parent_size
        nfit = int((lastscreen if last else screen) * keep)
        w_ = max(lev, 1)
        res = beam_level(st[0], st[1], st[2], st[3], st[4], st[5], st[6], np.ascontiguousarray(st[7][:, :w_]),
                         st[8], np.ascontiguousarray(st[9][:, :w_ + 1]), st[10], st[11], st[12], QT, var_of, lev_of,
                         kind, ncode, vcp, vcols, tptr, tabv, vrp, vrows, cntw, bvalid, y, c, yc, allbin, colh, varh,
                         seen, V, child_size, nfit, keep, sigmax, bound, tol, mb)
        if not res[0]:
            break
        st = (res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12],
              res[13])
    return st


def beam_search(D, k, parent_size=10, child_size=10, deadline=np.inf):
    """Beam search over supports, one numba kernel per level (beam_level). Every column's gain for every parent is
    bounded by one Newton step on the new coefficient (from per-code histograms of the gradient and curvature
    rows); each parent proposes its child_size best variables (each at its best column), and only the SCREEN *
    beam best proposals overall are fitted exactly."""
    hi5 = _ints5(D.d + D.V)
    colh = hi5[:D.d].astype(np.uint64) * np.uint64(2) + np.uint64(1)
    varh = hi5[D.d:].astype(np.uint64) * np.uint64(2) + np.uint64(1)
    inv, ng, gy, gc, rep = regroup(np.zeros(D.n, np.int64), 1, (D.y > 0).astype(np.int64), 2, D.y, D.c)
    npos = D.c[D.y > 0].sum()
    w0 = np.log(npos / (D.N - npos))
    mg = gy * w0
    lmax = max(k, 1)
    st = (inv[None, :].copy(), np.array([ng], np.int64), np.array([0, ng], np.int64), gy, gc, rep, mg,
          np.zeros((1, lmax), np.int64), np.zeros(1, np.int64), np.full((1, lmax + 1), w0),
          np.array([total_loss(mg, gc)]), np.zeros(1, np.uint64), np.zeros(1, np.uint64))
    seen = _new_dict_u64()
    bound = float(COEF_BOUND)
    nlev = min(k, int(D.valid.sum()))
    t_start = time.perf_counter()
    for lev in range(nlev):
        if lev == 1:
            # the remaining levels in one kernel when they surely fit in the time left (level 0, one parent, took
            # dt; a level of the full beam costs at most ~parent_size times that)
            dt = time.perf_counter() - t_start
            if t_start + 2.0 * parent_size * nlev * max(dt, 1e-3) < deadline:
                return beam_levels(st, 1, nlev, D.QT, D.var_of, D.lev_of, D.kind, D.ncode, D.vcp, D.vcols, D.tptr,
                                   D.tabv, D.vrp, D.vrows, D.cntw, D.bvalid, D.y, D.c, D.yc, D.allbin, colh, varh,
                                   seen, D.V, child_size, parent_size, FINAL_POOL, SCREEN, LASTSCREEN, SIGMAX, bound,
                                   BTOL, MS_BAND * D.msup)
        last = lev == nlev - 1
        if time.perf_counter() > deadline and st[1].shape[0] > 1:
            # out of time: finish the support greedily from the best parent
            ng0 = st[1][0]
            st = (st[0][:1], st[1][:1], st[2][:2], st[3][:ng0], st[4][:ng0], st[5][:ng0], st[6][:ng0], st[7][:1],
                  st[8][:1], st[9][:1], st[10][:1], st[11][:1], st[12][:1])
        keep = FINAL_POOL if last else parent_size
        nfit = int((LASTSCREEN if last else SCREEN) * keep)
        res = beam_level(st[0], st[1], st[2], st[3], st[4], st[5], st[6], np.ascontiguousarray(st[7][:, :max(lev, 1)]),
                         st[8], np.ascontiguousarray(st[9][:, :max(lev, 1) + 1]), st[10], st[11], st[12], D.QT,
                         D.var_of, D.lev_of, D.kind, D.ncode, D.vcp, D.vcols, D.tptr, D.tabv, D.vrp, D.vrows, D.cntw,
                         D.bvalid, D.y, D.c, D.yc, D.allbin, colh, varh, seen, D.V, child_size, nfit, keep, SIGMAX,
                         bound, BTOL, MS_BAND * D.msup)
        if not res[0]:
            break
        st = res[1:]
    return st


# ------------------------------------------------------- calibrated rounding
@nb.njit(cache=False, inline="always")
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


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
    seen_r = np.empty((n_mult, p))
    nseen = 0
    WPc = np.zeros(0)
    WNc = np.zeros(0)
    pres = np.zeros(0, np.bool_)
    svb = np.empty(0)
    Wpb = np.empty(0)
    Wnb = np.empty(0)
    Eb = np.empty(0)
    Enb = np.empty(0)
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
        # a rounding proportional to an earlier one has the same calibrated loss: skip it
        gg = 0
        for q in range(p):
            gg = _gcd(gg, int(abs(r[q])))
        sgn = 0.0
        for q in range(p):
            if r[q] != 0.0:
                sgn = 1.0 if r[q] > 0 else -1.0
                break
        dup = False
        for t2 in range(nseen):
            eq_ = True
            for q in range(p):
                if seen_r[t2, q] != sgn * r[q] / gg:
                    eq_ = False
                    break
            if eq_:
                dup = True
                break
        if dup:
            continue
        for q in range(p):
            seen_r[nseen, q] = sgn * r[q] / gg
        nseen += 1
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
        # integer scores: calibrate on the distinct score values (counted in reused buffers)
        integral = True
        for g in range(ng):
            if sg[g] != np.floor(sg[g]):
                integral = False
                break
        if integral and mx - mn <= 4 * ng + 1024:
            R = int(mx - mn) + 1
            if R > WPc.shape[0]:
                WPc = np.zeros(R)
                WNc = np.zeros(R)
                pres = np.zeros(R, np.bool_)
                svb = np.empty(R)
                Wpb = np.empty(R)
                Wnb = np.empty(R)
                Eb = np.empty(2 * R)
                Enb = np.empty(2 * R)
            for t2 in range(R):
                WPc[t2] = 0.0
                WNc[t2] = 0.0
                pres[t2] = False
            for g in range(ng):
                ix = int(sg[g] - mn)
                pres[ix] = True
                if gy[g] > 0:
                    WPc[ix] += gc[g]
                else:
                    WNc[ix] += gc[g]
            mb_ = 0
            for t2 in range(R):
                if pres[t2]:
                    svb[mb_] = (mn + t2) / sd
                    Wpb[mb_] = WPc[t2]
                    Wnb[mb_] = WNc[t2]
                    mb_ += 1
            loss, _, _ = calibrate_bins_buf(svb[:mb_], Wpb[:mb_], Wnb[:mb_], 0.0, 0.0, Eb, Enb)
        else:
            inv_, sv, Wp, Wn = bin_scores(sg, gy, gc)
            loss, _, _ = calibrate_bins(sv / sd, Wp, Wn, 0.0, 0.0)
        if loss < best_l:
            best_l = loss
            best_r[:] = r
    return best_r, best_l


@nb.njit(cache=False)
def round_all(GOFF, PNG, GY, GC, GREP, PS, plen, PW, QT, var_of, tptr, tabv, n_mult, bound, hv, N):
    """calib_round for every final beam node: rounded points (aligned with PS), calibrated loss / N, and a key of
    the score vector (hv . s) that identifies equivalent points."""
    P = PNG.shape[0]
    R = np.zeros((P, PS.shape[1]))
    L = np.empty(P)
    K = np.empty(P)
    n = QT.shape[1]
    for t in range(P):
        p = plen[t]
        a0 = GOFF[t]
        ng = PNG[t]
        S = PS[t, :p].copy()
        XS = np.empty((ng, p))
        for g in range(ng):
            for q in range(p):
                XS[g, q] = xval(QT, var_of, tptr, tabv, GREP[a0 + g], S[q])
        r, l = calib_round_kernel(XS, GY[a0:a0 + ng], GC[a0:a0 + ng], PW[t, 1:p + 1].copy(), n_mult, bound)
        R[t, :p] = r
        L[t] = l / N
        sc = score_kernel(QT, var_of, tptr, tabv, S, r)
        kk = 0.0
        for i in range(n):
            kk += hv[i] * sc[i]
        K[t] = kk
    return R, L, K


@nb.njit(cache=False)
def refit_round(w, QT, var_of, lev_of, kind, ncode, tptr, tabv, y, c, bound, tol, n_mult):
    """Continuous logistic fit (box [-bound, bound]) on the support of w, rounded by the calibrated loss over a
    grid of scales (as the beam's final nodes are): a move that changes every point at once."""
    d = w.shape[0]
    n = QT.shape[1]
    cnt = 0
    for j in range(d):
        if w[j] != 0.0:
            cnt += 1
    S = np.empty(cnt, np.int64)
    u = 0
    for j in range(d):
        if w[j] != 0.0:
            S[u] = j
            u += 1
    ginv = np.zeros(n, np.int64)
    code = np.empty(n, np.int64)
    for i in range(n):
        code[i] = 1 if y[i] > 0 else 0
    ginv, ng, gy, gc, rep = regroup(ginv, 1, code, 2, y, c)
    for q in range(cnt):
        j = S[q]
        v = var_of[j]
        if kind[v] == 0:
            l = lev_of[j]
            for i in range(n):
                code[i] = 1 if QT[v, i] >= l else 0
            nc = 2
        else:
            for i in range(n):
                code[i] = QT[v, i]
            nc = ncode[v]
        ginv, ng, gy, gc, rep = regroup(ginv, ng, code, nc, y, c)
    Z = group_design(QT, var_of, tptr, tabv, gy, rep, S)
    npos = 0.0
    tot = 0.0
    for g in range(ng):
        tot += gc[g]
        if gy[g] > 0:
            npos += gc[g]
    beta = np.zeros(cnt + 1)
    beta[0] = np.log(npos / (tot - npos))
    lo = np.full(cnt + 1, -bound)
    hi = np.full(cnt + 1, bound)
    lo[0] = -1e300
    hi[0] = 1e300
    newton_fit(Z, gc, beta, lo, hi, 50, tol)
    XS = np.empty((ng, cnt))
    for g in range(ng):
        for q in range(cnt):
            XS[g, q] = Z[g, q + 1] * gy[g]
    r, l = calib_round_kernel(XS, gy, gc, beta[1:].copy(), n_mult, bound)
    out = np.zeros(d)
    for q in range(cnt):
        out[S[q]] = r[q]
    return out, l


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
    """calibrate() on the bins (score sv, weight Wp of y = +1, Wn of y = -1) without building the 2m-row arrays:
    the same terms in the same order (all y = +1 terms, then all y = -1 terms), so the same result."""
    m = sv.shape[0]
    return calibrate_bins_buf(sv, Wp, Wn, a, b, np.empty(2 * m), np.empty(2 * m))


@nb.njit(cache=False)
def calibrate_bins_buf(sv, Wp, Wn, a, b, E, En):
    """calibrate_bins with caller work buffers E, En (length >= 2m)."""
    m = sv.shape[0]
    cur = 0.0
    for i in range(m):
        lv, E[i] = _lrow_e(a * sv[i] + b)
        cur += Wp[i] * lv
    for i in range(m):
        lv, E[m + i] = _lrow_e(-(a * sv[i] + b))
        cur += Wn[i] * lv
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for i in range(m):
            z = a * sv[i] + b
            p = _sig_e(z, E[i])
            w = Wp[i] * p * (1.0 - p)
            gi = -Wp[i] * p
            ga += gi * sv[i]
            gb += gi
            haa += w * sv[i] * sv[i]
            hab += w * sv[i]
            hbb += w
        for i in range(m):
            z = -(a * sv[i] + b)
            p = _sig_e(z, E[m + i])
            w = Wn[i] * p * (1.0 - p)
            gi = Wn[i] * p
            ga += gi * sv[i]
            gb += gi
            haa += w * sv[i] * sv[i]
            hab += w * sv[i]
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
            for i in range(m):
                lv, En[i] = _lrow_e(na * sv[i] + nb_)
                new += Wp[i] * lv
            for i in range(m):
                lv, En[m + i] = _lrow_e(-(na * sv[i] + nb_))
                new += Wn[i] * lv
            if new <= cur - 1e-4 * t * dec:
                break
            t *= 0.5
        if t <= 1e-10:
            break
        a, b = na, nb_
        E, En = En, E
        improv = cur - new
        cur = new
        if improv < 1e-12 * (1.0 + cur):
            break
    return cur, a, b


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
def eval_cells(q, deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b, tlo, TB):
    """out[:, q, r]: estimated calibrated loss and loss at fixed (a, b) after s += deltas[q, r] * x, from the
    touched cells (bin cb, value cx, weights cwp / cwn of y = +1 / -1). The estimate is the exact loss at the
    current (a, b) minus a trust-region Newton step in (a, b)."""
    nd = deltas.shape[1]
    wk = np.zeros((9, nd))  # one allocation for the nine work rows
    stepa = wk[0]
    stepb = wk[1]
    hyb = wk[2]
    dL = wk[3]
    dGa = wk[4]
    dGb = wk[5]
    dHaa = wk[6]
    dHab = wk[7]
    dHbb = wk[8]
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
            tf = sn_ - tlo
            if TB.shape[1] > 0 and tf == np.floor(tf) and tf >= 0.0 and tf < TB.shape[1]:
                # integer scores: the loss terms at (a, b) come from a table over the score range
                ti = int(tf)
                lpn = TB[0, ti]
                lnn = TB[1, ti]
                ppn = TB[2, ti]
                pnn = TB[3, ti]
            else:
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
    # table of the per-score loss terms at (a, b) when scores and moves are integers
    tlo = 0.0
    TB = np.zeros((4, 0))
    integral = True
    lo = np.inf
    hi = -np.inf
    for q in range(nbins):
        if sv[q] != np.floor(sv[q]):
            integral = False
        lo = min(lo, sv[q])
        hi = max(hi, sv[q])
    dlo = 0.0
    dhi = 0.0
    for t in range(m):
        for r in range(deltas.shape[1]):
            dv = deltas[qidx[t], r]
            if dv != np.floor(dv):
                integral = False
            dlo = min(dlo, dv)
            dhi = max(dhi, dv)
    if integral and nbins > 0 and hi - lo + dhi - dlo < 4096:
        tlo = lo + dlo
        R = int(hi + dhi - tlo) + 1
        TB = np.empty((4, R))
        for ti in range(R):
            z = a * (tlo + ti) + b
            e = np.exp(-abs(z))
            l1 = np.log1p(e)
            if z > 0:
                TB[0, ti] = l1
                TB[1, ti] = z + l1
                TB[2, ti] = e / (1.0 + e)
                TB[3, ti] = 1.0 / (1.0 + e)
            else:
                TB[0, ti] = -z + l1
                TB[1, ti] = l1
                TB[2, ti] = 1.0 / (1.0 + e)
                TB[3, ti] = e / (1.0 + e)
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
            eval_cells(qidx[t], deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b, tlo, TB)
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


@nb.njit(cache=False, inline="always")
def _qbest(G, H, a, vb):
    """min(0, min over integer v in [-vb, vb], v != 0 of u G + u^2 H / 2 with u = a v): the quadratic is convex in v
    (H >= 0), so only the integers next to its minimiser (and -1, 1 around 0, or the ends) can attain it."""
    best = 0.0
    den = a * H
    if den > 0.0 or den < 0.0:
        vs = -G / den
        lo = np.floor(vs)
        if lo < -vb:
            lo = -vb
        if lo > vb:
            lo = vb
        hi = lo + 1.0
        if hi > vb:
            hi = vb
        for v in (lo, hi, -1.0, 1.0):
            if v == 0.0:
                continue
            u = a * v
            f = u * G + 0.5 * u * u * H
            if f < best:
                best = f
    else:
        for v in (-vb, vb):
            u = a * v
            f = u * G + 0.5 * u * u * H
            if f < best:
                best = f
    return best


@nb.njit(cache=False)
def screen_cols(G12, p, cols, a, dl, m):
    """The m columns (ascending) of `cols` with the best second-order score min_delta u G + u^2 H / 2 (u = a d),
    G and H in columns p and p + 1 of G12."""
    nc = cols.shape[0]
    sc = np.empty(nc)
    vb = 0.0
    for r in range(dl.shape[0]):
        vb = max(vb, abs(dl[r]))
    for t in range(nc):
        sc[t] = _qbest(G12[cols[t], p], G12[cols[t], p + 1], a, vb)
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
               cntw, mb, nzv, n_screen, ne, tr_a, tr_b, cinv, csv, recal=False):
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
    AB = np.empty((k, 2))
    nb0 = csv.shape[0]
    key = np.empty(n, np.int64)
    for q in range(k):
        j = S[q]
        v = var_of[j]
        base = tptr[j]
        if kind[v] == 0:
            # binary column: the removal state's bins are the current bins split by x_j (rows keyed by (bin, x))
            l = lev_of[j]
            pres = np.zeros(2 * nb0, np.bool_)
            Wp2k = np.zeros(2 * nb0)
            Wn2k = np.zeros(2 * nb0)
            for i in range(n):
                kk = 2 * cinv[i] + (1 if QT[v, i] >= l else 0)
                key[i] = kk
                pres[kk] = True
                if y[i] > 0:
                    Wp2k[kk] += c[i]
                else:
                    Wn2k[kk] += c[i]
            npres = 0
            for kk in range(2 * nb0):
                npres += pres[kk]
            ks = np.empty(npres, np.int64)
            ksc = np.empty(npres)
            u = 0
            for kk in range(2 * nb0):
                if pres[kk]:
                    ks[u] = kk
                    ksc[u] = csv[kk >> 1] - w[j] * (kk & 1)
                    u += 1
            o = np.argsort(ksc, kind="mergesort")
            newid = np.empty(2 * nb0, np.int64)
            svt = np.empty(npres)
            m2 = -1
            for t in range(npres):
                if t == 0 or ksc[o[t]] != svt[m2]:
                    m2 += 1
                    svt[m2] = ksc[o[t]]
                newid[ks[o[t]]] = m2
            m2 += 1
            sv = svt[:m2].copy()
            Wp = np.zeros(m2)
            Wn = np.zeros(m2)
            for kk in range(2 * nb0):
                if pres[kk]:
                    Wp[newid[kk]] += Wp2k[kk]
                    Wn[newid[kk]] += Wn2k[kk]
            lp = np.empty(m2)
            ln_ = np.empty(m2)
            pp = np.empty(m2)
            pn = np.empty(m2)
            wq = np.empty(m2)
            aq, bq = a, b
            if recal and m2 >= 2:
                _, aq, bq = calibrate_bins(sv, Wp, Wn, a, b)
            AB[q, 0] = aq
            AB[q, 1] = bq
            T = bin_stats(sv, Wp, Wn, aq, bq, lp, ln_, pp, pn, wq)
            inv = np.empty(n, np.int64)
            for i in range(n):
                g = newid[key[i]]
                inv[i] = g
                R[i, 2 * q] = (-pp[g] if y[i] > 0 else pn[g]) * c[i]
                R[i, 2 * q + 1] = wq[g] * c[i]
        else:
            dj = np.empty(n)
            for i in range(n):
                dj[i] = w[j] * tabv[base + QT[v, i]]
            s2, inv, sv, Wp, Wn, lp, ln_, pp, pn, wq, T, r_, h_ = removal_state(s, dj, y, c, a, b)
            AB[q, 0] = a
            AB[q, 1] = b
            if recal and sv.shape[0] >= 2:
                _, aq, bq = calibrate_bins(sv, Wp, Wn, a, b)
                s2, inv, sv, Wp, Wn, lp, ln_, pp, pn, wq, T, r_, h_ = removal_state(s, dj, y, c, aq, bq)
                AB[q, 0] = aq
                AB[q, 1] = bq
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
        aq = AB[q, 0]
        bq = AB[q, 1]
        cols = screen_cols(G12, 2 * q, nonSq, aq, nzv, n_screen)
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
        eval_vars(cols[o], o, deltas, inv, sv.shape[0], sv, lp, ln_, pp, pn, wq, T, aq, bq, y, c, QT, var_of,
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
def support_hist(S, inv, nb_, y, c, QT, var_of, kind, ncode, lev_of, radius):
    """For every support column of a chain: weights of y = +1 / -1 per (score bin, code of its variable), suffix-
    summed over codes (entry [q, b, l]: rows of bin b with code >= l), from one pass over the rows."""
    k = S.shape[0]
    n = inv.shape[0]
    vs = np.empty(k, np.int64)
    mc = 1
    for q in range(k):
        vs[q] = var_of[S[q]]
        if kind[vs[q]] == 0:
            mc = max(mc, ncode[vs[q]] + 1)
    HP = np.zeros((k, nb_, mc))
    HN = np.zeros((k, nb_, mc))
    for q in range(k):
        if kind[vs[q]] != 0:
            vs[q] = -1
    for i in range(n):
        bq = inv[i]
        ci = c[i]
        if y[i] > 0:
            for q in range(k):
                if vs[q] >= 0:
                    HP[q, bq, QT[vs[q], i]] += ci
        else:
            for q in range(k):
                if vs[q] >= 0:
                    HN[q, bq, QT[vs[q], i]] += ci
    for q in range(k):
        if vs[q] < 0:
            continue
        nc = ncode[vs[q]]
        # suffix sums down to the lowest level a value change or a slide reads
        lo_q = max(0, lev_of[S[q]] - radius)
        for bq in range(nb_):
            for qq in range(nc - 1, lo_q - 1, -1):
                HP[q, bq, qq] += HP[q, bq, qq + 1]
                HN[q, bq, qq] += HN[q, bq, qq + 1]
    return HP, HN


@nb.njit(cache=False)
def make_tb(sv, nbins, dlo, dhi, a, b):
    """Loss terms at (a, b) tabulated over the integer scores sv + [dlo, dhi] (empty when scores are not integers)."""
    tlo = 0.0
    TB = np.zeros((4, 0))
    integral = True
    lo = np.inf
    hi = -np.inf
    for q in range(nbins):
        if sv[q] != np.floor(sv[q]):
            integral = False
        lo = min(lo, sv[q])
        hi = max(hi, sv[q])
    if dlo != np.floor(dlo) or dhi != np.floor(dhi):
        integral = False
    if integral and nbins > 0 and hi - lo + dhi - dlo < 4096:
        tlo = lo + dlo
        R = int(hi + dhi - tlo) + 1
        TB = np.empty((4, R))
        for ti in range(R):
            z = a * (tlo + ti) + b
            e = np.exp(-abs(z))
            l1 = np.log1p(e)
            if z > 0:
                TB[0, ti] = l1
                TB[1, ti] = z + l1
                TB[2, ti] = e / (1.0 + e)
                TB[3, ti] = 1.0 / (1.0 + e)
            else:
                TB[0, ti] = -z + l1
                TB[1, ti] = l1
                TB[2, ti] = 1.0 / (1.0 + e)
                TB[3, ti] = e / (1.0 + e)
    return tlo, TB


@nb.njit(cache=False)
def eval_support(S, deltas, HP, HN, lev_of, nb_, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b):
    """Value changes of the support columns (all chain columns) from the support histograms."""
    m = S.shape[0]
    dlo = 0.0
    dhi = 0.0
    integral = True
    for q in range(m):
        for r in range(deltas.shape[1]):
            dv = deltas[q, r]
            if dv != np.floor(dv):
                integral = False
            dlo = min(dlo, dv)
            dhi = max(dhi, dv)
    if integral:
        tlo, TB = make_tb(sv, nb_, dlo, dhi, a, b)
    else:
        tlo, TB = 0.0, np.zeros((4, 0))
    cb = np.empty(nb_, np.int64)
    cx = np.ones(nb_)
    cwp = np.empty(nb_)
    cwn = np.empty(nb_)
    for q in range(m):
        l = lev_of[S[q]]
        nt = 0
        for bq in range(nb_):
            if HP[q, bq, l] > 0.0 or HN[q, bq, l] > 0.0:
                cb[nt] = bq
                cwp[nt] = HP[q, bq, l]
                cwn[nt] = HN[q, bq, l]
                nt += 1
        eval_cells(q, deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b, tlo, TB)


@nb.njit(cache=False)
def slide_moves(w, S, free, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of, kind, ncode, vrp,
                vrows, vcols, vcp, cntw, mb, ne, tr_a, tr_b, radius, SHP, SHN):
    """Threshold slides: a support column of a chain moves to another threshold of its variable (at most `radius`
    levels away) with the same points. The score changes by +-w_j on the band of codes between the two levels,
    so every slide of a column is estimated from one (bin, code) histogram of its variable. Returns the ne best
    per support column (estimate, removed column, added column, value)."""
    m = S.shape[0]
    d = w.shape[0]
    nb_ = sv.shape[0]
    o_est = np.full(m * ne, np.inf)
    o_rj = np.full(m * ne, -1, np.int64)
    o_aj = np.full(m * ne, -1, np.int64)
    o_v = np.zeros(m * ne)
    tb0 = np.zeros((4, 0))
    cb = np.empty(nb_, np.int64)
    cx = np.empty(nb_)
    cwp = np.empty(nb_)
    cwn = np.empty(nb_)
    for q in range(m):
        j = S[q]
        v = var_of[j]
        nc = ncode[v]
        if kind[v] != 0 or nc <= 2:
            continue
        l = lev_of[j]
        mask = forbid_kernel(S, j, var_of, kind, vcols, vcp, cntw, mb, d)
        lo_l = max(1, l - radius)
        hi_l = min(nc - 1, l + radius)
        ntar = 0
        tl = np.empty(hi_l - lo_l + 1, np.int64)
        for l2 in range(lo_l, hi_l + 1):
            j2 = vcols[vcp[v] + l2 - 1]
            if l2 != l and free[j2] and not mask[j2]:
                tl[ntar] = l2
                ntar += 1
        if ntar == 0:
            continue
        HP = SHP[q]
        HN = SHN[q]
        deltas = np.full((ntar, 1), w[j])
        tlo, TB = make_tb(sv, nb_, -abs(w[j]), abs(w[j]), a, b)
        out = np.empty((2, ntar, 1))
        for t in range(ntar):
            l2 = tl[t]
            lo2 = min(l, l2)
            hi2 = max(l, l2)
            sg = 1.0 if l2 < l else -1.0  # rows of the band gain (lower level) or lose the indicator
            nt = 0
            for bq in range(nb_):
                wp_ = HP[bq, lo2] - HP[bq, hi2]
                wn_ = HN[bq, lo2] - HN[bq, hi2]
                if wp_ > 0.0 or wn_ > 0.0:
                    cb[nt] = bq
                    cx[nt] = sg
                    cwp[nt] = wp_
                    cwn[nt] = wn_
                    nt += 1
            eval_cells(t, deltas, cb, cx, cwp, cwn, nt, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b, tlo, TB)
        fo = np.argsort(out[0, :, 0], kind="mergesort")
        for u in range(min(ne, ntar)):
            t = fo[u]
            o_est[q * ne + u] = out[0, t, 0]
            o_rj[q * ne + u] = j
            o_aj[q * ne + u] = vcols[vcp[v] + tl[t] - 1]
            o_v[q * ne + u] = w[j]
    return o_est, o_rj, o_aj, o_v


@nb.njit(cache=False)
def exact_support(S, deltas, HP, HN, lev_of, sv, Wp, Wn, a, b, out):
    """out[0, q, r] = exact calibrated loss of the value change deltas[q, r] of chain column S[q], from its (bin, x)
    cells (support histograms)."""
    nb_ = sv.shape[0]
    m2 = 2 * nb_
    sc = np.empty(m2)
    cwp = np.empty(m2)
    cwn = np.empty(m2)
    svq = np.empty(m2)
    Wpq = np.empty(m2)
    Wnq = np.empty(m2)
    E = np.empty(2 * m2)
    En = np.empty(2 * m2)
    for q in range(S.shape[0]):
        l = lev_of[S[q]]
        for bq in range(nb_):
            cwp[2 * bq] = Wp[bq] - HP[q, bq, l]
            cwn[2 * bq] = Wn[bq] - HN[q, bq, l]
            cwp[2 * bq + 1] = HP[q, bq, l]
            cwn[2 * bq + 1] = HN[q, bq, l]
        for r in range(deltas.shape[1]):
            dv = deltas[q, r]
            for bq in range(nb_):
                sc[2 * bq] = sv[bq]
                sc[2 * bq + 1] = sv[bq] + dv
            o = np.argsort(sc, kind="mergesort")
            g = -1
            for u in range(m2):
                t = o[u]
                if cwp[t] <= 0.0 and cwn[t] <= 0.0:
                    continue
                if g < 0 or sc[t] != svq[g]:
                    g += 1
                    svq[g] = sc[t]
                    Wpq[g] = 0.0
                    Wnq[g] = 0.0
                Wpq[g] += cwp[t]
                Wnq[g] += cwn[t]
            g += 1
            if g < 2:
                out[0, q, r] = np.inf
                continue
            L2, a2, b2 = calibrate_bins_buf(svq[:g], Wpq[:g], Wnq[:g], a, b, E, En)
            out[0, q, r] = L2


@nb.njit(cache=False)
def main_phase(w, S, free, k, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of, lev_of, kind, ncode, tptr,
               tabv, vrp, vrows, vcols, vcp, cntw, mb, allv, nzv, ne, tr_a, tr_b, add_screen, SHP, SHN, Wp, Wn):
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
        allch = True
        for q in range(m):
            if kind[var_of[S[q]]] != 0:
                allch = False
        if allch:
            if EXV > 0:
                exact_support(S, deltas, SHP, SHN, lev_of, sv, Wp, Wn, a, b, out)
            else:
                eval_support(S, deltas, SHP, SHN, lev_of, nb_, sv, lp, ln_, pp, pn, wq, T, a, b, out, tr_a, tr_b)
        else:
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
def check_kernel(s, w, L, est, rjs, ajs, vs, a, b, margin, y, c, QT, var_of, tptr, tabv, kind, lev_of, cinv, csv):
    """Exact calibrated loss of the candidate moves (in order); returns the index of the best improving one (-1 if
    none) and its score vector, bins and calibration. Moves on binary columns are scored from rows keyed by
    (score bin, x_removed, x_added) in one pass."""
    n = s.shape[0]
    bi = -1
    bL = np.inf
    ba = a
    bb = b
    nb0 = csv.shape[0]
    Wp4 = np.zeros(4 * nb0)
    Wn4 = np.zeros(4 * nb0)
    sc = np.empty(4 * nb0)
    kp = np.empty(4 * nb0, np.int64)
    for t in range(est.shape[0]):
        if est[t] > L * (1.0 + margin):
            continue  # the estimate says the move does not help
        rj = rjs[t]
        aj = ajs[t]
        va = var_of[aj]
        if kind[va] == 0 and (rj < 0 or kind[var_of[rj]] == 0):
            la = lev_of[aj]
            dv = vs[t] - w[aj]
            vr = var_of[rj] if rj >= 0 else 0
            lr = lev_of[rj] if rj >= 0 else 0
            wr = w[rj] if rj >= 0 else 0.0
            Wp4[:] = 0.0
            Wn4[:] = 0.0
            for i in range(n):
                kk = 4 * cinv[i] + (1 if QT[va, i] >= la else 0)
                if rj >= 0 and QT[vr, i] >= lr:
                    kk += 2
                if y[i] > 0:
                    Wp4[kk] += c[i]
                else:
                    Wn4[kk] += c[i]
            m = 0
            for kk in range(4 * nb0):
                if Wp4[kk] > 0.0 or Wn4[kk] > 0.0:
                    kp[m] = kk
                    sc[m] = csv[kk >> 2] - wr * ((kk >> 1) & 1) + dv * (kk & 1)
                    m += 1
            o = np.argsort(sc[:m], kind="mergesort")
            sv = np.empty(m)
            Wp = np.zeros(m)
            Wn = np.zeros(m)
            g = -1
            for u in range(m):
                kk = kp[o[u]]
                if g < 0 or sc[o[u]] != sv[g]:
                    g += 1
                    sv[g] = sc[o[u]]
                Wp[g] += Wp4[kk]
                Wn[g] += Wn4[kk]
            g += 1
            if g < 2:
                continue
            L2, a2, b2 = calibrate_bins(sv[:g], Wp[:g], Wn[:g], a, b)
        else:
            s2 = s.copy()
            if rj >= 0:
                v = var_of[rj]
                base = tptr[rj]
                for i in range(n):
                    s2[i] -= w[rj] * tabv[base + QT[v, i]]
            dv = vs[t] - w[aj]
            base = tptr[aj]
            for i in range(n):
                s2[i] += dv * tabv[base + QT[va, i]]
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
            bi, bL, ba, bb = t, L2, a2, b2
            if CHKFIRST > 0:
                break  # first improving candidate in estimate order
    if bi < 0:
        return bi, bL, s, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0), a, b
    # the winner's score vector and bins
    s2 = s.copy()
    rj = rjs[bi]
    if rj >= 0:
        v = var_of[rj]
        base = tptr[rj]
        for i in range(n):
            s2[i] -= w[rj] * tabv[base + QT[v, i]]
    aj = ajs[bi]
    dv = vs[bi] - w[aj]
    v = var_of[aj]
    base = tptr[aj]
    for i in range(n):
        s2[i] += dv * tabv[base + QT[v, i]]
    inv, sv, Wp, Wn = bin_scores(s2, y, c)
    return bi, bL, s2, inv, sv, Wp, Wn, ba, bb


@nb.njit(cache=False)
def exact_values(w, S, s, L, a, b, inv, sv, Wp, Wn, y, c, QT, var_of, kind, ncode, lev_of, tptr, tabv, allv):
    """Exact calibrated loss of every value change of every support column (chain columns from one (bin, code)
    histogram pass; other columns by a row pass each). Returns (best total loss, column, value); column -1 if none
    improves."""
    nb_ = sv.shape[0]
    SHP, SHN = support_hist(S, inv, nb_, y, c, QT, var_of, kind, ncode, lev_of, 0)
    bL = L - 1e-9 * L
    bj = -1
    bv = 0.0
    m2 = 2 * nb_
    sc = np.empty(m2)
    cwp = np.empty(m2)
    cwn = np.empty(m2)
    svq = np.empty(m2)
    Wpq = np.empty(m2)
    Wnq = np.empty(m2)
    E = np.empty(2 * m2)
    En = np.empty(2 * m2)
    n = s.shape[0]
    for q in range(S.shape[0]):
        j = S[q]
        v = var_of[j]
        if kind[v] == 0:
            l = lev_of[j]
            for bq in range(nb_):
                cwp[2 * bq] = Wp[bq] - SHP[q, bq, l]
                cwn[2 * bq] = Wn[bq] - SHN[q, bq, l]
                cwp[2 * bq + 1] = SHP[q, bq, l]
                cwn[2 * bq + 1] = SHN[q, bq, l]
            c0p = np.empty(nb_)
            c0n = np.empty(nb_)
            c1p = np.empty(nb_)
            c1n = np.empty(nb_)
            for bq in range(nb_):
                c0p[bq] = cwp[2 * bq]
                c0n[bq] = cwn[2 * bq]
                c1p[bq] = cwp[2 * bq + 1]
                c1n[bq] = cwn[2 * bq + 1]
            # loss at the current (a, b) of every value; the POLVV best calibrated exactly
            fl = np.full(allv.shape[0], np.inf)
            for r in range(allv.shape[0]):
                dv = allv[r] - w[j]
                if dv == 0.0:
                    continue
                t = 0.0
                for bq in range(nb_):
                    if c1p[bq] <= 0.0 and c1n[bq] <= 0.0:
                        continue
                    z = a * (sv[bq] + dv) + b
                    l1 = np.log1p(np.exp(-abs(z)))
                    if z > 0:
                        t += c1p[bq] * l1 + c1n[bq] * (z + l1)
                    else:
                        t += c1p[bq] * (l1 - z) + c1n[bq] * l1
                fl[r] = t
            fo = np.argsort(fl)
            for r0 in range(min(POLVV, allv.shape[0] - 1)):
                r = fo[r0]
                dv = allv[r] - w[j]
                L2 = _merge_calib(sv, c0p, c0n, c1p, c1n, dv, a, b, svq, Wpq, Wnq, E, En, bL)
                if L2 < bL:
                    bL, bj, bv = L2, j, allv[r]
        else:
            base = tptr[j]
            s2 = np.empty(n)
            for r in range(allv.shape[0]):
                dv = allv[r] - w[j]
                if dv == 0.0:
                    continue
                for i in range(n):
                    s2[i] = s[i] + dv * tabv[base + QT[v, i]]
                inv2, sv2, Wp2, Wn2 = bin_scores(s2, y, c)
                if sv2.shape[0] < 2:
                    continue
                L2, a2, b2 = calibrate_bins(sv2, Wp2, Wn2, a, b)
                if L2 < bL:
                    bL, bj, bv = L2, j, allv[r]
    return bL, bj, bv


@nb.njit(cache=False)
def var_dp(w, s, a, b, v, rmax, y, c, QT, var_of, lev_of, ncode, vcols, vcp, valid, bound):
    """Best step function of chain variable v given the rest of the score, at fixed (a, b): jumps (integer, |.| <=
    bound) at up to rmax levels whose columns are valid; the variable's contribution is the cumulative sum of the
    jumps at or below a row's code. Exact DP over (code, cumulative value, jumps used) on the (bin of the rest,
    code) histogram. Returns the new points of the chain's columns (by level) and the DP loss at (a, b)."""
    n = s.shape[0]
    nc = ncode[v]
    base = vcp[v]
    # contribution of v now: cumulative over levels
    cur = np.zeros(nc)
    for l in range(1, nc):
        cur[l] = cur[l - 1] + w[vcols[base + l - 1]]
    sm = np.empty(n)
    for i in range(n):
        sm[i] = s[i] - cur[QT[v, i]]
    inv, sv, Wp, Wn = bin_scores(sm, y, c)
    nb_ = sv.shape[0]
    HP = np.zeros((nc, nb_))
    HN = np.zeros((nc, nb_))
    for i in range(n):
        if y[i] > 0:
            HP[QT[v, i], inv[i]] += c[i]
        else:
            HN[QT[v, i], inv[i]] += c[i]
    U = int(bound) * rmax
    nu = 2 * U + 1
    cost = np.zeros((nc, nu))
    for cc in range(nc):
        for ui in range(nu):
            u = ui - U
            t = 0.0
            for q in range(nb_):
                wp = HP[cc, q]
                wn = HN[cc, q]
                if wp == 0.0 and wn == 0.0:
                    continue
                z = a * (sv[q] + u) + b
                l1 = np.log1p(np.exp(-abs(z)))
                if z > 0:
                    t += wp * l1 + wn * (z + l1)
                else:
                    t += wp * (l1 - z) + wn * l1
            cost[cc, ui] = t
    INF = np.inf
    F = np.full((nc, nu, rmax + 1), INF)
    BK = np.full((nc, nu, rmax + 1), -127, np.int8)  # jump at this code (0: none)
    F[0, U, 0] = cost[0, U]
    ib = int(bound)
    for cc in range(1, nc):
        ok = valid[vcols[base + cc - 1]]
        for ui in range(nu):
            for r in range(rmax + 1):
                best = F[cc - 1, ui, r]
                bj = 0
                if ok and r > 0:
                    for dl in range(-ib, ib + 1):
                        if dl == 0:
                            continue
                        up = ui - dl
                        if up < 0 or up >= nu:
                            continue
                        f = F[cc - 1, up, r - 1]
                        if f < best:
                            best = f
                            bj = dl
                if best < INF:
                    F[cc, ui, r] = best + cost[cc, ui]
                    BK[cc, ui, r] = bj
    bl = INF
    bu = 0
    br = 0
    for ui in range(nu):
        for r in range(rmax + 1):
            if F[nc - 1, ui, r] < bl:
                bl = F[nc - 1, ui, r]
                bu = ui
                br = r
    newv = np.zeros(nc - 1)
    ui = bu
    r = br
    for cc in range(nc - 1, 0, -1):
        dl = BK[cc, ui, r]
        if dl != 0:
            newv[cc - 1] = dl
            ui -= dl
            r -= 1
    return newv, bl


@nb.njit(cache=False)
def calib_target(sv, Wp, Wn, a, b, target, E, En):
    """calibrate_bins_buf that stops early: returns the loss once it is below target (an upper bound of the
    calibrated loss, so the move surely beats target), or np.inf once even twice the Newton decrement cannot reach
    target (rejected), else the converged loss."""
    m = sv.shape[0]
    cur = 0.0
    for i in range(m):
        lv, E[i] = _lrow_e(a * sv[i] + b)
        cur += Wp[i] * lv
    for i in range(m):
        lv, E[m + i] = _lrow_e(-(a * sv[i] + b))
        cur += Wn[i] * lv
    if cur < target:
        return cur
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for i in range(m):
            z = a * sv[i] + b
            p = _sig_e(z, E[i])
            w = Wp[i] * p * (1.0 - p)
            gi = -Wp[i] * p
            ga += gi * sv[i]
            gb += gi
            haa += w * sv[i] * sv[i]
            hab += w * sv[i]
            hbb += w
        for i in range(m):
            z = -(a * sv[i] + b)
            p = _sig_e(z, E[m + i])
            w = Wn[i] * p * (1.0 - p)
            gi = Wn[i] * p
            ga += gi * sv[i]
            gb += gi
            haa += w * sv[i] * sv[i]
            hab += w * sv[i]
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
        if cur - REJ * dec > target:
            return np.inf
        t = 1.0
        new = cur
        while t > 1e-10:
            na, nb_ = a - t * da, b - t * db
            new = 0.0
            for i in range(m):
                lv, En[i] = _lrow_e(na * sv[i] + nb_)
                new += Wp[i] * lv
            for i in range(m):
                lv, En[m + i] = _lrow_e(-(na * sv[i] + nb_))
                new += Wn[i] * lv
            if new <= cur - 1e-4 * t * dec:
                break
            t *= 0.5
        if t <= 1e-10:
            break
        a, b = na, nb_
        E, En = En, E
        improv = cur - new
        cur = new
        if cur < target:
            return cur
        if improv < 1e-12 * (1.0 + cur):
            break
    return cur


@nb.njit(cache=False)
def _merge_calib(sv, cwp0, cwn0, cwp1, cwn1, dv, a, b, svq, Wpq, Wnq, E, En, target):
    """Exact calibrated loss of the cells (score sv[q], weights cw*0) and (score sv[q] + dv, weights cw*1), sv
    ascending: merge of the two sorted lists, equal scores combined."""
    m = sv.shape[0]
    i0 = 0
    i1 = 0
    g = -1
    while i0 < m or i1 < m:
        if i1 >= m or (i0 < m and sv[i0] <= sv[i1] + dv):
            sc = sv[i0]
            wp = cwp0[i0]
            wn = cwn0[i0]
            i0 += 1
        else:
            sc = sv[i1] + dv
            wp = cwp1[i1]
            wn = cwn1[i1]
            i1 += 1
        if wp <= 0.0 and wn <= 0.0:
            continue
        if g < 0 or sc != svq[g]:
            g += 1
            svq[g] = sc
            Wpq[g] = 0.0
            Wnq[g] = 0.0
        Wpq[g] += wp
        Wnq[g] += wn
    g += 1
    if g < 2:
        return np.inf
    return calib_target(svq[:g], Wpq[:g], Wnq[:g], a, b, target, E, En)


@nb.njit(cache=False)
def exact_moves(s, w, S, free, k, y, c, a, b, QT, var_of, lev_of, kind, ncode, tptr, tabv, vrp, vrows, vcols, vcp,
                nzv, n_screen, L0):
    """Exact best swap / addition: per removal (and no removal when |S| < k) the n_screen best columns by the
    second-order screen at the removal state, at the current (a, b) and at the refitted (a, b) each (one colsum for
    all removals); the POLV best values of every screened chain column (by the loss at the refitted (a, b)) scored
    by their exact calibrated loss on (bin, x) cells. Returns (total loss, removed column or -1, added column,
    value); added column -1 when nothing beats L0."""
    n = s.shape[0]
    m = S.shape[0]
    d = w.shape[0]
    nv = nzv.shape[0]
    vb = 0.0
    for r in range(nv):
        vb = max(vb, abs(nzv[r]))
    bL = L0
    brj = -1
    baj = -1
    bv = 0.0
    nfree = 0
    for t in range(d):
        if free[t] and kind[var_of[t]] == 0:
            nfree += 1
    if nfree == 0:
        return bL, brj, baj, bv
    fcols = np.empty(nfree, np.int64)
    u = 0
    for t in range(d):
        if free[t] and kind[var_of[t]] == 0:
            fcols[u] = t
            u += 1
    q0 = -1 if m < k else 0
    nq = m - q0
    PC = 4 if POLOLD > 0 else 2
    R = np.empty((n, PC * nq))
    INV = np.empty((nq, n), np.int64)
    BINS = List()
    AB = np.empty((nq, 2))
    s2 = np.empty(n)
    for qi in range(nq):
        q = q0 + qi
        if q >= 0:
            j = S[q]
            base = tptr[j]
            vj = var_of[j]
            for i in range(n):
                s2[i] = s[i] - w[j] * tabv[base + QT[vj, i]]
        else:
            for i in range(n):
                s2[i] = s[i]
        inv, sv, Wp, Wn = bin_scores(s2, y, c)
        nb_ = sv.shape[0]
        if nb_ >= 2:
            _, aq, bq = calibrate_bins(sv, Wp, Wn, a, b)
        else:
            aq, bq = a, b
        AB[qi, 0] = aq
        AB[qi, 1] = bq
        INV[qi] = inv
        st3 = np.empty((3, nb_))
        st3[0] = sv
        st3[1] = Wp
        st3[2] = Wn
        BINS.append(st3)
        lp = np.empty(nb_)
        ln_ = np.empty(nb_)
        pp = np.empty(nb_)
        pn = np.empty(nb_)
        wq = np.empty(nb_)
        if POLOLD > 0:
            bin_stats(sv, Wp, Wn, a, b, lp, ln_, pp, pn, wq)
            for i in range(n):
                g = inv[i]
                R[i, PC * qi + 2] = (-pp[g] if y[i] > 0 else pn[g]) * c[i]
                R[i, PC * qi + 3] = wq[g] * c[i]
        bin_stats(sv, Wp, Wn, aq, bq, lp, ln_, pp, pn, wq)
        for i in range(n):
            g = inv[i]
            R[i, PC * qi] = (-pp[g] if y[i] > 0 else pn[g]) * c[i]
            R[i, PC * qi + 1] = wq[g] * c[i]
    pw = np.zeros(PC * nq, np.bool_)
    for qi in range(nq):
        pw[PC * qi + 1] = True
        if PC == 4:
            pw[PC * qi + 3] = True
    G = colsum(QT, R, kind, ncode, vcp, vcols, lev_of, tptr, tabv, pw, vrp, vrows)
    fl = np.zeros(nv)
    for qi in range(nq):
        q = q0 + qi
        j = S[q] if q >= 0 else -1
        aq = AB[qi, 0]
        bq = AB[qi, 1]
        st3 = BINS[qi]
        sv = st3[0]
        Wp = st3[1]
        Wn = st3[2]
        nb_ = sv.shape[0]
        inv = INV[qi]
        c2 = screen_cols(G, PC * qi, fcols, aq, nzv, n_screen)
        if POLOLD > 0:
            c1 = screen_cols(G, PC * qi + 2, fcols, a, nzv, n_screen)
            cols = np.unique(np.concatenate((c1, c2)))
        else:
            cols = c2
        cwp0 = np.empty(nb_)
        cwn0 = np.empty(nb_)
        cwp1 = np.empty(nb_)
        cwn1 = np.empty(nb_)
        svq = np.empty(2 * nb_)
        Wpq = np.empty(2 * nb_)
        Wnq = np.empty(2 * nb_)
        E = np.empty(4 * nb_)
        En = np.empty(4 * nb_)
        tlo, TB = make_tb(sv, nb_, -vb, vb, aq, bq)
        # columns grouped by variable: one (bin, code) histogram per variable, suffix-summed over codes
        vo = np.empty(cols.shape[0], np.int64)
        for t in range(cols.shape[0]):
            vo[t] = var_of[cols[t]] * (d + 1) + lev_of[cols[t]]
        cols = cols[np.argsort(vo)]
        cur_v = -1
        HPv = np.zeros((1, 1))
        HNv = np.zeros((1, 1))
        for t in range(cols.shape[0]):
            l = cols[t]
            if l == j:
                continue
            v = var_of[l]
            lv = lev_of[l]
            if v != cur_v:
                cur_v = v
                nc = ncode[v]
                HPv = np.zeros((nc + 1, nb_))
                HNv = np.zeros((nc + 1, nb_))
                for tt in range(vrp[v], vrp[v + 1]):
                    i = vrows[tt]
                    if y[i] > 0:
                        HPv[QT[v, i], inv[i]] += c[i]
                    else:
                        HNv[QT[v, i], inv[i]] += c[i]
                for cc in range(nc - 1, 0, -1):
                    for g in range(nb_):
                        HPv[cc, g] += HPv[cc + 1, g]
                        HNv[cc, g] += HNv[cc + 1, g]
            for g in range(nb_):
                cwp1[g] = HPv[lv, g]
                cwn1[g] = HNv[lv, g]
            for g in range(nb_):
                cwp0[g] = Wp[g] - cwp1[g]
                cwn0[g] = Wn[g] - cwn1[g]
            fl[:] = 0.0
            for g in range(nb_):
                if cwp1[g] <= 0.0 and cwn1[g] <= 0.0:
                    continue
                for r in range(nv):
                    tf = sv[g] + nzv[r] - tlo
                    if TB.shape[1] > 0 and tf >= 0.0 and tf < TB.shape[1] and tf == np.floor(tf):
                        ti = int(tf)
                        fl[r] += cwp1[g] * TB[0, ti] + cwn1[g] * TB[1, ti]
                        continue
                    z = aq * (sv[g] + nzv[r]) + bq
                    l1 = np.log1p(np.exp(-abs(z)))
                    if z > 0:
                        fl[r] += cwp1[g] * l1 + cwn1[g] * (z + l1)
                    else:
                        fl[r] += cwp1[g] * (l1 - z) + cwn1[g] * l1
            fo = np.argsort(fl)
            for r0 in range(min(POLV, nv)):
                r = fo[r0]
                L2 = _merge_calib(sv, cwp0, cwn0, cwp1, cwn1, nzv[r], aq, bq, svq, Wpq, Wnq, E, En, bL)
                if L2 < bL:
                    bL, brj, baj, bv = L2, j, l, nzv[r]
    return bL, brj, baj, bv


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
    # the candidates of a failing swap phase at the final point (reused by the refit swaps)
    has_c = False
    c_oe = np.zeros(0)
    c_orj = np.zeros(0, np.int64)
    c_oaj = np.zeros(0, np.int64)
    c_ov = np.zeros(0)
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
        SHP, SHN = support_hist(S, inv, nb_, y, c, QT, var_of, kind, ncode, lev_of, SLIDER)
        oe, oaj, ov, ofx = main_phase(w, S, free, k, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of,
                                      lev_of, kind, ncode, tptr, tabv, vrp, vrows, vcols, vcp, cntw, mb, allv, nzv,
                                      ne, tr_a, tr_b, add_screen, SHP, SHN, Wp, Wn)
        orj = np.full(oe.shape[0], -1, np.int64)
        if SLIDES == 1 and cntS > 0:
            se, srj, saj, sv_ = slide_moves(w, S, free, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of,
                                            lev_of, kind, ncode, vrp, vrows, vcols, vcp, cntw, mb, ne, tr_a, tr_b,
                                            SLIDER, SHP, SHN)
            oe = np.concatenate((oe, se))
            orj = np.concatenate((orj, srj))
            oaj = np.concatenate((oaj, saj))
            ov = np.concatenate((ov, sv_))
        sel = top_order(oe, m_chk)
        bi, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, oe[sel], orj[sel], oaj[sel], ov[sel], a, b,
                                                               margin, y, c, QT, var_of, tptr, tabv, kind, lev_of,
                                                               inv, sv)
        rj = -1
        aj = -1
        v = 0.0
        from_main = False
        if bi >= 0:
            rj = orj[sel[bi]]
            aj = oaj[sel[bi]]
            v = ov[sel[bi]]
            from_main = True
            m_sel = sel
            m_bi = bi
        if bi < 0 and SLIDES == 2 and cntS > 0 and nfree > 0 and swaps and L / N <= gate:
            # threshold slides before the full swap phase
            se, srj, saj, sv_ = slide_moves(w, S, free, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, QT, var_of,
                                            lev_of, kind, ncode, vrp, vrows, vcols, vcp, cntw, mb, ne, tr_a, tr_b,
                                            SLIDER, SHP, SHN)
            sel = top_order(se, m_chk)
            bi, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, se[sel], srj[sel], saj[sel], sv_[sel],
                                                                   a, b, margin, y, c, QT, var_of, tptr, tabv, kind,
                                                                   lev_of, inv, sv)
            if bi >= 0:
                rj = srj[sel[bi]]
                aj = saj[sel[bi]]
                v = sv_[sel[bi]]
        if bi >= 0:
            pass
        elif nfree > 0 and swaps and L / N <= gate:
            rc = RG > 0 and abs(a) * (sv[sv.shape[0] - 1] - sv[0]) >= RG
            oe, orj, oaj, ov, ofx = swap_phase(s, w, S, free, y, c, a, b, QT, var_of, lev_of, kind, ncode, tptr, tabv,
                                               vrp, vrows, vcols, vcp, cntw, mb, nzv, n_screen, ne, tr_a, tr_b, inv, sv,
                                               rc)
            sel = top_order(oe, m_chk)
            bi, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, oe[sel], orj[sel], oaj[sel], ov[sel], a,
                                                                   b, margin, y, c, QT, var_of, tptr, tabv, kind,
                                                                   lev_of, inv, sv)
            if bi < 0 and RCFB > 0:
                # fallback: swaps screened at removal states with a refitted score-to-risk map
                oe2, orj2, oaj2, ov2, ofx2 = swap_phase(s, w, S, free, y, c, a, b, QT, var_of, lev_of, kind, ncode,
                                                        tptr, tabv, vrp, vrows, vcols, vcp, cntw, mb, nzv, n_screen,
                                                        ne, tr_a, tr_b, inv, sv, True)
                sel2 = top_order(oe2, RCFB)
                bi2, L2, s2, inv2, sv2, Wp2, Wn2, a2, b2 = check_kernel(s, w, L, oe2[sel2] * 0.0 - np.inf,
                                                                        orj2[sel2], oaj2[sel2], ov2[sel2], a, b,
                                                                        margin, y, c, QT, var_of, tptr, tabv, kind,
                                                                        lev_of, inv, sv)
                if bi2 >= 0:
                    bi = bi2
                    oe, orj, oaj, ov, sel = oe2, orj2, oaj2, ov2, sel2
            if bi >= 0:
                rj = orj[sel[bi]]
                aj = oaj[sel[bi]]
                v = ov[sel[bi]]
            else:
                has_c = True
                c_oe, c_orj, c_oaj, c_ov = oe, orj, oaj, ov
        if bi < 0:
            break
        if rj >= 0:
            w[rj] = 0.0
        w[aj] = v
        s, L, a, b, inv, sv, Wp, Wn = s2, L2, a2, b2, inv2, sv2, Wp2, Wn2
        if MULTI > 0 and from_main:
            # the other checked value changes (other columns), exactly on the new point, before a new main phase
            used = np.zeros(1, np.int64)
            used[0] = aj
            for t in range(m_sel.shape[0]):
                if t == m_bi:
                    continue
                q = m_sel[t]
                cj = oaj[q]
                if orj[q] >= 0 or w[cj] == 0.0 or cj == aj:
                    continue  # value changes of support columns only
                dup = False
                for u in range(used.shape[0]):
                    if used[u] == cj:
                        dup = True
                if dup:
                    continue
                one_e = np.full(1, -np.inf)
                bi3, L3, s3, inv3, sv3, Wp3, Wn3, a3, b3 = check_kernel(s, w, L, one_e, orj[q:q + 1], oaj[q:q + 1],
                                                                       ov[q:q + 1], a, b, margin, y, c, QT, var_of,
                                                                       tptr, tabv, kind, lev_of, inv, sv)
                if bi3 >= 0:
                    w[cj] = ov[q]
                    s, L, a, b, inv, sv, Wp, Wn = s3, L3, a3, b3, inv3, sv3, Wp3, Wn3
                    used = np.append(used, cj)
    return L / N, w, has_c, c_oe, c_orj, c_oaj, c_ov


@nb.njit(cache=False)
def start_state(w, QT, var_of, tptr, tabv, y, c, N):
    """ScoreState of the points w in one kernel: scores, bins and the calibrated (L, a, b)."""
    d = w.shape[0]
    cnt = 0
    for j in range(d):
        if w[j] != 0.0:
            cnt += 1
    S = np.empty(cnt, np.int64)
    wS = np.empty(cnt)
    u = 0
    for j in range(d):
        if w[j] != 0.0:
            S[u] = j
            wS[u] = w[j]
            u += 1
    s = score_kernel(QT, var_of, tptr, tabv, S, wS)
    inv, sv, Wp, Wn = bin_scores(s, y, c)
    if sv.shape[0] < 2:
        npos = 0.0
        for i in range(y.shape[0]):
            if y[i] > 0:
                npos += c[i]
        L, a, b = calibrate_bins(sv, Wp, Wn, 0.0, np.log(npos / (N - npos)))
    else:
        sd = np.std(s)
        L, a, b = calibrate_bins(sv / sd, Wp, Wn, 0.0, 0.0)
        a = a / sd
    return s, L, a, b, inv, sv, Wp, Wn


class ILS:
    """Best-improvement local search over integer points, scored by the calibrated loss."""

    def __init__(self, D, k, n_exact=2, n_screen=4, n_rank=1):
        self.D, self.k, self.n_exact, self.n_screen = D, k, n_exact, n_screen
        self.visited = set()
        self.n_rank = n_rank
        self.est_margin = 1e-4
        self.nevals = 0
        self.hr = _normals(3, k + 1)
        self.hw = _normals(11, D.d)  # hash of a point: its random projection
        self.vis = _new_dict_f64()
        self.cands = {}  # final point -> candidates of its failing swap phase
        self.allv = np.arange(-COEF_BOUND, COEF_BOUND + 1, dtype=np.float64)
        self.nzv = self.allv[self.allv != 0]

    def run(self, w, max_iter=100, deadline=np.inf, swaps=True, gate=np.inf):
        D, k = self.D, self.k
        w = w.astype(np.float64).copy()
        st = start_state(w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
        allv, nzv = self.allv, self.nzv
        out = ils_kernel(w, st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], self.vis, self.hw, k, max_iter,
                          swaps, float(gate), D.N, self.n_exact, self.n_screen, ADDSCREEN, self.est_margin,
                          MS_BAND * D.msup, D.y, D.c, D.QT, D.var_of, D.lev_of, D.kind, D.ncode, D.tptr, D.tabv,
                          D.vrp, D.vrows, D.vcols, D.vcp, D.cntw, D.valid, allv, nzv, TR_A, TR_B, False,
                          self.hr)
        if out[2]:
            self.cands[out[1].tobytes()] = out[3:]
        return out[0], out[1]


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, n_starts=5):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.n_starts = n_starts

    def _rswap(self, D, ils, w0, t0, gate=np.inf):
        """Swaps with every point refitted: the RSWAP best swaps by the estimate at w0 (the candidates of its failing
        swap phase when its run kept them), each followed by a continuous refit of its support and a calibrated
        rounding; local search from the best rounding. Returns (loss, points) or None."""
        if np.count_nonzero(w0) < 2:
            return None
        key_b = w0.tobytes()
        if key_b in ils.cands:
            oe, orj, oaj, ov = ils.cands[key_b]
        else:
            st0 = start_state(w0, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
            S0 = np.flatnonzero(w0).astype(np.int64)
            free0 = (w0 == 0) & D.valid
            oe, orj, oaj, ov, ofx = swap_phase(st0[0], w0, S0, free0, D.y, D.c, st0[2], st0[3], D.QT, D.var_of,
                                               D.lev_of, D.kind, D.ncode, D.tptr, D.tabv, D.vrp, D.vrows, D.vcols,
                                               D.vcp, D.cntw, MS_BAND * D.msup, ils.nzv, NSCREEN, NEXACT, TR_A, TR_B,
                                               st0[4], st0[5])
        o = np.argsort(oe, kind="mergesort")[:RSWAP]
        bw, bl = None, np.inf
        for t in o:
            if not np.isfinite(oe[t]):
                break
            w1 = w0.copy()
            w1[orj[t]] = 0.0
            w1[oaj[t]] = ov[t]
            wr, lr = refit_round(w1, D.QT, D.var_of, D.lev_of, D.kind, D.ncode, D.tptr, D.tabv, D.y, D.c,
                                 float(COEF_BOUND), 1e-8, 20)
            if np.isfinite(lr) and lr < bl and wr.tobytes() not in self.rs_tried:
                bw, bl = wr, lr
        if bw is None:
            return None
        self.rs_tried.add(bw.tobytes())
        return ils.run(bw, deadline=t0 + 0.9 * self.time_limit, gate=gate)

    def _vardp(self, D, ils, best_l, best_w, t0):
        """Variable re-optimisation: for every chain variable of the support, the best step function given the rest
        of the score (DP at the current (a, b)); local search from any exact improvement; repeat."""
        k = self.k
        for _ in range(VARDP * 3):
            if time.perf_counter() > t0 + 0.9 * self.time_limit:
                break
            w0 = best_w
            st0 = start_state(w0, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
            S0 = np.flatnonzero(w0)
            vs = sorted(set(int(D.var_of[j]) for j in S0 if D.kind[D.var_of[j]] == 0 and D.ncode[D.var_of[j]] > 2))
            improved = False
            for v in vs:
                inv_cnt = int(np.sum(D.var_of[S0] == v))
                rmax = min(k - (len(S0) - inv_cnt), max(inv_cnt + 1, 2), 3)
                cols = D.vcols[D.vcp[v]:D.vcp[v + 1]]
                newv, bl = var_dp(w0, st0[0], st0[2], st0[3], v, rmax, D.y, D.c, D.QT, D.var_of, D.lev_of, D.ncode,
                                  D.vcols, D.vcp, D.valid, float(COEF_BOUND))
                w1 = w0.copy()
                w1[cols] = newv
                if np.array_equal(w1, w0) or np.count_nonzero(w1) == 0:
                    continue
                st1 = start_state(w1, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
                l1 = st1[1] / D.N
                if l1 < best_l - 1e-12:
                    l, w = ils.run(w1, deadline=t0 + 0.9 * self.time_limit)
                    if l1 < l:
                        l, w = l1, w1
                    best_l, best_w = l, w
                    improved = True
                    break
            if not improved:
                break
        return best_l, best_w

    def _polish(self, D, ils, best_l, best_w, t0):
        """Exact polish of the best point: every value change of a support column checked exactly, and swaps
        screened at removal states with a refitted score-to-risk map; a local search from any improvement."""
        allv = ils.allv
        if POLG > 0:
            # only where the calibrated map spans a wide logit range (near-separable scores), where the move
            # estimate (one Newton step in (a, b)) is unreliable
            stg = start_state(best_w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
            if abs(stg[2]) * (stg[5].max() - stg[5].min()) < POLG:
                return best_l, best_w
        for _ in range(POLISH):
            if time.perf_counter() > t0 + 0.9 * self.time_limit:
                break
            w0 = best_w
            st0 = start_state(w0, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
            S0 = np.flatnonzero(w0).astype(np.int64)
            rj, aj, vv = [], [], []
            ej = -1
            if POLVAL:
                eL, ej, ev = exact_values(w0, S0, st0[0], st0[1], st0[2], st0[3], st0[4], st0[5], st0[6], st0[7], D.y,
                                          D.c, D.QT, D.var_of, D.kind, D.ncode, D.lev_of, D.tptr, D.tabv, allv)
            if ej >= 0:
                rj.append(-1); aj.append(ej); vv.append(ev)
            free0 = (w0 == 0) & D.valid
            if POLX > 0 and free0.any():
                xL, xr, xa, xv = exact_moves(st0[0], w0, S0, free0, self.k, D.y, D.c, st0[2], st0[3], D.QT, D.var_of,
                                             D.lev_of, D.kind, D.ncode, D.tptr, D.tabv, D.vrp, D.vrows, D.vcols,
                                             D.vcp, ils.nzv, POLNS, st0[1] * (1 - 1e-9))
                if xa >= 0 and xL < st0[1] * (1 - 1e-9) and (ej < 0 or xL < eL):
                    rj, aj, vv = [xr], [xa], [xv]
            elif free0.any() and POLM > 0:
                oe, orj, oaj, ov, ofx = swap_phase(st0[0], w0, S0, free0, D.y, D.c, st0[2], st0[3], D.QT, D.var_of,
                                                   D.lev_of, D.kind, D.ncode, D.tptr, D.tabv, D.vrp, D.vrows, D.vcols,
                                                   D.vcp, D.cntw, MS_BAND * D.msup, ils.nzv, POLNS, POLNE, TR_A,
                                                   TR_B, st0[4], st0[5], True)
                for t in np.argsort(oe, kind="mergesort")[:POLM]:
                    if np.isfinite(oe[t]):
                        rj.append(orj[t]); aj.append(oaj[t]); vv.append(ov[t])
            if not rj:
                break
            est = np.full(len(rj), -np.inf)
            bi, L2 = check_kernel(st0[0], w0, st0[1], est, np.array(rj, np.int64), np.array(aj, np.int64),
                                  np.array(vv, np.float64), st0[2], st0[3], 1e-4, D.y, D.c, D.QT, D.var_of, D.tptr,
                                  D.tabv, D.kind, D.lev_of, st0[4], st0[5])[:2]
            if bi < 0:
                break
            w1 = w0.copy()
            if rj[bi] >= 0:
                w1[rj[bi]] = 0.0
            w1[aj[bi]] = vv[bi]
            l, w = ils.run(w1, deadline=t0 + 0.9 * self.time_limit)
            l1 = L2 / D.N
            if l1 < l:
                l, w = l1, w1
            if l < best_l - 1e-12:
                best_l, best_w = l, w
            else:
                break
        return best_l, best_w

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
        hv = _normals(7, D.n)
        st = parents
        RR, LL, KK = round_all(st[2], st[1], st[3], st[4], st[5], st[7], st[8], st[9], D.QT, D.var_of, D.tptr, D.tabv,
                               20, float(COEF_BOUND), hv, D.N)
        for t in range(len(LL)):
            w = np.zeros(D.d)
            w[st[7][t, :st[8][t]]] = RR[t, :st[8][t]]
            l = float(LL[t])
            key = round(float(KK[t]), 6)  # equivalent points (duplicate columns) give one start
            if key in seen or not np.isfinite(l):
                continue
            seen.add(key)
            starts.append((l, w))
            if l < best_l:
                best_l, best_w = l, w
        tm.append(time.perf_counter())
        # integer local search from the best few distinct rounded solutions
        ils = ILS(D, self.k, n_exact=NEXACT, n_screen=NSCREEN)
        self.rs_tried = set()
        starts.sort(key=lambda t: t[0])
        self.start_losses_ = []
        nofail = 0
        ends = []
        for si, (l0, w0) in enumerate(starts[:self.n_starts]):
            if time.perf_counter() > t0 + 0.8 * self.time_limit:
                break
            if SPAT > 0 and nofail >= SPAT:
                break
            if SPATG > 0 and si > 0 and nofail >= SPATG and np.isfinite(best_l):
                stg = start_state(best_w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
                if abs(stg[2]) * (stg[5].max() - stg[5].min()) >= POLG:
                    break
            l, w = ils.run(w0, deadline=t0 + 0.9 * self.time_limit)
            nofail = nofail + 1 if l >= best_l * (1.0 - SPEPS) - 1e-12 else 0
            self.start_losses_.append((l0, l))
            ends.append((l, w))
            if l < best_l:
                best_l, best_w = l, w
        if RSWAP > 0 and np.isfinite(best_l) and np.count_nonzero(best_w) > 1:
            # swaps with every point refitted, from the best point (RSF fails in all)
            fails = 0
            rsf = RSF
            if RSFG > 0:
                stg = start_state(best_w, D.QT, D.var_of, D.tptr, D.tabv, D.y, D.c, D.N)
                if abs(stg[2]) * (stg[5].max() - stg[5].min()) >= POLG:
                    rsf = RSFG
            while fails < rsf and time.perf_counter() < t0 + 0.8 * self.time_limit:
                res = self._rswap(D, ils, best_w, t0, gate=best_l * (1 + RSQ) if RSQ >= 0 else np.inf)
                if res is not None and res[0] < best_l - 1e-12:
                    best_l, best_w = res
                    fails = 0
                else:
                    fails += 1
                if res is None:
                    break
        if VARDP > 0 and np.isfinite(best_l) and np.count_nonzero(best_w) > 0:
            best_l, best_w = self._vardp(D, ils, best_l, best_w, t0)
        if POLISH > 0 and np.isfinite(best_l) and np.count_nonzero(best_w) > 0:
            best_l, best_w = self._polish(D, ils, best_l, best_w, t0)
            if POLALL > 1:
                # also polish the next best distinct start ends
                ends.sort(key=lambda t: t[0])
                seen_e = {best_w.tobytes()}
                cnt = 1
                for le, we in ends:
                    if cnt >= POLALL:
                        break
                    if we.tobytes() in seen_e:
                        continue
                    seen_e.add(we.tobytes())
                    cnt += 1
                    lp_, wp_ = self._polish(D, ils, le, we, t0)
                    if lp_ < best_l - 1e-12:
                        best_l, best_w = lp_, wp_
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
    # binary columns only: the byte code matrix from the start
    Xb2 = (rng.random((n, 16)) < 0.4).astype(float)
    yb2 = (Xb2[:, 0] + Xb2[:, 1] + rng.random(n) > 1.2).astype(np.int64)
    SparseIntegerClassifier(k=3, time_limit=1e9).fit(Xb2, yb2)
    # more than 256 codes in a column: the int32 code matrix
    n = 400
    X = np.hstack([(rng.random((n, 10)) < 0.4).astype(float), rng.random((n, 2))])
    y = (X[:, 0] + X[:, 10] + rng.random(n) > 1.2).astype(np.int64)
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
