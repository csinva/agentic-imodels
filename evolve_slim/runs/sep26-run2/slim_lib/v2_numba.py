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

MODEL_NAME = "v2_numba"
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
def beam_search(D, k, parent_size=10, child_size=10):
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
                support = np.array(key, dtype=np.int64)
                loss = finetune(ym, b0, beta, support, D.X, D.y, D.c, D.lb, D.ub, 50, 1e-10)
                children.append((loss, ym, b0, beta))
        if not children:
            break
        children.sort(key=lambda t: t[0])
        parents = children[:parent_size]
    return parents[0]


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


# ------------------------------------------------------------------ model
class SparseIntegerClassifier:
    def __init__(self, k=5, time_limit=60.0, parent_size=10, child_size=10, select_top_m=50,
                 gap_tolerance=0.05, max_attempts=50, num_ray_search=20):
        self.k, self.time_limit = k, time_limit
        self.parent_size, self.child_size = parent_size, child_size
        self.select_top_m, self.gap_tolerance = select_top_m, gap_tolerance
        self.max_attempts, self.num_ray_search = max_attempts, num_ray_search

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        D = Data(X, y)
        best = beam_search(D, self.k, self.parent_size, self.child_size)
        pool = diverse_pool(D, best, self.gap_tolerance, self.select_top_m, self.max_attempts)
        seen = set()
        best_w, best_l = np.zeros(D.d), np.inf
        for _, b0, beta in pool:
            w = np.round(line_search(D, b0, beta, self.num_ray_search))
            key = w.tobytes()
            if key in seen:
                continue
            seen.add(key)
            l = calibrated_loss(D, w)
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
