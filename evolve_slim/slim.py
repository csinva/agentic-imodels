"""Sparse integer linear model (risk score) solver: the file the agent edits.

Starting point ``pyfasterrisk_v1``: FasterRisk (Liu et al., NeurIPS 2022,
github.com/jiachangliu/FasterRisk, BSD 3-Clause) flattened into one file with the
same algorithm and defaults:

1. sparse beam search: grow the support one feature at a time, expanding each of
   the ``parent_size`` best supports by the ``child_size`` features with the largest
   gradient and fine-tuning each child by coordinate descent (box-constrained so
   the coefficients stay within the point bounds);
2. diverse pool: swap each support feature for up to ``max_attempts`` others and keep
   the solutions whose loss is within ``gap_tolerance`` of the best;
3. star-ray search: for every pool solution, scale it by a grid of multipliers and
   round each scaled solution with sequential rounding (the auxiliary-loss rule);
   keep the integer solution with the smallest logistic loss.

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

MODEL_NAME = "pyfasterrisk_v1"
DESCRIPTION = "FasterRisk flattened into one file: beam search, diverse pool, star-ray search with sequential rounding"
INTEGER = True
COEF_BOUND = 5


# ---------------------------------------------------------------- helpers
def logistic_loss_from_exp(expyxb):
    return float(np.sum(np.log(1.0 + np.reciprocal(expyxb))))


def support_of(betas):
    return np.flatnonzero(np.abs(betas) > 1e-9)


def nonsupport_of(betas):
    return np.flatnonzero(np.abs(betas) <= 1e-9)


class LogRegModel:
    """Logistic regression on column-normalised features with box constraints."""

    def __init__(self, X, y, lambda2=1e-8, lb=-COEF_BOUND, ub=COEF_BOUND):
        self.X_mean = X.mean(axis=0)
        Xc = X - self.X_mean
        self.X_norm = np.linalg.norm(Xc, axis=0)
        self.scaled = np.flatnonzero(self.X_norm >= 1e-9)
        Xc[:, self.scaled] /= self.X_norm[self.scaled]
        self.n, self.p = Xc.shape
        self.y = y.astype(float)
        self.yXT = np.ascontiguousarray((self.y[:, None] * Xc).T)
        self.lambda2 = lambda2
        self.two_lambda2 = 2 * lambda2
        self.lipschitz = 0.25 + self.two_lambda2
        self.lbs = lb * np.ones(self.p)
        self.ubs = ub * np.ones(self.p)
        self.lbs[self.scaled] *= self.X_norm[self.scaled]
        self.ubs[self.scaled] *= self.X_norm[self.scaled]
        self.beta0, self.betas = 0.0, np.zeros(self.p)
        self.expyxb = np.ones(self.n)

    def to_original(self, beta0, betas):
        out = betas.copy()
        out[self.scaled] = out[self.scaled] / self.X_norm[self.scaled]
        return beta0 - self.X_mean @ out, out

    def step_coord(self, expyxb, betas, j):
        yx_j = self.yXT[j]
        grad = -np.inner(np.reciprocal(1 + expyxb), yx_j) + self.two_lambda2 * betas[j]
        new = max(self.lbs[j], min(self.ubs[j], betas[j] - grad / self.lipschitz))
        diff = new - betas[j]
        betas[j] = new
        expyxb *= np.exp(yx_j * diff)

    def finetune(self, expyxb, beta0, betas, steps=100):
        support = support_of(betas)
        grad = -self.yXT[support] @ np.reciprocal(1 + expyxb) + self.two_lambda2 * betas[support]
        support = support[np.argsort(-np.abs(grad))]
        before = logistic_loss_from_exp(expyxb) + self.lambda2 * betas[support] @ betas[support]
        for s in range(steps):
            g0 = -np.reciprocal(1 + expyxb) @ self.y
            d0 = g0 / (self.n * 0.25)
            beta0 -= d0
            expyxb *= np.exp(self.y * (-d0))
            for j in support:
                self.step_coord(expyxb, betas, j)
            if s % 10 == 0:
                after = logistic_loss_from_exp(expyxb) + self.lambda2 * betas[support] @ betas[support]
                if abs(before - after) / after < 1e-8:
                    break
                before = after
        return expyxb, beta0, betas


# ------------------------------------------------------------- beam search
def beam_search(model, k, parent_size=10, child_size=10):
    n, p = model.n, model.p
    npos = (np.sum(model.y) + n) / 2
    model.beta0 = np.log(npos / (n - npos))
    model.expyxb = np.exp(model.y * model.beta0)
    par_e = np.zeros((parent_size, n)); par_b0 = np.zeros(parent_size); par_b = np.zeros((parent_size, p))
    par_e[0], par_b0[0] = model.expyxb, model.beta0
    num_parent = 1
    total = parent_size * child_size
    ch_e = np.zeros((total, n)); ch_b0 = np.zeros(total); ch_b = np.zeros((total, p))
    ch_loss = np.full(total, 1e12)
    forbidden = set()
    for _ in range(min(k, p)):
        ch_loss.fill(1e12)
        added = 0
        for i in range(num_parent):
            non_support, support = nonsupport_of(par_b[i]), support_of(par_b[i])
            grad = model.yXT[non_support] @ np.reciprocal(1 + par_e[i])
            m = min(child_size, len(non_support))
            new_js = non_support[np.argsort(-np.abs(grad))][:m]
            lo, hi = i * child_size, i * child_size + m
            ch_e[lo:hi] = par_e[i]
            ch_b[lo:hi] = 0
            ch_b[lo:hi, support] = par_b[i, support]
            ch_b0[lo:hi] = par_b0[i]
            bnew = np.zeros(m)
            step, diff_max = 0, 1e3
            while step < 10 and diff_max > 1e-3:
                prev = bnew.copy()
                g = -np.sum(model.yXT[new_js] * np.reciprocal(1.0 + ch_e[lo:hi]), axis=1) + model.two_lambda2 * bnew
                bnew = np.clip(prev - g / model.lipschitz, model.lbs[new_js], model.ubs[new_js])
                diff = bnew - prev
                ch_e[lo:hi] *= np.exp(model.yXT[new_js] * diff[:, None])
                diff_max = np.max(np.abs(diff))
                step += 1
            for l in range(m):
                c = lo + l
                ch_b[c, new_js[l]] = bnew[l]
                key = str(support_of(ch_b[c]))
                if key not in forbidden:
                    added += 1
                    forbidden.add(key)
                    ch_e[c], ch_b0[c], ch_b[c] = model.finetune(ch_e[c], ch_b0[c], ch_b[c])
                    ch_loss[c] = logistic_loss_from_exp(ch_e[c])
        keep = np.argsort(ch_loss)[:min(parent_size, added)]
        num_parent = len(keep)
        par_e[:num_parent], par_b0[:num_parent], par_b[:num_parent] = ch_e[keep], ch_b0[keep], ch_b[keep]
    model.expyxb, model.beta0, model.betas = par_e[0].copy(), par_b0[0], par_b[0].copy()


# ------------------------------------------------------------ diverse pool
def diverse_pool(model, gap_tolerance=0.05, select_top_m=50, max_attempts=50):
    nz, z = support_of(model.betas), nonsupport_of(model.betas)
    attempts = min(max_attempts, len(z))
    total = 1 + len(nz) * attempts
    pool_b = np.zeros((total, model.p)); pool_b[:, nz] = model.betas[nz]
    pool_b0 = model.beta0 * np.ones(total)
    pool_e = np.zeros((total, model.n))
    pool_loss = np.full(total, 1e12)
    pool_e[-1] = model.expyxb
    pool_loss[-1] = logistic_loss_from_exp(model.expyxb) + model.lambda2 * model.betas[nz] @ model.betas[nz]
    sq = model.betas[nz] @ model.betas[nz]
    count = 1
    for num_old, old_j in enumerate(nz):
        lo, hi = num_old * attempts, (num_old + 1) * attempts
        pool_e[lo:hi] = model.expyxb * np.exp(-model.yXT[old_j] * model.betas[old_j])
        pool_b[lo:hi, old_j] = 0
        sq_wo = sq - model.betas[old_j] ** 2
        grad = -model.yXT[z] @ np.reciprocal(1 + pool_e[lo])
        new_js = z[np.argsort(-np.abs(grad))[:attempts]]
        for num_new, new_j in enumerate(new_js):
            idx = lo + num_new
            for _ in range(10):
                model.step_coord(pool_e[idx], pool_b[idx], new_j)
            loss = logistic_loss_from_exp(pool_e[idx]) + model.lambda2 * (sq_wo + pool_b[idx, new_j] ** 2)
            if (loss - pool_loss[-1]) / pool_loss[-1] < gap_tolerance:
                count += 1
                pool_e[idx], pool_b0[idx], pool_b[idx] = model.finetune(pool_e[idx], pool_b0[idx], pool_b[idx])
                pool_loss[idx] = logistic_loss_from_exp(pool_e[idx]) + model.lambda2 * (sq_wo + pool_b[idx, new_j] ** 2)
    sel = np.argsort(pool_loss)[:count][:select_top_m]
    out_b = np.zeros((len(sel), model.p))
    out_b[:, model.scaled] = pool_b[sel][:, model.scaled] / model.X_norm[model.scaled]
    out_b0 = pool_b0[sel] - out_b @ model.X_mean
    return out_b0, out_b


# ---------------------------------------------------------- star-ray search
class StarRay:
    def __init__(self, X, y, lb=-COEF_BOUND, ub=COEF_BOUND, num_ray_search=20, early_stop_tolerance=0.001):
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        self.yX = y[:, None] * X1
        self.p = X1.shape[1]
        self.ub_arr = ub * np.ones(self.p); self.ub_arr[0] = 100.0
        self.lb_arr = lb * np.ones(self.p); self.lb_arr[0] = -100.0
        self.num_ray_search = num_ray_search
        self.early_stop_tolerance = early_stop_tolerance

    def multipliers(self, betas, idx):
        pos, neg = betas > 1e-8, betas < -1e-8
        largest = 1e8
        if pos.any():
            largest = min(largest, np.min(self.ub_arr[idx][pos] / betas[pos]))
        if neg.any():
            largest = min(largest, np.min(self.lb_arr[idx][neg] / betas[neg]))
        if largest > 1:
            return np.linspace(1, largest, self.num_ray_search)
        return np.linspace(1, 0.5, self.num_ray_search)

    def line_search(self, betas):
        idx = support_of(betas)
        yx = self.yX[:, idx]
        b = betas[idx]
        loss_cont = float(np.sum(np.log1p(np.exp(-(yx @ b)))))
        best_loss, best_b = 1e12, np.zeros(len(idx))
        for mult in self.multipliers(b, idx):
            scaled = self.sequential_round(b * mult, yx / mult)
            loss = float(np.sum(np.log1p(np.exp(-(yx @ (scaled / mult))))))
            if loss < best_loss:
                best_loss, best_b, best_m = loss, scaled.copy(), mult
            if (loss - loss_cont) / loss_cont < self.early_stop_tolerance:
                break
        out = np.zeros(self.p)
        out[idx] = best_b
        return best_m, out

    @staticmethod
    def sequential_round(betas, yx):
        floor, ceil = np.floor(betas), np.ceil(betas)
        d_floor, d_ceil = floor - betas, ceil - betas
        todo = list(np.flatnonzero(floor != ceil))
        gamma = floor[None, :] + 1.0 * (yx <= 0)
        l_fac = np.reciprocal(1 + np.exp(np.sum(yx * gamma, axis=1)))
        lyx = l_fac[:, None] * yx
        lyx_sq = np.sum(lyx * lyx, axis=0)
        ub_arr = np.full(2 * yx.shape[1], 1e12)
        diff = np.zeros(yx.shape[0])
        cur = 0.0
        while todo:
            ub_arr.fill(1e12)
            for j in todo:
                expect = cur - lyx_sq[j] * d_floor[j] * d_ceil[j]
                ub_arr[2 * j + 1] = np.sum((diff + d_ceil[j] * lyx[:, j]) ** 2)
                if ub_arr[2 * j + 1] > expect:
                    ub_arr[2 * j] = np.sum((diff + d_floor[j] * lyx[:, j]) ** 2)
            best = int(np.argmin(ub_arr))
            cur = ub_arr[best]
            j, is_ceil = best // 2, best % 2
            step = d_ceil[j] if is_ceil else d_floor[j]
            betas[j] += step
            diff = diff + step * lyx[:, j]
            todo.remove(j)
        return betas


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
        ys = np.where(np.asarray(y) > 0, 1.0, -1.0)
        model = LogRegModel(X, ys)
        beam_search(model, self.k, self.parent_size, self.child_size)
        b0s, bs = diverse_pool(model, self.gap_tolerance, self.select_top_m, self.max_attempts)
        ray = StarRay(X, ys, num_ray_search=self.num_ray_search)
        best = (np.inf, None, None)
        seen = set()
        for b0, b in zip(b0s, bs):
            mult, sol = ray.line_search(np.concatenate([[b0], b]))
            key = support_of(sol).tobytes()
            if key in seen:
                continue
            seen.add(key)
            z = (sol[0] + X @ sol[1:]) / mult
            loss = float(np.sum(np.log1p(np.exp(-ys * z))))
            if loss < best[0]:
                best = (loss, sol, mult)
        _, sol, mult = best
        self.intercept_, self.coef_, self.multiplier_ = float(sol[0]), np.round(sol[1:]), float(mult)
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
