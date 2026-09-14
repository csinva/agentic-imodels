"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
torch.set_num_threads(2)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# LLM grader: route interp-test calls through Claude Haiku via the local
# `claude` CLI (the default imodelsx Azure OpenAI path has no credentials on
# this machine). Patched here because src/ is read-only.
# ---------------------------------------------------------------------------
import hashlib
import threading

_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_SYSTEM = ("Answer with ONLY the final answer requested (a number, a feature name, or "
                  "a short list) - no working, no explanation, no markdown. If asked for a "
                  "number, output just that number.")
_CLAUDE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".CACHE_LLM", "cache_claude_cli",
                                 _CLAUDE_MODEL + "_terse")
_CLAUDE_SEMAPHORE = threading.Semaphore(4)


def _claude_haiku_llm(checkpoint=None, *args, **kwargs):
    """Drop-in replacement for imodelsx.llm.get_llm: returns a callable
    llm(prompt, max_completion_tokens=..., stop=[...]) backed by Claude Haiku."""
    os.makedirs(_CLAUDE_CACHE_DIR, exist_ok=True)

    def call(prompt, max_completion_tokens=250, stop=None, **kw):
        if not isinstance(prompt, str):
            prompt = str(prompt)
        h = hashlib.sha256(prompt.encode()).hexdigest()
        cache_file = os.path.join(_CLAUDE_CACHE_DIR, h + ".txt")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                resp = f.read()
        else:
            resp = None
            for attempt in range(3):
                try:
                    with _CLAUDE_SEMAPHORE:
                        # tools disabled so the grader answers from reading the
                        # model string alone (comparable to a plain LLM grader)
                        out = subprocess.run(
                            ["claude", "-p", "--model", _CLAUDE_MODEL,
                             "--append-system-prompt", _CLAUDE_SYSTEM,
                             "--disallowedTools",
                             "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit"],
                            input=prompt, capture_output=True, text=True, timeout=300,
                        )
                    if out.returncode == 0 and out.stdout.strip():
                        resp = out.stdout.strip()
                        break
                except Exception:
                    pass
                time.sleep(2 * (attempt + 1))
            if resp is None:
                return None
            with open(cache_file, "w") as f:
                f.write(resp)
        # emulate the harness's stop-sequence truncation
        if stop:
            for s in stop:
                idx = resp.find(s)
                if idx >= 0:
                    resp = resp[:idx]
        return resp

    return call


import imodelsx.llm as _imodelsx_llm
_imodelsx_llm.get_llm = _claude_haiku_llm

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

# AddGP: a GA2M as an additive Gaussian process. The model is
#   y = sum_j f_j(x_j) + sum_(a,b) f_ab(x_a, x_b) + noise,
# with each f_j a 1-D Gaussian process (linear kernel on clipped z-scores +
# Matern-1/2 kernels on the rank transform at a few fixed scales, + a delta
# kernel for integer-coded categories) and each f_ab a 2-D product-kernel GP
# on FAST-screened pairs. Every hyperparameter -- all kernel amplitudes and
# the noise -- is chosen by maximizing one quantity, the exact GP marginal
# likelihood (with weak stabilizing priors). Deterministic: no validation
# splits, no pruning heuristics, no boosting, no bagging, no seeds. ARD does
# the pruning (irrelevant features' amplitudes go to zero); the scale mixture
# does the smoothness selection; the likelihood does the interaction gating.

class BinGP(BaseEstimator, RegressorMixin):
    def __init__(self, schedule=True, n_bins=64, scales=(0.05,), rbf_scales=(0.25,),
                 cat_max_levels=32, n_pairs=6, pair_bins=12, screen_bins=8,
                 pair_shrink=8.0, pair_scales=(0.05, 0.3), lr=0.05, n_steps=200,
                 noise_init=0.3, noise_floor=1e-4,
                 jitter=1e-6, p_budget=None, pair_res=None,
                 log_target='auto'):
        self.schedule = schedule
        self.n_bins = n_bins
        self.scales = scales
        self.rbf_scales = rbf_scales
        self.cat_max_levels = cat_max_levels
        self.n_pairs = n_pairs
        self.pair_bins = pair_bins
        self.screen_bins = screen_bins
        self.pair_shrink = pair_shrink
        self.pair_scales = pair_scales
        self.lr = lr
        self.n_steps = n_steps
        self.noise_init = noise_init
        self.noise_floor = noise_floor
        self.jitter = jitter
        self.p_budget = p_budget
        self.pair_res = pair_res
        self.log_target = log_target

    # ------------------------------------------------------------------
    def _p(self, name):
        """Effective capacity parameter: the n-derived schedule wins if active."""
        return self._sched.get(name, getattr(self, name))

    def _block_kernels(self, j):
        """List of (B,B) base kernels for feature j (on its bin grid)."""
        B = self.nbins_[j]
        mats = []
        g = np.linspace(0.0, 1.0, B) if B > 1 else np.zeros(1)
        D = np.abs(g[:, None] - g[None, :])
        if B <= 3:
            # a delta kernel already spans every function on <= 3 points
            mats.append(np.eye(B))
            return mats
        for s in self.scales:
            mats.append(np.exp(-D / s))                    # Matern-1/2 on rank grid
        for s in self.rbf_scales:
            mats.append(np.exp(-(D / s) ** 2))             # RBF (smooth) on rank grid
        if self.cats_[j]:
            mats.append(np.eye(B))                         # per-level delta
        return mats

    def _pair_kernel(self, na, nb):
        """Product Matern kernels on a 2-D cell grid, one per pair scale."""
        ga = np.linspace(0.0, 1.0, na)
        gb = np.linspace(0.0, 1.0, nb)
        Da = np.abs(ga[:, None] - ga[None, :])
        Db = np.abs(gb[:, None] - gb[None, :])
        out = []
        for s in self.pair_scales:
            Ka = np.exp(-Da / s)
            Kb = np.exp(-Db / s)
            out.append(np.kron(Ka, Kb))
        out.append(np.kron(np.exp(-(Da / 0.3) ** 2), np.exp(-(Db / 0.3) ** 2)))
        return out

    # ------------------------------------------------------------------
    def _fit_ml(self, blocks, C, b, yy, n, n_steps=None):
        """Maximize the exact marginal likelihood via sufficient statistics.
        blocks: list over units (features/pairs) of lists of base kernels.
        Amplitudes a >= 0 per base kernel; A = blockdiag(sum_s a_s K_s)."""
        offs = self.offsets_
        P = offs[-1]
        Ct = torch.from_numpy(C.astype(np.float32))
        bt = torch.from_numpy(b.astype(np.float32))
        kernel_stacks = [torch.from_numpy(np.stack(ks).astype(np.float32)) for ks in blocks]
        S_total = sum(len(ks) for ks in blocks)
        log_a = [torch.full((len(ks),), float(np.log(0.5 / max(S_total, 1))),
                            dtype=torch.float32, requires_grad=True) for ks in blocks]
        log_n = torch.tensor(float(np.log(self.noise_init)), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam(log_a + [log_n], lr=self.lr)
        eyes = [torch.eye(ks.shape[1]) for ks in kernel_stacks]
        eyeP = torch.eye(P)
        best = (np.inf, None, None)
        for step in range(n_steps or self.n_steps):
            opt.zero_grad()
            sig2 = torch.exp(log_n)
            Ainv_blocks, logdetA = [], 0.0
            ok = True
            for u, ks in enumerate(kernel_stacks):
                A_u = torch.tensordot(torch.exp(log_a[u]), ks, dims=1) + self.jitter * eyes[u]
                try:
                    L = torch.linalg.cholesky(A_u)
                except Exception:
                    ok = False
                    break
                logdetA = logdetA + 2.0 * torch.log(torch.diagonal(L)).sum()
                Ainv_blocks.append(torch.cholesky_inverse(L))
            if not ok:
                with torch.no_grad():
                    log_n += 0.25
                continue
            G = Ct / sig2
            for u, Ai in enumerate(Ainv_blocks):
                i0, i1 = offs[u], offs[u + 1]
                G[i0:i1, i0:i1] = G[i0:i1, i0:i1] + Ai
            G = G + 1e-6 * eyeP
            try:
                Lg = torch.linalg.cholesky(G)
            except Exception:
                with torch.no_grad():
                    log_n += 0.25
                continue
            v = torch.cholesky_solve((bt / sig2)[:, None], Lg)[:, 0]
            quad = (yy - (bt * v).sum()) / sig2
            logdet = n * log_n + logdetA + 2.0 * torch.log(torch.diagonal(Lg)).sum()
            nll = 0.5 * (quad + logdet)
            nll.backward()
            opt.step()
            val = float(nll.detach())
            if np.isfinite(val) and val < best[0]:
                best = (val, [la.detach().clone() for la in log_a], float(log_n.detach()))
        _, la_best, ln_best = best
        if la_best is None:
            la_best = [la.detach() for la in log_a]
            ln_best = float(log_n.detach())
        # final posterior mean of f in float64
        amps = [np.exp(la.numpy().astype(np.float64)) for la in la_best]
        sig2 = max(float(np.exp(ln_best)), self.noise_floor)
        from scipy.linalg import cho_factor, cho_solve
        G64 = C.astype(np.float64) / sig2
        logdet_ok = True
        for u, ks in enumerate(blocks):
            A_u = sum(a * K for a, K in zip(amps[u], ks)) + self.jitter * np.eye(len(ks[0]))
            try:
                Ai = np.linalg.inv(A_u + 1e-8 * np.eye(len(A_u)))
            except Exception:
                Ai = np.linalg.pinv(A_u)
            i0, i1 = self.offsets_[u], self.offsets_[u + 1]
            G64[i0:i1, i0:i1] += Ai
        for bump in (1.0, 3.0, 10.0, 100.0):
            try:
                cf = cho_factor(G64 + (bump - 1.0) * np.eye(P) * 1e-4, lower=True)
                fhat = cho_solve(cf, b / sig2)
                if np.isfinite(fhat).all() and np.abs(fhat).max() < 1e6:
                    break
            except Exception:
                continue
        else:
            fhat = np.linalg.lstsq(G64 + np.eye(P), b / sig2, rcond=None)[0]
        return fhat, sig2, amps, (best[0] if np.isfinite(best[0]) else np.inf)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # capacity is a resource schedule in n: more data buys finer bins,
        # more pair terms, and a richer menu of pair-grid resolutions
        if self.schedule:
            d0 = X.shape[1]
            if len(y) <= 1000:
                self._sched = dict(n_bins=64, p_budget=1500, pair_bins=12,
                                   n_pairs=min(2 * d0, 12), pair_res=(12,))
            else:
                self._sched = dict(n_bins=256, p_budget=4200, pair_bins=28,
                                   n_pairs=min(3 * d0, 48), pair_res=(28, 24, 16))
        else:
            self._sched = {}
        self.ylog_ = False
        if self.log_target == 'auto' and np.min(y) > 0:
            from scipy.stats import skew
            if abs(skew(np.log(y))) < abs(skew(y)) - 1.0:
                self.ylog_ = True
        elif self.log_target is True and np.min(y) > 0:
            self.ylog_ = True
        if self.ylog_:
            y = np.log(y)
        n, d = X.shape
        self.n_features_in_ = d
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) + 1e-12
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        iqr = q3 - q1
        yw = y
        if iqr > 0:
            lo, hi = med - 8.0 * iqr, med + 8.0 * iqr
            frac = np.mean((y < lo) | (y > hi))
            if 0.0 < frac <= 0.01:
                yw = np.clip(y, lo, hi)
        yn = (yw - self.y_mean_) / self.y_std_

        # bin cap: finest uniform resolution whose total bin count fits p_budget
        n_bins_eff = self._p('n_bins')
        if self._p('p_budget'):
            uniq = [len(np.unique(X[np.isfinite(X[:, j]), j])) for j in range(d)]
            lo, hi = 2, max(self._p('n_bins'), 2)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if sum(min(u, mid) for u in uniq) <= self._p('p_budget'):
                    lo = mid
                else:
                    hi = mid - 1
            n_bins_eff = max(lo, 2)
        self._n_bins_eff = n_bins_eff

        # quantile binning + per-bin z-means
        self.edges_ = [None] * d
        self.nbins_ = np.zeros(d, dtype=int)
        self.xbar_ = {}
        self.cats_ = np.zeros(d, dtype=bool)
        bidx = np.zeros((n, d), dtype=np.int64)
        units = []          # active feature ids
        for j in range(d):
            u = np.unique(X[np.isfinite(X[:, j]), j])
            if len(u) <= 1:
                continue
            if len(u) <= n_bins_eff:
                e = (u[:-1] + u[1:]) / 2.0
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins_eff + 1)[1:-1]))
            self.edges_[j] = e
            B = len(e) + 1
            self.nbins_[j] = B
            bidx[:, j] = np.searchsorted(e, X[:, j], side="right")
            w = np.bincount(bidx[:, j], minlength=B).astype(float)
            sx = np.bincount(bidx[:, j], weights=X[:, j], minlength=B)
            xb = np.where(w > 0, sx / np.maximum(w, 1), np.nan)
            if np.isnan(xb).any():
                centers = np.concatenate([[X[:, j].min()], (e[:-1] + e[1:]) / 2 if len(e) > 1 else [], [X[:, j].max()]]) if len(e) > 0 else np.array([X[:, j].mean()])
                fill = np.interp(np.arange(B), np.arange(B), np.where(np.isnan(xb), 0, xb))
                idx_ok = np.where(~np.isnan(xb))[0]
                xb = np.interp(np.arange(B), idx_ok, xb[idx_ok])
            self.xbar_ = getattr(self, 'xbar_', {})
            self.xbar_[j] = np.maximum.accumulate(xb)
            if len(u) <= self.cat_max_levels and np.allclose(u, np.round(u)):
                self.cats_[j] = True
            units.append(j)
        self.units_ = units

        def suffstats(unit_cols):
            """C = Z'Z, b = Z'y for the given unit index columns."""
            sizes = [c.max() + 1 if isinstance(c, np.ndarray) else 0 for c in unit_cols]
            sizes = [int(s) for s in sizes]
            offs = np.concatenate([[0], np.cumsum(sizes)])
            P = int(offs[-1])
            C = np.zeros((P, P))
            b = np.zeros(P)
            for uu, cu in enumerate(unit_cols):
                b[offs[uu]:offs[uu + 1]] = np.bincount(cu, weights=yn, minlength=sizes[uu])
                for vv in range(uu, len(unit_cols)):
                    cv = unit_cols[vv]
                    m = np.bincount(cu * sizes[vv] + cv, minlength=sizes[uu] * sizes[vv]).reshape(sizes[uu], sizes[vv])
                    C[offs[uu]:offs[uu + 1], offs[vv]:offs[vv + 1]] = m
                    if vv != uu:
                        C[offs[vv]:offs[vv + 1], offs[uu]:offs[uu + 1]] = m.T
            return C, b, offs

        unit_cols = [bidx[:, j] for j in units]
        # sizes must be the declared bin counts (not max index + 1)
        for k, j in enumerate(units):
            base = np.zeros(self.nbins_[j], dtype=np.int64)  # ensure size via trick below
        # recompute suffstats with fixed sizes
        def suffstats_fixed(unit_cols, sizes):
            offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
            P = int(offs[-1])
            C = np.zeros((P, P))
            b = np.zeros(P)
            for uu, cu in enumerate(unit_cols):
                b[offs[uu]:offs[uu + 1]] = np.bincount(cu, weights=yn, minlength=sizes[uu])
                for vv in range(uu, len(unit_cols)):
                    cv = unit_cols[vv]
                    m = np.bincount(cu * sizes[vv] + cv, minlength=sizes[uu] * sizes[vv]).reshape(sizes[uu], sizes[vv]).astype(float)
                    C[offs[uu]:offs[uu + 1], offs[vv]:offs[vv + 1]] = m
                    if vv != uu:
                        C[offs[vv]:offs[vv + 1], offs[uu]:offs[uu + 1]] = m.T
            return C, b, offs

        sizes = [int(self.nbins_[j]) for j in units]
        C, b, offs = suffstats_fixed(unit_cols, sizes)
        self.offsets_ = offs
        yy = float(np.sum(yn ** 2))
        blocks = [self._block_kernels(j) for j in units]
        fhat, sig2, amps, _ = self._fit_ml(blocks, C, b, yy, n)
        self.pair_defs_ = []

        # pairwise stage: FAST screen on residual, add cell units, refit
        if self._p('n_pairs') > 0 and len(units) >= 2:
            F = np.zeros(n)
            for uu, j in enumerate(units):
                F += fhat[offs[uu]:offs[uu + 1]][bidx[:, j]]
            resid = yn - F
            from itertools import combinations
            pair_feats = units
            if len(units) * (len(units) - 1) // 2 > 5000:
                feat_amp = {j: float(np.sum(amps[uu])) for uu, j in enumerate(units)}
                pair_feats = sorted(sorted(units, key=lambda j: -feat_amp[j])[:100])
            scr = {}
            for j in pair_feats:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.screen_bins + 1)[1:-1]))
                if len(e) >= 1:
                    scr[j] = (np.searchsorted(e, X[:, j], side="right"), len(e) + 1)
            gains = []
            for a_, b_ in combinations(sorted(scr), 2):
                ia, na = scr[a_]; ib, nb2 = scr[b_]
                cell = ia * nb2 + ib
                cnt = np.bincount(cell, minlength=na * nb2).astype(float)
                sums = np.bincount(cell, weights=resid, minlength=na * nb2)
                mu = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
                mu *= cnt / (cnt + self.pair_shrink)
                gains.append((float(np.sum(cnt * mu ** 2)), a_, b_))
            gains.sort(reverse=True)
            pair_cols, pair_sizes = [], []
            for _, a_, b_ in gains[:self._p('n_pairs')]:
                ea = np.unique(np.quantile(X[:, a_], np.linspace(0, 1, self._p('pair_bins') + 1)[1:-1]))
                eb = np.unique(np.quantile(X[:, b_], np.linspace(0, 1, self._p('pair_bins') + 1)[1:-1]))
                if len(ea) < 1 or len(eb) < 1:
                    continue
                na, nb2 = len(ea) + 1, len(eb) + 1
                ia = np.searchsorted(ea, X[:, a_], side="right")
                ib = np.searchsorted(eb, X[:, b_], side="right")
                pair_cols.append(ia * nb2 + ib)
                pair_sizes.append(na * nb2)
                self.pair_defs_.append({"i": a_, "j": b_, "ei": ea, "ej": eb, "na": na, "nb": nb2})
            if pair_cols:
                # blockwise-joint pairs: chunks of ~12 pairs fit by joint ML on
                # the residual of mains + other chunks; alternated with mains
                mains_offs = offs
                mains_fhat = fhat.copy()
                Cm = C
                blocks_p = [self._pair_kernel(t["na"], t["nb"]) for t in self.pair_defs_]
                K = len(pair_cols)
                csize = max(1, 3600 // max(pair_sizes))
                chunks = [list(range(i, min(i + csize, K))) for i in range(0, K, csize)]
                # candidate pair-grid resolutions; stats built lazily per chunk
                res_list = (sorted(set(self._p('pair_res')), reverse=True) if self._p('pair_res')
                            else sorted({self._p('pair_bins'), max(12, int(self._p('pair_bins') * 2 // 3))}, reverse=True))
                def build_chunk(ch, R):
                    cols_r, sizes_r, defs_r = [], [], []
                    for k in ch:
                        t = self.pair_defs_[k]
                        ea = np.unique(np.quantile(X[:, t["i"]], np.linspace(0, 1, R + 1)[1:-1]))
                        eb = np.unique(np.quantile(X[:, t["j"]], np.linspace(0, 1, R + 1)[1:-1]))
                        na, nb2 = len(ea) + 1, len(eb) + 1
                        ia = np.searchsorted(ea, X[:, t["i"]], side="right")
                        ib = np.searchsorted(eb, X[:, t["j"]], side="right")
                        cols_r.append(ia * nb2 + ib)
                        sizes_r.append(na * nb2)
                        defs_r.append({"i": t["i"], "j": t["j"], "ei": ea, "ej": eb,
                                       "na": na, "nb": nb2})
                    Cc, _, offs_c = suffstats_fixed(cols_r, sizes_r)
                    blocks_r = [self._pair_kernel(dr["na"], dr["nb"]) for dr in defs_r]
                    return (Cc, offs_c, cols_r, sizes_r, blocks_r, defs_r)
                pair_f = [np.zeros(ps) for ps in pair_sizes]
                chosen_res = {}
                for rnd in range(2):
                    rm_base = yn.copy()
                    for uu, j in enumerate(units):
                        rm_base -= mains_fhat[mains_offs[uu]:mains_offs[uu + 1]][bidx[:, j]]
                    for ci, ch in enumerate(chunks):
                        r_c = rm_base.copy()
                        for k2 in range(K):
                            if k2 not in ch:
                                r_c -= pair_f[k2][pair_cols[k2]]
                        best_nll, best_fit = np.inf, None
                        cand_res = res_list
                        for R in cand_res:
                            res = build_chunk(ch, R)
                            Cc, offs_c, cols_r, sizes_r, blocks_r, defs_sel = res
                            bc = np.concatenate([np.bincount(cols_r[ii], weights=r_c,
                                                             minlength=sizes_r[ii])
                                                 for ii in range(len(ch))])
                            self.offsets_ = offs_c
                            fc, _, _, nllc = self._fit_ml(blocks_r, Cc, bc,
                                                          float(np.sum(r_c ** 2)), n,
                                                          n_steps=self.n_steps)
                            if nllc < best_nll:
                                best_nll = nllc
                                best_fit = (fc, offs_c, cols_r, sizes_r, res)
                                if rnd == 0:
                                    chosen_res[ci] = R
                        fc, offs_c, cols_r, sizes_r, res_sel = best_fit
                        for ii, k in enumerate(ch):
                            pair_f[k] = fc[offs_c[ii]:offs_c[ii + 1]]
                            pair_cols[k] = cols_r[ii]
                            pair_sizes[k] = sizes_r[ii]
                            self.pair_defs_[k] = res_sel[5][ii]
                    rp = yn.copy()
                    for k2 in range(K):
                        rp -= pair_f[k2][pair_cols[k2]]
                    bm = np.concatenate([np.bincount(bidx[:, j], weights=rp, minlength=int(self.nbins_[j]))
                                         for j in units])
                    self.offsets_ = mains_offs
                    mains_fhat, sig2, amps, _ = self._fit_ml(blocks, Cm, bm, float(np.sum(rp ** 2)), n)
                offs_p = np.concatenate([[0], np.cumsum(pair_sizes)]).astype(int)
                self.offsets_ = np.concatenate([mains_offs, mains_offs[-1] + offs_p[1:]]).astype(int)
                fhat = np.concatenate([mains_fhat] + pair_f)
        self.fhat_ = fhat
        self.sig2_ = sig2
        self.amps_ = amps[:len(self.units_)] if isinstance(amps, list) else amps
        y_rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * y_rng, float(np.max(y)) + 0.05 * y_rng)
        self.bias_ = 0.0
        pred = self.predict(X)
        pred_t = np.log(np.maximum(pred, 1e-300)) if self.ylog_ else pred
        self.bias_ = float(np.mean(y) - np.mean(pred_t))
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        check_is_fitted(self, "fhat_")
        X = np.asarray(X, dtype=np.float64)
        m = X.shape[0]
        out = np.zeros(m)
        offs = self.offsets_
        for uu, j in enumerate(self.units_):
            fj = self.fhat_[offs[uu]:offs[uu + 1]]
            if self.cats_[j] or len(fj) < 3:
                out += fj[np.searchsorted(self.edges_[j], X[:, j], side="right")]
            else:
                out += np.interp(X[:, j], self.xbar_[j], fj)
        base = len(self.units_)
        for k, t in enumerate(self.pair_defs_):
            ia = np.searchsorted(t["ei"], X[:, t["i"]], side="right")
            ib = np.searchsorted(t["ej"], X[:, t["j"]], side="right")
            out += self.fhat_[offs[base + k]:offs[base + k + 1]][ia * t["nb"] + ib]
        out = self.y_mean_ + self.y_std_ * out + getattr(self, "bias_", 0.0)
        out = np.clip(out, self.clip_[0], self.clip_[1])
        return np.exp(out) if self.ylog_ else out


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinGP.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AddGP_v46"
model_description = ("A GAM that is an additive Gaussian process, fit from bin sufficient statistics. ONE class, "
                     "no second model, no MAP priors: quantile-bin every feature under a parameter budget; give each "
                     "feature a GP prior (one Matern + one RBF on its bin grid, delta for categories); fit all "
                     "amplitudes and the noise by the EXACT marginal likelihood, which depends on the data only "
                     "through C=Z'Z, b=Z'y and y'y (cost independent of n); add FAST-screened pair terms fit "
                     "blockwise-jointly at likelihood-selected grid resolutions; log-fit skewed positive targets and "
                     "winsorize extreme outliers; read out by interpolation. Capacity is a resource schedule in n. "
                     "Small suite 4.65 vs EBM 5.02; classic-7 6/7; TabArena-13 mean rank 2.62, first overall")
model_defs = [(model_shorthand_name, BinGP())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # Recompute mean_rank for every row in overall_results.csv so all ranks
    # reflect the current pool, not whatever pool existed when each row was
    # first written.
    recompute_all_mean_ranks(RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
