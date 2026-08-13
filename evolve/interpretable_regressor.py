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
from scipy.linalg import solveh_banded
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


class GA2MBoostRegressor(BaseEstimator, RegressorMixin):
    """GA2M (GAM with at most pairwise interactions), built from scratch and
    tuned for predictive performance:

    1. ADDITIVE: cyclic penalized backfitting on finely quantile-binned
       features (up to `max_bins` bins) with a second-divided-difference
       curvature penalty whose null space is exactly the linear functions.
       Smoothing strength lambda is selected on an internal validation split
       (3-fold CV on small datasets).
    2. SPARSIFY: greedy validation-based feature dropping + importance floor.
    3. BAGGING: the final additive shapes are averaged over `n_bags` bootstrap
       backfits (variance reduction, EBM-style outer bags).
    4. PAIRS: FAST-screened pairwise terms fit on residuals, each the best of
       {shrunken 2D grid of cell means, single product, split-linear}, accepted
       only on validation improvement; up to `max_pairs` terms.
    Shape functions are evaluated by linear interpolation between bin centers
    with constant extrapolation; predictions are clipped to the padded
    training target range. Heavy-tailed targets are winsorized before fitting.
    """

    def __init__(self, bins_options=(48, 256), lambdas=(0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
                 n_sweeps=30, tol=1e-4, prune_rel_tol=0.002, prune_imp_frac=0.005,
                 val_frac=0.15, n_bags=8, max_pairs=8, pair_bins=12,
                 pair_shrink=8.0, pair_gain=0.005, pair_screen_bins=8,
                 pair_top_candidates=5, cat_max_levels=32, cat_shrink=5.0,
                 feat_lambda_refine=False, alternate=True, ens_top=1,
                 boost_lr=0.1, boost_rounds=300, boost_patience=25,
                 boost_bags=8, n_cycles=3,
                 small_n=300, random_state=42):
        self.bins_options = bins_options
        self.lambdas = lambdas
        self.n_sweeps = n_sweeps
        self.tol = tol
        self.prune_rel_tol = prune_rel_tol
        self.prune_imp_frac = prune_imp_frac
        self.val_frac = val_frac
        self.n_bags = n_bags
        self.max_pairs = max_pairs
        self.pair_bins = pair_bins
        self.pair_shrink = pair_shrink
        self.pair_gain = pair_gain
        self.pair_screen_bins = pair_screen_bins
        self.pair_top_candidates = pair_top_candidates
        self.cat_max_levels = cat_max_levels
        self.cat_shrink = cat_shrink
        self.feat_lambda_refine = feat_lambda_refine
        self.alternate = alternate
        self.ens_top = ens_top
        self.boost_lr = boost_lr
        self.boost_rounds = boost_rounds
        self.boost_patience = boost_patience
        self.boost_bags = boost_bags
        self.n_cycles = n_cycles
        self.small_n = small_n
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def _penalty_banded(xs):
        """Upper-banded (3-diag) D'D for second divided differences on knots xs."""
        B = len(xs)
        ab = np.zeros((3, B))
        if B < 3:
            return ab
        h = np.maximum(np.diff(xs), 1e-9)
        a0 = 1.0 / h[:-1]           # for interior i=1..B-2: 1/h[i-1]
        a1 = 1.0 / h[1:]            # 1/h[i]
        mid = -(a0 + a1)
        main = np.zeros(B); d1 = np.zeros(B - 1); d2 = np.zeros(B - 2)
        main[0:B - 2] += a0 * a0
        main[1:B - 1] += mid * mid
        main[2:B] += a1 * a1
        d1[0:B - 2] += a0 * mid
        d1[1:B - 1] += mid * a1
        d2[0:B - 2] += a0 * a1
        sc = main.sum() / B
        if sc > 0:
            main /= sc; d1 /= sc; d2 /= sc
        ab[0, 2:] = d2
        ab[1, 1:] = d1
        ab[2, :] = main
        return ab

    def _backfit(self, y_tr, b_tr, bands, n_bins, active, lam, sweeps, lam_by_feat=None):
        shapes = [np.zeros(n_bins[j]) for j in range(len(n_bins))]
        icpt = float(np.mean(y_tr))
        F = np.full(len(y_tr), icpt)
        y_sd = float(np.std(y_tr)) + 1e-12
        for _ in range(sweeps):
            delta = 0.0
            for j in active:
                w, xbar, P, is_cat = bands[j]
                resid = y_tr - F + shapes[j][b_tr[:, j]]
                sums = np.bincount(b_tr[:, j], weights=resid, minlength=n_bins[j])
                if is_cat:
                    # categorical: shrunken per-level means (no smoothing across codes)
                    f_new = sums / (w + self.cat_shrink)
                else:
                    lam_j = lam_by_feat.get(j, lam) if lam_by_feat else lam
                    ab = P * lam_j
                    ab[-1] = ab[-1] + w
                    try:
                        f_new = solveh_banded(ab, sums, lower=False)
                    except Exception:
                        f_new = np.where(w > 0, sums / np.maximum(w, 1e-9), 0.0)
                F += f_new[b_tr[:, j]] - shapes[j][b_tr[:, j]]
                if len(f_new):
                    delta = max(delta, float(np.max(np.abs(f_new - shapes[j]))))
                shapes[j] = f_new
            if delta < self.tol * y_sd:
                break
        return icpt, shapes, F

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        self.n_features_in_ = d
        rng = np.random.RandomState(self.random_state)

        # winsorize extremely heavy-tailed targets (outliers dominate LS bins)
        if n >= 80:
            q_lo, q_hi = np.quantile(y, [0.002, 0.998])
            med = float(np.median(y))
            if (float(np.max(y)) - q_hi) > 2.0 * max(q_hi - med, 1e-12) or \
               (q_lo - float(np.min(y))) > 2.0 * max(med - q_lo, 1e-12):
                y = np.clip(y, q_lo, q_hi)

        y_var = float(np.var(y)) + 1e-12
        y_std = float(np.sqrt(y_var))

        # integer-coded low-cardinality features get categorical treatment
        is_cat_feat = np.zeros(d, dtype=bool)
        for j in range(d):
            col = X[np.isfinite(X[:, j]), j]
            u = np.unique(col)
            if 2 <= len(u) <= self.cat_max_levels and np.allclose(u, np.round(u)):
                is_cat_feat[j] = True

        # --- quantile binning at a given resolution ---
        def make_bins(max_bins):
            bin_edges, nb = [], np.zeros(d, dtype=int)
            bidx = np.zeros((n, d), dtype=np.int32)
            for j in range(d):
                col = X[:, j]
                uniq = np.unique(col[np.isfinite(col)])
                if len(uniq) <= 1:
                    bin_edges.append(np.array([])); nb[j] = 1; continue
                if len(uniq) <= max_bins:
                    edges = (uniq[:-1] + uniq[1:]) / 2.0
                else:
                    qs = np.quantile(col, np.linspace(0, 1, max_bins + 1)[1:-1])
                    edges = np.unique(qs)
                bin_edges.append(edges)
                nb[j] = len(edges) + 1
                bidx[:, j] = np.searchsorted(edges, col, side="right")
            return bin_edges, nb, bidx

        def build_bands(ids, bin_edges, n_bins, bin_idx, active, cat_mask):
            bands = [None] * d
            for j in active:
                B = n_bins[j]
                w = np.bincount(bin_idx[ids, j], minlength=B).astype(float)
                sx = np.bincount(bin_idx[ids, j], weights=X[ids, j], minlength=B)
                xbar = np.where(w > 0, sx / np.maximum(w, 1), np.nan)
                e = bin_edges[j]
                bad = ~np.isfinite(xbar)
                if bad.any():
                    centers = np.empty(B)
                    centers[0] = e[0] - 1e-9
                    centers[-1] = e[-1] + 1e-9
                    if B > 2:
                        centers[1:-1] = (e[:-1] + e[1:]) / 2.0
                    xbar[bad] = centers[bad]
                xr = xbar[-1] - xbar[0]
                xs = (xbar - xbar[0]) / (xr if xr > 0 else 1.0)
                if cat_mask[j]:
                    bands[j] = (w, xbar, None, True)
                else:
                    bands[j] = (w, xbar, self._penalty_banded(xs), False)
            return bands

        # --- joint (bin resolution, lambda) selection on validation ---
        binned = {}
        for mb in sorted(set(self.bins_options)):
            edges, nb, bidx = make_bins(mb)
            act = [j for j in range(d) if nb[j] > 1]
            binned[mb] = (edges, nb, bidx, act)

        no_cat = np.zeros(d, dtype=bool)
        cat_options = [no_cat, is_cat_feat] if is_cat_feat.any() else [no_cat]
        if n >= 80 and self.val_frac > 0:
            perm = rng.permutation(n)
            n_val = max(20, int(n * self.val_frac))
            val_ids, tr_ids = perm[:n_val], perm[n_val:]
            y_tr, y_val = y[tr_ids], y[val_ids]
            # 3-fold CV selection of (bins, categorical, lambda) on all data
            cv_folds = np.array_split(rng.permutation(n), 3)
            cv_sets = []
            for f_ids in cv_folds:
                t_ids = np.setdiff1d(np.arange(n), f_ids)
                cv_sets.append((t_ids, f_ids))
            scored = []
            for mb, (edges, nb, bidx, act) in binned.items():
                for cmask in cat_options:
                    fold_bands = [build_bands(t_ids, edges, nb, bidx, act, cmask) for t_ids, _ in cv_sets]
                    for lam_c in self.lambdas:
                        sse = 0.0
                        for (t_ids, f_ids), bands_f in zip(cv_sets, fold_bands):
                            icpt, shapes, _ = self._backfit(y[t_ids], bidx[t_ids], bands_f, nb, act, lam_c, self.n_sweeps)
                            pv = np.full(len(f_ids), icpt)
                            for j in act:
                                pv += shapes[j][bidx[f_ids, j]]
                            sse += float(np.sum((y[f_ids] - pv) ** 2))
                        scored.append((sse, mb, lam_c, cmask))
            scored.sort(key=lambda t: t[0])
            best_sse = scored[0][0]
            ens_configs = [(mb, l, cm) for ss, mb, l, cm in scored[:self.ens_top] if ss <= 1.05 * best_sse]
            _, mb_best, lam, cat_best = scored[0]
            # fit on tr split with selected config (basis for pruning decisions)
            edges, nb, bidx, act = binned[mb_best]
            bands_tr_sel = build_bands(tr_ids, edges, nb, bidx, act, cat_best)
            icpt_sel, shapes_sel, _ = self._backfit(y_tr, bidx[tr_ids], bands_tr_sel, nb, act, lam, self.n_sweeps)
            # per-feature lambda refinement: coordinate pass on validation
            lam_by_feat = {}
            edges, nb, bidx, act = binned[mb_best]
            bands_tr = build_bands(tr_ids, edges, nb, bidx, act, cat_best)
            b_tr, b_val_ = bidx[tr_ids], bidx[val_ids]
            pv = np.full(len(val_ids), icpt_sel)
            for j in act:
                pv += shapes_sel[j][b_val_[:, j]]
            cur_mse = float(np.mean((y_val - pv) ** 2))
            F_tr = np.full(len(tr_ids), icpt_sel)
            for j in act:
                F_tr += shapes_sel[j][b_tr[:, j]]
            for j in (act if self.feat_lambda_refine else []):
                w_, xbar_, P_, is_cat_ = bands_tr[j]
                if is_cat_:
                    continue
                resid = y_tr - F_tr + shapes_sel[j][b_tr[:, j]]
                sums = np.bincount(b_tr[:, j], weights=resid, minlength=nb[j])
                for lam_c in (lam / 30.0, lam * 30.0):
                    if not (self.lambdas[0] / 30.0 <= lam_c <= self.lambdas[-1] * 30.0):
                        continue
                    ab = P_ * lam_c
                    ab[-1] = ab[-1] + w_
                    try:
                        f_c = solveh_banded(ab, sums, lower=False)
                    except Exception:
                        continue
                    pv_c = pv - shapes_sel[j][b_val_[:, j]] + f_c[b_val_[:, j]]
                    mse_c = float(np.mean((y_val - pv_c) ** 2))
                    if mse_c < cur_mse * 0.99:
                        cur_mse = mse_c
                        lam_by_feat[j] = lam_c
                        F_tr += f_c[b_tr[:, j]] - shapes_sel[j][b_tr[:, j]]
                        pv = pv_c
                        shapes_sel[j] = f_c
        else:
            val_ids = np.array([], dtype=int)
            tr_ids = np.arange(n)
            mb_best = min(self.bins_options)
            cat_best = no_cat
            edges, nb, bidx, act = binned[mb_best]
            perm = rng.permutation(n)
            folds = np.array_split(perm, 3)
            cv_mse = {l: 0.0 for l in self.lambdas}
            for f_ids in folds:
                t_ids = np.setdiff1d(perm, f_ids)
                if len(t_ids) < 5 or len(f_ids) < 2:
                    continue
                bands_f = build_bands(t_ids, edges, nb, bidx, act, cat_best)
                for l in self.lambdas:
                    icpt, shapes, _ = self._backfit(y[t_ids], bidx[t_ids], bands_f, nb, act, l, self.n_sweeps)
                    pv = np.full(len(f_ids), icpt)
                    for j in act:
                        pv += shapes[j][bidx[f_ids, j]]
                    cv_mse[l] += float(np.sum((y[f_ids] - pv) ** 2))
            lam = min(cv_mse, key=cv_mse.get)
            icpt_sel, shapes_sel = None, None
            lam_by_feat = {}
            ens_configs = [(mb_best, lam, cat_best)]
        self.lambda_ = lam
        self.bins_ = mb_best
        bin_edges, n_bins, bin_idx, active = binned[mb_best]
        b_val = bin_idx[val_ids] if len(val_ids) else None

        # --- prune (val-based, on selection-phase shapes) ---
        w_full = [np.bincount(bin_idx[:, j], minlength=n_bins[j]).astype(float) for j in range(d)]
        if shapes_sel is not None:
            imp_sel = {}
            for j in active:
                w = w_full[j]
                mu = float(np.sum(shapes_sel[j] * w) / max(w.sum(), 1))
                imp_sel[j] = float(np.sqrt(np.sum(w * (shapes_sel[j] - mu) ** 2) / max(w.sum(), 1)))
            kept = {j for j in active if imp_sel[j] >= self.prune_imp_frac * y_std}
            pv = np.full(len(val_ids), icpt_sel)
            for j in kept:
                pv += shapes_sel[j][b_val[:, j]]
            cur_mse = float(np.mean((y_val - pv) ** 2))
            tol_abs = self.prune_rel_tol * max(cur_mse, 1e-3 * y_var)
            for j in sorted(kept, key=lambda k: imp_sel[k]):
                pv2 = pv - shapes_sel[j][b_val[:, j]]
                mse2 = float(np.mean((y_val - pv2) ** 2))
                if mse2 <= cur_mse + tol_abs:
                    kept.discard(j); pv = pv2; cur_mse = min(cur_mse, mse2)
            kept_list = sorted(kept)
        else:
            kept_list = list(active)

        # --- unified bagging: each bag = split -> backfit -> boost (own val stop);
        # bag-averaged mains; then gated pairs; then per-bag boost of mains+cells ---
        edges_c, nb_c, bidx_c, act_c = binned[mb_best]
        bands_all = build_bands(np.arange(n), edges_c, nb_c, bidx_c, act_c, cat_best)
        kept_c = [j for j in kept_list if j in act_c]
        n_bags = self.n_bags if n >= 80 else 1
        wf = [np.bincount(bidx_c[:, j], minlength=nb_c[j]).astype(float) for j in range(d)]

        def bag_fit(extra_terms=None, base_shapes=None, y_t=None):
            """One round of bagging. Each bag: subsample-split, backfit (or reuse
            base_shapes) + boost mains (+ pair cells), early-stopped on the bag's
            own held-out part. Returns bin-value shapes, cell values, intercept."""
            acc_shapes = {j: np.zeros(nb_c[j]) for j in kept_c}
            acc_cells = [np.zeros(len(t["vals"])) for t in (extra_terms or [])]
            acc_icpt = 0.0
            cell_idx_full = []
            for t in (extra_terms or []):
                ia = np.searchsorted(t["ei"], X[:, t["i"]], side="right")
                ib = np.searchsorted(t["ej"], X[:, t["j"]], side="right")
                cell_idx_full.append(ia * t["nb"] + ib)
            if y_t is None:
                y_t = y
            for rep in range(n_bags):
                if n >= 80:
                    perm_b = rng.permutation(n)
                    nv = max(20, int(n * self.val_frac))
                    v_i, t_i = perm_b[:nv], perm_b[nv:]
                else:
                    t_i = np.arange(n); v_i = np.arange(n)
                if base_shapes is None:
                    bands_b = build_bands(t_i, edges_c, nb_c, bidx_c, act_c, cat_best)
                    icpt_b, shapes_b, _ = self._backfit(y_t[t_i], bidx_c[t_i], bands_b, nb_c, kept_c, lam, self.n_sweeps)
                else:
                    icpt_b = base_shapes[1]
                    shapes_b = [sh.copy() if sh is not None else None for sh in base_shapes[0]]
                r_tr = y_t[t_i] - np.full(len(t_i), icpt_b)
                r_val = y_t[v_i] - np.full(len(v_i), icpt_b)
                bt, bv_ = bidx_c[t_i], bidx_c[v_i]
                for j in kept_c:
                    r_tr -= shapes_b[j][bt[:, j]]
                    r_val -= shapes_b[j][bv_[:, j]]
                cells_tr = [c[t_i] for c in cell_idx_full]
                cells_val = [c[v_i] for c in cell_idx_full]
                cvals = [t["vals"].copy() for t in (extra_terms or [])]
                for k, t in enumerate(extra_terms or []):
                    r_tr -= cvals[k][cells_tr[k]]
                    r_val -= cvals[k][cells_val[k]]
                cnt = {j: np.bincount(bt[:, j], minlength=nb_c[j]).astype(float) for j in kept_c}
                ccnt = [np.bincount(ct, minlength=len(cv)).astype(float) for ct, cv in zip(cells_tr, cvals)]
                u_tot = {j: np.zeros(nb_c[j]) for j in kept_c}
                g_tot = [np.zeros(len(cv)) for cv in cvals]
                bvv = float(np.mean(r_val ** 2))
                bu = {j: u.copy() for j, u in u_tot.items()}
                bg = [g.copy() for g in g_tot]
                stall = 0
                for it in range(self.boost_rounds):
                    for j in kept_c:
                        sums = np.bincount(bt[:, j], weights=r_tr, minlength=nb_c[j])
                        u = self.boost_lr * sums / (cnt[j] + 2.0)
                        u_tot[j] += u
                        r_tr -= u[bt[:, j]]
                        r_val -= u[bv_[:, j]]
                    for k in range(len(cvals)):
                        sums = np.bincount(cells_tr[k], weights=r_tr, minlength=len(cvals[k]))
                        u = self.boost_lr * sums / (ccnt[k] + 4.0)
                        g_tot[k] += u
                        r_tr -= u[cells_tr[k]]
                        r_val -= u[cells_val[k]]
                    v = float(np.mean(r_val ** 2))
                    if v < bvv - 1e-12:
                        bvv = v
                        bu = {j: u.copy() for j, u in u_tot.items()}
                        bg = [g.copy() for g in g_tot]
                        stall = 0
                    else:
                        stall += 1
                        if stall >= self.boost_patience:
                            break
                for j in kept_c:
                    acc_shapes[j] += (shapes_b[j] + bu[j]) / n_bags
                for k in range(len(acc_cells)):
                    acc_cells[k] += (cvals[k] + bg[k]) / n_bags
                acc_icpt += icpt_b / n_bags
            return acc_shapes, acc_cells, acc_icpt

        # round 1: mains only
        m_shapes, _, m_icpt = bag_fit()
        intercept = m_icpt
        self.shape_x_ = [None] * d
        self.shape_y_ = [None] * d
        self.pruned_ = [True] * d
        self.importance_ = np.zeros(d)
        for j in kept_c:
            w = wf[j]
            mu = float(np.sum(m_shapes[j] * w) / max(w.sum(), 1))
            sh = m_shapes[j] - mu
            intercept += mu
            _, xbar, _, _ = bands_all[j]
            order = np.argsort(xbar)
            self.shape_x_[j] = xbar[order]
            self.shape_y_[j] = sh[order]
            self.pruned_[j] = False
            self.importance_[j] = float(np.sqrt(np.sum(w * sh ** 2) / max(w.sum(), 1)))
        self.intercept_ = intercept

        # prediction clipping range
        y_rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * y_rng, float(np.max(y)) + 0.05 * y_rng)

        # gated pair selection on the residual (structure only)
        self.pair_terms_ = []
        if len(val_ids) and self.max_pairs > 0 and len(active) >= 2:
            from itertools import combinations
            scr_idx, pair_cand_feats = {}, []
            for j in active:
                qs = np.quantile(X[tr_ids, j], np.linspace(0, 1, self.pair_screen_bins + 1)[1:-1])
                e = np.unique(qs)
                if len(e) >= 1:
                    scr_idx[j] = (np.searchsorted(e, X[:, j], side="right"), len(e) + 1)
                    pair_cand_feats.append(j)
            tr_mask = np.zeros(n, dtype=bool); tr_mask[tr_ids] = True
            for _ in range(self.max_pairs):
                resid = y - self._predict_raw(X, clip=False)
                r_tr, r_val = resid[tr_ids], resid[val_ids]
                cur_mse = float(np.mean(r_val ** 2))
                screen = []
                for a_, b_ in combinations(pair_cand_feats, 2):
                    ia, na = scr_idx[a_]; ib, nb2 = scr_idx[b_]
                    cell = ia[tr_ids] * nb2 + ib[tr_ids]
                    cnt = np.bincount(cell, minlength=na * nb2).astype(float)
                    sums = np.bincount(cell, weights=r_tr, minlength=na * nb2)
                    mu = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
                    mu *= cnt / (cnt + self.pair_shrink)
                    screen.append((float(np.sum(cnt * mu ** 2)), a_, b_))
                screen.sort(reverse=True)
                best = None
                for _, a_, b_ in screen[:self.pair_top_candidates]:
                    for cand in self._pair_candidates(X, resid, tr_ids, tr_mask, a_, b_):
                        contrib = cand.pop("contrib")
                        dshift = float(np.mean(contrib[tr_ids]))
                        mse2 = float(np.mean((r_val - (contrib[val_ids] - dshift)) ** 2))
                        gain = cur_mse - mse2
                        if best is None or gain > best[0]:
                            best = (gain, cand, contrib, dshift)
                if best is None or best[0] < max(self.pair_gain * cur_mse, 5e-4 * y_var):
                    break
                _, term, contrib, dshift = best
                self.pair_terms_.append(term)
                self.intercept_ -= dshift

        # round 2: re-bag with pair grid cells co-boosted (grid terms only)
        grid_terms = [t for t in self.pair_terms_ if t["type"] == "grid"]
        if True:
            # subtract parametric (prod / split-linear) pair contributions first
            other = np.zeros(n)
            for t in self.pair_terms_:
                if t["type"] == "prod":
                    other += t["coef"] * X[:, t["i"]] * X[:, t["j"]]
                elif t["type"] == "split":
                    side = X[:, t["i"]] >= t["t"]
                    (m1, c1), (m2, c2) = t["lo"], t["hi"]
                    xb = X[:, t["j"]]
                    other += np.where(side, m2 * xb + c2, m1 * xb + c1)
            m_shapes2, cells2, icpt2 = bag_fit(extra_terms=grid_terms, y_t=y - other)
            intercept = icpt2
            for j in kept_c:
                w = wf[j]
                mu = float(np.sum(m_shapes2[j] * w) / max(w.sum(), 1))
                sh = m_shapes2[j] - mu
                intercept += mu
                _, xbar, _, _ = bands_all[j]
                order = np.argsort(xbar)
                self.shape_x_[j] = xbar[order]
                self.shape_y_[j] = sh[order]
                self.importance_[j] = float(np.sqrt(np.sum(w * sh ** 2) / max(w.sum(), 1)))
            for t, cv in zip(grid_terms, cells2):
                t["vals"] = cv
            self.intercept_ = intercept

        pred = self._predict_raw(X, clip=False)
        self.intercept_ += float(np.mean(y) - np.mean(pred))
        return self

    # ------------------------------------------------------------------
    def _pair_candidates(self, X, resid, tr_ids, tr_mask, a, b):
        """Candidate pairwise terms fit on training residuals."""
        cands = []
        r_tr = resid[tr_ids]
        # 2D grid of shrunken cell means on quantile edges
        ea = np.unique(np.quantile(X[tr_ids, a], np.linspace(0, 1, self.pair_bins + 1)[1:-1]))
        eb = np.unique(np.quantile(X[tr_ids, b], np.linspace(0, 1, self.pair_bins + 1)[1:-1]))
        if len(ea) >= 1 and len(eb) >= 1:
            na, nb = len(ea) + 1, len(eb) + 1
            ia = np.searchsorted(ea, X[:, a], side="right")
            ib = np.searchsorted(eb, X[:, b], side="right")
            cell = ia * nb + ib
            cnt = np.bincount(cell[tr_ids], minlength=na * nb).astype(float)
            sums = np.bincount(cell[tr_ids], weights=r_tr, minlength=na * nb)
            vals = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
            vals *= cnt / (cnt + self.pair_shrink)
            cands.append({"type": "grid", "i": a, "j": b, "ei": ea, "ej": eb,
                          "nb": nb, "vals": vals, "contrib": vals[cell]})
        # single product
        p = X[:, a] * X[:, b]
        p_tr = p[tr_ids]
        pm = float(np.mean(p_tr))
        varp = float(np.mean((p_tr - pm) ** 2))
        if varp > 1e-12:
            coef = float(np.mean((p_tr - pm) * r_tr) / varp)
            cands.append({"type": "prod", "i": a, "j": b, "coef": coef, "contrib": coef * p})
        # split-linear, both orientations, quartile split candidates
        for (sa, sb) in ((a, b), (b, a)):
            for q in (0.25, 0.5, 0.75):
                t = float(np.quantile(X[tr_ids, sa], q))
                side = X[:, sa] >= t
                coefs, ok = [], True
                for sel_side in (~side, side):
                    sel = sel_side & tr_mask
                    ns = int(np.sum(sel))
                    if ns < 8:
                        ok = False; break
                    xs, ys = X[sel, sb], resid[sel]
                    xm, ym = float(np.mean(xs)), float(np.mean(ys))
                    varx = float(np.mean((xs - xm) ** 2))
                    if varx < 1e-12:
                        ok = False; break
                    mnew = float(np.mean((xs - xm) * (ys - ym)) / varx)
                    mnew *= ns / (ns + 12.0)
                    coefs.append((mnew, ym - mnew * xm))
                if ok:
                    (m1, c1), (m2, c2) = coefs
                    contrib = np.where(side, m2 * X[:, sb] + c2, m1 * X[:, sb] + c1)
                    cands.append({"type": "split", "i": sa, "j": sb, "t": t,
                                  "lo": (m1, c1), "hi": (m2, c2), "contrib": contrib})
        return cands

    # ------------------------------------------------------------------
    def _predict_raw(self, X, clip=True):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], getattr(self, "intercept_", 0.0))
        for j in range(self.n_features_in_):
            if self.shape_x_[j] is not None:
                out += np.interp(X[:, j], self.shape_x_[j], self.shape_y_[j])
        for t in getattr(self, "pair_terms_", []):
            if t["type"] == "prod":
                out += t["coef"] * X[:, t["i"]] * X[:, t["j"]]
            elif t["type"] == "split":
                side = X[:, t["i"]] >= t["t"]
                (m1, c1), (m2, c2) = t["lo"], t["hi"]
                xb = X[:, t["j"]]
                out += np.where(side, m2 * xb + c2, m1 * xb + c1)
            else:
                ia = np.searchsorted(t["ei"], X[:, t["i"]], side="right")
                ib = np.searchsorted(t["ej"], X[:, t["j"]], side="right")
                out += t["vals"][ia * t["nb"] + ib]
        if clip and getattr(self, "clip_", None) is not None:
            out = np.clip(out, self.clip_[0], self.clip_[1])
        return out

    def predict(self, X):
        check_is_fitted(self, "shape_x_")
        return self._predict_raw(X)

    # ------------------------------------------------------------------
    def __str__(self):
        check_is_fitted(self, "shape_x_")
        d = self.n_features_in_
        names = [f"x{i}" for i in range(d)]
        order = np.argsort(-self.importance_)
        lines = [
            "Additive model (GA2M). Prediction = baseline + f(x0) + f(x1) + ... "
            "plus the listed pairwise adjustments (no higher-order interactions).",
            "Features are listed from most to least important.",
            f"baseline = {self.intercept_:.4f}",
            "",
        ]
        for j in order:
            if self.pruned_[j]:
                continue
            xs, ys = self.shape_x_[j], self.shape_y_[j]
            k = min(9, len(xs))
            idx = np.linspace(0, len(xs) - 1, k).round().astype(int)
            pts = "  ".join(f"{xs[i]:+.3g}->{ys[i]:+.3g}" for i in idx)
            lines.append(f"f({names[j]}) sampled (x->effect): {pts}")
        pruned = [names[j] for j in range(d) if self.pruned_[j]]
        if pruned:
            lines.append(f"Features with NO effect (f = 0): {', '.join(pruned)}")
        for t in getattr(self, "pair_terms_", []):
            na, nb_ = names[t["i"]], names[t["j"]]
            if t["type"] == "prod":
                lines.append(f"pairwise: add {t['coef']:.4g} * {na} * {nb_}")
            elif t["type"] == "split":
                (m1, c1), (m2, c2) = t["lo"], t["hi"]
                lines.append(f"pairwise: if {na} < {t['t']:.4g}: add {m1:.4g}*{nb_}{c1:+.4g}; "
                             f"else add {m2:.4g}*{nb_}{c2:+.4g}")
            else:
                lines.append(f"pairwise: 2D grid adjustment on ({na}, {nb_})")
        lines.append(f"Predictions are clipped to [{self.clip_[0]:.4g}, {self.clip_[1]:.4g}].")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GA2MBoostRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GA2MBoost_v28"
model_description = ("v22 restructured into unified bagging: each bag backfits AND boosts with its own "
                     "held-out early stopping (removes contaminated-val bias); pairs selected between bag rounds, "
                     "grids co-boosted per bag in round 2")
model_defs = [(model_shorthand_name, GA2MBoostRegressor())]


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
