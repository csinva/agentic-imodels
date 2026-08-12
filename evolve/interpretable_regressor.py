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

    def __init__(self, max_bins=256, lambdas=(0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0),
                 n_sweeps=30, tol=1e-4, prune_rel_tol=0.005, prune_imp_frac=0.005,
                 val_frac=0.15, n_bags=8, max_pairs=8, pair_bins=12,
                 pair_shrink=8.0, pair_gain=0.005, pair_screen_bins=8,
                 pair_top_candidates=5, small_n=300, random_state=42):
        self.max_bins = max_bins
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

    def _backfit(self, y_tr, b_tr, bands, n_bins, active, lam, sweeps):
        shapes = [np.zeros(n_bins[j]) for j in range(len(n_bins))]
        icpt = float(np.mean(y_tr))
        F = np.full(len(y_tr), icpt)
        y_sd = float(np.std(y_tr)) + 1e-12
        for _ in range(sweeps):
            delta = 0.0
            for j in active:
                w, xbar, P = bands[j]
                resid = y_tr - F + shapes[j][b_tr[:, j]]
                sums = np.bincount(b_tr[:, j], weights=resid, minlength=n_bins[j])
                ab = P * lam
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

        # --- quantile binning ---
        bin_edges, n_bins = [], np.zeros(d, dtype=int)
        bin_idx = np.zeros((n, d), dtype=np.int32)
        for j in range(d):
            col = X[:, j]
            uniq = np.unique(col[np.isfinite(col)])
            if len(uniq) <= 1:
                bin_edges.append(np.array([])); n_bins[j] = 1; continue
            if len(uniq) <= self.max_bins:
                edges = (uniq[:-1] + uniq[1:]) / 2.0
            else:
                qs = np.quantile(col, np.linspace(0, 1, self.max_bins + 1)[1:-1])
                edges = np.unique(qs)
            bin_edges.append(edges)
            n_bins[j] = len(edges) + 1
            bin_idx[:, j] = np.searchsorted(edges, col, side="right")
        active = [j for j in range(d) if n_bins[j] > 1]

        def build_bands(ids):
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
                bands[j] = (w, xbar, self._penalty_banded(xs))
            return bands

        # --- lambda selection ---
        if n >= 80 and self.val_frac > 0:
            perm = rng.permutation(n)
            n_val = max(20, int(n * self.val_frac))
            val_ids, tr_ids = perm[:n_val], perm[n_val:]
            bands_tr = build_bands(tr_ids)
            y_tr, y_val = y[tr_ids], y[val_ids]
            b_tr, b_val = bin_idx[tr_ids], bin_idx[val_ids]
            best = (np.inf, None, None, None)
            for lam in self.lambdas:
                icpt, shapes, _ = self._backfit(y_tr, b_tr, bands_tr, n_bins, active, lam, self.n_sweeps)
                pv = np.full(len(val_ids), icpt)
                for j in active:
                    pv += shapes[j][b_val[:, j]]
                mse = float(np.mean((y_val - pv) ** 2))
                if mse < best[0]:
                    best = (mse, lam, icpt, shapes)
            _, lam, icpt_sel, shapes_sel = best
        else:
            val_ids = np.array([], dtype=int)
            tr_ids = np.arange(n)
            perm = rng.permutation(n)
            folds = np.array_split(perm, 3)
            cv_mse = {l: 0.0 for l in self.lambdas}
            for f_ids in folds:
                t_ids = np.setdiff1d(perm, f_ids)
                if len(t_ids) < 5 or len(f_ids) < 2:
                    continue
                bands_f = build_bands(t_ids)
                for l in self.lambdas:
                    icpt, shapes, _ = self._backfit(y[t_ids], bin_idx[t_ids], bands_f, n_bins, active, l, self.n_sweeps)
                    pv = np.full(len(f_ids), icpt)
                    for j in active:
                        pv += shapes[j][bin_idx[f_ids, j]]
                    cv_mse[l] += float(np.sum((y[f_ids] - pv) ** 2))
            lam = min(cv_mse, key=cv_mse.get)
            icpt_sel, shapes_sel = None, None
        self.lambda_ = lam

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

        # --- bagged final backfit on all data ---
        bands_full = build_bands(np.arange(n))
        n_bags = self.n_bags if n >= 80 else 1
        acc = [np.zeros(n_bins[j]) for j in range(d)]
        icpt_acc = 0.0
        for b in range(n_bags):
            if b == 0:
                ids = np.arange(n)
                bands_b = bands_full
            else:
                ids = rng.randint(0, n, size=n)
                bands_b = build_bands(ids)
            icpt_b, shapes_b, _ = self._backfit(y[ids], bin_idx[ids], bands_b, n_bins, kept_list, lam, self.n_sweeps)
            for j in kept_list:
                acc[j] += shapes_b[j] / n_bags
            icpt_acc += icpt_b / n_bags
        intercept = icpt_acc

        # center shapes, store as interpolation tables
        self.shape_x_ = [None] * d
        self.shape_y_ = [None] * d
        self.pruned_ = [True] * d
        self.importance_ = np.zeros(d)
        for j in kept_list:
            w = w_full[j]
            mu = float(np.sum(acc[j] * w) / max(w.sum(), 1))
            sh = acc[j] - mu
            intercept += mu
            _, xbar, _ = bands_full[j]
            order = np.argsort(xbar)
            self.shape_x_[j] = xbar[order]
            self.shape_y_[j] = sh[order]
            self.pruned_[j] = False
            self.importance_[j] = float(np.sqrt(np.sum(w * sh ** 2) / max(w.sum(), 1)))
        self.intercept_ = intercept

        # prediction clipping range
        y_rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * y_rng, float(np.max(y)) + 0.05 * y_rng)

        # --- pairwise stage (residual, val-gated) ---
        self.pair_terms_ = []
        if len(val_ids) and self.max_pairs > 0 and len(kept_list) >= 2:
            from itertools import combinations
            # screening cell index per feature
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
                    ia, na = scr_idx[a_]; ib, nb = scr_idx[b_]
                    cell = ia[tr_ids] * nb + ib[tr_ids]
                    cnt = np.bincount(cell, minlength=na * nb).astype(float)
                    sums = np.bincount(cell, weights=r_tr, minlength=na * nb)
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
model_shorthand_name = "GA2MBoost_v10"
model_description = ("performance-focused GA2M: 256-bin penalized backfitting (curvature penalty, val-selected lambda), "
                     "val pruning, 8 bagged backfits averaged, then up to 8 val-gated pairwise terms (2D grid / product / "
                     "split-linear) on residuals; interp shapes via interpolation tables")
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
