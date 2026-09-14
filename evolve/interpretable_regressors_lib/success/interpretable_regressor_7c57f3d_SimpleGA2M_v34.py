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


class SimpleGA2M(BaseEstimator, RegressorMixin):
    """A deliberately simple GA2M (additive model with at most pairwise
    interactions), averaged over random seeds. One fitting mechanism:
    penalized least squares on quantile-binned features.

    Per seed:
      1. BIN     quantile-bin every feature (coarse or fine resolution).
      2. SELECT  3-fold CV picks the bin resolution, the smoothing strength
                 lambda (curvature penalty whose null space is the linear
                 functions, so linear signal is never shrunk), and whether
                 integer-coded features are treated as categories
                 (shrunken per-level means instead of smoothing).
      3. FIT     cyclic backfitting on all data.
      4. PRUNE   greedily drop features that do not help a held-out slice.
      5. PAIRS   FAST-screen feature pairs on the residual; add the best of
                 {2D grid of shrunken cell means, product term, split-linear
                 term} while a held-out slice keeps improving.
    Predictions are the average over `n_seeds` such fits (an average of
    GA2Ms is a GA2M), clipped to the padded training range.
    """

    def __init__(self, bins_options=(48, 256),
                 lambdas=(0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
                 cat_max_levels=32, cat_shrink=5.0, n_sweeps=30, tol=1e-4,
                 val_frac=0.15, prune_rel_tol=0.002, prune_imp_frac=0.005,
                 max_pairs=8, pair_bins=12, pair_shrink=8.0, pair_gain=0.005,
                 screen_bins=8, pair_top_candidates=5, n_seeds=12, random_state=42):
        self.bins_options = bins_options
        self.lambdas = lambdas
        self.cat_max_levels = cat_max_levels
        self.cat_shrink = cat_shrink
        self.n_sweeps = n_sweeps
        self.tol = tol
        self.val_frac = val_frac
        self.prune_rel_tol = prune_rel_tol
        self.prune_imp_frac = prune_imp_frac
        self.max_pairs = max_pairs
        self.pair_bins = pair_bins
        self.pair_shrink = pair_shrink
        self.pair_gain = pair_gain
        self.screen_bins = screen_bins
        self.pair_top_candidates = pair_top_candidates
        self.n_seeds = n_seeds
        self.random_state = random_state

    # -- one penalized-backfit engine ----------------------------------
    @staticmethod
    def _penalty_banded(xs):
        """Upper-banded (3,B) second-divided-difference penalty D'D."""
        B = len(xs)
        ab = np.zeros((3, B))
        if B < 3:
            return ab
        h = np.maximum(np.diff(xs), 1e-9)
        a0, a1 = 1.0 / h[:-1], 1.0 / h[1:]
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
        ab[0, 2:] = d2; ab[1, 1:] = d1; ab[2, :] = main
        return ab

    def _backfit(self, y, b_idx, bands, n_bins, active, lam, sweeps):
        """Cyclic penalized backfitting; categorical features use shrunken
        per-level means, smooth features a banded curvature-penalized solve."""
        shapes = {j: np.zeros(n_bins[j]) for j in active}
        icpt = float(np.mean(y))
        F = np.full(len(y), icpt)
        y_sd = float(np.std(y)) + 1e-12
        for _ in range(sweeps):
            delta = 0.0
            for j in active:
                w, xbar, P, is_cat = bands[j]
                resid = y - F + shapes[j][b_idx[:, j]]
                sums = np.bincount(b_idx[:, j], weights=resid, minlength=n_bins[j])
                if is_cat:
                    f_new = sums / (w + self.cat_shrink)
                else:
                    ab = P * lam
                    ab[-1] = ab[-1] + w + 1e-8
                    try:
                        f_new = solveh_banded(ab, sums, lower=False)
                    except Exception:
                        f_new = np.where(w > 0, sums / np.maximum(w, 1e-9), 0.0)
                F += f_new[b_idx[:, j]] - shapes[j][b_idx[:, j]]
                delta = max(delta, float(np.max(np.abs(f_new - shapes[j]))) if len(f_new) else 0.0)
                shapes[j] = f_new
            if delta < self.tol * y_sd:
                break
        return icpt, shapes, F

    # -- one seed ------------------------------------------------------
    def _fit_single(self, X, y, seed):
        n, d = X.shape
        rng = np.random.RandomState(seed)
        y_var = float(np.var(y)) + 1e-12
        y_std = float(np.sqrt(y_var))

        is_cat = np.zeros(d, dtype=bool)
        for j in range(d):
            u = np.unique(X[np.isfinite(X[:, j]), j])
            if 2 <= len(u) <= self.cat_max_levels and np.allclose(u, np.round(u)):
                is_cat[j] = True

        def make_bins(max_bins):
            edges_l, nb = [], np.zeros(d, dtype=int)
            bidx = np.zeros((n, d), dtype=np.int32)
            for j in range(d):
                col = X[:, j]
                u = np.unique(col[np.isfinite(col)])
                if len(u) <= 1:
                    edges_l.append(np.array([])); nb[j] = 1; continue
                if len(u) <= max_bins:
                    e = (u[:-1] + u[1:]) / 2.0
                else:
                    e = np.unique(np.quantile(col, np.linspace(0, 1, max_bins + 1)[1:-1]))
                edges_l.append(e)
                nb[j] = len(e) + 1
                bidx[:, j] = np.searchsorted(e, col, side="right")
            act = [j for j in range(d) if nb[j] > 1]
            return edges_l, nb, bidx, act

        def build_bands(ids, edges_l, nb, bidx, act, cmask):
            bands = [None] * d
            for j in act:
                B = nb[j]
                w = np.bincount(bidx[ids, j], minlength=B).astype(float)
                sx = np.bincount(bidx[ids, j], weights=X[ids, j], minlength=B)
                xbar = np.where(w > 0, sx / np.maximum(w, 1), np.nan)
                e = edges_l[j]
                bad = ~np.isfinite(xbar)
                if bad.any():
                    c = np.empty(B)
                    c[0] = e[0] - 1e-9; c[-1] = e[-1] + 1e-9
                    if B > 2:
                        c[1:-1] = (e[:-1] + e[1:]) / 2.0
                    xbar[bad] = c[bad]
                xr = xbar[-1] - xbar[0]
                if cmask[j]:
                    bands[j] = (w, xbar, None, True)
                else:
                    P = self._penalty_banded((xbar - xbar[0]) / (xr if xr > 0 else 1.0))
                    bands[j] = (w, xbar, P, False)
            return bands

        binned = {mb: make_bins(mb) for mb in sorted(set(self.bins_options))}
        no_cat = np.zeros(d, dtype=bool)
        cat_options = [no_cat, is_cat] if is_cat.any() else [no_cat]

        # SELECT (bins, categorical, lambda) by 3-fold CV
        folds = np.array_split(rng.permutation(n), 3)
        cv_sets = [(np.setdiff1d(np.arange(n), f), f) for f in folds if len(f) >= 2]
        best = (np.inf, None, None, None)
        for mb, (edges_l, nb, bidx, act) in binned.items():
            for cmask in cat_options:
                fold_bands = [build_bands(t, edges_l, nb, bidx, act, cmask) for t, _ in cv_sets]
                for lam in self.lambdas:
                    sse = 0.0
                    for (t, f), bands_f in zip(cv_sets, fold_bands):
                        icpt, shapes, _ = self._backfit(y[t], bidx[t], bands_f, nb, act, lam, self.n_sweeps)
                        pv = np.full(len(f), icpt)
                        for j in act:
                            pv += shapes[j][bidx[f, j]]
                        sse += float(np.sum((y[f] - pv) ** 2))
                    if sse < best[0]:
                        best = (sse, mb, lam, cmask)
        _, mb, lam, cmask = best
        edges_l, nb, bidx, act = binned[mb]

        # PRUNE on a held-out slice (fit on the rest)
        kept = list(act)
        if n >= 80:
            perm = rng.permutation(n)
            n_val = max(20, int(n * self.val_frac))
            val_ids, tr_ids = perm[:n_val], perm[n_val:]
            bands_tr = build_bands(tr_ids, edges_l, nb, bidx, act, cmask)
            icpt_s, shapes_s, _ = self._backfit(y[tr_ids], bidx[tr_ids], bands_tr, nb, act, lam, self.n_sweeps)
            w_full = {j: np.bincount(bidx[:, j], minlength=nb[j]).astype(float) for j in act}
            imp = {}
            for j in act:
                w = w_full[j]
                mu = float(np.sum(shapes_s[j] * w) / max(w.sum(), 1))
                imp[j] = float(np.sqrt(np.sum(w * (shapes_s[j] - mu) ** 2) / max(w.sum(), 1)))
            kept = [j for j in act if imp[j] >= self.prune_imp_frac * y_std]
            pv = np.full(len(val_ids), icpt_s)
            for j in kept:
                pv += shapes_s[j][bidx[val_ids, j]]
            cur = float(np.mean((y[val_ids] - pv) ** 2))
            tol_abs = self.prune_rel_tol * max(cur, 1e-3 * y_var)
            for j in sorted(kept, key=lambda k: imp[k]):
                pv2 = pv - shapes_s[j][bidx[val_ids, j]]
                m2 = float(np.mean((y[val_ids] - pv2) ** 2))
                if m2 <= cur + tol_abs:
                    kept.remove(j); pv = pv2; cur = min(cur, m2)
        else:
            val_ids = np.array([], dtype=int)
            tr_ids = np.arange(n)

        # FIT on all data with the selected configuration
        bands = build_bands(np.arange(n), edges_l, nb, bidx, act, cmask)
        icpt, shapes, F = self._backfit(y, bidx, bands, nb, kept, lam, self.n_sweeps)

        snap = {"icpt": icpt, "shape_x": [None] * d, "shape_y": [None] * d,
                "pairs": [], "lam": lam, "bins": mb}
        imp_out = np.zeros(d)
        for j in kept:
            w = np.bincount(bidx[:, j], minlength=nb[j]).astype(float)
            mu = float(np.sum(shapes[j] * w) / max(w.sum(), 1))
            sh = shapes[j] - mu
            snap["icpt"] += mu
            _, xbar, _, _ = bands[j]
            order = np.argsort(xbar)
            snap["shape_x"][j] = xbar[order]
            snap["shape_y"][j] = sh[order]
            imp_out[j] = float(np.sqrt(np.sum(w * sh ** 2) / max(w.sum(), 1)))

        # PAIRS: FAST-screen, add while the held-out slice improves
        y_rng_s = float(np.max(y) - np.min(y))
        clip_s = (float(np.min(y)) - 0.05 * y_rng_s, float(np.max(y)) + 0.05 * y_rng_s)
        if len(val_ids) and self.max_pairs > 0 and len(act) >= 2:
            from itertools import combinations
            scr = {}
            for j in act:
                e = np.unique(np.quantile(X[tr_ids, j], np.linspace(0, 1, self.screen_bins + 1)[1:-1]))
                if len(e) >= 1:
                    scr[j] = (np.searchsorted(e, X[:, j], side="right"), len(e) + 1)
            tr_mask = np.zeros(n, dtype=bool); tr_mask[tr_ids] = True
            for _ in range(self.max_pairs):
                resid = y - self._snap_predict(snap, X, d, clip=clip_s)
                r_tr, r_val = resid[tr_ids], resid[val_ids]
                cur = float(np.mean(r_val ** 2))
                screen = []
                for a, b in combinations(sorted(scr), 2):
                    ia, na = scr[a]; ib, nb2 = scr[b]
                    cell = ia[tr_ids] * nb2 + ib[tr_ids]
                    cnt = np.bincount(cell, minlength=na * nb2).astype(float)
                    sums = np.bincount(cell, weights=r_tr, minlength=na * nb2)
                    mu = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
                    mu *= cnt / (cnt + self.pair_shrink)
                    screen.append((float(np.sum(cnt * mu ** 2)), a, b))
                screen.sort(reverse=True)
                best_c = None
                for _, a, b in screen[:self.pair_top_candidates]:
                    for cand in self._pair_candidates(X, resid, tr_ids, tr_mask, a, b):
                        contrib = cand.pop("contrib")
                        dshift = float(np.mean(contrib[tr_ids]))
                        m2 = float(np.mean((r_val - (contrib[val_ids] - dshift)) ** 2))
                        gain = cur - m2
                        if best_c is None or gain > best_c[0]:
                            best_c = (gain, cand, dshift)
                if best_c is None or best_c[0] < max(self.pair_gain * cur, 5e-4 * y_var):
                    break
                _, term, dshift = best_c
                snap["pairs"].append(term)
                snap["icpt"] -= dshift
        return snap, imp_out

    def _pair_candidates(self, X, resid, tr_ids, tr_mask, a, b):
        cands = []
        r_tr = resid[tr_ids]
        ea = np.unique(np.quantile(X[tr_ids, a], np.linspace(0, 1, self.pair_bins + 1)[1:-1]))
        eb = np.unique(np.quantile(X[tr_ids, b], np.linspace(0, 1, self.pair_bins + 1)[1:-1]))
        if len(ea) >= 1 and len(eb) >= 1:
            na, nb2 = len(ea) + 1, len(eb) + 1
            ia = np.searchsorted(ea, X[:, a], side="right")
            ib = np.searchsorted(eb, X[:, b], side="right")
            cell = ia * nb2 + ib
            cnt = np.bincount(cell[tr_ids], minlength=na * nb2).astype(float)
            sums = np.bincount(cell[tr_ids], weights=r_tr, minlength=na * nb2)
            vals = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
            vals *= cnt / (cnt + self.pair_shrink)
            cands.append({"type": "grid", "i": a, "j": b, "ei": ea, "ej": eb,
                          "nb": nb2, "vals": vals, "contrib": vals[cell]})
        p = X[:, a] * X[:, b]
        pm = float(np.mean(p[tr_ids]))
        varp = float(np.mean((p[tr_ids] - pm) ** 2))
        if varp > 1e-12:
            coef = float(np.mean((p[tr_ids] - pm) * r_tr) / varp)
            cands.append({"type": "prod", "i": a, "j": b, "coef": coef, "contrib": coef * p})
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
                    mnew = float(np.mean((xs - xm) * (ys - ym)) / varx) * ns / (ns + 12.0)
                    coefs.append((mnew, ym - mnew * xm))
                if ok:
                    (m1, c1), (m2, c2) = coefs
                    contrib = np.where(side, m2 * X[:, sb] + c2, m1 * X[:, sb] + c1)
                    cands.append({"type": "split", "i": sa, "j": sb, "t": t,
                                  "lo": (m1, c1), "hi": (m2, c2), "contrib": contrib})
        return cands

    # -- ensemble over seeds -------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.snapshots_ = []
        self.importance_ = np.zeros(X.shape[1])
        for k in range(self.n_seeds):
            snap, imp = self._fit_single(X, y, self.random_state + 1000 * k)
            self.snapshots_.append(snap)
            self.importance_ += imp / self.n_seeds
        y_rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * y_rng, float(np.max(y)) + 0.05 * y_rng)
        # expose the first snapshot's tables for introspection / __str__
        self.shape_x_ = self.snapshots_[0]["shape_x"]
        self.shape_y_ = self.snapshots_[0]["shape_y"]
        self.intercept_ = self.snapshots_[0]["icpt"]
        pred = self.predict(X)
        shift = float(np.mean(y) - np.mean(pred))
        for snap in self.snapshots_:
            snap["icpt"] += shift
        return self

    @staticmethod
    def _snap_predict(snap, X, d, clip):
        out = np.full(X.shape[0], snap["icpt"])
        for j in range(d):
            if snap["shape_x"][j] is not None:
                out += np.interp(X[:, j], snap["shape_x"][j], snap["shape_y"][j])
        for t in snap["pairs"]:
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
        if clip is not None:
            out = np.clip(out, clip[0], clip[1])
        return out

    def predict(self, X):
        check_is_fitted(self, "snapshots_")
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        for snap in self.snapshots_:
            out += self._snap_predict(snap, X, self.n_features_in_, None) / len(self.snapshots_)
        return np.clip(out, self.clip_[0], self.clip_[1])

    def __str__(self):
        check_is_fitted(self, "snapshots_")
        d = self.n_features_in_
        names = [f"x{i}" for i in range(d)]
        snap = self.snapshots_[0]
        lines = ["Additive model (GA2M), average of seed fits; first fit shown.",
                 f"baseline = {snap['icpt']:.4f}"]
        for j in np.argsort(-self.importance_):
            if snap["shape_x"][j] is None:
                continue
            xs, ys = snap["shape_x"][j], snap["shape_y"][j]
            k = min(9, len(xs))
            idx = np.linspace(0, len(xs) - 1, k).round().astype(int)
            pts = "  ".join(f"{xs[i]:+.3g}->{ys[i]:+.3g}" for i in idx)
            lines.append(f"f({names[j]}): {pts}")
        for t in snap["pairs"]:
            lines.append(f"pairwise term ({t['type']}) on ({names[t['i']]}, {names[t['j']]})")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SimpleGA2M.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SimpleGA2M_v34"
model_description = ("elegant final GA2M: per seed, 3-fold CV picks (bin resolution, categorical treatment, curvature "
                     "lambda); penalized backfit; held-out greedy pruning; FAST-screened held-out-gated pairwise terms; "
                     "8 seed fits prediction-averaged. No boosting/bagging/winsorization - ablations showed them redundant")
model_defs = [(model_shorthand_name, SimpleGA2M())]


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
