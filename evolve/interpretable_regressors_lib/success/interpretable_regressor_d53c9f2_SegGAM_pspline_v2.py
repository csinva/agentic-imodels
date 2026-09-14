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
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


def _round_sig(v, sig=4):
    if v == 0 or not np.isfinite(v):
        return 0.0
    return float(f"{v:.{sig}g}")


class SegmentedGAMRegressor(BaseEstimator, RegressorMixin):
    """Additive model (GAM) built from scratch in three stages:

    1. FIT: cyclic penalized backfitting on quantile-binned features with a
       second-divided-difference curvature penalty (P-spline style). The
       penalty's null space is exactly the linear functions, so linear signals
       are recovered unshrunk while wiggle is suppressed. The smoothing
       strength lambda is selected on an internal validation split.
    2. SPARSIFY: features are greedily dropped (weakest first) whenever
       removing them does not hurt validation error, plus an absolute
       importance floor. Dropped features are reported as having no effect.
    3. DISTILL: each remaining shape function is compressed by dynamic
       programming into at most `max_segments` weighted-least-squares linear
       segments ("if a <= x < b: f = m*x + c").
    4. REFIT: with breakpoints frozen, the segment slopes/constants are
       re-estimated by backfitting least squares against the actual training
       residuals (removes distillation bias), then rounded. predict()
       evaluates exactly these rounded segments, so the printed model is
       100% faithful to actual predictions.
    """

    def __init__(self, max_bins=48, lambdas=(0.3, 3.0, 30.0, 300.0, 3000.0),
                 n_sweeps=40, tol=1e-4, max_segments=6, seg_penalty_frac=0.0008,
                 prune_rel_tol=0.005, prune_imp_frac=0.01, val_frac=0.15,
                 refit_sweeps=6, min_slope_samples=8, random_state=42):
        self.max_bins = max_bins
        self.lambdas = lambdas
        self.n_sweeps = n_sweeps
        self.tol = tol
        self.max_segments = max_segments
        self.seg_penalty_frac = seg_penalty_frac
        self.prune_rel_tol = prune_rel_tol
        self.prune_imp_frac = prune_imp_frac
        self.val_frac = val_frac
        self.refit_sweeps = refit_sweeps
        self.min_slope_samples = min_slope_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _backfit(self, y_tr, b_tr, bands, n_bins, active, lam_scale, sweeps):
        """Cyclic backfitting; returns intercept, per-feature bin values, fit."""
        shapes = [np.zeros(n_bins[j]) for j in range(len(n_bins))]
        icpt = float(np.mean(y_tr))
        F = np.full(len(y_tr), icpt)
        for _ in range(sweeps):
            delta = 0.0
            for j in active:
                w, xbar, P = bands[j]
                resid = y_tr - F + shapes[j][b_tr[:, j]]
                sums = np.bincount(b_tr[:, j], weights=resid, minlength=n_bins[j])
                ab = P * lam_scale
                ab[-1] = ab[-1] + w
                try:
                    f_new = solveh_banded(ab, sums, lower=False)
                except Exception:
                    f_new = np.where(w > 0, sums / np.maximum(w, 1e-9), 0.0)
                F += f_new[b_tr[:, j]] - shapes[j][b_tr[:, j]]
                if len(f_new):
                    delta = max(delta, float(np.max(np.abs(f_new - shapes[j]))))
                shapes[j] = f_new
            if delta < self.tol * (np.std(y_tr) + 1e-12):
                break
        return icpt, shapes, F

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        self.n_features_in_ = d
        rng = np.random.RandomState(self.random_state)

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

        y_var = float(np.var(y)) + 1e-12
        y_std = float(np.sqrt(y_var))

        def build_bands(ids):
            """Per-feature (bin weights, bin x-means, banded curvature penalty)."""
            bands = [None] * d
            for j in active:
                B = n_bins[j]
                w = np.bincount(bin_idx[ids, j], minlength=B).astype(float)
                sx = np.bincount(bin_idx[ids, j], weights=X[ids, j], minlength=B)
                xbar = np.where(w > 0, sx / np.maximum(w, 1), np.nan)
                e = bin_edges[j]
                for b in range(B):
                    if not np.isfinite(xbar[b]):
                        lo = e[b - 1] if b > 0 else e[0] - 1e-9
                        hi = e[b] if b < len(e) else e[-1] + 1e-9
                        xbar[b] = (lo + hi) / 2
                xr = xbar[-1] - xbar[0]
                xs = (xbar - xbar[0]) / (xr if xr > 0 else 1.0)
                P = np.zeros((3, B))
                if B >= 3:
                    h = np.maximum(np.diff(xs), 1e-6)
                    DtD = np.zeros((B, B))
                    for i in range(1, B - 1):
                        a0, a1 = 1.0 / h[i - 1], 1.0 / h[i]
                        row = np.zeros(B)
                        row[i - 1] = a0; row[i] = -(a0 + a1); row[i + 1] = a1
                        DtD += np.outer(row, row)
                    P[0, 2:] = np.diag(DtD, 2)
                    P[1, 1:] = np.diag(DtD, 1)
                    P[2, :] = np.diag(DtD)
                    sc = np.trace(DtD) / B
                    if sc > 0:
                        P /= sc
                bands[j] = (w, xbar, P)
            return bands

        # --- choose smoothing strength lambda ---
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
            # small n: 3-fold CV over lambdas
            val_ids = np.array([], dtype=int)
            perm = rng.permutation(n)
            folds = np.array_split(perm, 3)
            cv_mse = {lam: 0.0 for lam in self.lambdas}
            for f_ids in folds:
                t_ids = np.setdiff1d(perm, f_ids)
                if len(t_ids) < 5 or len(f_ids) < 2:
                    continue
                bands_f = build_bands(t_ids)
                for lam in self.lambdas:
                    icpt, shapes, _ = self._backfit(y[t_ids], bin_idx[t_ids], bands_f, n_bins, active, lam, self.n_sweeps)
                    pv = np.full(len(f_ids), icpt)
                    for j in active:
                        pv += shapes[j][bin_idx[f_ids, j]]
                    cv_mse[lam] += float(np.sum((y[f_ids] - pv) ** 2))
            lam = min(cv_mse, key=cv_mse.get)
            icpt_sel, shapes_sel = None, None
        self.lambda_ = lam

        # --- sparsify (decided on held-out val using selection-phase shapes) ---
        w_full = [np.bincount(bin_idx[:, j], minlength=n_bins[j]).astype(float) for j in range(d)]
        if shapes_sel is not None:
            # center selection shapes for importance measurement
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
                    kept.discard(j)
                    pv = pv2
                    cur_mse = min(cur_mse, mse2)
        else:
            kept = set(active)

        # --- final backfit on ALL data with chosen lambda, kept features only ---
        bands_full = build_bands(np.arange(n))
        kept_list = sorted(kept)
        intercept, shapes, _ = self._backfit(y, bin_idx, bands_full, n_bins, kept_list, lam, self.n_sweeps)
        imp = {}
        for j in kept_list:
            w = w_full[j]
            mu = float(np.sum(shapes[j] * w) / max(w.sum(), 1))
            shapes[j] = shapes[j] - mu
            intercept += mu
            imp[j] = float(np.sqrt(np.sum(w * shapes[j] ** 2) / max(w.sum(), 1)))
        # absolute-floor prune after final fit too (small n path relies on this)
        kept_list = [j for j in kept_list if imp[j] >= self.prune_imp_frac * y_std]

        # --- distill kept shapes into piecewise-linear segments ---
        self.segments_ = [[] for _ in range(d)]
        self.pruned_ = [True] * d
        self.importance_ = np.zeros(d)
        for j in kept_list:
            _, xbar, _ = bands_full[j]
            segs = self._dp_segments(xbar, shapes[j], w_full[j], bin_edges[j], y_std)
            self.segments_[j] = segs
            self.pruned_[j] = False
            self.importance_[j] = imp[j]
        self.intercept_ = intercept

        # --- refit segment coefficients against actual residuals (breakpoints frozen) ---
        self._refit_segments(X, y, kept_list)

        pred = self._predict_raw(X)
        self.intercept_ += float(np.mean(y) - np.mean(pred))
        self.intercept_ = _round_sig(self.intercept_, 5)
        return self

    # ------------------------------------------------------------------
    def _refit_segments(self, X, y, kept_list):
        """Backfitting LS re-estimation of segment (slope, const) with fixed
        breakpoints, evaluated on the real training data; then rounding."""
        n = X.shape[0]
        if not kept_list:
            return
        seg_ids = {}
        fvals = {}
        for j in kept_list:
            segs = self.segments_[j]
            breaks = np.array([s[1] for s in segs[:-1]])
            seg_ids[j] = np.searchsorted(breaks, X[:, j], side="right")
            fvals[j] = self._feature_effect(j, X[:, j])
        pred = np.full(n, self.intercept_)
        for j in kept_list:
            pred += fvals[j]
        for _ in range(self.refit_sweeps):
            for j in kept_list:
                partial = y - pred + fvals[j]
                segs = self.segments_[j]
                new_segs = []
                xj = X[:, j]
                for s_idx, (lo, hi, m, c) in enumerate(segs):
                    sel = seg_ids[j] == s_idx
                    ns = int(np.sum(sel))
                    if ns == 0:
                        new_segs.append((lo, hi, m, c))
                        continue
                    xs, ys = xj[sel], partial[sel]
                    xm, ym = float(np.mean(xs)), float(np.mean(ys))
                    varx = float(np.mean((xs - xm) ** 2))
                    if ns >= self.min_slope_samples and varx > 1e-12:
                        m_new = float(np.mean((xs - xm) * (ys - ym)) / varx)
                        c_new = ym - m_new * xm
                    else:
                        m_new, c_new = 0.0, ym
                    new_segs.append((lo, hi, m_new, c_new))
                self.segments_[j] = new_segs
                f_new = self._feature_effect(j, xj)
                pred += f_new - fvals[j]
                fvals[j] = f_new
            resid_mean = float(np.mean(y - pred))
            self.intercept_ += resid_mean
            pred += resid_mean
        # round coefficients (predict uses the rounded values)
        for j in kept_list:
            self.segments_[j] = [(lo, hi, _round_sig(m, 4), _round_sig(c, 4))
                                 for (lo, hi, m, c) in self.segments_[j]]

    # ------------------------------------------------------------------
    def _dp_segments(self, xbar, vals, w, edges, y_std):
        """Optimal partition of bins into few weighted-LS linear pieces."""
        B = len(vals)
        W = np.concatenate([[0], np.cumsum(w)])
        Sx = np.concatenate([[0], np.cumsum(w * xbar)])
        Sy = np.concatenate([[0], np.cumsum(w * vals)])
        Sxx = np.concatenate([[0], np.cumsum(w * xbar * xbar)])
        Sxy = np.concatenate([[0], np.cumsum(w * xbar * vals)])
        Syy = np.concatenate([[0], np.cumsum(w * vals * vals)])

        def seg_fit(i, k):
            ww = W[k + 1] - W[i]
            if ww <= 0:
                return 0.0, 0.0, 0.0
            sx = Sx[k + 1] - Sx[i]; sy = Sy[k + 1] - Sy[i]
            sxx = Sxx[k + 1] - Sxx[i]; sxy = Sxy[k + 1] - Sxy[i]
            syy = Syy[k + 1] - Syy[i]
            varx = sxx - sx * sx / ww
            m = 0.0 if varx < 1e-12 else (sxy - sx * sy / ww) / varx
            c = (sy - m * sx) / ww
            sse = syy - 2 * m * sxy - 2 * c * sy + m * m * sxx + 2 * m * c * sx + c * c * ww
            return max(sse, 0.0), m, c

        penalty = self.seg_penalty_frac * float(W[-1]) * y_std ** 2
        K = self.max_segments
        INF = np.inf
        dp = np.full((K + 1, B + 1), INF)
        back = np.zeros((K + 1, B + 1), dtype=int)
        dp[0][0] = 0.0
        for s in range(1, K + 1):
            for b in range(1, B + 1):
                bestc, besti = INF, 0
                for i in range(b):
                    if dp[s - 1][i] == INF:
                        continue
                    sse, _, _ = seg_fit(i, b - 1)
                    cost = dp[s - 1][i] + sse + penalty
                    if cost < bestc:
                        bestc, besti = cost, i
                dp[s][b] = bestc; back[s][b] = besti
        s_best = int(np.argmin(dp[:, B]))
        bounds = []
        b, s = B, s_best
        while s > 0:
            i = back[s][b]; bounds.append((i, b - 1)); b = i; s -= 1
        bounds.reverse()
        segs = []
        for (i, k) in bounds:
            _, m, c = seg_fit(i, k)
            lo = -np.inf if i == 0 else _round_sig(float(edges[i - 1]), 4)
            hi = np.inf if k == B - 1 else _round_sig(float(edges[k]), 4)
            segs.append((lo, hi, _round_sig(m, 4), _round_sig(c, 4)))
        return segs

    # ------------------------------------------------------------------
    def _feature_effect(self, j, x):
        segs = self.segments_[j]
        if not segs:
            return np.zeros_like(x)
        out = np.zeros_like(x, dtype=float)
        for (lo, hi, m, c) in segs:
            sel = (x >= lo) & (x < hi) if np.isfinite(hi) else (x >= lo)
            out[sel] = m * x[sel] + c
        return out

    def _predict_raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], getattr(self, "intercept_", 0.0))
        for j in range(self.n_features_in_):
            out += self._feature_effect(j, X[:, j])
        return out

    def predict(self, X):
        check_is_fitted(self, "segments_")
        return self._predict_raw(X)

    # ------------------------------------------------------------------
    def __str__(self):
        check_is_fitted(self, "segments_")
        d = self.n_features_in_
        names = [f"x{i}" for i in range(d)]
        order = np.argsort(-self.importance_)
        lines = [
            "Additive model (GAM). Prediction = baseline + f(x0) + f(x1) + ... "
            "(each feature contributes INDEPENDENTLY; no interactions).",
            "Features are listed from most to least important.",
            f"baseline = {self.intercept_}",
            "",
        ]
        n_rules = 0
        for j in order:
            if self.pruned_[j]:
                continue
            segs = self.segments_[j]; nm = names[j]
            if len(segs) == 1 and segs[0][2] == 0.0:
                lines.append(f"f({nm}) = {segs[0][3]}   (constant)"); n_rules += 1
            elif len(segs) == 1:
                lines.append(f"f({nm}) = {segs[0][2]}*{nm} + {segs[0][3]}   (linear)"); n_rules += 1
            else:
                lines.append(f"f({nm}) is piecewise linear:")
                for (lo, hi, m, c) in segs:
                    if np.isinf(lo):
                        cond = f"{nm} < {hi}"
                    elif np.isinf(hi):
                        cond = f"{nm} >= {lo}"
                    else:
                        cond = f"{lo} <= {nm} < {hi}"
                    if m == 0.0:
                        lines.append(f"    if {cond}:  f({nm}) = {c}")
                    else:
                        lines.append(f"    if {cond}:  f({nm}) = {m}*{nm} + {c}")
                    n_rules += 1
        pruned = [names[j] for j in range(d) if self.pruned_[j]]
        if pruned:
            lines.append("")
            lines.append(f"Features with NO effect (f = 0): {', '.join(pruned)}")
        lines.append("")
        lines.append(f"Total pieces: {n_rules}. To predict: add baseline plus each feature's f value.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SegmentedGAMRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SegGAM_pspline_v2"
model_description = ("v1 + final full-data backfit with selected lambda, 3-fold-CV lambda on small n, DP distill to <=6 "
                     "segments, then segment slopes/consts refit by backfitting LS on true residuals (breakpoints frozen) "
                     "before rounding; predict() evaluates exactly the printed segments")
model_defs = [(model_shorthand_name, SegmentedGAMRegressor())]


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
