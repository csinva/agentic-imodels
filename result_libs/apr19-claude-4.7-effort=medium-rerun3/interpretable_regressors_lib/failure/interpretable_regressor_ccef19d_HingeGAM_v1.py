"""Interpretable regressor autoresearch script. Usage: uv run interpretable_regressor.py"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance

# =========== CLASS ============


class HingeGAMRegressor(BaseEstimator, RegressorMixin):
    """Per-feature basis: {x, max(0,x-med), max(0,-x+med)} → Lasso-select, OLS refit.
    Output grouped as per-feature piecewise-linear shape f_j(x_j)."""

    def __init__(self, max_basis=10):
        self.max_basis = max_basis

    def _basis(self, X):
        meds = self.medians_[None, :]
        pos = np.maximum(0.0, X - meds)
        neg = np.maximum(0.0, meds - X)
        return np.hstack([X, pos, neg])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape; self.d_ = d
        self.medians_ = np.median(X, axis=0)
        B = self._basis(X)
        scaler = StandardScaler().fit(B)
        Bs = scaler.transform(B)
        lasso = LassoCV(cv=min(3, max(2, n // 20)), n_alphas=25, max_iter=5000).fit(Bs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_basis]
        if not kept:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(B[:, self.support_], y)
        self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return self._basis(np.asarray(X, dtype=float))[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        d = self.d_
        per_feat = defaultdict(list)
        for c, idx in zip(self.coef_, self.support_):
            if idx < d: per_feat[idx].append(("lin", c, None))
            elif idx < 2*d:
                j = idx - d; per_feat[j].append(("pos", c, float(self.medians_[j])))
            else:
                j = idx - 2*d; per_feat[j].append(("neg", c, float(self.medians_[j])))
        lines = [
            "HingeGAM — additive piecewise-linear per feature:  y = intercept + sum_j f_j(x_j)",
            "  where f_j(x) = a_j * x + b_j * max(0, x - m_j) + c_j * max(0, m_j - x)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Per-feature shape functions:",
        ]
        for j in range(d):
            terms = per_feat.get(j, [])
            if not terms:
                lines.append(f"\n  x{j}: f(x{j}) = 0  (feature excluded)")
                continue
            parts = []
            for kind, c, m in terms:
                if kind == "lin": parts.append(f"{c:+.4g} * x{j}")
                elif kind == "pos": parts.append(f"{c:+.4g} * max(0, x{j} - {m:+.3f})")
                else: parts.append(f"{c:+.4g} * max(0, {m:+.3f} - x{j})")
            lines.append(f"\n  x{j}: f(x{j}) = " + " + ".join(parts))
        return "\n".join(lines)


class _Unused4(BaseEstimator, RegressorMixin):
    """Sum of K decision stumps (depth-1 trees), each on one feature. Stagewise boosting with shrinkage."""

    def __init__(self, n_stumps=30, learning_rate=0.1, max_leaf_nodes=2):
        self.n_stumps = n_stumps
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.init_ = float(np.mean(y))
        self.stumps_ = []
        resid = y - self.init_
        for _ in range(self.n_stumps):
            stump = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=self.max_leaf_nodes,
                                          random_state=42).fit(X, resid)
            pred = stump.predict(X)
            if np.allclose(pred.std(), 0): break
            resid = resid - self.learning_rate * pred
            self.stumps_.append(stump)
        return self

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.init_, dtype=float)
        for s in self.stumps_:
            out += self.learning_rate * s.predict(X)
        return out

    def __str__(self):
        check_is_fitted(self, "stumps_")
        # Aggregate per-feature contributions
        agg = defaultdict(list)  # j -> list of (threshold, left_val, right_val) scaled by lr
        for s in self.stumps_:
            tree = s.tree_
            j = int(tree.feature[0])
            thr = float(tree.threshold[0])
            left_idx = tree.children_left[0]
            right_idx = tree.children_right[0]
            lv = float(tree.value[left_idx][0, 0]) * self.learning_rate
            rv = float(tree.value[right_idx][0, 0]) * self.learning_rate
            agg[j].append((thr, lv, rv))
        lines = [
            f"BoostedStumps: y = {self.init_:+.4f} + sum_k f_k(x)   ({len(self.stumps_)} stumps, lr={self.learning_rate})",
            "  each f_k is: if x_j <= thr then left_val else right_val",
            "",
            "Per-feature stump contributions (additive, summed over all stumps on that feature):",
        ]
        used = sorted(agg.keys())
        for j in used:
            lines.append(f"\n  x{j}:")
            for thr, lv, rv in agg[j][:8]:
                lines.append(f"    if x{j} <= {thr:+.3f}: {lv:+.4f}  else: {rv:+.4f}")
            if len(agg[j]) > 8:
                lines.append(f"    ... ({len(agg[j]) - 8} more stumps on x{j})")
        inactive = [f"x{i}" for i in range(self.d_) if i not in agg]
        if inactive:
            lines.append(f"\nFeatures with no stumps (zero effect on y): {', '.join(inactive)}")
        return "\n".join(lines)


class _Unused3(BaseEstimator, RegressorMixin):
    """Fit full OLS, drop coefs with |coef_std| < threshold * max (threshold=0.05), refit OLS."""

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.d_ = d
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        full = LinearRegression().fit(Xs, y)
        std_coef = np.abs(full.coef_)
        if std_coef.max() > 0:
            keep_mask = std_coef >= self.threshold * std_coef.max()
        else:
            keep_mask = np.zeros(d, bool); keep_mask[0] = True
        self.support_ = sorted(np.where(keep_mask)[0].tolist())
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float)[:, self.support_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = " + ".join(f"{c:+.4g}*x{j}" for c, j in zip(self.coef_, self.support_))
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [
            f"ThresholdOLS (full OLS then drop features with |std-coef| < {self.threshold}*max):",
            f"  y = {self.intercept_:+.4f} + {terms}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for c, j in zip(self.coef_, self.support_):
            lines.append(f"  x{j}: {c:.4f}")
        if excluded:
            lines.append(f"Features excluded (negligible effect): {', '.join(excluded)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeGAM_v1"
model_description = "GAM with per-feature {x, (x-med)_+, (med-x)_+} basis, Lasso-select 10, OLS refit; grouped per-feature output"
model_defs = [(model_shorthand_name, HingeGAMRegressor())]


# ============ EVAL ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o')
    args = parser.parse_args()
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results); total = len(interp_results)
    dataset_rmses = evaluate_all_regressors(model_defs)
    try: git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    def _suite(t):
        if t.startswith("insight_"): return "insight"
        if t.startswith("hard_"):    return "hard"
        return "standard"
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name: existing_interp.append(row)
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
                   "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
                   "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(existing_interp + new_interp)

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name: existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
                              "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf: by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1): r["rank"] = rank_idx
        for r in rows:
            if r["rmse"] in ("", None): r["rank"] = ""
    with open(perf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=perf_fields); w.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]: w.writerow(row)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None): all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else: all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))
    upsert_overall_results([{"commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed/total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description}], RESULTS_DIR)
    recompute_all_mean_ranks(RESULTS_DIR)
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
