"""157 generalization interpretability tests across 6 cognitive-operation
categories. The suite is designed to defeat memorization of canonical Q/A
tables that a prior model (RegimeMixtureTeacherAuditAtlas) exploited:

Design rules enforced in this file:
- No sub-family contains more than 3 tests.
- Correct answers are spread across different feature indices — the right
  answer to any given subfamily is NOT always x0.
- Question wording is rephrased to avoid exact or near-exact matches with
  the original 43-test suite in evolve/src/interp_eval.py.
- Query points, thresholds, and feature indices differ from the original
  suite (no 'x0=2.0, x1=0.0, x2=0.0' style samples).
- Training data is capped at n=100 rows so any model can fit quickly.

Totals per category:
    feature_attribution (26), point_simulation (26), sensitivity (26),
    counterfactual (26), structural (26), complex_fn (27).
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evolve", "src"))
from interp_eval import get_model_str, ask_llm, _safe_clone  # noqa: E402


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def _nums(text):
    return re.findall(r"-?\d+\.?\d*", text or "")


def _any_num_close(text, target, tol):
    for n in reversed(_nums(text)):
        try:
            v = float(n)
        except ValueError:
            continue
        if abs(v - target) <= tol:
            return True
    return False


def _feat_list(text):
    return set(re.findall(r"x\d+", (text or "").lower()))


def _first_feat(text):
    m = re.search(r"x\d+", (text or "").lower())
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Synthetic data factories — permutation-aware so answers can target any
# feature index.
# ---------------------------------------------------------------------------

def _lin(coefs, n=100, seed=0, noise=0.5, intercept=0.0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, len(coefs))
    y = X @ np.asarray(coefs, dtype=float) + intercept + rng.randn(n) * noise
    return X, y


def _lin_at(coef_map, n_feat, n=100, seed=0, noise=0.5, intercept=0.0):
    """Linear data where only features in coef_map have non-zero coefficients."""
    coefs = [coef_map.get(i, 0.0) for i in range(n_feat)]
    return _lin(coefs, n=n, seed=seed, noise=noise, intercept=intercept)


def _step_at(idx, thresh=0.0, low=0.0, high=2.0, n_feat=3, n=100, seed=0, noise=0.1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = np.where(X[:, idx] > thresh, high, low) + rng.randn(n) * noise
    return X, y


def _relu_at(idx, coef=3.0, n_feat=3, n=100, seed=0, noise=0.2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = coef * np.maximum(0.0, X[:, idx]) + rng.randn(n) * noise
    return X, y


def _quad_at(i_sq_a, i_sq_b, i_lin, c0=2.0, c1=-1.5, c2=1.0,
             n_feat=5, n=100, seed=0, noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (c0 * X[:, i_sq_a] ** 2 + c1 * X[:, i_sq_b] ** 2
         + c2 * X[:, i_lin] + rng.randn(n) * noise)
    return X, y


def _interact_at(i0, i1, c0=3.0, c1=2.0, c_int=2.0, n_feat=4, n=100, seed=0, noise=0.4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (c0 * X[:, i0] + c1 * X[:, i1]
         + c_int * X[:, i0] * X[:, i1] + rng.randn(n) * noise)
    return X, y


def _triple_at(i0, i1, i2, i3, seed=0, n=100, n_feat=6, noise=0.4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (2.0 * X[:, i0] * X[:, i1]
         + 3.0 * X[:, i1] * X[:, i2]
         + X[:, i0] * X[:, i2] * X[:, i3]
         + rng.randn(n) * noise)
    return X, y


def _fried_at(perm, seed=0, n=100, noise=0.5):
    """perm: 5 indices (a,b,c,d,e) → y = 10 sin(pi Xa Xb)+20(Xc-0.5)^2+10 Xd+5 Xe."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 10))
    a, b, c, d, e = perm
    y = (10.0 * np.sin(np.pi * X[:, a] * X[:, b])
         + 20.0 * (X[:, c] - 0.5) ** 2
         + 10.0 * X[:, d]
         + 5.0 * X[:, e]
         + rng.randn(n) * noise)
    return X, y


def _cascade_at(i_gate, i_pos, i_neg, seed=0, n=100, n_feat=6, noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = np.where(X[:, i_gate] > 0, 2.5 * X[:, i_pos], -2.5 * X[:, i_neg]) + rng.randn(n) * noise
    return X, y


def _exp_decay_at(i_exp, i_lin, seed=0, n=100, n_feat=4, noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = 4.0 * np.exp(-X[:, i_exp]) + 2.0 * X[:, i_lin] + rng.randn(n) * noise
    return X, y


def _piecewise3_at(idx, seed=0, n=100, n_feat=4, noise=0.2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    x = X[:, idx]
    y = np.where(x < -1.0, -2.5,
                 np.where(x < 1.0, 2.5 * x, 2.5 + 0.4 * (x - 1.0))) + rng.randn(n) * noise
    return X, y


def _sine_at(i_sin, i_cos, i_lin, seed=0, n=100, n_feat=5, noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (3.0 * np.sin(X[:, i_sin])
         + 2.0 * np.cos(X[:, i_cos])
         + 1.2 * X[:, i_lin] + rng.randn(n) * noise)
    return X, y


def _absv_at(i_pos, i_neg, i_lin, seed=0, n=100, n_feat=5, noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    y = (2.5 * np.abs(X[:, i_pos])
         - 1.5 * np.abs(X[:, i_neg])
         + 1.2 * X[:, i_lin] + rng.randn(n) * noise)
    return X, y


# ---------------------------------------------------------------------------
# Core fit helper
# ---------------------------------------------------------------------------

def _fit_with_names(model, data_fn, min_r2=None):
    X, y = data_fn()
    n_feat = X.shape[1]
    names = [f"x{i}" for i in range(n_feat)]
    m = _safe_clone(model)
    m.fit(X, y)
    if min_r2 is not None:
        r2 = r2_score(y, m.predict(X))
        if r2 < min_r2:
            return None
    return m, names


def _fmt_sample(sample, names):
    return ", ".join(f"{n}={float(sample[i]):g}" for i, n in enumerate(names))


# ---------------------------------------------------------------------------
# Attribution runners
# ---------------------------------------------------------------------------

def _run_top_feature(test_name, model, llm, data_fn, target):
    """Ask for the feature whose +1 perturbation from zeros shifts the
    prediction the most. Different question phrasing than the originals'
    'which single feature is most important'."""
    fit = _fit_with_names(model, data_fn, min_r2=0.4)
    if fit is None:
        return dict(test=test_name, passed=False, error="low_r2",
                    ground_truth=target, response=None)
    m, names = fit
    q = ("Imagine starting from an all-zeros input and perturbing ONE feature "
         "at a time by 1 unit. Which perturbation shifts the model's output the "
         "furthest in absolute value? Give just the feature name.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = bool(response and _first_feat(response) == target)
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


def _run_bottom_feature(test_name, model, llm, data_fn, target):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    q = ("Among all input features, which one produces the SMALLEST absolute "
         "shift in the model's output when perturbed by 1 unit from the origin? "
         "Give a single feature name.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = bool(response and _first_feat(response) == target)
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


def _run_rank_top2(test_name, model, llm, data_fn, ordered):
    """Only the top-2 matter. Accept if the first two listed features match."""
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    target_str = ", ".join(ordered[:2])
    q = ("Order the two features whose +/-1 perturbations have the LARGEST "
         "absolute effect on the model's output, from largest to smallest. "
         f"Reply with a comma-separated list (e.g., '{target_str}').")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = False
    if response:
        r = response.lower()
        positions = [r.find(t) for t in ordered[:2]]
        passed = all(p != -1 for p in positions) and positions[0] < positions[1]
    return dict(test=test_name, passed=passed, ground_truth=target_str, response=response)


def _run_zero_effect(test_name, model, llm, data_fn, active_feats, total_feats):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    q = ("Identify every feature whose contribution to the model's output is "
         "essentially zero across typical inputs. Answer as a comma-separated "
         "list of feature names.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = False
    irrelevant = {f"x{i}" for i in range(total_feats)} - set(active_feats)
    if response:
        named = _feat_list(response)
        hit = len(irrelevant & named)
        spurious = len(named & set(active_feats))
        passed = hit >= max(2, len(irrelevant) // 2) and spurious == 0
    return dict(test=test_name, passed=passed,
                ground_truth=f"zero_effect={sorted(irrelevant)}", response=response)


def _run_delta_unit(test_name, model, llm, data_fn, feature_idx):
    """Signed delta from +1 perturbation at a chosen feature_idx."""
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    zero = np.zeros((1, len(names)))
    bump = zero.copy()
    bump[0, feature_idx] = 1.0
    delta = float(m.predict(bump)[0]) - float(m.predict(zero)[0])
    q = (f"Start from an all-zeros input. Replace x{feature_idx} with 1.0 while "
         f"leaving every other feature at 0. Report the signed shift in the "
         "model's output as a single number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(delta) * 0.25, 0.6)
    passed = _any_num_close(response, delta, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(delta, 3), response=response)


def _run_dominant_sample(test_name, model, llm, data_fn, sample, target):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    feat_str = _fmt_sample(sample, names)
    q = (f"At the specific input {feat_str}, which single feature is "
         f"contributing the most in absolute value to this prediction? Give "
         "just the feature name.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = bool(response and _first_feat(response) == target)
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


def _run_second_feature(test_name, model, llm, data_fn, target):
    """Second-most-influential feature — genuinely new task."""
    fit = _fit_with_names(model, data_fn, min_r2=0.4)
    if fit is None:
        return dict(test=test_name, passed=False, error="low_r2",
                    ground_truth=target, response=None)
    m, names = fit
    q = ("Rank the features by the absolute effect of a +1 perturbation (from "
         "zeros). Which feature lands in SECOND place? Answer with just the "
         "feature name.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = bool(response and _first_feat(response) == target)
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


def _run_count_active(test_name, model, llm, data_fn, active_feats):
    fit = _fit_with_names(model, data_fn, min_r2=0.3)
    if fit is None:
        return dict(test=test_name, passed=False, error="low_r2",
                    ground_truth=len(active_feats), response=None)
    m, names = fit
    target = int(len(active_feats))
    q = ("How many features have a non-negligible effect on the model's "
         "output? Answer with just an integer.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = False
    nums = _nums(response)
    if nums:
        try:
            v = int(round(float(nums[0])))
            passed = abs(v - target) <= 1  # allow off-by-one
        except ValueError:
            pass
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


def _run_pair_importance(test_name, model, llm, data_fn, a, b, target):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    q = (f"Considering only x{a} and x{b}, which of the two has a LARGER "
         f"absolute effect on the model's output? Reply with just the feature "
         f"name (either 'x{a}' or 'x{b}').")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = bool(response and _first_feat(response) == target)
    return dict(test=test_name, passed=passed, ground_truth=target, response=response)


# ---------------------------------------------------------------------------
# Point-simulation runner
# ---------------------------------------------------------------------------

def _run_point_pred(test_name, model, llm, data_fn, sample,
                    tol_rel=0.2, tol_abs=1.0, min_r2=None, question=None):
    fit = _fit_with_names(model, data_fn, min_r2=min_r2)
    if fit is None:
        return dict(test=test_name, passed=False, error="low_r2",
                    ground_truth=None, response=None)
    m, names = fit
    sample_arr = np.asarray(sample, dtype=float).reshape(1, -1)
    true_pred = float(m.predict(sample_arr)[0])
    feat_str = _fmt_sample(sample, names)
    if question is None:
        q = (f"Evaluate the model at input {feat_str}. Report its output as a "
             "single number.")
    else:
        q = question.format(feat_str=feat_str)
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(true_pred) * tol_rel, tol_abs)
    passed = False
    for num_str in reversed(_nums(response)):
        try:
            v = float(num_str)
        except ValueError:
            continue
        if abs(v - true_pred) < tol:
            passed = True
            break
    return dict(test=test_name, passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def _run_sample_diff(test_name, model, llm, data_fn, sample_a, sample_b,
                     tol_rel=0.2, tol_abs=1.0):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    pa = float(m.predict(np.asarray(sample_a).reshape(1, -1))[0])
    pb = float(m.predict(np.asarray(sample_b).reshape(1, -1))[0])
    delta = pb - pa
    q = (f"Sample α has features {_fmt_sample(sample_a, names)}. "
         f"Sample β has features {_fmt_sample(sample_b, names)}. "
         "Report prediction(β) minus prediction(α) as a single signed number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(delta) * tol_rel, tol_abs)
    passed = _any_num_close(response, delta, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(delta, 3), response=response)


# ---------------------------------------------------------------------------
# Sensitivity runners
# ---------------------------------------------------------------------------

def _run_sensitivity(test_name, model, llm, data_fn, k, base, new_val,
                     tol_rel=0.2, tol_abs=1.0, wording="change"):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    base_arr = np.asarray(base, dtype=float).reshape(1, -1)
    new_arr = base_arr.copy()
    new_arr[0, k] = new_val
    delta = float(m.predict(new_arr)[0]) - float(m.predict(base_arr)[0])
    feat_str = _fmt_sample(base, names)
    if wording == "change":
        q = (f"The current input is {feat_str}. If x{k} is changed to {new_val} "
             "(leaving every other feature as-is), what is the signed shift in "
             "the model's output? Answer as a single number.")
    elif wording == "decrease":
        dec = float(base[k]) - float(new_val)
        q = (f"From input {feat_str}, x{k} is decreased by {dec:g} (others stay "
             "the same). Report the signed change in the model's output.")
    elif wording == "multi":
        q = (f"Starting at {feat_str}, x{k} is raised to {new_val}. What is the "
             "resulting change in the model's prediction? Give a signed number.")
    else:
        q = (f"Input: {feat_str}. Transform x{k} to {new_val} (others fixed). "
             "Signed output shift? Answer with a single number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(delta) * tol_rel, tol_abs)
    passed = _any_num_close(response, delta, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(delta, 3), response=response)


def _run_crossing(test_name, model, llm, data_fn, k, fixed_base, target_level,
                  search_range=(-4.0, 4.0), tol=0.7):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    n = len(names)
    xs = np.linspace(search_range[0], search_range[1], 201)
    grid = np.tile(np.asarray(fixed_base, dtype=float), (len(xs), 1))
    grid[:, k] = xs
    ys = m.predict(grid)
    diffs = ys - target_level
    sign_changes = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0]
    if len(sign_changes) == 0:
        return dict(test=test_name, passed=False, error="target_not_crossed",
                    ground_truth=None, response=None)
    i = sign_changes[0]
    xa, xb = xs[i], xs[i + 1]
    ya, yb = ys[i], ys[i + 1]
    crossing = (xa + xb) / 2 if yb == ya else xa + (target_level - ya) * (xb - xa) / (yb - ya)
    fixed_str = ", ".join(f"x{j}={fixed_base[j]}" for j in range(n) if j != k)
    q = (f"With {fixed_str} held fixed, locate the value of x{k} at which the "
         f"model's output first reaches {target_level:.2f}. Give just a number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = _any_num_close(response, crossing, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(float(crossing), 3), response=response)


def _run_two_feature(test_name, model, llm, data_fn, base, k1, new1, k2, new2,
                     tol_rel=0.2, tol_abs=1.0):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    base_arr = np.asarray(base, dtype=float).reshape(1, -1)
    new_arr = base_arr.copy()
    new_arr[0, k1] = new1
    new_arr[0, k2] = new2
    delta = float(m.predict(new_arr)[0]) - float(m.predict(base_arr)[0])
    feat_str = _fmt_sample(base, names)
    q = (f"From input {feat_str}, simultaneously set x{k1} to {new1} and x{k2} "
         f"to {new2} (others unchanged). Report the signed change in the "
         "model's output as a single number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(delta) * tol_rel, tol_abs)
    passed = _any_num_close(response, delta, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(delta, 3), response=response)


# ---------------------------------------------------------------------------
# Counterfactual runner
# ---------------------------------------------------------------------------

def _run_counterfactual(test_name, model, llm, data_fn, k, base, b_k,
                        tol_abs=0.6, tol_rel=0.2, yb_tol_rel=0.2, yb_tol_abs=1.0):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    n = len(names)
    base_arr = np.asarray(base, dtype=float).reshape(1, -1)
    B = base_arr.copy()
    B[0, k] = b_k
    y_A = float(m.predict(base_arr)[0])
    y_B = float(m.predict(B)[0])
    if abs(y_B - y_A) < 0.3:
        return dict(test=test_name, passed=False, error="counterfactual_trivial",
                    ground_truth=round(b_k, 3), response=None)
    fixed_str = ", ".join(f"x{j}={base[j]}" for j in range(n) if j != k)
    q = (f"At input {_fmt_sample(base, names)} the model outputs about "
         f"{y_A:.2f}. Suppose only x{k} is allowed to move (keeping "
         f"{fixed_str}). What value of x{k} makes the output equal {y_B:.2f}? "
         "Give just a number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    nums = _nums(response)
    passed = False
    if nums:
        try:
            llm_x = float(nums[0])
            direct_tol = max(abs(b_k) * tol_rel, tol_abs)
            if abs(llm_x - b_k) <= direct_tol:
                passed = True
            else:
                probe = base_arr.copy()
                probe[0, k] = llm_x
                y_probe = float(m.predict(probe)[0])
                yb_tol = max(abs(y_B) * yb_tol_rel, yb_tol_abs)
                passed = abs(y_probe - y_B) <= yb_tol
        except ValueError:
            pass
    return dict(test=test_name, passed=passed,
                ground_truth=round(b_k, 3), response=response)


# ---------------------------------------------------------------------------
# Structural runners
# ---------------------------------------------------------------------------

def _run_compactness(test_name, model, llm, data_fn, op_threshold=10):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    q = (f"Could the model's prediction be computed from the inputs using at "
         f"most {op_threshold} arithmetic operations or branching rules? "
         "Answer with exactly 'yes' or 'no'.")
    response = ask_llm(llm, get_model_str(m, names), q, max_tokens=5)
    passed = bool(response and "yes" in response.lower())
    return dict(test=test_name, passed=passed, ground_truth=None, response=response)


def _run_argmax(test_name, model, llm, data_fn, k, fixed_base,
                search_range=(-3.0, 3.0), tol=0.8):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    xs = np.linspace(search_range[0], search_range[1], 401)
    grid = np.tile(np.asarray(fixed_base, dtype=float), (len(xs), 1))
    grid[:, k] = xs
    ys = m.predict(grid)
    x_star = float(xs[int(np.argmax(ys))])
    top = np.max(ys)
    plateau = xs[ys >= top - max(abs(top) * 0.02, 0.05)]
    lo, hi = float(plateau.min()), float(plateau.max())
    fixed_str = ", ".join(f"x{j}={fixed_base[j]}" for j in range(len(names)) if j != k)
    q = (f"Fix {fixed_str}. Over x{k} ∈ [{search_range[0]:.1f}, "
         f"{search_range[1]:.1f}], at which value of x{k} is the model's "
         "output MAXIMAL? Reply with a single number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = False
    nums = _nums(response)
    if nums:
        try:
            v = float(nums[0])
            passed = (lo - tol) <= v <= (hi + tol)
        except ValueError:
            pass
    return dict(test=test_name, passed=passed,
                ground_truth=round(x_star, 3), response=response)


def _run_argmin(test_name, model, llm, data_fn, k, fixed_base,
                search_range=(-3.0, 3.0), tol=0.8):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    xs = np.linspace(search_range[0], search_range[1], 401)
    grid = np.tile(np.asarray(fixed_base, dtype=float), (len(xs), 1))
    grid[:, k] = xs
    ys = m.predict(grid)
    x_star = float(xs[int(np.argmin(ys))])
    bot = np.min(ys)
    plateau = xs[ys <= bot + max(abs(bot) * 0.02, 0.05)]
    lo, hi = float(plateau.min()), float(plateau.max())
    fixed_str = ", ".join(f"x{j}={fixed_base[j]}" for j in range(len(names)) if j != k)
    q = (f"Fix {fixed_str}. Over x{k} ∈ [{search_range[0]:.1f}, "
         f"{search_range[1]:.1f}], at which value of x{k} is the model's "
         "output MINIMAL? Reply with a single number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    passed = False
    nums = _nums(response)
    if nums:
        try:
            v = float(nums[0])
            passed = (lo - tol) <= v <= (hi + tol)
        except ValueError:
            pass
    return dict(test=test_name, passed=passed,
                ground_truth=round(x_star, 3), response=response)


def _run_monotonic(test_name, model, llm, data_fn, k, fixed_base,
                   search_range=(-2.0, 2.0)):
    """Classify the model's behavior along x_k as 'increasing', 'decreasing',
    or 'non-monotonic'. Ground truth from a 401-point sweep."""
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    xs = np.linspace(search_range[0], search_range[1], 401)
    grid = np.tile(np.asarray(fixed_base, dtype=float), (len(xs), 1))
    grid[:, k] = xs
    ys = m.predict(grid)
    diffs = np.diff(ys)
    tot = float(max(np.max(ys) - np.min(ys), 1e-6))
    pos = float(np.sum(diffs[diffs > 0])) / tot
    neg = float(-np.sum(diffs[diffs < 0])) / tot
    if pos > 0.95 and neg < 0.1:
        gt = "increasing"
    elif neg > 0.95 and pos < 0.1:
        gt = "decreasing"
    else:
        gt = "non-monotonic"
    fixed_str = ", ".join(f"x{j}={fixed_base[j]}" for j in range(len(names)) if j != k)
    q = (f"With {fixed_str} fixed, sweep x{k} across "
         f"[{search_range[0]:.1f}, {search_range[1]:.1f}]. Is the model's "
         "output strictly increasing, strictly decreasing, or non-monotonic? "
         "Answer with exactly one of 'increasing', 'decreasing', 'non-monotonic'.")
    response = ask_llm(llm, get_model_str(m, names), q, max_tokens=12)
    passed = bool(response and gt in response.lower())
    return dict(test=test_name, passed=passed, ground_truth=gt, response=response)


def _run_output_range(test_name, model, llm, data_fn, k, fixed_base,
                      search_range=(-2.0, 2.0), tol_rel=0.2, tol_abs=1.0):
    fit = _fit_with_names(model, data_fn)
    m, names = fit
    xs = np.linspace(search_range[0], search_range[1], 401)
    grid = np.tile(np.asarray(fixed_base, dtype=float), (len(xs), 1))
    grid[:, k] = xs
    ys = m.predict(grid)
    span = float(np.max(ys) - np.min(ys))
    fixed_str = ", ".join(f"x{j}={fixed_base[j]}" for j in range(len(names)) if j != k)
    q = (f"With {fixed_str} fixed, vary x{k} over "
         f"[{search_range[0]:.1f}, {search_range[1]:.1f}]. By how much does the "
         "model's output span between its minimum and maximum on that range? "
         "Reply with just a (positive) number.")
    response = ask_llm(llm, get_model_str(m, names), q)
    tol = max(abs(span) * tol_rel, tol_abs)
    passed = _any_num_close(response, span, tol)
    return dict(test=test_name, passed=passed,
                ground_truth=round(span, 3), response=response)


# ---------------------------------------------------------------------------
# Test specifications — the arbiters of what questions exist.
# ---------------------------------------------------------------------------
# All answers below were chosen so each subfamily's correct answer set is NOT
# a constant across tests (feature indices vary), and none of the questions
# lexically match any of the original 43 tests in evolve/src/interp_eval.py.

# ===== FEATURE ATTRIBUTION (26) ===========================================
# 9 subfamilies × (3,3,3,3,3,3,3,3,2) = 26.

_FA_SPECS = []

# fa_top_feature (3 tests, answers: x2, x3, x5)
for i, (coef_map, n_feat, seed, tgt) in enumerate([
    ({2: 6.5, 0: 1.2, 1: -0.4, 3: 0.3}, 5, 4101, "x2"),
    ({3: 7.0, 0: -0.6, 1: 1.8, 4: -0.8, 2: 0.2}, 6, 4102, "x3"),
    ({5: 8.0, 1: 2.0, 3: -1.2, 0: 0.4, 6: -0.3}, 8, 4103, "x5"),
]):
    _FA_SPECS.append((
        f"fa_top_feature_{i + 1:02d}", "top_feature",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             target=tgt),
    ))

# fa_bottom_feature (3, answers: x3, x5, x6)
for i, (coef_map, n_feat, seed, tgt) in enumerate([
    ({0: 3.0, 1: -2.0, 2: 1.2, 3: 0.0, 4: 1.5}, 5, 4201, "x3"),
    ({0: 2.5, 1: -3.0, 2: 1.0, 4: 1.8, 5: 0.0}, 6, 4202, "x5"),
    ({0: -2.5, 1: 2.0, 2: 1.0, 3: -1.5, 4: 0.8, 5: -0.5, 6: 0.0}, 7, 4203, "x6"),
]):
    _FA_SPECS.append((
        f"fa_bottom_feature_{i + 1:02d}", "bottom_feature",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             target=tgt),
    ))

# fa_rank_top2 (3, answers: [x3,x1], [x4,x2], [x5,x0])
for i, (coef_map, n_feat, seed, ordered) in enumerate([
    ({3: 6.0, 1: 4.0, 0: 1.0, 2: -0.5}, 4, 4301, ["x3", "x1", "x0"]),
    ({4: 7.0, 2: 5.0, 0: 1.5, 1: -1.0, 3: 0.3}, 5, 4302, ["x4", "x2", "x0"]),
    ({5: 8.0, 0: 5.5, 2: 1.5, 3: -0.8, 4: 0.4, 1: 0.2}, 6, 4303, ["x5", "x0", "x2"]),
]):
    _FA_SPECS.append((
        f"fa_rank_top2_{i + 1:02d}", "rank_top2",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             ordered=ordered),
    ))

# fa_zero_effect (3, active sets vary — spread active indices)
for i, (coef_map, n_feat, seed) in enumerate([
    ({1: 4.0, 3: -2.5}, 6, 4401),
    ({2: 5.0, 5: -3.0, 7: 2.0}, 9, 4402),
    ({0: 4.0, 4: -3.0, 8: 2.5, 11: 1.5}, 12, 4403),
]):
    active = [f"x{k}" for k in coef_map]
    _FA_SPECS.append((
        f"fa_zero_effect_{i + 1:02d}", "zero_effect",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             active_feats=active, total_feats=n_feat),
    ))

# fa_delta_unit (3, feature_idx: 2, 4, 1 — never 0)
for i, (coef_map, n_feat, seed, k) in enumerate([
    ({0: 2.0, 1: -1.5, 2: 3.5, 3: 0.8}, 4, 4501, 2),
    ({0: -1.0, 2: 2.0, 4: -4.0, 5: 0.5}, 6, 4502, 4),
    ({0: 2.5, 1: 3.0, 3: -1.2, 4: 0.7}, 5, 4503, 1),
]):
    _FA_SPECS.append((
        f"fa_delta_unit_{i + 1:02d}", "delta_unit",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             feature_idx=k),
    ))

# fa_dominant_sample (3, answers: x3, x2, x4 — sample chosen to make that
# feature the dominant contributor)
for i, (coef_map, n_feat, seed, sample, tgt) in enumerate([
    ({0: 2.0, 1: 1.5, 2: -1.0, 3: 3.5, 4: 0.5}, 5, 4601,
     [0.1, 0.2, 0.1, 1.6, 0.2], "x3"),
    ({0: -1.5, 1: 1.0, 2: 4.0, 3: 0.5, 4: -0.8}, 5, 4602,
     [0.2, 0.3, 1.4, 0.2, 0.1], "x2"),
    ({0: 1.0, 1: -1.2, 2: 0.8, 3: -0.6, 4: 3.0, 5: 0.4}, 6, 4603,
     [0.2, 0.3, 0.2, 0.1, 1.5, 0.2], "x4"),
]):
    _FA_SPECS.append((
        f"fa_dominant_sample_{i + 1:02d}", "dominant_sample",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             sample=sample, target=tgt),
    ))

# fa_second_feature (3, answers: x1, x4, x0)
for i, (coef_map, n_feat, seed, tgt) in enumerate([
    ({2: 6.0, 1: 4.0, 0: 0.8, 3: -0.3}, 4, 4701, "x1"),
    ({3: 7.0, 4: 4.5, 0: 1.0, 1: -0.8, 2: 0.4}, 5, 4702, "x4"),
    ({5: 7.5, 0: 5.0, 2: 0.8, 1: -0.5, 3: 0.3, 4: -0.2}, 6, 4703, "x0"),
]):
    _FA_SPECS.append((
        f"fa_second_feature_{i + 1:02d}", "second_feature",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             target=tgt),
    ))

# fa_count_active (3, answers: 2, 3, 4 active features)
for i, (coef_map, n_feat, seed) in enumerate([
    ({0: 3.5, 4: -2.5}, 6, 4801),
    ({1: 4.0, 3: -3.0, 5: 2.0}, 7, 4802),
    ({0: 3.0, 2: 2.5, 4: -2.0, 6: 1.5}, 8, 4803),
]):
    _FA_SPECS.append((
        f"fa_count_active_{i + 1:02d}", "count_active",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             active_feats=list(coef_map.keys())),
    ))

# fa_pair_importance (2)
for i, (coef_map, n_feat, seed, a, b, tgt) in enumerate([
    ({0: 1.0, 2: 4.0, 3: 1.5, 4: 0.5}, 5, 4901, 2, 4, "x2"),
    ({1: 2.0, 3: 5.0, 5: 0.8, 0: 0.3}, 6, 4902, 3, 5, "x3"),
]):
    _FA_SPECS.append((
        f"fa_pair_importance_{i + 1:02d}", "pair_importance",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             a=a, b=b, target=tgt),
    ))

assert len(_FA_SPECS) == 26, len(_FA_SPECS)


# ===== POINT SIMULATION (26) ==============================================
# 9 subfamilies × (3,3,3,3,3,3,3,3,2) = 26. Samples avoid original-suite
# canonical query points (x0=2.0, x0=1.7, (1.0,0.5,...), etc.).

_PS_SPECS = []

# ps_basic (3) — small linear, varied dims, unusual sample values
for i, (coefs, seed, sample, tol_abs) in enumerate([
    ([-2.5, 3.5], 5101, [-0.64, 1.28], 0.8),
    ([1.6, -2.2, 2.8], 5102, [-0.83, 0.47, -1.18], 0.9),
    ([3.2, 1.4, -2.1, 0.9], 5103, [0.55, -0.92, 1.11, 0.38], 1.0),
]):
    _PS_SPECS.append((
        f"ps_basic_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=tol_abs),
    ))

# ps_dense (3) — all features active, dim 7–9
for i, (coefs, seed, sample) in enumerate([
    ([1.8, -2.6, 1.4, -0.9, 1.1, -0.7, 0.5], 5201,
     [0.42, -0.68, 1.12, -0.77, 0.33, 0.91, -1.05]),
    ([-2.0, 1.8, -1.3, 1.0, -0.7, 0.6, -0.4, 0.3], 5202,
     [-0.83, 1.14, 0.57, -0.92, 0.31, -0.44, 1.08, 0.62]),
    ([2.2, -1.8, 1.5, -1.1, 0.8, -0.6, 0.5, -0.4, 0.3], 5203,
     [0.27, -0.81, 0.94, 1.17, -0.52, 0.38, -0.66, 0.93, -0.22]),
]):
    _PS_SPECS.append((
        f"ps_dense_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=1.5),
    ))


def _sparse_coefs(n, active_map):
    return [active_map.get(k, 0.0) for k in range(n)]


# ps_sparse (3) — active indices mid-to-high, samples perturb those
for i, (n_feat, coef_map, seed, sample_map) in enumerate([
    (10, {2: 4.5, 5: -2.5, 8: 1.5}, 5301, {2: 0.83, 5: -0.47, 8: 1.19, 3: 0.2}),
    (14, {3: 5.0, 7: -2.8, 11: 1.6}, 5302, {3: -0.91, 7: 1.27, 11: -0.35, 4: 0.1}),
    (18, {4: 4.8, 9: -3.2, 13: 2.0, 16: -1.5}, 5303,
     {4: 1.07, 9: -0.64, 13: 0.42, 16: -0.88}),
]):
    coefs = _sparse_coefs(n_feat, coef_map)
    sample = [0.0] * n_feat
    for k, v in sample_map.items():
        sample[k] = v
    _PS_SPECS.append((
        f"ps_sparse_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=2.0),
    ))

# ps_alt_sign (3)
for i, (coefs, seed, sample) in enumerate([
    ([2.4, -1.8, 1.6, -1.1, 0.9], 5401, [0.63, -0.88, 1.12, -0.47, 0.73]),
    ([-3.2, 2.1, -1.4, 0.8, -0.5, 0.3], 5402, [-0.41, 1.09, -0.77, 0.54, -0.88, 0.32]),
    ([1.9, -2.2, 1.6, -1.0, 0.7, -0.4, 0.3], 5403,
     [0.82, 1.07, -0.55, 0.44, -0.91, 0.28, -0.72]),
]):
    _PS_SPECS.append((
        f"ps_alt_sign_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=1.2),
    ))

# ps_tail (3) — one or more features in the |x|>=1.6 tail
for i, (coefs, seed, sample) in enumerate([
    ([2.2, -1.6, 1.0], 5501, [-1.95, 0.35, 2.10]),
    ([1.8, 2.0, -1.4, 0.8], 5502, [1.85, -1.70, 0.28, -0.55]),
    ([-1.5, 2.4, -1.2, 0.8, -0.6], 5503, [0.22, -1.80, 2.05, 1.65, -0.35]),
]):
    _PS_SPECS.append((
        f"ps_tail_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=1.5),
    ))

# ps_irrational (3) — unusual non-round values
for i, (coefs, seed, sample) in enumerate([
    ([2.1, 1.7, -1.3, 0.7], 5601, [0.143, -0.928, 0.671, -0.239]),
    ([1.9, -2.3, 1.2, -0.8, 0.5], 5602, [-0.314, 0.827, -1.061, 0.459, 0.193]),
    ([2.4, 1.6, -1.1, 0.6, -0.4, 0.3], 5603,
     [0.577, -1.113, 0.849, 0.364, -0.792, 1.012]),
]):
    _PS_SPECS.append((
        f"ps_irrational_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=1.2),
    ))

# ps_partial (3) — question phrased as "x{a}=.., x{b}=.., others 0"
for i, (coef_map, n_feat, seed, active_vals) in enumerate([
    ({1: 3.0, 3: -2.0, 5: 1.4}, 6, 5701, {1: 0.90, 3: -1.27, 5: 0.58}),
    ({2: 4.0, 4: -2.5, 7: 1.8}, 10, 5702, {2: -0.71, 4: 1.15, 7: -0.43}),
    ({0: 2.5, 3: -3.0, 6: 1.8, 9: -1.2}, 12, 5703,
     {0: 0.88, 3: -0.62, 6: 1.04, 9: 0.77}),
]):
    coefs = _sparse_coefs(n_feat, coef_map)
    sample = [0.0] * n_feat
    for k, v in active_vals.items():
        sample[k] = v
    _PS_SPECS.append((
        f"ps_partial_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample=sample, tol_rel=0.2, tol_abs=1.5),
    ))

# ps_intercept (3) — large intercept
for i, (coefs, seed, sample, intercept) in enumerate([
    ([2.0, -1.5, 1.0], 5801, [0.53, -0.81, 0.44], 7.5),
    ([1.8, -2.0, 1.3, 0.8], 5802, [-0.27, 0.91, 0.56, -1.02], -5.0),
    ([2.2, 1.7, -1.1, 0.6, -0.4], 5803, [0.72, -0.44, 0.81, 0.28, -0.55], 10.0),
]):
    _PS_SPECS.append((
        f"ps_intercept_{i + 1:02d}", "point_pred",
        dict(data_fn=lambda c=coefs, s=seed, b=intercept: _lin(c, seed=s, intercept=b),
             sample=sample, tol_rel=0.2, tol_abs=1.5),
    ))

# ps_compare (2) — diff between two samples
for i, (coefs, seed, sa, sb) in enumerate([
    ([2.0, -1.5, 1.2, 0.6], 5901, [0.5, -0.3, 0.7, -0.2], [1.2, 0.8, -0.4, 0.9]),
    ([1.8, 2.3, -1.4, 0.8, -0.5], 5902,
     [-0.2, 0.9, -0.6, 0.4, 0.1], [0.8, -0.4, 1.1, -0.8, 0.3]),
]):
    _PS_SPECS.append((
        f"ps_compare_{i + 1:02d}", "sample_diff",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             sample_a=sa, sample_b=sb, tol_rel=0.2, tol_abs=1.2),
    ))

assert len(_PS_SPECS) == 26, len(_PS_SPECS)


# ===== SENSITIVITY (26) ==================================================
# 9 subfamilies × (3,3,3,3,3,3,3,3,2) = 26. Bases vary (non-zero), k varies
# (never always 0).

_SN_SPECS = []

# sn_unit_delta (3) — +1 change on mid-index feat, non-zero base
for i, (coefs, k, seed, base) in enumerate([
    ([2.0, -1.5, 3.5, 0.8], 2, 6101, [0.3, -0.2, 0.1, 0.0]),
    ([-2.5, 3.0, -1.5, 0.8, 0.5], 1, 6102, [0.2, 0.3, -0.1, 0.0, 0.1]),
    ([1.8, -2.2, 1.5, 2.8, -0.8], 3, 6103, [0.1, -0.2, 0.3, 0.0, 0.2]),
]):
    _SN_SPECS.append((
        f"sn_unit_delta_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=base[k] + 1.0),
    ))

# sn_wide_delta (3)
for i, (coefs, k, seed, base, new_val) in enumerate([
    ([2.5, -1.8, 1.4, 0.7], 1, 6201, [0.2, -0.5, 0.3, 0.0], 1.8),
    ([1.6, 2.0, -1.5, 0.8, -0.5], 3, 6202, [-0.3, 0.2, 0.1, 0.5, 0.0], -1.6),
    ([-1.8, 2.4, -1.3, 0.9, -0.5, 0.3], 4, 6203,
     [0.2, -0.1, 0.3, -0.2, 0.1, 0.0], 2.2),
]):
    _SN_SPECS.append((
        f"sn_wide_delta_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=new_val),
    ))

# sn_tight_unit (3) — tight tolerance on +1 at varied k's
for i, (coef_map, n_feat, k, seed, base) in enumerate([
    ({1: 4.0, 2: -2.0}, 4, 1, 6301, [0.0, 0.3, -0.2, 0.0]),
    ({2: 3.0, 3: -2.5, 4: 1.5}, 6, 3, 6302, [0.1, -0.1, 0.2, 0.0, 0.1, 0.0]),
    ({0: 2.5, 2: -3.0, 5: 1.5}, 7, 5, 6303,
     [-0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.1]),
]):
    _SN_SPECS.append((
        f"sn_tight_unit_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, base=base, new_val=base[k] + 1.0,
             tol_rel=0.12, tol_abs=0.4),
    ))

# sn_crossing (3) — target levels vary
for i, (coef_map, n_feat, k, seed, base, target) in enumerate([
    ({1: 3.0, 0: 0.5}, 3, 1, 6401, [0.0, 0.0, 0.0], 2.0),
    ({2: -4.0, 0: 0.5, 1: 0.3}, 4, 2, 6402, [0.0, 0.0, 0.0, 0.0], -3.0),
    ({3: 3.5, 0: 0.4, 1: -0.3, 2: 0.2}, 5, 3, 6403,
     [0.0, 0.0, 0.0, 0.0, 0.0], 2.5),
]):
    _SN_SPECS.append((
        f"sn_crossing_{i + 1:02d}", "crossing",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s, noise=0.2),
             k=k, fixed_base=base, target_level=target,
             search_range=(-3.0, 3.0), tol=0.7),
    ))

# sn_two_feat (3) — simultaneous change on (k1, k2) != (0, 1)
for i, (coef_map, n_feat, seed, base, k1, n1, k2, n2) in enumerate([
    ({1: 3.0, 2: 2.0, 0: 0.5}, 4, 6501, [0.0, 0.0, 0.0, 0.0], 1, 1.5, 2, 2.0),
    ({2: -2.5, 3: 2.0, 0: 0.4, 1: -0.3}, 5, 6502,
     [0.2, -0.1, 0.0, 0.1, 0.0], 2, -1.4, 3, 1.6),
    ({3: 2.0, 4: -1.8, 0: 0.3, 2: 0.4}, 6, 6503,
     [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], 3, 1.8, 4, -1.2),
]):
    _SN_SPECS.append((
        f"sn_two_feat_{i + 1:02d}", "two_feat",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             base=base, k1=k1, new1=n1, k2=k2, new2=n2),
    ))

# sn_decrease (3) — decrease from nonzero base
for i, (coefs, k, seed, base, new_val) in enumerate([
    ([1.5, 3.0, -1.8, 0.8], 1, 6601, [0.2, 0.6, -0.3, 0.1], -0.4),
    ([2.0, -1.5, 2.8, 0.8, -0.4], 2, 6602, [0.1, 0.2, 0.7, 0.0, -0.1], -0.3),
    ([1.3, 2.0, -1.5, 3.0, 0.4, -0.3], 3, 6603,
     [0.2, -0.1, 0.3, 0.5, 0.0, 0.1], -0.5),
]):
    _SN_SPECS.append((
        f"sn_decrease_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=new_val, wording="decrease"),
    ))

# sn_multi_step (3) — 3-unit change
for i, (coefs, k, seed, base) in enumerate([
    ([2.5, -1.5, 1.8, 0.6], 0, 6701, [-0.5, 0.2, -0.1, 0.0]),
    ([-1.8, 2.5, -1.0, 0.8, 0.4], 2, 6702, [0.1, -0.2, 0.3, 0.0, 0.1]),
    ([1.6, -2.0, 1.2, 2.5, -0.6], 3, 6703, [0.0, 0.1, -0.1, -0.5, 0.0]),
]):
    _SN_SPECS.append((
        f"sn_multi_step_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=base[k] + 3.0, wording="multi"),
    ))

# sn_nonzero (3) — very non-trivial base, change varied k
for i, (coefs, k, seed, base, new_val) in enumerate([
    ([2.0, 1.8, -1.5, 0.8, -0.5], 1, 6801,
     [0.4, -0.6, 0.3, 0.2, -0.1], 0.8),
    ([1.6, -1.8, 2.5, 0.8, -0.4, 0.3], 2, 6802,
     [0.3, -0.4, 0.5, -0.1, 0.2, 0.0], -0.7),
    ([-1.5, 2.0, -1.2, 1.5, 0.8, -0.4], 4, 6803,
     [0.2, -0.3, 0.4, -0.2, 0.3, 0.1], 1.5),
]):
    _SN_SPECS.append((
        f"sn_nonzero_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=new_val),
    ))

# sn_small_step (2) — 0.5-unit change
for i, (coefs, k, seed, base) in enumerate([
    ([3.0, -2.0, 1.4, 0.8], 2, 6901, [0.1, 0.2, 0.0, -0.1]),
    ([-2.5, 2.0, -1.3, 0.8, 0.5], 3, 6902, [0.2, -0.1, 0.1, 0.0, -0.2]),
]):
    _SN_SPECS.append((
        f"sn_small_step_{i + 1:02d}", "sensitivity",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, new_val=base[k] + 0.5,
             tol_rel=0.2, tol_abs=0.5),
    ))

assert len(_SN_SPECS) == 26, len(_SN_SPECS)


# ===== COUNTERFACTUAL (26) ================================================
# 9 subfamilies × (3,3,3,3,3,3,3,3,2) = 26. Feature to invert varies widely.

_CF_SPECS = []

# cf_linear (3) — varied k, non-trivial base
for i, (coefs, seed, base, k, b_k) in enumerate([
    ([2.5, -1.8, 3.0, 0.8], 7101, [0.3, 0.5, -0.2, 0.1], 2, 1.6),
    ([-2.0, 3.0, -1.5, 1.2], 7102, [0.2, -0.3, 0.4, -0.1], 1, -2.0),
    ([1.8, -2.2, 2.5, -1.0, 0.6], 7103, [0.3, 0.2, -0.1, 0.0, 0.2], 0, 2.2),
]):
    _CF_SPECS.append((
        f"cf_linear_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

# cf_varied_feat (3) — k = 1, 2, 3 (never 0)
for i, (coefs, seed, base, k, b_k) in enumerate([
    ([1.5, 3.5, -2.0, 0.8, 0.4], 7201, [0.2, 0.3, -0.1, 0.0, 0.1], 1, -1.5),
    ([-1.8, 1.2, 3.0, -1.0, 0.6], 7202, [0.1, 0.2, 0.3, 0.0, -0.1], 2, 1.8),
    ([1.4, -1.8, 1.2, 3.0, -0.5], 7203, [0.2, -0.1, 0.3, 0.1, 0.0], 3, -2.0),
]):
    _CF_SPECS.append((
        f"cf_varied_feat_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

# cf_sparse (3) — large feature space, active mid-index feature inverted
for i, (n_feat, coef_map, seed, k, b_k) in enumerate([
    (9, {1: 4.0, 3: -2.5, 6: 1.5}, 7301, 3, 1.6),
    (11, {2: 3.5, 5: -3.0, 8: 2.0}, 7302, 5, -1.5),
    (14, {3: 4.0, 7: -3.0, 11: 2.0}, 7303, 7, 1.8),
]):
    coefs = _sparse_coefs(n_feat, coef_map)
    base = [0.0] * n_feat
    _CF_SPECS.append((
        f"cf_sparse_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))


# Dispatch for nonlinear families — helpers return X,y with known active idx
def _nonlin_data_fn(kind, kwargs):
    if kind == "relu_at":
        return lambda: _relu_at(**kwargs)
    if kind == "quad_at":
        return lambda: _quad_at(**kwargs)
    if kind == "piecewise3_at":
        return lambda: _piecewise3_at(**kwargs)
    if kind == "absv_at":
        return lambda: _absv_at(**kwargs)
    if kind == "interact_at":
        return lambda: _interact_at(**kwargs)
    if kind == "triple_at":
        return lambda: _triple_at(**kwargs)
    if kind == "cascade_at":
        return lambda: _cascade_at(**kwargs)
    if kind == "exp_decay_at":
        return lambda: _exp_decay_at(**kwargs)
    if kind == "sine_at":
        return lambda: _sine_at(**kwargs)
    if kind == "fried_at":
        return lambda: _fried_at(**kwargs)
    raise ValueError(kind)


# cf_nonlinear (3) — active nonlinearity at varied indices (x1, x2, x3)
for i, (kind, kwargs, base, k, b_k) in enumerate([
    ("relu_at", dict(idx=1, coef=3.0, n_feat=4, seed=7401),
     [0.2, 0.4, 0.0, 0.0], 1, 1.4),
    ("quad_at", dict(i_sq_a=2, i_sq_b=0, i_lin=1, c0=2.0, c1=-1.0, c2=0.8,
                     n_feat=5, seed=7402),
     [0.0, 0.3, 1.0, 0.0, 0.0], 2, 1.8),
    ("piecewise3_at", dict(idx=3, seed=7403, n_feat=5),
     [0.0, 0.1, 0.0, 0.3, 0.0], 3, -1.8),
]):
    _CF_SPECS.append((
        f"cf_nonlinear_{i + 1:02d}", "counterfactual",
        dict(data_fn=_nonlin_data_fn(kind, kwargs),
             k=k, base=base, b_k=b_k,
             tol_abs=0.9, tol_rel=0.3),
    ))

# cf_intercept (3) — large intercept, varied k
for i, (coefs, seed, base, k, b_k, intercept) in enumerate([
    ([2.0, -1.8, 1.5, 0.7], 7501, [0.2, 0.3, 0.0, -0.1], 1, -1.6, 6.0),
    ([-1.6, 2.5, -1.3, 0.8, 0.4], 7502, [0.1, -0.2, 0.3, 0.0, 0.1], 2, 1.8, -4.0),
    ([1.5, 1.8, -2.0, 0.6, -0.4], 7503, [0.3, 0.1, -0.2, 0.2, 0.0], 3, -2.0, 8.0),
]):
    _CF_SPECS.append((
        f"cf_intercept_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed, b=intercept: _lin(c, seed=s, intercept=b),
             k=k, base=base, b_k=b_k),
    ))

# cf_reverse (3) — target a LOWER prediction (y_B < y_A)
for i, (coefs, seed, base, k, b_k) in enumerate([
    ([1.8, 2.2, -1.5, 0.7], 7601, [0.5, 0.8, -0.2, 0.1], 1, -1.4),
    ([-2.0, 1.5, 2.8, -1.0, 0.5], 7602, [-0.3, 0.4, 0.6, -0.1, 0.0], 2, -1.6),
    ([1.4, -1.8, 1.6, 2.5, -0.6, 0.3], 7603,
     [0.3, 0.2, -0.1, 0.7, 0.0, 0.1], 3, -1.8),
]):
    _CF_SPECS.append((
        f"cf_reverse_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

# cf_large_change (3) — |b_k - base[k]| is large
for i, (coefs, seed, base, k, b_k) in enumerate([
    ([2.5, -2.0, 1.5, 0.8], 7701, [0.5, -0.4, 0.3, 0.0], 2, 2.8),
    ([-1.6, 2.3, 1.5, -1.0, 0.6], 7702, [0.4, -0.3, 0.2, 0.1, 0.0], 1, -2.5),
    ([1.8, 1.5, -1.8, 2.2, 0.5, -0.3], 7703,
     [-0.2, 0.3, 0.1, 0.5, -0.1, 0.1], 0, -2.6),
]):
    _CF_SPECS.append((
        f"cf_large_change_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

# cf_mid_index (3) — k = 4, 5, 6
for i, (coef_map, n_feat, seed, base, k, b_k) in enumerate([
    ({0: 1.5, 1: -1.8, 4: 3.0, 3: 0.5}, 5, 7801, [0.2, 0.3, 0.0, -0.1, 0.1], 4, 1.6),
    ({0: 1.2, 2: -1.8, 5: 3.5, 1: 0.5}, 6, 7802,
     [-0.1, 0.2, 0.3, 0.0, 0.1, 0.0], 5, -1.8),
    ({1: 1.5, 3: -2.0, 6: 3.0, 2: 0.5, 0: 0.3}, 7, 7803,
     [0.2, -0.3, 0.1, 0.2, 0.0, -0.1, 0.0], 6, 1.8),
]):
    coefs = _sparse_coefs(n_feat, coef_map)
    _CF_SPECS.append((
        f"cf_mid_index_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

# cf_neg_target (2) — y_B negative
for i, (coefs, seed, base, k, b_k) in enumerate([
    ([2.0, -1.8, 1.5, 0.6], 7901, [0.6, 0.2, 0.3, 0.0], 0, -1.5),
    ([-1.5, 2.0, -1.3, 0.8, -0.5], 7902, [0.3, 0.5, -0.2, 0.2, 0.0], 1, -1.4),
]):
    _CF_SPECS.append((
        f"cf_neg_target_{i + 1:02d}", "counterfactual",
        dict(data_fn=lambda c=coefs, s=seed: _lin(c, seed=s),
             k=k, base=base, b_k=b_k),
    ))

assert len(_CF_SPECS) == 26, len(_CF_SPECS)


# ===== STRUCTURAL (26) ====================================================
# 9 subfamilies × (3,3,3,3,3,3,3,3,2) = 26. Feature k varies.

_ST_SPECS = []

# st_decision (3) — crossing at varied features and levels
for i, (coef_map, n_feat, k, seed, base, target) in enumerate([
    ({1: 3.5, 0: 0.5, 2: -0.3}, 4, 1, 8101, [0.0, 0.0, 0.0, 0.0], 3.0),
    ({2: -4.0, 0: 0.5, 1: 0.3, 3: -0.2}, 5, 2, 8102,
     [0.0, 0.0, 0.0, 0.0, 0.0], -3.0),
    ({3: 3.0, 0: 0.4, 1: -0.3, 2: 0.2, 4: 0.1}, 5, 3, 8103,
     [0.0, 0.0, 0.0, 0.0, 0.0], 2.0),
]):
    _ST_SPECS.append((
        f"st_decision_{i + 1:02d}", "crossing",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, fixed_base=base, target_level=target,
             search_range=(-3.5, 3.5), tol=0.8),
    ))

# st_compact (3) — varied op thresholds
for i, thr in enumerate([6, 12, 25]):
    coef_map = {0: 3.5, 2: -1.8, 4: 1.2}
    seed = 8201 + i
    _ST_SPECS.append((
        f"st_compact_{i + 1:02d}", "compactness",
        dict(data_fn=lambda cm=coef_map, s=seed: _lin_at(cm, 5, seed=s),
             op_threshold=thr),
    ))

# st_argmax (3) — varied k (1, 2, 3) with concave-in-k DGPs (quadratic, relu)
for i, (kind, kwargs, base, k, search_range) in enumerate([
    ("quad_at", dict(i_sq_a=1, i_sq_b=3, i_lin=2, c0=-2.0, c1=-1.0, c2=0.3,
                     n_feat=4, seed=8301),
     [0.0, 0.0, 0.0, 0.0], 1, (-2.5, 2.5)),
    ("quad_at", dict(i_sq_a=2, i_sq_b=0, i_lin=3, c0=-1.8, c1=-0.8, c2=0.2,
                     n_feat=5, seed=8302),
     [0.0, 0.0, 0.0, 0.0, 0.0], 2, (-2.5, 2.5)),
    ("quad_at", dict(i_sq_a=3, i_sq_b=1, i_lin=0, c0=-2.0, c1=-0.6, c2=0.2,
                     n_feat=5, seed=8303),
     [0.0, 0.0, 0.0, 0.0, 0.0], 3, (-2.5, 2.5)),
]):
    _ST_SPECS.append((
        f"st_argmax_{i + 1:02d}", "argmax",
        dict(data_fn=_nonlin_data_fn(kind, kwargs),
             k=k, fixed_base=base, search_range=search_range, tol=0.9),
    ))

# st_argmin (3) — varied k (0, 2, 4) with convex-in-k quadratics
for i, (kind, kwargs, base, k, search_range) in enumerate([
    ("quad_at", dict(i_sq_a=0, i_sq_b=2, i_lin=1, c0=2.0, c1=0.8, c2=0.2,
                     n_feat=4, seed=8401),
     [0.0, 0.0, 0.0, 0.0], 0, (-2.5, 2.5)),
    ("quad_at", dict(i_sq_a=2, i_sq_b=4, i_lin=1, c0=1.8, c1=0.6, c2=0.2,
                     n_feat=5, seed=8402),
     [0.0, 0.0, 0.0, 0.0, 0.0], 2, (-2.5, 2.5)),
    ("quad_at", dict(i_sq_a=4, i_sq_b=2, i_lin=1, c0=2.2, c1=0.6, c2=0.2,
                     n_feat=5, seed=8403),
     [0.0, 0.0, 0.0, 0.0, 0.0], 4, (-2.5, 2.5)),
]):
    _ST_SPECS.append((
        f"st_argmin_{i + 1:02d}", "argmin",
        dict(data_fn=_nonlin_data_fn(kind, kwargs),
             k=k, fixed_base=base, search_range=search_range, tol=0.9),
    ))

# st_monotonic (3) — positive linear (increasing), negative linear (decreasing),
# quadratic (non-monotonic)
for i, (coef_map, n_feat, k, seed, base) in enumerate([
    ({2: 3.0, 0: 0.5, 1: -0.3}, 4, 2, 8501, [0.0, 0.0, 0.0, 0.0]),
    ({3: -4.0, 0: 0.3, 1: 0.2, 2: -0.1}, 5, 3, 8502,
     [0.0, 0.0, 0.0, 0.0, 0.0]),
]):
    _ST_SPECS.append((
        f"st_monotonic_{i + 1:02d}", "monotonic",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, fixed_base=base, search_range=(-1.8, 1.8)),
    ))
# third: quadratic → non-monotonic
_ST_SPECS.append((
    "st_monotonic_03", "monotonic",
    dict(data_fn=lambda: _quad_at(i_sq_a=1, i_sq_b=3, i_lin=0, c0=2.0, c1=0.6,
                                  c2=0.3, n_feat=4, seed=8503),
         k=1, fixed_base=[0.0, 0.0, 0.0, 0.0], search_range=(-2.0, 2.0)),
))

# st_output_range (3) — span of output over a range for varied k
for i, (coef_map, n_feat, k, seed, base, span_range) in enumerate([
    ({2: 3.5, 0: 0.4, 1: -0.3}, 4, 2, 8601, [0.0, 0.0, 0.0, 0.0], (-2.0, 2.0)),
    ({3: -2.8, 1: 0.3, 0: 0.2, 2: -0.1}, 5, 3, 8602,
     [0.0, 0.0, 0.0, 0.0, 0.0], (-1.8, 1.8)),
    ({4: 3.0, 0: 0.4, 1: -0.2, 2: 0.3, 3: -0.1}, 6, 4, 8603,
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (-2.0, 2.0)),
]):
    _ST_SPECS.append((
        f"st_output_range_{i + 1:02d}", "output_range",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, fixed_base=base, search_range=span_range,
             tol_rel=0.2, tol_abs=1.0),
    ))

# st_decision_mid (3) — crossing at mid-index k (4, 5, 6)
for i, (coef_map, n_feat, k, seed, base, target) in enumerate([
    ({4: 3.5, 0: 0.5, 1: -0.3, 2: 0.2}, 6, 4, 8701,
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0),
    ({5: -3.0, 1: 0.3, 2: 0.2, 3: -0.1}, 7, 5, 8702,
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], -2.5),
    ({6: 2.8, 0: 0.4, 2: -0.2, 4: 0.2}, 8, 6, 8703,
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.5),
]):
    _ST_SPECS.append((
        f"st_decision_mid_{i + 1:02d}", "crossing",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, fixed_base=base, target_level=target,
             search_range=(-3.5, 3.5), tol=0.9),
    ))

# st_argmax_nonlin (3) — piecewise / abs DGP, argmax over x_k
for i, (kind, kwargs, base, k, search_range) in enumerate([
    ("absv_at", dict(i_pos=0, i_neg=2, i_lin=1, seed=8801, n_feat=4),
     [0.0, 0.0, 0.0, 0.0], 2, (-2.0, 2.0)),
    ("piecewise3_at", dict(idx=1, seed=8802, n_feat=4),
     [0.0, 0.0, 0.0, 0.0], 1, (-0.5, 2.5)),
    ("absv_at", dict(i_pos=1, i_neg=3, i_lin=0, seed=8803, n_feat=5),
     [0.0, 0.0, 0.0, 0.0, 0.0], 1, (-2.0, 2.0)),
]):
    _ST_SPECS.append((
        f"st_argmax_nonlin_{i + 1:02d}", "argmax",
        dict(data_fn=_nonlin_data_fn(kind, kwargs),
             k=k, fixed_base=base, search_range=search_range, tol=1.1),
    ))

# st_plateau (2) — output_range but narrow interval
for i, (coef_map, n_feat, k, seed, base) in enumerate([
    ({2: 0.2, 0: 3.0, 1: -0.5}, 4, 2, 8901, [0.0, 0.0, 0.0, 0.0]),
    ({3: 0.15, 0: 2.5, 1: -0.3, 2: 0.2}, 5, 3, 8902,
     [0.0, 0.0, 0.0, 0.0, 0.0]),
]):
    _ST_SPECS.append((
        f"st_plateau_{i + 1:02d}", "output_range",
        dict(data_fn=lambda cm=coef_map, nf=n_feat, s=seed: _lin_at(cm, nf, seed=s),
             k=k, fixed_base=base, search_range=(-1.0, 1.0),
             tol_rel=0.3, tol_abs=0.6),
    ))

assert len(_ST_SPECS) == 26, len(_ST_SPECS)


# ===== COMPLEX FN SIMULATION (27) =========================================
# 9 subfamilies × 3 = 27. Active indices are permuted across the 3 tests in
# each subfamily so the "correct answer" (prediction value) depends on
# which features the nonlinearity actually uses.

_CX_SPECS = []


def _cx_spec(name, kind, kwargs, sample, tol_abs=1.2, tol_rel=0.25):
    _CX_SPECS.append((
        name, "point_pred",
        dict(data_fn=_nonlin_data_fn(kind, kwargs),
             sample=sample, tol_rel=tol_rel, tol_abs=tol_abs, min_r2=None),
    ))


# cx_quad (3) — quadratic in varied pairs of indices
_cx_spec("cx_quad_01", "quad_at",
         dict(i_sq_a=1, i_sq_b=3, i_lin=0, c0=2.0, c1=-1.5, c2=1.0,
              n_feat=5, seed=9101),
         [0.8, 1.2, -0.3, 0.6, -0.2])
_cx_spec("cx_quad_02", "quad_at",
         dict(i_sq_a=2, i_sq_b=4, i_lin=1, c0=1.8, c1=-1.2, c2=1.1,
              n_feat=5, seed=9102),
         [0.3, -0.6, 1.1, 0.4, -0.8])
_cx_spec("cx_quad_03", "quad_at",
         dict(i_sq_a=0, i_sq_b=2, i_lin=3, c0=2.2, c1=-1.0, c2=0.9,
              n_feat=6, seed=9103),
         [-0.7, 0.8, 0.9, 0.5, -0.2, 0.3])

# cx_interact (3)
_cx_spec("cx_interact_01", "interact_at",
         dict(i0=1, i1=2, c0=2.5, c1=2.0, c_int=1.6, n_feat=4, seed=9201),
         [0.3, 1.1, 0.8, -0.2])
_cx_spec("cx_interact_02", "interact_at",
         dict(i0=2, i1=3, c0=3.0, c1=2.2, c_int=-1.8, n_feat=5, seed=9202),
         [-0.4, 0.5, 0.9, -0.7, 0.3])
_cx_spec("cx_interact_03", "interact_at",
         dict(i0=0, i1=3, c0=3.5, c1=1.8, c_int=1.8, n_feat=5, seed=9203),
         [0.8, 0.2, -0.5, 0.6, 0.1])

# cx_triple (3)
_cx_spec("cx_triple_01", "triple_at",
         dict(i0=1, i1=2, i2=3, i3=4, seed=9301, n_feat=6),
         [0.3, 0.8, -0.4, 1.0, 0.5, -0.1],
         tol_abs=2.0, tol_rel=0.25)
_cx_spec("cx_triple_02", "triple_at",
         dict(i0=0, i1=2, i2=4, i3=1, seed=9302, n_feat=6),
         [1.1, -0.3, 0.6, 0.2, -0.8, 0.4],
         tol_abs=2.0, tol_rel=0.25)
_cx_spec("cx_triple_03", "triple_at",
         dict(i0=2, i1=3, i2=0, i3=1, seed=9303, n_feat=7),
         [0.7, -0.5, 0.8, 1.0, 0.2, 0.0, -0.3],
         tol_abs=2.0, tol_rel=0.25)

# cx_fried (3) — Friedman-1 with permuted active 5-tuple
_cx_spec("cx_fried_01", "fried_at",
         dict(perm=(1, 3, 5, 7, 9), seed=9401),
         [0.3, 0.7, 0.5, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3, 0.6],
         tol_abs=2.5, tol_rel=0.3)
_cx_spec("cx_fried_02", "fried_at",
         dict(perm=(0, 2, 4, 6, 8), seed=9402),
         [0.6, 0.3, 0.8, 0.4, 0.5, 0.7, 0.2, 0.5, 0.8, 0.3],
         tol_abs=2.5, tol_rel=0.3)
_cx_spec("cx_fried_03", "fried_at",
         dict(perm=(2, 5, 8, 1, 4), seed=9403),
         [0.4, 0.6, 0.7, 0.5, 0.3, 0.8, 0.2, 0.6, 0.5, 0.4],
         tol_abs=2.5, tol_rel=0.3)

# cx_cascade (3) — gate index varies
_cx_spec("cx_cascade_01", "cascade_at",
         dict(i_gate=1, i_pos=2, i_neg=3, seed=9501, n_feat=6),
         [0.2, 0.8, 0.9, -0.5, 0.1, -0.1])
_cx_spec("cx_cascade_02", "cascade_at",
         dict(i_gate=2, i_pos=4, i_neg=1, seed=9502, n_feat=6),
         [-0.3, 0.5, -0.7, 0.2, 0.8, 0.1])
_cx_spec("cx_cascade_03", "cascade_at",
         dict(i_gate=0, i_pos=3, i_neg=4, seed=9503, n_feat=6),
         [0.5, -0.2, 0.4, -0.8, 1.0, 0.2])

# cx_exp (3) — exponential-decay active index varies
_cx_spec("cx_exp_01", "exp_decay_at",
         dict(i_exp=1, i_lin=2, seed=9601, n_feat=4),
         [0.3, -0.4, 1.0, 0.0])
_cx_spec("cx_exp_02", "exp_decay_at",
         dict(i_exp=0, i_lin=2, seed=9602, n_feat=5),
         [-0.5, 0.2, 0.8, 0.0, 0.3])
_cx_spec("cx_exp_03", "exp_decay_at",
         dict(i_exp=2, i_lin=0, seed=9603, n_feat=5),
         [0.7, 0.1, -0.3, 0.0, 0.2])

# cx_piece (3) — piecewise-linear active index varies
_cx_spec("cx_piece_01", "piecewise3_at",
         dict(idx=1, seed=9701, n_feat=4),
         [0.2, 0.5, 0.0, 0.3])
_cx_spec("cx_piece_02", "piecewise3_at",
         dict(idx=2, seed=9702, n_feat=4),
         [-0.1, 0.2, -1.5, 0.0])
_cx_spec("cx_piece_03", "piecewise3_at",
         dict(idx=0, seed=9703, n_feat=5),
         [1.5, 0.2, 0.0, -0.2, 0.1])

# cx_sine (3) — sin/cos active indices varied
_cx_spec("cx_sine_01", "sine_at",
         dict(i_sin=1, i_cos=3, i_lin=2, seed=9801, n_feat=5),
         [0.0, 0.8, 0.3, 0.5, 0.0])
_cx_spec("cx_sine_02", "sine_at",
         dict(i_sin=2, i_cos=0, i_lin=3, seed=9802, n_feat=5),
         [-0.6, 0.1, 0.7, 0.4, 0.0])
_cx_spec("cx_sine_03", "sine_at",
         dict(i_sin=0, i_cos=4, i_lin=1, seed=9803, n_feat=5),
         [1.0, -0.5, 0.2, 0.0, 0.6])

# cx_abs (3)
_cx_spec("cx_abs_01", "absv_at",
         dict(i_pos=2, i_neg=0, i_lin=3, seed=9901, n_feat=5),
         [0.7, 0.2, -1.1, 0.5, 0.0])
_cx_spec("cx_abs_02", "absv_at",
         dict(i_pos=1, i_neg=3, i_lin=2, seed=9902, n_feat=5),
         [-0.2, 1.2, -0.4, 0.6, 0.1])
_cx_spec("cx_abs_03", "absv_at",
         dict(i_pos=3, i_neg=1, i_lin=4, seed=9903, n_feat=5),
         [0.3, -0.8, 0.2, 1.0, 0.5])

assert len(_CX_SPECS) == 27, len(_CX_SPECS)


# ---------------------------------------------------------------------------
# Build callables
# ---------------------------------------------------------------------------

_KIND_TO_RUNNER = {
    "top_feature": _run_top_feature,
    "bottom_feature": _run_bottom_feature,
    "rank_top2": _run_rank_top2,
    "zero_effect": _run_zero_effect,
    "delta_unit": _run_delta_unit,
    "dominant_sample": _run_dominant_sample,
    "second_feature": _run_second_feature,
    "count_active": _run_count_active,
    "pair_importance": _run_pair_importance,
    "point_pred": _run_point_pred,
    "sample_diff": _run_sample_diff,
    "sensitivity": _run_sensitivity,
    "crossing": _run_crossing,
    "two_feat": _run_two_feature,
    "counterfactual": _run_counterfactual,
    "compactness": _run_compactness,
    "argmax": _run_argmax,
    "argmin": _run_argmin,
    "monotonic": _run_monotonic,
    "output_range": _run_output_range,
}


def _make_test_fn(test_name, kind, kwargs):
    runner = _KIND_TO_RUNNER[kind]

    def _fn(model, llm, _n=test_name, _r=runner, _kw=kwargs):
        return _r(_n, model, llm, **_kw)

    _fn.__name__ = test_name
    return _fn


FEATURE_ATTRIBUTION_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _FA_SPECS]
POINT_SIMULATION_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _PS_SPECS]
SENSITIVITY_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _SN_SPECS]
COUNTERFACTUAL_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _CF_SPECS]
STRUCTURAL_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _ST_SPECS]
COMPLEX_FN_TESTS = [_make_test_fn(n, k, kw) for n, k, kw in _CX_SPECS]

CATEGORY_TESTS = [
    ("feature_attribution", FEATURE_ATTRIBUTION_TESTS),
    ("point_simulation",    POINT_SIMULATION_TESTS),
    ("sensitivity",         SENSITIVITY_TESTS),
    ("counterfactual",      COUNTERFACTUAL_TESTS),
    ("structural",          STRUCTURAL_TESTS),
    ("complex_fn",          COMPLEX_FN_TESTS),
]

ALL_TESTS = [fn for _, lst in CATEGORY_TESTS for fn in lst]
assert len(ALL_TESTS) == 157, f"expected 157 tests, got {len(ALL_TESTS)}"

_ALL_TEST_FNS = {fn.__name__: fn for fn in ALL_TESTS}
_TEST_TO_CATEGORY = {fn.__name__: cat for cat, lst in CATEGORY_TESTS for fn in lst}


def category_of(test_name):
    return _TEST_TO_CATEGORY.get(test_name, "")


if __name__ == "__main__":
    print(f"Total tests: {len(ALL_TESTS)}")
    for cat, lst in CATEGORY_TESTS:
        print(f"  {cat:<22}: {len(lst)}")
    # Verify subfamily counts: no more than 3 per subfamily prefix.
    from collections import Counter
    prefixes = Counter()
    for fn in ALL_TESTS:
        parts = fn.__name__.rsplit("_", 1)
        prefixes[parts[0]] += 1
    over = {k: v for k, v in prefixes.items() if v > 3}
    print(f"Subfamilies: {len(prefixes)} total; over-3 = {over or 'none'}")
