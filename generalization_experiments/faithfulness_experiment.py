"""
Display-vs-predictor faithfulness experiment (rebuttal: mpaA W1/Q2, TdWu Q2).

For each of the 10 curated library models, we reconstruct predictions by
EXECUTING ONLY the equation/rules printed in the model's __str__ (the display
an LLM would read) and compare them against the model's actual .predict() on
held-out test inputs.

We report, per model, on the pooled held-out points across several real
regression datasets (y standardized so SD=1):
  R2(display -> predict) : how much of predict variance the display explains
  MAE / SD               : mean abs display-predict gap in outcome-SD units
  % within tol           : fraction of points with |display-predict| < 0.20*SD
                           (0.20*SD ~ the point-simulation grading tolerance)

A faithful display -> R2~1, MAE/SD~0, %within~1.
A display-predict decoupled model -> gap = magnitude of the hidden corrector.

Run:  python faithfulness_experiment.py
Out:  scratchpad/faithfulness/{faithfulness_long.csv, faithfulness_summary.csv}
"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pmlb import fetch_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "result_libs_processed", "agentic-imodels"))
from agentic_imodels import (
    HingeEBMRegressor, HybridGAM, SmartAdditiveRegressor, HingeGAMRegressor,
    TeacherStudentRuleSplineRegressor, DualPathSparseSymbolicRegressor,
    SparseSignedBasisPursuitRegressor, DistilledTreeBlendAtlasRegressor,
    WinsorizedSparseOLSRegressor, TinyDTDepth2Regressor)

OUT = "/private/tmp/claude-501/-Users-chandansingh-Downloads/f32dcc95-9574-4888-b12b-77a8bb0d41d5/scratchpad/faithfulness"
os.makedirs(OUT, exist_ok=True)
TOL = 0.20  # outcome-SD units

# ---------------------------------------------------------------------------
# Display-equation executor
# ---------------------------------------------------------------------------
NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"

def _split_top_terms(rhs):
    """Split an equation RHS on top-level +/- (parens-aware). Returns signed strings."""
    terms, buf, depth = [], "", 0
    i = 0
    # normalize: ensure leading sign
    rhs = rhs.strip()
    while i < len(rhs):
        c = rhs[i]
        if c in "([":
            depth += 1; buf += c
        elif c in ")]":
            depth -= 1; buf += c
        elif c in "+-" and depth == 0 and buf.strip() != "" and rhs[i-1] not in "eE(*":
            terms.append(buf); buf = c
        else:
            buf += c
        i += 1
    if buf.strip():
        terms.append(buf)
    return [t.strip() for t in terms if t.strip() not in ("", "+", "-")]

def _strip_wrap(body):
    """Strip a single fully-wrapping outer paren pair, if present."""
    body = body.strip()
    while body.startswith("(") and body.endswith(")"):
        depth = 0; wraps = True
        for i, c in enumerate(body):
            if c == "(": depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(body) - 1:
                    wraps = False; break
        if wraps: body = body[1:-1].strip()
        else: break
    return body

def _eval_term(term, X):
    """Evaluate one signed symbolic term over rows X (n,d) -> (n,) array."""
    t = term.replace(" ", "")
    n = X.shape[0]
    # fold the leading connective sign (from term splitting) into the coefficient
    sign = 1.0
    while t and t[0] in "+-":
        if t[0] == "-": sign = -sign
        t = t[1:]
    # (C)*BODY  or C*BODY  or bare constant
    m = re.match(r"^\(?(%s)\)?\*(.+)$" % NUM, t)
    if m:
        coef = sign * float(m.group(1)); body = m.group(2)
    else:
        # bare constant (intercept) or bare coef*x handled below
        m2 = re.match(r"^(%s)$" % NUM, t)
        if m2:
            return np.full(n, sign * float(m2.group(1)))
        coef, body = sign, t
    body = _strip_wrap(body)

    def xi(k):
        return X[:, int(k)]
    # 1[x{i} > t] * x{j}
    mm = re.match(r"^1\[x(\d+)>(%s)\]\*x(\d+)$" % NUM, body)
    if mm: return coef * (xi(mm.group(1)) > float(mm.group(2))).astype(float) * xi(mm.group(3))
    # standalone 1[x{i} > t]
    mm = re.match(r"^1\[x(\d+)>(%s)\]$" % NUM, body)
    if mm: return coef * (xi(mm.group(1)) > float(mm.group(2))).astype(float)
    # max(0, x{i} - t)
    mm = re.match(r"^max\(0,x(\d+)-(%s)\)$" % NUM, body)
    if mm: return coef * np.maximum(0.0, xi(mm.group(1)) - float(mm.group(2)))
    # max(0, t - x{i})
    mm = re.match(r"^max\(0,(%s)-x(\d+)\)$" % NUM, body)
    if mm: return coef * np.maximum(0.0, float(mm.group(1)) - xi(mm.group(2)))
    # (x{i} - t)^2
    mm = re.match(r"^\(?x(\d+)-(%s)\)?\^2$" % NUM, body)
    if mm: return coef * (xi(mm.group(1)) - float(mm.group(2)))**2
    # (x{i})^2 or x{i}^2
    mm = re.match(r"^\(?x(\d+)\)?\^2$", body)
    if mm: return coef * xi(mm.group(1))**2
    # (x{i} - t)   (centered linear)
    mm = re.match(r"^\(?x(\d+)-(%s)\)?$" % NUM, body)
    if mm: return coef * (xi(mm.group(1)) - float(mm.group(2)))
    # (x{i} - a)*(x{j} - b)  centered pairwise interaction
    mm = re.match(r"^\(x(\d+)-(%s)\)\*\(x(\d+)-(%s)\)$" % (NUM, NUM), body)
    if mm: return coef * (xi(mm.group(1)) - float(mm.group(2))) * (xi(mm.group(3)) - float(mm.group(4)))
    # x{i}*x{j}
    mm = re.match(r"^\(?x(\d+)\*x(\d+)\)?$", body)
    if mm: return coef * xi(mm.group(1)) * xi(mm.group(2))
    # 1[x{i} > t] * x{j}
    mm = re.match(r"^\(?1\[x(\d+)>(%s)\]\*x(\d+)\)?$" % NUM, body)
    if mm: return coef * (xi(mm.group(1)) > float(mm.group(2))).astype(float) * xi(mm.group(3))
    # plain x{i}
    mm = re.match(r"^x(\d+)$", body)
    if mm: return coef * xi(mm.group(1))
    raise ValueError("unparsed term: %r" % term)

def eval_equation(eq_rhs, X):
    terms = _split_top_terms(eq_rhs)
    out = np.zeros(X.shape[0])
    for t in terms:
        out = out + _eval_term(t, X)
    return out

def parse_linear_from_coef_table(s, d):
    """Parse 'name: value' coefficient table + intercept -> (coef vec, intercept)."""
    coef = np.zeros(d); inter = 0.0
    for line in s.splitlines():
        m = re.match(r"\s*x(\d+):\s*(%s)\s*$" % NUM, line)
        if m:
            j = int(m.group(1))
            if j < d: coef[j] = float(m.group(2))
        m2 = re.match(r"\s*intercept:\s*(%s)\s*$" % NUM, line)
        if m2: inter = float(m2.group(1))
    return coef, inter

def parse_piecewise_blocks(s):
    """Parse GAM 'f(xk):' piecewise correction tables.
    Returns dict k -> list of (lo, hi, val) with lo/hi possibly +-inf."""
    blocks = {}
    cur = None
    for line in s.splitlines():
        mh = re.match(r"\s*f\(x(\d+)\):\s*$", line)
        if mh:
            cur = int(mh.group(1)); blocks[cur] = []; continue
        if cur is None: continue
        ls = line.strip()
        # x{k} <= T: V
        m = re.match(r"x\d+\s*<=\s*(%s):\s*(%s)$" % (NUM, NUM), ls)
        if m: blocks[cur].append((-np.inf, float(m.group(1)), float(m.group(2)))); continue
        # A < x{k} <= B: V
        m = re.match(r"(%s)\s*<\s*x\d+\s*<=\s*(%s):\s*(%s)$" % (NUM, NUM, NUM), ls)
        if m: blocks[cur].append((float(m.group(1)), float(m.group(2)), float(m.group(3)))); continue
        # x{k} > T: V
        m = re.match(r"x\d+\s*>\s*(%s):\s*(%s)$" % (NUM, NUM), ls)
        if m: blocks[cur].append((float(m.group(1)), np.inf, float(m.group(2)))); continue
        if ls == "" or ls.startswith("Features with zero"):
            cur = None
    return blocks

def eval_piecewise(blocks, X):
    out = np.zeros(X.shape[0])
    for k, rules in blocks.items():
        xk = X[:, k]
        vals = np.zeros(X.shape[0])
        for lo, hi, v in rules:
            mask = (xk > lo) & (xk <= hi)
            vals[mask] = v
        out += vals
    return out

# ---------------------------------------------------------------------------
# Per-model display executor: returns reconstructed prediction from __str__
# ---------------------------------------------------------------------------
def display_predict(name, s, X):
    d = X.shape[1]
    if name in ("HingeEBM", "WinsorizedSparseOLS", "TinyDT"):
        coef, inter = parse_linear_from_coef_table(s, d)
        return X @ coef + inter
    if name in ("HybridGAM", "SmartAdditive", "HingeGAM"):
        coef, inter = parse_linear_from_coef_table(s, d)
        base = X @ coef + inter
        return base + eval_piecewise(parse_piecewise_blocks(s), X)
    if name in ("TeacherStudentRuleSpline", "DualPathSparseSymbolic", "SparseSignedBasisPursuit"):
        # find the 'y = ...' equation line
        for line in s.splitlines():
            ls = line.strip()
            if ls.startswith("y ="):
                return eval_equation(ls[3:].strip(), X)
        raise ValueError("no equation line for %s" % name)
    if name == "DistilledTreeBlendAtlas":
        # parse the printed 'sparse_equation: y = ...'
        for line in s.splitlines():
            ls = line.strip()
            if ls.startswith("sparse_equation:"):
                rhs = ls.split("y =", 1)[1].strip()
                return eval_equation(rhs, X)
        raise ValueError("no sparse_equation for DistilledTreeBlendAtlas")
    raise ValueError(name)

MODELS = {
    "HingeEBM": HingeEBMRegressor, "HybridGAM": HybridGAM,
    "SmartAdditive": SmartAdditiveRegressor, "HingeGAM": HingeGAMRegressor,
    "TeacherStudentRuleSpline": TeacherStudentRuleSplineRegressor,
    "DualPathSparseSymbolic": DualPathSparseSymbolicRegressor,
    "SparseSignedBasisPursuit": SparseSignedBasisPursuitRegressor,
    "DistilledTreeBlendAtlas": DistilledTreeBlendAtlasRegressor,
    "WinsorizedSparseOLS": WinsorizedSparseOLSRegressor,
    "TinyDT": TinyDTDepth2Regressor,
}
# design intent from README (honest vs decoupled)
INTENT = {
    "HingeEBM": "decoupled", "HybridGAM": "decoupled",
    "DistilledTreeBlendAtlas": "decoupled", "DualPathSparseSymbolic": "decoupled",
    "TeacherStudentRuleSpline": "decoupled",
    "SmartAdditive": "honest", "HingeGAM": "honest", "WinsorizedSparseOLS": "honest",
    "SparseSignedBasisPursuit": "honest", "TinyDT": "honest",
}

DATASETS = ["1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA", "1096_FacultySalaries",
            "197_cpu_act", "215_2dplanes", "537_houses", "561_cpu", "engel"]

def run():
    long_rows = []
    for name, cls in MODELS.items():
        for ds in DATASETS:
            try:
                df = fetch_data(ds, local_cache_dir="/tmp/pmlbcache")
            except Exception:
                continue
            y = df["target"].values.astype(float)
            X = df.drop(columns=["target"]).values.astype(np.float64)
            v = ~np.isnan(y); X, y = X[v], y[v]
            if len(X) < 40 or X.shape[1] < 1:
                continue
            if len(X) > 1200:
                rng = np.random.RandomState(42); idx = rng.choice(len(X), 1200, replace=False)
                X, y = X[idx], y[idx]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
            ym, ys = ytr.mean(), ytr.std()
            if ys <= 0: continue
            ytr = (ytr - ym) / ys
            m = cls().fit(Xtr, ytr)
            pred = np.asarray(m.predict(Xte), dtype=float)
            try:
                disp = np.asarray(display_predict(name, str(m), Xte), dtype=float)
            except Exception as e:
                print(f"  PARSE-FAIL {name} on {ds}: {e}")
                continue
            ok = np.isfinite(pred) & np.isfinite(disp)
            if ok.sum() < 5: continue
            pred, disp = pred[ok], disp[ok]
            gap = np.abs(disp - pred)  # already in SD units (ytr standardized)
            long_rows.append(dict(model=name, dataset=ds, intent=INTENT[name],
                n=len(pred), r2=r2_score(pred, disp) if np.var(pred) > 1e-9 else np.nan,
                mae_sd=float(gap.mean()), within_tol=float((gap < TOL).mean())))
    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(OUT, "faithfulness_long.csv"), index=False)
    # pooled summary per model (weight by n)
    def wavg(g, col): return np.average(g[col], weights=g["n"])
    summ = []
    for name, g in long.groupby("model"):
        summ.append(dict(model=name, intent=INTENT[name],
            R2=float(np.average(g["r2"].fillna(0), weights=g["n"])),
            MAE_over_SD=float(wavg(g, "mae_sd")),
            pct_within_tol=float(wavg(g, "within_tol")),
            n_datasets=len(g)))
    s = pd.DataFrame(summ).sort_values(["intent", "MAE_over_SD"])
    s.to_csv(os.path.join(OUT, "faithfulness_summary.csv"), index=False)
    pd.set_option("display.width", 140)
    print("\n==== FAITHFULNESS SUMMARY (display eq executed from __str__ vs .predict()) ====")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    run()
