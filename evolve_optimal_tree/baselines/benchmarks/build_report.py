"""Render the benchmark comparison report (HTML with inline SVG charts) from summary.csv.

Usage::

    uv run python benchmarks/build_report.py

Writes ``REPORT.html`` at the package root.  All charts are inline SVG drawn
from the data, so the page has no runtime dependencies; it is a body fragment
(no ``<html>``/``<head>``) so it can be published as an Artifact as-is and also
opens directly in a browser.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"

LAMS = [0.1, 0.05, 0.02, 0.01, 0.005]
STOP_LABEL = {"optimal": "optimal", "time": "time cap", "timeout": "time cap", "memory": "memory cap", "": "no run"}
STOP_CLASS = {"optimal": "ok", "time": "warn", "timeout": "warn", "memory": "crit", "": "none"}


# ----------------------------------------------------------------- helpers
def esc(s) -> str:
    return html.escape(str(s))


def fnum(v, digits=3):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    av = abs(float(v))
    if av == 0:
        return "0"
    if av >= 100:
        return f"{v:,.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.{digits}g}"


def ftime(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    v = float(v)
    if v < 0.001:
        return "<1 ms"
    if v < 1:
        return f"{v * 1000:.0f} ms"
    if v < 60:
        return f"{v:.1f} s"
    return f"{v / 60:.1f} min"


def fobj(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{float(v):.4f}"


def chip(stop: str) -> str:
    return f'<span class="chip chip-{STOP_CLASS.get(stop, "none")}">{esc(STOP_LABEL.get(stop, stop))}</span>'


# ------------------------------------------------------------- baselines
PAIRS = RESULTS / "pair_results.csv"
MODELS = [
    ("gosdt", "reference GOSDT (C++)", "The ICML 2020 code as published, default bounds, one thread."),
    ("pygosdt_v1", "pygosdt_v1", "The pure-Python re-implementation this report is about."),
    ("streed", "STreeD", "Separable-tree dynamic programming (van der Linden et al. 2023), cost-complex-accuracy task "
                        "with cost_complexity = λ, depth cap 20, on the same binarization."),
    ("gosdt_guesses", "gosdt-guesses (exact)", "The newer GOSDT C++ code base (McTavish et al. 2022) in exact mode: same "
                                              "binarization, no threshold or label guesses, no depth budget."),
    ("gosdt_guesses_guided", "gosdt-guesses (guided)", "The same code with the paper's guesses: thresholds from a 40-stump "
                                                       "gradient-boosted ensemble and its predictions as reference labels. "
                                                       "This shrinks the search space, so it is a heuristic, not an exact solver."),
]
EXACT = {"gosdt", "pygosdt_v1", "streed", "gosdt_guesses"}
SUITE_CAP = 30.0
FULL_CAP = 600.0


def load_pairs() -> pd.DataFrame:
    p = pd.read_csv(PAIRS)
    p["status"] = p["status"].fillna("")
    p["verdict"] = p["verdict"].fillna("")
    # the solver's own optimisation time; when a wrapper reports none, the fit wall time
    p["t"] = np.where(pd.to_numeric(p["seconds"], errors="coerce").fillna(0) > 0,
                      pd.to_numeric(p["seconds"], errors="coerce"), pd.to_numeric(p["wall"], errors="coerce"))
    p["t"] = p["t"].fillna(FULL_CAP).clip(upper=FULL_CAP)
    known = pd.read_csv(ROOT.parent / "src" / "known_optima.csv")
    p = p.drop(columns=["known_objective", "known_certified"]).merge(
        known[["dataset", "lam", "objective", "certified"]].rename(
            columns={"objective": "known_objective", "certified": "known_certified"}),
        on=["dataset", "lam"], how="left")

    def verdict(r):
        if pd.isna(r["objective"]):
            return "no_tree"
        if pd.isna(r["known_objective"]):
            return "ok"
        claimed = r["status"] == "optimal" and r["model"] in EXACT
        if r["objective"] > r["known_objective"] + 1e-6:
            return "WRONG" if (claimed and r["known_certified"]) else "worse_than_known"
        if r["objective"] < r["known_objective"] - 1e-6:
            return "WRONG" if (claimed and r["known_certified"]) else "better_than_known"
        return "ok"

    p["verdict"] = p.apply(verdict, axis=1)
    return p


def model_stats(p: pd.DataFrame) -> pd.DataFrame:
    best = p.dropna(subset=["objective"]).groupby(["dataset", "lam"])["objective"].min().rename("best")
    q = p.merge(best, on=["dataset", "lam"], how="left")
    rows = []
    for name, label, _ in MODELS:
        d = q[q["model"] == name]
        if not len(d):
            continue
        has = d["objective"].notna()
        cert = d["status"] == "optimal"
        rows.append({
            "model": name, "label": label, "n_pairs": len(d), "trees": int(has.sum()),
            "certified": int(cert.sum()), "certified_suite": int((cert & (d["t"] <= SUITE_CAP)).sum()),
            "matched_best": int((has & ((d["objective"] - d["best"]).abs() < 1e-6)).sum()),
            "worse": int((has & (d["objective"] > d["best"] + 1e-6)).sum()),
            "wrong": int((d["verdict"] == "WRONG").sum()),
            "mean_t": float(d["t"].mean()), "median_t": float(d["t"].median()),
            "time_cap": int(d["status"].isin(["time", "timeout"]).sum()),
            "memory_cap": int((d["status"] == "memory").sum()),
            "exact": name in EXACT,
        })
    return pd.DataFrame(rows)


def per_dataset_models(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, d in p.groupby("dataset", sort=False):
        rec = {"dataset": ds, "n": int(d["n"].iloc[0])}
        for name, _, _ in MODELS:
            m = d[d["model"] == name]
            rec[f"{name}_cert"] = int((m["status"] == "optimal").sum())
            rec[f"{name}_geo"] = float(np.exp(np.log(m["t"].clip(lower=1e-3)).mean())) if len(m) else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values("n").reset_index(drop=True)


def speed_vs_py(p: pd.DataFrame) -> list:
    """Geometric mean of (model time / pygosdt_v1 time) over pairs both certified."""
    py = p[(p["model"] == "pygosdt_v1") & (p["status"] == "optimal")].set_index(["dataset", "lam"])["t"]
    out = []
    for name, label, _ in MODELS:
        if name == "pygosdt_v1":
            continue
        m = p[(p["model"] == name) & (p["status"] == "optimal")].set_index(["dataset", "lam"])["t"]
        both = m.index.intersection(py.index)
        if len(both) == 0:
            continue
        ratio = (m.loc[both].clip(lower=1e-3) / py.loc[both].clip(lower=1e-3))
        out.append((name, label, float(np.exp(np.log(ratio).mean())), len(both)))
    return out


def disagreements(p: pd.DataFrame) -> pd.DataFrame:
    """Exact solvers' results that are not the best known objective, and the guided heuristic's."""
    d = p[p["objective"].notna() & p["verdict"].isin(["WRONG", "worse_than_known"])]
    return d.sort_values(["dataset", "lam", "model"], ascending=[True, False, True])


# ------------------------------------------------------------------- stats
def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "summary.csv")
    df["ref_stop"] = df["ref_stop"].fillna("")
    df["py_stop"] = df["py_stop"].fillna("")
    return df


def compute(df: pd.DataFrame) -> dict:
    both = df.dropna(subset=["ref_objective", "py_objective"]).copy()
    both["ratio"] = both["ref_time"] / both["py_time"]
    pos = both[(both["ref_time"] > 0) & (both["py_time"] > 0)]
    wins = both[both["objective_diff"] < -1e-6].copy()
    per = []
    for name, d in df.groupby("dataset", sort=False):
        b = d.dropna(subset=["ref_objective", "py_objective"])
        r = (b["ref_time"] / b["py_time"]).replace([0, np.inf], np.nan).dropna()
        per.append({
            "dataset": name, "n": int(d["n"].iloc[0]), "p": int(d["p"].iloc[0]),
            "binary": int(d["py_binary_features"].max()),
            "compared": len(b), "same": int((b["objective_diff"].abs() < 1e-6).sum()),
            "better": int((b["objective_diff"] < -1e-6).sum()), "worse": int((b["objective_diff"] > 1e-6).sum()),
            "ref_optimal": int((d["ref_stop"] == "optimal").sum()), "py_optimal": int((d["py_stop"] == "optimal").sum()),
            "ref_total": float(d["ref_time"].fillna(0).sum()), "py_total": float(d["py_time"].sum()),
            "geo": float(np.exp(np.log(r).mean())) if len(r) else float("nan"),
        })
    per = pd.DataFrame(per)
    return {
        "pairs": len(df), "compared": len(both),
        "same": int((both["objective_diff"].abs() < 1e-6).sum()),
        "better": int((both["objective_diff"] < -1e-6).sum()),
        "worse": int((both["objective_diff"] > 1e-6).sum()),
        "geo": float(np.exp(np.log(pos["ratio"]).mean())), "median": float(pos["ratio"].median()),
        "py_faster": int((both["ratio"] > 1).sum()),
        "ref_faster": int(((both["ratio"] < 1) & (both["ref_time"] > 0)).sum()),
        "ref_zero": int((both["ref_time"] == 0).sum()),
        "ref_time_cap": int(df["ref_stop"].isin(["time", "timeout"]).sum()),
        "ref_memory": int((df["ref_stop"] == "memory").sum()),
        "ref_optimal": int((df["ref_stop"] == "optimal").sum()),
        "py_time_cap": int((df["py_stop"] == "time").sum()),
        "py_memory": int((df["py_stop"] == "memory").sum()),
        "py_optimal": int((df["py_stop"] == "optimal").sum()),
        "ref_trees": int(df["ref_objective"].notna().sum()), "py_trees": int(df["py_objective"].notna().sum()),
        "ref_mean": float(both["ref_time"].mean()), "ref_median": float(both["ref_time"].median()),
        "py_mean": float(both["py_time"].mean()), "py_median": float(both["py_time"].median()),
        "ref_total": float(both["ref_time"].sum()), "py_total": float(both["py_time"].sum()),
        "wins": wins, "per": per, "both": both,
        **node_stats(df),
    }


def node_stats(df: pd.DataFrame) -> dict:
    """Subproblem counts and per-expansion cost on pairs both sides solved to optimality."""
    b = df[(df["ref_stop"] == "optimal") & (df["py_stop"] == "optimal") & (df["ref_time"] > 0) & (df["py_time"] > 0)].copy()
    ratio = b["ref_size"] / b["py_size"]
    slow = b[b["ref_time"] > 1]
    return {
        "both_opt": len(b),
        "node_ratio_med": float(ratio.median()), "node_ratio_max": float(ratio.max()),
        "ref_us": float((slow["ref_time"] / slow["ref_size"] * 1e6).median()),
        "py_us": float((slow["py_time"] / slow["py_size"] * 1e6).median()),
    }


# ------------------------------------------------------------------ charts
def svg_scatter(both: pd.DataFrame) -> str:
    d = both[(both["ref_time"] > 0) & (both["py_time"] > 0)]
    W, H = 640, 420
    L, R, T, B = 64, 20, 20, 52
    lo, hi = 1e-3, 3e3
    lx = lambda v: L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)
    ly = lambda v: T + (H - T - B) - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (H - T - B)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Optimisation time of the reference against pygosdt, log scales">']
    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    tlabel = {1e-3: "1 ms", 1e-2: "10 ms", 1e-1: "0.1 s", 1: "1 s", 10: "10 s", 100: "100 s", 1000: "1000 s"}
    for t in ticks:
        out.append(f'<line class="grid" x1="{lx(t):.1f}" y1="{T}" x2="{lx(t):.1f}" y2="{H - B}"/>')
        out.append(f'<line class="grid" x1="{L}" y1="{ly(t):.1f}" x2="{W - R}" y2="{ly(t):.1f}"/>')
        out.append(f'<text class="tick" x="{lx(t):.1f}" y="{H - B + 16}" text-anchor="middle">{tlabel[t]}</text>')
        out.append(f'<text class="tick" x="{L - 8}" y="{ly(t) + 4:.1f}" text-anchor="end">{tlabel[t]}</text>')
    # shaded region where pygosdt is faster (below diagonal)
    out.append(f'<polygon class="faster-region" points="{lx(lo):.1f},{ly(lo):.1f} {lx(hi):.1f},{ly(hi):.1f} {lx(hi):.1f},{ly(lo):.1f}"/>')
    out.append(f'<line class="diag" x1="{lx(lo):.1f}" y1="{ly(lo):.1f}" x2="{lx(hi):.1f}" y2="{ly(hi):.1f}"/>')
    out.append(f'<text class="annot" x="{lx(2e-3):.1f}" y="{ly(3e-3) - 6:.1f}">equal time</text>')
    out.append(f'<text class="annot" x="{lx(30):.1f}" y="{ly(0.02):.1f}" text-anchor="middle">pygosdt faster</text>')
    for _, r in d.iterrows():
        cap = r["py_stop"] != "optimal" or r["ref_stop"] != "optimal"
        cls = "pt pt-cap" if cap else "pt"
        title = f'{r["dataset"]}, λ={r["lam"]:g}: reference {ftime(r["ref_time"])}, pygosdt {ftime(r["py_time"])}'
        out.append(f'<circle class="{cls}" cx="{lx(r["ref_time"]):.1f}" cy="{ly(r["py_time"]):.1f}" r="5"><title>{esc(title)}</title></circle>')
    labels = [("iris", 0.005, "iris λ=0.005", 8, -8), ("gaussian_1k", 0.02, "gaussian_1k λ=0.02", 8, 4),
              ("compas_processed", 0.1, "compas_processed λ=0.1", -8, 14), ("tic-tac-toe", 0.01, "tic-tac-toe λ=0.01", 8, 4),
              ("fico_binary", 0.005, "fico_binary λ=0.005", 8, 4)]
    for ds, lam, text, dx, dy in labels:
        r = d[(d["dataset"] == ds) & (d["lam"] == lam)]
        if len(r):
            r = r.iloc[0]
            anchor = "end" if dx < 0 else "start"
            out.append(f'<text class="label" x="{lx(r["ref_time"]) + dx:.1f}" y="{ly(r["py_time"]) + dy:.1f}" text-anchor="{anchor}">{esc(text)}</text>')
    out.append(f'<text class="axis" x="{(L + W - R) / 2:.1f}" y="{H - 8}" text-anchor="middle">reference C++ optimisation time</text>')
    out.append(f'<text class="axis" transform="translate(14 {(T + H - B) / 2:.1f}) rotate(-90)" text-anchor="middle">pygosdt optimisation time</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_ratio_bars(per: pd.DataFrame) -> str:
    d = per.dropna(subset=["geo"]).sort_values("geo", ascending=False)
    rowh, top, left, right, bottom = 26, 24, 150, 90, 36
    W = 640
    H = top + rowh * len(d) + bottom
    lo, hi = 0.5, 30000
    lx = lambda v: left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - left - right)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Geometric mean of reference time over pygosdt time per dataset">']
    for t in [1, 10, 100, 1000, 10000]:
        out.append(f'<line class="grid" x1="{lx(t):.1f}" y1="{top - 6}" x2="{lx(t):.1f}" y2="{H - bottom + 4}"/>')
        out.append(f'<text class="tick" x="{lx(t):.1f}" y="{H - bottom + 18}" text-anchor="middle">{t:,}×</text>')
    out.append(f'<line class="diag" x1="{lx(1):.1f}" y1="{top - 6}" x2="{lx(1):.1f}" y2="{H - bottom + 4}"/>')
    for i, (_, r) in enumerate(d.iterrows()):
        y = top + i * rowh
        x0, x1 = lx(1), lx(max(r["geo"], 0.51))
        xa, xb = min(x0, x1), max(x0, x1)
        note = "" if r["compared"] == 5 else f" ({r['compared']} of 5 pairs)"
        out.append(f'<rect class="bar" x="{xa:.1f}" y="{y + 4}" width="{max(xb - xa, 1):.1f}" height="{rowh - 8}" rx="3"><title>{esc(r["dataset"])}: {r["geo"]:.1f}×{esc(note)}</title></rect>')
        out.append(f'<text class="cat" x="{left - 8}" y="{y + rowh / 2 + 4:.1f}" text-anchor="end">{esc(r["dataset"])}</text>')
        out.append(f'<text class="val" x="{xb + 6:.1f}" y="{y + rowh / 2 + 4:.1f}">{r["geo"]:,.0f}×{esc(note)}</text>')
    out.append(f'<text class="axis" x="{(left + W - right) / 2:.1f}" y="{H - 4}" text-anchor="middle">reference time ÷ pygosdt time (log scale, geometric mean over λ)</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_wins(wins: pd.DataFrame) -> str:
    d = wins.sort_values(["dataset", "lam"], ascending=[True, False])
    rowh, top, left, right, bottom = 44, 30, 170, 80, 40
    W = 640
    H = top + rowh * len(d) + bottom
    hi = 0.5
    lx = lambda v: left + v / hi * (W - left - right)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Objective of the reference and pygosdt on the six pairs where they differ">']
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        out.append(f'<line class="grid" x1="{lx(t):.1f}" y1="{top - 8}" x2="{lx(t):.1f}" y2="{H - bottom + 4}"/>')
        out.append(f'<text class="tick" x="{lx(t):.1f}" y="{H - bottom + 18}" text-anchor="middle">{t:g}</text>')
    out.append(f'<rect class="bar-ref" x="{left}" y="6" width="12" height="12" rx="2"/><text class="cat" x="{left + 18}" y="16">reference C++</text>')
    out.append(f'<rect class="bar" x="{left + 130}" y="6" width="12" height="12" rx="2"/><text class="cat" x="{left + 148}" y="16">pygosdt</text>')
    for i, (_, r) in enumerate(d.iterrows()):
        y = top + i * rowh
        out.append(f'<text class="cat" x="{left - 8}" y="{y + 14}" text-anchor="end">{esc(r["dataset"])}</text>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y + 30}" text-anchor="end">λ = {r["lam"]:g}</text>')
        out.append(f'<rect class="bar-ref" x="{left}" y="{y + 4}" width="{lx(r["ref_objective"]) - left:.1f}" height="14" rx="3"><title>reference {r["ref_objective"]:.4f} ({esc(STOP_LABEL[r["ref_stop"]])})</title></rect>')
        out.append(f'<text class="val" x="{lx(r["ref_objective"]) + 6:.1f}" y="{y + 15}">{r["ref_objective"]:.4f}</text>')
        out.append(f'<rect class="bar" x="{left}" y="{y + 22}" width="{lx(r["py_objective"]) - left:.1f}" height="14" rx="3"><title>pygosdt {r["py_objective"]:.4f} ({esc(STOP_LABEL[r["py_stop"]])})</title></rect>')
        out.append(f'<text class="val" x="{lx(r["py_objective"]) + 6:.1f}" y="{y + 33}">{r["py_objective"]:.4f}</text>')
    out.append(f'<text class="axis" x="{(left + W - right) / 2:.1f}" y="{H - 6}" text-anchor="middle">objective = misclassification rate + λ × leaves (lower is better)</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_speed_vs_py(ratios: list) -> str:
    rowh, top, left, right, bottom = 30, 16, 190, 110, 36
    W = 640
    H = top + rowh * len(ratios) + bottom
    lo, hi = 0.3, 300
    lx = lambda v: left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - left - right)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Geometric mean of each baseline\'s time over pygosdt_v1\'s on pairs both certified">']
    for t in [1, 10, 100]:
        out.append(f'<line class="grid" x1="{lx(t):.1f}" y1="{top - 6}" x2="{lx(t):.1f}" y2="{H - bottom + 4}"/>')
        out.append(f'<text class="tick" x="{lx(t):.1f}" y="{H - bottom + 18}" text-anchor="middle">{t}×</text>')
    out.append(f'<line class="diag" x1="{lx(1):.1f}" y1="{top - 6}" x2="{lx(1):.1f}" y2="{H - bottom + 4}"/>')
    for i, (name, label, geo, n) in enumerate(ratios):
        y = top + i * rowh
        x0, x1 = lx(1), lx(min(max(geo, lo), hi))
        xa, xb = min(x0, x1), max(x0, x1)
        cls = "bar-ref" if name == "gosdt" else ("bar-other" if name in EXACT else "bar-heur")
        out.append(f'<rect class="{cls}" x="{xa:.1f}" y="{y + 5}" width="{max(xb - xa, 1):.1f}" height="{rowh - 10}" rx="3"><title>{esc(label)}: {geo:.1f}× pygosdt_v1 time over {n} pairs</title></rect>')
        out.append(f'<text class="cat" x="{left - 8}" y="{y + rowh / 2 + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<text class="val" x="{xb + 6:.1f}" y="{y + rowh / 2 + 4:.1f}">{geo:.1f}× ({n} pairs)</text>')
    out.append(f'<text class="axis" x="{(left + W - right) / 2:.1f}" y="{H - 4}" text-anchor="middle">time ÷ pygosdt_v1 time, geometric mean over pairs both certified (log scale)</text>')
    out.append("</svg>")
    return "\n".join(out)


def baselines_table_html(st: pd.DataFrame) -> str:
    rows = []
    for _, r in st.iterrows():
        cls = ' class="py"' if r["model"] == "pygosdt_v1" else ""
        sw = {"gosdt": "ref", "pygosdt_v1": "py", "streed": "st", "gosdt_guesses": "gg", "gosdt_guesses_guided": "ggh"}[r["model"]]
        wrong = str(r["wrong"]) if r["exact"] else "n/a"
        rows.append(f'<tr{cls}><th scope="row"><span class="swatch {sw}"></span>{esc(r["label"])}</th>'
                    f'<td>{r["trees"]} of {r["n_pairs"]}</td><td>{r["certified"]}</td><td>{r["certified_suite"]}</td>'
                    f'<td>{r["matched_best"]}</td><td>{r["worse"]}</td><td>{wrong}</td>'
                    f'<td>{ftime(r["mean_t"])}</td><td>{ftime(r["median_t"])}</td><td>{r["time_cap"]}</td><td>{r["memory_cap"]}</td></tr>')
    return ('<table class="headline"><thead><tr><th>solver</th><th>pairs with a tree</th><th>certified optimal<br><span class="sub">600 s cap</span></th>'
            '<th>certified within 30 s</th><th>best known objective<br><span class="sub">matched</span></th><th>worse than best known</th>'
            '<th>false certificates</th><th>mean fit time</th><th>median fit time</th><th>stopped at 600 s</th><th>stopped at 6 GB</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def per_dataset_models_html(pdm: pd.DataFrame) -> str:
    head = "".join(f'<th>{esc(label)}<br><span class="sub">certified / geo. mean time</span></th>' for _, label, _ in MODELS)
    rows = []
    for _, r in pdm.iterrows():
        cells = "".join(f'<td>{int(r[f"{n}_cert"])} of 5 · {ftime(r[f"{n}_geo"])}</td>' for n, _, _ in MODELS)
        rows.append(f'<tr><th scope="row">{esc(r["dataset"])}</th><td>{int(r["n"]):,}</td>{cells}</tr>')
    return f'<table class="data"><thead><tr><th>dataset</th><th>rows</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def disagreements_html(d: pd.DataFrame) -> str:
    labels = {n: l for n, l, _ in MODELS}
    rows = []
    for _, r in d.iterrows():
        rows.append(f'<tr><th scope="row">{esc(labels[r["model"]])}</th><td>{esc(r["dataset"])}</td><td>{r["lam"]:g}</td>'
                    f'<td>{fobj(r["objective"])}</td><td>{fobj(r["known_objective"])}{"*" if r["known_certified"] else ""}</td>'
                    f'<td>{chip(r["status"] if r["status"] != "heuristic" else "")}</td><td>{esc(r["verdict"])}</td></tr>')
    return ('<table class="data"><thead><tr><th>solver</th><th>dataset</th><th>λ</th><th>objective</th><th>best known</th>'
            '<th>status</th><th>verdict</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def all_rows_html(p: pd.DataFrame) -> str:
    labels = {n: l for n, l, _ in MODELS}
    rows = []
    for _, r in p.sort_values(["dataset", "lam", "model"], ascending=[True, False, True]).iterrows():
        rows.append(f'<tr><th scope="row">{esc(r["dataset"])}</th><td>{r["lam"]:g}</td><td>{esc(labels.get(r["model"], r["model"]))}</td>'
                    f'<td>{fobj(r["objective"])}</td><td>{"" if pd.isna(r["leaves"]) else int(r["leaves"])}</td>'
                    f'<td>{ftime(r["t"])}</td><td>{chip(r["status"] if r["status"] != "heuristic" else "")}</td><td>{esc(r["verdict"])}</td></tr>')
    return ('<table class="data"><thead><tr><th>dataset</th><th>λ</th><th>solver</th><th>objective</th><th>leaves</th>'
            '<th>time</th><th>status</th><th>verdict</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


# ------------------------------------------------------------------ tables
def fit_matrix_html(df: pd.DataFrame) -> str:
    rows = []
    for name, d in df.groupby("dataset", sort=False):
        cells = []
        for lam in LAMS:
            r = d[d["lam"] == lam]
            if not len(r):
                cells.append("<td></td>")
                continue
            r = r.iloc[0]
            if pd.isna(r["ref_objective"]):
                cells.append(f'<td class="cell cell-none" title="reference produced no tree ({esc(STOP_LABEL[r["ref_stop"]])})">no C++ tree<br><span class="sub">pygosdt {fobj(r["py_objective"])}</span></td>')
            elif r["objective_diff"] < -1e-6:
                cells.append(f'<td class="cell cell-better">pygosdt better<br><span class="sub">{fobj(r["py_objective"])} vs {fobj(r["ref_objective"])}</span></td>')
            elif r["objective_diff"] > 1e-6:
                cells.append(f'<td class="cell cell-worse">pygosdt worse<br><span class="sub">{fobj(r["py_objective"])} vs {fobj(r["ref_objective"])}</span></td>')
            else:
                cells.append(f'<td class="cell cell-same">identical<br><span class="sub">{fobj(r["py_objective"])}</span></td>')
        rows.append(f'<tr><th scope="row">{esc(name)}</th>{"".join(cells)}</tr>')
    head = "".join(f"<th>λ = {l:g}</th>" for l in LAMS)
    return f'<table class="matrix"><thead><tr><th>dataset</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def stop_matrix_html(df: pd.DataFrame) -> str:
    rows = []
    for name, d in df.groupby("dataset", sort=False):
        cells = []
        for lam in LAMS:
            r = d[d["lam"] == lam]
            if not len(r):
                cells.append("<td></td>")
                continue
            r = r.iloc[0]
            cells.append(f'<td class="stops"><div>{chip(r["ref_stop"])} <span class="sub">{ftime(r["ref_time"]) if r["ref_stop"] != "memory" else ftime(r["ref_wall"])}</span></div>'
                         f'<div>{chip(r["py_stop"])} <span class="sub">{ftime(r["py_time"])}</span></div></td>')
        rows.append(f'<tr><th scope="row">{esc(name)}</th>{"".join(cells)}</tr>')
    head = "".join(f"<th>λ = {l:g}</th>" for l in LAMS)
    return (f'<table class="matrix"><thead><tr><th>dataset</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def per_dataset_html(per: pd.DataFrame) -> str:
    rows = []
    for _, r in per.iterrows():
        geo = f"{r['geo']:,.1f}×" if not math.isnan(r["geo"]) else "–"
        rows.append(f"<tr><th scope=\"row\">{esc(r['dataset'])}</th><td>{r['n']:,}</td><td>{r['p']}</td><td>{r['binary']:,}</td>"
                    f"<td>{r['compared']}</td><td>{r['same']}</td><td>{r['better']}</td><td>{r['worse']}</td>"
                    f"<td>{r['ref_optimal']}</td><td>{r['py_optimal']}</td><td>{ftime(r['ref_total'])}</td><td>{ftime(r['py_total'])}</td><td>{geo}</td></tr>")
    return ('<table class="data"><thead><tr><th>dataset</th><th>rows</th><th>source features</th><th>binary features</th>'
            '<th>pairs with both trees</th><th>identical</th><th>pygosdt better</th><th>pygosdt worse</th>'
            '<th>C++ certified optimal</th><th>pygosdt certified optimal</th><th>C++ total time</th><th>pygosdt total time</th>'
            '<th>C++ ÷ pygosdt time</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def full_table_html(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        diff = "" if pd.isna(r["objective_diff"]) else (f"{r['objective_diff']:+.4f}" if abs(r["objective_diff"]) > 1e-6 else "0")
        ref_leaves = "" if pd.isna(r["ref_leaves"]) else int(r["ref_leaves"])
        ref_nodes = "" if pd.isna(r["ref_size"]) or r["ref_size"] < 0 else f"{int(r['ref_size']):,}"
        rows.append(f"<tr><th scope=\"row\">{esc(r['dataset'])}</th><td>{r['lam']:g}</td>"
                    f"<td>{fobj(r['ref_objective'])}</td><td>{fobj(r['py_objective'])}</td><td>{diff}</td>"
                    f"<td>{ref_leaves}</td><td>{int(r['py_leaves'])}</td>"
                    f"<td>{ftime(r['ref_time'])}</td><td>{ftime(r['py_time'])}</td>"
                    f"<td>{ref_nodes}</td><td>{int(r['py_size']):,}</td>"
                    f"<td>{chip(r['ref_stop'])}</td><td>{chip(r['py_stop'])}</td></tr>")
    return ('<table class="data"><thead><tr><th>dataset</th><th>λ</th><th>C++ objective</th><th>pygosdt objective</th>'
            '<th>difference</th><th>C++ leaves</th><th>pygosdt leaves</th><th>C++ time</th><th>pygosdt time</th>'
            '<th>C++ graph nodes</th><th>pygosdt graph nodes</th><th>C++ stop</th><th>pygosdt stop</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


# ------------------------------------------------------------------- HTML
CSS = """
:root {
  color-scheme: light;
  --bg: #f6f6f3; --surface: #ffffff; --ink: #17191d; --ink-2: #4d525b; --ink-3: #7b8089;
  --rule: #dcdcd6; --rule-soft: #ebebe6;
  --ref: #2a78d6; --py: #eb6834; --py-soft: #fbe6dc; --ref-soft: #dbe8fa;
  --st: #1baf7a; --gg: #4a3aa7; --ggh: #a8a4b8;
  --ok: #0ca30c; --warn: #b87a00; --crit: #d03b3b;
  --ok-bg: #e5f5e5; --warn-bg: #fdf1d8; --crit-bg: #fae0e0; --none-bg: #ecece8;
  --better-bg: #e5f5e5; --same-bg: #ffffff; --worse-bg: #fae0e0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #1a1a19; --surface: #232322; --ink: #f2f2ee; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --rule: #3a3a37; --rule-soft: #2e2e2c;
    --ref: #3987e5; --py: #f0784a; --py-soft: #4a2a1c; --ref-soft: #1c3557;
    --st: #199e70; --gg: #9085e9; --ggh: #7d7a8c;
    --ok: #3fbf3f; --warn: #e0a640; --crit: #ef6b6b;
    --ok-bg: #1f3a1f; --warn-bg: #3d3115; --crit-bg: #472222; --none-bg: #2c2c2a;
    --better-bg: #1f3a1f; --same-bg: #232322; --worse-bg: #472222;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1a1a19; --surface: #232322; --ink: #f2f2ee; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --rule: #3a3a37; --rule-soft: #2e2e2c;
  --ref: #3987e5; --py: #f0784a; --py-soft: #4a2a1c; --ref-soft: #1c3557;
  --st: #199e70; --gg: #9085e9; --ggh: #7d7a8c;
  --ok: #3fbf3f; --warn: #e0a640; --crit: #ef6b6b;
  --ok-bg: #1f3a1f; --warn-bg: #3d3115; --crit-bg: #472222; --none-bg: #2c2c2a;
  --better-bg: #1f3a1f; --same-bg: #232322; --worse-bg: #472222;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  font-size: 15.5px; line-height: 1.55; padding-block: 0 64px; padding-inline: 20px; }
.wrap { max-width: 1040px; margin: 0 auto; }
.prose { max-width: 70ch; }
h1, h2, h3 { font-family: "Source Serif 4", Georgia, "Times New Roman", serif; font-weight: 600; line-height: 1.15; text-wrap: balance; margin: 0; }
h1 { font-size: 2.4rem; letter-spacing: -0.01em; }
h2 { font-size: 1.55rem; margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--rule); }
h3 { font-size: 1.15rem; margin-top: 28px; }
p { margin: 12px 0; }
a { color: var(--ref); }
header { padding-block: 48px 8px; }
.eyebrow { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 14px; font-weight: 500; }
.lede { font-size: 1.12rem; color: var(--ink-2); max-width: 64ch; margin-top: 14px; }
.headline-wrap { margin: 28px 0 10px; }
.headline th, .headline td { padding: 12px 14px; }
.headline thead th { font-size: 0.8rem; line-height: 1.25; vertical-align: bottom; }
.headline thead .sub { display: block; font-weight: 400; color: var(--ink-3); font-size: 0.72rem; }
.headline tbody th { font-family: "IBM Plex Sans", sans-serif; font-size: 1rem; font-weight: 600; }
.headline td { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.95rem; white-space: nowrap; }
.headline th, .headline td { padding: 10px 12px; }
.headline tr.py td { color: var(--py); font-weight: 500; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 8px; vertical-align: 0; }
.swatch.ref { background: var(--ref); } .swatch.py { background: var(--py); }
.swatch.st { background: var(--st); } .swatch.gg { background: var(--gg); } .swatch.ggh { background: var(--ggh); }
.chart .bar-other { fill: var(--st); }
.chart .bar-heur { fill: var(--ggh); }
.tablenote { color: var(--ink-2); font-size: 0.9rem; max-width: 80ch; margin-top: 6px; }
figure { margin: 22px 0; }
figcaption { color: var(--ink-2); font-size: 0.88rem; margin-top: 8px; max-width: 70ch; }
.chart { width: 100%; height: auto; max-width: 100%; display: block; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
.chart .grid { stroke: var(--rule-soft); stroke-width: 1; }
.chart .diag { stroke: var(--ink-3); stroke-width: 1.5; stroke-dasharray: 5 4; }
.chart .faster-region { fill: var(--py-soft); opacity: 0.55; }
.chart .tick { font: 11px "IBM Plex Mono", ui-monospace, monospace; fill: var(--ink-3); }
.chart .axis { font: 12px "IBM Plex Sans", sans-serif; fill: var(--ink-2); }
.chart .annot { font: italic 12px "IBM Plex Sans", sans-serif; fill: var(--ink-3); }
.chart .label { font: 11.5px "IBM Plex Sans", sans-serif; fill: var(--ink-2); }
.chart .cat { font: 12.5px "IBM Plex Sans", sans-serif; fill: var(--ink); }
.chart .val { font: 11.5px "IBM Plex Mono", ui-monospace, monospace; fill: var(--ink-2); }
.chart .pt { fill: var(--py); stroke: var(--surface); stroke-width: 2; }
.chart .pt-cap { fill: var(--surface); stroke: var(--py); stroke-width: 2; }
.chart .bar { fill: var(--py); }
.chart .bar-ref { fill: var(--ref); }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.86rem; color: var(--ink-2); margin: 8px 0 0; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; background: var(--py); }
.legend .hollow::before { background: var(--surface); border: 2px solid var(--py); width: 8px; height: 8px; }
.scroll { overflow-x: auto; margin: 18px 0; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule-soft); }
thead th { font-weight: 600; color: var(--ink-2); font-size: 0.8rem; letter-spacing: 0.02em; background: var(--surface); position: sticky; top: 0; }
tbody th { font-weight: 500; white-space: nowrap; }
td { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.82rem; }
.matrix td.cell { font-family: "IBM Plex Sans", sans-serif; font-size: 0.84rem; min-width: 128px; }
.matrix .cell-better { background: var(--better-bg); color: var(--ok); font-weight: 600; }
.matrix .cell-worse { background: var(--worse-bg); color: var(--crit); font-weight: 600; }
.matrix .cell-same { color: var(--ink-2); }
.matrix .cell-none { background: var(--none-bg); color: var(--ink-3); }
.matrix .sub, .stops .sub { font-family: "IBM Plex Mono", ui-monospace, monospace; font-weight: 400; color: var(--ink-3); font-size: 0.78rem; }
.matrix .cell-better .sub { color: var(--ink-2); }
.stops div { white-space: nowrap; margin: 2px 0; }
.chip { display: inline-block; font-family: "IBM Plex Sans", sans-serif; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em; padding: 1px 8px; border-radius: 999px; min-width: 84px; text-align: center; }
.chip-ok { background: var(--ok-bg); color: var(--ok); }
.chip-warn { background: var(--warn-bg); color: var(--warn); }
.chip-crit { background: var(--crit-bg); color: var(--crit); }
.chip-none { background: var(--none-bg); color: var(--ink-3); }
.key { display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.84rem; color: var(--ink-2); margin-top: 8px; align-items: center; }
details { margin-top: 18px; }
summary { cursor: pointer; font-weight: 600; color: var(--ink-2); }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--py); outline-offset: 2px; }
ul { padding-left: 22px; } li { margin: 6px 0; }
code { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.86em; background: var(--surface); border: 1px solid var(--rule-soft); border-radius: 3px; padding: 0 4px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 720px) { .two { grid-template-columns: 1fr; } h1 { font-size: 1.9rem; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def build_html(df: pd.DataFrame, s: dict) -> str:
    bs = s["baselines"]
    wins = s["wins"].sort_values(["dataset", "lam"], ascending=[True, False])
    win_rows = "".join(
        f"<tr><th scope=\"row\">{esc(r['dataset'])}</th><td>{r['lam']:g}</td>"
        f"<td>{fobj(r['ref_objective'])}</td><td>{int(r['ref_errors'])} / {int(r['ref_leaves'])}</td><td>{chip(r['ref_stop'])} <span class=\"sub\">{ftime(r['ref_time'])}</span></td>"
        f"<td>{fobj(r['py_objective'])}</td><td>{int(r['py_errors'])} / {int(r['py_leaves'])}</td><td>{chip(r['py_stop'])} <span class=\"sub\">{ftime(r['py_time'])}</span></td></tr>"
        for _, r in wins.iterrows())
    datasets = df.groupby("dataset", sort=False).first()
    ds_rows = "".join(f"<tr><th scope=\"row\">{esc(n)}</th><td>{int(r['n']):,}</td><td>{int(r['p'])}</td><td>{int(r['py_binary_features']):,}</td></tr>"
                      for n, r in datasets.iterrows())
    return f"""<title>pygosdt Benchmark Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">Benchmark report · optimal sparse decision trees</div>
  <h1>pygosdt against the reference GOSDT</h1>
  <p class="lede">A pure-Python re-implementation of Generalized Optimal Sparse Decision Trees, compared with the
  reference C++ binary, STreeD and gosdt-guesses on 15 datasets and 5 regularization strengths: does it find the
  same trees, and how fast?</p>
  <div class="scroll headline-wrap">{baselines_table_html(bs)}</div>
  <p class="tablenote">All five solvers were run on the same 75 (dataset, λ) pairs, the same binarized features and the same 600 s / 6 GB caps,
  and every returned tree's objective was recomputed independently. "Best known objective" is the best value any solver returned on that pair;
  a false certificate is an exact solver certifying a value that disagrees with a certified optimum. Fit time is the solver's own optimisation
  time, with pairs that produced no tree counted at 600 s; the guided gosdt-guesses wrapper reports no solver time, so its child-process wall
  time (about 1 s of start-up) is shown. Head to head on the {s['compared']} pairs where both returned a tree, pygosdt_v1 is
  {s['geo']:.1f}× faster than the reference on the geometric mean (median {s['median']:.1f}×) and never worse in objective.</p>
</header>

<section class="prose">
<h2>What was compared</h2>
<p>Both implementations minimise the same objective, <em>misclassification rate + λ × number of leaves</em>, over all
decision trees on the binarized features. Each (dataset, λ) pair was run once per implementation on the same CSV,
single-threaded, sequentially on an otherwise idle Apple M5 (16 GB), with a 600 s time cap and a 6 GB memory cap for
both. The objective of every returned tree was recomputed independently from the tree and the raw data. Times are
optimisation only; CSV parsing and binarization are excluded for both. The reference is the ICML 2020 code
(<code>gosdt</code>) with its default bounds and <code>worker_limit = 1</code>.</p>
</section>
<div class="scroll"><table class="data"><thead><tr><th>dataset</th><th>rows</th><th>source features</th><th>binary features</th></tr></thead><tbody>{ds_rows}</tbody></table></div>
<p class="prose">λ ∈ {{0.1, 0.05, 0.02, 0.01, 0.005}} for every dataset. Missing values were filled with 0 for both implementations.</p>

<section class="prose">
<h2>Five solvers on the same problem</h2>
<p>Besides the reference and pygosdt_v1, three further baselines were run through the same scorer:</p>
<ul>
{"".join(f"<li><strong>{esc(label)}</strong> — {esc(text)}</li>" for _, label, text in MODELS)}
</ul>
<p>Four of the five are exact solvers of the same objective, so they must agree whenever both certify. They do: across the
{s['exact_certified_pairs']} pairs certified by at least two exact solvers, every certified objective agrees, with the single
exception of the reference's tic-tac-toe certificate discussed below. The guided variant of gosdt-guesses is not exact by
construction; it returned a worse tree than the best known on {s['guided_worse']} of its 75 pairs.</p>
</section>
<figure>{svg_speed_vs_py(s['speed_ratios'])}<figcaption>How much longer each solver takes than pygosdt_v1, as the geometric mean of the time ratio
over the pairs both certified. The guided heuristic's time is dominated by process start-up and is not comparable.</figcaption></figure>
<div class="scroll">{per_dataset_models_html(s['per_dataset_models'])}</div>
<p class="prose">Certified pairs (of 5 λ values) and geometric-mean fit time per dataset; time counts uncertified pairs at the 600 s cap.</p>

<h3>Results that are not the best known objective</h3>
<p class="prose">Every exact solver's returned tree that is worse than the best known objective, and the guided heuristic's. A time-capped
incumbent that is worse is expected (verdict <code>worse_than_known</code>); a <em>certified</em> result that is worse is an
exactness violation (<code>WRONG</code>). Best known values marked * are certified.</p>
<div class="scroll">{disagreements_html(s['disagreements'])}</div>

<section class="prose">
<h2>pygosdt_v1 against the reference: does it find the same trees?</h2>
<p>Yes, or better. In none of the {s['compared']} pairs where both implementations produced a tree was pygosdt's objective worse.
{s['same']} objectives are identical; in {s['better']} pairs pygosdt's tree is strictly better. Every cell below is one
(dataset, λ) pair. Grey cells are pairs where the reference was stopped by the memory cap before it could write any model.</p>
</section>
<div class="scroll">{fit_matrix_html(df)}</div>

<h3>The six pairs where the trees differ</h3>
<p class="prose">Two of these are not timeouts: on <strong>tic-tac-toe at λ = 0.02</strong> the reference reports a certified optimum of
0.3246, yet a 6-leaf tree with 190 errors (objective 0.3183) exists and pygosdt finds it. The reference's pairwise
"feature exchange" pruning discards features for whole subtrees using bounds computed at the parent, which is not exact, and it
sorts integer thresholds as strings. The other four are cases where the reference hit its time cap and returned an incumbent;
pygosdt certified the optimum in under five minutes, and on sine_1k both timed out with pygosdt's incumbent ahead.</p>
<div class="scroll"><table class="data"><thead><tr><th>dataset</th><th>λ</th><th>C++ objective</th><th>C++ errors / leaves</th><th>C++ status</th>
<th>pygosdt objective</th><th>pygosdt errors / leaves</th><th>pygosdt status</th></tr></thead><tbody>{win_rows}</tbody></table></div>
<figure>{svg_wins(s['wins'])}<figcaption>Objective of the two implementations on the six pairs that differ. Lower is better; both bars start at zero.</figcaption></figure>

<section class="prose">
<h2>pygosdt_v1 against the reference: speed</h2>
<p>Each point is one pair where both implementations returned a tree, placed by the reference's time (across) and pygosdt's
time (up), on log scales. Points below the dashed line are pairs where pygosdt was faster. Hollow points are pairs where at
least one side stopped at a cap, so their time is the cap rather than a solve time. pygosdt was faster on {s['py_faster']} pairs;
the {s['ref_faster'] + s['ref_zero']} pairs where the reference was faster all take the reference 0–2 ms (its timer has millisecond
resolution and reports 0 on {s['ref_zero']} of them, which the log plot cannot show).</p>
</section>
<figure>{svg_scatter(s['both'])}
<div class="legend"><span>solved to optimality by both</span><span class="hollow">at least one side stopped at a cap</span></div>
<figcaption>Optimisation time per pair. Geometric mean of reference ÷ pygosdt time: {s['geo']:.1f}×; median {s['median']:.1f}×.</figcaption></figure>

<section class="prose">
<p>The gap widens as λ shrinks and the search gets harder: pygosdt's depth-first search with a strong incumbent visits far fewer
subproblems than the reference's best-first message passing (iris at λ = 0.01: 3,812 vs 215,881 graph nodes, 0.14 s vs 151 s),
which more than pays for the interpreter overhead per node.</p>
</section>
<figure>{svg_ratio_bars(s['per'])}<figcaption>Per-dataset geometric mean of reference time ÷ pygosdt time over the λ values where both returned a tree. Bars left of 1× would mean the reference was faster; none reach it. compas_processed has a single comparable pair (295 s vs 14 ms).</figcaption></figure>
<div class="scroll">{per_dataset_html(s['per'])}</div>

<section class="prose">
<h2>Why pygosdt is faster</h2>
<p>Both implementations do the same kind of work per subproblem: for every binary feature, count how the rows reaching the
node split by class. What differs is how many subproblems they expand and what each expansion costs.</p>
<ul>
<li><strong>Far fewer subproblems.</strong> pygosdt runs a depth-first search that starts from a greedy tree and immediately
tightens its incumbent with a "both children as leaves" bound, so at every node it only descends into splits whose lower bound
beats the best tree found so far. The reference is best-first: a priority queue of messages ordered by support and lower bound
expands many subproblems breadth-wise before any complete tree exists to prune against. Over the {s['both_opt']} pairs both
sides solved to optimality, the reference expanded a median {s['node_ratio_med']:.1f}× more subproblems (up to
{s['node_ratio_max']:.0f}× on iris, where it visits over 200,000 nodes for a 4-leaf tree). Missing bounds are not the reason:
the two use the same bounds, and pygosdt's extra pruning of neighbouring numeric thresholds is applied per node only.</li>
<li><strong>Cheaper expansions.</strong> pygosdt gathers a node's per-feature counts in one vectorised call (a numba kernel over
packed 64-bit words), then filters and orders candidate splits with numpy; the interpreter only touches the few splits that
survive. The reference copies full row bitmasks into every message and child task, and routes each expansion through several
concurrent hash maps (vertices, children, edges, bounds, queue membership). On pairs taking the reference more than a second,
the median cost per expanded subproblem was {s['ref_us']:.0f} µs for the reference and {s['py_us']:.0f} µs for pygosdt.</li>
<li><strong>Lower memory.</strong> A pygosdt subproblem is one big integer key plus a small record; the reference's per-vertex
bitmask copies and per-edge tables exhausted 6 GB on sine_10k within a minute at every λ, while pygosdt stayed under the same
cap for the full 600 s with 300,000–500,000 memoised subproblems.</li>
</ul>
</section>

<section class="prose">
<h2>What goes wrong in the reference implementation</h2>
<ul>
<li><strong>It certifies suboptimal trees.</strong> When the reference computes a vertex's lower bound (the minimum over its
splits, in <code>store_children</code> and <code>load_children</code>) it skips every split whose bound exceeds the vertex's
current <em>scope</em>, the budget handed down by the parent. That lower bound is only valid for that scope, but it is cached and
never lowered: when a parent later revisits the vertex with a wider budget, the stale bound stays and the search prunes the
subtree that held the optimum. On tic-tac-toe at λ = 0.02 the reference reports 0.3246 with a zero optimality gap while a tree
with 0.3183 exists; it does so with every optional bound, look-ahead and cancellation disabled. Removing the two scope-conditional
skips (patch <code>baselines/gosdt_patches/scope-lowerbound.patch</code>, two lines) makes it report 0.3183, matching pygosdt. pygosdt
avoids the problem by construction: when a subproblem fails its budget it records a lower bound that is valid unconditionally
(the minimum over all pruned and solved splits), so revisiting it with a larger budget is always safe.</li>
<li><strong>Its numeric encoder sorts integer thresholds as strings</strong> ("10" &lt; "2"), so the threshold-adjacency the
continuous-feature-exchange bound relies on does not hold for integer columns with values above 9.</li>
<li><strong>Its pairwise feature-exchange bound prunes features for entire subtrees</strong> using dominance established at
the parent, which does not carry over to descendants; pygosdt applies the provably valid per-node version only.</li>
<li><strong>It overruns its own time limit</strong> because the clock is checked every 10,000 iterations: with a 600 s cap it ran
for up to 1,321 s, and its memory use is high enough that it was killed at 6 GB on 12 of 75 pairs.</li>
<li><strong>It does not build as published on current toolchains</strong> (x86-only compiler flags and SIMD headers, an
allocator type oneTBB 2021+ rejects); two lines and a direct clang build fix that, see <code>baselines/gosdt_patches/</code>.</li>
</ul>
</section>

<section class="prose">
<h2>Where each implementation ran out of time or memory</h2>
<p>Each cell shows how the reference (top) and pygosdt (bottom) stopped, with the time used. The reference certified optimality on
{s['ref_optimal']} pairs, hit the time cap on {s['ref_time_cap']} (it checks the clock only every 10,000 iterations and overran the cap by up to
500 s) and was stopped by the memory cap on {s['ref_memory']}, including all five sine_10k cases within about a minute. pygosdt certified
optimality on {s['py_optimal']} pairs, hit the time cap on {s['py_time_cap']} and the memory cap on {s['py_memory']} (compas_processed, where each memoised subproblem
key is a 12,381-bit integer). When it stops early pygosdt returns its incumbent tree together with a certified lower bound.</p>
</section>
<div class="scroll">{stop_matrix_html(df)}</div>
<div class="key">{chip('optimal')} certified optimum &nbsp; {chip('time')} stopped at 600 s, incumbent returned &nbsp; {chip('memory')} stopped at 6 GB</div>

<section class="prose">
<h2>Method notes</h2>
<ul>
<li>pygosdt implements the reference's bounds (equivalent points, leaf support, incremental accuracy, one-step look-ahead,
similar support, threshold exchange) in a memoised depth-first branch-and-bound with Python big-integer bitsets and an optional
numba kernel for the per-node counts. Both engines give identical trees; numba is 1.1–2× faster on wide numeric data.</li>
<li>Exactness is tested against an exhaustive dynamic program on 50 random problems (2 and 3 classes, random cost matrices)
and pinned on 14 real (dataset, λ) pairs; 80 tests in total.</li>
<li>The reference's non-exact pairwise feature-exchange bound is deliberately not replicated.</li>
<li>Missing numeric values never satisfy a split in pygosdt; the reference parses them as 0. The benchmark fills them with 0 for both.</li>
<li>Reproduce with <code>uv run baselines/benchmarks/run_benchmark.py</code>, then <code>summarize.py</code> and <code>build_report.py</code> in <code>baselines/benchmarks/</code>.</li>
</ul>
</section>

<h2>All {s['pairs']} pairs</h2>
<details open><summary>pygosdt_v1 against the reference, per pair</summary>
<div class="scroll">{full_table_html(df)}</div>
</details>
<details><summary>Every solver, every pair ({s['all_rows']} rows)</summary>
<div class="scroll">{all_rows_html(s['pairs_df'])}</div>
</details>
</div>
"""


def main():
    df = load()
    s = compute(df)
    p = load_pairs()
    bs = model_stats(p)
    s["baselines"] = bs
    s["per_dataset_models"] = per_dataset_models(p)
    s["speed_ratios"] = speed_vs_py(p)
    s["disagreements"] = disagreements(p)
    s["pairs_df"] = p
    s["all_rows"] = len(p)
    s["guided_worse"] = int(bs.loc[bs["model"] == "gosdt_guesses_guided", "worse"].sum())
    cert = p[(p["status"] == "optimal") & p["model"].isin(EXACT)]
    s["exact_certified_pairs"] = int((cert.groupby(["dataset", "lam"]).size() >= 2).sum())
    (ROOT / "REPORT.html").write_text(build_html(df, s))
    print(f"wrote {ROOT / 'REPORT.html'}")
    print(f"pairs {s['pairs']} compared {s['compared']} same {s['same']} better {s['better']} worse {s['worse']} geo {s['geo']:.2f} median {s['median']:.2f} "
          f"py_faster {s['py_faster']} ref_faster {s['ref_faster']} ref_zero {s['ref_zero']}")


if __name__ == "__main__":
    main()
