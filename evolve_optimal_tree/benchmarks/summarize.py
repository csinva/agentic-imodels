"""Merge benchmark CSVs into one table, a markdown report and a plot.

Reference columns (``ref_*``) and Python columns (``py_*``) are taken from the
most recently modified CSV that contains them for each (dataset, lam) pair, so
the Python side can be re-run alone after optimisations.

Usage::

    uv run python benchmarks/summarize.py [--results benchmarks/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_all(results: Path) -> pd.DataFrame:
    files = sorted(results.glob("benchmark*.csv"), key=lambda p: p.stat().st_mtime)
    ref_rows: dict[tuple, pd.Series] = {}
    py_rows: dict[tuple, pd.Series] = {}
    meta: dict[tuple, dict] = {}
    for f in files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            key = (row["dataset"], float(row["lam"]))
            meta[key] = {"dataset": row["dataset"], "n": int(row["n"]), "p": int(row["p"]), "lam": float(row["lam"])}
            if "ref_objective" in row and not pd.isna(row.get("ref_objective", np.nan)):
                ref_rows[key] = row[[c for c in df.columns if c.startswith("ref_")]]
            if "py_objective" in row and not pd.isna(row.get("py_objective", np.nan)):
                py_rows[key] = row[[c for c in df.columns if c.startswith("py_")]]
    out = []
    for key, m in meta.items():
        rec = dict(m)
        if key in ref_rows:
            rec.update(ref_rows[key].to_dict())
        if key in py_rows:
            rec.update(py_rows[key].to_dict())
        out.append(rec)
    df = pd.DataFrame(out)
    order = {d: i for i, d in enumerate(dict.fromkeys(df.sort_values("n")["dataset"]))}
    df["_o"] = df["dataset"].map(order)
    df = df.sort_values(["_o", "lam"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    if "ref_objective" in df and "py_objective" in df:
        df["objective_diff"] = df["py_objective"] - df["ref_objective"]
        df["speedup_py_vs_cpp"] = df["ref_time"] / df["py_time"]
    return df


def fmt(v, digits=6):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["dataset", "n", "lam", "ref_binary_features", "ref_objective", "py_objective", "objective_diff",
            "ref_time", "py_time", "speedup_py_vs_cpp", "ref_size", "py_size", "ref_gap", "py_optimal"]
    cols = [c for c in cols if c in df.columns]
    head = {"ref_binary_features": "binary feats", "ref_objective": "C++ objective", "py_objective": "py objective",
            "objective_diff": "py - C++", "ref_time": "C++ time (s)", "py_time": "py time (s)",
            "speedup_py_vs_cpp": "C++ time / py time", "ref_size": "C++ graph", "py_size": "py graph",
            "ref_gap": "C++ gap", "py_optimal": "py optimal"}
    lines = ["| " + " | ".join(head.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("ref_time", "py_time", "speedup_py_vs_cpp"):
                cells.append(fmt(float(v), 3) if not pd.isna(v) else "")
            elif c in ("ref_size", "py_size", "n", "ref_binary_features"):
                cells.append("" if pd.isna(v) else str(int(v)))
            else:
                cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["ref_time", "py_time"]).copy()
    d = d[(d["ref_time"] > 0) & (d["py_time"] > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    datasets = list(dict.fromkeys(d["dataset"]))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(datasets):
        s = d[d["dataset"] == name]
        ax.scatter(s["ref_time"], s["py_time"], label=name, color=cmap(i % 20), s=36)
    lo = min(d["ref_time"].min(), d["py_time"].min()) * 0.5
    hi = max(d["ref_time"].max(), d["py_time"].max()) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="equal time")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reference C++ optimisation time (s)")
    ax.set_ylabel("pygosdt optimisation time (s)")
    ax.set_title("Optimisation time per (dataset, λ)")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    d2 = d.copy()
    d2["speed"] = d2["ref_time"] / d2["py_time"]
    for i, name in enumerate(datasets):
        s = d2[d2["dataset"] == name].sort_values("lam")
        ax.plot(s["lam"], s["speed"], "o-", label=name, color=cmap(i % 20))
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("regularization λ")
    ax.set_ylabel("C++ time / pygosdt time (>1: pygosdt faster)")
    ax.set_title("Relative speed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args(argv)
    results = Path(args.results)
    df = load_all(results)
    df.to_csv(results / "summary.csv", index=False)
    md = to_markdown(df)
    (results / "summary.md").write_text(md + "\n")
    print(md)
    try:
        plot(df, results / "benchmark.png")
        print(f"plot written to {results / 'benchmark.png'}")
    except Exception as exc:  # matplotlib optional
        print(f"plot skipped: {exc}")
    both = df.dropna(subset=["ref_objective", "py_objective"])
    same = (both["objective_diff"].abs() < 1e-6).sum()
    better = (both["objective_diff"] < -1e-6).sum()
    worse = (both["objective_diff"] > 1e-6).sum()
    print(f"\npairs compared: {len(both)}  identical objective: {same}  pygosdt better: {better}  pygosdt worse: {worse}")
    if "speedup_py_vs_cpp" in both:
        s = both["speedup_py_vs_cpp"].replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s > 0]
        print(f"geometric mean of C++ time / py time: {np.exp(np.log(s).mean()):.2f}  (median {s.median():.2f})")


if __name__ == "__main__":
    main()
