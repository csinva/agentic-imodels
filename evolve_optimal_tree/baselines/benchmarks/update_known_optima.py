"""Refresh ``src/known_optima.csv`` from ``results/pair_results.csv``.

For every (dataset, λ) the best objective any baseline returned becomes the
known optimum.  It is marked *certified* only when a trusted solver
(``pygosdt_v1`` or ``streed``) certified that objective: the reference
``gosdt`` has been observed to issue false certificates, so its certificates
are never trusted, and other solvers are treated the same way until verified.

Usage (from ``evolve_optimal_tree``)::

    uv run baselines/benchmarks/update_known_optima.py
"""

from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAIRS = os.path.join(HERE, "results", "pair_results.csv")
KNOWN = os.path.join(ROOT, "src", "known_optima.csv")
TRUSTED = {"pygosdt_v1", "streed"}
TOL = 1e-6

if __name__ == "__main__":
    pairs = pd.read_csv(PAIRS).dropna(subset=["objective"])
    old = pd.read_csv(KNOWN) if os.path.exists(KNOWN) else pd.DataFrame(columns=["dataset", "lam"])
    order = list(dict.fromkeys(zip(old["dataset"], old["lam"]))) if len(old) else []
    rows = []
    for (name, lam), d in pairs.groupby(["dataset", "lam"], sort=False):
        best = float(d["objective"].min())
        at_best = d[d["objective"] <= best + TOL]
        certified = bool(((at_best["status"] == "optimal") & at_best["model"].isin(TRUSTED)).any())
        source = at_best[at_best["status"] == "optimal"]["model"].iloc[0] if certified else at_best["model"].iloc[0]
        rows.append({"dataset": name, "lam": float(lam), "objective": round(best, 6),
                     "certified": certified, "source": source})
    new = pd.DataFrame(rows)
    key = {(ds, float(lam)): i for i, (ds, lam) in enumerate(order)}
    new["_o"] = [key.get((r.dataset, float(r.lam)), len(key)) for r in new.itertuples()]
    new = new.sort_values(["_o", "dataset", "lam"], ascending=[True, True, False]).drop(columns="_o")
    if len(old):
        merged = new.merge(old, on=["dataset", "lam"], how="left", suffixes=("", "_old"))
        changed = merged[((merged["objective"] - merged["objective_old"]).abs() > TOL)
                         | (merged["certified"] != merged["certified_old"])]
        for r in changed.itertuples():
            print(f"{r.dataset:17s} λ={r.lam:<6} {r.objective_old:.6f}{'*' if r.certified_old else ''} -> "
                  f"{r.objective:.6f}{'*' if r.certified else ''} ({r.source})")
        print(f"{len(changed)} pairs changed")
    new.to_csv(KNOWN, index=False)
    print(f"{len(new)} pairs, {int(new['certified'].sum())} certified -> {KNOWN}")
