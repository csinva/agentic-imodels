"""Every solver on the three panels: mean training loss, mean test AUC, geometric-mean fit time,
problems without a score, and problems where its loss is below FasterRisk's.

    uv run benchmarks/baseline_table.py            # printed table
    uv run benchmarks/baseline_table.py --markdown # the README table
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = {  # panel -> files with its per-problem rows, and the shipped solver's model name there
    "visible": (["results/problem_results.csv", "results/t1200_problem_results.csv",
                 "runs/sep26-run2/results/problem_results.csv"], "v35_scratch"),
    "hidden": (["results/hidden_problem_results.csv", "results/hidden_t1200_problem_results.csv"], "v35_scratch_run2"),
    "full": (["results/hidden_full_t600_problem_results.csv"], "v35_scratch"),
}
MODELS = [("SHIP", "FastRiskScore (v35_scratch)"), ("fasterrisk", "FasterRisk"),
          ("fasterrisk_wide", "FasterRisk, wide search"), ("riskslim", "RiskSLIM (CPLEX CE)"),
          ("cpa_highs", "cutting planes, HiGHS"), ("slim_milp", "SLIM (HiGHS)"),
          ("abess_seqround", "abess + seq. rounding"), ("fastsparse_seqround", "fastSparse + seq. rounding"),
          ("okridge_seqround", "OKRidge + seq. rounding"), ("l1path_seqround", "L1 path + seq. rounding"),
          ("psl", "probabilistic scoring list"), ("autoscore", "AutoScore"), ("rounded_lr", "rounded L1 logistic"),
          ("unit_weighting", "unit weighting"), ("imodels_slim", "imodels SLIMClassifier"),
          ("continuous_beam", "real-valued k-sparse (not integer)")]


def load(files):
    df = pd.concat([pd.read_csv(os.path.join(ROOT, f)) for f in files if os.path.exists(os.path.join(ROOT, f))])
    return df.drop_duplicates(["model", "dataset", "k"])


def stats():
    out = {}
    for panel, (files, ship) in PANELS.items():
        df = load(files)
        fr = df[df.model == "fasterrisk"].set_index(["dataset", "k"])["loss"]
        for key, _ in MODELS:
            g = df[df.model == (ship if key == "SHIP" else key)]
            if not len(g):
                continue
            gl = g.set_index(["dataset", "k"])["loss"]
            out[(key, panel)] = dict(
                loss=g["loss"].mean(), auc=g["auc_test"].mean(),
                time=float(np.exp(np.log(np.maximum(g["seconds"], 1e-3)).mean())),
                nomodel=int(g["status"].isin(["no_model", "killed"]).sum()),
                below_fr=int((gl < fr.reindex(gl.index) - 1e-6).sum()), n=len(g))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    s = stats()
    if args := ap.parse_args():
        pass
    if args.markdown:
        print("| solver | " + " | ".join(f"{p}: loss | AUC | time" for p in PANELS) + " |")
        print("|---|" + "---|---|---|" * len(PANELS))
        for key, label in MODELS:
            cells = []
            for p in PANELS:
                v = s.get((key, p))
                t = "" if v is None else (f"{v['time']:.3f} s" if v["time"] < 1 else f"{v['time']:.3g} s")
                cells += ["", "", ""] if v is None else [f"{v['loss']:.4f}", f"{v['auc']:.3f}", t]
            print(f"| {label} | " + " | ".join(cells) + " |")
    else:
        print(f"{'solver':36s}" + "".join(f"{p:>44s}" for p in PANELS))
        for key, label in MODELS:
            row = f"{label:36s}"
            for p in PANELS:
                v = s.get((key, p))
                row += " " * 44 if v is None else (f"  {v['loss']:.4f} {v['auc']:.3f} {v['time']:8.3f}s "
                                                   f"none={v['nomodel']:<3d} <FR={v['below_fr']:>3d}/{v['n']:<3d}")
            print(row)
