"""Backfill ``mean_objective`` and ``mean_regret`` (see src/evaluate.py) into an
overall_results.csv from the per-pair rows of the same results folder, falling back to the
top-level results/pair_results.csv (baselines) and to the full benchmark for rows whose
pairs are missing.  Usage: uv run backfill_criterion.py RESULTS_DIR [RESULTS_DIR ...]"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import pandas as pd
from evaluate import OVERALL_CSV_COLS, criterion_summary, rows_from_benchmark
from suite import load_known_optima

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PAIRS = os.path.join(HERE, "results", "pair_results.csv")
known = load_known_optima()


def pair_rows(model, results_dir):
    for path in (os.path.join(results_dir, "pair_results.csv"), MAIN_PAIRS):
        if os.path.exists(path):
            t = pd.read_csv(path)
            t = t[t["model"] == model]
            if len(t):
                return [{"dataset": r.dataset, "lam": r.lam, "objective": r.objective,
                         "known_objective": known.get((r.dataset, float(r.lam)), (float("nan"), False))[0]}
                        for r in t.itertuples()]
    return rows_from_benchmark(model)


for results_dir in sys.argv[1:]:
    path = os.path.join(results_dir, "overall_results.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        pr = pair_rows(r["model_name"], results_dir)
        if pr is None:
            print(f"  {r['model_name']}: no per-pair rows found, left blank"); r["mean_objective"] = r["mean_regret"] = ""
            continue
        mo, mr = criterion_summary(pr)
        r["mean_objective"], r["mean_regret"] = f"{mo:.4f}", f"{mr:.4f}"
        print(f"  {r['model_name']:40s} pairs {len(pr):3d} mean_objective {mo:.4f} mean_regret {mr:.4f}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{path}: {len(rows)} rows rewritten with the criterion columns")
