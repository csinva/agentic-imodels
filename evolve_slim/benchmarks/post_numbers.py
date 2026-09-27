"""Every number quoted in the text of the FastRiskScore post, computed from the result files.

    uv run benchmarks/post_numbers.py

The shipped solver is run 2's v35_scratch; FasterRisk is the default baseline. Per-problem
comparisons use the first run of each (the repeats only enter the figure's means).
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def rows(path, model):
    df = pd.read_csv(os.path.join(ROOT, path))
    return df[df["model"] == model].set_index(["dataset", "k"]).sort_index()


def geo(s):
    return float(np.exp(np.mean(np.log(np.maximum(s, 1e-3)))))


def compare(ship, fr, name):
    d = ship["loss"] - fr["loss"]
    sp = fr["seconds"] / ship["seconds"]
    print(f"[{name}] problems {len(d)}: lower {int((d < -1e-6).sum())}, equal {int((d.abs() <= 1e-6).sum())}, "
          f"higher {int((d > 1e-6).sum())} (largest +{d.max():.4f}, largest drop {d.min():.4f})")
    print(f"[{name}] geo time ship {geo(ship['seconds']):.3f} s, FasterRisk {geo(fr['seconds']):.3f} s; "
          f"speed-up geo {geo(fr['seconds']) / geo(ship['seconds']):.0f}x, median {sp.median():.0f}x, "
          f"range {sp.min():.1f}x to {sp.max():.0f}x; slowest fit {ship['seconds'].max():.2f} s vs "
          f"{fr['seconds'].max():.1f} s")
    print(f"[{name}] mean loss ship {ship['loss'].mean():.5f}, FasterRisk {fr['loss'].mean():.5f} "
          f"(drop {fr['loss'].mean() - ship['loss'].mean():.5f}); test AUC ship {ship['auc_test'].mean():.4f}, "
          f"FasterRisk {fr['auc_test'].mean():.4f}")


vis_ship = rows("runs/sep26-run2/results/problem_results.csv", "v35_scratch")
vis_fr = rows("results/problem_results.csv", "fasterrisk")
compare(vis_ship, vis_fr, "visible")
wide = rows("results/t1200_problem_results.csv", "fasterrisk_wide")
print(f"[visible] wide FasterRisk geo time {geo(wide['seconds']):.1f} s ({geo(wide['seconds']) / geo(vis_fr['seconds']):.0f}x "
      f"default), mean loss {wide['loss'].mean():.5f}; ship lower than wide on "
      f"{int((vis_ship['loss'] < wide['loss'] - 1e-6).sum())}, higher on {int((vis_ship['loss'] > wide['loss'] + 1e-6).sum())}")
hid_ship = rows("results/hidden_problem_results.csv", "v35_scratch_run2")
hid_fr = rows("results/hidden_problem_results.csv", "fasterrisk")
compare(hid_ship, hid_fr, "hidden")
cont = rows("results/hidden_problem_results.csv", "continuous_beam")
print(f"[hidden] real-valued reference mean loss {cont['loss'].mean():.5f}: the drop closes "
      f"{(hid_fr['loss'].mean() - hid_ship['loss'].mean()) / (hid_fr['loss'].mean() - cont['loss'].mean()):.0%} of the gap")
for suite, path in (("visible", "results/problem_results.csv"), ("hidden", "results/hidden_problem_results.csv")):
    rs = rows(path, "riskslim")
    print(f"[{suite}] RiskSLIM no model on {int((rs['status'] == 'no_model').sum())} of {len(rs)}")
full_path = "results/hidden_full_t600_problem_results.csv"
if os.path.exists(os.path.join(ROOT, full_path)):
    fs, ff = rows(full_path, "v35_scratch"), rows(full_path, "fasterrisk")
    compare(fs, ff, "full")
    for m in ("riskslim", "slim_milp", "rounded_lr", "imodels_slim", "continuous_beam"):
        r = rows(full_path, m)
        if len(r):
            print(f"[full] {m}: geo {geo(r['seconds']):.1f} s, mean loss {r['loss'].mean():.5f}, AUC "
                  f"{r['auc_test'].mean():.4f}, statuses {r['status'].value_counts().to_dict()}")
    print(f"[full] ship statuses {fs['status'].value_counts().to_dict()}; fasterrisk {ff['status'].value_counts().to_dict()}")
