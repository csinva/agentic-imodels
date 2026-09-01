"""Sibling-free small-suite harness: uv run perf_iter5.py <module> <class> <name> [k=v ...]"""
import importlib, os, sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0, _os.path.join(_HERE, "..", "model"))
sys.path.insert(0, _os.path.join(_ROOT, "evolve", "src"))
from performance_eval import evaluate_all_regressors, RESULTS_DIR

mod, cls, name = sys.argv[1], sys.argv[2], sys.argv[3]
kwargs = {}
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    try: kwargs[k] = eval(v)
    except Exception: kwargs[k] = v
M = getattr(importlib.import_module(mod), cls)
dataset_rmses = evaluate_all_regressors([(name, M(**kwargs))])

SIB = ("SegGAM","GA2M","SB_","GCV","Simple","AddGP","BinGP","TVG","PFN","SPGP","EMGP")
perf = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(RESULTS_DIR, "performance_results.csv"))):
    if r["rmse"] and not r["model"].startswith(SIB):
        perf[r["dataset"]][r["model"]] = float(r["rmse"])
for ds, mr in dataset_rmses.items():
    v = mr.get(name, float("nan"))
    if np.isfinite(v): perf[ds][name] = v

ranks = defaultdict(list)
for ds, mv in perf.items():
    if name not in mv: continue
    for i, m in enumerate(sorted(mv, key=lambda m: mv[m])): ranks[m].append(i + 1)
print("== mean ranks (sibling-free pool) ==")
for m in sorted(ranks, key=lambda m: np.mean(ranks[m]))[:8]:
    print(f"  {m:<18}: {np.mean(ranks[m]):.2f}{'  <<<' if m == name else ''}")
common = [ds for ds in perf if name in perf[ds]]
print("\n== richer metrics (medNRMSE, GMvsBest, top3, >EBM) ==")
for m in [name, "TabPFN", "EBM", "GBM", "RF"]:
    vals = [perf[ds].get(m, np.nan) for ds in common]
    ratios = [perf[ds][m] / min(perf[ds].values()) for ds in common if m in perf[ds]]
    top3 = sum(1 for ds in common if m in perf[ds] and sorted(perf[ds].values()).index(perf[ds][m]) < 3)
    gt = sum(1 for ds in common if m in perf[ds] and "EBM" in perf[ds] and perf[ds][m] < perf[ds]["EBM"])
    print(f"{m:<20} {np.nanmedian(vals):>8.4f} {np.exp(np.nanmean(np.log(ratios))):>9.3f} {top3:>5} {gt:>5}")
