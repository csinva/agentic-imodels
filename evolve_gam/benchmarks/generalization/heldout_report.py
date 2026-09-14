import os
"""Held-out summary: v48 and v47 vs EBM on the de-duplicated TabArena + CTR23 sets."""
import csv, os
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
J = os.path.dirname(os.path.abspath(__file__))
R = defaultdict(dict); suite = {}
for r in csv.DictReader(open(f"{J}/results_heldout.csv")):
    v = float(r["rmse"])
    if np.isfinite(v): R[r["dataset"]][r["model"]] = v; suite[r["dataset"]] = r["suite"]
def block(label, names):
    print(f"\n{label} ({len(names)} datasets)")
    print(f"  {'model':<5}{'wins vs EBM':>13}{'GM ratio':>10}{'p90':>7}{'sign p':>9}")
    for m in ("v49h", "v49", "v48", "v47"):
        c = [d for d in names if m in R[d] and "EBM" in R[d]]
        if not c: continue
        rat = np.array([R[d][m] / R[d]["EBM"] for d in c]); k = int((rat < 1).sum())
        print(f"  {m:<5}{k:>8}/{len(c):<4}{np.exp(np.log(rat).mean()):>10.3f}{np.percentile(rat, 90):>7.3f}{binomtest(k, len(c)).pvalue:>9.3f}")
    c = [d for d in names if "v48" in R[d] and "v47" in R[d]]
    if c:
        rat = np.array([R[d]["v48"] / R[d]["v47"] for d in c]); k = int((rat < 1).sum())
        print(f"  v48 vs v47: better on {k}/{len(c)}, GM {np.exp(np.log(rat).mean()):.3f}, sign p={binomtest(k, len(c)).pvalue:.3f}")
names = sorted(R)
block("TabArena (houses removed)", [d for d in names if suite[d] == "tabarena"])
block("CTR23 (imodels + TabArena overlaps removed)", [d for d in names if suite[d] == "ctr23"])
block("POOLED held-out", names)
print(f"\n{'dataset':<32}{'EBM':>8}{'v47':>8}{'v48':>8}{'v49h':>8}{'v49h/EBM':>10}")
for d in sorted(names, key=lambda d: R[d].get("v49h", R[d].get("v48", np.nan)) / R[d].get("EBM", np.nan)):
    e, a, b, c = (R[d].get(k, np.nan) for k in ("EBM", "v47", "v48", "v49h"))
    print(f"{d:<32}{e:>8.4f}{a:>8.4f}{b:>8.4f}{c:>8.4f}{c/e:>10.3f}")
