"""Cross-judge agreement: independent Claude judge vs GPT-4o (judge1), run1, Claude agent."""
import csv, numpy as np
from scipy.stats import pearsonr, spearmanr

# My independent (Claude/Haiku-role) scores: (correctness, completeness, clarity)
claude_judge = {
 "standard/affairs": (6,4,4), "standard/amtl": (7,6,6), "standard/boxes": (5,3,3),
 "standard/caschools": (8,7,7), "standard/crofoot": (4,2,2), "standard/fertility": (8,6,6),
 "standard/fish": (6,5,5), "standard/hurricane": (7,5,5), "standard/mortgage": (6,6,5),
 "standard/panda_nuts": (7,5,6), "standard/reading": (6,4,4), "standard/soccer": (7,6,6),
 "standard/teachingratings": (8,6,6),
 "custom_v2/affairs": (7,7,7), "custom_v2/amtl": (9,8,8), "custom_v2/boxes": (9,9,9),
 "custom_v2/caschools": (9,9,9), "custom_v2/crofoot": (5,3,3), "custom_v2/fertility": (8,5,5),
 "custom_v2/fish": (8,8,8), "custom_v2/hurricane": (8,7,8), "custom_v2/mortgage": (8,8,8),
 "custom_v2/panda_nuts": (8,7,8), "custom_v2/reading": (7,6,6), "custom_v2/soccer": (8,7,7),
 "custom_v2/teachingratings": (8,7,8),
}

# GPT-4o judge1 scores (from repo CSVs)
gpt = {
 "standard/affairs": (8,7,8), "standard/amtl": (8,7,8), "standard/boxes": (3,2,3),
 "standard/caschools": (9,8,9), "standard/crofoot": (5,3,4), "standard/fertility": (9,7,8),
 "standard/fish": (4,5,6), "standard/hurricane": (9,8,8), "standard/mortgage": (5,6,5),
 "standard/panda_nuts": (7,6,7), "standard/reading": (7,5,6), "standard/soccer": (8,7,6),
 "standard/teachingratings": (8,7,8),
 "custom_v2/affairs": (9,8,9), "custom_v2/amtl": (9,8,9), "custom_v2/boxes": (9,8,9),
 "custom_v2/caschools": (9,9,9), "custom_v2/crofoot": (5,3,4), "custom_v2/fertility": (9,8,9),
 "custom_v2/fish": (9,8,9), "custom_v2/hurricane": (9,8,9), "custom_v2/mortgage": (9,9,9),
 "custom_v2/panda_nuts": (8,7,8), "custom_v2/reading": (9,8,8), "custom_v2/soccer": (9,8,9),
 "custom_v2/teachingratings": (9,8,9),
}

keys = sorted(claude_judge)
def overall(d,k): return np.mean(d[k])
cl = np.array([overall(claude_judge,k) for k in keys])
gp = np.array([overall(gpt,k) for k in keys])

print("=== Cross-judge agreement (overall = mean of 3 dims), n=%d items ===" % len(keys))
r,p = pearsonr(cl,gp); rs,ps = spearmanr(cl,gp)
print(f"Pearson r  = {r:.3f} (p={p:.1e})")
print(f"Spearman r = {rs:.3f} (p={ps:.1e})")
# per-dimension pearson
for i,dim in enumerate(["correctness","completeness","clarity"]):
    a=np.array([claude_judge[k][i] for k in keys]); b=np.array([gpt[k][i] for k in keys])
    print(f"  {dim:12} Pearson r = {pearsonr(a,b)[0]:.3f}")

# mean absolute difference on overall
print(f"Mean |Claude - GPT4o| overall = {np.abs(cl-gp).mean():.2f} points (0-10 scale)")

# gain persistence: custom_v2 vs standard under each judge
def modemean(d, mode):
    ks=[k for k in keys if k.startswith(mode)]
    return np.mean([overall(d,k) for k in ks])
print("\n=== custom_v2 vs standard (overall mean) ===")
for jname,d in [("GPT-4o (judge1)",gpt),("Claude (independent)",claude_judge)]:
    s=modemean(d,"standard"); c=modemean(d,"custom_v2")
    print(f"  {jname:22}: standard={s:.2f}  custom_v2={c:.2f}  gain=+{c-s:.2f} ({100*(c-s)/s:.0f}%)")

# paired improvement count (per dataset, custom_v2 > standard) under Claude judge
datasets=sorted(set(k.split("/")[1] for k in keys))
imp=0
for ds in datasets:
    if overall(claude_judge,f"custom_v2/{ds}")>overall(claude_judge,f"standard/{ds}"): imp+=1
print(f"\nUnder Claude judge: custom_v2 > standard on {imp}/{len(datasets)} datasets")
