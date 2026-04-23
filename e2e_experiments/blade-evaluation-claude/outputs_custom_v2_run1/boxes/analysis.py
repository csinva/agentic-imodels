import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nValue counts y:", df['y'].value_counts().sort_index())
print("Culture counts:", df['culture'].value_counts().sort_index())

# ── 2. Create outcome: chose_majority (binary) ────────────────────────────────
df['chose_majority'] = (df['y'] == 2).astype(int)
print("\nMajority choice rate:", df['chose_majority'].mean().round(3))

# ── 3. Bivariate: age vs majority choice ─────────────────────────────────────
print("\n--- Bivariate: age vs. chose_majority ---")
corr, pval = stats.pointbiserialr(df['age'], df['chose_majority'])
print(f"Point-biserial r = {corr:.4f}, p = {pval:.4g}")

# Age-group majority rates
age_groups = pd.cut(df['age'], bins=[3,6,9,12,15], labels=['4-6','7-9','10-12','13-14'])
print("\nMajority rate by age group:")
print(df.groupby(age_groups)['chose_majority'].mean().round(3))

# ── 4. Cross-cultural descriptives ────────────────────────────────────────────
print("\nMajority rate by culture:")
print(df.groupby('culture')['chose_majority'].agg(['mean','count']).round(3))

# ── 5. Classical OLS with controls ────────────────────────────────────────────
print("\n--- OLS with controls ---")
X_ols = sm.add_constant(df[['age', 'culture', 'gender', 'majority_first']])
ols = sm.OLS(df['chose_majority'], X_ols).fit()
print(ols.summary())

# ── 6. Logistic regression ────────────────────────────────────────────────────
print("\n--- Logistic regression with controls ---")
logit = sm.Logit(df['chose_majority'], X_ols).fit(maxiter=200, disp=False)
print(logit.summary())
print("\nOdds Ratios:")
print(np.exp(logit.params).round(4))

# ── 7. Interaction: age * culture ─────────────────────────────────────────────
print("\n--- OLS with age x culture interaction ---")
df['age_x_culture'] = df['age'] * df['culture']
X_int = sm.add_constant(df[['age', 'culture', 'gender', 'majority_first', 'age_x_culture']])
ols_int = sm.OLS(df['chose_majority'], X_int).fit()
print(f"age coef={ols_int.params['age']:.4f} p={ols_int.pvalues['age']:.4g}")
print(f"culture coef={ols_int.params['culture']:.4f} p={ols_int.pvalues['culture']:.4g}")
print(f"age*culture coef={ols_int.params['age_x_culture']:.4f} p={ols_int.pvalues['age_x_culture']:.4g}")

# Per-culture age slopes
print("\nPer-culture regression of age -> chose_majority:")
culture_slopes = {}
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    if len(sub) < 10:
        continue
    r, p = stats.pointbiserialr(sub['age'], sub['chose_majority'])
    culture_slopes[c] = (r, p, len(sub))
    print(f"  culture {c}: r={r:.3f}, p={p:.4g}, n={len(sub)}")

# ── 8. agentic_imodels ────────────────────────────────────────────────────────
feature_cols = ['age', 'culture', 'gender', 'majority_first']
X = df[feature_cols]
y = df['chose_majority'].values.astype(float)

print("\n=== SmartAdditiveRegressor ===")
m1 = SmartAdditiveRegressor()
m1.fit(X, y)
print(m1)

print("\n=== HingeEBMRegressor ===")
m2 = HingeEBMRegressor()
m2.fit(X, y)
print(m2)

# ── 9. Synthesize evidence ─────────────────────────────────────────────────────
age_coef   = ols.params['age']
age_pval   = ols.pvalues['age']
int_pval   = ols_int.pvalues['age_x_culture']

# Majority choice rate by age (young vs old)
young_rate = df[df['age'] <= 7]['chose_majority'].mean()
old_rate   = df[df['age'] >= 12]['chose_majority'].mean()

print(f"\nMajority rate young (age<=7): {young_rate:.3f}")
print(f"Majority rate old   (age>=12): {old_rate:.3f}")
print(f"\nOLS age coef={age_coef:.4f}, p={age_pval:.4g}")
print(f"Age*culture interaction p={int_pval:.4g}")

# ── 10. Write conclusion ────────────────────────────────────────────────────────
# Research question: "How do children's reliance on majority preference develop
# over growth in age across different cultural contexts?"
# Score 0-100: strong No=0, strong Yes=100
#
# Evidence summary:
# - Age NOT significant: OLS β=0.0019, p=0.803; logistic p=0.801
# - HingeEBM ranks age LAST (coef=0.0019), majority_first & gender dominate
# - SmartAdditive shows U-shaped age pattern (youngest and oldest higher), but
#   the corrections are modest relative to majority_first (0.281)
# - No significant age*culture interaction (p=0.513)
# - Per-culture age slopes: all non-significant (p>0.13), mixed positive/negative
# - Majority rate only goes 44.5% → 49.3% across age groups (small, non-sig)
# - Overall: weak and inconsistent evidence for age-based development → 25-35

score = 28

explanation = (
    f"The data provide weak, non-significant evidence for an age-related increase "
    f"in children's majority-preference reliance. "
    f"OLS with controls yields β={age_coef:.4f} for age (p={age_pval:.3g}), and logistic "
    f"regression gives OR=1.009 (p=0.80). "
    f"HingeEBMRegressor assigns age the smallest coefficient (0.0019), far behind "
    f"majority_first (0.2325) and gender (0.0351). "
    f"SmartAdditiveRegressor reveals a non-linear U-shape (youngest ≤4.5 and oldest "
    f">12.5 children show modest upward corrections), but the pattern is inconsistent "
    f"with a monotonic developmental trajectory. "
    f"Raw majority-choice rates change minimally across age groups (44–48% for ages "
    f"4–12, 60% for 13–14 in one bin, but N is small). "
    f"The age×culture interaction is not significant (p={int_pval:.3g}), and "
    f"within-culture age slopes are all non-significant (all p>0.13) with mixed "
    f"positive and negative directions across the 8 societies. "
    f"The dominant predictors are presentation order and gender, not age. "
    f"Overall, the evidence does not support a robust developmental increase in "
    f"majority preference across age in these cross-cultural data."
)

print("\nScore:", score)
print("Explanation:", explanation)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": score, "explanation": explanation}, f)

print("\nconclusion.txt written.")
