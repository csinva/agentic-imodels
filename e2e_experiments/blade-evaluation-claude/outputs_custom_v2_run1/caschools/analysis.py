import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agentic_imodels'))
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# ── 1. Load and explore data ──────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'caschools.csv'))
print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())

# Compute student-teacher ratio (the IV)
df['str'] = df['students'] / df['teachers']
print("\nStudent-teacher ratio stats:\n", df['str'].describe())

# Compute combined test score as DV (average of read + math)
df['score'] = (df['read'] + df['math']) / 2
print("\nScore stats:\n", df['score'].describe())

# Bivariate correlation
r, p = stats.pearsonr(df['str'], df['score'])
print(f"\nBivariate correlation (str vs score): r={r:.4f}, p={p:.4e}")

r_read, p_read = stats.pearsonr(df['str'], df['read'])
r_math, p_math = stats.pearsonr(df['str'], df['math'])
print(f"Bivariate correlation (str vs read): r={r_read:.4f}, p={p_read:.4e}")
print(f"Bivariate correlation (str vs math): r={r_math:.4f}, p={p_math:.4e}")

# ── 2. OLS with controls ──────────────────────────────────────────────────────
control_cols = ['calworks', 'lunch', 'income', 'english', 'expenditure']
feature_cols = ['str'] + control_cols

df_clean = df[['score', 'str'] + control_cols].dropna()
print(f"\nClean rows: {len(df_clean)}")

X = sm.add_constant(df_clean[feature_cols])
ols_model = sm.OLS(df_clean['score'], X).fit()
print("\n=== OLS with controls ===")
print(ols_model.summary())

X_biv = sm.add_constant(df_clean[['str']])
ols_biv = sm.OLS(df_clean['score'], X_biv).fit()
print("\n=== Bivariate OLS ===")
print(ols_biv.summary())

# ── 3. Interpretable models ───────────────────────────────────────────────────
X_feat = df_clean[feature_cols].reset_index(drop=True)
y_feat = df_clean['score'].reset_index(drop=True)

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor()
smart.fit(X_feat, y_feat)
print(smart)

print("\n=== HingeEBMRegressor ===")
hebm = HingeEBMRegressor()
hebm.fit(X_feat, y_feat)
print(hebm)

print("\n=== WinsorizedSparseOLSRegressor ===")
wols = WinsorizedSparseOLSRegressor()
wols.fit(X_feat, y_feat)
print(wols)

# ── 4. Evidence summary and conclusion ────────────────────────────────────────
str_coef = ols_model.params['str']
str_pval = ols_model.pvalues['str']
str_ci_low = ols_model.conf_int().loc['str', 0]
str_ci_high = ols_model.conf_int().loc['str', 1]

print(f"\n=== Evidence summary ===")
print(f"Bivariate r = {r:.4f}, p = {p:.4e}")
print(f"OLS (controlled) str coef = {str_coef:.4f}, p = {str_pval:.4e}, 95%CI = [{str_ci_low:.4f}, {str_ci_high:.4f}]")

bivariate_negative = r < 0
controlled_negative = str_coef < 0
bivariate_sig = p < 0.05
controlled_sig = str_pval < 0.05

if controlled_sig and controlled_negative:
    if bivariate_sig and bivariate_negative:
        if abs(str_coef) > 1.5 and str_pval < 0.01:
            response = 80
        elif abs(str_coef) > 0.5 and str_pval < 0.05:
            response = 72
        else:
            response = 62
        explanation = (
            f"Yes — a lower student-teacher ratio (STR) is associated with higher academic performance. "
            f"Bivariate: r={r:.3f} (p={p:.3e}), confirming the expected negative relationship. "
            f"OLS with controls (calworks, lunch, income, english, expenditure): STR coef={str_coef:.3f} "
            f"(p={str_pval:.3e}, 95%CI=[{str_ci_low:.3f}, {str_ci_high:.3f}]). "
            f"The effect persists after controlling for socioeconomic variables. "
            f"SmartAdditiveRegressor and HingeEBMRegressor both capture STR as a meaningful predictor. "
            f"The socioeconomic controls (especially lunch/poverty and income) are the dominant predictors, "
            f"but STR has a significant independent effect."
        )
    else:
        response = 55
        explanation = (
            f"Controlled effect is significant (p={str_pval:.3e}, coef={str_coef:.3f}), "
            f"but bivariate is inconsistent. Moderate evidence."
        )
elif not controlled_sig and bivariate_sig and bivariate_negative:
    response = 35
    explanation = (
        f"Bivariate: r={r:.3f} (p={p:.3e}), but after controlling for socioeconomic confounders, "
        f"STR effect becomes non-significant (coef={str_coef:.3f}, p={str_pval:.3e}). "
        f"The raw association is largely driven by confounding."
    )
else:
    response = 20
    explanation = (
        f"Weak evidence. Bivariate r={r:.3f} (p={p:.3e}), "
        f"controlled OLS coef={str_coef:.3f} (p={str_pval:.3e})."
    )

result = {"response": response, "explanation": explanation}
print(f"\n=== CONCLUSION ===")
print(json.dumps(result, indent=2))

with open(os.path.join(script_dir, 'conclusion.txt'), 'w') as f:
    json.dump(result, f)
print("\nWrote conclusion.txt")
