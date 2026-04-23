import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeGAMRegressor

# Load data
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.head())
print("\nSummary stats:")
print(df.describe())
print("\nValue counts - sex:", df['sex'].value_counts().to_dict())
print("Value counts - help:", df['help'].value_counts().to_dict())
print("Value counts - hammer:", df['hammer'].value_counts().to_dict())

# Compute nut-cracking efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']
print("\nEfficiency stats:", df['efficiency'].describe())

# Encode categoricals
df['sex_m'] = (df['sex'] == 'm').astype(int)
df['help_y'] = (df['help'] == 'y').astype(int)

print("\n--- Bivariate tests ---")
eff_male = df[df['sex_m'] == 1]['efficiency']
eff_female = df[df['sex_m'] == 0]['efficiency']
t_sex, p_sex = stats.ttest_ind(eff_male, eff_female)
print(f"Sex effect (t-test): t={t_sex:.3f}, p={p_sex:.3f}, male_mean={eff_male.mean():.4f}, female_mean={eff_female.mean():.4f}")

eff_help = df[df['help_y'] == 1]['efficiency']
eff_nohelp = df[df['help_y'] == 0]['efficiency']
t_help, p_help = stats.ttest_ind(eff_help, eff_nohelp)
print(f"Help effect (t-test): t={t_help:.3f}, p={p_help:.3f}, help_mean={eff_help.mean():.4f}, nohelp_mean={eff_nohelp.mean():.4f}")

r_age, p_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"Age-efficiency correlation: r={r_age:.3f}, p={p_age:.3f}")

print("\n--- OLS with controls ---")
X_ols = sm.add_constant(df[['age', 'sex_m', 'help_y', 'seconds']])
ols = sm.OLS(df['efficiency'], X_ols).fit()
print(ols.summary())

print("\n--- OLS without session_duration control ---")
X_ols2 = sm.add_constant(df[['age', 'sex_m', 'help_y']])
ols2 = sm.OLS(df['efficiency'], X_ols2).fit()
print(ols2.summary())

# Interpretable models
X_interp = df[['age', 'sex_m', 'help_y', 'seconds']].copy()
y_interp = df['efficiency'].copy()

print("\n=== SmartAdditiveRegressor ===")
sar = SmartAdditiveRegressor()
sar.fit(X_interp, y_interp)
print(sar)

print("\n=== HingeGAMRegressor ===")
hgr = HingeGAMRegressor()
hgr.fit(X_interp, y_interp)
print(hgr)

# Summarize findings
print("\n--- Summary ---")
age_coef = ols2.params['age']
age_pval = ols2.pvalues['age']
sex_coef = ols2.params['sex_m']
sex_pval = ols2.pvalues['sex_m']
help_coef = ols2.params['help_y']
help_pval = ols2.pvalues['help_y']

print(f"Age: coef={age_coef:.4f}, p={age_pval:.4f}")
print(f"Sex(male): coef={sex_coef:.4f}, p={sex_pval:.4f}")
print(f"Help: coef={help_coef:.4f}, p={help_pval:.4f}")

# Compute Likert score
# Research question asks about age, sex, AND help collectively.
# We'll assess overall evidence.
effects = []
sig_count = 0
for coef, pval, name in [(age_coef, age_pval, 'age'), (sex_coef, sex_pval, 'sex'), (help_coef, help_pval, 'help')]:
    if pval < 0.05:
        sig_count += 1
        print(f"  {name}: SIGNIFICANT, coef={coef:.4f}")
    else:
        print(f"  {name}: not significant (p={pval:.4f})")

# Determine score: if most variables are significant with reasonable effect size, score high
# The question is about influence collectively
if sig_count == 3:
    score = 85
elif sig_count == 2:
    score = 70
elif sig_count == 1:
    score = 45
else:
    score = 20

# Fine-tune based on direction/magnitude
explanation = (
    f"Research question: How do age, sex, and help influence nut-cracking efficiency (nuts/second). "
    f"OLS without controls: age (beta={age_coef:.4f}, p={age_pval:.3f}), "
    f"sex_male (beta={sex_coef:.4f}, p={sex_pval:.3f}), "
    f"help (beta={help_coef:.4f}, p={help_pval:.3f}). "
    f"Age correlation with efficiency: r={r_age:.3f} (p={p_age:.3f}). "
    f"Bivariate: sex p={p_sex:.3f}, help p={p_help:.3f}. "
    f"Significant predictors: {sig_count}/3. "
    f"SmartAdditiveRegressor and HingeGAMRegressor were fitted to characterize directions and shapes. "
    f"Overall: {'Strong' if sig_count >= 2 else 'Weak'} evidence that age, sex, and help collectively influence nut-cracking efficiency."
)

result = {"response": score, "explanation": explanation}
print("\nResult:", result)

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
