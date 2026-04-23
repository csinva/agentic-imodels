import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agentic_imodels'))
from agentic_imodels import SmartAdditiveRegressor, HingeGAMRegressor

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df.head())
print("\nGenus counts:")
print(df['genus'].value_counts())
print("\nTooth class counts:")
print(df['tooth_class'].value_counts())

# Outcome: proportion of teeth lost
df['amtl_rate'] = df['num_amtl'] / df['sockets']
print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].describe())

# ── 2. Classical GLM (Binomial) ──────────────────────────────────────────────
# Encode species: is_human indicator
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)
df['is_pan'] = (df['genus'] == 'Pan').astype(int)
df['is_pongo'] = (df['genus'] == 'Pongo').astype(int)

# Tooth class dummies (reference = Anterior)
tooth_dummies = pd.get_dummies(df['tooth_class'], drop_first=True, prefix='tooth')

df_model = pd.concat([df[['is_human', 'is_pan', 'is_pongo', 'age', 'prob_male', 'amtl_rate']], tooth_dummies], axis=1)
df_model = df_model.dropna()

X_glm = sm.add_constant(df_model.drop(columns='amtl_rate').astype(float))
y_glm = df_model['amtl_rate']

# Binomial GLM using proportion outcome with frequency weights
n_success = df.loc[df_model.index, 'num_amtl'].values.astype(float)
n_total = df.loc[df_model.index, 'sockets'].values.astype(float)
endog = np.stack([n_success, n_total - n_success], axis=1)
glm_model = sm.GLM(endog, X_glm, family=sm.families.Binomial()).fit()
print("\n=== GLM Binomial Summary ===")
print(glm_model.summary())

# Bivariate comparison: t-test on AMTL rates
human_rates = df.loc[df['genus'] == 'Homo sapiens', 'amtl_rate'].dropna()
nonhuman_rates = df.loc[df['genus'] != 'Homo sapiens', 'amtl_rate'].dropna()
tstat, pval = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nBivariate t-test: human vs non-human AMTL rate")
print(f"  Human mean: {human_rates.mean():.4f}, Non-human mean: {nonhuman_rates.mean():.4f}")
print(f"  t={tstat:.3f}, p={pval:.4e}")

# ── 3. Interpretable models ──────────────────────────────────────────────────
feature_cols = ['is_human', 'is_pan', 'is_pongo', 'age', 'prob_male'] + list(tooth_dummies.columns)
X_interp = df_model[feature_cols]
y_interp = df_model['amtl_rate']

print("\n=== SmartAdditiveRegressor ===")
sar = SmartAdditiveRegressor()
sar.fit(X_interp, y_interp)
print(sar)

print("\n=== HingeGAMRegressor ===")
hgam = HingeGAMRegressor()
hgam.fit(X_interp, y_interp)
print(hgam)

# ── 4. Conclusion ─────────────────────────────────────────────────────────────
is_human_coef = glm_model.params.get('is_human', None)
is_human_pval = glm_model.pvalues.get('is_human', None)

human_mean = human_rates.mean()
nonhuman_mean = nonhuman_rates.mean()

print(f"\nSummary:")
print(f"  GLM is_human coef (log-odds): {is_human_coef:.4f}, p={is_human_pval:.4e}")
print(f"  Bivariate: human={human_mean:.4f}, non-human={nonhuman_mean:.4f}")

# Calibrate Likert score
# Strong significant effect persisting across models → 75-100
# is_human coef direction: positive → humans have higher AMTL
if is_human_pval < 0.001 and is_human_coef > 0:
    score = 90
    reasoning = "very strong evidence"
elif is_human_pval < 0.05 and is_human_coef > 0:
    score = 70
    reasoning = "moderate evidence"
elif is_human_pval >= 0.05 and is_human_coef > 0:
    score = 35
    reasoning = "weak/marginal evidence"
else:
    score = 15
    reasoning = "no or negative evidence"

explanation = (
    f"Research question: Do modern humans have higher AMTL compared to non-human primates "
    f"(Pan, Pongo, Papio) after controlling for age, sex, and tooth class? "
    f"Bivariate analysis: human mean AMTL rate={human_mean:.4f} vs non-human={nonhuman_mean:.4f} "
    f"(t-test p={pval:.4e}). "
    f"Binomial GLM with controls (age, prob_male, tooth class, genus dummies): "
    f"is_human coefficient={is_human_coef:.4f} (log-odds), p={is_human_pval:.4e}. "
    f"SmartAdditiveRegressor and HingeGAMRegressor both fitted to examine feature importance and shape. "
    f"Evidence: {reasoning}. "
    f"Conclusion: Humans show {'higher' if is_human_coef > 0 else 'lower'} AMTL than non-human primates "
    f"after controlling for confounders. Score={score}/100."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
