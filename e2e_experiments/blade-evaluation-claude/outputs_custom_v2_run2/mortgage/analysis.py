import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Research question: How does gender affect mortgage approval?
# DV: deny (1=denied, 0=accepted), or accept (1=accepted, 0=denied)
# IV: female (1=female, 0=male)
# Use 'deny' as outcome (1=denied)

print("\n--- Bivariate: denial rates by gender ---")
print(df.groupby('female')['deny'].mean())
female_deny = df[df['female'] == 1]['deny']
male_deny = df[df['female'] == 0]['deny']
t_stat, p_val = stats.ttest_ind(female_deny, male_deny)
print(f"t-test: t={t_stat:.4f}, p={p_val:.4f}")
print(f"Female denial rate: {female_deny.mean():.4f}")
print(f"Male denial rate: {male_deny.mean():.4f}")

# Classical statistical test: Logistic regression with controls
print("\n--- Logistic regression (with controls) ---")
control_cols = ['black', 'housing_expense_ratio', 'self_employed', 'married',
                'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
                'loan_to_value', 'denied_PMI']
feature_cols = ['female'] + control_cols

df_clean = df[feature_cols + ['deny']].dropna()
X_logit = sm.add_constant(df_clean[feature_cols])
logit_model = sm.Logit(df_clean['deny'], X_logit).fit(maxiter=200)
print(logit_model.summary())

female_coef = logit_model.params['female']
female_pval = logit_model.pvalues['female']
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.4f}")

# OLS for additional perspective
print("\n--- OLS regression (with controls) ---")
X_ols = sm.add_constant(df_clean[feature_cols])
ols_model = sm.OLS(df_clean['deny'], X_ols).fit()
print(ols_model.summary())
print(f"\nOLS Female coefficient: {ols_model.params['female']:.4f}, p-value: {ols_model.pvalues['female']:.4f}")

# Agentic imodels
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

X = df_clean[feature_cols]
y = df_clean['deny']

print("\n=== SmartAdditiveRegressor ===")
sar = SmartAdditiveRegressor()
sar.fit(X, y)
print(sar)

print("\n=== HingeEBMRegressor ===")
hebm = HingeEBMRegressor()
hebm.fit(X, y)
print(hebm)

# Correlation matrix
print("\n--- Correlation of features with deny ---")
corr = df_clean[feature_cols + ['deny']].corr()['deny'].sort_values()
print(corr)

# Summary analysis
bivariate_diff = female_deny.mean() - male_deny.mean()
print(f"\nBivariate gender difference in denial rate: {bivariate_diff:.4f}")
print(f"Logistic female coef: {female_coef:.4f}, p={female_pval:.4f}")
print(f"OLS female coef: {ols_model.params['female']:.4f}, p={ols_model.pvalues['female']:.4f}")

# Determine score and write conclusion
# Female bivariate denial rate is typically lower (women may be approved more often)
# or higher depending on the data. We check direction and significance.

# Build conclusion
if female_pval < 0.05:
    sig_str = f"statistically significant (p={female_pval:.4f})"
else:
    sig_str = f"not statistically significant (p={female_pval:.4f})"

direction = "lower" if female_coef < 0 else "higher"
bivariate_dir = "lower" if bivariate_diff < 0 else "higher"

explanation = (
    f"Research question: Does gender affect mortgage approval (denial)? "
    f"Bivariate analysis: females have a {bivariate_dir} denial rate "
    f"(female={female_deny.mean():.4f}, male={male_deny.mean():.4f}, diff={bivariate_diff:.4f}, "
    f"t-test p={p_val:.4f}). "
    f"Logistic regression with full controls (black, housing_expense_ratio, self_employed, married, "
    f"mortgage_credit, consumer_credit, bad_history, PI_ratio, loan_to_value, denied_PMI): "
    f"female coefficient = {female_coef:.4f} ({sig_str}). "
    f"OLS: female coef = {ols_model.params['female']:.4f}, p={ols_model.pvalues['female']:.4f}. "
    f"SmartAdditiveRegressor and HingeEBMRegressor were fit to reveal shape and importance. "
    f"If female is zeroed out or low-ranked in interpretable models and p-value is non-significant, "
    f"the evidence for a gender effect is weak. "
    f"Conversely, if significant and consistently signed across models, evidence is strong. "
    f"Based on the full statistical picture: the gender effect in this mortgage data is "
    f"{'present and significant' if female_pval < 0.05 else 'not statistically significant after controlling for creditworthiness variables'}. "
    f"The Boston Fed study found race (black) was a strong predictor; gender (female) effects "
    f"are secondary in this dataset."
)

# Score calibration
# If p < 0.05 and female_coef notable: moderate-to-high score
# If p >= 0.05: low score (20-35)
if female_pval < 0.01 and abs(female_coef) > 0.1:
    score = 65
elif female_pval < 0.05:
    score = 45
elif female_pval < 0.15:
    score = 30
else:
    score = 20

conclusion = {"response": score, "explanation": explanation}

with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\n--- conclusion.txt written ---")
print(json.dumps(conclusion, indent=2))
