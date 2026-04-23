import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col])

# Compute fertility proxy: estimate cycle day and proximity to ovulation
# Fertile window is typically around ovulation, ~14 days before next period
df['days_since_last_period'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
df['computed_cycle_length'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Use reported cycle length if computed is missing or odd
df['cycle_length'] = df['ReportedCycleLength'].fillna(df['computed_cycle_length'])

# Days until next period
df['days_until_next_period'] = df['cycle_length'] - df['days_since_last_period']

# Ovulation ~14 days before next period; proximity to ovulation = fertility
# fertility_score: higher = closer to ovulation
df['days_to_ovulation'] = df['days_until_next_period'] - 14
df['fertility_score'] = -np.abs(df['days_to_ovulation'])  # higher = nearer ovulation

# High-fertility binary indicator (within 3 days of ovulation)
df['high_fertility'] = (np.abs(df['days_to_ovulation']) <= 3).astype(int)

print("\nFertility score stats:")
print(df[['days_since_last_period', 'cycle_length', 'days_to_ovulation', 'fertility_score', 'high_fertility']].describe())

# Religiosity composite
df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)
print("\nReligiosity stats:", df['religiosity'].describe())
print("High fertility N:", df['high_fertility'].sum())

# Drop rows with missing key variables
df_clean = df.dropna(subset=['religiosity', 'fertility_score', 'Relationship', 'Sure1', 'Sure2'])
print(f"\nClean rows: {len(df_clean)}")

# Bivariate correlation
r, p = stats.pearsonr(df_clean['fertility_score'], df_clean['religiosity'])
print(f"\nBivariate Pearson r(fertility_score, religiosity) = {r:.4f}, p = {p:.4f}")

r2, p2 = stats.pointbiserialr(df_clean['high_fertility'], df_clean['religiosity'])
print(f"Point-biserial r(high_fertility, religiosity) = {r2:.4f}, p = {p2:.4f}")

# t-test: high vs low fertility
high = df_clean[df_clean['high_fertility'] == 1]['religiosity']
low = df_clean[df_clean['high_fertility'] == 0]['religiosity']
t, p_t = stats.ttest_ind(high, low)
print(f"\nHigh fertility mean: {high.mean():.3f}, Low fertility mean: {low.mean():.3f}")
print(f"t-test t={t:.4f}, p={p_t:.4f}")

# Classical OLS with controls
X_ols = sm.add_constant(df_clean[['fertility_score', 'Relationship', 'Sure1', 'Sure2', 'cycle_length']])
ols_model = sm.OLS(df_clean['religiosity'], X_ols).fit()
print("\n=== OLS Summary ===")
print(ols_model.summary())

# Also OLS with high_fertility binary
X_ols2 = sm.add_constant(df_clean[['high_fertility', 'Relationship', 'Sure1', 'Sure2', 'cycle_length']])
ols_model2 = sm.OLS(df_clean['religiosity'], X_ols2).fit()
print("\n=== OLS with high_fertility binary ===")
print(ols_model2.summary())

# Interpretable models from agentic_imodels
feature_cols = ['fertility_score', 'high_fertility', 'Relationship', 'Sure1', 'Sure2', 'cycle_length', 'days_since_last_period']
X = df_clean[feature_cols].values
y = df_clean['religiosity'].values

print("\n=== SmartAdditiveRegressor ===")
sam = SmartAdditiveRegressor()
sam.fit(X, y)
print(sam)

print("\n=== HingeEBMRegressor ===")
hebm = HingeEBMRegressor()
hebm.fit(X, y)
print(hebm)

print("\n=== WinsorizedSparseOLSRegressor ===")
wols = WinsorizedSparseOLSRegressor()
wols.fit(X, y)
print(wols)

# Summary of evidence
fert_coef = ols_model.params['fertility_score']
fert_pval = ols_model.pvalues['fertility_score']
high_fert_coef = ols_model2.params['high_fertility']
high_fert_pval = ols_model2.pvalues['high_fertility']

print(f"\n--- Evidence Summary ---")
print(f"OLS fertility_score coef={fert_coef:.4f}, p={fert_pval:.4f}")
print(f"OLS high_fertility coef={high_fert_coef:.4f}, p={high_fert_pval:.4f}")
print(f"Bivariate r={r:.4f}, p={p:.4f}")
print(f"High vs low t-test p={p_t:.4f}")

# Assess evidence strength and write conclusion
# Significant = p < 0.05 with controlled analysis
# Need to check if fertility_score (or high_fertility) has a clear effect

evidence_notes = (
    f"Bivariate Pearson r={r:.4f} (p={p:.4f}). "
    f"OLS with controls: fertility_score coef={fert_coef:.4f} (p={fert_pval:.4f}). "
    f"High-fertility binary coef={high_fert_coef:.4f} (p={high_fert_pval:.4f}). "
    f"High fertility mean religiosity={high.mean():.3f} vs low={low.mean():.3f}."
)
print("\nEvidence notes:", evidence_notes)

# Score calibration:
# - Both OLS p-values for fertility predictors > 0.05 with controls → weak/null evidence
# - Bivariate r also likely small
# Check p-values to assign score
if fert_pval < 0.01 and abs(r) > 0.15:
    score = 75
    explanation = f"Strong significant effect of fertility on religiosity. {evidence_notes}"
elif fert_pval < 0.05 or high_fert_pval < 0.05:
    score = 55
    explanation = f"Moderate evidence: fertility measure significant under at least one specification. {evidence_notes}"
elif fert_pval < 0.1 or high_fert_pval < 0.1:
    score = 35
    explanation = f"Weak / marginal evidence of fertility effect on religiosity. {evidence_notes}"
else:
    score = 15
    explanation = f"Little to no evidence of hormonal fertility effect on religiosity. Neither continuous nor binary fertility measure is significant after controls. {evidence_notes}"

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print("\nWritten conclusion.txt")
