import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nBasic stats:")
print(df[['reader_view', 'speed', 'dyslexia', 'dyslexia_bin']].describe())

# The research question: Does Reader View improve reading speed for dyslexic individuals?
# DV: speed, IV: reader_view, moderated by dyslexia
# Focus on dyslexic participants (dyslexia > 0 or dyslexia_bin == 1)

print("\n=== Dyslexia distribution ===")
print(df['dyslexia'].value_counts())
print(df['dyslexia_bin'].value_counts())

print("\n=== Reader View distribution ===")
print(df['reader_view'].value_counts())

# Filter to dyslexic individuals
dyslexic = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic participants rows: {len(dyslexic)}")

print("\n=== Speed stats by reader_view (dyslexic only) ===")
print(dyslexic.groupby('reader_view')['speed'].describe())

# Log-transform speed (highly skewed)
df['log_speed'] = np.log1p(df['speed'])
dyslexic['log_speed'] = np.log1p(dyslexic['speed'])

print("\n=== Bivariate t-test (dyslexic): reader_view effect on speed ===")
rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed']
rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed']
t, p = stats.ttest_ind(rv0, rv1)
print(f"Reader View=0: mean={rv0.mean():.1f}, n={len(rv0)}")
print(f"Reader View=1: mean={rv1.mean():.1f}, n={len(rv1)}")
print(f"t={t:.3f}, p={p:.4f}")

# Also on log scale
lv0 = dyslexic[dyslexic['reader_view'] == 0]['log_speed']
lv1 = dyslexic[dyslexic['reader_view'] == 1]['log_speed']
t2, p2 = stats.ttest_ind(lv0, lv1)
print(f"\nLog-speed: Reader View=0 mean={lv0.mean():.3f}, Reader View=1 mean={lv1.mean():.3f}")
print(f"t={t2:.3f}, p={p2:.4f}")

# OLS with controls on full dataset with interaction term
# Encode categorical variables
df_enc = df.copy()
df_enc['log_speed'] = np.log1p(df_enc['speed'])
df_enc = pd.get_dummies(df_enc, columns=['device', 'education', 'english_native'], drop_first=True)
df_enc = df_enc.dropna(subset=['age', 'log_speed', 'reader_view', 'dyslexia_bin'])

# For OLS on full data with interaction
feature_cols = ['reader_view', 'dyslexia_bin', 'age', 'gender', 'retake_trial', 'Flesch_Kincaid', 'num_words']
# Add device and education dummies
device_cols = [c for c in df_enc.columns if c.startswith('device_')]
educ_cols = [c for c in df_enc.columns if c.startswith('education_')]
eng_cols = [c for c in df_enc.columns if c.startswith('english_native_')]
feature_cols += device_cols + educ_cols + eng_cols

df_model = df_enc[feature_cols + ['log_speed']].dropna().astype(float)
df_model['reader_view_x_dyslexia'] = df_model['reader_view'] * df_model['dyslexia_bin']

print("\n=== OLS: Full dataset with interaction term (reader_view x dyslexia_bin) ===")
X_ols = sm.add_constant(df_model[feature_cols + ['reader_view_x_dyslexia']])
y_ols = df_model['log_speed']
ols_result = sm.OLS(y_ols, X_ols).fit()
print(ols_result.summary())

# OLS on dyslexic only
dyslexic_enc = dyslexic.copy()
dyslexic_enc['log_speed'] = np.log1p(dyslexic_enc['speed'])
dyslexic_enc = pd.get_dummies(dyslexic_enc, columns=['device', 'education', 'english_native'], drop_first=True)
device_cols2 = [c for c in dyslexic_enc.columns if c.startswith('device_')]
educ_cols2 = [c for c in dyslexic_enc.columns if c.startswith('education_')]
eng_cols2 = [c for c in dyslexic_enc.columns if c.startswith('english_native_')]
feature_cols2 = ['reader_view', 'age', 'gender', 'retake_trial', 'Flesch_Kincaid', 'num_words']
feature_cols2 += device_cols2 + educ_cols2 + eng_cols2
df_dyslex_model = dyslexic_enc[feature_cols2 + ['log_speed']].dropna().astype(float)

print("\n=== OLS: Dyslexic participants only ===")
X_ols2 = sm.add_constant(df_dyslex_model[feature_cols2])
y_ols2 = df_dyslex_model['log_speed']
ols_result2 = sm.OLS(y_ols2, X_ols2).fit()
print(ols_result2.summary())

reader_view_coef = ols_result2.params.get('reader_view', None)
reader_view_pval = ols_result2.pvalues.get('reader_view', None)
print(f"\nreader_view coef (dyslexic only): {reader_view_coef:.4f}, p={reader_view_pval:.4f}")

# Interpretable models
print("\n\n=== Interpretable Models ===")

# Add agentic_imodels to path
sys.path.insert(0, '.')
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# Prepare data for agentic_imodels (dyslexic only, numeric features)
numeric_feats = ['reader_view', 'age', 'gender', 'retake_trial', 'Flesch_Kincaid', 'num_words']
available = [c for c in numeric_feats if c in df_dyslex_model.columns]
X_interp = df_dyslex_model[available]  # DataFrame preserves column names
y_interp = df_dyslex_model['log_speed'].values

# Also run on full data with dyslexia as feature
numeric_feats_full = ['reader_view', 'dyslexia_bin', 'age', 'gender', 'retake_trial', 'Flesch_Kincaid', 'num_words']
X_full_df = df_model[numeric_feats_full]  # DataFrame preserves column names
y_full_arr = df_model['log_speed'].values

print("\n--- SmartAdditiveRegressor (dyslexic only, honest GAM) ---")
sar = SmartAdditiveRegressor()
sar.fit(X_interp, y_interp)
print(sar)

print("\n--- HingeEBMRegressor (dyslexic only, high-rank) ---")
hebm = HingeEBMRegressor()
hebm.fit(X_interp, y_interp)
print(hebm)

print("\n--- SmartAdditiveRegressor (full data, honest GAM) ---")
sar_full = SmartAdditiveRegressor()
sar_full.fit(X_full_df, y_full_arr)
print(sar_full)

print("\n--- HingeEBMRegressor (full data) ---")
hebm_full = HingeEBMRegressor()
hebm_full.fit(X_full_df, y_full_arr)
print(hebm_full)

# Synthesize results
# Research question: Does Reader View IMPROVE reading speed for dyslexic individuals?
# "improve reading speed" = higher speed (more words per minute)
# reader_view=1 → higher speed means Reader View helps

coef_sign = np.sign(reader_view_coef) if reader_view_coef is not None else 0
positive_effect = (coef_sign > 0)
significant = (reader_view_pval < 0.05) if reader_view_pval is not None else False

print(f"\n=== SYNTHESIS ===")
print(f"Bivariate: Reader View=0 mean speed={rv0.mean():.1f}, Reader View=1 mean speed={rv1.mean():.1f}")
print(f"Bivariate t-test (raw speed): p={p:.4f}")
print(f"OLS (dyslexic only, controlled): reader_view coef={reader_view_coef:.4f}, p={reader_view_pval:.4f}")
print(f"Positive effect: {positive_effect}, Significant: {significant}")

# Determine Likert score
# Check direction from bivariate
bivariate_positive = rv1.mean() > rv0.mean()
print(f"Bivariate positive (rv1 > rv0): {bivariate_positive}")

# Calibrate score
if significant and positive_effect:
    # Strong or moderate? Check magnitude
    effect_size = abs(reader_view_coef) if reader_view_coef else 0
    if effect_size > 0.3:
        score = 75
    else:
        score = 60
elif not significant and positive_effect:
    score = 40
elif not significant and not positive_effect:
    score = 30
else:
    score = 20

print(f"\nLikert score: {score}")

# Write conclusion
explanation = (
    f"The research question is whether Reader View improves reading speed for dyslexic individuals. "
    f"Bivariate analysis (dyslexic participants, n={len(dyslexic)}): "
    f"Reader View=0 mean speed={rv0.mean():.1f} wpm, Reader View=1 mean speed={rv1.mean():.1f} wpm "
    f"({'Reader View faster' if bivariate_positive else 'Reader View slower'}), t-test p={p:.4f}. "
    f"OLS regression (dyslexic only, controlling for age, gender, device, education, Flesch-Kincaid, num_words): "
    f"reader_view coefficient={reader_view_coef:.4f}, p={reader_view_pval:.4f} "
    f"({'significant' if significant else 'not significant'}). "
    f"The interaction term (reader_view x dyslexia_bin) in the full-dataset OLS had coefficient "
    f"{ols_result.params.get('reader_view_x_dyslexia', 'N/A'):.4f}, "
    f"p={ols_result.pvalues.get('reader_view_x_dyslexia', float('nan')):.4f}. "
    f"Interpretable models (SmartAdditiveRegressor, HingeEBMRegressor) were fit on the dyslexic subsample "
    f"and on the full dataset to assess direction, magnitude, shape, and robustness of the reader_view effect. "
    f"{'The effect is positive (Reader View increases speed) and statistically significant, supporting a Yes answer.' if significant and positive_effect else 'The evidence for Reader View improving speed in dyslexic individuals is weak or mixed based on significance and direction from controlled models.'}"
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\n=== conclusion.txt written ===")
print(json.dumps(result, indent=2))
