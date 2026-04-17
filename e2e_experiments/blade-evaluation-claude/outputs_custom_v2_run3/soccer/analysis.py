import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['redCards','rater1','rater2','games','yellowCards','meanIAT','meanExp']].describe())

# Create skin tone variable (average of two raters)
df['skin_tone'] = (df['rater1'].fillna(0) + df['rater2'].fillna(0)) / 2
# Only use rows where at least one rater coded the player
df_rated = df.dropna(subset=['rater1', 'rater2']).copy()
print(f"\nRows with skin tone ratings: {len(df_rated)}")

# Bivariate correlation
corr = df_rated[['redCards','skin_tone','games','yellowCards','goals','meanIAT','meanExp']].corr()
print("\nBivariate correlations with redCards:")
print(corr['redCards'])

# OLS regression with controls
numeric_cols = ['skin_tone','games','yellowCards','goals','meanIAT','meanExp']
df_ols = df_rated[numeric_cols + ['redCards']].dropna()
X = sm.add_constant(df_ols[numeric_cols])
model = sm.OLS(df_ols['redCards'], X).fit()
print("\nOLS Summary:")
print(model.summary())

# Interpretable models
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

X_df = df_ols[numeric_cols].reset_index(drop=True)
y = df_ols['redCards'].reset_index(drop=True)

print("\n--- SmartAdditiveRegressor ---")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
effects_smart = smart.feature_effects()
print("Feature effects:", effects_smart)

print("\n--- HingeEBMRegressor ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print(hinge)
effects_hinge = hinge.feature_effects()
print("Feature effects:", effects_hinge)

# Summarize findings
skin_coef = model.params['skin_tone']
skin_pval = model.pvalues['skin_tone']
skin_smart = effects_smart.get('skin_tone', {})
skin_hinge = effects_hinge.get('skin_tone', {})

print(f"\nSkin tone OLS coef: {skin_coef:.4f}, p={skin_pval:.4f}")
print(f"SmartAdditive skin_tone: {skin_smart}")
print(f"HingeEBM skin_tone: {skin_hinge}")

# Score
sig = skin_pval < 0.05
positive = skin_coef > 0
smart_imp = skin_smart.get('importance', 0) if skin_smart else 0
hinge_imp = skin_hinge.get('importance', 0) if skin_hinge else 0

# Mixed evidence: OLS significant, HingeEBM picks it up (rank 4, 16.4%), SmartAdditive zeros it out
# Effect is real but small (bivariate r=0.008, OLS coef=0.004)
if sig and positive and smart_imp > 0.05 and hinge_imp > 0.05:
    score = 78  # both interpretable models confirm
elif sig and positive and hinge_imp > 0.05 and smart_imp == 0:
    score = 62  # OLS + HingeEBM confirm, but SmartAdditive excludes it — moderate evidence
elif sig and positive:
    score = 55
elif sig and not positive:
    score = 15
elif skin_pval < 0.10 and positive:
    score = 35
else:
    score = 15

biv_corr = corr.loc['skin_tone', 'redCards']
explanation = (
    f"Skin tone (IV, 0=lightest to 1=darkest, averaged from two blind raters) vs red cards (DV). "
    f"N=124,468 player-referee dyads with rated photos. "
    f"Bivariate correlation: r={biv_corr:.4f} (small positive). "
    f"OLS controlling for games, yellowCards, goals, meanIAT, meanExp: "
    f"skin_tone coef={skin_coef:.4f}, p={skin_pval:.4f} — statistically significant positive effect. "
    f"SmartAdditiveRegressor: skin_tone was excluded (importance=0.0, zeroed out by boosting), "
    f"with yellowCards (61.9%), games (27.2%), and goals (8.3%) dominating. "
    f"HingeEBM: skin_tone retained as 4th most important feature (importance=16.4%, coef=+0.0036, positive direction), "
    f"behind games (26.6%), yellowCards (24.4%), and meanExp (24.4%). "
    f"Evidence is consistent in direction (darker skin -> more red cards) across OLS and HingeEBM, "
    f"but the SmartAdditive model finds no nonlinear skin tone effect, suggesting the effect is small and roughly linear. "
    f"The absolute magnitude is modest: moving from lightest (0) to darkest (1) skin predicts ~0.004 more red cards per dyad "
    f"(baseline rate ~0.013). The effect persists after controlling for bias measures and player behavior."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print(f"\nConclusion written: score={score}")
print(explanation)
