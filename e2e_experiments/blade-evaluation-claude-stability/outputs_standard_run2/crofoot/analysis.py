import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('crofoot.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df['win'].mean())

# Derived features
df['rel_size'] = df['n_focal'] / df['n_other']          # >1 means focal is larger
df['size_diff'] = df['n_focal'] - df['n_other']
df['rel_males'] = df['m_focal'] / df['m_other']
df['location_adv'] = df['dist_other'] - df['dist_focal'] # positive = focal closer to home (home advantage)
df['dist_ratio'] = df['dist_focal'] / df['dist_other']   # <1 = focal closer to home

print("\n--- Correlation with win ---")
for col in ['rel_size', 'size_diff', 'rel_males', 'location_adv', 'dist_ratio']:
    r, p = stats.pointbiserialr(df[col], df['win'])
    print(f"  {col}: r={r:.3f}, p={p:.4f}")

# Logistic regression with statsmodels for p-values
print("\n--- Logistic Regression (statsmodels) ---")
X = df[['rel_size', 'location_adv']].copy()
X = sm.add_constant(X)
logit_model = sm.Logit(df['win'], X)
result = logit_model.fit(disp=0)
print(result.summary())

# Full model
print("\n--- Full Logistic Regression ---")
X2 = df[['rel_size', 'rel_males', 'location_adv']].copy()
X2 = sm.add_constant(X2)
logit_full = sm.Logit(df['win'], X2)
result_full = logit_full.fit(disp=0)
print(result_full.summary())

# t-tests: wins vs losses for each predictor
print("\n--- T-tests: Winners vs Losers ---")
wins = df[df['win'] == 1]
losses = df[df['win'] == 0]
for col in ['rel_size', 'size_diff', 'location_adv', 'dist_focal', 'dist_other']:
    t, p = stats.ttest_ind(wins[col], losses[col])
    print(f"  {col}: wins_mean={wins[col].mean():.3f}, losses_mean={losses[col].mean():.3f}, t={t:.3f}, p={p:.4f}")

# Home range analysis: is focal closer to home when they win?
df['focal_home'] = df['dist_focal'] < df['dist_other']  # focal closer to home
home_win = df[df['focal_home']]['win'].mean()
away_win = df[~df['focal_home']]['win'].mean()
print(f"\nWin rate when focal closer to home: {home_win:.3f} (n={df['focal_home'].sum()})")
print(f"Win rate when focal farther from home: {away_win:.3f} (n={(~df['focal_home']).sum()})")
chi2, p_chi = stats.chi2_contingency(pd.crosstab(df['focal_home'], df['win']))[:2]
print(f"Chi2 test p={p_chi:.4f}")

# Size advantage analysis
df['focal_larger'] = df['n_focal'] > df['n_other']
large_win = df[df['focal_larger']]['win'].mean()
small_win = df[~df['focal_larger']]['win'].mean()
print(f"\nWin rate when focal is larger: {large_win:.3f} (n={df['focal_larger'].sum()})")
print(f"Win rate when focal is NOT larger: {small_win:.3f} (n={(~df['focal_larger']).sum()})")
chi2_s, p_chi_s = stats.chi2_contingency(pd.crosstab(df['focal_larger'], df['win']))[:2]
print(f"Chi2 test p={p_chi_s:.4f}")

# Collect key results
rel_size_pval = result.pvalues['rel_size']
loc_adv_pval = result.pvalues['location_adv']
size_ttest_p = stats.ttest_ind(wins['rel_size'], losses['rel_size'])[1]
loc_ttest_p = stats.ttest_ind(wins['location_adv'], losses['location_adv'])[1]

print(f"\n=== SUMMARY ===")
print(f"Relative group size logit p-value: {rel_size_pval:.4f}")
print(f"Location advantage logit p-value: {loc_adv_pval:.4f}")
print(f"Relative size t-test p-value: {size_ttest_p:.4f}")
print(f"Location advantage t-test p-value: {loc_ttest_p:.4f}")

# Determine response score
# Both variables significant = strong yes (high score)
# One significant = moderate yes
# Neither = low score
both_sig = (rel_size_pval < 0.05) and (loc_adv_pval < 0.05)
size_sig = rel_size_pval < 0.05 or size_ttest_p < 0.05
loc_sig = loc_adv_pval < 0.05 or loc_ttest_p < 0.05

if both_sig:
    response = 85
    explanation = (
        f"Both relative group size (logit p={rel_size_pval:.4f}) and contest location/home advantage "
        f"(logit p={loc_adv_pval:.4f}) significantly influence the probability of winning. "
        f"Focal groups that are relatively larger win more often (mean rel_size winners={wins['rel_size'].mean():.2f} vs losers={losses['rel_size'].mean():.2f}), "
        f"and groups closer to their home range center have a higher win probability "
        f"(location advantage: winners={wins['location_adv'].mean():.1f}m vs losers={losses['location_adv'].mean():.1f}m). "
        f"Both factors jointly shape contest outcomes."
    )
elif size_sig and loc_sig:
    response = 80
    explanation = (
        f"Both relative group size (p={min(rel_size_pval, size_ttest_p):.4f}) and contest location "
        f"(p={min(loc_adv_pval, loc_ttest_p):.4f}) show significant effects on winning probability. "
        f"Larger groups and groups closer to their home range center are more likely to win."
    )
elif size_sig:
    response = 65
    explanation = (
        f"Relative group size significantly predicts contest outcomes (p={min(rel_size_pval, size_ttest_p):.4f}), "
        f"but contest location does not reach significance (location logit p={loc_adv_pval:.4f}). "
        f"Larger groups win more often; location effect is weaker."
    )
elif loc_sig:
    response = 65
    explanation = (
        f"Contest location significantly predicts winning (p={min(loc_adv_pval, loc_ttest_p):.4f}), "
        f"but relative group size does not (p={min(rel_size_pval, size_ttest_p):.4f}). "
        f"Home advantage is key; group size effect is weaker."
    )
else:
    response = 30
    explanation = (
        f"Neither relative group size (p={min(rel_size_pval, size_ttest_p):.4f}) nor contest location "
        f"(p={min(loc_adv_pval, loc_ttest_p):.4f}) significantly predict contest outcomes in this dataset."
    )

print(f"\nFinal response: {response}")
print(f"Explanation: {explanation}")

import json
with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)
print("\nconclusion.txt written.")
