import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agentic_imodels'))
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# ── 1. Load & explore ────────────────────────────────────────────────────────
df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nOutcome distribution:\n", df['y'].value_counts(normalize=True).sort_index())
print("\nCulture distribution:\n", df['culture'].value_counts().sort_index())

# Binary DV: did child choose majority (y==2)?
df['majority_choice'] = (df['y'] == 2).astype(int)
print("\nMajority choice rate:", df['majority_choice'].mean().round(3))

# Bivariate: majority choice by age
print("\nMajority choice rate by age:")
print(df.groupby('age')['majority_choice'].mean().round(3))

# Bivariate: majority choice by culture
print("\nMajority choice rate by culture:")
print(df.groupby('culture')['majority_choice'].mean().round(3))

# ── 2. Classical statistical test ───────────────────────────────────────────
# Logistic regression: age + controls (gender, majority_first, culture dummies)
culture_dummies = pd.get_dummies(df['culture'], prefix='cult', drop_first=True)
X_logit = pd.concat([df[['age', 'gender', 'majority_first']], culture_dummies], axis=1)
X_logit = sm.add_constant(X_logit.astype(float))
logit_model = sm.Logit(df['majority_choice'], X_logit).fit(disp=0)
print("\n=== Logistic Regression (majority_choice ~ age + gender + majority_first + culture) ===")
print(logit_model.summary())

# Age-only logistic (bivariate)
X_age_only = sm.add_constant(df[['age']].astype(float))
logit_age = sm.Logit(df['majority_choice'], X_age_only).fit(disp=0)
print("\n=== Bivariate Logistic (majority_choice ~ age) ===")
print(logit_age.summary())

# Correlation: age vs majority choice
r, p = stats.pointbiserialr(df['age'], df['majority_choice'])
print(f"\nPoint-biserial r(age, majority_choice) = {r:.3f}, p = {p:.4f}")

# Age × culture interaction
df['age_x_culture'] = df['age'] * df['culture']
X_inter = sm.add_constant(df[['age', 'culture', 'age_x_culture', 'gender', 'majority_first']].astype(float))
logit_inter = sm.Logit(df['majority_choice'], X_inter).fit(disp=0)
print("\n=== Logistic with age×culture interaction ===")
print(logit_inter.summary2())

# ── 3. Interpretable models ──────────────────────────────────────────────────
feature_cols = ['age', 'gender', 'majority_first', 'culture']
X = df[feature_cols].astype(float)
y = df['majority_choice'].astype(float)

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    m = cls()
    m.fit(X, y)
    print(f"\n=== {cls.__name__} ===")
    print(m)
    from sklearn.metrics import r2_score
    r2 = r2_score(y, m.predict(X))
    print(f"  R^2 (train): {r2:.3f}")

# ── 4. Per-culture age-majority correlations ─────────────────────────────────
print("\n=== Per-culture: correlation of age with majority choice ===")
for cult, grp in df.groupby('culture'):
    if len(grp) > 10:
        r_c, p_c = stats.pointbiserialr(grp['age'], grp['majority_choice'])
        n = len(grp)
        print(f"  culture={cult} n={n}  r={r_c:.3f}  p={p_c:.4f}")

# ── 5. Write conclusion ──────────────────────────────────────────────────────
age_coef = logit_model.params['age']
age_pval = logit_model.pvalues['age']
bivar_coef = logit_age.params['age']
bivar_pval = logit_age.pvalues['age']
interact_pval = logit_inter.pvalues['age_x_culture']
r_bivar, p_bivar = stats.pointbiserialr(df['age'], df['majority_choice'])

explanation = (
    f"The research question asks whether children's reliance on majority preference increases with age, "
    f"and whether this differs across cultural contexts. "
    f"The outcome is binary (chose majority vs. not). "
    f"Bivariate logistic regression: age coefficient = {bivar_coef:.3f}, p = {bivar_pval:.4f} "
    f"(point-biserial r = {r_bivar:.3f}, p = {p_bivar:.4f}). "
    f"Controlled model (age + gender + majority_first + culture dummies): "
    f"age coef = {age_coef:.3f}, p = {age_pval:.4f}. "
    f"Age×culture interaction p = {interact_pval:.4f}. "
    f"Interpretable models (SmartAdditiveRegressor and HingeEBMRegressor) were fitted on all features "
    f"to characterize effect direction, magnitude, and shape. "
    f"If age is significant (p < 0.05) in the controlled model with a positive coefficient, "
    f"children become more reliant on majority preference as they age; "
    f"interaction term addresses cultural moderation. "
    f"Per-culture correlations further assess cross-cultural consistency. "
    f"Overall: {'age shows a significant positive effect on majority choice' if age_pval < 0.05 and age_coef > 0 else 'age effect on majority choice is not clearly significant or positive'}, "
    f"and the interaction (age × culture) is {'significant' if interact_pval < 0.05 else 'not significant'}, "
    f"suggesting cultural context {'does' if interact_pval < 0.05 else 'does not'} moderate the age trend."
)

# Calibrate Likert score
if age_pval < 0.01 and age_coef > 0:
    score = 80
elif age_pval < 0.05 and age_coef > 0:
    score = 65
elif age_pval < 0.10 and age_coef > 0:
    score = 50
elif age_coef > 0:
    score = 35
else:
    score = 20

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print("\nWrote conclusion.txt")
print(json.dumps(result, indent=2))
