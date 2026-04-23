import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv("affairs.csv")
print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# --- Feature engineering ---
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

# --- Bivariate exploration ---
print("\n=== Bivariate: children vs affairs ===")
grp = df.groupby("children")["affairs"]
print(grp.describe())
yes_affairs = df.loc[df["children"] == "yes", "affairs"]
no_affairs = df.loc[df["children"] == "no", "affairs"]
t_stat, p_val = stats.ttest_ind(yes_affairs, no_affairs)
print(f"Mean affairs (children=yes): {yes_affairs.mean():.3f}")
print(f"Mean affairs (children=no):  {no_affairs.mean():.3f}")
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Correlation
corr_pearson, corr_p = stats.pearsonr(df["children_bin"], df["affairs"])
print(f"Pearson r(children, affairs): {corr_pearson:.3f}, p={corr_p:.4f}")

# --- Classical OLS with controls ---
control_cols = ["gender_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
X_ols = sm.add_constant(df[["children_bin"] + control_cols])
ols_model = sm.OLS(df["affairs"], X_ols).fit()
print("\n=== OLS with controls ===")
print(ols_model.summary())

# --- Interpretable models ---
from agentic_imodels import SmartAdditiveRegressor, HingeGAMRegressor, WinsorizedSparseOLSRegressor

feature_cols = ["children_bin", "gender_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
X = df[feature_cols]
y = df["affairs"]

print("\n=== SmartAdditiveRegressor ===")
sar = SmartAdditiveRegressor()
sar.fit(X, y)
print(sar)

print("\n=== HingeGAMRegressor ===")
hgam = HingeGAMRegressor()
hgam.fit(X, y)
print(hgam)

print("\n=== WinsorizedSparseOLSRegressor ===")
wols = WinsorizedSparseOLSRegressor()
wols.fit(X, y)
print(wols)

# --- Summarize findings ---
children_coef = ols_model.params["children_bin"]
children_pval = ols_model.pvalues["children_bin"]
children_ci = ols_model.conf_int().loc["children_bin"]

print(f"\n=== Key finding: children coefficient ===")
print(f"OLS beta (controlled): {children_coef:.4f}, p={children_pval:.4f}, CI=[{children_ci[0]:.3f}, {children_ci[1]:.3f}]")

# Calibrate Likert score
# Research question: "Does having children *decrease* engagement in extramarital affairs?"
# We check sign of children_bin coef (positive = more affairs with children; negative = fewer)
# Also check bivariate vs controlled direction

bivar_diff = yes_affairs.mean() - no_affairs.mean()  # positive means children -> more affairs

print(f"\nBivariate diff (yes-no): {bivar_diff:.3f}")
print(f"Controlled beta: {children_coef:.3f} (positive = more affairs, negative = fewer)")

# Write conclusion
# The question asks if children DECREASE affairs
# Need to check: is coefficient significantly negative?

if children_pval < 0.05 and children_coef < 0:
    # Significant decrease
    if children_coef < -0.5:
        score = 75
    else:
        score = 60
    direction = "decrease"
elif children_pval < 0.1 and children_coef < 0:
    score = 45
    direction = "weak/marginal decrease"
elif children_pval >= 0.05 and abs(children_coef) < 0.3:
    score = 25
    direction = "no significant effect"
elif children_pval < 0.05 and children_coef > 0:
    score = 10
    direction = "increase (opposite direction)"
else:
    score = 30
    direction = "inconsistent/unclear"

explanation = (
    f"Research question: Does having children decrease extramarital affairs? "
    f"Bivariate: mean affairs is {yes_affairs.mean():.2f} for those with children vs {no_affairs.mean():.2f} without "
    f"(diff={bivar_diff:.2f}, t={t_stat:.2f}, p={p_val:.3f}). "
    f"OLS with controls (gender, age, yearsmarried, religiousness, education, occupation, rating): "
    f"children_bin beta={children_coef:.3f}, p={children_pval:.4f}, 95% CI=[{children_ci[0]:.3f}, {children_ci[1]:.3f}]. "
    f"Direction assessed as: {direction}. "
    f"SmartAdditiveRegressor and HingeGAMRegressor provide shape and importance info. "
    f"Key confounders: rating (marriage satisfaction) and religiousness are typically strong predictors. "
    f"The bivariate pattern shows children=yes is associated with {'more' if bivar_diff>0 else 'fewer'} affairs, "
    f"but under controls the effect may change. "
    f"Calibrated Likert score (0=strong No, 100=strong Yes for decrease): {score}."
)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\n=== Conclusion ===")
print(json.dumps(result, indent=2))
print("\nWritten to conclusion.txt")
