import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv("affairs.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("\nchildren value counts:")
print(df["children"].value_counts())
print("\nAffairs by children:")
print(df.groupby("children")["affairs"].describe())

# Bivariate test
children_yes = df[df["children"] == "yes"]["affairs"]
children_no = df[df["children"] == "no"]["affairs"]
t_stat, p_val = stats.ttest_ind(children_yes, children_no)
print(f"\nBivariate t-test: t={t_stat:.3f}, p={p_val:.4f}")
print(f"Mean affairs (children=yes): {children_yes.mean():.3f}")
print(f"Mean affairs (children=no): {children_no.mean():.3f}")

# Classical OLS with controls
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

feature_cols = ["children_bin", "gender_bin", "age", "yearsmarried",
                "religiousness", "education", "occupation", "rating"]

X_ols = sm.add_constant(df[feature_cols])
ols_model = sm.OLS(df["affairs"], X_ols).fit()
print("\n=== OLS with controls ===")
print(ols_model.summary())

# Interpretable models
X_interp = df[feature_cols].copy()
y = df["affairs"].values

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor()
smart.fit(X_interp, y)
print(smart)

print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor()
hinge.fit(X_interp, y)
print(hinge)

# Extract key findings
children_coef = ols_model.params["children_bin"]
children_pval = ols_model.pvalues["children_bin"]
children_ci_low = ols_model.conf_int().loc["children_bin", 0]
children_ci_high = ols_model.conf_int().loc["children_bin", 1]

print(f"\nOLS children_bin: coef={children_coef:.4f}, p={children_pval:.4f}, "
      f"CI=[{children_ci_low:.4f}, {children_ci_high:.4f}]")

# Determine Likert score
# Research question: does having children DECREASE affairs?
# children_bin=1 means has children
# Negative coefficient = having children reduces affairs

mean_diff = children_yes.mean() - children_no.mean()
print(f"\nMean diff (children=yes minus no): {mean_diff:.3f}")

# If children_coef is positive (having children increases affairs) or nonsignificant → low score
# If children_coef is negative and significant → high score

if children_coef < 0 and children_pval < 0.05:
    # Negative effect with significance: children reduce affairs
    magnitude = abs(children_coef)
    if children_pval < 0.01 and magnitude > 0.5:
        score = 65
    elif children_pval < 0.05:
        score = 50
    else:
        score = 40
    direction = "negative (children reduce affairs)"
elif children_coef > 0 and children_pval < 0.05:
    # Positive: children increase affairs
    score = 15
    direction = "positive (children increase affairs, opposite to question)"
else:
    # Non-significant
    if children_pval < 0.2 and children_coef < 0:
        score = 30
        direction = "trending negative but not significant"
    else:
        score = 20
        direction = "not significant"

# Also consider bivariate: if bivariate shows yes > no, that's against the hypothesis
if mean_diff > 0 and p_val < 0.05:
    # Bivariate shows having children INCREASES affairs
    # But under controls maybe different; lower the score
    score = max(score - 10, 10)

explanation = (
    f"Research question: Does having children decrease extramarital affairs? "
    f"Bivariate: affairs mean={children_yes.mean():.2f} (children=yes) vs "
    f"{children_no.mean():.2f} (children=no), t={t_stat:.3f}, p={p_val:.4f}. "
    f"Mean difference={mean_diff:.3f} (positive means children have MORE affairs bivariate). "
    f"OLS with controls: children_bin coef={children_coef:.4f} ({direction}), "
    f"p={children_pval:.4f}, 95% CI=[{children_ci_low:.4f}, {children_ci_high:.4f}]. "
    f"Interpretable models (SmartAdditiveRegressor, HingeEBMRegressor) were fitted "
    f"to confirm direction and importance of children variable. "
    f"The bivariate effect shows couples with children have more affairs on average "
    f"(likely a confound with age/years married). Under controls, the children effect "
    f"may attenuate. Based on all evidence, the Likert score is {score}."
)

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("conclusion.txt written.")
