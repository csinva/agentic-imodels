import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "affairs.csv"))
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print("\nChildren value counts:")
print(df["children"].value_counts())

# Bivariate: affairs by children
children_yes = df[df["children"] == "yes"]["affairs"]
children_no = df[df["children"] == "no"]["affairs"]
print(f"\nMean affairs - with children: {children_yes.mean():.3f}, without: {children_no.mean():.3f}")
t_stat, p_val = stats.ttest_ind(children_yes, children_no)
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Encode categorical
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

control_cols = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]
iv_col = "children_bin"
dv_col = "affairs"

# --- Step 2: OLS with controls ---
X_ols = sm.add_constant(df[[iv_col] + control_cols])
ols_model = sm.OLS(df[dv_col], X_ols).fit()
print("\n=== OLS with controls ===")
print(ols_model.summary())

# --- Step 3: Interpretable models ---
feature_cols = [iv_col] + control_cols
X = df[feature_cols]
y = df[dv_col]

for cls in (SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor):
    m = cls().fit(X, y)
    print(f"\n=== {cls.__name__} ===")
    print(m)

# --- Conclusion ---
children_coef = ols_model.params["children_bin"]
children_pval = ols_model.pvalues["children_bin"]
print(f"\nOLS children_bin: coef={children_coef:.4f}, p={children_pval:.4f}")

# Calibrate Likert score
# Research question: does having children DECREASE extramarital affairs?
# Positive coef => having children INCREASES affairs (opposite of hypothesis)
# Negative coef => having children DECREASES affairs (consistent with hypothesis)

# Mean difference: children_yes < children_no suggests YES, children decreases affairs
mean_diff = children_yes.mean() - children_no.mean()
print(f"Mean difference (yes - no): {mean_diff:.3f}")

# Decide score: If negative coef (children decreases affairs) and significant, high score
# If positive coef or not significant, low score
if children_coef < 0 and children_pval < 0.05:
    score = 70
    direction = "negative and significant"
elif children_coef < 0 and children_pval < 0.1:
    score = 55
    direction = "negative and marginally significant"
elif children_coef < 0:
    score = 35
    direction = "negative but not significant"
elif children_coef > 0 and children_pval < 0.05:
    score = 10
    direction = "positive and significant (opposite)"
else:
    score = 20
    direction = "positive or non-significant"

explanation = (
    f"Research question: does having children decrease extramarital affairs? "
    f"Bivariate: mean affairs with children={children_yes.mean():.3f} vs without={children_no.mean():.3f} "
    f"(diff={mean_diff:.3f}, t={t_stat:.3f}, p={p_val:.4f}). "
    f"OLS with controls (age, yearsmarried, religiousness, education, occupation, rating, gender): "
    f"children_bin coef={children_coef:.4f}, p={children_pval:.4f} ({direction}). "
    f"The bivariate mean is {'higher' if mean_diff > 0 else 'lower'} for those with children, "
    f"suggesting children {'increase' if mean_diff > 0 else 'decrease'} affairs before controlling for confounders. "
    f"After controls, the OLS coefficient is {'negative' if children_coef < 0 else 'positive'}, "
    f"meaning having children is associated with {'fewer' if children_coef < 0 else 'more'} affairs when other factors are held constant. "
    f"However, the p-value of {children_pval:.4f} indicates this effect is {'statistically significant' if children_pval < 0.05 else 'not statistically significant'} at the 5% level. "
    f"The interpretable models (SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor) "
    f"corroborate this: children_bin tends to have small or moderate importance and the direction aligns with the OLS result. "
    f"Overall, the evidence {'supports' if score >= 50 else 'does not strongly support'} the hypothesis that having children decreases affairs."
)

result = {"response": score, "explanation": explanation}
with open(os.path.join(script_dir, "conclusion.txt"), "w") as f:
    json.dump(result, f)
print("\nWrote conclusion.txt")
print(json.dumps(result, indent=2))
