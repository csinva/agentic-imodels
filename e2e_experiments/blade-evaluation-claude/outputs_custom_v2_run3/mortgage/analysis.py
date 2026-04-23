import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# --- Load data ---
df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# --- Research question: does gender (female) affect mortgage approval (accept)? ---
print("\n=== Bivariate: female vs accept ===")
print(df.groupby("female")["accept"].mean())

# Chi-square test
ct = pd.crosstab(df["female"], df["accept"])
chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
print(f"Chi2={chi2:.4f}, p={p_chi2:.4f}")

# --- Classical logistic regression with controls ---
control_cols = ["black", "housing_expense_ratio", "self_employed", "married",
                "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
                "loan_to_value", "denied_PMI"]

# Drop rows with any NaN in the relevant columns
cols_needed = ["female", "accept"] + control_cols
df_clean = df[cols_needed].dropna()
print(f"\nRows after dropna: {len(df_clean)}")

X_logit = sm.add_constant(df_clean[["female"] + control_cols])
logit_model = sm.Logit(df_clean["accept"], X_logit).fit(disp=0)
print("\n=== Logistic Regression Summary ===")
print(logit_model.summary2())

female_coef = logit_model.params["female"]
female_pval = logit_model.pvalues["female"]
print(f"\nfemale coef={female_coef:.4f}, p={female_pval:.4f}")

# --- Interpretable regressors (fit on 0/1 outcome) ---
feature_cols = ["female", "black", "housing_expense_ratio", "self_employed", "married",
                "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
                "loan_to_value", "denied_PMI"]

X = df_clean[feature_cols]
y = df_clean["accept"]

print("\n=== SmartAdditiveRegressor ===")
m1 = SmartAdditiveRegressor()
m1.fit(X, y)
print(m1)

print("\n=== HingeEBMRegressor ===")
m2 = HingeEBMRegressor()
m2.fit(X, y)
print(m2)

# --- Synthesize conclusion ---
# bivariate: compare approval rates
female_approve = df.groupby("female")["accept"].mean()
bivariate_diff = female_approve[1.0] - female_approve[0.0]
print(f"\nBivariate approval rate diff (female - male): {bivariate_diff:.4f}")

# Decision on Likert score
# female coef in logit (positive = higher accept if female)
# p-value significance
# whether SmartAdditive and HingeEBM include female as a notable feature
# Based on mortgage discrimination literature, gender effect is typically weak/absent
# while race is strong

if female_pval < 0.05:
    if abs(female_coef) > 0.3:
        score = 65
    else:
        score = 45
else:
    if abs(bivariate_diff) < 0.03:
        score = 15
    else:
        score = 25

explanation = (
    f"The research question asks whether gender (female=1) affects mortgage approval. "
    f"Bivariate analysis shows approval rates of {female_approve.get(0.0, float('nan')):.3f} (male) "
    f"and {female_approve.get(1.0, float('nan')):.3f} (female), a difference of {bivariate_diff:.3f}. "
    f"Chi-squared test: chi2={chi2:.3f}, p={p_chi2:.4f}. "
    f"Logistic regression with full controls (credit scores, debt ratios, race, marital status, etc.) "
    f"yields female coefficient={female_coef:.4f} (p={female_pval:.4f}). "
    f"SmartAdditiveRegressor and HingeEBMRegressor were also fitted on the 0/1 accept outcome; "
    f"their printed forms characterize the direction, magnitude, and shape of each feature. "
    f"If the female coefficient in the controlled model is statistically significant and positive, "
    f"there is evidence that being female is associated with higher acceptance; if non-significant "
    f"or zeroed out by the interpretable models, gender does not meaningfully affect approval. "
    f"The calibrated Likert score reflects: p={'<0.05 significant' if female_pval < 0.05 else '>0.05 non-significant'}, "
    f"coefficient direction={'positive (females more likely approved)' if female_coef > 0 else 'negative (females less likely approved)'}, "
    f"bivariate diff={bivariate_diff:.3f}."
)

result = {"response": score, "explanation": explanation}
print("\n=== Final Result ===")
print(json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclustion.txt written.")
