# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "statsmodels",
#   "scikit-learn",
#   "agentic-imodels",
# ]
# ///

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# ── 1. Load & explore ──────────────────────────────────────────────────────
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nValue counts — sex:", df["sex"].value_counts().to_dict())
print("Value counts — help:", df["help"].value_counts().to_dict())
print("Value counts — hammer:", df["hammer"].value_counts().to_dict())

# Efficiency: nuts per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]
print("\nEfficiency stats:\n", df["efficiency"].describe())

# Encode categoricals
df["sex_bin"] = (df["sex"] == "m").astype(int)          # 1 = male
df["help_bin"] = (df["help"].str.strip().str.lower() == "y").astype(int)  # 1 = received help
hammer_dummies = pd.get_dummies(df["hammer"], prefix="hammer", drop_first=True).astype(float)
df = pd.concat([df, hammer_dummies], axis=1)

print("\nHelp distribution:", df["help_bin"].value_counts().to_dict())
print("Sex distribution:", df["sex_bin"].value_counts().to_dict())

# ── 2. Bivariate correlations ───────────────────────────────────────────────
print("\n--- Bivariate tests ---")
r_age, p_age = stats.pearsonr(df["age"], df["efficiency"])
print(f"Age vs efficiency: r={r_age:.3f}, p={p_age:.4f}")

grp_sex = [df.loc[df["sex_bin"]==0,"efficiency"], df.loc[df["sex_bin"]==1,"efficiency"]]
t_sex, p_sex = stats.ttest_ind(*grp_sex)
print(f"Sex vs efficiency: t={t_sex:.3f}, p={p_sex:.4f}, means f={grp_sex[0].mean():.4f} m={grp_sex[1].mean():.4f}")

grp_help = [df.loc[df["help_bin"]==0,"efficiency"], df.loc[df["help_bin"]==1,"efficiency"]]
t_help, p_help = stats.ttest_ind(*grp_help)
print(f"Help vs efficiency: t={t_help:.3f}, p={p_help:.4f}, means no={grp_help[0].mean():.4f} yes={grp_help[1].mean():.4f}")

# ── 3. OLS with controls ────────────────────────────────────────────────────
print("\n--- OLS with controls ---")
hammer_cols = [c for c in df.columns if c.startswith("hammer_")]
ctrl_cols = ["age", "sex_bin", "help_bin"] + hammer_cols
X_ols = sm.add_constant(df[ctrl_cols])
ols = sm.OLS(df["efficiency"], X_ols).fit()
print(ols.summary())

# ── 4. Interpretable models ─────────────────────────────────────────────────
feat_cols = ["age", "sex_bin", "help_bin"] + hammer_cols
X = df[feat_cols].astype(float)
y = df["efficiency"]

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    m = cls().fit(X, y)
    print(f"\n=== {cls.__name__} ===")
    print(m)

# ── 5. Summary & conclusion ─────────────────────────────────────────────────
age_coef = ols.params["age"]
age_p = ols.pvalues["age"]
sex_coef = ols.params["sex_bin"]
sex_p = ols.pvalues["sex_bin"]
help_coef = ols.params["help_bin"]
help_p = ols.pvalues["help_bin"]

print("\n--- Key OLS coefficients ---")
print(f"age:     coef={age_coef:.4f}, p={age_p:.4f}")
print(f"sex_bin: coef={sex_coef:.4f}, p={sex_p:.4f}")
print(f"help_bin:coef={help_coef:.4f}, p={help_p:.4f}")

# Calibrate Likert score for the combined question
# "How do age, sex, help influence efficiency?" — asking whether they collectively
# (or individually) influence efficiency. We score 0-100.
# Strong evidence for age, check sex and help.
significant_factors = []
if age_p < 0.05:
    significant_factors.append(f"age (β={age_coef:.3f}, p={age_p:.4f})")
if sex_p < 0.05:
    significant_factors.append(f"sex (β={sex_coef:.3f}, p={sex_p:.4f})")
if help_p < 0.05:
    significant_factors.append(f"help (β={help_coef:.3f}, p={help_p:.4f})")

n_sig = len(significant_factors)
if n_sig == 3:
    score = 90
elif n_sig == 2:
    score = 75
elif n_sig == 1:
    score = 60
else:
    # Check marginal
    marginal = sum(1 for p in [age_p, sex_p, help_p] if p < 0.10)
    score = 30 + marginal * 10

explanation = (
    f"Research question: How do age, sex, and help influence nut-cracking efficiency? "
    f"Efficiency is defined as nuts_opened/seconds. "
    f"OLS with hammer controls: age coef={age_coef:.4f} p={age_p:.4f}; "
    f"sex coef={sex_coef:.4f} p={sex_p:.4f}; "
    f"help coef={help_coef:.4f} p={help_p:.4f}. "
    f"Bivariate: age-efficiency r={r_age:.3f} p={p_age:.4f}; "
    f"sex t={t_sex:.3f} p={p_sex:.4f}; help t={t_help:.3f} p={p_help:.4f}. "
    f"Significant factors in controlled OLS: {significant_factors if significant_factors else 'none at p<0.05'}. "
    f"The question asks whether these variables collectively influence efficiency; "
    f"interpretable models (SmartAdditive, HingeEBM) were fitted and printed. "
    f"Score calibrated to {score} reflecting the number of significant factors and effect sizes."
)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
