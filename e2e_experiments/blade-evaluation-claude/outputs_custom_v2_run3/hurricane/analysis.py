"""
Analysis of the hurricane dataset.
Research question: Hurricanes with more feminine names are perceived as less threatening
and hence lead to fewer precautionary measures by the general public.
(Proxy: feminine names -> more deaths because people underestimate the storm)
"""
# /// script
# dependencies = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn"]
# ///

import sys
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# ── 1. Load and explore ──────────────────────────────────────────────────────
df = pd.read_csv("hurricane.csv")
print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nSummary stats:")
print(df.describe())

# Key variables
# masfem: femininity index (higher = more feminine)
# alldeaths: total deaths
# gender_mf: binary (1=female, 0=male)
# Controls: min (pressure), category, ndam, year, wind

print("\nCorrelation masfem vs alldeaths:", df["masfem"].corr(df["alldeaths"]))
print("Correlation gender_mf vs alldeaths:", df["gender_mf"].corr(df["alldeaths"]))

# Bivariate t-test: female vs male deaths
female_deaths = df[df["gender_mf"] == 1]["alldeaths"]
male_deaths = df[df["gender_mf"] == 0]["alldeaths"]
t_stat, p_val = stats.ttest_ind(female_deaths, male_deaths)
print(f"\nBivariate t-test (female vs male deaths): t={t_stat:.3f}, p={p_val:.3f}")
print(f"Mean deaths female: {female_deaths.mean():.1f}, male: {male_deaths.mean():.1f}")

# Pearson correlation masfem vs alldeaths
r, p_r = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson r(masfem, alldeaths): r={r:.3f}, p={p_r:.3f}")

# ── 2. Classical OLS with controls ───────────────────────────────────────────
control_cols = ["min", "category", "ndam", "year", "wind"]
df_clean = df[["masfem", "alldeaths"] + control_cols].dropna()

X_ols = sm.add_constant(df_clean[["masfem"] + control_cols])
ols_model = sm.OLS(df_clean["alldeaths"], X_ols).fit()
print("\n=== OLS with controls ===")
print(ols_model.summary())

# Log-transformed deaths (more appropriate for skewed count)
df_clean2 = df_clean.copy()
df_clean2["log_deaths"] = np.log1p(df_clean2["alldeaths"])
X_ols2 = sm.add_constant(df_clean2[["masfem"] + control_cols])
ols_log = sm.OLS(df_clean2["log_deaths"], X_ols2).fit()
print("\n=== OLS (log deaths) with controls ===")
print(ols_log.summary())

# ── 3. Interpretable models ───────────────────────────────────────────────────
feature_cols = ["masfem", "min", "category", "ndam", "year", "wind"]
df_model = df[feature_cols + ["alldeaths"]].dropna()
X = df_model[feature_cols]
y = df_model["alldeaths"]

# Log-transform y for modeling
y_log = np.log1p(y)

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor()
smart.fit(X, y_log)
print(smart)

print("\n=== HingeEBMRegressor ===")
hinge_ebm = HingeEBMRegressor()
hinge_ebm.fit(X, y_log)
print(hinge_ebm)

print("\n=== WinsorizedSparseOLSRegressor ===")
winsor = WinsorizedSparseOLSRegressor()
winsor.fit(X, y_log)
print(winsor)

# ── 4. Calibrated conclusion ──────────────────────────────────────────────────
masfem_coef_ols = ols_model.params.get("masfem", 0)
masfem_pval_ols = ols_model.pvalues.get("masfem", 1)
masfem_coef_log = ols_log.params.get("masfem", 0)
masfem_pval_log = ols_log.pvalues.get("masfem", 1)

print("\n=== Summary ===")
print(f"OLS (deaths): masfem coef={masfem_coef_ols:.3f}, p={masfem_pval_ols:.3f}")
print(f"OLS (log deaths): masfem coef={masfem_coef_log:.3f}, p={masfem_pval_log:.3f}")

# Score calibration
# The research question asks if feminine names lead to MORE deaths (perceived as less threatening)
# 100 = strong yes (feminine names do lead to more deaths, supporting the hypothesis)
# 0 = strong no

# Check direction: positive coef on masfem -> more feminine = more deaths = supports hypothesis
direction_supports = masfem_coef_log > 0
significant_log = masfem_pval_log < 0.05
significant_raw = masfem_pval_ols < 0.05

print(f"\nDirection supports hypothesis (feminine -> more deaths): {direction_supports}")
print(f"Significant at 0.05 (raw deaths): {significant_raw}")
print(f"Significant at 0.05 (log deaths): {significant_log}")

# Bivariate correlation direction
bivariate_supports = r > 0
print(f"Bivariate correlation positive (r={r:.3f}, p={p_r:.3f}): {bivariate_supports}")

# Score reasoning
# - Bivariate: r=positive but p-value?
# - OLS controlled: direction and significance?
# Literature context: The original paper (Jung et al. 2014) was heavily criticized;
# many replications and meta-analyses found the effect vanishes with controls,
# particularly after removing Katrina/Audrey.
# We need to rely on the data.

if significant_log and direction_supports:
    score = 65  # moderate evidence
elif not significant_log and direction_supports and masfem_pval_log < 0.15:
    score = 40  # marginal, direction consistent
elif not significant_log and direction_supports:
    score = 25  # direction consistent but not significant
else:
    score = 15  # no clear support

explanation = (
    f"Research question: do feminine hurricane names lead to more deaths (via reduced precautionary behavior)? "
    f"Bivariate: r(masfem, alldeaths)={r:.3f} (p={p_r:.3f}); mean deaths female={female_deaths.mean():.1f} vs male={male_deaths.mean():.1f} (t-test p={p_val:.3f}). "
    f"OLS with controls (min pressure, category, normalized damage, year, wind): masfem coef={masfem_coef_ols:.3f} (p={masfem_pval_ols:.3f}). "
    f"OLS log-deaths with controls: masfem coef={masfem_coef_log:.3f} (p={masfem_pval_log:.3f}). "
    f"Interpretable models (SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor) fitted on feature set [masfem, min, category, ndam, year, wind]. "
    f"The effect of masfem on deaths is {'positive' if direction_supports else 'negative'} in direction, "
    f"{'significant' if significant_log else 'not significant'} after controlling for storm severity. "
    f"The literature (Jung et al. 2014 reanalyzed) found that controlling for damage/severity largely eliminates this effect; "
    f"the dominant predictors are storm intensity variables. "
    f"Overall, there is {'some' if direction_supports and masfem_pval_log < 0.15 else 'little'} evidence that feminine names lead to more deaths. "
    f"Score {score}/100 reflecting {'moderate' if score >= 40 else 'weak'} support for the hypothesis."
)

print("\n=== Final Score ===", score)
print("Explanation:", explanation)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("\nconclusion.txt written.")
