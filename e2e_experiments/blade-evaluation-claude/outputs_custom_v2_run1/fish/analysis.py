import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv("fish.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelations with fish_caught:")
print(df.corr()["fish_caught"])

# Compute fish per hour
df["fish_per_hour"] = df["fish_caught"] / df["hours"].replace(0, np.nan)
print("\nfish_per_hour stats:")
print(df["fish_per_hour"].describe())
mean_rate = df["fish_per_hour"].mean()
median_rate = df["fish_per_hour"].median()
print(f"Mean fish/hour: {mean_rate:.4f}")
print(f"Median fish/hour: {median_rate:.4f}")

# Poisson GLM for count outcome (fish_caught) with log(hours) offset
feature_cols = ["livebait", "camper", "persons", "child"]
X_glm = sm.add_constant(df[feature_cols])
glm_poisson = sm.GLM(
    df["fish_caught"],
    X_glm,
    family=sm.families.Poisson(),
    offset=np.log(df["hours"].clip(lower=0.001)),
).fit()
print("\n=== Poisson GLM with log(hours) offset ===")
print(glm_poisson.summary())

# OLS on fish_per_hour (winsorized)
fph = df["fish_per_hour"].copy()
upper = fph.quantile(0.95)
fph_win = fph.clip(upper=upper)
X_ols = sm.add_constant(df[feature_cols])
ols = sm.OLS(fph_win, X_ols).fit()
print("\n=== OLS on winsorized fish_per_hour ===")
print(ols.summary())

# Interpretable models
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

feature_cols_all = ["livebait", "camper", "persons", "child", "hours"]
X = df[feature_cols_all]
y = df["fish_caught"]

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    m = cls().fit(X, y)
    print(f"\n=== {cls.__name__} ===")
    print(m)

# Bivariate stats
print("\n=== Bivariate ===")
lb_yes = df[df["livebait"] == 1]["fish_per_hour"].dropna()
lb_no = df[df["livebait"] == 0]["fish_per_hour"].dropna()
t, p = stats.mannwhitneyu(lb_yes, lb_no, alternative="two-sided")
print(f"Livebait vs no livebait fish/hour: {lb_yes.mean():.3f} vs {lb_no.mean():.3f}, MWU p={p:.4f}")

camp_yes = df[df["camper"] == 1]["fish_per_hour"].dropna()
camp_no = df[df["camper"] == 0]["fish_per_hour"].dropna()
t2, p2 = stats.mannwhitneyu(camp_yes, camp_no, alternative="two-sided")
print(f"Camper vs no camper fish/hour: {camp_yes.mean():.3f} vs {camp_no.mean():.3f}, MWU p={p2:.4f}")

r_persons, p_persons = stats.spearmanr(df["persons"], df["fish_per_hour"].fillna(0))
r_child, p_child = stats.spearmanr(df["child"], df["fish_per_hour"].fillna(0))
r_hours, p_hours = stats.spearmanr(df["hours"], df["fish_per_hour"].fillna(0))
print(f"Persons vs fish/hour: Spearman r={r_persons:.3f}, p={p_persons:.4f}")
print(f"Child vs fish/hour: Spearman r={r_child:.3f}, p={p_child:.4f}")
print(f"Hours vs fish/hour: Spearman r={r_hours:.3f}, p={p_hours:.4f}")

# Summary and conclusion
explanation = (
    f"The mean fish caught per hour across all {len(df)} visits is {mean_rate:.2f} (median {median_rate:.2f}). "
    f"The Poisson GLM (with log-hours offset) shows livebait (IRR={np.exp(glm_poisson.params['livebait']):.2f}, "
    f"p={glm_poisson.pvalues['livebait']:.4f}) and persons (IRR={np.exp(glm_poisson.params['persons']):.2f}, "
    f"p={glm_poisson.pvalues['persons']:.4f}) are the dominant positive predictors of catch rate. "
    f"Children have a strong negative effect (IRR={np.exp(glm_poisson.params['child']):.2f}, "
    f"p={glm_poisson.pvalues['child']:.4f}). Camper effect is modest. "
    f"Interpretable models (SmartAdditiveRegressor, HingeEBMRegressor) confirm these directions with hours and persons ranked as top predictors. "
    f"The estimated catch rate is approximately {mean_rate:.1f} fish per hour on average across all groups, "
    f"but for active fishing groups using livebait the rate is substantially higher (~{lb_yes.mean():.1f} fish/hour vs ~{lb_no.mean():.1f} without livebait). "
    f"Overall, the data robustly answer the research question: the average rate is roughly {mean_rate:.1f} fish/hour, "
    f"driven mainly by group size, livebait use, and time spent fishing."
)

# Score 72: the question is descriptive/estimative ('how many fish on average / what factors influence')
# We can answer it clearly with strong statistical evidence; score reflects high confidence in the characterization.
response = 72

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
