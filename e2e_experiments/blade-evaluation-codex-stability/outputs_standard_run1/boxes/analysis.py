import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# 1) Load metadata and data
info = json.loads(Path("info.json").read_text())
research_question = info.get("research_questions", ["Unknown question"])[0]
df = pd.read_csv("boxes.csv")

required_cols = {"y", "gender", "age", "majority_first", "culture"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Clean/derive columns
for col in ["y", "gender", "age", "majority_first", "culture"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["y", "gender", "age", "majority_first", "culture"]).copy()

df["y_majority"] = (df["y"] == 2).astype(int)
df["age_group"] = pd.cut(
    df["age"],
    bins=[3.5, 6.5, 9.5, 12.5, 14.5],
    labels=["4-6", "7-9", "10-12", "13-14"],
    include_lowest=True,
)

# 2) Explore data: summaries, distributions, correlations
summary = df[["y", "y_majority", "age", "gender", "majority_first", "culture"]].describe()
y_dist = df["y"].value_counts().sort_index()
majority_rate_by_age = df.groupby("age")["y_majority"].mean()
majority_rate_by_culture = df.groupby("culture")["y_majority"].mean()
corr = df[["y", "y_majority", "age", "gender", "majority_first", "culture"]].corr(numeric_only=True)

print("Research question:")
print(research_question)
print("\nRows:", len(df))
print("\nSummary stats:\n", summary)
print("\nOutcome distribution (y):\n", y_dist)
print("\nMajority-choice rate by age:\n", majority_rate_by_age)
print("\nMajority-choice rate by culture:\n", majority_rate_by_culture)
print("\nCorrelation matrix:\n", corr)

# 3) Statistical tests
# 3a) Age-majority correlation
r_age, p_age_corr = stats.pearsonr(df["age"], df["y_majority"])

# 3b) Chi-square test for age group and majority choice
contingency_age = pd.crosstab(df["age_group"], df["y_majority"])
chi2_age, p_chi_age, _, _ = stats.chi2_contingency(contingency_age)

# 3c) Logistic regression for majority choice
logit_base = smf.logit("y_majority ~ age + gender + majority_first + C(culture)", data=df).fit(disp=0)
logit_int = smf.logit("y_majority ~ age + gender + majority_first + C(culture) + age:C(culture)", data=df).fit(disp=0)

llr_stat = 2 * (logit_int.llf - logit_base.llf)
df_diff = int(logit_int.df_model - logit_base.df_model)
p_lr_interaction = stats.chi2.sf(llr_stat, df_diff)

# 3d) OLS (linear probability style) for interpretable ANOVA-style table
ols_int = smf.ols("y_majority ~ age + gender + majority_first + C(culture) + age:C(culture)", data=df).fit()
anova_tbl = anova_lm(ols_int, typ=2)

print("\nStatistical tests:")
print(f"Pearson r(age, majority_choice) = {r_age:.4f}, p = {p_age_corr:.4g}")
print(f"Chi-square(age_group x majority_choice): chi2 = {chi2_age:.4f}, p = {p_chi_age:.4g}")
print(f"Logit base age coef = {logit_base.params['age']:.4f}, p = {logit_base.pvalues['age']:.4g}")
print(
    f"Likelihood-ratio test for age*culture interaction: chi2 = {llr_stat:.4f}, "
    f"df = {df_diff}, p = {p_lr_interaction:.4g}"
)
print("\nANOVA (OLS with interaction):\n", anova_tbl)

# 4) Interpretable models: sklearn + imodels
X = pd.get_dummies(df[["age", "gender", "majority_first", "culture"]], columns=["culture"], drop_first=True)
y = df["y_majority"].values
feature_names = X.columns.tolist()

# Train/test split for simple out-of-sample interpretability sanity-check
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lin = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.001, max_iter=20000).fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42).fit(X_train, y_train)

lin_coefs = pd.Series(lin.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
ridge_coefs = pd.Series(ridge.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
lasso_coefs = pd.Series(lasso.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
tree_importance = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)

y_pred_tree = tree.predict(X_test)
if hasattr(tree, "predict_proba"):
    y_prob_tree = tree.predict_proba(X_test)[:, 1]
    tree_auc = roc_auc_score(y_test, y_prob_tree)
else:
    tree_auc = np.nan
tree_acc = accuracy_score(y_test, y_pred_tree)

print("\nLinearRegression top coefficients:\n", lin_coefs.head(10))
print("\nRidge top coefficients:\n", ridge_coefs.head(10))
print("\nLasso top coefficients:\n", lasso_coefs.head(10))
print("\nDecisionTree feature importances:\n", tree_importance.head(10))
print(f"\nDecisionTree performance: accuracy={tree_acc:.3f}, auc={safe_float(tree_auc):.3f}")

imodels_results = {}
try:
    from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

    # RuleFit
    try:
        rf = RuleFitRegressor(random_state=42)
    except TypeError:
        rf = RuleFitRegressor()
    try:
        rf.fit(X_train.values, y_train, feature_names=feature_names)
        if hasattr(rf, "get_rules"):
            rules_df = rf.get_rules()
        else:
            rules_df = rf._get_rules()
        rules_df = rules_df[rules_df["coef"] != 0].copy()
        rules_df["abs_coef"] = rules_df["coef"].abs()
        top_rules = rules_df.sort_values("abs_coef", ascending=False).head(8)
        imodels_results["rulefit_top_rules"] = top_rules[["rule", "coef", "support"]].to_dict("records")
    except Exception as e:
        imodels_results["rulefit_error"] = repr(e)

    # FIGS
    try:
        figs = FIGSRegressor(random_state=42)
    except TypeError:
        figs = FIGSRegressor()
    try:
        figs.fit(X_train.values, y_train, feature_names=feature_names)
        if hasattr(figs, "feature_importances_"):
            figs_importance = (
                pd.Series(figs.feature_importances_, index=feature_names)
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            )
        else:
            figs_importance = None
        imodels_results["figs_feature_importance"] = figs_importance
    except Exception as e:
        imodels_results["figs_error"] = repr(e)

    # HSTree
    try:
        hst = HSTreeRegressor(random_state=42)
    except TypeError:
        hst = HSTreeRegressor()
    try:
        hst.fit(X_train.values, y_train, feature_names=feature_names)
        hst_importance = None
        if hasattr(hst, "feature_importances_"):
            hst_importance = (
                pd.Series(hst.feature_importances_, index=feature_names)
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            )
        elif hasattr(hst, "estimator_") and hasattr(hst.estimator_, "feature_importances_"):
            hst_importance = (
                pd.Series(hst.estimator_.feature_importances_, index=feature_names)
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            )
        imodels_results["hst_feature_importance"] = hst_importance
    except Exception as e:
        imodels_results["hst_error"] = repr(e)

    print("\nimodels RuleFit top rules:")
    for r in imodels_results.get("rulefit_top_rules", []):
        print(r)
    if "rulefit_error" in imodels_results:
        print("RuleFit error:", imodels_results["rulefit_error"])
    print("\nimodels FIGS feature importance:\n", imodels_results.get("figs_feature_importance"))
    if "figs_error" in imodels_results:
        print("FIGS error:", imodels_results["figs_error"])
    print("\nimodels HSTree feature importance:\n", imodels_results.get("hst_feature_importance"))
    if "hst_error" in imodels_results:
        print("HSTree error:", imodels_results["hst_error"])

except Exception as e:
    print("\nimodels import failed:", repr(e))

# 5) Interpret and produce Likert response
age_coef = float(logit_base.params.get("age", np.nan))
age_p = float(logit_base.pvalues.get("age", np.nan))

# Approximate odds-ratio per additional year of age
odds_ratio_age = float(np.exp(age_coef)) if np.isfinite(age_coef) else np.nan

age_sig = np.isfinite(age_p) and age_p < 0.05
age_positive = np.isfinite(age_coef) and age_coef > 0
interaction_sig = np.isfinite(p_lr_interaction) and p_lr_interaction < 0.05
chi_age_sig = np.isfinite(p_chi_age) and p_chi_age < 0.05

score = 50
score += 25 if age_sig else -25
score += 10 if age_positive else -10
score += 10 if chi_age_sig else -5
score += 10 if interaction_sig else -5

# Small boost if age is one of the most important tree features
if "age" in tree_importance.index[:3]:
    score += 5

score = int(np.clip(round(score), 0, 100))

if age_sig and age_positive:
    main_effect_text = "majority-choice increased with age, supporting developmental growth in majority reliance"
elif age_sig and not age_positive:
    main_effect_text = "majority-choice decreased with age, indicating reduced majority reliance with development"
else:
    main_effect_text = "there was no statistically significant age trend in majority-choice"

explanation = (
    f"Question: {research_question} "
    f"Using 629 children, {main_effect_text} "
    f"(logistic coef={age_coef:.3f}, OR/year={odds_ratio_age:.3f}, p={age_p:.3g}; "
    f"Pearson r={r_age:.3f}, p={p_age_corr:.3g}; chi-square age-group p={p_chi_age:.3g}). "
    f"Age-by-culture interaction test p={p_lr_interaction:.3g}, indicating "
    f"{'significant heterogeneity across cultures' if interaction_sig else 'limited evidence that age slopes differ by culture'}. "
    f"Interpretable models (linear coefficients, decision-tree importance, and imodels rule/tree models) were used to assess feature relationships."
)

conclusion = {"response": score, "explanation": explanation}
Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=True))

print("\nWrote conclusion.txt:")
print(json.dumps(conclusion, indent=2))
