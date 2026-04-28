import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


def clamp_int(x, low=0, high=100):
    return int(max(low, min(high, round(x))))


def main():
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["No question found"])[0]

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Basic EDA
    print("Research question:", question)
    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False))

    binary_cols = [c for c in df.columns if df[c].dropna().nunique() <= 2]
    print("\nBinary variable distributions:")
    for col in binary_cols:
        vc = df[col].value_counts(dropna=False, normalize=True).sort_index()
        print(f"{col}:\n{vc}\n")

    corr_with_accept = df[numeric_cols].corr(numeric_only=True)["accept"].sort_values(ascending=False)
    print("\nCorrelations with accept:")
    print(corr_with_accept)

    # Focused gender-outcome EDA
    fx = df[["female", "accept"]].dropna()
    group_rates = fx.groupby("female")["accept"].agg(["mean", "count"])
    print("\nApproval rate by female:")
    print(group_rates)

    # Statistical tests
    female_accept = fx.loc[fx["female"] == 1, "accept"]
    male_accept = fx.loc[fx["female"] == 0, "accept"]

    ttest = stats.ttest_ind(female_accept, male_accept, equal_var=False, nan_policy="omit")
    corr_test = stats.pointbiserialr(fx["female"], fx["accept"])

    contingency = pd.crosstab(fx["female"], fx["accept"])
    chi2, chi2_p, chi2_dof, _ = chi2_contingency(contingency)

    print("\nStatistical tests for female vs accept:")
    print(f"Welch t-test p-value: {ttest.pvalue:.6g}")
    print(f"Point-biserial correlation: r={corr_test.statistic:.6g}, p={corr_test.pvalue:.6g}")
    print(f"Chi-square p-value: {chi2_p:.6g}, chi2={chi2:.6g}, dof={chi2_dof}")

    predictors = [c for c in df.columns if c not in ["accept", "deny"]]
    reg_df = df[["accept"] + predictors].dropna().copy()

    # OLS (linear probability models) for interpretability and p-values
    ols_unadj = sm.OLS(
        reg_df["accept"], sm.add_constant(reg_df[["female"]], has_constant="add")
    ).fit(cov_type="HC3")

    ols_adj = sm.OLS(
        reg_df["accept"], sm.add_constant(reg_df[predictors], has_constant="add")
    ).fit(cov_type="HC3")

    # Logistic model as robustness check
    logit_adj = sm.Logit(
        reg_df["accept"], sm.add_constant(reg_df[predictors], has_constant="add")
    ).fit(disp=False)

    # ANOVA on linear model
    formula = "accept ~ " + " + ".join(predictors)
    anova_model = smf.ols(formula=formula, data=reg_df).fit()
    anova_tbl = sm.stats.anova_lm(anova_model, typ=2)

    print("\nOLS (unadjusted):")
    print(ols_unadj.summary().tables[1])
    print("\nOLS (adjusted):")
    print(ols_adj.summary().tables[1])
    print("\nLogit (adjusted) female coefficient + p-value:")
    print({"coef": float(logit_adj.params["female"]), "p": float(logit_adj.pvalues["female"])})
    print("\nANOVA (type II):")
    print(anova_tbl)

    # Interpretable sklearn models
    X = df[predictors]
    y = df["accept"]
    female_idx = predictors.index("female")

    linear_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.002, max_iter=20000),
    }
    lin_results = {}

    for name, model in linear_models.items():
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X, y)
        female_coef = float(pipe.named_steps["model"].coef_[female_idx])
        lin_results[name] = {"female_coef_std": female_coef}

    tree_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ]
    )
    tree_pipe.fit(X, y)
    tree_model = tree_pipe.named_steps["model"]
    tree_female_importance = float(tree_model.feature_importances_[female_idx])
    tree_rules = export_text(tree_model, feature_names=predictors)

    print("\nSklearn interpretable models:")
    print("Linear coefficients (standardized):", lin_results)
    print("Decision tree female importance:", tree_female_importance)
    print("Decision tree rules:\n", tree_rules)

    # Interpretable imodels models
    X_imp = SimpleImputer(strategy="median").fit_transform(X)

    rulefit = RuleFitRegressor(random_state=0, tree_size=3, n_estimators=80)
    rulefit.fit(X_imp, y, feature_names=predictors)
    if hasattr(rulefit, "_get_rules"):
        rf_rules = rulefit._get_rules()
    else:
        rf_rules = pd.DataFrame()

    female_linear_coef_rulefit = np.nan
    female_nonlinear_rules = 0
    if not rf_rules.empty and "rule" in rf_rules.columns:
        female_linear = rf_rules[(rf_rules["rule"] == "female") & (rf_rules["type"] == "linear")]
        if len(female_linear) > 0:
            female_linear_coef_rulefit = float(female_linear["coef"].iloc[0])
        female_nonlinear_rules = int(
            rf_rules[
                rf_rules["rule"].astype(str).str.contains("female", na=False)
                & (rf_rules["rule"] != "female")
                & (rf_rules["coef"] != 0)
            ].shape[0]
        )

    figs = FIGSRegressor(max_rules=12, random_state=0)
    figs.fit(X_imp, y, feature_names=predictors)
    figs_female_importance = float(figs.feature_importances_[female_idx])
    figs_text = str(figs)

    hstree = HSTreeRegressor(max_leaf_nodes=8, random_state=0)
    hstree.fit(X_imp, y, feature_names=predictors)
    if hasattr(hstree, "estimator_") and hasattr(hstree.estimator_, "feature_importances_"):
        hstree_female_importance = float(hstree.estimator_.feature_importances_[female_idx])
        hstree_rules = export_text(hstree.estimator_, feature_names=predictors)
    else:
        hstree_female_importance = float("nan")
        hstree_rules = "N/A"

    print("\nimodels interpretable models:")
    print(
        {
            "RuleFit_female_linear_coef": female_linear_coef_rulefit,
            "RuleFit_nonlinear_rules_with_female": female_nonlinear_rules,
            "FIGS_female_importance": figs_female_importance,
            "HSTree_female_importance": hstree_female_importance,
        }
    )
    print("FIGS model text:\n", figs_text)
    print("HSTree rules:\n", hstree_rules)

    # Build final conclusion
    unadj_diff = float(female_accept.mean() - male_accept.mean())
    unadj_p = float(chi2_p)
    adj_coef = float(ols_adj.params["female"])
    adj_p = float(ols_adj.pvalues["female"])
    logit_coef = float(logit_adj.params["female"])
    logit_p = float(logit_adj.pvalues["female"])
    anova_p = float(anova_tbl.loc["female", "PR(>F)"]) if "female" in anova_tbl.index else np.nan

    # Likert scoring: significant adjusted effect with weak unadjusted difference => moderate evidence of effect
    score = 50
    if adj_p < 0.01:
        score += 20
    elif adj_p < 0.05:
        score += 15
    elif adj_p < 0.10:
        score += 8
    else:
        score -= 15

    if logit_p < 0.05:
        score += 8
    if unadj_p > 0.10:
        score -= 10
    if abs(unadj_diff) < 0.01:
        score -= 5
    if abs(adj_coef) < 0.02:
        score -= 5
    if tree_female_importance == 0 and figs_female_importance == 0 and female_nonlinear_rules == 0:
        score -= 3

    response_score = clamp_int(score)

    explanation = (
        f"Raw approval rates are nearly identical by gender (female-male difference={unadj_diff:.4f}, "
        f"chi-square p={unadj_p:.3g}; t-test p={ttest.pvalue:.3g}), so unadjusted evidence of a gender gap is weak. "
        f"After adjusting for credit and risk controls, female has a small positive association with approval "
        f"(OLS coef={adj_coef:.4f}, p={adj_p:.3g}; logit coef={logit_coef:.3f}, p={logit_p:.3g}; "
        f"ANOVA p={anova_p:.3g}). Interpretable linear models (sklearn + RuleFit) also give small positive female coefficients, "
        f"while tree/rule structures give low female split importance, indicating a modest but not dominant effect."
    )

    result = {"response": response_score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))

    print("\nFinal conclusion JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
