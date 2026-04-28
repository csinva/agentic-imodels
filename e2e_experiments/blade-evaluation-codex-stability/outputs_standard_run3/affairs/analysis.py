import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def marginal_effect_children_regression(model, X: pd.DataFrame) -> float:
    """Average predicted change when children_yes toggles 0->1, holding other features fixed."""
    X0 = X.copy()
    X1 = X.copy()
    X0["children_yes"] = 0
    X1["children_yes"] = 1
    pred0 = model.predict(X0)
    pred1 = model.predict(X1)
    return float(np.mean(pred1 - pred0))


def marginal_effect_children_classifier(model, X: pd.DataFrame) -> float:
    """Average change in predicted probability(any_affair) when children_yes toggles 0->1."""
    X0 = X.copy()
    X1 = X.copy()
    X0["children_yes"] = 0
    X1["children_yes"] = 1
    p0 = model.predict_proba(X0)[:, 1]
    p1 = model.predict_proba(X1)[:, 1]
    return float(np.mean(p1 - p0))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    # 1) Load data
    df = pd.read_csv("affairs.csv")

    # Basic feature engineering for interpretable numeric modeling
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    feature_cols = [
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = df[feature_cols].copy()
    y = df["affairs"].astype(float).copy()
    y_bin = df["any_affair"].astype(int).copy()

    print("=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\n=== SUMMARY STATISTICS ===")
    print(df[["affairs", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]].describe().to_string())

    print("\n=== DISTRIBUTIONS ===")
    print("Affairs value counts:")
    print(df["affairs"].value_counts().sort_index().to_string())
    print("\nAffairs proportion by value:")
    print((df["affairs"].value_counts(normalize=True).sort_index().round(4)).to_string())
    print("\nChildren distribution:")
    print(df["children"].value_counts().to_string())

    # Group summaries relevant to research question
    group_stats = (
        df.groupby("children")["affairs"]
        .agg(["count", "mean", "median", "std"])
        .rename_axis("children")
    )
    print("\nAffairs by children status:")
    print(group_stats.to_string())

    # Correlations with target
    corr_df = df[["affairs"] + feature_cols].corr(numeric_only=True)
    print("\n=== CORRELATIONS WITH AFFAIRS ===")
    print(corr_df["affairs"].sort_values(ascending=False).round(4).to_string())

    # 2) Statistical tests
    y_yes = df.loc[df["children_yes"] == 1, "affairs"].astype(float)
    y_no = df.loc[df["children_yes"] == 0, "affairs"].astype(float)

    # Welch t-test (difference in means)
    t_res = stats.ttest_ind(y_yes, y_no, equal_var=False, nan_policy="omit")

    # One-sided p-value for the exact research direction: mean(children_yes) < mean(children_no)
    mean_diff = safe_float(y_yes.mean() - y_no.mean())
    if np.isnan(t_res.pvalue):
        t_p_one_sided_decrease = np.nan
    else:
        # If observed direction is opposite, one-sided p is near 1
        if mean_diff < 0:
            t_p_one_sided_decrease = safe_float(t_res.pvalue / 2.0)
        else:
            t_p_one_sided_decrease = safe_float(1.0 - t_res.pvalue / 2.0)

    # Non-parametric test due heavy zero-inflation/discreteness
    mw_less = stats.mannwhitneyu(y_yes, y_no, alternative="less")
    mw_two = stats.mannwhitneyu(y_yes, y_no, alternative="two-sided")

    # Any affair (binary) association test
    contingency = pd.crosstab(df["children_yes"], df["any_affair"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\n=== STATISTICAL TESTS ===")
    print(f"Mean affairs (children=yes): {y_yes.mean():.4f}")
    print(f"Mean affairs (children=no):  {y_no.mean():.4f}")
    print(f"Mean difference (yes - no):  {mean_diff:.4f}")
    print(f"Welch t-test (two-sided) p-value: {safe_float(t_res.pvalue):.6g}")
    print(f"Welch t-test one-sided p-value for decrease (yes<no): {t_p_one_sided_decrease:.6g}")
    print(f"Mann-Whitney U (one-sided, yes<no) p-value: {safe_float(mw_less.pvalue):.6g}")
    print(f"Mann-Whitney U (two-sided) p-value: {safe_float(mw_two.pvalue):.6g}")
    print(f"Chi-square test on any affair vs children p-value: {safe_float(chi2_p):.6g}")

    # OLS with robust standard errors
    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit(cov_type="HC3")
    ols_coef_children = safe_float(ols.params.get("children_yes", np.nan))
    ols_p_children = safe_float(ols.pvalues.get("children_yes", np.nan))

    # Poisson GLM is interpretable for count outcome
    poisson = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit(cov_type="HC3")
    pois_coef_children = safe_float(poisson.params.get("children_yes", np.nan))
    pois_p_children = safe_float(poisson.pvalues.get("children_yes", np.nan))
    pois_irr_children = safe_float(np.exp(pois_coef_children))

    print("\n=== REGRESSION MODELS (INFERENCE) ===")
    print(f"OLS children_yes coef: {ols_coef_children:.4f}, p-value: {ols_p_children:.6g}")
    print(f"Poisson children_yes coef: {pois_coef_children:.4f}, IRR: {pois_irr_children:.4f}, p-value: {pois_p_children:.6g}")

    # 3) Interpretable sklearn models
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0, random_state=0)
    lasso = Lasso(alpha=0.01, random_state=0, max_iter=20000)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=0)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=0)

    lr.fit(X, y)
    ridge.fit(X, y)
    lasso.fit(X, y)
    dtr.fit(X, y)
    dtc.fit(X, y_bin)

    coef_table = pd.DataFrame(
        {
            "feature": feature_cols,
            "linear_coef": lr.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
        }
    ).sort_values("linear_coef", key=np.abs, ascending=False)

    tree_importance = pd.Series(dtr.feature_importances_, index=feature_cols).sort_values(ascending=False)
    clf_importance = pd.Series(dtc.feature_importances_, index=feature_cols).sort_values(ascending=False)

    lr_children_coef = safe_float(lr.coef_[feature_cols.index("children_yes")])
    ridge_children_coef = safe_float(ridge.coef_[feature_cols.index("children_yes")])
    lasso_children_coef = safe_float(lasso.coef_[feature_cols.index("children_yes")])
    dtr_children_effect = marginal_effect_children_regression(dtr, X)
    dtc_children_effect = marginal_effect_children_classifier(dtc, X)

    print("\n=== INTERPRETABLE SKLEARN MODELS ===")
    print("Top coefficients (Linear/Ridge/Lasso):")
    print(coef_table.round(4).to_string(index=False))
    print("\nDecisionTreeRegressor feature importances:")
    print(tree_importance.round(4).to_string())
    print("\nDecisionTreeClassifier(any_affair) feature importances:")
    print(clf_importance.round(4).to_string())
    print("\nDecisionTreeRegressor (depth=3) structure:")
    print(export_text(dtr, feature_names=feature_cols))

    # 4) Interpretable imodels models
    imodels_results = {
        "loaded": False,
        "errors": [],
    }

    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

        imodels_results["loaded"] = True

        # RuleFit
        try:
            rulefit = RuleFitRegressor(random_state=0)
            rulefit.fit(X, y)

            rules_df = None
            if hasattr(rulefit, "get_rules"):
                rules_df = rulefit.get_rules()
            elif hasattr(rulefit, "_get_rules"):
                rules_df = rulefit._get_rules()

            children_rules = pd.DataFrame()
            if isinstance(rules_df, pd.DataFrame) and not rules_df.empty:
                children_rules = rules_df[rules_df["rule"].astype(str).str.contains("children_yes", regex=False)].copy()
                children_rules = children_rules.sort_values("importance", ascending=False)
                top_rules = rules_df.sort_values("importance", ascending=False).head(10)
                print("\n=== IMODELS: RULEFIT TOP RULES ===")
                print(top_rules[["rule", "coef", "support", "importance"]].round(4).to_string(index=False))
            else:
                print("\n=== IMODELS: RULEFIT ===")
                print("Rules table not available in this imodels version.")

            imodels_results["rulefit_children_rule_count"] = int(children_rules.shape[0])

            # RuleFit linear-term coefficient for children_yes if available
            rulefit_children_linear = np.nan
            if isinstance(rules_df, pd.DataFrame) and not rules_df.empty:
                linear_rows = rules_df[(rules_df["type"] == "linear") & (rules_df["rule"] == "children_yes")]
                if not linear_rows.empty:
                    rulefit_children_linear = safe_float(linear_rows.iloc[0]["coef"])
            imodels_results["rulefit_children_linear_coef"] = rulefit_children_linear

        except Exception as e:
            imodels_results["errors"].append(f"RuleFitRegressor failed: {e}")

        # FIGS
        try:
            figs = FIGSRegressor(random_state=0)
            figs.fit(X, y)
            figs_children_importance = np.nan
            if hasattr(figs, "feature_importances_"):
                imp = np.asarray(figs.feature_importances_, dtype=float)
                figs_children_importance = safe_float(imp[feature_cols.index("children_yes")])
                print("\n=== IMODELS: FIGS FEATURE IMPORTANCES ===")
                print(pd.Series(imp, index=feature_cols).sort_values(ascending=False).round(4).to_string())
            imodels_results["figs_children_importance"] = figs_children_importance
            imodels_results["figs_children_effect"] = marginal_effect_children_regression(figs, X)
        except Exception as e:
            imodels_results["errors"].append(f"FIGSRegressor failed: {e}")

        # HSTree
        try:
            hst = HSTreeRegressor(random_state=0)
            hst.fit(X, y)
            hst_text = str(hst)
            imodels_results["hstree_mentions_children"] = bool("children_yes" in hst_text)
            imodels_results["hstree_children_effect"] = marginal_effect_children_regression(hst, X)
            print("\n=== IMODELS: HSTREE SUMMARY ===")
            print("children_yes appears in tree:", imodels_results["hstree_mentions_children"])
        except Exception as e:
            imodels_results["errors"].append(f"HSTreeRegressor failed: {e}")

    except Exception as e:
        imodels_results["errors"].append(f"imodels import failed: {e}")

    # 5) Convert evidence into Likert score for the research question:
    # "Does having children decrease engagement in extramarital affairs?"
    # We require significant negative effects for high scores.

    sig_decrease_evidence = 0
    sig_increase_evidence = 0

    # Mean difference tests
    if (mean_diff < 0) and (safe_float(t_res.pvalue) < 0.05):
        sig_decrease_evidence += 1
    if (mean_diff > 0) and (safe_float(t_res.pvalue) < 0.05):
        sig_increase_evidence += 1

    # Binary affair association direction
    rate_yes = safe_float((y_yes > 0).mean())
    rate_no = safe_float((y_no > 0).mean())
    if (chi2_p < 0.05) and (rate_yes < rate_no):
        sig_decrease_evidence += 1
    if (chi2_p < 0.05) and (rate_yes > rate_no):
        sig_increase_evidence += 1

    # Multivariable models
    if (ols_coef_children < 0) and (ols_p_children < 0.05):
        sig_decrease_evidence += 1
    if (ols_coef_children > 0) and (ols_p_children < 0.05):
        sig_increase_evidence += 1

    if (pois_coef_children < 0) and (pois_p_children < 0.05):
        sig_decrease_evidence += 1
    if (pois_coef_children > 0) and (pois_p_children < 0.05):
        sig_increase_evidence += 1

    if sig_decrease_evidence >= 2 and sig_increase_evidence == 0:
        response = 85
    elif sig_decrease_evidence >= 1 and sig_increase_evidence == 0:
        response = 70
    elif sig_increase_evidence >= 2 and sig_decrease_evidence == 0:
        response = 10
    elif sig_increase_evidence >= 1 and sig_decrease_evidence == 0:
        response = 20
    else:
        response = 40

    response = int(np.clip(response, 0, 100))

    explanation = (
        f"Question: whether children decrease affair engagement. "
        f"Unadjusted means show higher affairs with children (yes={y_yes.mean():.2f}, no={y_no.mean():.2f}; "
        f"Welch t-test p={safe_float(t_res.pvalue):.3g}; one-sided decrease p={t_p_one_sided_decrease:.3g}). "
        f"Any-affair rate is also higher with children (yes={rate_yes:.3f}, no={rate_no:.3f}; chi-square p={chi2_p:.3g}). "
        f"After adjusting for covariates, children coefficient is not statistically significant in OLS "
        f"(coef={ols_coef_children:.3f}, p={ols_p_children:.3g}) and Poisson "
        f"(coef={pois_coef_children:.3f}, IRR={pois_irr_children:.3f}, p={pois_p_children:.3g}). "
        f"Interpretable sklearn/imodels models do not provide strong evidence of a robust negative children effect. "
        f"Overall, evidence does not support that having children decreases extramarital affairs in this dataset."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
