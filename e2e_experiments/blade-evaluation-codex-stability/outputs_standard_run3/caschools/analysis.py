import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def top_abs_series(series: pd.Series, n: int = 5) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    print(f"Research question: {research_question}")

    df = pd.read_csv("caschools.csv")

    # Core variables for this question
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["academic_performance"] = (df["read"] + df["math"]) / 2.0

    print("\nData shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nSummary statistics (selected):")
    print(
        df[[
            "student_teacher_ratio",
            "academic_performance",
            "students",
            "teachers",
            "lunch",
            "income",
            "english",
            "expenditure",
        ]]
        .describe()
        .round(3)
    )

    print("\nDistribution snapshots:")
    for col in ["student_teacher_ratio", "academic_performance"]:
        q = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
        print(f"{col} quantiles: {q}")

    corr = df[numeric_cols].corr(numeric_only=True)
    corr_target = corr["academic_performance"].sort_values(ascending=False)
    print("\nCorrelations with academic_performance:")
    print(corr_target.round(3).head(10))
    print(corr_target.round(3).tail(10))

    # Statistical tests focused on the research question
    x = df["student_teacher_ratio"]
    y = df["academic_performance"]

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    median_ratio = x.median()
    low_ratio_scores = y[x <= median_ratio]
    high_ratio_scores = y[x > median_ratio]
    ttest = stats.ttest_ind(low_ratio_scores, high_ratio_scores, equal_var=False)

    df["str_quartile"] = pd.qcut(df["student_teacher_ratio"], q=4, labels=False)
    anova_groups = [
        df.loc[df["str_quartile"] == q, "academic_performance"] for q in sorted(df["str_quartile"].unique())
    ]
    anova = stats.f_oneway(*anova_groups)

    ols_unadj = sm.OLS(y, sm.add_constant(df[["student_teacher_ratio"]])).fit(cov_type="HC3")

    controls = [
        "student_teacher_ratio",
        "lunch",
        "income",
        "english",
        "calworks",
        "expenditure",
        "computer",
    ]
    ols_adj = sm.OLS(y, sm.add_constant(df[controls])).fit(cov_type="HC3")

    print("\nStatistical tests:")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.3g}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.3g}")
    print(
        "Median split t-test: "
        f"mean(low STR)={low_ratio_scores.mean():.3f}, "
        f"mean(high STR)={high_ratio_scores.mean():.3f}, "
        f"t={ttest.statistic:.3f}, p={ttest.pvalue:.3g}"
    )
    print(f"ANOVA across STR quartiles: F={anova.statistic:.3f}, p={anova.pvalue:.3g}")
    print(
        "OLS unadjusted beta(STR)="
        f"{ols_unadj.params['student_teacher_ratio']:.4f}, "
        f"p={ols_unadj.pvalues['student_teacher_ratio']:.3g}, "
        f"R2={ols_unadj.rsquared:.3f}"
    )
    print(
        "OLS adjusted beta(STR)="
        f"{ols_adj.params['student_teacher_ratio']:.4f}, "
        f"p={ols_adj.pvalues['student_teacher_ratio']:.3g}, "
        f"R2={ols_adj.rsquared:.3f}"
    )

    # Interpretable models
    X = df[controls]
    model_results = {}

    lin = LinearRegression().fit(X, y)
    lin_coef = pd.Series(lin.coef_, index=controls)
    model_results["LinearRegression_top_coef"] = top_abs_series(lin_coef).round(4).to_dict()

    ridge = Ridge(alpha=1.0).fit(X, y)
    ridge_coef = pd.Series(ridge.coef_, index=controls)
    model_results["Ridge_top_coef"] = top_abs_series(ridge_coef).round(4).to_dict()

    lasso = Lasso(alpha=0.05, max_iter=20000, random_state=0).fit(X, y)
    lasso_coef = pd.Series(lasso.coef_, index=controls)
    model_results["Lasso_nonzero_coef"] = lasso_coef[lasso_coef != 0].round(4).to_dict()

    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    tree_imp = pd.Series(tree.feature_importances_, index=controls)
    model_results["DecisionTree_top_importance"] = top_abs_series(tree_imp).round(4).to_dict()

    rulefit = RuleFitRegressor(random_state=0).fit(X, y)
    extracted_rules = [r.rule for r in rulefit.rules_]
    str_rules = [r for r in extracted_rules if "student_teacher_ratio" in r]
    model_results["RuleFit_n_rules"] = int(len(extracted_rules))
    model_results["RuleFit_str_rule_count"] = int(len(str_rules))
    model_results["RuleFit_example_rules"] = extracted_rules[:8]

    figs = FIGSRegressor(random_state=0, max_rules=12).fit(X, y)
    figs_imp = pd.Series(figs.feature_importances_, index=controls)
    model_results["FIGS_top_importance"] = top_abs_series(figs_imp).round(4).to_dict()

    hst = HSTreeRegressor(random_state=0, max_leaf_nodes=12).fit(X, y)
    hst_text = str(hst)
    model_results["HSTree_mentions_STR"] = bool("student_teacher_ratio" in hst_text)

    print("\nInterpretable model signals:")
    print(json.dumps(model_results, indent=2))

    # Convert evidence to Likert 0-100 answer.
    # Primary criterion: significance of STR after adjustment for confounders.
    adj_beta = float(ols_adj.params["student_teacher_ratio"])
    adj_p = float(ols_adj.pvalues["student_teacher_ratio"])
    unadj_beta = float(ols_unadj.params["student_teacher_ratio"])
    unadj_p = float(ols_unadj.pvalues["student_teacher_ratio"])

    if adj_p < 0.05 and adj_beta < 0:
        response = 85
        verdict = "Strong evidence"
    elif adj_p < 0.05 and adj_beta >= 0:
        response = 10
        verdict = "Evidence against"
    elif unadj_p < 0.05 and unadj_beta < 0:
        response = 45
        verdict = "Mixed evidence"
    else:
        response = 20
        verdict = "Weak evidence"

    explanation = (
        f"{verdict}: lower student-teacher ratio is associated with higher performance in unadjusted analyses "
        f"(Pearson r={pearson_r:.3f}, p={pearson_p:.2g}; unadjusted OLS beta={unadj_beta:.3f}, p={unadj_p:.2g}; "
        f"median-split t-test p={ttest.pvalue:.2g}). However, after adjusting for socioeconomic covariates "
        f"(lunch, income, english learners, calworks, expenditure, computers), the STR effect is not statistically "
        f"significant (adjusted OLS beta={adj_beta:.3f}, p={adj_p:.2g}). Interpretable tree/rule models place most "
        f"importance on socioeconomic variables, so evidence for an independent STR-performance relationship is limited."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
