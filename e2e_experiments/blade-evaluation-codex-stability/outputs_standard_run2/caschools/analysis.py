import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def hist_summary(series: pd.Series, bins: int = 10):
    counts, edges = np.histogram(series, bins=bins)
    return [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def top_abs_coefs(feature_names, coefs, k: int = 5):
    coef_s = pd.Series(coefs, index=feature_names)
    return coef_s.reindex(coef_s.abs().sort_values(ascending=False).index).head(k)


def main():
    # 1) Load question + data
    with open("info.json", "r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    df = pd.read_csv("caschools.csv")

    # 2) Feature engineering for the question
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["computer_per_student"] = df["computer"] / df["students"]
    df["grade_KK08"] = (df["grades"] == "KK-08").astype(int)

    # 3) EDA
    print("Research question:", research_question)
    print("\nShape:", df.shape)
    print("\nNumeric summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    dist_cols = ["student_teacher_ratio", "avg_score", "income", "lunch", "english", "expenditure"]
    print("\nDistribution summaries (10-bin histograms):")
    for col in dist_cols:
        print(f"\n{col}:")
        for h in hist_summary(df[col], bins=10):
            print(h)

    corr_with_score = (
        df[numeric_cols]
        .corr(numeric_only=True)["avg_score"]
        .sort_values(key=np.abs, ascending=False)
    )
    print("\nCorrelations with avg_score (sorted by absolute value):")
    print(corr_with_score)

    # 4) Statistical tests focused on question
    y = df["avg_score"]
    ratio = df["student_teacher_ratio"]

    pearson_r, pearson_p = stats.pearsonr(ratio, y)
    spearman_rho, spearman_p = stats.spearmanr(ratio, y)

    median_ratio = ratio.median()
    low_ratio_scores = y[ratio <= median_ratio]
    high_ratio_scores = y[ratio > median_ratio]
    t_stat, t_p = stats.ttest_ind(low_ratio_scores, high_ratio_scores, equal_var=False)

    df["ratio_quartile"] = pd.qcut(
        ratio,
        q=4,
        labels=["Q1_lowest_ratio", "Q2", "Q3", "Q4_highest_ratio"],
    )
    quartile_means = df.groupby("ratio_quartile", observed=False)["avg_score"].mean()
    groups = [df[df["ratio_quartile"] == q]["avg_score"] for q in quartile_means.index]
    anova_f, anova_p = stats.f_oneway(*groups)

    print("\nStatistical tests:")
    print(f"Pearson r(student_teacher_ratio, avg_score) = {pearson_r:.4f}, p = {pearson_p:.3g}")
    print(f"Spearman rho(student_teacher_ratio, avg_score) = {spearman_rho:.4f}, p = {spearman_p:.3g}")
    print(
        "Median-split t-test (low ratio vs high ratio avg_score): "
        f"t = {t_stat:.4f}, p = {t_p:.3g}; "
        f"means = {low_ratio_scores.mean():.3f} vs {high_ratio_scores.mean():.3f}"
    )
    print(f"ANOVA across ratio quartiles: F = {anova_f:.4f}, p = {anova_p:.3g}")
    print("Quartile means:")
    print(quartile_means)

    # OLS: simple and adjusted
    X_simple = sm.add_constant(df[["student_teacher_ratio"]])
    ols_simple = sm.OLS(y, X_simple).fit()

    controls = [
        "student_teacher_ratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer_per_student",
        "grade_KK08",
    ]
    X_multi = sm.add_constant(df[controls])
    ols_multi = sm.OLS(y, X_multi).fit()

    simple_coef = float(ols_simple.params["student_teacher_ratio"])
    simple_p = float(ols_simple.pvalues["student_teacher_ratio"])
    multi_coef = float(ols_multi.params["student_teacher_ratio"])
    multi_p = float(ols_multi.pvalues["student_teacher_ratio"])
    multi_ci_low, multi_ci_high = ols_multi.conf_int().loc["student_teacher_ratio"].tolist()

    print("\nOLS results:")
    print(
        "Simple OLS avg_score ~ student_teacher_ratio: "
        f"coef = {simple_coef:.4f}, p = {simple_p:.3g}, R^2 = {ols_simple.rsquared:.4f}"
    )
    print(
        "Adjusted OLS (with controls): "
        f"coef = {multi_coef:.4f}, p = {multi_p:.3g}, "
        f"95% CI = [{multi_ci_low:.4f}, {multi_ci_high:.4f}], R^2 = {ols_multi.rsquared:.4f}"
    )

    # 5) Interpretable ML models
    X = df[controls]
    y_np = y.values
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=0.25, random_state=42
    )

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_r2 = r2_score(y_test, linear.predict(X_test))

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    ridge.fit(X_train, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))
    ridge_coef = ridge.named_steps["ridge"].coef_

    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05, random_state=42, max_iter=10000))
    lasso.fit(X_train, y_train)
    lasso_r2 = r2_score(y_test, lasso.predict(X_test))
    lasso_coef = lasso.named_steps["lasso"].coef_

    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    tree_r2 = r2_score(y_test, tree.predict(X_test))
    tree_importance = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)

    # imodels
    rulefit = RuleFitRegressor(n_estimators=200, max_rules=40, random_state=42)
    rulefit.fit(X_train.values, y_train, feature_names=feature_names)
    rulefit_r2 = r2_score(y_test, rulefit.predict(X_test.values))
    if hasattr(rulefit, "get_rules"):
        rules = rulefit.get_rules()
    else:
        rules = rulefit._get_rules(exclude_zero_coef=False)
    active_rules = rules[(rules["type"] == "rule") & (rules["coef"] != 0)].copy()
    ratio_rules = active_rules[
        active_rules["rule"].str.contains("student_teacher_ratio", regex=False, na=False)
    ].copy()
    if not ratio_rules.empty:
        ratio_rules = ratio_rules.reindex(ratio_rules["coef"].abs().sort_values(ascending=False).index).head(5)

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_train.values, y_train, feature_names=feature_names)
    figs_r2 = r2_score(y_test, figs.predict(X_test.values))
    figs_importance = pd.Series(figs.feature_importances_, index=feature_names).sort_values(ascending=False)

    hst = HSTreeRegressor(max_leaf_nodes=20, random_state=42)
    hst.fit(X_train.values, y_train, feature_names=feature_names)
    hst_r2 = r2_score(y_test, hst.predict(X_test.values))
    if hasattr(hst, "feature_importances_"):
        hst_importance_values = hst.feature_importances_
    elif hasattr(hst, "estimator_") and hasattr(hst.estimator_, "feature_importances_"):
        hst_importance_values = hst.estimator_.feature_importances_
    else:
        hst_importance_values = np.zeros(len(feature_names))
    hst_importance = pd.Series(hst_importance_values, index=feature_names).sort_values(ascending=False)

    print("\nInterpretable model summaries (holdout R^2):")
    print(f"LinearRegression R^2: {linear_r2:.4f}")
    print(f"Ridge R^2: {ridge_r2:.4f}")
    print(f"Lasso R^2: {lasso_r2:.4f}")
    print(f"DecisionTreeRegressor R^2: {tree_r2:.4f}")
    print(f"RuleFitRegressor R^2: {rulefit_r2:.4f}")
    print(f"FIGSRegressor R^2: {figs_r2:.4f}")
    print(f"HSTreeRegressor R^2: {hst_r2:.4f}")

    lin_coef = pd.Series(linear.coef_, index=feature_names)
    print("\nTop absolute LinearRegression coefficients:")
    print(top_abs_coefs(feature_names, linear.coef_, k=8))
    print("\nTop absolute Ridge coefficients (standardized-space):")
    print(top_abs_coefs(feature_names, ridge_coef, k=8))
    print("\nTop absolute Lasso coefficients (standardized-space):")
    print(top_abs_coefs(feature_names, lasso_coef, k=8))
    print("\nDecisionTree feature importances:")
    print(tree_importance)
    print("\nFIGS feature importances:")
    print(figs_importance)
    print("\nHSTree feature importances:")
    print(hst_importance)

    if ratio_rules.empty:
        print("\nRuleFit active rules involving student_teacher_ratio: none")
    else:
        print("\nRuleFit active rules involving student_teacher_ratio:")
        print(ratio_rules[["rule", "coef", "support"]])

    # 6) Evidence aggregation into 0-100 response
    # Rules: significant support -> yes; significant contradiction -> no;
    # non-significance is treated as evidence against strong relationship.
    tests = [
        {
            "name": "pearson",
            "weight": 20,
            "significant": pearson_p < 0.05,
            "supports_yes": pearson_r < 0,
        },
        {
            "name": "t_test",
            "weight": 15,
            "significant": t_p < 0.05,
            "supports_yes": low_ratio_scores.mean() > high_ratio_scores.mean(),
        },
        {
            "name": "anova",
            "weight": 10,
            "significant": anova_p < 0.05,
            "supports_yes": quartile_means["Q1_lowest_ratio"] > quartile_means["Q4_highest_ratio"],
        },
        {
            "name": "simple_ols",
            "weight": 20,
            "significant": simple_p < 0.05,
            "supports_yes": simple_coef < 0,
        },
        {
            "name": "adjusted_ols",
            "weight": 35,
            "significant": multi_p < 0.05,
            "supports_yes": multi_coef < 0,
        },
    ]

    yes_score = 0.0
    no_score = 0.0

    for t in tests:
        w = t["weight"]
        if t["significant"] and t["supports_yes"]:
            yes_score += w
        elif t["significant"] and not t["supports_yes"]:
            no_score += w
        else:
            # Non-significant findings count against claiming a relationship.
            no_score += 0.8 * w
            if t["supports_yes"]:
                yes_score += 0.2 * w

    # Small extra weight from direction consistency in interpretable linear models
    direction_votes_yes = 0
    direction_votes_no = 0
    for coef in [
        lin_coef["student_teacher_ratio"],
        ridge_coef[feature_names.index("student_teacher_ratio")],
        lasso_coef[feature_names.index("student_teacher_ratio")],
    ]:
        if coef < 0:
            direction_votes_yes += 1
        elif coef > 0:
            direction_votes_no += 1

    yes_score += 2 * direction_votes_yes
    no_score += 2 * direction_votes_no

    response = int(np.clip(round(100 * yes_score / (yes_score + no_score)), 0, 100))

    explanation = (
        f"Question: {research_question} "
        f"Pearson correlation between student-teacher ratio and average test score was r={pearson_r:.3f} (p={pearson_p:.3g}). "
        f"In adjusted OLS with socioeconomic controls, the student-teacher-ratio coefficient was {multi_coef:.3f} "
        f"(p={multi_p:.3g}, 95% CI [{multi_ci_low:.3f}, {multi_ci_high:.3f}]). "
        f"Median-split t-test p={t_p:.3g} and quartile ANOVA p={anova_p:.3g}; quartile means declined from "
        f"{quartile_means['Q1_lowest_ratio']:.2f} (lowest ratio) to {quartile_means['Q4_highest_ratio']:.2f} (highest ratio). "
        f"Interpretable models (Linear/Ridge/Lasso/Tree/RuleFit/FIGS/HSTree) were also fit, and linear-model coefficients for "
        f"student-teacher ratio were directionally {'consistent' if direction_votes_yes >= 2 else 'mixed'} with the hypothesis."
    )

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)

    print("\nFinal response:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
