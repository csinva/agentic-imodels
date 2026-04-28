import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def top_series(series: pd.Series, n: int = 8) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("teachingratings.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    df = pd.read_csv(data_path)

    print_header("Research Question")
    print(research_question)

    print_header("Data Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values per column:")
    print(df.isna().sum())

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print_header("Numeric Summary Statistics")
    print(df[numeric_cols].describe().T.round(3))

    print_header("Categorical Distributions")
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        vc_pct = (df[col].value_counts(normalize=True, dropna=False) * 100).round(2)
        print(f"\n{col}:")
        for level, count in vc.items():
            print(f"  {level}: {count} ({vc_pct[level]}%)")

    print_header("Correlations (Numeric)")
    corr = df[numeric_cols].corr(numeric_only=True)
    print(corr.round(3))
    if "eval" in corr.columns:
        print("\nCorrelation with eval:")
        print(corr["eval"].sort_values(ascending=False).round(3))

    # Focus variables
    beauty = df["beauty"].astype(float)
    eval_score = df["eval"].astype(float)

    print_header("Statistical Tests: Beauty vs Teaching Evaluation")
    pearson_r, pearson_p = stats.pearsonr(beauty, eval_score)
    spearman_rho, spearman_p = stats.spearmanr(beauty, eval_score)
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.4g}")

    median_beauty = beauty.median()
    low_eval = eval_score[beauty < median_beauty]
    high_eval = eval_score[beauty >= median_beauty]
    t_stat, t_p = stats.ttest_ind(high_eval, low_eval, equal_var=False)
    pooled_sd = np.sqrt((low_eval.var(ddof=1) + high_eval.var(ddof=1)) / 2)
    cohen_d = (high_eval.mean() - low_eval.mean()) / pooled_sd
    print(
        "Median split t-test (high beauty vs low beauty): "
        f"t = {t_stat:.4f}, p = {t_p:.4g}, "
        f"mean_high = {high_eval.mean():.3f}, mean_low = {low_eval.mean():.3f}, d = {cohen_d:.3f}"
    )

    beauty_quartile = pd.qcut(beauty, q=4, labels=False)
    quartile_eval = [eval_score[beauty_quartile == i] for i in range(4)]
    anova_f, anova_p = stats.f_oneway(*quartile_eval)
    quartile_means = [arr.mean() for arr in quartile_eval]
    print(
        f"ANOVA across beauty quartiles: F = {anova_f:.4f}, p = {anova_p:.4g}, "
        f"quartile_means = {[round(x, 3) for x in quartile_means]}"
    )

    # OLS models (inference)
    simple_X = sm.add_constant(df[["beauty"]])
    ols_simple = sm.OLS(eval_score, simple_X).fit()
    simple_coef = ols_simple.params["beauty"]
    simple_p = ols_simple.pvalues["beauty"]
    simple_ci = ols_simple.conf_int().loc["beauty"].tolist()
    print("\nSimple OLS: eval ~ beauty")
    print(
        f"coef_beauty = {simple_coef:.4f}, p = {simple_p:.4g}, "
        f"95% CI = [{simple_ci[0]:.4f}, {simple_ci[1]:.4f}], R^2 = {ols_simple.rsquared:.4f}"
    )

    base_features = [
        "beauty",
        "age",
        "students",
        "allstudents",
        "gender",
        "minority",
        "credits",
        "division",
        "native",
        "tenure",
    ]

    mult_df = df[base_features].copy()
    mult_df_dummies = pd.get_dummies(mult_df, drop_first=True, dtype=float)
    mult_X = sm.add_constant(mult_df_dummies)
    ols_multi = sm.OLS(eval_score, mult_X).fit()
    multi_coef = ols_multi.params["beauty"]
    multi_p = ols_multi.pvalues["beauty"]
    multi_ci = ols_multi.conf_int().loc["beauty"].tolist()
    print("\nMultiple OLS (with controls)")
    print(
        f"coef_beauty = {multi_coef:.4f}, p = {multi_p:.4g}, "
        f"95% CI = [{multi_ci[0]:.4f}, {multi_ci[1]:.4f}], R^2 = {ols_multi.rsquared:.4f}"
    )

    no_beauty_dummies = mult_df_dummies.drop(columns=["beauty"])
    no_beauty_X = sm.add_constant(no_beauty_dummies)
    ols_no_beauty = sm.OLS(eval_score, no_beauty_X).fit()
    delta_r2 = ols_multi.rsquared - ols_no_beauty.rsquared
    print(f"Incremental R^2 from adding beauty to controlled OLS: {delta_r2:.4f}")

    print_header("Interpretable Models (scikit-learn + imodels)")
    X = df[base_features]
    y = eval_score

    num_features = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_onehot(), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    X_enc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    beauty_feature_name = "num__beauty" if "num__beauty" in feature_names else "beauty"

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    lin = LinearRegression()
    lin.fit(X_enc, y)
    lin_cv_r2 = cross_val_score(lin, X_enc, y, cv=cv, scoring="r2").mean()
    lin_coef = pd.Series(lin.coef_, index=feature_names)
    print(f"LinearRegression CV R^2 (mean): {lin_cv_r2:.4f}")
    print("Top linear coefficients (absolute magnitude):")
    print(top_series(lin_coef).round(4))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_enc, y)
    ridge_cv_r2 = cross_val_score(ridge, X_enc, y, cv=cv, scoring="r2").mean()
    ridge_coef = pd.Series(ridge.coef_, index=feature_names)
    print(f"\nRidge CV R^2 (mean): {ridge_cv_r2:.4f}")
    print("Top ridge coefficients (absolute magnitude):")
    print(top_series(ridge_coef).round(4))

    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    lasso.fit(X_enc, y)
    lasso_cv_r2 = cross_val_score(lasso, X_enc, y, cv=cv, scoring="r2").mean()
    lasso_coef = pd.Series(lasso.coef_, index=feature_names)
    print(f"\nLasso CV R^2 (mean): {lasso_cv_r2:.4f}")
    print("Top lasso coefficients (absolute magnitude):")
    print(top_series(lasso_coef).round(4))

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
    tree.fit(X_enc, y)
    tree_cv_r2 = cross_val_score(tree, X_enc, y, cv=cv, scoring="r2").mean()
    tree_imp = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nDecisionTreeRegressor CV R^2 (mean): {tree_cv_r2:.4f}")
    print("Top decision-tree feature importances:")
    print(tree_imp.head(8).round(4))

    # imodels
    rulefit = RuleFitRegressor(n_estimators=100, tree_size=4, max_rules=30, random_state=42)
    rulefit.fit(X_enc, y, feature_names=feature_names)
    rulefit_rules = rulefit._get_rules()
    active_rules = (
        rulefit_rules.loc[rulefit_rules["importance"] > 0, ["rule", "type", "coef", "importance"]]
        .sort_values("importance", ascending=False)
        .head(8)
    )
    print("\nRuleFitRegressor top active rules/terms:")
    if active_rules.empty:
        print("No active non-zero-importance rules found.")
    else:
        print(active_rules.to_string(index=False))

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_enc, y, feature_names=feature_names)
    figs_imp = pd.Series(figs.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFIGSRegressor top feature importances:")
    print(figs_imp.head(8).round(4))

    hstree = HSTreeRegressor(max_leaf_nodes=12, random_state=42)
    hstree.fit(X_enc, y, feature_names=feature_names)
    hs_imp = pd.Series(hstree.estimator_.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nHSTreeRegressor top feature importances:")
    print(hs_imp.head(8).round(4))

    beauty_top_count = 0
    if beauty_feature_name in top_series(lin_coef, n=5).index:
        beauty_top_count += 1
    if beauty_feature_name in tree_imp.head(5).index:
        beauty_top_count += 1
    if beauty_feature_name in figs_imp.head(5).index:
        beauty_top_count += 1
    if beauty_feature_name in hs_imp.head(5).index:
        beauty_top_count += 1

    # Convert evidence into a Likert response (0-100)
    score = 30
    if simple_p < 0.05:
        score += 10
    if multi_p < 0.05:
        score += 12
    if pearson_p < 0.05:
        score += 8
    if t_p < 0.05:
        score += 4
    if anova_p < 0.05:
        score += 4
    if simple_coef > 0 and multi_coef > 0 and pearson_r > 0:
        score += 8
    if abs(pearson_r) >= 0.3:
        score += 6
    elif abs(pearson_r) >= 0.2:
        score += 3
    elif abs(pearson_r) < 0.1:
        score -= 5
    if delta_r2 > 0.02:
        score += 6
    if beauty_top_count >= 2:
        score += 4

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Beauty shows a statistically significant positive association with teaching evaluations "
        f"(Pearson r={pearson_r:.3f}, p={pearson_p:.2e}; simple OLS coef={simple_coef:.3f}, "
        f"p={simple_p:.2e}; controlled OLS coef={multi_coef:.3f}, p={multi_p:.2e}). "
        f"Group tests are also significant (median-split t-test p={t_p:.3f}, ANOVA p={anova_p:.2e}). "
        f"Interpretable tree/rule models rank beauty among important predictors, but effect size is modest "
        f"(small-moderate correlation; incremental R^2 from beauty={delta_r2:.3f})."
    )

    output = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print_header("Final Likert Response")
    print(output)
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
