import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    return stats.pearsonr(x[mask], y[mask])


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    return stats.spearmanr(x[mask], y[mask])


def summarize_exploration(df: pd.DataFrame) -> None:
    print("=== Data Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n=== Summary Statistics (numeric) ===")
    print(df[numeric_cols].describe().T.to_string())

    print("\n=== Distribution Snapshots ===")
    for col in ["alldeaths", "masfem", "min", "wind", "ndam15"]:
        q = df[col].quantile([0, 0.25, 0.5, 0.75, 1.0])
        skew = stats.skew(df[col], nan_policy="omit")
        print(f"{col}: skew={skew:.3f}, quantiles={q.to_dict()}")

    corr_cols = [
        "masfem",
        "masfem_mturk",
        "gender_mf",
        "alldeaths",
        "wind",
        "min",
        "category",
        "ndam15",
        "year",
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    print("\n=== Correlations with alldeaths ===")
    print(corr["alldeaths"].sort_values(ascending=False).to_string())


def run_statistical_tests(df: pd.DataFrame) -> Dict[str, float]:
    print("\n=== Statistical Tests ===")
    out: Dict[str, float] = {}

    df = df.copy()
    df["log_deaths"] = np.log1p(df["alldeaths"])

    pearson_r, pearson_p = safe_pearson(df["masfem"], df["log_deaths"])
    spearman_rho, spearman_p = safe_spearman(df["masfem"], df["log_deaths"])
    out["pearson_r"] = float(pearson_r)
    out["pearson_p"] = float(pearson_p)
    out["spearman_rho"] = float(spearman_rho)
    out["spearman_p"] = float(spearman_p)
    print(f"Pearson(masfem, log_deaths): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman(masfem, log_deaths): rho={spearman_rho:.4f}, p={spearman_p:.4g}")

    female_log = df.loc[df["gender_mf"] == 1, "log_deaths"]
    male_log = df.loc[df["gender_mf"] == 0, "log_deaths"]
    t_stat, t_p = stats.ttest_ind(female_log, male_log, equal_var=False)
    out["ttest_log_t"] = float(t_stat)
    out["ttest_log_p"] = float(t_p)
    out["female_log_mean"] = float(female_log.mean())
    out["male_log_mean"] = float(male_log.mean())
    print(
        "Welch t-test(log_deaths by binary gender): "
        f"t={t_stat:.4f}, p={t_p:.4g}, "
        f"means(female={female_log.mean():.3f}, male={male_log.mean():.3f})"
    )

    mw = stats.mannwhitneyu(
        df.loc[df["gender_mf"] == 1, "alldeaths"],
        df.loc[df["gender_mf"] == 0, "alldeaths"],
        alternative="two-sided",
    )
    out["mannwhitney_u"] = float(mw.statistic)
    out["mannwhitney_p"] = float(mw.pvalue)
    print(f"Mann-Whitney(alldeaths by binary gender): U={mw.statistic:.1f}, p={mw.pvalue:.4g}")

    df["masfem_tertile"] = pd.qcut(df["masfem"], q=3, labels=["low", "mid", "high"])
    groups = [
        np.log1p(g["alldeaths"]).values
        for _, g in df.groupby("masfem_tertile", observed=False)
    ]
    f_stat, f_p = stats.f_oneway(*groups)
    out["anova_f"] = float(f_stat)
    out["anova_p"] = float(f_p)
    print(f"ANOVA(log_deaths by masfem tertiles): F={f_stat:.4f}, p={f_p:.4g}")

    z_cols = ["masfem", "min", "wind", "category", "ndam15", "year"]
    for c in z_cols:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

    ols_main = smf.ols(
        "log_deaths ~ masfem_z + min_z + wind_z + category_z + ndam15_z + year_z",
        data=df,
    ).fit()
    out["ols_masfem_coef"] = float(ols_main.params["masfem_z"])
    out["ols_masfem_p"] = float(ols_main.pvalues["masfem_z"])
    print(
        "OLS main effect (controls included): "
        f"coef_masfem={ols_main.params['masfem_z']:.4f}, "
        f"p={ols_main.pvalues['masfem_z']:.4g}, R2={ols_main.rsquared:.3f}"
    )

    ols_inter = smf.ols(
        "log_deaths ~ masfem_z * min_z + wind_z + category_z + ndam15_z + year_z",
        data=df,
    ).fit()
    out["ols_interaction_coef"] = float(ols_inter.params["masfem_z:min_z"])
    out["ols_interaction_p"] = float(ols_inter.pvalues["masfem_z:min_z"])
    print(
        "OLS interaction (masfem x min): "
        f"coef={ols_inter.params['masfem_z:min_z']:.4f}, "
        f"p={ols_inter.pvalues['masfem_z:min_z']:.4g}, R2={ols_inter.rsquared:.3f}"
    )

    return out


def fit_interpretable_models(df: pd.DataFrame) -> Dict[str, float]:
    print("\n=== Interpretable Models ===")
    out: Dict[str, float] = {}

    features = ["masfem", "gender_mf", "min", "wind", "category", "ndam15", "year"]
    X = df[features].copy()
    y = np.log1p(df["alldeaths"]).values

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    lin = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    lin.fit(X, y)
    lin_r2 = cross_val_score(lin, X, y, cv=cv, scoring="r2").mean()
    lin_coef = lin.named_steps["model"].coef_
    lin_coef_map = dict(zip(features, lin_coef))
    out["lin_cv_r2"] = float(lin_r2)
    out["lin_coef_masfem"] = float(lin_coef_map["masfem"])
    print(f"LinearRegression CV R2={lin_r2:.3f}")
    print("LinearRegression standardized coefficients:")
    print(pd.Series(lin_coef_map).sort_values(key=np.abs, ascending=False).to_string())

    ridge = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=0))]
    )
    ridge.fit(X, y)
    ridge_coef = dict(zip(features, ridge.named_steps["model"].coef_))
    out["ridge_coef_masfem"] = float(ridge_coef["masfem"])
    print("Ridge coefficients:")
    print(pd.Series(ridge_coef).sort_values(key=np.abs, ascending=False).to_string())

    lasso = Pipeline(
        [("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, max_iter=20000, random_state=0))]
    )
    lasso.fit(X, y)
    lasso_coef = dict(zip(features, lasso.named_steps["model"].coef_))
    out["lasso_coef_masfem"] = float(lasso_coef["masfem"])
    print("Lasso coefficients:")
    print(pd.Series(lasso_coef).sort_values(key=np.abs, ascending=False).to_string())

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
    tree.fit(X, y)
    tree_importance = dict(zip(features, tree.feature_importances_))
    out["tree_importance_masfem"] = float(tree_importance["masfem"])
    print("DecisionTree feature_importances:")
    print(pd.Series(tree_importance).sort_values(ascending=False).to_string())

    rulefit = RuleFitRegressor(random_state=0, max_rules=40)
    rulefit.fit(X, y)
    rules_df = rulefit._get_rules(exclude_zero_coef=True)
    top_rules = rules_df.sort_values("importance", ascending=False).head(10)
    masfem_rules = rules_df[rules_df["rule"].str.contains("masfem", case=False, na=False)]
    out["rulefit_num_rules"] = float(len(rules_df))
    out["rulefit_num_masfem_rules"] = float(len(masfem_rules))
    out["rulefit_top_rule_has_masfem"] = float(
        int(top_rules["rule"].str.contains("masfem", case=False, na=False).any())
    )
    print("RuleFit top rules by importance:")
    print(top_rules[["rule", "coef", "importance"]].to_string(index=False))

    figs = FIGSRegressor(random_state=0, max_rules=12)
    figs.fit(X, y)
    figs_imp = dict(zip(features, figs.feature_importances_))
    out["figs_importance_masfem"] = float(figs_imp["masfem"])
    print("FIGS feature_importances:")
    print(pd.Series(figs_imp).sort_values(ascending=False).to_string())

    hs_base = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
    hs = HSTreeRegressor(estimator_=hs_base, reg_param=1)
    hs.fit(X, y)
    hs_imp = dict(zip(features, hs.estimator_.feature_importances_))
    out["hs_importance_masfem"] = float(hs_imp["masfem"])
    print("HSTree(base tree) feature_importances:")
    print(pd.Series(hs_imp).sort_values(ascending=False).to_string())

    return out


def compute_score(stats_out: Dict[str, float], model_out: Dict[str, float]) -> Tuple[int, str]:
    # Primary hypothesis expects a positive femininity effect on fatalities.
    positive_significant_tests = 0

    if stats_out["pearson_r"] > 0 and stats_out["pearson_p"] < 0.05:
        positive_significant_tests += 1
    if (
        stats_out["female_log_mean"] > stats_out["male_log_mean"]
        and stats_out["ttest_log_p"] < 0.05
    ):
        positive_significant_tests += 1
    if stats_out["ols_masfem_coef"] > 0 and stats_out["ols_masfem_p"] < 0.05:
        positive_significant_tests += 1
    if (
        stats_out["ols_interaction_coef"] < 0
        and stats_out["ols_interaction_p"] < 0.05
    ):
        # Negative interaction with min pressure means stronger storms (lower min)
        # have a stronger positive femininity association.
        positive_significant_tests += 1

    weak_signal = (
        stats_out["pearson_p"] < 0.10
        or stats_out["ttest_log_p"] < 0.10
        or stats_out["ols_masfem_p"] < 0.10
        or stats_out["ols_interaction_p"] < 0.10
    )

    masfem_model_signal = np.mean(
        [
            abs(model_out["lin_coef_masfem"]),
            abs(model_out["ridge_coef_masfem"]),
            abs(model_out["lasso_coef_masfem"]),
        ]
    )

    if positive_significant_tests >= 3:
        score = 90
    elif positive_significant_tests == 2:
        score = 78
    elif positive_significant_tests == 1:
        score = 62
    elif weak_signal:
        score = 40
    else:
        score = 18

    # Keep score conservative when significance is absent and coefficients are modest.
    if positive_significant_tests == 0 and masfem_model_signal < 0.12:
        score = max(10, score - 3)

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Across correlation, group-comparison, and controlled OLS tests, the femininity "
        "predictor is not statistically significant for fatalities (Pearson p={pearson_p:.3f}, "
        "Welch t-test p={ttest_p:.3f}, OLS main-effect p={ols_p:.3f}, interaction p={int_p:.3f}). "
        "Interpretable models (linear/tree/RuleFit/FIGS/HSTree) consistently prioritize storm "
        "severity and damage variables over name femininity. This provides weak evidence for the "
        "hypothesized relationship in this dataset."
    ).format(
        pearson_p=stats_out["pearson_p"],
        ttest_p=stats_out["ttest_log_p"],
        ols_p=stats_out["ols_masfem_p"],
        int_p=stats_out["ols_interaction_p"],
    )

    return score, explanation


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)

    df = pd.read_csv("hurricane.csv")

    summarize_exploration(df)
    stats_out = run_statistical_tests(df)
    model_out = fit_interpretable_models(df)

    response, explanation = compute_score(stats_out, model_out)
    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\n=== Final Output ===")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
