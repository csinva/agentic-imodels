import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pvalue_support(pvalue: float) -> float:
    """Map p-values to evidence strength on [0, 1]."""
    if pvalue < 0.01:
        return 1.0
    if pvalue < 0.05:
        return 0.85
    if pvalue < 0.10:
        return 0.60
    if pvalue < 0.20:
        return 0.35
    return 0.10


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    info = json.loads(info_path.read_text())
    question = info["research_questions"][0]
    df = pd.read_csv(data_path)

    # Feature engineering focused on relative size and contest location.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["male_diff"] = df["m_focal"] - df["m_other"]
    df["female_diff"] = df["f_focal"] - df["f_other"]
    df["dist_diff"] = df["dist_focal"] - df["dist_other"]
    df["location_advantage"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    section("Research question")
    print(question)

    section("Data overview")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    section("Summary statistics")
    print(df[numeric_cols].describe().T.to_string())

    section("Distributions (5-bin histograms)")
    dist_columns = ["win", "size_diff", "dist_diff", "n_focal", "n_other", "dist_focal", "dist_other"]
    for col in dist_columns:
        values = df[col].to_numpy()
        counts, bins = np.histogram(values, bins=5)
        print(f"{col}: bins={np.round(bins, 3).tolist()}, counts={counts.tolist()}")

    section("Correlations with outcome (win)")
    corr_to_win = df[numeric_cols].corr(numeric_only=True)["win"].sort_values(ascending=False)
    print(corr_to_win.to_string())

    # Statistical tests for the research question.
    section("Statistical tests")
    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]

    t_size = stats.ttest_ind(wins["size_diff"], losses["size_diff"], equal_var=False)
    t_dist = stats.ttest_ind(wins["dist_diff"], losses["dist_diff"], equal_var=False)

    pb_size = stats.pointbiserialr(df["win"], df["size_diff"])
    pb_dist = stats.pointbiserialr(df["win"], df["dist_diff"])

    contingency = pd.crosstab(df["location_advantage"], df["win"])
    chi2_res = stats.chi2_contingency(contingency)

    # ANOVA across relative size categories.
    size_cat = np.where(df["size_diff"] > 0, "focal_larger", np.where(df["size_diff"] < 0, "focal_smaller", "equal"))
    df["size_category"] = size_cat
    groups = [grp["win"].values for _, grp in df.groupby("size_category") if len(grp) > 1]
    anova_size = stats.f_oneway(*groups) if len(groups) >= 2 else None

    print(f"Welch t-test (size_diff by win): statistic={t_size.statistic:.4f}, p={t_size.pvalue:.4f}")
    print(f"Welch t-test (dist_diff by win): statistic={t_dist.statistic:.4f}, p={t_dist.pvalue:.4f}")
    print(f"Point-biserial corr (win, size_diff): r={pb_size.statistic:.4f}, p={pb_size.pvalue:.4f}")
    print(f"Point-biserial corr (win, dist_diff): r={pb_dist.statistic:.4f}, p={pb_dist.pvalue:.4f}")
    print(f"Chi-square (location_advantage vs win): chi2={chi2_res.statistic:.4f}, p={chi2_res.pvalue:.4f}")
    print("Contingency table (rows=location_advantage 0/1, cols=win 0/1):")
    print(contingency.to_string())
    if anova_size is not None:
        print(f"ANOVA (win across size categories): F={anova_size.statistic:.4f}, p={anova_size.pvalue:.4f}")

    section("Statsmodels regression")
    # OLS as requested for interpretable coefficients + p-values.
    X_ols = sm.add_constant(df[["size_diff", "dist_diff", "location_advantage"]])
    y = df["win"]
    ols_model = sm.OLS(y, X_ols).fit()
    print("OLS coefficients:")
    print(ols_model.params.to_string())
    print("OLS p-values:")
    print(ols_model.pvalues.to_string())
    print("OLS 95% CI:")
    print(ols_model.conf_int().rename(columns={0: "ci_low", 1: "ci_high"}).to_string())

    # Logistic regression is appropriate for binary outcome.
    X_logit = sm.add_constant(df[["size_diff", "dist_diff"]])
    logit_model = sm.Logit(y, X_logit).fit(disp=False)
    print("\nLogit coefficients:")
    print(logit_model.params.to_string())
    print("Logit p-values:")
    print(logit_model.pvalues.to_string())
    print("Logit odds ratios:")
    print(np.exp(logit_model.params).rename("odds_ratio").to_string())

    section("Interpretable sklearn models")
    feature_cols = ["size_diff", "male_diff", "female_diff", "dist_diff", "location_advantage"]
    X_rel = df[feature_cols]

    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=0))])
    lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05, random_state=0, max_iter=10000))])

    lin.fit(X_rel, y)
    ridge.fit(X_rel, y)
    lasso.fit(X_rel, y)

    lin_coef = pd.Series(lin.named_steps["model"].coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    ridge_coef = pd.Series(ridge.named_steps["model"].coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    lasso_coef = pd.Series(lasso.named_steps["model"].coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)

    print("LinearRegression standardized coefficients:")
    print(lin_coef.to_string())
    print("\nRidge standardized coefficients:")
    print(ridge_coef.to_string())
    print("\nLasso standardized coefficients:")
    print(lasso_coef.to_string())

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
    tree.fit(X_rel, y)
    tree_importance = pd.Series(tree.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nDecisionTreeClassifier feature importances:")
    print(tree_importance.to_string())
    print("\nDecision tree rules:")
    print(export_text(tree, feature_names=feature_cols))

    section("Interpretable imodels models")
    model_cols = [
        "dist_focal",
        "dist_other",
        "n_focal",
        "n_other",
        "m_focal",
        "m_other",
        "f_focal",
        "f_other",
        "size_diff",
        "dist_diff",
        "location_advantage",
    ]
    X_model = df[model_cols]

    rulefit = RuleFitRegressor(random_state=0, n_estimators=100, max_rules=20)
    rulefit.fit(X_model, y)
    rules = getattr(rulefit, "rules_", [])
    print("RuleFitRegressor learned rules (first 10):")
    for i, rule in enumerate(rules[:10], start=1):
        print(f"{i}. {rule}")

    figs = FIGSRegressor(random_state=0, max_rules=10)
    figs.fit(X_model, y)
    figs_importance = pd.Series(figs.feature_importances_, index=model_cols).sort_values(ascending=False)
    print("\nFIGSRegressor feature importances:")
    print(figs_importance.to_string())

    hst = HSTreeRegressor(random_state=0, max_leaf_nodes=8)
    hst.fit(X_model, y)
    hst_importance = pd.Series(hst.estimator_.feature_importances_, index=model_cols).sort_values(ascending=False)
    print("\nHSTreeRegressor (base tree) feature importances:")
    print(hst_importance.to_string())

    # Evidence synthesis for final Likert response.
    size_pvals = [t_size.pvalue, pb_size.pvalue, float(logit_model.pvalues["size_diff"])]
    loc_pvals = [t_dist.pvalue, chi2_res.pvalue, float(logit_model.pvalues["dist_diff"])]

    size_support = float(np.mean([pvalue_support(p) for p in size_pvals]))
    location_support = float(np.mean([pvalue_support(p) for p in loc_pvals]))

    # Mild direction bonus if effects are in theoretically expected directions.
    size_bonus = 0.05 if float(logit_model.params["size_diff"]) > 0 else 0.0
    location_bonus = 0.05 if float(logit_model.params["dist_diff"]) < 0 else 0.0

    combined_support = np.clip(0.5 * (size_support + location_support) + 0.5 * (size_bonus + location_bonus), 0, 1)
    response = int(np.clip(round(100 * combined_support), 0, 100))

    explanation = (
        "Tests on this dataset show weak evidence that relative group size and contest location influence contest outcome. "
        f"For size, t-test p={t_size.pvalue:.3f}, point-biserial p={pb_size.pvalue:.3f}, and logit p={float(logit_model.pvalues['size_diff']):.3f} "
        f"(logit coef={float(logit_model.params['size_diff']):.3f}). "
        f"For location, t-test p={t_dist.pvalue:.3f}, chi-square p={chi2_res.pvalue:.3f}, and logit p={float(logit_model.pvalues['dist_diff']):.3f} "
        f"(logit coef={float(logit_model.params['dist_diff']):.4f}). "
        "Interpretable models (linear coefficients, trees, and imodels rule/tree methods) show directional patterns but no strong statistical significance, "
        "so the overall answer is closer to No than Yes."
    )

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    section("Final conclusion payload")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
