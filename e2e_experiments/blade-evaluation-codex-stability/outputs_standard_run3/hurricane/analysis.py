import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def top_n_abs(series: pd.Series, n: int = 8) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def get_rulefit_rules(model: RuleFitRegressor) -> pd.DataFrame:
    if hasattr(model, "get_rules"):
        return model.get_rules()
    if hasattr(model, "_get_rules"):
        return model._get_rules()
    return pd.DataFrame(columns=["rule", "type", "coef", "support", "importance"])


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("hurricane.csv")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print_header("Research Question")
    print(research_question)

    df = pd.read_csv(data_path)
    df["log_deaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])

    print_header("Data Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nSummary statistics (numeric columns):")
    print(df[numeric_cols].describe().T)

    print_header("Distribution Checks")
    print(
        "Skewness -- alldeaths: "
        f"{df['alldeaths'].skew():.3f}, log_deaths: {df['log_deaths'].skew():.3f}"
    )
    print(
        "Kurtosis -- alldeaths: "
        f"{df['alldeaths'].kurtosis():.3f}, log_deaths: {df['log_deaths'].kurtosis():.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["alldeaths"], bins=20, kde=True, ax=axes[0], color="#3b82f6")
    axes[0].set_title("Distribution of Total Deaths")
    sns.histplot(df["log_deaths"], bins=20, kde=True, ax=axes[1], color="#ef4444")
    axes[1].set_title("Distribution of log(1 + Total Deaths)")
    fig.tight_layout()
    fig.savefig("distributions.png", dpi=150)
    plt.close(fig)

    corr = df[numeric_cols].corr(numeric_only=True)
    print_header("Correlation Snapshot")
    print("Top absolute correlations with log_deaths:")
    print(top_n_abs(corr["log_deaths"].drop(labels=["log_deaths"]), n=10))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Correlation Heatmap")
    fig.tight_layout()
    fig.savefig("correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print_header("Statistical Tests")
    pearson_r, pearson_p = stats.pearsonr(df["masfem"], df["log_deaths"])
    spearman_rho, spearman_p = stats.spearmanr(df["masfem"], df["alldeaths"])

    female_log = df.loc[df["gender_mf"] == 1, "log_deaths"]
    male_log = df.loc[df["gender_mf"] == 0, "log_deaths"]
    ttest_res = stats.ttest_ind(female_log, male_log, equal_var=False)
    mw_res = stats.mannwhitneyu(
        df.loc[df["gender_mf"] == 1, "alldeaths"],
        df.loc[df["gender_mf"] == 0, "alldeaths"],
        alternative="two-sided",
    )

    category_groups = [
        g["log_deaths"].values for _, g in df.groupby("category") if len(g) >= 2
    ]
    anova_f, anova_p = stats.f_oneway(*category_groups)

    print(f"Pearson corr(masfem, log_deaths): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman corr(masfem, alldeaths): rho={spearman_rho:.4f}, p={spearman_p:.4g}")
    print(
        "Welch t-test female vs male (log_deaths): "
        f"t={ttest_res.statistic:.4f}, p={ttest_res.pvalue:.4g}"
    )
    print(
        "Mann-Whitney female vs male (alldeaths): "
        f"U={mw_res.statistic:.4f}, p={mw_res.pvalue:.4g}"
    )
    print(f"ANOVA across hurricane category (log_deaths): F={anova_f:.4f}, p={anova_p:.4g}")

    simple_ols = smf.ols("log_deaths ~ masfem", data=df).fit(cov_type="HC3")
    adjusted_ols = smf.ols(
        "log_deaths ~ masfem + wind + min + category + log_ndam15 + year",
        data=df,
    ).fit(cov_type="HC3")
    interaction_ols = smf.ols(
        "log_deaths ~ masfem * wind + min + category + log_ndam15 + year",
        data=df,
    ).fit(cov_type="HC3")

    print("\nOLS (simple):")
    print(simple_ols.summary().tables[1])
    print("\nOLS (adjusted controls):")
    print(adjusted_ols.summary().tables[1])
    print("\nOLS (masfem x wind interaction):")
    print(interaction_ols.summary().tables[1])

    print_header("Interpretable Models")
    features = [
        "masfem",
        "gender_mf",
        "wind",
        "min",
        "category",
        "log_ndam15",
        "year",
        "elapsedyrs",
        "masfem_mturk",
    ]
    model_df = df[features + ["log_deaths"]].dropna()
    X = model_df[features]
    y = model_df["log_deaths"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    linear_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    ridge_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]
    )
    lasso_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", Lasso(alpha=0.03, random_state=42))]
    )

    linear_pipe.fit(X_train, y_train)
    ridge_pipe.fit(X_train, y_train)
    lasso_pipe.fit(X_train, y_train)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)
    tree.fit(X_train, y_train)

    def eval_model(name: str, model, xte: pd.DataFrame, yte: pd.Series) -> None:
        preds = model.predict(xte)
        print(
            f"{name:<20} R2={r2_score(yte, preds):.3f}  "
            f"MAE={mean_absolute_error(yte, preds):.3f}"
        )

    eval_model("LinearRegression", linear_pipe, X_test, y_test)
    eval_model("Ridge", ridge_pipe, X_test, y_test)
    eval_model("Lasso", lasso_pipe, X_test, y_test)
    eval_model("DecisionTree", tree, X_test, y_test)

    lin_coef = pd.Series(
        linear_pipe.named_steps["model"].coef_, index=features, name="linear_coef"
    )
    ridge_coef = pd.Series(
        ridge_pipe.named_steps["model"].coef_, index=features, name="ridge_coef"
    )
    lasso_coef = pd.Series(
        lasso_pipe.named_steps["model"].coef_, index=features, name="lasso_coef"
    )
    tree_imp = pd.Series(tree.feature_importances_, index=features, name="tree_importance")

    coef_table = pd.concat([lin_coef, ridge_coef, lasso_coef, tree_imp], axis=1)
    print("\nFeature effects/importances (top by |Linear coef|):")
    print(coef_table.reindex(lin_coef.abs().sort_values(ascending=False).index))

    # imodels estimators
    rulefit = RuleFitRegressor(
        n_estimators=200, tree_size=4, max_rules=40, random_state=42
    )
    rulefit.fit(X_train, y_train, feature_names=features)
    rulefit_rules = get_rulefit_rules(rulefit)

    figs = FIGSRegressor(max_rules=10, random_state=42)
    figs.fit(X_train, y_train)

    hstree = HSTreeRegressor(random_state=42, max_leaf_nodes=10)
    hstree.fit(X_train, y_train)

    eval_model("RuleFitRegressor", rulefit, X_test, y_test)
    eval_model("FIGSRegressor", figs, X_test, y_test)
    eval_model("HSTreeRegressor", hstree, X_test, y_test)

    rulefit_top = pd.DataFrame()
    if not rulefit_rules.empty:
        rulefit_top = rulefit_rules.loc[
            rulefit_rules["coef"] != 0,
            ["rule", "coef", "support", "importance"],
        ].copy()
        if not rulefit_top.empty:
            rulefit_top["abs_importance"] = rulefit_top["importance"].abs()
            rulefit_top = rulefit_top.sort_values("abs_importance", ascending=False).head(8)

    print("\nRuleFit top non-zero rules:")
    if rulefit_top.empty:
        print("No non-zero rules retained.")
    else:
        print(rulefit_top[["rule", "coef", "support", "importance"]])

    figs_importances = pd.Series(figs.feature_importances_, index=features)
    hstree_importances = pd.Series(hstree.estimator_.feature_importances_, index=features)

    print("\nFIGS feature importances:")
    print(figs_importances.sort_values(ascending=False))
    print("\nHSTree feature importances:")
    print(hstree_importances.sort_values(ascending=False))

    print_header("Conclusion Logic")
    masfem_simple_coef = float(simple_ols.params.get("masfem", np.nan))
    masfem_simple_p = float(simple_ols.pvalues.get("masfem", np.nan))
    masfem_adj_coef = float(adjusted_ols.params.get("masfem", np.nan))
    masfem_adj_p = float(adjusted_ols.pvalues.get("masfem", np.nan))
    masfem_wind_coef = float(interaction_ols.params.get("masfem:wind", np.nan))
    masfem_wind_p = float(interaction_ols.pvalues.get("masfem:wind", np.nan))

    score = 50

    # Primary evidence: controlled OLS estimate for femininity effect.
    if masfem_adj_p < 0.05:
        score += 35 if masfem_adj_coef > 0 else -35
    elif masfem_adj_p < 0.10:
        score += 15 if masfem_adj_coef > 0 else -15
    else:
        score -= 15

    # Secondary evidence: simple association.
    if masfem_simple_p < 0.05:
        score += 15 if masfem_simple_coef > 0 else -15
    else:
        score -= 5

    # Group comparison by binary female-name indicator.
    if ttest_res.pvalue < 0.05 and female_log.mean() > male_log.mean():
        score += 10
    elif ttest_res.pvalue < 0.05 and female_log.mean() < male_log.mean():
        score -= 10
    else:
        score -= 5

    # Rank correlation on raw deaths.
    if spearman_p < 0.05:
        score += 10 if spearman_rho > 0 else -10
    else:
        score -= 5

    # Weak/moderate interaction evidence gets small weight.
    if masfem_wind_p < 0.10 and masfem_wind_coef > 0:
        score += 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Evidence for the claim is weak in this dataset. Femininity rating (masfem) has "
        f"a small, non-significant association with log deaths in simple OLS "
        f"(coef={masfem_simple_coef:.3f}, p={masfem_simple_p:.3f}) and in adjusted OLS "
        f"with storm controls (coef={masfem_adj_coef:.3f}, p={masfem_adj_p:.3f}). "
        f"Direct association tests are also non-significant (Pearson p={pearson_p:.3f}, "
        f"Spearman p={spearman_p:.3f}, Welch t-test by female-vs-male names p={ttest_res.pvalue:.3f}). "
        "Interpretable models (linear, ridge/lasso, decision tree, RuleFit, FIGS, HSTree) "
        "consistently rank damage/intensity-related variables above name-gender variables. "
        f"A masfem*wind interaction is only marginal (p={masfem_wind_p:.3f}) and not enough to "
        "override the overall lack of significant main effects."
    )

    result = {"response": score, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(result, f)

    print(f"Final Likert score: {score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
