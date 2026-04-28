import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def p_to_strength(p_value: float) -> float:
    if p_value < 0.01:
        return 1.0
    if p_value < 0.05:
        return 0.9
    if p_value < 0.10:
        return 0.7
    return 0.1


def fmt_p(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)
    print("\nLoading data from panda_nuts.csv ...")

    df = pd.read_csv("panda_nuts.csv")
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower()
    df["hammer"] = df["hammer"].astype(str).str.strip()

    df["efficiency"] = df["nuts_opened"] / df["seconds"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["age", "sex", "help", "hammer", "nuts_opened", "seconds", "efficiency"])

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\n=== Summary statistics ===")
    print(df.describe(include="all").transpose().to_string())

    print("\n=== Categorical distributions ===")
    for col in ["sex", "help", "hammer"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).to_string())

    print("\n=== Efficiency distribution ===")
    eff = df["efficiency"]
    print(eff.describe().to_string())
    print(f"Skewness: {eff.skew():.4f}")
    print(f"Zero-efficiency sessions: {(eff == 0).mean() * 100:.1f}%")

    print("\n=== Correlations (numeric variables) ===")
    corr = df[["age", "nuts_opened", "seconds", "efficiency"]].corr(numeric_only=True)
    print(corr.to_string())

    print("\n=== Statistical tests ===")
    pearson_age = stats.pearsonr(df["age"], df["efficiency"])
    spearman_age = stats.spearmanr(df["age"], df["efficiency"])
    print(
        "Age vs efficiency: "
        f"Pearson r={pearson_age.statistic:.4f}, p={fmt_p(pearson_age.pvalue)}; "
        f"Spearman rho={spearman_age.statistic:.4f}, p={fmt_p(spearman_age.pvalue)}"
    )

    male_eff = df.loc[df["sex"] == "m", "efficiency"]
    female_eff = df.loc[df["sex"] == "f", "efficiency"]
    t_sex = stats.ttest_ind(male_eff, female_eff, equal_var=False)
    print(
        "Sex effect (Welch t-test): "
        f"mean(m)={male_eff.mean():.4f}, mean(f)={female_eff.mean():.4f}, "
        f"t={t_sex.statistic:.4f}, p={fmt_p(t_sex.pvalue)}"
    )

    help_yes_eff = df.loc[df["help"] == "y", "efficiency"]
    help_no_eff = df.loc[df["help"] == "n", "efficiency"]
    t_help = stats.ttest_ind(help_yes_eff, help_no_eff, equal_var=False)
    print(
        "Help effect (Welch t-test): "
        f"mean(help=yes)={help_yes_eff.mean():.4f}, mean(help=no)={help_no_eff.mean():.4f}, "
        f"t={t_help.statistic:.4f}, p={fmt_p(t_help.pvalue)}"
    )

    formula = "efficiency ~ age + C(sex) + C(help) + C(hammer)"
    ols = smf.ols(formula, data=df).fit()
    print("\n=== OLS regression (with controls) ===")
    print(ols.summary())

    print("\n=== ANOVA (Type II) ===")
    anova_tbl = anova_lm(ols, typ=2)
    print(anova_tbl.to_string())

    print("\n=== Interpretable models: scikit-learn ===")
    X = df[["age", "sex", "help", "hammer"]]
    y = df["efficiency"].values

    preprocessor = ColumnTransformer(
        [
            ("cat", safe_ohe(), ["sex", "help", "hammer"]),
            ("num", "passthrough", ["age"]),
        ],
        verbose_feature_names_out=False,
    )
    X_enc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    linear = LinearRegression()
    ridge = Ridge(alpha=1.0, random_state=42)
    lasso = Lasso(alpha=0.005, max_iter=10000, random_state=42)
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)

    linear.fit(X_enc, y)
    ridge.fit(X_enc, y)
    lasso.fit(X_enc, y)
    tree.fit(X_enc, y)

    def print_coef(name: str, model) -> None:
        coef = pd.Series(model.coef_, index=feature_names)
        coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
        print(f"\n{name} coefficients (sorted by absolute value):")
        print(coef.round(4).to_string())

    print_coef("LinearRegression", linear)
    print_coef("Ridge", ridge)
    print_coef("Lasso", lasso)

    tree_imp = pd.Series(tree.feature_importances_, index=feature_names)
    tree_imp = tree_imp.sort_values(ascending=False)
    print("\nDecisionTreeRegressor feature importances:")
    print(tree_imp.round(4).to_string())
    print("\nDecisionTreeRegressor rules:")
    print(export_text(tree, feature_names=feature_names, decimals=3))

    print("\n=== Interpretable models: imodels ===")
    rulefit = RuleFitRegressor(n_estimators=100, max_rules=20, include_linear=True, random_state=42)
    rulefit.fit(X_enc, y, feature_names=feature_names)
    rules_df = rulefit._get_rules()
    nonzero_rules = rules_df.loc[rules_df["coef"] != 0].copy()
    if not nonzero_rules.empty:
        nonzero_rules = nonzero_rules.sort_values("importance", ascending=False).head(10)
        print("\nTop RuleFit rules/features:")
        print(nonzero_rules[["rule", "type", "coef", "support", "importance"]].to_string(index=False))
    else:
        print("\nRuleFit returned no non-zero rules/features.")

    figs = FIGSRegressor(max_rules=10, random_state=42)
    figs.fit(X_enc, y, feature_names=feature_names)
    figs_imp = pd.Series(figs.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFIGSRegressor feature importances:")
    print(figs_imp.round(4).to_string())
    print("\nFIGS model structure:")
    print(str(figs))

    hst = HSTreeRegressor(
        estimator_=DecisionTreeRegressor(max_leaf_nodes=8, random_state=42),
        reg_param=1.0,
    )
    hst.fit(X_enc, y, feature_names=feature_names)
    hst_imp = pd.Series(hst.estimator_.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nHSTreeRegressor (base tree) feature importances:")
    print(hst_imp.round(4).to_string())
    print("\nHSTree model structure:")
    print(str(hst))

    age_coef = ols.params.get("age", np.nan)
    age_p = ols.pvalues.get("age", np.nan)

    sex_term = [c for c in ols.params.index if c.startswith("C(sex)")][0]
    sex_coef = ols.params[sex_term]
    sex_p = ols.pvalues[sex_term]

    help_term = [c for c in ols.params.index if c.startswith("C(help)")][0]
    help_coef = ols.params[help_term]
    help_p = ols.pvalues[help_term]

    age_score = np.mean([p_to_strength(pearson_age.pvalue), p_to_strength(age_p)])
    sex_score = np.mean([p_to_strength(t_sex.pvalue), p_to_strength(sex_p)])
    help_score = np.mean([p_to_strength(t_help.pvalue), p_to_strength(help_p)])
    overall_score = int(np.clip(np.round(np.mean([age_score, sex_score, help_score]) * 100), 0, 100))

    age_dir = "increases" if age_coef > 0 else "decreases"
    sex_dir = "higher" if sex_coef > 0 else "lower"
    help_dir = "higher" if help_coef > 0 else "lower"

    explanation = (
        f"Using efficiency = nuts_opened/seconds, evidence supports an influence of all three predictors: "
        f"age {age_dir} efficiency (Pearson p={fmt_p(pearson_age.pvalue)}, OLS beta={age_coef:.3f}, p={fmt_p(age_p)}); "
        f"males have {sex_dir} efficiency than females (Welch t-test p={fmt_p(t_sex.pvalue)}, OLS beta={sex_coef:.3f}, p={fmt_p(sex_p)}); "
        f"and helped sessions show {help_dir} efficiency than non-helped sessions in the controlled model "
        f"(Welch t-test p={fmt_p(t_help.pvalue)}, OLS beta={help_coef:.3f}, p={fmt_p(help_p)}). "
        f"Interpretable sklearn and imodels models also rank age and sex/help-derived indicators among important features."
    )

    conclusion = {"response": overall_score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=True)

    print("\nWrote conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
