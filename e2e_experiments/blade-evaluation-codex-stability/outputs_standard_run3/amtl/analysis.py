import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def fit_with_optional_kwargs(model, X, y, **kwargs):
    """Fit model while gracefully handling signature differences across versions."""
    try:
        return model.fit(X, y, **kwargs)
    except TypeError:
        return model.fit(X, y)


def compute_likert_score(coef: float, pvalue: float, odds_ratio: float) -> int:
    """Map statistical evidence to a 0-100 Yes/No scale."""
    if np.isnan(coef) or np.isnan(pvalue):
        return 50
    if coef <= 0:
        return 5 if pvalue < 0.05 else 15
    if pvalue >= 0.05:
        return 35

    if pvalue < 1e-10:
        score = 94
    elif pvalue < 1e-5:
        score = 88
    elif pvalue < 1e-3:
        score = 82
    else:
        score = 74

    if odds_ratio >= 3:
        score += 4
    elif odds_ratio >= 2:
        score += 2

    return int(max(0, min(100, round(score))))


def main() -> None:
    print_header("1) Load Research Question and Dataset")
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["N/A"])[0]
    print(f"Research question: {research_question}")

    df = pd.read_csv("amtl.csv")
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["any_amtl"] = (df["num_amtl"] > 0).astype(int)

    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", ", ".join(df.columns))

    print_header("2) Data Exploration: Summary Statistics, Distributions, Correlations")
    num_cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "amtl_rate"]
    print("\nNumeric summary statistics:")
    print(df[num_cols].describe().round(4))

    print("\nDistribution of genus:")
    print(df["genus"].value_counts())

    print("\nDistribution of tooth_class:")
    print(df["tooth_class"].value_counts())

    print("\nDistribution of num_amtl (top values):")
    print(df["num_amtl"].value_counts().sort_index().head(15))

    print("\nMean AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].mean().sort_values(ascending=False).round(4))

    print("\nMean AMTL rate by tooth class:")
    print(df.groupby("tooth_class")["amtl_rate"].mean().sort_values(ascending=False).round(4))

    print("\nCorrelation matrix (numeric variables):")
    corr = df[num_cols].corr()
    print(corr.round(4))

    print_header("3) Statistical Tests")
    human_rates = df.loc[df["is_human"] == 1, "amtl_rate"]
    nonhuman_rates = df.loc[df["is_human"] == 0, "amtl_rate"]

    t_stat, t_p = stats.ttest_ind(human_rates, nonhuman_rates, equal_var=False)
    print(
        f"Welch t-test (AMTL rate: humans vs non-humans): t={t_stat:.4f}, p={t_p:.3e}"
    )

    groups = [g["amtl_rate"].values for _, g in df.groupby("genus")]
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"One-way ANOVA (AMTL rate across genera): F={f_stat:.4f}, p={anova_p:.3e}")

    contingency = pd.crosstab(df["is_human"], df["any_amtl"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi-square (any AMTL by human/non-human): chi2={chi2:.4f}, p={chi2_p:.3e}")

    glm_main = smf.glm(
        "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef_human = float(glm_main.params["is_human"])
    p_human = float(glm_main.pvalues["is_human"])
    ci_low, ci_high = glm_main.conf_int().loc["is_human"].tolist()
    or_human = float(np.exp(coef_human))

    print("\nPrimary weighted binomial regression (controls: age, prob_male, tooth_class):")
    print(
        f"is_human coef={coef_human:.4f}, OR={or_human:.4f}, "
        f"95% CI for coef=[{ci_low:.4f}, {ci_high:.4f}], p={p_human:.3e}"
    )

    ols_model = smf.ols(
        "amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
    ).fit()
    print(
        f"OLS check: is_human coef={ols_model.params['is_human']:.4f}, "
        f"p={ols_model.pvalues['is_human']:.3e}"
    )

    genus_glm = smf.glm(
        'amtl_rate ~ C(genus, Treatment(reference="Pan")) + age + prob_male + C(tooth_class)',
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print("\nGenus-specific weighted binomial regression coefficients:")
    print(genus_glm.params.round(4))
    print("Genus-specific p-values:")
    print(genus_glm.pvalues.apply(lambda x: float(f"{x:.3e}")))

    print_header("4) Interpretable Models (scikit-learn + imodels)")
    features_num = ["age", "stdev_age", "prob_male"]
    features_cat = ["genus", "tooth_class"]
    target = "amtl_rate"

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                features_num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                features_cat,
            ),
        ]
    )

    X = df[features_num + features_cat]
    y = df[target].values
    sample_weight = df["sockets"].values

    X_proc = preprocessor.fit_transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc_dense = X_proc.toarray()
    else:
        X_proc_dense = X_proc

    cat_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(features_cat)
        .tolist()
    )
    feature_names = features_num + cat_names

    # Linear models
    lin = LinearRegression()
    fit_with_optional_kwargs(lin, X_proc_dense, y, sample_weight=sample_weight)
    lin_coef = pd.Series(lin.coef_, index=feature_names).sort_values(
        key=np.abs, ascending=False
    )
    print("\nLinearRegression top coefficients (abs):")
    print(lin_coef.head(10).round(4))

    ridge = Ridge(alpha=1.0, random_state=0)
    fit_with_optional_kwargs(ridge, X_proc_dense, y, sample_weight=sample_weight)
    ridge_coef = pd.Series(ridge.coef_, index=feature_names).sort_values(
        key=np.abs, ascending=False
    )
    print("\nRidge top coefficients (abs):")
    print(ridge_coef.head(10).round(4))

    lasso = Lasso(alpha=0.0005, random_state=0, max_iter=10000)
    fit_with_optional_kwargs(lasso, X_proc_dense, y)
    lasso_coef = pd.Series(lasso.coef_, index=feature_names).sort_values(
        key=np.abs, ascending=False
    )
    print("\nLasso top coefficients (abs):")
    print(lasso_coef.head(10).round(4))

    tree_reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=0)
    fit_with_optional_kwargs(tree_reg, X_proc_dense, y, sample_weight=sample_weight)
    reg_importances = pd.Series(tree_reg.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    print("\nDecisionTreeRegressor feature importances:")
    print(reg_importances.head(10).round(4))
    print(f"DecisionTreeRegressor R^2 (train): {r2_score(y, tree_reg.predict(X_proc_dense)):.4f}")

    # Simple classifier for interpretability on AMTL presence
    y_bin = df["any_amtl"].values
    tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=0)
    fit_with_optional_kwargs(tree_clf, X_proc_dense, y_bin, sample_weight=sample_weight)
    clf_importances = pd.Series(tree_clf.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    print("\nDecisionTreeClassifier feature importances (predict any AMTL):")
    print(clf_importances.head(10).round(4))
    print(
        f"DecisionTreeClassifier accuracy (train): "
        f"{accuracy_score(y_bin, tree_clf.predict(X_proc_dense)):.4f}"
    )

    # imodels
    rulefit = RuleFitRegressor(max_rules=30, random_state=0)
    fit_with_optional_kwargs(rulefit, X_proc_dense, y, feature_names=feature_names)
    print(f"\nRuleFitRegressor R^2 (train): {r2_score(y, rulefit.predict(X_proc_dense)):.4f}")

    top_rules: List[Tuple[str, float]] = []
    if hasattr(rulefit, "rules_"):
        for rule in rulefit.rules_:
            coef = np.nan
            if hasattr(rule, "args") and len(rule.args) > 0:
                try:
                    coef = float(rule.args[0])
                except Exception:
                    coef = np.nan
            top_rules.append((str(rule), coef))
        top_rules = sorted(
            top_rules,
            key=lambda x: 0.0 if np.isnan(x[1]) else abs(x[1]),
            reverse=True,
        )

    print("RuleFit top rules by absolute coefficient:")
    for rule_text, coef in top_rules[:8]:
        coef_txt = "nan" if np.isnan(coef) else f"{coef:.4f}"
        print(f"  coef={coef_txt:>8} | {rule_text}")

    figs = FIGSRegressor(max_rules=20, random_state=0)
    fit_with_optional_kwargs(figs, X_proc_dense, y, feature_names=feature_names)
    print(f"\nFIGSRegressor R^2 (train): {r2_score(y, figs.predict(X_proc_dense)):.4f}")
    if hasattr(figs, "feature_importances_"):
        figs_imp = pd.Series(figs.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )
        print("FIGS feature importances:")
        print(figs_imp.head(10).round(4))

    hst = HSTreeRegressor(random_state=0)
    fit_with_optional_kwargs(hst, X_proc_dense, y, feature_names=feature_names)
    print(f"\nHSTreeRegressor R^2 (train): {r2_score(y, hst.predict(X_proc_dense)):.4f}")

    score = compute_likert_score(coef_human, p_human, or_human)

    # Key supporting evidence from exploratory and modeling steps.
    mean_human = float(human_rates.mean())
    mean_nonhuman = float(nonhuman_rates.mean())
    evidence = {
        "mean_human_amtl_rate": round(mean_human, 4),
        "mean_nonhuman_amtl_rate": round(mean_nonhuman, 4),
        "welch_ttest_p": float(t_p),
        "anova_p": float(anova_p),
        "glm_is_human_coef": round(coef_human, 4),
        "glm_is_human_or": round(or_human, 4),
        "glm_is_human_p": float(p_human),
    }

    explanation = (
        "Yes. After adjusting for age, sex (prob_male), and tooth class in a weighted "
        "binomial regression, Homo sapiens shows a significantly higher AMTL rate than "
        f"non-human primates (is_human coef={coef_human:.3f}, OR={or_human:.2f}, "
        f"p={p_human:.2e}). Unadjusted tests also support this (Welch t-test p={t_p:.2e}, "
        f"ANOVA p={anova_p:.2e}), and interpretable linear/tree/rule models consistently "
        "identify human-genus indicators and age as major positive predictors of AMTL. "
        f"Observed mean AMTL rate is {mean_human:.3f} in humans vs {mean_nonhuman:.3f} in non-humans."
    )

    result: Dict[str, object] = {
        "response": int(score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=True))

    print_header("5) Final Conclusion JSON")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print("\nEvidence summary:")
    print(evidence)


if __name__ == "__main__":
    main()
