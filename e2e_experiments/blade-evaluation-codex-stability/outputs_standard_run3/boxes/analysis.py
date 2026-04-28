import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def sort_named(values, names, top_n=6):
    pairs = [(str(n), float(v)) for n, v in zip(names, values)]
    return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]


def main():
    info = json.loads(Path("info.json").read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv("boxes.csv")
    required = {"y", "gender", "age", "majority_first", "culture"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("Research question:", question)
    print("\nData shape:", df.shape)
    print("\nMissing values per column:\n", df.isna().sum())

    print("\nSummary statistics:\n", df.describe(include="all"))

    print("\nOutcome distribution (y):\n", df["y"].value_counts().sort_index())
    print(
        "\nMajority choice rate overall:",
        round(float(df["majority_choice"].mean()), 4),
    )
    print(
        "\nMajority choice rate by culture:\n",
        df.groupby("culture")["majority_choice"].mean().sort_index(),
    )
    print(
        "\nMajority choice rate by age:\n",
        df.groupby("age")["majority_choice"].mean().sort_index(),
    )

    corr_cols = ["y", "majority_choice", "gender", "age", "majority_first", "culture"]
    print("\nCorrelation matrix:\n", df[corr_cols].corr())

    pearson_r, pearson_p = stats.pearsonr(df["age"], df["majority_choice"])
    spearman_rho, spearman_p = stats.spearmanr(df["age"], df["majority_choice"])

    age_majority = df.loc[df["majority_choice"] == 1, "age"]
    age_not_majority = df.loc[df["majority_choice"] == 0, "age"]
    t_stat, t_p = stats.ttest_ind(age_majority, age_not_majority, equal_var=False)

    groups = [df.loc[df["y"] == level, "age"] for level in sorted(df["y"].unique())]
    anova_f, anova_p = stats.f_oneway(*groups)

    contingency = pd.crosstab(df["culture"], df["majority_choice"])
    chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)

    ols = smf.ols(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=df
    ).fit()
    logit = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    logit_interaction = smf.logit(
        "majority_choice ~ age * C(culture) + gender + majority_first", data=df
    ).fit(disp=False, maxiter=300)

    lr_stat = 2 * (logit_interaction.llf - logit.llf)
    lr_df = float(logit_interaction.df_model - logit.df_model)
    lr_p = float(stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan

    print("\nStatistical tests:")
    print(f"Pearson(age, majority_choice): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman(age, majority_choice): rho={spearman_rho:.4f}, p={spearman_p:.4g}")
    print(f"Welch t-test age by majority choice: t={t_stat:.4f}, p={t_p:.4g}")
    print(f"ANOVA age across y classes: F={anova_f:.4f}, p={anova_p:.4g}")
    print(
        f"Chi-square culture vs majority choice: chi2={chi2_stat:.4f}, "
        f"dof={chi2_dof}, p={chi2_p:.4g}"
    )
    print(
        f"Logit age coefficient: beta={logit.params['age']:.4f}, "
        f"p={logit.pvalues['age']:.4g}"
    )
    print(
        f"Logit majority_first coefficient: beta={logit.params['majority_first']:.4f}, "
        f"p={logit.pvalues['majority_first']:.4g}"
    )
    print(
        f"Likelihood-ratio test for age*culture interaction: "
        f"LR={lr_stat:.4f}, dof={int(lr_df)}, p={lr_p:.4g}"
    )

    X = pd.get_dummies(
        df[["gender", "age", "majority_first", "culture"]].astype({"culture": "category"}),
        drop_first=True,
    )
    y = df["majority_choice"].astype(float)
    feature_names = list(X.columns)

    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X, y)
    lasso = Lasso(alpha=0.01, max_iter=20000, random_state=RANDOM_STATE).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE)
    tree.fit(X, y.astype(int))

    print("\nInterpretable sklearn model signals:")
    print("Top LinearRegression coefficients:", sort_named(lin.coef_, feature_names))
    print("Top Ridge coefficients:", sort_named(ridge.coef_, feature_names))
    print("Top Lasso coefficients:", sort_named(lasso.coef_, feature_names))
    print(
        "Top DecisionTreeClassifier importances:",
        sort_named(tree.feature_importances_, feature_names),
    )

    rulefit = RuleFitRegressor(random_state=RANDOM_STATE, max_rules=40)
    rulefit.fit(X, y)
    n_features = len(feature_names)
    linear_part = np.array(rulefit.coef[:n_features], dtype=float)
    rule_part = np.array(rulefit.coef[n_features:], dtype=float)
    top_rulefit_linear = sort_named(linear_part, feature_names)
    rule_pairs = [
        (str(rule), float(coef))
        for rule, coef in zip(rulefit.rules_, rule_part)
        if abs(float(coef)) > 1e-8
    ]
    top_rulefit_rules = sorted(rule_pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

    figs = FIGSRegressor(random_state=RANDOM_STATE, max_rules=12)
    figs.fit(X, y)
    top_figs = sort_named(figs.feature_importances_, feature_names)

    hst = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=8)
    hst.fit(X, y)
    hst_importances = hst.estimator_.feature_importances_
    top_hst = sort_named(hst_importances, feature_names)

    print("\nInterpretable imodels signals:")
    print("Top RuleFit linear terms:", top_rulefit_linear)
    print("Top RuleFit rules:", top_rulefit_rules)
    print("Top FIGS importances:", top_figs)
    print("Top HSTree importances:", top_hst)

    # Likert score calibrated from significance and interpretability signals.
    score = 50
    age_beta = float(logit.params.get("age", np.nan))
    age_p = float(logit.pvalues.get("age", np.nan))

    if np.isfinite(age_p):
        if age_p < 0.05 and age_beta > 0:
            score += 25
        elif age_p < 0.05 and age_beta <= 0:
            score -= 10
        else:
            score -= 15

    if np.isfinite(spearman_p):
        if spearman_p < 0.05 and spearman_rho > 0:
            score += 12
        elif spearman_p < 0.05 and spearman_rho <= 0:
            score -= 8
        else:
            score -= 7

    if np.isfinite(lr_p):
        score += 12 if lr_p < 0.05 else -5

    if np.isfinite(chi2_p):
        score += 8 if chi2_p < 0.05 else -3

    age_idx = feature_names.index("age")
    age_small_in_models = (
        abs(float(lin.coef_[age_idx])) < 0.03
        and float(tree.feature_importances_[age_idx]) < 0.2
        and float(figs.feature_importances_[age_idx]) < 0.2
    )
    if age_small_in_models:
        score -= 3

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Evidence for increasing reliance on majority preference with age across cultures is weak: "
        f"age was not significant in logistic regression (beta={age_beta:.3f}, p={age_p:.3g}), "
        f"age-majority correlations were near zero (Pearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f}), "
        f"and age-by-culture interaction was not significant (LR p={lr_p:.3g}). "
        f"Culture differences in majority choice were also not significant by chi-square (p={chi2_p:.3g}). "
        f"Interpretable models consistently highlighted majority_first and gender above age."
    )

    payload = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(payload), encoding="utf-8")
    print("\nWrote conclusion.txt:", payload)


if __name__ == "__main__":
    main()
