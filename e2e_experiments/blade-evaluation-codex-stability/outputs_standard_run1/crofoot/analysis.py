import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize_distribution(series: pd.Series) -> dict:
    return {
        "mean": safe_float(series.mean()),
        "std": safe_float(series.std()),
        "min": safe_float(series.min()),
        "q25": safe_float(series.quantile(0.25)),
        "median": safe_float(series.median()),
        "q75": safe_float(series.quantile(0.75)),
        "max": safe_float(series.max()),
    }


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def main():
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0]

    df = pd.read_csv("crofoot.csv")

    # Feature engineering centered on the research question.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["location_advantage"] = df["dist_other"] - df["dist_focal"]
    df["location_advantage_100m"] = df["location_advantage"] / 100.0
    df["rel_males"] = df["m_focal"] - df["m_other"]
    df["rel_females"] = df["f_focal"] - df["f_other"]
    df["size_category"] = np.select(
        [df["rel_group_size"] < 0, df["rel_group_size"] == 0, df["rel_group_size"] > 0],
        ["smaller", "equal", "larger"],
        default="equal",
    )
    df["location_side"] = np.where(df["location_advantage"] > 0, "closer_to_focal", "closer_to_other")

    print("Research question:", question)
    print("\nData shape:", df.shape)
    print("Missing values by column:\n", df.isna().sum())

    key_cols = [
        "win",
        "rel_group_size",
        "location_advantage",
        "dist_focal",
        "dist_other",
        "n_focal",
        "n_other",
    ]

    print("\nSummary statistics (key variables):")
    print(df[key_cols].describe().round(3))

    print("\nDistribution summaries:")
    dist_summary = {
        "rel_group_size": summarize_distribution(df["rel_group_size"]),
        "location_advantage": summarize_distribution(df["location_advantage"]),
        "win_rate": safe_float(df["win"].mean()),
    }
    print(json.dumps(dist_summary, indent=2))

    corr_cols = ["win", "rel_group_size", "location_advantage", "rel_males", "rel_females"]
    corr = df[corr_cols].corr(numeric_only=True)
    print("\nCorrelation matrix:")
    print(corr.round(3))

    # Statistical tests aligned to the question.
    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]

    t_size = stats.ttest_ind(
        wins["rel_group_size"], losses["rel_group_size"], equal_var=False, nan_policy="omit"
    )
    t_loc = stats.ttest_ind(
        wins["location_advantage"], losses["location_advantage"], equal_var=False, nan_policy="omit"
    )

    r_size, p_size_corr = stats.pearsonr(df["rel_group_size"], df["win"])
    r_loc, p_loc_corr = stats.pearsonr(df["location_advantage"], df["win"])

    size_groups = [g["win"].values for _, g in df.groupby("size_category") if len(g) > 1]
    anova_size = stats.f_oneway(*size_groups) if len(size_groups) >= 2 else None

    contingency = pd.crosstab(df["location_side"], df["win"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\nStatistical tests:")
    print(f"Welch t-test rel_group_size by win: t={t_size.statistic:.3f}, p={fmt_p(t_size.pvalue)}")
    print(f"Welch t-test location_advantage by win: t={t_loc.statistic:.3f}, p={fmt_p(t_loc.pvalue)}")
    print(f"Pearson(win, rel_group_size): r={r_size:.3f}, p={fmt_p(p_size_corr)}")
    print(f"Pearson(win, location_advantage): r={r_loc:.3f}, p={fmt_p(p_loc_corr)}")
    if anova_size is not None:
        print(f"ANOVA win ~ size_category: F={anova_size.statistic:.3f}, p={fmt_p(anova_size.pvalue)}")
    print(f"Chi-square win vs location_side: chi2={chi2_stat:.3f}, p={fmt_p(chi2_p)}")

    # Regression with inferential statistics.
    y = df["win"].astype(float)
    X_infer = sm.add_constant(df[["rel_group_size", "location_advantage_100m"]])

    logit_result = None
    logit_label = "Logit"
    try:
        logit_result = sm.Logit(y, X_infer).fit(disp=False)
    except Exception:
        logit_result = sm.GLM(y, X_infer, family=sm.families.Binomial()).fit()
        logit_label = "GLM-Binomial"

    ols_result = sm.OLS(y, X_infer).fit()

    print(f"\n{logit_label} coefficients:")
    print(logit_result.params.round(4))
    print(f"{logit_label} p-values:")
    print(logit_result.pvalues.round(4))

    print("\nOLS coefficients:")
    print(ols_result.params.round(4))
    print("OLS p-values:")
    print(ols_result.pvalues.round(4))

    # Interpretable sklearn models.
    feature_cols = ["rel_group_size", "location_advantage_100m", "rel_males", "rel_females"]
    X = df[feature_cols]

    lr = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=0).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0).fit(X, y.astype(int))

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "linear_coef": lr.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
            "tree_importance": tree.feature_importances_,
        }
    ).sort_values("tree_importance", ascending=False)

    print("\nInterpretable sklearn model effects:")
    print(coef_df.round(4).to_string(index=False))

    # Interpretable imodels models.
    rulefit = RuleFitRegressor(random_state=0, max_rules=20)
    rulefit.fit(X, y)

    figs = FIGSRegressor(random_state=0, max_rules=10)
    figs.fit(X, y)

    hstree = HSTreeRegressor(random_state=0, max_leaf_nodes=8)
    hstree.fit(X, y)

    # RuleFit human-readable rule list (supports show prevalence).
    top_rules = sorted(rulefit.rules_, key=lambda r: r.support, reverse=True)[:5]
    print("\nTop RuleFit rules by support:")
    for idx, rule in enumerate(top_rules, start=1):
        print(f"{idx}. {rule.rule} [support={rule.support:.3f}]")

    figs_importance = dict(zip(feature_cols, figs.feature_importances_))
    print("\nFIGS feature importances:")
    print({k: round(v, 4) for k, v in figs_importance.items()})

    hs_importance = None
    if hasattr(hstree, "estimator_") and hasattr(hstree.estimator_, "feature_importances_"):
        hs_importance = dict(zip(feature_cols, hstree.estimator_.feature_importances_))

    if hs_importance is None:
        pi = permutation_importance(
            hstree, X, y, n_repeats=30, random_state=0, scoring="neg_mean_squared_error"
        )
        hs_importance = dict(zip(feature_cols, pi.importances_mean))

    print("\nHSTree feature importances:")
    print({k: round(v, 4) for k, v in hs_importance.items()})

    # Evidence synthesis for a 0-100 Likert response.
    p_size_logit = safe_float(logit_result.pvalues.get("rel_group_size", np.nan))
    p_loc_logit = safe_float(logit_result.pvalues.get("location_advantage_100m", np.nan))
    beta_size = safe_float(logit_result.params.get("rel_group_size", np.nan))
    beta_loc = safe_float(logit_result.params.get("location_advantage_100m", np.nan))

    p_size_ols = safe_float(ols_result.pvalues.get("rel_group_size", np.nan))
    p_loc_ols = safe_float(ols_result.pvalues.get("location_advantage_100m", np.nan))

    score = 0
    score += 40 if p_size_logit < 0.05 else (20 if min(p_size_ols, safe_float(t_size.pvalue)) < 0.05 else 5)
    score += 40 if p_loc_logit < 0.05 else (20 if min(p_loc_ols, safe_float(t_loc.pvalue)) < 0.05 else 5)

    if p_size_logit < 0.05 and p_loc_logit < 0.05:
        score += 10
    if beta_size > 0 and beta_loc > 0:
        score += 5

    response_score = int(np.clip(round(score), 0, 100))

    direction_size = "increases" if beta_size > 0 else "decreases"
    direction_loc = "increases" if beta_loc > 0 else "decreases"

    if response_score >= 75:
        evidence_sentence = "Overall evidence is strong that relative size and contest location influence contest outcomes."
    elif response_score >= 50:
        evidence_sentence = (
            "Evidence is mixed but suggests at least one of the predictors influences contest outcomes."
        )
    else:
        evidence_sentence = (
            "Overall evidence is weak: neither predictor reaches statistical significance, so support for an influence is low."
        )

    explanation = (
        f"Both predictors were evaluated with inferential statistics and interpretable models. "
        f"Relative group size {direction_size} focal win probability "
        f"(logit beta={beta_size:.3f}, p={fmt_p(p_size_logit)}; t-test p={fmt_p(t_size.pvalue)}). "
        f"Location advantage (being closer to focal home range center) {direction_loc} win probability "
        f"(logit beta per 100m={beta_loc:.3f}, p={fmt_p(p_loc_logit)}; t-test p={fmt_p(t_loc.pvalue)}). "
        f"Tree/rule models frequently split on these variables, but p-values drive the Likert score. "
        f"{evidence_sentence}"
    )

    conclusion = {"response": response_score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print("\nWrote conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
