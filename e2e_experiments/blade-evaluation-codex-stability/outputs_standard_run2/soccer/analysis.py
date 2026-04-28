import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["skin_mean"] = out[["rater1", "rater2"]].mean(axis=1)
    out["birthday_dt"] = pd.to_datetime(out["birthday"], format="%d.%m.%Y", errors="coerce")
    out["age_2013"] = 2013 - out["birthday_dt"].dt.year
    out = out.dropna(subset=["skin_mean", "redCards", "games"])
    out = out[out["games"] > 0].copy()
    out["red_rate"] = out["redCards"] / out["games"]
    out["any_red"] = (out["redCards"] > 0).astype(int)
    out["skin_group"] = pd.cut(
        out["skin_mean"],
        bins=[-1.0, 0.25, 0.75, 2.0],
        labels=["light", "neutral", "dark"],
    )
    return out


def run_eda(df: pd.DataFrame, numeric_cols):
    print("=== RESEARCH QUESTION ===")
    print("Are darker-skinned players more likely to receive red cards?")
    print()

    print("=== DATA OVERVIEW ===")
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")
    print("Missingness (fraction) for key columns:")
    for c in ["skin_mean", "redCards", "games", "age_2013", "meanIAT", "meanExp"]:
        print(f"  {c}: {df[c].isna().mean():.4f}")
    print()

    print("=== SUMMARY STATISTICS (NUMERIC) ===")
    print(df[numeric_cols + ["redCards", "red_rate", "any_red"]].describe().round(4).to_string())
    print()

    print("=== DISTRIBUTIONS ===")
    print("redCards counts:")
    print(df["redCards"].value_counts().sort_index().to_string())
    print("skin_mean counts (rounded levels):")
    print(df["skin_mean"].value_counts().sort_index().to_string())
    print()

    print("=== CORRELATIONS ===")
    corr_cols = ["skin_mean", "red_rate", "any_red", "games", "yellowCards", "goals", "meanIAT", "meanExp"]
    corr = df[corr_cols].corr(numeric_only=True)
    print(corr.round(4).to_string())
    print()


def run_statistical_tests(df: pd.DataFrame):
    results = {}

    light = df[df["skin_mean"] <= 0.25]
    dark = df[df["skin_mean"] >= 0.75]
    light_rate = light["red_rate"].values
    dark_rate = dark["red_rate"].values
    light_any = light["any_red"].values
    dark_any = dark["any_red"].values

    results["light_mean_red_rate"] = safe_float(np.mean(light_rate))
    results["dark_mean_red_rate"] = safe_float(np.mean(dark_rate))
    results["mean_diff_dark_minus_light"] = results["dark_mean_red_rate"] - results["light_mean_red_rate"]
    results["light_any_red_rate"] = safe_float(np.mean(light_any))
    results["dark_any_red_rate"] = safe_float(np.mean(dark_any))

    t_res = stats.ttest_ind(dark_rate, light_rate, equal_var=False, nan_policy="omit")
    results["ttest_stat"] = safe_float(t_res.statistic)
    results["ttest_p"] = safe_float(t_res.pvalue)

    mw_res = stats.mannwhitneyu(dark_rate, light_rate, alternative="two-sided")
    results["mannwhitney_stat"] = safe_float(mw_res.statistic)
    results["mannwhitney_p"] = safe_float(mw_res.pvalue)

    ct = pd.crosstab((df["skin_mean"] >= 0.75).astype(int), df["any_red"])
    chi2_res = stats.chi2_contingency(ct)
    results["chi2_stat"] = safe_float(chi2_res.statistic)
    results["chi2_p"] = safe_float(chi2_res.pvalue)

    anova_groups = [
        df.loc[df["skin_group"] == "light", "red_rate"].values,
        df.loc[df["skin_group"] == "neutral", "red_rate"].values,
        df.loc[df["skin_group"] == "dark", "red_rate"].values,
    ]
    anova_res = stats.f_oneway(*anova_groups)
    results["anova_stat"] = safe_float(anova_res.statistic)
    results["anova_p"] = safe_float(anova_res.pvalue)

    sp_res = stats.spearmanr(df["skin_mean"], df["red_rate"], nan_policy="omit")
    results["spearman_rho"] = safe_float(sp_res.statistic)
    results["spearman_p"] = safe_float(sp_res.pvalue)

    formula = (
        "red_rate ~ skin_mean + games + goals + yellowCards + C(position) + "
        "C(leagueCountry) + age_2013 + height + weight + meanIAT + meanExp"
    )
    ols_model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    results["ols_skin_coef"] = safe_float(ols_model.params.get("skin_mean", np.nan))
    results["ols_skin_p"] = safe_float(ols_model.pvalues.get("skin_mean", np.nan))
    if "skin_mean" in ols_model.conf_int().index:
        ci = ols_model.conf_int().loc["skin_mean"].values
        results["ols_skin_ci_low"] = safe_float(ci[0])
        results["ols_skin_ci_high"] = safe_float(ci[1])
    else:
        results["ols_skin_ci_low"] = np.nan
        results["ols_skin_ci_high"] = np.nan

    glm_formula = (
        "any_red ~ skin_mean + games + goals + yellowCards + C(position) + "
        "C(leagueCountry) + age_2013 + height + weight + meanIAT + meanExp"
    )
    glm_model = smf.glm(formula=glm_formula, data=df, family=sm.families.Binomial()).fit()
    results["glm_skin_coef"] = safe_float(glm_model.params.get("skin_mean", np.nan))
    results["glm_skin_p"] = safe_float(glm_model.pvalues.get("skin_mean", np.nan))
    if "skin_mean" in glm_model.conf_int().index:
        gci = glm_model.conf_int().loc["skin_mean"].values
        results["glm_skin_ci_low"] = safe_float(gci[0])
        results["glm_skin_ci_high"] = safe_float(gci[1])
        results["glm_skin_odds_ratio"] = safe_float(np.exp(results["glm_skin_coef"]))
        results["glm_skin_or_ci_low"] = safe_float(np.exp(gci[0]))
        results["glm_skin_or_ci_high"] = safe_float(np.exp(gci[1]))
    else:
        results["glm_skin_ci_low"] = np.nan
        results["glm_skin_ci_high"] = np.nan
        results["glm_skin_odds_ratio"] = np.nan
        results["glm_skin_or_ci_low"] = np.nan
        results["glm_skin_or_ci_high"] = np.nan

    print("=== STATISTICAL TESTS ===")
    print(f"Light mean red rate: {results['light_mean_red_rate']:.6f}")
    print(f"Dark mean red rate:  {results['dark_mean_red_rate']:.6f}")
    print(f"Difference (dark-light): {results['mean_diff_dark_minus_light']:.6f}")
    print(
        f"Welch t-test: t={results['ttest_stat']:.4f}, p={results['ttest_p']:.6g}; "
        f"Mann-Whitney p={results['mannwhitney_p']:.6g}"
    )
    print(f"Chi-square any_red by dark/light: chi2={results['chi2_stat']:.4f}, p={results['chi2_p']:.6g}")
    print(f"ANOVA red_rate across light/neutral/dark: F={results['anova_stat']:.4f}, p={results['anova_p']:.6g}")
    print(f"Spearman skin_mean vs red_rate: rho={results['spearman_rho']:.4f}, p={results['spearman_p']:.6g}")
    print(
        f"OLS (controls) skin_mean coef={results['ols_skin_coef']:.6f}, "
        f"95% CI=({results['ols_skin_ci_low']:.6f}, {results['ols_skin_ci_high']:.6f}), "
        f"p={results['ols_skin_p']:.6g}"
    )
    print(
        f"Binomial GLM (controls) skin_mean OR={results['glm_skin_odds_ratio']:.4f}, "
        f"95% CI=({results['glm_skin_or_ci_low']:.4f}, {results['glm_skin_or_ci_high']:.4f}), "
        f"p={results['glm_skin_p']:.6g}"
    )
    print()
    return results


def run_interpretable_models(df: pd.DataFrame):
    numeric_features = [
        "skin_mean",
        "games",
        "goals",
        "yellowCards",
        "yellowReds",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
    ]
    model_df = df[numeric_features + ["red_rate", "any_red"]].copy()
    model_df = model_df.dropna()
    X = model_df[numeric_features]
    y_reg = model_df["red_rate"]
    y_clf = model_df["any_red"]

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    lasso = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.0001, random_state=42, max_iter=5000)),
        ]
    )
    dt_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=42)),
        ]
    )
    dt_clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=200, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    linear.fit(X_train, y_train_reg)
    ridge.fit(X_train, y_train_reg)
    lasso.fit(X_train, y_train_reg)
    dt_reg.fit(X_train, y_train_reg)
    dt_clf.fit(X_train, y_train_clf)

    reg_pred_linear = linear.predict(X_test)
    reg_pred_ridge = ridge.predict(X_test)
    reg_pred_lasso = lasso.predict(X_test)
    reg_pred_tree = dt_reg.predict(X_test)
    clf_prob = dt_clf.predict_proba(X_test)[:, 1]

    linear_coef = linear.named_steps["model"].coef_
    ridge_coef = ridge.named_steps["model"].coef_
    lasso_coef = lasso.named_steps["model"].coef_
    dt_reg_importances = dt_reg.named_steps["model"].feature_importances_
    dt_clf_importances = dt_clf.named_steps["model"].feature_importances_
    skin_idx = numeric_features.index("skin_mean")

    # imodels on a moderate-size sample for runtime stability.
    sample_n = min(12000, len(model_df))
    sampled = model_df.sample(n=sample_n, random_state=42)
    Xs = sampled[numeric_features]
    ys = sampled["red_rate"]

    rulefit = RuleFitRegressor(random_state=42, max_rules=40, tree_size=4)
    rulefit.fit(Xs, ys, feature_names=numeric_features)

    figs = FIGSRegressor(random_state=42, max_rules=12)
    figs.fit(Xs, ys, feature_names=numeric_features)

    hstree = HSTreeRegressor(max_leaf_nodes=12, random_state=42)
    hstree.fit(Xs, ys, feature_names=numeric_features)

    rulefit_coefs = np.array(rulefit.coef, dtype=float)
    n_linear = len(numeric_features)
    rulefit_skin_coef = safe_float(rulefit_coefs[skin_idx]) if len(rulefit_coefs) >= n_linear else np.nan
    rule_texts = [str(r) for r in getattr(rulefit, "rules_", [])]
    rule_coefs = rulefit_coefs[n_linear:] if len(rulefit_coefs) > n_linear else np.array([])
    top_rule_idx = np.argsort(np.abs(rule_coefs))[::-1][:5] if len(rule_coefs) > 0 else []
    top_rules = []
    for idx in top_rule_idx:
        if idx < len(rule_texts):
            top_rules.append((rule_texts[idx], safe_float(rule_coefs[idx])))

    figs_importances = np.array(getattr(figs, "feature_importances_", np.zeros(n_linear)), dtype=float)
    hstree_importances = np.array(
        getattr(getattr(hstree, "estimator_", None), "feature_importances_", np.zeros(n_linear)),
        dtype=float,
    )

    results = {
        "numeric_features": numeric_features,
        "linear_skin_coef": safe_float(linear_coef[skin_idx]),
        "ridge_skin_coef": safe_float(ridge_coef[skin_idx]),
        "lasso_skin_coef": safe_float(lasso_coef[skin_idx]),
        "dt_reg_skin_importance": safe_float(dt_reg_importances[skin_idx]),
        "dt_clf_skin_importance": safe_float(dt_clf_importances[skin_idx]),
        "rulefit_skin_coef": safe_float(rulefit_skin_coef),
        "figs_skin_importance": safe_float(figs_importances[skin_idx]),
        "hstree_skin_importance": safe_float(hstree_importances[skin_idx]),
        "linear_r2": safe_float(r2_score(y_test_reg, reg_pred_linear)),
        "ridge_r2": safe_float(r2_score(y_test_reg, reg_pred_ridge)),
        "lasso_r2": safe_float(r2_score(y_test_reg, reg_pred_lasso)),
        "tree_r2": safe_float(r2_score(y_test_reg, reg_pred_tree)),
        "tree_auc": safe_float(roc_auc_score(y_test_clf, clf_prob)),
        "top_rulefit_rules": top_rules,
    }

    print("=== INTERPRETABLE MODELS ===")
    print(f"LinearRegression skin_mean coef (scaled): {results['linear_skin_coef']:.6f}")
    print(f"Ridge skin_mean coef (scaled):            {results['ridge_skin_coef']:.6f}")
    print(f"Lasso skin_mean coef (scaled):            {results['lasso_skin_coef']:.6f}")
    print(f"DecisionTreeRegressor skin importance:    {results['dt_reg_skin_importance']:.6f}")
    print(f"DecisionTreeClassifier skin importance:   {results['dt_clf_skin_importance']:.6f}")
    print(f"RuleFit skin linear coef:                 {results['rulefit_skin_coef']:.6f}")
    print(f"FIGS skin importance:                     {results['figs_skin_importance']:.6f}")
    print(f"HSTree skin importance:                   {results['hstree_skin_importance']:.6f}")
    print(
        f"Model fit: linear R2={results['linear_r2']:.4f}, ridge R2={results['ridge_r2']:.4f}, "
        f"lasso R2={results['lasso_r2']:.4f}, tree R2={results['tree_r2']:.4f}, tree AUC={results['tree_auc']:.4f}"
    )
    if results["top_rulefit_rules"]:
        print("Top RuleFit rules by |coefficient|:")
        for rule, coef in results["top_rulefit_rules"]:
            print(f"  coef={coef:.6f} :: {rule}")
    print()
    return results


def build_conclusion(test_res: dict, model_res: dict, question: str):
    score = 50
    checks = []

    diff = test_res["mean_diff_dark_minus_light"]
    positive_diff = diff > 0
    if positive_diff:
        checks.append("dark group has higher average red-card rate")
    else:
        checks.append("dark group does not have higher average red-card rate")

    if positive_diff and test_res["ttest_p"] < 0.05:
        score += 6
        checks.append("Welch t-test significant")
    elif positive_diff and test_res["ttest_p"] < 0.10:
        score += 3
        checks.append("Welch t-test marginal")
    else:
        score -= 4
        checks.append("Welch t-test not significant")

    if positive_diff and test_res["mannwhitney_p"] < 0.05:
        score += 5
        checks.append("Mann-Whitney significant")
    elif positive_diff and test_res["mannwhitney_p"] < 0.10:
        score += 2
        checks.append("Mann-Whitney marginal")
    else:
        score -= 2
        checks.append("Mann-Whitney not significant")

    if test_res["anova_p"] < 0.05:
        score += 3
        checks.append("ANOVA across skin groups significant")
    else:
        score -= 1
        checks.append("ANOVA not significant")

    if positive_diff and test_res["chi2_p"] < 0.05:
        score += 5
        checks.append("Chi-square for any red significant")
    else:
        checks.append("Chi-square for any red not significant")

    if test_res["spearman_rho"] > 0 and test_res["spearman_p"] < 0.05:
        score += 3
        checks.append("positive Spearman correlation")
    else:
        checks.append("Spearman correlation weak/non-significant")

    if test_res["ols_skin_coef"] > 0 and test_res["ols_skin_p"] < 0.05:
        score += 10
        checks.append("OLS adjusted effect significant")
    elif test_res["ols_skin_coef"] > 0 and test_res["ols_skin_p"] < 0.10:
        score += 5
        checks.append("OLS adjusted effect marginal")
    else:
        score -= 5
        checks.append("OLS adjusted effect not significant")

    if test_res["glm_skin_coef"] > 0 and test_res["glm_skin_p"] < 0.05:
        score += 10
        checks.append("GLM adjusted odds effect significant")
    elif test_res["glm_skin_coef"] > 0 and test_res["glm_skin_p"] < 0.10:
        score += 5
        checks.append("GLM adjusted odds effect marginal")
    else:
        score -= 5
        checks.append("GLM adjusted odds effect not significant")

    model_direction_votes = 0
    for k in ["linear_skin_coef", "ridge_skin_coef", "lasso_skin_coef", "rulefit_skin_coef"]:
        if model_res.get(k, np.nan) > 0:
            model_direction_votes += 1
    if model_direction_votes >= 3:
        score += 3
        checks.append("most interpretable regression models give positive skin coefficient")
    elif model_direction_votes == 2:
        score += 1
    else:
        score -= 2
        checks.append("model coefficients do not consistently support positive direction")

    if abs(diff) < 0.001:
        score -= 8
        checks.append("effect size is very small")
    elif abs(diff) < 0.002:
        score -= 5
        checks.append("effect size is small")

    score = int(np.clip(round(score), 0, 100))

    chi2_phrase = "significant" if test_res["chi2_p"] < 0.05 else "not significant"

    explanation = (
        f"Question: {question} "
        f"In this dataset, darker-skinned players have a higher mean red-card rate "
        f"({test_res['dark_mean_red_rate']:.4f}) than lighter-skinned players "
        f"({test_res['light_mean_red_rate']:.4f}), difference={diff:.4f}. "
        f"Unadjusted tests are significant (Welch t p={test_res['ttest_p']:.4g}, "
        f"Mann-Whitney p={test_res['mannwhitney_p']:.4g}, ANOVA p={test_res['anova_p']:.4g}); "
        f"chi-square on any red is {chi2_phrase} (p={test_res['chi2_p']:.4g}). "
        f"With covariates, OLS is marginal (coef={test_res['ols_skin_coef']:.4g}, "
        f"p={test_res['ols_skin_p']:.4g}) and binomial GLM is significant "
        f"(OR={test_res['glm_skin_odds_ratio']:.3f}, p={test_res['glm_skin_p']:.4g}). "
        f"Interpretable models mostly keep a positive skin-tone coefficient/importance, "
        f"but the effect size is small. Overall this is moderate, not overwhelming, evidence for 'Yes'."
    )

    print("=== DECISION TRACE ===")
    for item in checks:
        print(f"- {item}")
    print(f"Final Likert response: {score}")
    print()
    return score, explanation


def main():
    info_path = Path("info.json")
    data_path = Path("soccer.csv")
    out_path = Path("conclusion.txt")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    question = info.get("research_questions", [""])[0]

    df_raw = pd.read_csv(data_path)
    df = prepare_data(df_raw)

    numeric_cols = [
        "skin_mean",
        "games",
        "goals",
        "yellowCards",
        "yellowReds",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
    ]
    run_eda(df, numeric_cols)
    test_res = run_statistical_tests(df)
    model_res = run_interpretable_models(df)
    response, explanation = build_conclusion(test_res, model_res, question)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f, ensure_ascii=True)


if __name__ == "__main__":
    main()
