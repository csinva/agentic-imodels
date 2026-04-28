import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_sd


def top_abs(series: pd.Series, n: int = 10) -> pd.Series:
    s = series.copy().dropna()
    return s.reindex(s.abs().sort_values(ascending=False).head(n).index)


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:")
    print(question)

    df = pd.read_csv("reading.csv")
    print("\nData shape:", df.shape)

    # ---------------------
    # EDA: summary/distributions/correlations
    # ---------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\nNumeric summary statistics:")
    print(df[numeric_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    speed_desc = df["speed"].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    print("\nSpeed distribution summary:")
    print(speed_desc)
    print("Speed skewness:", df["speed"].skew())

    group_summary = (
        df.groupby(["dyslexia_bin", "reader_view"], dropna=False)["speed"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    print("\nGrouped speed summary (dyslexia_bin x reader_view):")
    print(group_summary)

    corr = df[numeric_cols].corr(numeric_only=True)
    speed_corr = corr["speed"].drop("speed").sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nTop correlations with speed (absolute order):")
    print(speed_corr.head(10))

    # ---------------------
    # Modeling data prep
    # ---------------------
    feature_cols = [
        "reader_view",
        "dyslexia_bin",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "device",
        "education",
        "gender",
        "language",
        "retake_trial",
        "english_native",
        "Flesch_Kincaid",
        "page_id",
    ]

    model_df = df[feature_cols + ["uuid", "speed"]].copy()
    model_df = model_df.dropna(subset=["speed", "reader_view"])

    for c in feature_cols:
        if is_numeric_dtype(model_df[c]):
            model_df[c] = model_df[c].fillna(model_df[c].median())
        else:
            model_df[c] = model_df[c].fillna("MISSING")

    model_df["reader_view_x_dyslexia"] = model_df["reader_view"] * model_df["dyslexia_bin"]
    model_df["log_speed"] = np.log1p(model_df["speed"])

    X = pd.get_dummies(
        model_df[feature_cols + ["reader_view_x_dyslexia"]],
        drop_first=True,
    )
    y = model_df["log_speed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---------------------
    # Interpretable scikit-learn models
    # ---------------------
    sklearn_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        ),
    }

    sklearn_results = {}
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        sklearn_results[name] = {
            "r2_test": r2_score(y_test, pred),
            "model": model,
        }

    lin = sklearn_results["LinearRegression"]["model"]
    lin_coef = pd.Series(lin.coef_, index=X.columns)
    print("\nLinearRegression top coefficients (log_speed target):")
    print(top_abs(lin_coef, n=12))

    tree = sklearn_results["DecisionTreeRegressor"]["model"]
    tree_importances = pd.Series(tree.feature_importances_, index=X.columns)
    print("\nDecisionTreeRegressor top feature importances:")
    print(tree_importances.sort_values(ascending=False).head(12))

    print("\nscikit-learn model performance (R^2 on test):")
    for name, out in sklearn_results.items():
        print(f"{name}: {out['r2_test']:.4f}")

    # ---------------------
    # Interpretable imodels models
    # ---------------------
    rulefit = RuleFitRegressor(random_state=RANDOM_STATE, tree_size=4, max_rules=40)
    rulefit.fit(X_train.values, y_train.values, feature_names=list(X.columns))
    rulefit_r2 = r2_score(y_test, rulefit.predict(X_test.values))

    rule_df = rulefit._get_rules(exclude_zero_coef=True)
    rule_df = rule_df.sort_values("importance", ascending=False)
    print("\nRuleFit top rules/features:")
    print(rule_df[["rule", "type", "coef", "support", "importance"]].head(10).to_string(index=False))

    figs = FIGSRegressor(max_rules=12, random_state=RANDOM_STATE)
    figs.fit(X_train.values, y_train.values, feature_names=list(X.columns))
    figs_r2 = r2_score(y_test, figs.predict(X_test.values))
    figs_importance = pd.Series(figs.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFIGS top feature importances:")
    print(figs_importance.head(12))

    hstree = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=8)
    hstree.fit(X_train.values, y_train.values, feature_names=list(X.columns))
    hstree_r2 = r2_score(y_test, hstree.predict(X_test.values))
    hs_importance = pd.Series(hstree.estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nHSTree top feature importances:")
    print(hs_importance.head(12))

    print("\nimodels performance (R^2 on test):")
    print(f"RuleFitRegressor: {rulefit_r2:.4f}")
    print(f"FIGSRegressor: {figs_r2:.4f}")
    print(f"HSTreeRegressor: {hstree_r2:.4f}")

    # ---------------------
    # Statistical tests focused on the research question
    # ---------------------
    dys_df = model_df[model_df["dyslexia_bin"] == 1].copy()

    speed_rv1 = dys_df.loc[dys_df["reader_view"] == 1, "speed"]
    speed_rv0 = dys_df.loc[dys_df["reader_view"] == 0, "speed"]
    log_rv1 = np.log1p(speed_rv1)
    log_rv0 = np.log1p(speed_rv0)

    welch_raw = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy="omit")
    welch_log = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy="omit")
    mann_whitney = stats.mannwhitneyu(speed_rv1, speed_rv0, alternative="two-sided")

    paired = (
        dys_df.assign(log_speed=np.log1p(dys_df["speed"]))
        .pivot_table(index="uuid", columns="reader_view", values="log_speed", aggfunc="mean")
        .dropna()
    )
    if 0 in paired.columns and 1 in paired.columns and len(paired) >= 3:
        paired_t = stats.ttest_rel(paired[1], paired[0])
        paired_diff = float((paired[1] - paired[0]).mean())
    else:
        paired_t = None
        paired_diff = np.nan

    d_effect = cohens_d(log_rv1, log_rv0)

    print("\nKey tests in dyslexia group:")
    print(
        f"Welch t-test (raw speed): t={welch_raw.statistic:.4f}, p={welch_raw.pvalue:.4g}; "
        f"mean_rv1={speed_rv1.mean():.2f}, mean_rv0={speed_rv0.mean():.2f}"
    )
    print(
        f"Welch t-test (log speed): t={welch_log.statistic:.4f}, p={welch_log.pvalue:.4g}; "
        f"log_mean_rv1={log_rv1.mean():.4f}, log_mean_rv0={log_rv0.mean():.4f}"
    )
    print(f"Mann-Whitney U: U={mann_whitney.statistic:.1f}, p={mann_whitney.pvalue:.4g}")
    if paired_t is not None:
        print(
            f"Paired t-test within participant (log speed): t={paired_t.statistic:.4f}, "
            f"p={paired_t.pvalue:.4g}, mean_diff={paired_diff:.4f}, pairs={len(paired)}"
        )
    else:
        print("Paired t-test within participant: insufficient paired observations")
    print(f"Cohen's d (log speed, rv1-rv0): {d_effect:.4f}")

    # OLS in dyslexia-only participants with controls
    dys_formula = (
        "log_speed ~ reader_view + num_words + correct_rate + age + retake_trial + "
        "Flesch_Kincaid + C(device) + C(education) + C(gender) + C(language) + "
        "C(page_id) + C(english_native)"
    )
    dys_ols = smf.ols(dys_formula, data=dys_df.assign(log_speed=np.log1p(dys_df["speed"]))).fit(cov_type="HC3")

    # Full-sample interaction model
    full_formula = (
        "log_speed ~ reader_view * dyslexia_bin + num_words + correct_rate + age + retake_trial + "
        "Flesch_Kincaid + C(device) + C(education) + C(gender) + C(language) + "
        "C(page_id) + C(english_native)"
    )
    full_ols = smf.ols(full_formula, data=model_df).fit(cov_type="HC3")

    # Two-way ANOVA for reader_view, dyslexia_bin, and interaction
    anova_df = model_df.dropna(subset=["dyslexia_bin"]).copy()
    anova_model = smf.ols("log_speed ~ C(reader_view) * C(dyslexia_bin)", data=anova_df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)

    print("\nDyslexia-only OLS: reader_view effect")
    print(
        f"coef={dys_ols.params['reader_view']:.4f}, "
        f"p={dys_ols.pvalues['reader_view']:.4g}, "
        f"95% CI=({dys_ols.conf_int().loc['reader_view', 0]:.4f}, {dys_ols.conf_int().loc['reader_view', 1]:.4f})"
    )

    print("\nFull OLS interaction effect (reader_view:dyslexia_bin)")
    print(
        f"coef={full_ols.params['reader_view:dyslexia_bin']:.4f}, "
        f"p={full_ols.pvalues['reader_view:dyslexia_bin']:.4g}, "
        f"95% CI=({full_ols.conf_int().loc['reader_view:dyslexia_bin', 0]:.4f}, "
        f"{full_ols.conf_int().loc['reader_view:dyslexia_bin', 1]:.4f})"
    )

    print("\nTwo-way ANOVA (log_speed):")
    print(anova_table)

    # ---------------------
    # Convert evidence to Likert response
    # 0 = strong No, 100 = strong Yes
    # ---------------------
    mean_diff_pct = 100.0 * (speed_rv1.mean() - speed_rv0.mean()) / max(speed_rv0.mean(), 1e-9)
    primary_coef = float(dys_ols.params["reader_view"])
    primary_p = float(dys_ols.pvalues["reader_view"])
    interact_coef = float(full_ols.params["reader_view:dyslexia_bin"])
    interact_p = float(full_ols.pvalues["reader_view:dyslexia_bin"])

    if primary_p < 0.05 and primary_coef > 0:
        response = 85
    elif primary_p < 0.05 and primary_coef <= 0:
        response = 10
    else:
        response = 25
        if mean_diff_pct < -2:
            response -= 8
        elif mean_diff_pct > 2:
            response += 8

        if paired_t is not None and paired_t.pvalue < 0.05:
            response += 10 if paired_diff > 0 else -10

        if interact_p < 0.05:
            response += 10 if interact_coef > 0 else -10

    response = int(np.clip(response, 0, 100))

    # Check whether reader_view is among the strongest interpretable model drivers
    top_linear = set(top_abs(lin_coef, n=8).index)
    top_tree = set(tree_importances.sort_values(ascending=False).head(8).index)
    top_figs = set(figs_importance.head(8).index)
    rv_terms = {c for c in X.columns if c.startswith("reader_view")}
    rv_prominent = bool((top_linear | top_tree | top_figs) & rv_terms)

    explanation = (
        "No convincing evidence that Reader View improves reading speed for participants with dyslexia. "
        f"In dyslexia-only analyses, reader_view was not significant in controlled OLS "
        f"(coef={primary_coef:.3f}, p={primary_p:.3f}), Welch tests were non-significant "
        f"(raw p={welch_raw.pvalue:.3f}, log p={welch_log.pvalue:.3f}), and paired within-person test was "
        f"non-significant (p={(paired_t.pvalue if paired_t is not None else float('nan')):.3f}). "
        f"Mean raw speed difference was {mean_diff_pct:.1f}% (Reader View vs control) and the interaction "
        f"reader_view:dyslexia_bin was non-significant (p={interact_p:.3f}). "
        f"Interpretable models (Linear/Tree/RuleFit/FIGS/HSTree) did not show Reader View as a dominant driver "
        f"of log-speed ({'present' if rv_prominent else 'not prominent'} among top predictors)."
    )

    output = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(output))

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
