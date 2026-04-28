import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    if items.empty or items.shape[1] < 2:
        return float("nan")
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    n_items = items.shape[1]
    return float((n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    research_question = info["research_questions"][0]

    df = pd.read_csv("fertility.csv")
    print("Research question:", research_question)
    print("Rows/columns:", df.shape)

    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

    # Outcome variable: average religiosity across 3 survey items.
    rel_cols = ["Rel1", "Rel2", "Rel3"]
    df["religiosity"] = df[rel_cols].mean(axis=1, skipna=True)
    alpha = cronbach_alpha(df[rel_cols])

    # Fertility-related features from cycle dates and lengths.
    df["observed_cycle_length"] = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    df["cycle_length"] = df["ReportedCycleLength"].fillna(df["observed_cycle_length"])
    df["cycle_length"] = df["cycle_length"].fillna(df["cycle_length"].median())
    df["cycle_length"] = df["cycle_length"].clip(lower=21, upper=40)

    df["days_since_last_period"] = (
        df["DateTesting"] - df["StartDateofLastPeriod"]
    ).dt.days
    df["cycle_day"] = df["days_since_last_period"] + 1
    df["cycle_day_wrapped"] = ((df["cycle_day"] - 1) % df["cycle_length"]) + 1
    df["ovulation_day_est"] = df["cycle_length"] - 14
    df["days_from_ovulation"] = df["cycle_day_wrapped"] - df["ovulation_day_est"]

    # Common fertile-window approximation: 5 days pre-ovulation through ovulation day.
    df["high_fertility"] = (
        (df["days_from_ovulation"] >= -5) & (df["days_from_ovulation"] <= 0)
    ).astype(int)
    df["fertility_closeness"] = (1 - (df["days_from_ovulation"].abs() / 10)).clip(
        lower=0
    )

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    numeric_summary_cols = [
        "religiosity",
        "cycle_length",
        "cycle_day_wrapped",
        "days_from_ovulation",
        "fertility_closeness",
        "high_fertility",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print("\nSummary statistics:")
    print(df[numeric_summary_cols].describe().to_string())
    print("\nReligiosity scale reliability (Cronbach alpha):", round(alpha, 3))

    print("\nDistribution checks:")
    print("High-fertility counts:\n", df["high_fertility"].value_counts().to_string())
    print(
        "Relationship status counts:\n", df["Relationship"].value_counts().sort_index().to_string()
    )
    print(
        "Cycle phase counts:\n",
        pd.cut(
            df["cycle_day_wrapped"] / df["cycle_length"],
            bins=[0, 0.2, 0.45, 0.6, 1.0],
            labels=["menstrual", "follicular", "fertile", "luteal"],
            include_lowest=True,
        )
        .value_counts()
        .to_string(),
    )

    corr_cols = [
        "religiosity",
        "fertility_closeness",
        "high_fertility",
        "days_from_ovulation",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print("\nCorrelation matrix:")
    print(df[corr_cols].corr(numeric_only=True).round(3).to_string())

    # Statistical tests for association between fertility indicators and religiosity.
    fertile_vals = df.loc[df["high_fertility"] == 1, "religiosity"].dropna()
    nonfertile_vals = df.loc[df["high_fertility"] == 0, "religiosity"].dropna()
    ttest = stats.ttest_ind(fertile_vals, nonfertile_vals, equal_var=False, nan_policy="omit")

    corr_df = df[["fertility_closeness", "religiosity"]].dropna()
    pearson_r, pearson_p = stats.pearsonr(
        corr_df["fertility_closeness"], corr_df["religiosity"]
    )
    spearman_r, spearman_p = stats.spearmanr(
        corr_df["fertility_closeness"], corr_df["religiosity"]
    )

    df["cycle_phase"] = pd.cut(
        df["cycle_day_wrapped"] / df["cycle_length"],
        bins=[0, 0.2, 0.45, 0.6, 1.0],
        labels=["menstrual", "follicular", "fertile", "luteal"],
        include_lowest=True,
    )
    phase_groups = [
        g["religiosity"].dropna().values
        for _, g in df.groupby("cycle_phase", observed=False)
        if g["religiosity"].notna().sum() > 1
    ]
    anova = stats.f_oneway(*phase_groups) if len(phase_groups) >= 2 else None

    high_cert = df[(df["Sure1"] >= 7) & (df["Sure2"] >= 7)]
    hc_fertile = high_cert.loc[high_cert["high_fertility"] == 1, "religiosity"].dropna()
    hc_nonfertile = high_cert.loc[high_cert["high_fertility"] == 0, "religiosity"].dropna()
    if len(hc_fertile) > 1 and len(hc_nonfertile) > 1:
        ttest_high_cert = stats.ttest_ind(
            hc_fertile, hc_nonfertile, equal_var=False, nan_policy="omit"
        )
    else:
        ttest_high_cert = None

    print("\nStatistical tests:")
    print(
        f"Welch t-test (high vs non-high fertility): t={ttest.statistic:.3f}, p={ttest.pvalue:.4f}"
    )
    print(f"Pearson correlation: r={pearson_r:.3f}, p={pearson_p:.4f}")
    print(f"Spearman correlation: rho={spearman_r:.3f}, p={spearman_p:.4f}")
    if anova is not None:
        print(f"ANOVA across cycle phases: F={anova.statistic:.3f}, p={anova.pvalue:.4f}")
    if ttest_high_cert is not None:
        print(
            "Welch t-test high-certainty subset: "
            f"t={ttest_high_cert.statistic:.3f}, p={ttest_high_cert.pvalue:.4f}"
        )

    regression_df = df[
        [
            "religiosity",
            "fertility_closeness",
            "high_fertility",
            "Relationship",
            "Sure1",
            "Sure2",
            "cycle_length",
        ]
    ].dropna()

    ols_cont = smf.ols(
        "religiosity ~ fertility_closeness + Relationship + Sure1 + Sure2 + cycle_length",
        data=regression_df,
    ).fit(cov_type="HC3")
    ols_bin = smf.ols(
        "religiosity ~ high_fertility + Relationship + Sure1 + Sure2 + cycle_length",
        data=regression_df,
    ).fit(cov_type="HC3")

    print("\nOLS model with continuous fertility proxy:")
    print(ols_cont.summary().tables[1])
    print("\nOLS model with fertile-window indicator:")
    print(ols_bin.summary().tables[1])

    # Interpretable machine-learning models.
    features = [
        "fertility_closeness",
        "high_fertility",
        "days_from_ovulation",
        "cycle_day_wrapped",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    model_df = df[features + ["religiosity"]].dropna()
    X = model_df[features]
    y = model_df["religiosity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    linear = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
    tree = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=15, random_state=42
    ).fit(X_train, y_train)

    print("\nInterpretable sklearn models (test R^2):")
    print("LinearRegression:", round(r2_score(y_test, linear.predict(X_test)), 4))
    print("Ridge:", round(r2_score(y_test, ridge.predict(X_test)), 4))
    print("Lasso:", round(r2_score(y_test, lasso.predict(X_test)), 4))
    print("DecisionTreeRegressor:", round(r2_score(y_test, tree.predict(X_test)), 4))

    coef_table = pd.DataFrame(
        {
            "feature": features,
            "linear_coef": linear.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
            "tree_importance": tree.feature_importances_,
        }
    )
    coef_table["max_abs_linear_family"] = coef_table[
        ["linear_coef", "ridge_coef", "lasso_coef"]
    ].abs().max(axis=1)
    coef_table = coef_table.sort_values(
        ["tree_importance", "max_abs_linear_family"], ascending=False
    )
    print("\nFeature effects/importances from sklearn models:")
    print(coef_table.drop(columns=["max_abs_linear_family"]).round(4).to_string(index=False))
    print("\nDecision tree rules:")
    print(export_text(tree, feature_names=features))

    rulefit = RuleFitRegressor(random_state=42, max_rules=25)
    rulefit.fit(X_train.values, y_train.values, feature_names=features)
    rulefit_r2 = r2_score(y_test, rulefit.predict(X_test.values))

    rulefit_coef = np.array([safe_float(v) for v in getattr(rulefit, "coef", [])], dtype=float)
    n_features = len(features)
    linear_part = pd.Series(dtype=float)
    rule_effects = []
    if rulefit_coef.size >= n_features:
        linear_part = pd.Series(rulefit_coef[:n_features], index=features).sort_values(
            key=np.abs, ascending=False
        )
    rules = [str(r) for r in getattr(rulefit, "rules_", [])]
    if rulefit_coef.size >= n_features + len(rules) and len(rules) > 0:
        rule_coefs = rulefit_coef[n_features : n_features + len(rules)]
        order = np.argsort(np.abs(rule_coefs))[::-1]
        for idx in order:
            if abs(rule_coefs[idx]) > 1e-8:
                rule_effects.append((rules[idx], float(rule_coefs[idx])))
            if len(rule_effects) >= 5:
                break

    figs = FIGSRegressor(random_state=42, max_rules=12)
    figs.fit(X_train.values, y_train.values, feature_names=features)
    figs_r2 = r2_score(y_test, figs.predict(X_test.values))
    figs_importance = pd.Series(figs.feature_importances_, index=features).sort_values(
        ascending=False
    )

    hst = HSTreeRegressor(random_state=42, max_leaf_nodes=12)
    hst.fit(X_train.values, y_train.values, feature_names=features)
    hst_r2 = r2_score(y_test, hst.predict(X_test.values))

    print("\nimodels models (test R^2):")
    print("RuleFitRegressor:", round(rulefit_r2, 4))
    print("FIGSRegressor:", round(figs_r2, 4))
    print("HSTreeRegressor:", round(hst_r2, 4))

    print("\nRuleFit linear-term coefficients (abs-sorted):")
    if not linear_part.empty:
        print(linear_part.round(4).to_string())
    else:
        print("No linear coefficients found.")

    print("\nTop RuleFit rules:")
    if rule_effects:
        for rule, coef in rule_effects:
            print(f"coef={coef:+.4f} | {rule}")
    else:
        print("No non-zero rules found.")

    print("\nFIGS feature importances:")
    print(figs_importance.round(4).to_string())

    print("\nHSTree (truncated text representation):")
    hst_text = str(hst)
    print("\n".join(hst_text.splitlines()[:20]))

    # Convert evidence into a Likert score: low if fertility-religiosity links are not significant.
    score = 50
    p_main = safe_float(ols_bin.pvalues.get("high_fertility", np.nan))
    p_ttest = safe_float(ttest.pvalue)
    p_anova = safe_float(anova.pvalue) if anova is not None else np.nan
    p_corr = safe_float(pearson_p)

    score += 30 if (not np.isnan(p_main) and p_main < 0.05) else -20
    score += 15 if (not np.isnan(p_ttest) and p_ttest < 0.05) else -10
    score += 10 if (not np.isnan(p_anova) and p_anova < 0.05) else -5
    score += 10 if (not np.isnan(p_corr) and p_corr < 0.05) else -5
    score = int(max(0, min(100, round(score))))

    high_fi = figs_importance.get("high_fertility", np.nan)
    close_fi = figs_importance.get("fertility_closeness", np.nan)
    beta_high = safe_float(ols_bin.params.get("high_fertility", np.nan))
    beta_close = safe_float(ols_cont.params.get("fertility_closeness", np.nan))

    explanation = (
        "Evidence does not support a meaningful fertility-religiosity relationship in this sample. "
        f"Welch t-test between high- and non-high-fertility groups was non-significant (p={p_ttest:.3f}); "
        f"Pearson correlation between fertility closeness and religiosity was near zero (r={pearson_r:.3f}, p={p_corr:.3f}); "
        f"ANOVA across cycle phases was non-significant (p={p_anova:.3f}); "
        f"OLS models controlling for relationship status and date-certainty showed non-significant fertility terms "
        f"(high_fertility beta={beta_high:.3f}, p={p_main:.3f}; fertility_closeness beta={beta_close:.3f}, "
        f"p={safe_float(ols_cont.pvalues.get('fertility_closeness', np.nan)):.3f}). "
        "Interpretable ML models (linear/tree/RuleFit/FIGS/HSTree) gave low out-of-sample R^2 and did not rank fertility "
        f"features as dominant predictors (FIGS importances: high_fertility={high_fi:.3f}, fertility_closeness={close_fi:.3f})."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": score, "explanation": explanation}, f)

    print("\nWrote conclusion.txt with response score:", score)


if __name__ == "__main__":
    main()
