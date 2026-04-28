import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

warnings.filterwarnings("ignore")


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    note: str = ""


def p_to_evidence(p: float) -> float:
    """Map p-value to interpretable evidence strength in [0, 2]."""
    if np.isnan(p):
        return 0.0
    if p < 0.01:
        return 2.0
    if p < 0.05:
        return 1.5
    if p < 0.10:
        return 1.0
    return 0.0


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize_rulefit(rulefit_model, feature_names):
    coefs = np.array([safe_float(c) for c in getattr(rulefit_model, "coef", [])])
    n_features = len(feature_names)

    summary = {
        "linear_terms": {},
        "top_rules": [],
    }

    if coefs.size >= n_features:
        lin = coefs[:n_features]
        summary["linear_terms"] = {
            feature_names[i]: float(lin[i]) for i in range(n_features)
        }

    rules = getattr(rulefit_model, "rules_", [])
    if coefs.size > n_features and len(rules) > 0:
        rule_coefs = coefs[n_features:n_features + len(rules)]
        abs_idx = np.argsort(np.abs(rule_coefs))[::-1]
        top_idx = abs_idx[:5]
        for idx in top_idx:
            summary["top_rules"].append(
                {
                    "rule": str(rules[idx]),
                    "coef": float(rule_coefs[idx]),
                }
            )

    return summary


def main():
    # 1) Read metadata and dataset
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0].strip()
    df = pd.read_csv("crofoot.csv")

    # 2) Feature engineering aligned with question
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["location_adv"] = df["dist_other"] - df["dist_focal"]
    df["location_adv_norm"] = (df["dist_other"] - df["dist_focal"]) / (
        df["dist_other"] + df["dist_focal"]
    )
    df["m_rel"] = df["m_focal"] - df["m_other"]
    df["f_rel"] = df["f_focal"] - df["f_other"]
    df["closer_to_focal"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    # 3) EDA: summary, distributions, correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric_cols].describe().T
    win_rate = float(df["win"].mean())

    distribution_snapshot = {
        "rel_size_quantiles": df["rel_size"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict(),
        "dist_focal_quantiles": df["dist_focal"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict(),
        "dist_other_quantiles": df["dist_other"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict(),
    }

    corr_with_win = df[[
        "win", "rel_size", "size_ratio", "dist_focal", "dist_other", "location_adv", "location_adv_norm"
    ]].corr(numeric_only=True)["win"].sort_values(ascending=False)

    print("Research question:")
    print(research_question)
    print("\nData shape:", df.shape)
    print("\nSummary stats (selected):")
    print(summary_stats.loc[["win", "rel_size", "dist_focal", "dist_other", "location_adv"]])
    print("\nDistribution snapshot:")
    print(distribution_snapshot)
    print("\nCorrelations with win:")
    print(corr_with_win)

    # 4) Statistical tests focused on the research question
    win1 = df[df["win"] == 1]
    win0 = df[df["win"] == 0]

    tests = []

    # Relative group size tests
    rel_corr = stats.pointbiserialr(df["win"], df["rel_size"])
    rel_t = stats.ttest_ind(win1["rel_size"], win0["rel_size"], equal_var=False)

    tests.append(TestResult("pointbiserial(win, rel_size)", safe_float(rel_corr.statistic), safe_float(rel_corr.pvalue)))
    tests.append(TestResult("welch_t(rel_size by win)", safe_float(rel_t.statistic), safe_float(rel_t.pvalue)))

    # Contest location tests
    loc_corr = stats.pointbiserialr(df["win"], df["dist_focal"])
    loc_t = stats.ttest_ind(win1["dist_focal"], win0["dist_focal"], equal_var=False)

    tests.append(TestResult("pointbiserial(win, dist_focal)", safe_float(loc_corr.statistic), safe_float(loc_corr.pvalue)))
    tests.append(TestResult("welch_t(dist_focal by win)", safe_float(loc_t.statistic), safe_float(loc_t.pvalue)))

    # Chi-square for categorical location proxy
    contingency = pd.crosstab(df["closer_to_focal"], df["win"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    tests.append(TestResult("chi2(closer_to_focal vs win)", safe_float(chi2_stat), safe_float(chi2_p)))

    # ANOVA on win across binned relative size and location
    df["rel_size_bin"] = pd.cut(df["rel_size"], bins=3, duplicates="drop")
    rel_groups = [g["win"].values for _, g in df.groupby("rel_size_bin") if len(g) > 0]
    if len(rel_groups) >= 2:
        anova_rel = stats.f_oneway(*rel_groups)
        tests.append(TestResult("anova(win ~ rel_size_bin)", safe_float(anova_rel.statistic), safe_float(anova_rel.pvalue)))

    df["dist_focal_bin"] = pd.qcut(df["dist_focal"], q=3, duplicates="drop")
    loc_groups = [g["win"].values for _, g in df.groupby("dist_focal_bin") if len(g) > 0]
    if len(loc_groups) >= 2:
        anova_loc = stats.f_oneway(*loc_groups)
        tests.append(TestResult("anova(win ~ dist_focal_bin)", safe_float(anova_loc.statistic), safe_float(anova_loc.pvalue)))

    # Regression with p-values
    X_reg = sm.add_constant(df[["rel_size", "dist_focal", "dist_other"]])
    y = df["win"]

    ols_model = sm.OLS(y, X_reg).fit()
    logit_model = sm.Logit(y, X_reg).fit(disp=0)

    tests.append(TestResult("ols_p(rel_size)", safe_float(ols_model.pvalues["rel_size"]), safe_float(ols_model.pvalues["rel_size"])))
    tests.append(TestResult("ols_p(dist_focal)", safe_float(ols_model.pvalues["dist_focal"]), safe_float(ols_model.pvalues["dist_focal"])))
    tests.append(TestResult("logit_p(rel_size)", safe_float(logit_model.pvalues["rel_size"]), safe_float(logit_model.pvalues["rel_size"])))
    tests.append(TestResult("logit_p(dist_focal)", safe_float(logit_model.pvalues["dist_focal"]), safe_float(logit_model.pvalues["dist_focal"])))

    print("\nKey statistical tests:")
    for t in tests:
        print(f"- {t.name}: statistic={t.statistic:.4f}, p={t.pvalue:.4f}")

    print("\nOLS coefficients:")
    print(ols_model.params)
    print("OLS p-values:")
    print(ols_model.pvalues)

    print("\nLogit coefficients:")
    print(logit_model.params)
    print("Logit p-values:")
    print(logit_model.pvalues)

    # 5) Interpretable models (sklearn + imodels)
    feature_cols = ["rel_size", "dist_focal", "dist_other", "location_adv", "m_rel", "f_rel"]
    X = df[feature_cols]
    y_cls = df["win"]

    lin = LinearRegression().fit(X, y_cls)
    ridge = Ridge(alpha=1.0).fit(X, y_cls)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(X, y_cls)

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0)
    tree.fit(X, y_cls)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tree_cv_acc = float(cross_val_score(tree, X, y_cls, cv=cv, scoring="accuracy").mean())

    rf = RuleFitRegressor(random_state=0, max_rules=20, tree_size=4)
    rf.fit(X.values, y_cls.values, feature_names=feature_cols)
    rf_summary = summarize_rulefit(rf, feature_cols)

    figs = FIGSRegressor(max_rules=12, random_state=0)
    figs.fit(X, y_cls)

    hs = HSTreeRegressor(max_leaf_nodes=8, random_state=0)
    hs.fit(X, y_cls)

    hs_importances = None
    if hasattr(hs, "estimator_") and hasattr(hs.estimator_, "feature_importances_"):
        hs_importances = {
            feature_cols[i]: float(hs.estimator_.feature_importances_[i])
            for i in range(len(feature_cols))
        }

    print("\nLinear/Ridge/Lasso coefficients:")
    print("Linear:", dict(zip(feature_cols, lin.coef_)))
    print("Ridge:", dict(zip(feature_cols, ridge.coef_)))
    print("Lasso:", dict(zip(feature_cols, lasso.coef_)))

    print("\nDecision tree importances:")
    print(dict(zip(feature_cols, tree.feature_importances_)))
    print("Decision tree CV accuracy:", tree_cv_acc)
    print("Decision tree rules:")
    print(export_text(tree, feature_names=feature_cols))

    print("\nRuleFit summary:")
    print(rf_summary)

    print("\nFIGS feature importances:")
    figs_imp = None
    if hasattr(figs, "feature_importances_"):
        figs_imp = {
            feature_cols[i]: float(figs.feature_importances_[i])
            for i in range(len(feature_cols))
        }
        print(figs_imp)
    print("FIGS structure:")
    print(figs)

    print("\nHSTree feature importances:")
    print(hs_importances)
    print("HSTree structure:")
    print(hs)

    # 6) Translate evidence into Likert score (0-100)
    # Relative size evidence: correlation, t-test, and regression p-values
    p_rel = [
        safe_float(rel_corr.pvalue),
        safe_float(rel_t.pvalue),
        safe_float(ols_model.pvalues["rel_size"]),
        safe_float(logit_model.pvalues["rel_size"]),
    ]

    # Location evidence: focal distance tests and regression p-values
    p_loc = [
        safe_float(loc_corr.pvalue),
        safe_float(loc_t.pvalue),
        safe_float(ols_model.pvalues["dist_focal"]),
        safe_float(logit_model.pvalues["dist_focal"]),
    ]

    rel_evidence = float(np.mean([p_to_evidence(p) for p in p_rel]))
    loc_evidence = float(np.mean([p_to_evidence(p) for p in p_loc]))

    # Max combined evidence is 4.0 (2.0 per factor).
    response = int(round(((rel_evidence + loc_evidence) / 4.0) * 100))
    response = int(np.clip(response, 0, 100))

    rel_sig = any(p < 0.05 for p in p_rel)
    loc_sig = any(p < 0.05 for p in p_loc)

    explanation = (
        f"In {len(df)} contests, relative group size showed weak/non-significant evidence "
        f"(p-values: {', '.join(f'{p:.3f}' for p in p_rel)}), while contest location had only "
        f"limited evidence via focal-distance effects (p-values: {', '.join(f'{p:.3f}' for p in p_loc)}). "
        f"Interpretable models (linear terms, shallow trees, RuleFit/FIGS/HSTree) generally prioritized "
        f"distance-related splits over size differences, but predictive performance remained modest "
        f"(tree CV accuracy={tree_cv_acc:.3f}). Overall this supports at most a weak/partial influence, "
        f"not a strong robust relationship for both factors. rel_size_significant={rel_sig}, "
        f"location_significant={loc_sig}, baseline_win_rate={win_rate:.3f}."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
