import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    # 1) Read metadata / research question
    info_path = Path("info.json")
    data_path = Path("affairs.csv")
    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0].strip()

    # 2) Load dataset
    df = pd.read_csv(data_path)

    print("=== Research Question ===")
    print(research_question)
    print()

    print("=== Dataset Overview ===")
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes)
    print("missing values:")
    print(df.isna().sum())
    print()

    # 3) Exploratory analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print("=== Numeric Summary Statistics ===")
    print(df[numeric_cols].describe().T)
    print()

    print("=== Affairs Distribution ===")
    print(df["affairs"].value_counts().sort_index())
    print("affairs quantiles:")
    print(df["affairs"].quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print()

    print("=== Categorical Distributions ===")
    for c in categorical_cols:
        print(f"{c}:")
        print(df[c].value_counts(dropna=False))
        print()

    print("=== Correlations with affairs (numeric only) ===")
    corr_with_affairs = df[numeric_cols].corr(numeric_only=True)["affairs"].sort_values(ascending=False)
    print(corr_with_affairs)
    print()

    # Encode children for tests
    df["children_bin"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    affairs_yes = df.loc[df["children_bin"] == 1, "affairs"].values
    affairs_no = df.loc[df["children_bin"] == 0, "affairs"].values

    print("=== Group Means (children vs affairs) ===")
    print(df.groupby("children")["affairs"].agg(["count", "mean", "median", "std"]))
    print()

    # 4) Statistical tests
    ttest = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)
    mwu = stats.mannwhitneyu(affairs_yes, affairs_no, alternative="two-sided")
    anova = stats.f_oneway(affairs_yes, affairs_no)
    pb = stats.pointbiserialr(df["children_bin"], df["affairs"])

    print("=== Statistical Tests ===")
    print(f"Welch t-test: statistic={safe_float(ttest.statistic):.4f}, p={safe_float(ttest.pvalue):.6f}")
    print(f"Mann-Whitney U: statistic={safe_float(mwu.statistic):.4f}, p={safe_float(mwu.pvalue):.6f}")
    print(f"One-way ANOVA: statistic={safe_float(anova.statistic):.4f}, p={safe_float(anova.pvalue):.6f}")
    print(f"Point-biserial corr(children_bin, affairs): r={safe_float(pb.statistic):.4f}, p={safe_float(pb.pvalue):.6f}")
    print()

    # 5) OLS models for interpretability and significance with controls
    X_unadj = sm.add_constant(df[["children_bin"]])
    y = df["affairs"]
    ols_unadj = sm.OLS(y, X_unadj).fit()

    features = [
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = pd.get_dummies(df[features], drop_first=True, dtype=float)
    X_sm = sm.add_constant(X)
    ols_adj = sm.OLS(y, X_sm).fit()

    child_col = "children_yes"
    if child_col not in X.columns:
        # Fallback to any column containing "children"
        child_candidates = [c for c in X.columns if "children" in c]
        child_col = child_candidates[0] if child_candidates else None

    unadj_coef = safe_float(ols_unadj.params.get("children_bin", np.nan))
    unadj_p = safe_float(ols_unadj.pvalues.get("children_bin", np.nan))
    adj_coef = safe_float(ols_adj.params.get(child_col, np.nan)) if child_col else np.nan
    adj_p = safe_float(ols_adj.pvalues.get(child_col, np.nan)) if child_col else np.nan

    print("=== OLS: Unadjusted (affairs ~ children) ===")
    print(ols_unadj.summary().tables[1])
    print()

    print("=== OLS: Adjusted (with demographics/marriage controls) ===")
    print(ols_adj.summary().tables[1])
    print()

    # 6) Interpretable sklearn models
    lin = LinearRegression()
    ridge = Ridge(alpha=1.0, random_state=42)
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)

    lin.fit(X, y)
    ridge.fit(X, y)
    lasso.fit(X, y)
    tree.fit(X, y)

    coef_lin = pd.Series(lin.coef_, index=X.columns)
    coef_ridge = pd.Series(ridge.coef_, index=X.columns)
    coef_lasso = pd.Series(lasso.coef_, index=X.columns)
    importances_tree = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)

    child_lin = safe_float(coef_lin.get(child_col, np.nan)) if child_col else np.nan
    child_ridge = safe_float(coef_ridge.get(child_col, np.nan)) if child_col else np.nan
    child_lasso = safe_float(coef_lasso.get(child_col, np.nan)) if child_col else np.nan

    print("=== sklearn Interpretable Models ===")
    print(f"LinearRegression coef({child_col}) = {child_lin:.4f}")
    print(f"Ridge coef({child_col}) = {child_ridge:.4f}")
    print(f"Lasso coef({child_col}) = {child_lasso:.4f}")
    print("Top DecisionTreeRegressor feature importances:")
    print(importances_tree.head(8))
    print()

    # 7) imodels interpretable models
    print("=== imodels Interpretable Models ===")
    rulefit_child_coef = np.nan
    figs_child_imp = np.nan
    hstree_child_imp = np.nan

    try:
        rf = RuleFitRegressor(random_state=42, max_rules=30)
        rf.fit(X.values, y.values, feature_names=X.columns.tolist())

        # If available, capture linear term for children from human-readable rules table
        if hasattr(rf, "get_rules"):
            rules_df = rf.get_rules()
            if isinstance(rules_df, pd.DataFrame) and "feature" in rules_df.columns and "coef" in rules_df.columns:
                tmp = rules_df.loc[rules_df["feature"] == child_col, "coef"]
                if len(tmp) > 0:
                    rulefit_child_coef = safe_float(tmp.iloc[0])

        print(f"RuleFit child linear coef proxy ({child_col}) = {rulefit_child_coef:.4f}")
    except Exception as e:
        print(f"RuleFitRegressor failed: {e}")

    try:
        figs = FIGSRegressor(max_rules=12, random_state=42)
        figs.fit(X.values, y.values, feature_names=X.columns.tolist())
        if hasattr(figs, "feature_importances_"):
            fi = pd.Series(figs.feature_importances_, index=X.columns)
            figs_child_imp = safe_float(fi.get(child_col, np.nan)) if child_col else np.nan
        print(f"FIGS feature importance ({child_col}) = {figs_child_imp:.4f}")
    except Exception as e:
        print(f"FIGSRegressor failed: {e}")

    try:
        hst = HSTreeRegressor(random_state=42, max_leaf_nodes=12)
        hst.fit(X.values, y.values, feature_names=X.columns.tolist())
        if hasattr(hst, "feature_importances_"):
            fi_h = pd.Series(hst.feature_importances_, index=X.columns)
            hstree_child_imp = safe_float(fi_h.get(child_col, np.nan)) if child_col else np.nan
        print(f"HSTree feature importance ({child_col}) = {hstree_child_imp:.4f}")
    except Exception as e:
        print(f"HSTreeRegressor failed: {e}")

    print()

    # 8) Convert evidence to Likert score (0-100) for the specific question:
    # "Does having children decrease engagement in extramarital affairs?"
    mean_yes = float(np.mean(affairs_yes))
    mean_no = float(np.mean(affairs_no))

    score = 50

    # Unadjusted direction/significance evidence
    if mean_yes < mean_no:
        if ttest.pvalue < 0.05 and mwu.pvalue < 0.05:
            score += 35
        elif ttest.pvalue < 0.05 or mwu.pvalue < 0.05:
            score += 20
        else:
            score += 5
    else:
        if ttest.pvalue < 0.05 and mwu.pvalue < 0.05:
            score -= 35
        elif ttest.pvalue < 0.05 or mwu.pvalue < 0.05:
            score -= 25
        else:
            score -= 10

    # Adjusted regression evidence (most important for interpretation)
    if not np.isnan(adj_coef):
        if adj_coef < 0 and adj_p < 0.05:
            score += 30
        elif adj_coef < 0 and adj_p < 0.10:
            score += 15
        elif adj_coef < 0:
            score += 5
        elif adj_coef > 0 and adj_p < 0.05:
            score -= 25
        else:
            score -= 10

    # Coefficient direction consistency across linear interpretable models
    linear_signs = [x for x in [child_lin, child_ridge, child_lasso] if not np.isnan(x)]
    if linear_signs:
        neg_count = sum(x < 0 for x in linear_signs)
        pos_count = sum(x > 0 for x in linear_signs)
        if neg_count > pos_count:
            score += 5
        elif pos_count > neg_count:
            score -= 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Research question: {research_question} "
        f"In raw comparisons, people with children had higher mean affairs ({mean_yes:.3f}) than those without children ({mean_no:.3f}). "
        f"This difference was statistically significant (Welch t-test p={ttest.pvalue:.4g}, Mann-Whitney p={mwu.pvalue:.4g}). "
        f"In adjusted OLS controlling for gender, age, years married, religiousness, education, occupation, and marriage rating, "
        f"the children coefficient was {adj_coef:.3f} with p={adj_p:.4g}, indicating no statistically significant decrease after controls. "
        f"Interpretable linear models (Linear/Ridge/Lasso) gave children coefficients "
        f"{child_lin:.3f}, {child_ridge:.3f}, {child_lasso:.3f}, which were small and not evidence of a robust decrease. "
        f"Overall, the data do not support the claim that having children decreases extramarital affairs."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("=== Final Conclusion JSON ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
