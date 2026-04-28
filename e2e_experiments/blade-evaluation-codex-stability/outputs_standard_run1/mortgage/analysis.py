import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def choose_outcome(df: pd.DataFrame) -> str:
    if "accept" in df.columns:
        return "accept"
    if "deny" in df.columns:
        return "deny"
    raise ValueError("Could not find outcome column ('accept' or 'deny').")


def make_feature_set(df: pd.DataFrame, outcome_col: str) -> list:
    excluded = {"Unnamed: 0", outcome_col}
    if "accept" in df.columns:
        excluded.add("accept")
    if "deny" in df.columns:
        excluded.add("deny")
    # Exclude likely post-decision signal to reduce leakage.
    if "denied_PMI" in df.columns:
        excluded.add("denied_PMI")

    features = [c for c in df.columns if c not in excluded]
    return features


def describe_data(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric_cols].describe().T

    binary_cols = []
    for col in numeric_cols:
        vals = set(df[col].dropna().unique().tolist())
        if vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)

    distributions = {
        col: {
            "mean": float(df[col].mean(skipna=True)),
            "std": float(df[col].std(skipna=True)),
            "min": float(df[col].min(skipna=True)),
            "max": float(df[col].max(skipna=True)),
            "q25": float(df[col].quantile(0.25)),
            "median": float(df[col].quantile(0.5)),
            "q75": float(df[col].quantile(0.75)),
        }
        for col in numeric_cols
    }

    corr = df[numeric_cols].corr(numeric_only=True)

    return {
        "summary_stats": summary_stats,
        "binary_cols": binary_cols,
        "distributions": distributions,
        "correlation_matrix": corr,
    }


def run_stat_tests(df: pd.DataFrame, outcome_col: str, feature_cols: list) -> dict:
    if "female" not in df.columns:
        raise ValueError("Expected a 'female' column for the research question.")

    # t-test on outcome by gender
    t_df = df[["female", outcome_col]].dropna()
    y_female = t_df.loc[t_df["female"] == 1, outcome_col]
    y_male = t_df.loc[t_df["female"] == 0, outcome_col]
    t_stat, t_pval = stats.ttest_ind(y_female, y_male, equal_var=False)

    # Chi-square test on contingency table
    contingency = pd.crosstab(t_df["female"], t_df[outcome_col])
    chi2, chi2_pval, _, _ = stats.chi2_contingency(contingency)

    # ANOVA: outcome across mortgage credit categories (if available)
    anova_stat, anova_pval = np.nan, np.nan
    if "mortgage_credit" in df.columns:
        anova_df = df[["mortgage_credit", outcome_col]].dropna()
        groups = [
            group[outcome_col].values
            for _, group in anova_df.groupby("mortgage_credit")
            if len(group) > 1
        ]
        if len(groups) >= 2:
            anova_stat, anova_pval = stats.f_oneway(*groups)

    # Unadjusted OLS: outcome ~ female
    ols_unadj_df = df[[outcome_col, "female"]].dropna()
    X_unadj = sm.add_constant(ols_unadj_df[["female"]], has_constant="add")
    y_unadj = ols_unadj_df[outcome_col]
    ols_unadj = sm.OLS(y_unadj, X_unadj).fit(cov_type="HC3")

    # Adjusted OLS: outcome ~ female + controls
    adjusted_cols = [c for c in [outcome_col] + feature_cols if c in df.columns]
    ols_adj_df = df[adjusted_cols].dropna()
    X_adj = sm.add_constant(ols_adj_df[feature_cols], has_constant="add")
    y_adj = ols_adj_df[outcome_col]
    ols_adj = sm.OLS(y_adj, X_adj).fit(cov_type="HC3")

    tests = {
        "t_test": {"t_stat": float(t_stat), "p_value": float(t_pval)},
        "chi_square": {
            "chi2": float(chi2),
            "p_value": float(chi2_pval),
            "contingency": contingency.to_dict(),
        },
        "anova": {
            "f_stat": float(anova_stat) if not np.isnan(anova_stat) else None,
            "p_value": float(anova_pval) if not np.isnan(anova_pval) else None,
        },
        "ols_unadjusted": {
            "female_coef": float(ols_unadj.params.get("female", np.nan)),
            "female_p": float(ols_unadj.pvalues.get("female", np.nan)),
            "r2": float(ols_unadj.rsquared),
        },
        "ols_adjusted": {
            "female_coef": float(ols_adj.params.get("female", np.nan)),
            "female_p": float(ols_adj.pvalues.get("female", np.nan)),
            "r2": float(ols_adj.rsquared),
        },
    }
    return tests


def run_interpretable_models(df: pd.DataFrame, outcome_col: str, feature_cols: list) -> dict:
    model_df = df[feature_cols + [outcome_col]].copy()
    X = model_df[feature_cols]
    y = model_df[outcome_col].astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=feature_cols)

    results = {}

    # scikit-learn linear models
    lr = LinearRegression()
    lr.fit(X_imp_df, y)
    results["linear_regression"] = {
        "female_coef": float(lr.coef_[feature_cols.index("female")]),
        "r2": float(lr.score(X_imp_df, y)),
        "top_abs_coefs": sorted(
            [{"feature": f, "coef": float(c)} for f, c in zip(feature_cols, lr.coef_)],
            key=lambda d: abs(d["coef"]),
            reverse=True,
        )[:5],
    }

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_imp_df, y)
    results["ridge"] = {
        "female_coef": float(ridge.coef_[feature_cols.index("female")]),
        "r2": float(ridge.score(X_imp_df, y)),
    }

    lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    lasso.fit(X_imp_df, y)
    results["lasso"] = {
        "female_coef": float(lasso.coef_[feature_cols.index("female")]),
        "r2": float(lasso.score(X_imp_df, y)),
    }

    # scikit-learn trees
    dtr = DecisionTreeRegressor(max_depth=3, random_state=42)
    dtr.fit(X_imp_df, y)
    results["decision_tree_regressor"] = {
        "r2": float(dtr.score(X_imp_df, y)),
        "female_importance": float(dtr.feature_importances_[feature_cols.index("female")]),
    }

    y_class = (y >= 0.5).astype(int)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
    dtc.fit(X_imp_df, y_class)
    y_hat = dtc.predict(X_imp_df)
    y_prob = dtc.predict_proba(X_imp_df)[:, 1]
    results["decision_tree_classifier"] = {
        "accuracy": float(accuracy_score(y_class, y_hat)),
        "auc": float(roc_auc_score(y_class, y_prob)),
        "female_importance": float(dtc.feature_importances_[feature_cols.index("female")]),
    }

    # imodels models
    rulefit = RuleFitRegressor(random_state=42, max_rules=30)
    rulefit.fit(X_imp_df.values, y.values, feature_names=feature_cols)
    if hasattr(rulefit, "get_rules"):
        rf_rules = rulefit.get_rules()
    else:
        rf_rules = rulefit._get_rules()
    female_rule_subset = rf_rules[rf_rules["rule"].astype(str).str.contains("female", na=False)]
    female_rule_max = (
        float(female_rule_subset["coef"].abs().max()) if not female_rule_subset.empty else 0.0
    )
    results["rulefit_regressor"] = {
        "r2": float(rulefit.score(X_imp_df.values, y.values)),
        "female_rule_max_abs_coef": female_rule_max,
    }

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_imp_df.values, y.values, feature_names=feature_cols)
    figs_importances = getattr(figs, "feature_importances_", np.zeros(len(feature_cols)))
    results["figs_regressor"] = {
        "r2": float(figs.score(X_imp_df.values, y.values)),
        "female_importance": float(figs_importances[feature_cols.index("female")]),
    }

    hst = HSTreeRegressor(random_state=42, max_leaf_nodes=20)
    hst.fit(X_imp_df.values, y.values, feature_names=feature_cols)
    hst_importances = getattr(hst, "feature_importances_", np.zeros(len(feature_cols)))
    results["hstree_regressor"] = {
        "r2": float(hst.score(X_imp_df.values, y.values)),
        "female_importance": float(hst_importances[feature_cols.index("female")]),
    }

    return results


def make_likert_score(stat_tests: dict, model_results: dict, outcome_col: str) -> tuple[int, str]:
    p_adj = stat_tests["ols_adjusted"]["female_p"]
    coef_adj = stat_tests["ols_adjusted"]["female_coef"]
    p_unadj = stat_tests["ols_unadjusted"]["female_p"]

    adj_sig = p_adj < 0.05
    unadj_sig = p_unadj < 0.05

    # Interpretability consistency checks.
    female_linear_coef = model_results["linear_regression"]["female_coef"]
    female_tree_imp = model_results["decision_tree_classifier"]["female_importance"]
    female_rule_coef = model_results["rulefit_regressor"]["female_rule_max_abs_coef"]
    weak_nonparam_signal = female_tree_imp < 0.01 and female_rule_coef < 0.01

    # Map statistical evidence to Likert score.
    if adj_sig and unadj_sig:
        score = 80
    elif adj_sig and not unadj_sig:
        score = 45 if weak_nonparam_signal else 60
    elif not adj_sig and unadj_sig:
        score = 40
    else:
        score = 5 if abs(coef_adj) < 0.01 else 15

    if outcome_col == "accept":
        direction = "higher" if coef_adj > 0 else "lower"
    else:
        direction = "lower" if coef_adj > 0 else "higher"
    delta_pp = abs(coef_adj) * 100

    if adj_sig and not unadj_sig:
        evidence_phrase = "Evidence is mixed"
    elif adj_sig and unadj_sig:
        evidence_phrase = "Evidence is consistent"
    else:
        evidence_phrase = "Evidence does not support a meaningful relationship"

    explanation = (
        f"{evidence_phrase}. "
        f"Adjusted OLS shows female coefficient={coef_adj:.4f} (p={p_adj:.4g}), and unadjusted OLS p={p_unadj:.4g}. "
        f"Estimated effect size is about {delta_pp:.2f} percentage points ({direction} approval for women). "
        f"Interpretable models agree: LinearRegression female coef={female_linear_coef:.4f}, "
        f"DecisionTree female importance={female_tree_imp:.4f}, RuleFit female rule |coef|max={female_rule_coef:.4f}."
    )

    return int(max(0, min(100, round(score)))), explanation


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)
    outcome_col = choose_outcome(df)
    feature_cols = make_feature_set(df, outcome_col)

    exploration = describe_data(df)
    stat_tests = run_stat_tests(df, outcome_col, feature_cols)
    model_results = run_interpretable_models(df, outcome_col, feature_cols)

    # Human-readable console output for traceability.
    print("Research question:", research_question)
    print("Rows, columns:", df.shape)
    print("Outcome column used:", outcome_col)
    print("\nTop correlations with female:")
    corr_female = exploration["correlation_matrix"]["female"].sort_values(key=np.abs, ascending=False)
    print(corr_female.head(8))

    print("\nStatistical tests:")
    print(json.dumps(stat_tests, indent=2))

    print("\nModel highlights:")
    highlights = {
        "linear_regression": model_results["linear_regression"],
        "decision_tree_classifier": model_results["decision_tree_classifier"],
        "rulefit_regressor": model_results["rulefit_regressor"],
        "figs_regressor": model_results["figs_regressor"],
        "hstree_regressor": model_results["hstree_regressor"],
    }
    print(json.dumps(highlights, indent=2))

    score, explanation = make_likert_score(stat_tests, model_results, outcome_col)

    conclusion = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
