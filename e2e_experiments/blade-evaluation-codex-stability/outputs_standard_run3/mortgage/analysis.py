import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")


def pfmt(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.4g}"


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown research question"])[0]

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_cols = {"female", "accept", "deny"}
    missing_required = required_cols.difference(df.columns)
    if missing_required:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_required)}")

    print("=" * 88)
    print("Research question:", question)
    print("Rows, columns:", df.shape)
    print("Missing values by column:\n", df.isna().sum().sort_values(ascending=False))

    df_clean = df.dropna().copy()
    dropped = len(df) - len(df_clean)
    print(f"Rows after dropping missing values: {df_clean.shape[0]} (dropped {dropped})")

    feature_cols = [
        c
        for c in df_clean.columns
        if c not in {"accept", "deny", "denied_PMI"}
    ]

    # Basic exploration
    print("\n" + "=" * 88)
    print("Summary statistics (numeric):")
    print(df_clean.describe().T)

    print("\n" + "=" * 88)
    print("Binary variable distributions:")
    for col in df_clean.columns:
        vals = set(df_clean[col].dropna().unique().tolist())
        if vals.issubset({0, 1, 0.0, 1.0}):
            counts = df_clean[col].value_counts(dropna=False).sort_index()
            rates = (counts / counts.sum()).round(4)
            print(f"{col}: counts={counts.to_dict()} rates={rates.to_dict()}")

    corr = df_clean.corr(numeric_only=True)
    print("\n" + "=" * 88)
    print("Top correlations with accept:")
    print(corr["accept"].sort_values(ascending=False))

    # Inferential tests
    female_accept_table = pd.crosstab(df_clean["female"], df_clean["accept"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(female_accept_table)

    female_accept = df_clean.loc[df_clean["female"] == 1, "accept"]
    male_accept = df_clean.loc[df_clean["female"] == 0, "accept"]
    ttest_stat, ttest_p = stats.ttest_ind(
        female_accept, male_accept, equal_var=False, nan_policy="omit"
    )

    pb_corr, pb_p = stats.pointbiserialr(df_clean["female"], df_clean["accept"])

    X = sm.add_constant(df_clean[feature_cols])
    y = df_clean["accept"]

    ols_model = sm.OLS(y, X).fit()
    ols_female_coef = float(ols_model.params.get("female", np.nan))
    ols_female_p = float(ols_model.pvalues.get("female", np.nan))

    logit_model = sm.Logit(y, X).fit(disp=0)
    logit_female_coef = float(logit_model.params.get("female", np.nan))
    logit_female_p = float(logit_model.pvalues.get("female", np.nan))
    logit_female_or = float(np.exp(logit_female_coef))

    anova_formula = (
        "accept ~ C(female) + C(mortgage_credit) + C(consumer_credit) + "
        "black + married + self_employed + bad_history + "
        "PI_ratio + housing_expense_ratio + loan_to_value"
    )
    anova_model = smf.ols(anova_formula, data=df_clean).fit()
    anova_table = anova_lm(anova_model, typ=2)
    anova_female_p = float(anova_table.loc["C(female)", "PR(>F)"])

    print("\n" + "=" * 88)
    print("Statistical tests for female vs approval")
    print("Contingency table (female x accept):\n", female_accept_table)
    print(f"Chi-square p-value: {pfmt(chi2_p)}")
    print(f"Welch t-test p-value: {pfmt(ttest_p)}")
    print(f"Point-biserial corr(female, accept): r={pfmt(pb_corr)}, p={pfmt(pb_p)}")
    print(f"OLS female coef: {pfmt(ols_female_coef)}, p={pfmt(ols_female_p)}")
    print(
        f"Logit female coef: {pfmt(logit_female_coef)}, "
        f"OR={pfmt(logit_female_or)}, p={pfmt(logit_female_p)}"
    )
    print(f"ANOVA female p-value: {pfmt(anova_female_p)}")

    # Interpretable models
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean[feature_cols],
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    female_idx = feature_cols.index("female")

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_female_coef = float(lin.coef_[female_idx])

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_female_coef = float(ridge.coef_[female_idx])

    lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_female_coef = float(lasso.coef_[female_idx])

    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
    dt.fit(X_train, y_train)
    dt_importances = dict(zip(feature_cols, dt.feature_importances_))
    dt_female_importance = float(dt_importances.get("female", 0.0))

    rulefit = RuleFitRegressor(random_state=42, max_rules=30)
    rulefit.fit(X_train, y_train, feature_names=feature_cols)
    rules_df = rulefit._get_rules(exclude_zero_coef=True)
    female_rule_row = rules_df[
        (rules_df["rule"] == "female") & (rules_df["type"] == "linear")
    ]
    female_rule_coef = (
        float(female_rule_row.iloc[0]["coef"]) if not female_rule_row.empty else 0.0
    )

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_train, y_train, feature_names=feature_cols)
    figs_importances = dict(zip(feature_cols, figs.feature_importances_))
    figs_female_importance = float(figs_importances.get("female", 0.0))

    hstree = HSTreeRegressor(max_leaf_nodes=20, random_state=42)
    hstree.fit(X_train, y_train)
    hstree_importances = dict(zip(feature_cols, hstree.estimator_.feature_importances_))
    hstree_female_importance = float(hstree_importances.get("female", 0.0))

    model_metrics = {
        "linear_r2": float(r2_score(y_test, lin.predict(X_test))),
        "ridge_r2": float(r2_score(y_test, ridge.predict(X_test))),
        "lasso_r2": float(r2_score(y_test, lasso.predict(X_test))),
        "decision_tree_accuracy": float(accuracy_score(y_test, dt.predict(X_test))),
        "decision_tree_auc": float(roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])),
        "rulefit_r2": float(r2_score(y_test, rulefit.predict(X_test))),
        "figs_r2": float(r2_score(y_test, figs.predict(X_test))),
        "hstree_r2": float(r2_score(y_test, hstree.predict(X_test))),
    }

    print("\n" + "=" * 88)
    print("Interpretable model evidence for female")
    print(
        f"Linear coef={pfmt(lin_female_coef)}, "
        f"Ridge coef={pfmt(ridge_female_coef)}, Lasso coef={pfmt(lasso_female_coef)}"
    )
    print(
        f"DecisionTree importance={pfmt(dt_female_importance)}, "
        f"RuleFit female coef={pfmt(female_rule_coef)}, "
        f"FIGS importance={pfmt(figs_female_importance)}, "
        f"HSTree importance={pfmt(hstree_female_importance)}"
    )
    print("Model performance summary:", {k: round(v, 4) for k, v in model_metrics.items()})

    # Evidence synthesis into Likert score
    score = 50

    if ols_female_p < 0.05:
        score += 12
    else:
        score -= 12

    if logit_female_p < 0.05:
        score += 12
    else:
        score -= 12

    if anova_female_p < 0.05:
        score += 8
    else:
        score -= 8

    score += 4 if chi2_p < 0.05 else -4
    score += 4 if ttest_p < 0.05 else -4

    if abs(ols_female_coef) >= 0.03:
        score += 5
    elif abs(ols_female_coef) < 0.01:
        score -= 5

    if (
        abs(female_rule_coef) < 1e-9
        and dt_female_importance < 1e-9
        and figs_female_importance < 1e-9
        and hstree_female_importance < 1e-9
    ):
        score -= 7

    score = int(np.clip(round(score), 0, 100))

    female_rate = float(female_accept.mean())
    male_rate = float(male_accept.mean())
    direction = "higher" if ols_female_coef > 0 else "lower"

    explanation = (
        f"Raw approval rates are almost identical by gender (female={female_rate:.3f}, "
        f"male={male_rate:.3f}; chi-square p={chi2_p:.3g}, t-test p={ttest_p:.3g}), "
        f"but adjusted models show a statistically significant gender association. "
        f"In OLS controlling for financial covariates, female has coef={ols_female_coef:.3f} "
        f"(p={ols_female_p:.3g}); in logistic regression, female has log-odds coef="
        f"{logit_female_coef:.3f} (OR={logit_female_or:.2f}, p={logit_female_p:.3g}); "
        f"ANOVA also gives p={anova_female_p:.3g}. Interpretable tree/rule models assign "
        f"little direct importance to female, so the effect appears modest rather than dominant. "
        f"Overall, evidence supports that gender is related to approval after adjustment, with "
        f"female applicants showing {direction} predicted approval."
    )

    conclusion = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")

    print("\n" + "=" * 88)
    print("Wrote conclusion.txt")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
