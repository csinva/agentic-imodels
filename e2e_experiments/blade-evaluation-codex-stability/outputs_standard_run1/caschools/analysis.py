import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]
    print_section("Research Question")
    print(research_question)

    df = pd.read_csv("caschools.csv")
    df["str_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    print_section("Data Overview")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print_section("Summary Statistics (Numeric Columns)")
    print(df[numeric_cols].describe().T)

    dist_cols = ["str_ratio", "avg_score", "income", "english", "lunch", "calworks", "expenditure"]
    print_section("Distribution Diagnostics (Skewness)")
    print(df[dist_cols].skew().sort_values(ascending=False))

    corr_cols = ["str_ratio", "avg_score", "read", "math", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    corr = df[corr_cols].corr(numeric_only=True)
    print_section("Correlation Matrix (Selected Variables)")
    print(corr.round(3))

    y = df["avg_score"].astype(float)
    x_str = df["str_ratio"].astype(float)

    print_section("Statistical Tests")
    pearson_r, pearson_p = stats.pearsonr(x_str, y)
    spearman_rho, spearman_p = stats.spearmanr(x_str, y)
    print(f"Pearson correlation (str_ratio vs avg_score): r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman correlation (str_ratio vs avg_score): rho={spearman_rho:.4f}, p={spearman_p:.4g}")

    median_str = x_str.median()
    low_str_scores = y[x_str <= median_str]
    high_str_scores = y[x_str > median_str]
    t_stat, t_p = stats.ttest_ind(low_str_scores, high_str_scores, equal_var=False)
    print(
        "Welch t-test (low STR <= median vs high STR > median): "
        f"t={t_stat:.4f}, p={t_p:.4g}, "
        f"mean_low={low_str_scores.mean():.3f}, mean_high={high_str_scores.mean():.3f}"
    )

    df["str_quartile"] = pd.qcut(df["str_ratio"], 4, labels=["Q1_lowest_STR", "Q2", "Q3", "Q4_highest_STR"])
    quartile_groups = [df.loc[df["str_quartile"] == q, "avg_score"] for q in df["str_quartile"].cat.categories]
    anova_f, anova_p = stats.f_oneway(*quartile_groups)
    quartile_means = df.groupby("str_quartile")["avg_score"].mean()
    print(f"ANOVA across STR quartiles: F={anova_f:.4f}, p={anova_p:.4g}")
    print("Quartile means:")
    print(quartile_means)

    print_section("OLS Regression")
    X_simple = sm.add_constant(df[["str_ratio"]])
    ols_simple = sm.OLS(y, X_simple).fit()
    print("Simple OLS: avg_score ~ str_ratio")
    print(ols_simple.summary())

    controls = ["str_ratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_multiple = sm.add_constant(df[controls])
    ols_multiple = sm.OLS(y, X_multiple).fit()
    print("\nMultiple OLS: avg_score ~ str_ratio + controls")
    print(ols_multiple.summary())

    print_section("Interpretable Models (scikit-learn)")
    features = ["str_ratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    X = df[features].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    sklearn_models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "Lasso(alpha=0.05)": Lasso(alpha=0.05, max_iter=20000),
        "DecisionTreeRegressor(max_depth=3)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42),
    }

    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_r2 = r2_score(y_test, preds)
        print(f"\n{name}: test R^2={test_r2:.4f}")

        if hasattr(model, "coef_"):
            coefs = pd.Series(model.coef_, index=features).sort_values(key=np.abs, ascending=False)
            print("Top coefficients (absolute magnitude):")
            print(coefs)
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            print("Feature importances:")
            print(fi)

    print_section("Interpretable Models (imodels)")
    imodels_ok = True
    imodels_results = {}
    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

        imodels_models = {
            "RuleFitRegressor": RuleFitRegressor(),
            "FIGSRegressor": FIGSRegressor(),
            "HSTreeRegressor": HSTreeRegressor(),
        }

        for name, model in imodels_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            test_r2 = r2_score(y_test, preds)
            imodels_results[name] = {"r2": safe_float(test_r2)}
            print(f"\n{name}: test R^2={test_r2:.4f}")

            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                print("Feature importances:")
                print(fi)

            if hasattr(model, "get_rules"):
                rules = model.get_rules()
                if isinstance(rules, pd.DataFrame) and {"rule", "coef"}.issubset(set(rules.columns)):
                    active_rules = rules.loc[rules["coef"] != 0].copy()
                    if not active_rules.empty:
                        active_rules["abs_coef"] = active_rules["coef"].abs()
                        active_rules = active_rules.sort_values("abs_coef", ascending=False).head(10)
                        print("Top active rules:")
                        print(active_rules[["rule", "coef", "support"]])
    except Exception as e:
        imodels_ok = False
        print(f"imodels models failed to run: {e}")

    print_section("Conclusion Synthesis")
    simple_coef = safe_float(ols_simple.params.get("str_ratio", np.nan))
    simple_p = safe_float(ols_simple.pvalues.get("str_ratio", np.nan))
    multiple_coef = safe_float(ols_multiple.params.get("str_ratio", np.nan))
    multiple_p = safe_float(ols_multiple.pvalues.get("str_ratio", np.nan))

    score = 50
    if (pearson_p < 0.05) and (pearson_r < 0):
        score += 20
    else:
        score -= 20

    if np.isfinite(simple_p) and (simple_p < 0.05) and (simple_coef < 0):
        score += 15
    else:
        score -= 15

    if np.isfinite(multiple_p) and (multiple_p < 0.05) and (multiple_coef < 0):
        score += 20
    else:
        score -= 20

    if (t_p < 0.05) and (low_str_scores.mean() > high_str_scores.mean()):
        score += 10
    else:
        score -= 10

    decreasing_quartile_means = all(np.diff(quartile_means.values) < 0)
    if (anova_p < 0.05) and decreasing_quartile_means:
        score += 10
    elif anova_p < 0.05:
        score += 5
    else:
        score -= 5

    score = int(np.clip(round(score), 0, 100))

    explanation_parts = [
        f"Pearson r between student-teacher ratio and average score was {pearson_r:.3f} (p={pearson_p:.3g}), indicating a {'negative' if pearson_r < 0 else 'non-negative'} association.",
        f"Simple OLS estimated str_ratio coefficient {simple_coef:.3f} (p={simple_p:.3g}).",
        f"Controlled OLS estimated str_ratio coefficient {multiple_coef:.3f} (p={multiple_p:.3g}) after adjusting for income, english learners, lunch, calworks, expenditure, and computer access.",
        f"Welch t-test comparing low vs high STR groups gave p={t_p:.3g} with mean scores {low_str_scores.mean():.2f} vs {high_str_scores.mean():.2f}.",
        f"ANOVA across STR quartiles gave p={anova_p:.3g} with quartile means {quartile_means.round(2).to_dict()}.",
    ]
    if imodels_ok:
        rf_r2 = imodels_results.get("RuleFitRegressor", {}).get("r2", np.nan)
        figs_r2 = imodels_results.get("FIGSRegressor", {}).get("r2", np.nan)
        hs_r2 = imodels_results.get("HSTreeRegressor", {}).get("r2", np.nan)
        explanation_parts.append(
            f"Interpretable imodels regressors achieved test R^2 values of RuleFit={rf_r2:.3f}, FIGS={figs_r2:.3f}, HSTree={hs_r2:.3f}, supporting interpretation consistency."
        )
    else:
        explanation_parts.append("imodels regressors could not be evaluated in this runtime, so the conclusion leans on statistical tests and sklearn interpretable models.")

    explanation = " ".join(explanation_parts)
    result = {"response": score, "explanation": explanation}
    print(f"Likert score (0-100): {score}")
    print(explanation)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
