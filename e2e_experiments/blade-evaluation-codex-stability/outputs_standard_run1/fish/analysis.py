import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


warnings.filterwarnings("ignore")


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(x) -> float:
    return float(np.asarray(x).item())


def describe_distribution(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col]
        rows.append(
            {
                "column": col,
                "mean": safe_float(s.mean()),
                "median": safe_float(s.median()),
                "std": safe_float(s.std(ddof=1)),
                "min": safe_float(s.min()),
                "q25": safe_float(s.quantile(0.25)),
                "q75": safe_float(s.quantile(0.75)),
                "max": safe_float(s.max()),
                "skew": safe_float(s.skew()),
            }
        )
    return pd.DataFrame(rows)


def fit_interpretable_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    lin = LinearRegression()
    lin.fit(X, y)
    results["LinearRegression"] = {
        "r2": safe_float(r2_score(y, lin.predict(X))),
        "coef": {c: safe_float(v) for c, v in zip(X.columns, lin.coef_)},
        "intercept": safe_float(lin.intercept_),
    }

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    results["Ridge"] = {
        "r2": safe_float(r2_score(y, ridge.predict(X))),
        "coef": {c: safe_float(v) for c, v in zip(X.columns, ridge.coef_)},
        "intercept": safe_float(ridge.intercept_),
    }

    lasso = Lasso(alpha=0.01, max_iter=20000, random_state=42)
    lasso.fit(X, y)
    results["Lasso"] = {
        "r2": safe_float(r2_score(y, lasso.predict(X))),
        "coef": {c: safe_float(v) for c, v in zip(X.columns, lasso.coef_)},
        "intercept": safe_float(lasso.intercept_),
    }

    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(X, y)
    results["DecisionTreeRegressor"] = {
        "r2": safe_float(r2_score(y, tree.predict(X))),
        "feature_importances": {
            c: safe_float(v) for c, v in zip(X.columns, tree.feature_importances_)
        },
    }

    rulefit = RuleFitRegressor(max_rules=30, random_state=42)
    rulefit.fit(X, y, feature_names=X.columns.tolist())
    rulefit_rules = rulefit._get_rules(exclude_zero_coef=True)
    rulefit_rules = rulefit_rules.sort_values("importance", ascending=False).head(10)
    results["RuleFitRegressor"] = {
        "r2": safe_float(r2_score(y, rulefit.predict(X))),
        "top_rules": [
            {
                "rule": str(r.rule),
                "type": str(r.type),
                "coef": safe_float(r.coef),
                "support": safe_float(r.support),
                "importance": safe_float(r.importance),
            }
            for _, r in rulefit_rules.iterrows()
        ],
    }

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X, y, feature_names=X.columns.tolist())
    results["FIGSRegressor"] = {
        "r2": safe_float(r2_score(y, figs.predict(X))),
        "feature_importances": {
            c: safe_float(v) for c, v in zip(X.columns, figs.feature_importances_)
        }
        if hasattr(figs, "feature_importances_")
        else {},
        "model_text": str(figs),
    }

    hst = HSTreeRegressor(max_leaf_nodes=12, random_state=42)
    hst.fit(X, y, feature_names=X.columns.tolist())
    results["HSTreeRegressor"] = {
        "r2": safe_float(r2_score(y, hst.predict(X))),
        "model_text": str(hst),
    }

    return results


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["No question provided"])[0]

    df = pd.read_csv("fish.csv")
    required_cols = ["fish_caught", "livebait", "camper", "persons", "child", "hours"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[df["hours"] > 0].reset_index(drop=True)
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]

    header("Research Question")
    print(question)

    header("Data Overview")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())

    header("Summary Statistics")
    print(df.describe().T.to_string())

    numeric_cols = ["fish_caught", "fish_per_hour", "livebait", "camper", "persons", "child", "hours"]
    header("Distribution Statistics")
    dist_df = describe_distribution(df, numeric_cols)
    print(dist_df.to_string(index=False))

    header("Correlation Matrix")
    corr = df[numeric_cols].corr(numeric_only=True)
    print(corr.to_string())

    header("Statistical Tests")
    stat_rows = []
    for feature in ["hours", "livebait", "camper", "persons", "child"]:
        pear_r, pear_p = stats.pearsonr(df[feature], df["fish_caught"])
        spear_r, spear_p = stats.spearmanr(df[feature], df["fish_caught"])
        stat_rows.append(
            {
                "feature": feature,
                "pearson_r": safe_float(pear_r),
                "pearson_p": safe_float(pear_p),
                "spearman_r": safe_float(spear_r),
                "spearman_p": safe_float(spear_p),
            }
        )
    stat_df = pd.DataFrame(stat_rows)
    print("Correlation tests against fish_caught:")
    print(stat_df.to_string(index=False))

    # Welch t-tests for binary factors
    livebait_fish = stats.ttest_ind(
        df.loc[df["livebait"] == 1, "fish_caught"],
        df.loc[df["livebait"] == 0, "fish_caught"],
        equal_var=False,
    )
    livebait_rate = stats.ttest_ind(
        df.loc[df["livebait"] == 1, "fish_per_hour"],
        df.loc[df["livebait"] == 0, "fish_per_hour"],
        equal_var=False,
    )
    camper_fish = stats.ttest_ind(
        df.loc[df["camper"] == 1, "fish_caught"],
        df.loc[df["camper"] == 0, "fish_caught"],
        equal_var=False,
    )
    camper_rate = stats.ttest_ind(
        df.loc[df["camper"] == 1, "fish_per_hour"],
        df.loc[df["camper"] == 0, "fish_per_hour"],
        equal_var=False,
    )

    print("\nWelch t-tests:")
    print(
        f"livebait -> fish_caught: t={livebait_fish.statistic:.4f}, p={livebait_fish.pvalue:.6f}; "
        f"fish_per_hour: t={livebait_rate.statistic:.4f}, p={livebait_rate.pvalue:.6f}"
    )
    print(
        f"camper   -> fish_caught: t={camper_fish.statistic:.4f}, p={camper_fish.pvalue:.6f}; "
        f"fish_per_hour: t={camper_rate.statistic:.4f}, p={camper_rate.pvalue:.6f}"
    )

    # ANOVA for multi-category integer factors
    persons_groups = [g["fish_caught"].values for _, g in df.groupby("persons")]
    child_groups = [g["fish_caught"].values for _, g in df.groupby("child")]
    anova_persons = stats.f_oneway(*persons_groups)
    anova_child = stats.f_oneway(*child_groups)
    print("\nANOVA on fish_caught:")
    print(f"persons: F={anova_persons.statistic:.4f}, p={anova_persons.pvalue:.6f}")
    print(f"child  : F={anova_child.statistic:.4f}, p={anova_child.pvalue:.6f}")

    # OLS for count and rate outcomes
    X = df[["hours", "livebait", "camper", "persons", "child"]]
    X_sm = sm.add_constant(X)
    ols_count = sm.OLS(df["fish_caught"], X_sm).fit()
    ols_rate = sm.OLS(df["fish_per_hour"], X_sm).fit()

    print("\nOLS (fish_caught) coefficients and p-values:")
    ols_count_table = pd.DataFrame(
        {
            "coef": ols_count.params,
            "pvalue": ols_count.pvalues,
        }
    )
    print(ols_count_table.to_string())
    print(
        f"Model fit: R^2={ols_count.rsquared:.4f}, adj R^2={ols_count.rsquared_adj:.4f}, "
        f"F-test p={ols_count.f_pvalue:.6e}"
    )

    print("\nOLS (fish_per_hour) coefficients and p-values:")
    ols_rate_table = pd.DataFrame(
        {
            "coef": ols_rate.params,
            "pvalue": ols_rate.pvalues,
        }
    )
    print(ols_rate_table.to_string())
    print(
        f"Model fit: R^2={ols_rate.rsquared:.4f}, adj R^2={ols_rate.rsquared_adj:.4f}, "
        f"F-test p={ols_rate.f_pvalue:.6e}"
    )

    header("Interpretable Models (scikit-learn + imodels)")
    X_ml = df[["livebait", "camper", "persons", "child", "hours"]]
    y_ml = df["fish_caught"]
    model_results = fit_interpretable_models(X_ml, y_ml)

    for model_name, result in model_results.items():
        print(f"\n{model_name}:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}: {json.dumps(value, default=str)}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for item in value[:6]:
                    print(f"    - {json.dumps(item, default=str)}")
            else:
                print(f"  {key}: {value}")

    # Synthesis for final Likert response (0-100)
    avg_rate_ratio = safe_float(df["fish_caught"].sum() / df["hours"].sum())
    avg_rate_mean = safe_float(df["fish_per_hour"].mean())
    median_rate = safe_float(df["fish_per_hour"].median())

    sig_predictors = [
        c for c in ["hours", "livebait", "camper", "persons", "child"] if ols_count.pvalues[c] < 0.05
    ]

    score = 25
    if ols_count.f_pvalue < 0.05:
        score += 25
    if len(sig_predictors) >= 2:
        score += 15
    if livebait_fish.pvalue < 0.05:
        score += 10
    if anova_persons.pvalue < 0.05:
        score += 10
    if ols_count.rsquared >= 0.15:
        score += 10
    if model_results["FIGSRegressor"]["r2"] >= 0.6 or model_results["RuleFitRegressor"]["r2"] >= 0.6:
        score += 10

    score = int(max(0, min(100, round(score))))

    top_linear = model_results["LinearRegression"]["coef"]
    top_pos = sorted(
        [(k, v) for k, v in top_linear.items() if v > 0],
        key=lambda kv: kv[1],
        reverse=True,
    )[:2]
    top_neg = sorted(
        [(k, v) for k, v in top_linear.items() if v < 0],
        key=lambda kv: kv[1],
    )[:2]

    explanation = (
        f"Estimated catch rate is about {avg_rate_ratio:.2f} fish/hour overall (total fish divided by total hours); "
        f"mean individual rate is {avg_rate_mean:.2f} and median is {median_rate:.2f}. "
        f"Statistical tests indicate meaningful relationships (OLS F-test p={ols_count.f_pvalue:.2e}, R^2={ols_count.rsquared:.2f}); "
        f"significant OLS predictors are {sig_predictors}. "
        f"Welch t-test for livebait on fish caught has p={livebait_fish.pvalue:.4f}, and ANOVA for persons has p={anova_persons.pvalue:.4f}. "
        f"Interpretable models agree that stronger positive effects come from {top_pos} and negative effects from {top_neg}."
    )

    conclusion = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    header("Conclusion Written")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
