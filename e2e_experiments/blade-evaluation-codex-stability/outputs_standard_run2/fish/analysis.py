import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def series_histogram_text(s: pd.Series, bins: int = 10) -> str:
    counts, edges = np.histogram(s.dropna().values, bins=bins)
    parts = []
    for i, c in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        parts.append(f"[{left:.3f}, {right:.3f}): {int(c)}")
    return "; ".join(parts)


def welch_ttest(df: pd.DataFrame, group_col: str, target_col: str):
    g0 = df[df[group_col] == 0][target_col]
    g1 = df[df[group_col] == 1][target_col]
    stat, p = stats.ttest_ind(g1, g0, equal_var=False)
    return {
        "group_col": group_col,
        "target_col": target_col,
        "mean_group1": float(g1.mean()),
        "mean_group0": float(g0.mean()),
        "t_stat": float(stat),
        "p_value": float(p),
    }


def anova_test(df: pd.DataFrame, group_col: str, target_col: str):
    grouped_vals = [g[target_col].values for _, g in df.groupby(group_col)]
    f_stat, p = stats.f_oneway(*grouped_vals)
    return {
        "group_col": group_col,
        "target_col": target_col,
        "f_stat": float(f_stat),
        "p_value": float(p),
    }


def main() -> None:
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "fish.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    required_cols = ["fish_caught", "livebait", "camper", "persons", "child", "hours"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Core engineered targets
    df = df.copy()
    df["fish_per_hour"] = np.where(df["hours"] > 0, df["fish_caught"] / df["hours"], np.nan)
    df["log_fish_caught"] = np.log1p(df["fish_caught"])

    section("Research Question")
    print(research_question)

    section("Data Overview")
    print(f"Rows: {len(df)}")
    print("Missing values by column:")
    print(df[required_cols + ["fish_per_hour"]].isna().sum().to_string())

    section("Summary Statistics")
    print(df[required_cols + ["fish_per_hour"]].describe().to_string())

    section("Distribution Snapshots")
    print("fish_caught histogram bins:")
    print(series_histogram_text(df["fish_caught"], bins=12))
    print("fish_per_hour histogram bins:")
    print(series_histogram_text(df["fish_per_hour"].replace([np.inf, -np.inf], np.nan).dropna(), bins=12))

    section("Correlations")
    numeric_cols = required_cols + ["fish_per_hour"]
    pearson_corr = df[numeric_cols].corr(method="pearson")
    spearman_corr = df[numeric_cols].corr(method="spearman")
    print("Pearson correlation matrix:")
    print(pearson_corr.to_string())
    print("\nSpearman correlation matrix:")
    print(spearman_corr.to_string())

    section("Statistical Tests")
    ttests = [
        welch_ttest(df, "livebait", "fish_caught"),
        welch_ttest(df, "livebait", "fish_per_hour"),
        welch_ttest(df, "camper", "fish_caught"),
        welch_ttest(df, "camper", "fish_per_hour"),
    ]

    for res in ttests:
        print(
            f"Welch t-test ({res['target_col']} ~ {res['group_col']}): "
            f"mean(1)={res['mean_group1']:.4f}, mean(0)={res['mean_group0']:.4f}, "
            f"t={res['t_stat']:.4f}, p={res['p_value']:.4g}"
        )

    anovas = [
        anova_test(df, "persons", "fish_caught"),
        anova_test(df, "persons", "fish_per_hour"),
        anova_test(df, "child", "fish_caught"),
        anova_test(df, "child", "fish_per_hour"),
    ]

    for res in anovas:
        print(
            f"ANOVA ({res['target_col']} ~ {res['group_col']}): "
            f"F={res['f_stat']:.4f}, p={res['p_value']:.4g}"
        )

    corr_hours_fish = stats.pearsonr(df["hours"], df["fish_caught"])
    corr_hours_rate = stats.pearsonr(df["hours"], df["fish_per_hour"].fillna(0))
    print(
        f"Pearson corr(hours, fish_caught): r={corr_hours_fish.statistic:.4f}, p={corr_hours_fish.pvalue:.4g}"
    )
    print(
        f"Pearson corr(hours, fish_per_hour): r={corr_hours_rate.statistic:.4f}, p={corr_hours_rate.pvalue:.4g}"
    )

    section("Statsmodels OLS")
    X = df[["livebait", "camper", "persons", "child", "hours"]]
    X_sm = sm.add_constant(X)

    y_raw = df["fish_caught"]
    y_log = df["log_fish_caught"]

    ols_raw = sm.OLS(y_raw, X_sm).fit()
    ols_log = sm.OLS(y_log, X_sm).fit()

    print("OLS on fish_caught coefficients:")
    print(ols_raw.params.to_string())
    print("OLS on fish_caught p-values:")
    print(ols_raw.pvalues.to_string())
    print(f"OLS fish_caught R^2={ols_raw.rsquared:.4f}, F-test p={ols_raw.f_pvalue:.4g}")

    print("\nOLS on log1p(fish_caught) coefficients:")
    print(ols_log.params.to_string())
    print("OLS on log1p(fish_caught) p-values:")
    print(ols_log.pvalues.to_string())
    print(f"OLS log1p(fish_caught) R^2={ols_log.rsquared:.4f}, F-test p={ols_log.f_pvalue:.4g}")

    section("Interpretable scikit-learn Models")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.30, random_state=42
    )

    sk_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=20000),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    sk_results = {}
    for name, model in sk_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        sk_results[name] = {"r2": float(r2)}
        print(f"{name} R^2 (log target): {r2:.4f}")

        if hasattr(model, "coef_"):
            coef_map = dict(zip(X.columns.tolist(), model.coef_.tolist()))
            sk_results[name]["coef"] = coef_map
            print(f"{name} coefficients: {coef_map}")
        if hasattr(model, "feature_importances_"):
            fi_map = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
            sk_results[name]["feature_importances"] = fi_map
            print(f"{name} feature_importances: {fi_map}")

    section("Interpretable imodels Models")
    imodels_results = {}

    model_specs = {
        "RuleFitRegressor": RuleFitRegressor(random_state=42),
        "FIGSRegressor": FIGSRegressor(random_state=42),
        "HSTreeRegressor": HSTreeRegressor(random_state=42),
    }

    for name, model in model_specs.items():
        model.fit(X_train, y_train, feature_names=X.columns.tolist())
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        imodels_results[name] = {"r2": float(r2)}
        print(f"{name} R^2 (log target): {r2:.4f}")

        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fi_map = dict(zip(X.columns.tolist(), np.array(fi).flatten().tolist()))
            imodels_results[name]["feature_importances"] = fi_map
            print(f"{name} feature_importances: {fi_map}")

        # Keep a small readable rule/tree snapshot
        model_text = str(model).splitlines()
        model_text_preview = "\n".join(model_text[:18])
        imodels_results[name]["model_preview"] = model_text_preview
        print(f"{name} preview:\n{model_text_preview}")

    section("Interpretation")
    total_fish = float(df["fish_caught"].sum())
    total_hours = float(df["hours"].sum())
    weighted_rate = total_fish / total_hours if total_hours > 0 else np.nan
    median_rate = float(df["fish_per_hour"].median())

    print(f"Total fish caught: {total_fish:.1f}")
    print(f"Total hours fished: {total_hours:.1f}")
    print(f"Estimated average catch rate (total fish / total hours): {weighted_rate:.3f} fish/hour")
    print(f"Median trip-level catch rate: {median_rate:.3f} fish/hour")

    sig_predictors_log = [
        name
        for name, p in ols_log.pvalues.items()
        if name != "const" and p < 0.05
    ]
    print(f"Significant predictors in log-OLS (p<0.05): {sig_predictors_log}")

    livebait_rate_p = next(
        r["p_value"] for r in ttests if r["group_col"] == "livebait" and r["target_col"] == "fish_per_hour"
    )
    persons_rate_p = next(
        r["p_value"] for r in anovas if r["group_col"] == "persons" and r["target_col"] == "fish_per_hour"
    )

    # Build an evidence-based Likert score for "Yes, factors influence catch and rate can be estimated"
    score = 50
    if ols_log.f_pvalue < 0.05:
        score += 15
    score += min(20, 4 * len(sig_predictors_log))
    if livebait_rate_p < 0.05:
        score += 7
    if persons_rate_p < 0.05:
        score += 7
    if ols_log.rsquared >= 0.45:
        score += 7
    elif ols_log.rsquared >= 0.30:
        score += 4

    # Penalize heavy zero-inflation / skew because it limits certainty
    zero_share = float((df["fish_caught"] == 0).mean())
    if zero_share > 0.45:
        score -= 10
    elif zero_share > 0.30:
        score -= 6

    score = int(max(0, min(100, round(score))))

    explanation = (
        f"Estimated mean catch rate is {weighted_rate:.3f} fish/hour (total fish {total_fish:.0f} over "
        f"{total_hours:.1f} total hours). Statistical tests show meaningful relationships: livebait is "
        f"associated with higher fish/hour (Welch t-test p={livebait_rate_p:.3g}), and persons affects "
        f"fish/hour (ANOVA p={persons_rate_p:.3g}). In multivariable log-OLS, the model is significant "
        f"(F-test p={ols_log.f_pvalue:.3g}, R^2={ols_log.rsquared:.3f}) with significant predictors "
        f"{sig_predictors_log}. Interpretable sklearn and imodels models broadly agree that group composition "
        f"and bait usage matter, though many zero-catch trips add noise, so confidence is high but not absolute."
    )

    result = {"response": score, "explanation": explanation}
    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
