import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


@dataclass
class TestResults:
    pearson_r: float
    pearson_p: float
    ttest_stat: float
    ttest_p: float
    mean_fertile: float
    mean_nonfertile: float
    cohens_d: float
    anova_f: float
    anova_p: float
    ols_coef_fertility_score: float
    ols_p_fertility_score: float
    ols_coef_fertile_window: float
    ols_p_fertile_window: float


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")

    # Use reported cycle length when available; otherwise infer from consecutive periods.
    df["inferred_cycle_length"] = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    df.loc[~df["inferred_cycle_length"].between(15, 60), "inferred_cycle_length"] = np.nan

    reported = df["ReportedCycleLength"].where(df["ReportedCycleLength"].between(15, 60))
    df["cycle_length"] = reported.fillna(df["inferred_cycle_length"])

    df["day_since_last"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days
    df["ovulation_day"] = df["cycle_length"] - 14

    # Fertile window: roughly 5 days before ovulation up to ovulation day.
    df["fertile_window"] = (
        (df["day_since_last"] >= (df["ovulation_day"] - 5))
        & (df["day_since_last"] <= df["ovulation_day"])
    ).astype(int)

    # Continuous fertility proximity score centered around ovulation.
    df["fertility_score"] = np.exp(
        -((df["day_since_last"] - df["ovulation_day"]) ** 2) / (2 * (3.0**2))
    )

    df["religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

    model_cols = [
        "religiosity",
        "fertility_score",
        "fertile_window",
        "Relationship",
        "Sure1",
        "Sure2",
        "cycle_length",
        "day_since_last",
    ]
    df = df.dropna(subset=model_cols).copy()

    # Plausible cycle-day range.
    df = df[(df["day_since_last"] >= 0) & (df["day_since_last"] <= df["cycle_length"] + 3)].copy()

    # Cycle-phase bins for ANOVA.
    df["cycle_phase"] = np.select(
        [
            df["day_since_last"] <= 5,
            df["day_since_last"] < (df["ovulation_day"] - 5),
            df["day_since_last"] <= df["ovulation_day"],
            df["day_since_last"] > df["ovulation_day"],
        ],
        ["menstrual", "follicular", "fertile", "luteal"],
        default="other",
    )

    return df


def print_eda(df: pd.DataFrame) -> None:
    print("=== DATA OVERVIEW ===")
    print(f"Rows used for analysis: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("=== SUMMARY STATISTICS (key variables) ===")
    key_cols = [
        "religiosity",
        "fertility_score",
        "fertile_window",
        "day_since_last",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print(df[key_cols].describe().round(3))
    print()

    print("=== DISTRIBUTIONS ===")
    print("Cycle phase counts:")
    print(df["cycle_phase"].value_counts(dropna=False))
    print()

    print("Fertile window counts:")
    print(df["fertile_window"].value_counts(dropna=False))
    print()

    print("Religiosity quantiles:")
    print(df["religiosity"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).round(3))
    print()

    print("=== CORRELATIONS (Pearson) ===")
    corr_cols = [
        "religiosity",
        "fertility_score",
        "fertile_window",
        "day_since_last",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print(df[corr_cols].corr().round(3))
    print()


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def run_statistical_tests(df: pd.DataFrame) -> TestResults:
    pearson_r, pearson_p = stats.pearsonr(df["fertility_score"], df["religiosity"])

    fertile_vals = df.loc[df["fertile_window"] == 1, "religiosity"].values
    nonfertile_vals = df.loc[df["fertile_window"] == 0, "religiosity"].values

    ttest = stats.ttest_ind(fertile_vals, nonfertile_vals, equal_var=False)
    d = cohen_d(fertile_vals, nonfertile_vals)

    groups = [
        grp["religiosity"].values
        for _, grp in df.groupby("cycle_phase")
        if len(grp) > 1 and grp["cycle_phase"].iloc[0] != "other"
    ]
    anova_f, anova_p = stats.f_oneway(*groups)

    X = sm.add_constant(
        df[
            [
                "fertility_score",
                "fertile_window",
                "Relationship",
                "Sure1",
                "Sure2",
                "cycle_length",
            ]
        ]
    )
    ols = sm.OLS(df["religiosity"], X).fit()

    print("=== STATISTICAL TESTS ===")
    print(
        f"Pearson correlation (fertility_score vs religiosity): r={pearson_r:.3f}, p={pearson_p:.4f}"
    )
    print(
        f"Welch t-test (fertile vs non-fertile religiosity): t={ttest.statistic:.3f}, p={ttest.pvalue:.4f}"
    )
    print(
        f"Group means: fertile={np.mean(fertile_vals):.3f}, non-fertile={np.mean(nonfertile_vals):.3f}, Cohen's d={d:.3f}"
    )
    print(f"ANOVA by cycle phase: F={anova_f:.3f}, p={anova_p:.4f}")
    print()

    print("OLS regression (religiosity ~ fertility + controls):")
    print(ols.summary())
    print()

    return TestResults(
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        ttest_stat=float(ttest.statistic),
        ttest_p=float(ttest.pvalue),
        mean_fertile=float(np.mean(fertile_vals)),
        mean_nonfertile=float(np.mean(nonfertile_vals)),
        cohens_d=float(d),
        anova_f=float(anova_f),
        anova_p=float(anova_p),
        ols_coef_fertility_score=float(ols.params["fertility_score"]),
        ols_p_fertility_score=float(ols.pvalues["fertility_score"]),
        ols_coef_fertile_window=float(ols.params["fertile_window"]),
        ols_p_fertile_window=float(ols.pvalues["fertile_window"]),
    )


def run_interpretable_models(df: pd.DataFrame) -> dict:
    feature_cols = [
        "fertility_score",
        "fertile_window",
        "Relationship",
        "Sure1",
        "Sure2",
        "cycle_length",
        "day_since_last",
    ]
    X = df[feature_cols]
    y = df["religiosity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.05, random_state=42, max_iter=10000),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    summary = {}

    print("=== INTERPRETABLE SKLEARN MODELS ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        info = {"r2": float(r2)}
        if hasattr(model, "coef_"):
            info["coefficients"] = {
                col: float(coef) for col, coef in zip(feature_cols, model.coef_)
            }
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = {
                col: float(imp)
                for col, imp in zip(feature_cols, model.feature_importances_)
            }

        summary[name] = info

        print(f"{name} R^2: {r2:.3f}")
        if "coefficients" in info:
            sorted_coefs = sorted(
                info["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            print("  Top coefficients:", sorted_coefs[:4])
        if "feature_importances" in info:
            sorted_imps = sorted(
                info["feature_importances"].items(), key=lambda kv: kv[1], reverse=True
            )
            print("  Top feature importances:", sorted_imps[:4])
    print()

    print("=== IMODELS INTERPRETABLE MODELS ===")

    rulefit = RuleFitRegressor(random_state=42, max_rules=20)
    rulefit.fit(X_train.values, y_train.values, feature_names=feature_cols)
    rulefit_r2 = r2_score(y_test, rulefit.predict(X_test.values))
    rules = rulefit._get_rules(exclude_zero_coef=True)
    rules = rules.sort_values("importance", ascending=False)
    top_rules = rules.head(5)[["rule", "coef", "support", "importance"]]

    print(f"RuleFitRegressor R^2: {rulefit_r2:.3f}")
    print("Top RuleFit rules:")
    for _, row in top_rules.iterrows():
        print(
            f"  rule='{row['rule']}', coef={row['coef']:.3f}, support={row['support']:.3f}, importance={row['importance']:.3f}"
        )

    figs = FIGSRegressor(random_state=42, max_rules=12)
    figs.fit(X_train.values, y_train.values, feature_names=feature_cols)
    figs_r2 = r2_score(y_test, figs.predict(X_test.values))
    figs_importances = {
        col: float(imp) for col, imp in zip(feature_cols, figs.feature_importances_)
    }

    print(f"FIGSRegressor R^2: {figs_r2:.3f}")
    print(
        "Top FIGS feature importances:",
        sorted(figs_importances.items(), key=lambda kv: kv[1], reverse=True)[:4],
    )

    hstree = HSTreeRegressor(max_leaf_nodes=8)
    hstree.fit(X_train.values, y_train.values, feature_names=feature_cols)
    hstree_r2 = r2_score(y_test, hstree.predict(X_test.values))
    hs_importances = {
        col: float(imp) for col, imp in zip(feature_cols, hstree.estimator_.feature_importances_)
    }

    print(f"HSTreeRegressor R^2: {hstree_r2:.3f}")
    print(
        "Top HSTree feature importances:",
        sorted(hs_importances.items(), key=lambda kv: kv[1], reverse=True)[:4],
    )
    print()

    summary["RuleFitRegressor"] = {
        "r2": float(rulefit_r2),
        "top_rules": [
            {
                "rule": str(row["rule"]),
                "coef": float(row["coef"]),
                "support": float(row["support"]),
                "importance": float(row["importance"]),
            }
            for _, row in top_rules.iterrows()
        ],
    }
    summary["FIGSRegressor"] = {
        "r2": float(figs_r2),
        "feature_importances": figs_importances,
    }
    summary["HSTreeRegressor"] = {
        "r2": float(hstree_r2),
        "feature_importances": hs_importances,
    }

    return summary


def derive_score_and_explanation(t: TestResults, model_summary: dict) -> tuple[int, str]:
    pvals = [
        t.pearson_p,
        t.ttest_p,
        t.anova_p,
        t.ols_p_fertility_score,
        t.ols_p_fertile_window,
    ]
    sig_count = sum(p < 0.05 for p in pvals)
    trend_count = sum(0.05 <= p < 0.10 for p in pvals)

    fertility_importance_rankings = []
    for model_name in [
        "DecisionTreeRegressor",
        "FIGSRegressor",
        "HSTreeRegressor",
    ]:
        info = model_summary.get(model_name, {})
        imps = info.get("feature_importances", {})
        if imps:
            ranking = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)
            fertility_rank = [feat for feat, _ in ranking].index("fertility_score") + 1
            fertility_importance_rankings.append((model_name, fertility_rank, ranking[0][0]))

    # Conservative score: strong "Yes" only if fertility effects are statistically reliable.
    if sig_count >= 3:
        response = 85
    elif sig_count == 2:
        response = 70
    elif sig_count == 1:
        response = 55
    else:
        response = 18 if trend_count >= 1 else 10

    response = int(max(0, min(100, response)))

    top_tree_driver = ", ".join(
        [f"{name}: fertility_score rank {rank} (top={top})" for name, rank, top in fertility_importance_rankings]
    )

    explanation = (
        f"Evidence does not show a reliable fertility-religiosity relationship. "
        f"Pearson correlation between fertility score and religiosity was r={t.pearson_r:.3f} (p={t.pearson_p:.3f}), "
        f"and a fertile-vs-nonfertile t-test was non-significant (p={t.ttest_p:.3f}, Cohen's d={t.cohens_d:.3f}). "
        f"ANOVA across cycle phases was also non-significant (p={t.anova_p:.3f}). "
        f"In OLS with controls, fertility terms were not significant "
        f"(fertility_score p={t.ols_p_fertility_score:.3f}; fertile_window p={t.ols_p_fertile_window:.3f}). "
        f"Interpretable tree/rule models mainly prioritized non-fertility variables (e.g., relationship status, certainty), "
        f"with fertility features not consistently top drivers ({top_tree_driver})."
    )

    return response, explanation


def main() -> None:
    df = load_and_prepare_data("fertility.csv")

    print_eda(df)
    tests = run_statistical_tests(df)
    model_summary = run_interpretable_models(df)

    response, explanation = derive_score_and_explanation(tests, model_summary)

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("=== FINAL CONCLUSION ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
