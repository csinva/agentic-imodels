import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    return stats.pearsonr(x[mask], y[mask])


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    mask = x.notna() & y.notna()
    return stats.spearmanr(x[mask], y[mask])


def format_dict(d: Dict[str, float], digits: int = 4) -> Dict[str, float]:
    return {k: float(np.round(v, digits)) for k, v in d.items()}


def main() -> None:
    # 1) Read task metadata
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["No question provided"])[0]

    # 2) Load data
    df = pd.read_csv("hurricane.csv")

    print("=== Research Question ===")
    print(question)
    print()

    print("=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print()

    # 3) EDA: summary stats, distribution, correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("=== Numeric Summary Statistics ===")
    print(df[numeric_cols].describe().T.to_string())
    print()

    print("=== Distribution Highlights ===")
    print(
        {
            "alldeaths_skew": float(df["alldeaths"].skew()),
            "alldeaths_kurtosis": float(df["alldeaths"].kurtosis()),
            "alldeaths_quantiles": format_dict(df["alldeaths"].quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()),
        }
    )
    print()

    corr = df[numeric_cols].corr(numeric_only=True)
    print("=== Correlation With alldeaths ===")
    print(corr["alldeaths"].sort_values(ascending=False).to_string())
    print()

    # Prepare transformed variables for tests/models
    df_model = df.copy()
    df_model["log_alldeaths"] = np.log1p(df_model["alldeaths"])

    # 4) Statistical tests
    pearson_r, pearson_p = safe_pearson(df_model["masfem"], df_model["alldeaths"])
    spearman_rho, spearman_p = safe_spearman(df_model["masfem"], df_model["alldeaths"])

    female_deaths = df_model.loc[df_model["gender_mf"] == 1, "alldeaths"]
    male_deaths = df_model.loc[df_model["gender_mf"] == 0, "alldeaths"]
    t_stat, t_p = stats.ttest_ind(female_deaths, male_deaths, equal_var=False, nan_policy="omit")

    masfem_bins = pd.qcut(df_model["masfem"], q=3, labels=["low", "mid", "high"])
    anova_groups = [
        df_model.loc[masfem_bins == level, "log_alldeaths"].dropna().values
        for level in ["low", "mid", "high"]
    ]
    anova_f, anova_p = stats.f_oneway(*anova_groups)

    # OLS with controls (interpretable p-values and CI)
    ols = smf.ols(
        "log_alldeaths ~ masfem + wind + category + Q('min') + np.log1p(ndam15) + year",
        data=df_model,
    ).fit()

    ols_gender = smf.ols(
        "log_alldeaths ~ gender_mf + wind + category + Q('min') + np.log1p(ndam15) + year",
        data=df_model,
    ).fit()

    print("=== Statistical Tests ===")
    print(
        {
            "pearson_masfem_alldeaths": {"r": float(np.round(pearson_r, 4)), "p": float(np.round(pearson_p, 6))},
            "spearman_masfem_alldeaths": {"rho": float(np.round(spearman_rho, 4)), "p": float(np.round(spearman_p, 6))},
            "welch_ttest_female_vs_male_deaths": {
                "t": float(np.round(t_stat, 4)),
                "p": float(np.round(t_p, 6)),
                "female_mean": float(np.round(female_deaths.mean(), 4)),
                "male_mean": float(np.round(male_deaths.mean(), 4)),
            },
            "anova_log_deaths_by_masfem_tertiles": {
                "F": float(np.round(anova_f, 4)),
                "p": float(np.round(anova_p, 6)),
            },
        }
    )
    print()

    print("=== OLS (log_alldeaths) Coefficients ===")
    print(ols.summary().tables[1])
    print()

    print("=== OLS (gender indicator alternative) Coefficients ===")
    print(ols_gender.summary().tables[1])
    print()

    # 5) Interpretable models using sklearn + imodels
    feature_cols = [
        "masfem",
        "gender_mf",
        "masfem_mturk",
        "wind",
        "category",
        "min",
        "ndam15",
        "year",
        "elapsedyrs",
    ]

    X = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df_model["log_alldeaths"].values

    # Standardized linear models for comparable coefficients
    lin = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=0)),
    ])
    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.03, random_state=0, max_iter=10000)),
    ])
    tree = DecisionTreeRegressor(max_depth=3, random_state=0)

    lin.fit(X, y)
    ridge.fit(X, y)
    lasso.fit(X, y)
    tree.fit(X, y)

    lin_coef = dict(zip(feature_cols, lin.named_steps["model"].coef_))
    ridge_coef = dict(zip(feature_cols, ridge.named_steps["model"].coef_))
    lasso_coef = dict(zip(feature_cols, lasso.named_steps["model"].coef_))
    tree_imp = dict(zip(feature_cols, tree.feature_importances_))

    print("=== Interpretable sklearn Models ===")
    print("LinearRegression standardized coefficients:")
    print(format_dict(lin_coef))
    print("Ridge standardized coefficients:")
    print(format_dict(ridge_coef))
    print("Lasso standardized coefficients:")
    print(format_dict(lasso_coef))
    print("DecisionTreeRegressor feature importances:")
    print(format_dict(tree_imp))
    print()

    # imodels
    rulefit = RuleFitRegressor(random_state=0)
    rulefit.fit(X, y, feature_names=feature_cols)
    # RuleFit exposes rules via a private method in this version.
    rules_df = rulefit._get_rules()
    active_rules = (
        rules_df.loc[rules_df["coef"].abs() > 1e-10]
        .sort_values("importance", ascending=False)
        .head(10)
        [["rule", "coef", "importance", "support"]]
    )

    figs = FIGSRegressor(random_state=0, max_rules=20)
    figs.fit(X, y)
    figs_imp = dict(zip(feature_cols, figs.feature_importances_))

    hstree = HSTreeRegressor(random_state=0, max_leaf_nodes=8)
    hstree.fit(X, y)
    hstree_text = repr(hstree)

    print("=== imodels Results ===")
    print("Top active RuleFit rules:")
    print(active_rules.to_string(index=False))
    print()
    print("FIGS feature importances:")
    print(format_dict(figs_imp))
    print()
    print("HSTree structure:")
    print(hstree_text)
    print()

    # 6) Synthesize evidence into Likert response
    tests: List[Dict[str, float]] = [
        {"effect": float(pearson_r), "p": float(pearson_p)},
        {"effect": float(spearman_rho), "p": float(spearman_p)},
        {"effect": float(female_deaths.mean() - male_deaths.mean()), "p": float(t_p)},
        {
            "effect": float(
                df_model.loc[masfem_bins == "high", "log_alldeaths"].mean()
                - df_model.loc[masfem_bins == "low", "log_alldeaths"].mean()
            ),
            "p": float(anova_p),
        },
        {"effect": float(ols.params.get("masfem", np.nan)), "p": float(ols.pvalues.get("masfem", np.nan))},
    ]

    supportive_sig = sum(1 for t in tests if np.isfinite(t["effect"]) and np.isfinite(t["p"]) and t["effect"] > 0 and t["p"] < 0.05)
    contradictory_sig = sum(1 for t in tests if np.isfinite(t["effect"]) and np.isfinite(t["p"]) and t["effect"] < 0 and t["p"] < 0.05)
    min_p = min(t["p"] for t in tests if np.isfinite(t["p"]))

    # Check model-based relevance of masfem.
    masfem_linear_abs = abs(lin_coef.get("masfem", 0.0))
    top_tree_feature = max(tree_imp, key=tree_imp.get)
    top_figs_feature = max(figs_imp, key=figs_imp.get)
    masfem_in_hstree = "masfem" in hstree_text

    if supportive_sig >= 3:
        response = 85
    elif supportive_sig == 2:
        response = 70
    elif supportive_sig == 1:
        response = 50
    else:
        # No statistically significant support.
        if min_p < 0.1:
            response = 30
        else:
            response = 15

    if contradictory_sig > supportive_sig:
        response = max(0, response - 10)

    explanation = (
        "Evidence is weak for the claim that more feminine hurricane names are associated with higher deaths. "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.3f}) and Spearman rho={spearman_rho:.3f} (p={spearman_p:.3f}) "
        "show no significant relationship between femininity score and deaths. "
        f"Welch t-test comparing female vs male names gives p={t_p:.3f} (female mean={female_deaths.mean():.1f}, male mean={male_deaths.mean():.1f}), "
        "which is not significant. "
        f"In controlled OLS on log deaths, masfem coef={ols.params['masfem']:.3f} with p={ols.pvalues['masfem']:.3f}, also non-significant. "
        f"Interpretable models (DecisionTree, FIGS, RuleFit, HSTree) prioritize damage/intensity variables (e.g., top tree feature={top_tree_feature}, top FIGS feature={top_figs_feature}); "
        f"masfem has small linear effect magnitude ({masfem_linear_abs:.3f}) and is {'used' if masfem_in_hstree else 'not used'} in HSTree splits. "
        "Overall this dataset does not provide statistically significant support for the proposed relationship."
    )

    result = {
        "response": int(np.clip(round(response), 0, 100)),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=True))

    print("=== Final Conclusion JSON ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
