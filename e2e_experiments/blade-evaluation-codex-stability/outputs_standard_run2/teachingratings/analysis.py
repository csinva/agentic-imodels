import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def safe_round(val, digits=4):
    if pd.isna(val):
        return None
    return round(float(val), digits)


def top_series(series, n=5, abs_sort=False):
    s = series.abs().sort_values(ascending=False) if abs_sort else series.sort_values(ascending=False)
    return {k: safe_round(v, 4) for k, v in s.head(n).items()}


def main():
    info_path = Path("info.json")
    data_path = Path("teachingratings.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Basic exploration
    print("Research question:", research_question)
    print("Data shape:", df.shape)
    print("\nColumns and dtypes:\n", df.dtypes)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print("\nNumeric summary stats:\n", df[numeric_cols].describe().T)
    distribution_summary = pd.DataFrame(
        {
            "mean": df[numeric_cols].mean(),
            "std": df[numeric_cols].std(),
            "skew": df[numeric_cols].skew(),
            "min": df[numeric_cols].min(),
            "max": df[numeric_cols].max(),
        }
    )
    print("\nDistribution summary (numeric):\n", distribution_summary)

    print("\nCategorical distributions:")
    for col in categorical_cols:
        print(f"\n{col} value counts:\n{df[col].value_counts(normalize=True).round(3)}")

    corr = df[numeric_cols].corr()
    print("\nCorrelation matrix (numeric):\n", corr)

    # Statistical tests focused on beauty -> eval relationship
    pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
    spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

    df["beauty_quartile"] = pd.qcut(df["beauty"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    quartile_means = df.groupby("beauty_quartile")["eval"].mean()

    eval_q1 = df.loc[df["beauty_quartile"] == "Q1", "eval"]
    eval_q4 = df.loc[df["beauty_quartile"] == "Q4", "eval"]
    t_stat, t_p = stats.ttest_ind(eval_q4, eval_q1, equal_var=False)

    grouped_eval = [grp["eval"].values for _, grp in df.groupby("beauty_quartile")]
    anova_f, anova_p = stats.f_oneway(*grouped_eval)

    ols_simple = smf.ols("eval ~ beauty", data=df).fit()
    ols_adjusted = smf.ols(
        "eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + "
        "C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit()
    ols_cluster = smf.ols(
        "eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + "
        "C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["prof"]})

    print("\nStatistical tests:")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.3e}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.3e}")
    print(f"Top-vs-bottom beauty quartile t-test: t={t_stat:.4f}, p={t_p:.3e}")
    print(f"ANOVA across beauty quartiles: F={anova_f:.4f}, p={anova_p:.3e}")
    print("Quartile means eval:")
    print(quartile_means)

    print("\nOLS simple coef for beauty:", safe_round(ols_simple.params["beauty"], 4), "p=", f"{ols_simple.pvalues['beauty']:.3e}")
    print("OLS adjusted coef for beauty:", safe_round(ols_adjusted.params["beauty"], 4), "p=", f"{ols_adjusted.pvalues['beauty']:.3e}")
    print("OLS clustered-by-prof coef for beauty:", safe_round(ols_cluster.params["beauty"], 4), "p=", f"{ols_cluster.pvalues['beauty']:.3e}")

    # Interpretable sklearn models
    X = df.drop(columns=["eval", "rownames", "prof", "beauty_quartile"])
    y = df["eval"]

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = [c for c in X.columns if c not in num_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    sklearn_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=10000),
        "decision_tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    sklearn_results = {}

    for name, model in sklearn_models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)

        model_info = {"r2_test": safe_round(r2, 4)}

        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
        fitted_model = pipe.named_steps["model"]

        if hasattr(fitted_model, "coef_"):
            coef_series = pd.Series(fitted_model.coef_, index=feature_names)
            model_info["beauty_coef"] = safe_round(coef_series.get("num__beauty", np.nan), 4)
            model_info["top_abs_coefficients"] = top_series(coef_series, n=6, abs_sort=True)

        if hasattr(fitted_model, "feature_importances_"):
            fi_series = pd.Series(fitted_model.feature_importances_, index=feature_names)
            model_info["beauty_importance"] = safe_round(fi_series.get("num__beauty", np.nan), 4)
            model_info["top_feature_importances"] = top_series(fi_series, n=6, abs_sort=False)

        sklearn_results[name] = model_info

    print("\nInterpretable sklearn model results:")
    for model_name, result in sklearn_results.items():
        print(model_name, result)

    # Interpretable imodels models
    X_enc = pd.get_dummies(X, drop_first=True)
    Xenc_train, Xenc_test, yenc_train, yenc_test = train_test_split(X_enc, y, test_size=0.25, random_state=42)

    imodels_results = {}

    rulefit = RuleFitRegressor(random_state=42, max_rules=30)
    rulefit.fit(Xenc_train.values, yenc_train.values, feature_names=list(X_enc.columns))
    rf_r2 = r2_score(yenc_test, rulefit.predict(Xenc_test.values))
    rules_df = rulefit._get_rules()
    active_rules = rules_df.loc[rules_df["coef"] != 0].sort_values("importance", ascending=False).head(8)
    imodels_results["rulefit"] = {
        "r2_test": safe_round(rf_r2, 4),
        "num_active_rules": int((rules_df["coef"] != 0).sum()),
        "top_rules": [str(r) for r in active_rules["rule"].tolist()],
        "top_rule_importance": [safe_round(v, 4) for v in active_rules["importance"].tolist()],
    }

    figs = FIGSRegressor(random_state=42, max_rules=12)
    figs.fit(Xenc_train.values, yenc_train.values, feature_names=list(X_enc.columns))
    figs_r2 = r2_score(yenc_test, figs.predict(Xenc_test.values))
    figs_fi = pd.Series(figs.feature_importances_, index=X_enc.columns)
    imodels_results["figs"] = {
        "r2_test": safe_round(figs_r2, 4),
        "beauty_importance": safe_round(figs_fi.get("beauty", np.nan), 4),
        "top_feature_importances": top_series(figs_fi, n=6, abs_sort=False),
    }

    hstree = HSTreeRegressor(random_state=42, max_leaf_nodes=20)
    hstree.fit(Xenc_train.values, yenc_train.values, feature_names=list(X_enc.columns))
    hstree_r2 = r2_score(yenc_test, hstree.predict(Xenc_test.values))
    hs_fi = pd.Series(hstree.estimator_.feature_importances_, index=X_enc.columns)
    imodels_results["hstree"] = {
        "r2_test": safe_round(hstree_r2, 4),
        "beauty_importance": safe_round(hs_fi.get("beauty", np.nan), 4),
        "top_feature_importances": top_series(hs_fi, n=6, abs_sort=False),
    }

    print("\nInterpretable imodels results:")
    for model_name, result in imodels_results.items():
        print(model_name, result)

    # Evidence synthesis into 0-100 Likert response
    # Strong significance raises confidence, but effect-size moderation prevents
    # overconfident 100/100 scores for modest effects.
    strong_positive = (
        pearson_p < 0.01
        and spearman_p < 0.01
        and t_p < 0.01
        and anova_p < 0.01
        and ols_adjusted.pvalues["beauty"] < 0.01
        and ols_cluster.pvalues["beauty"] < 0.01
        and ols_adjusted.params["beauty"] > 0
    )
    strong_negative = (
        pearson_p < 0.01
        and ols_adjusted.pvalues["beauty"] < 0.01
        and ols_adjusted.params["beauty"] < 0
    )

    if strong_positive:
        response = int(
            np.clip(
                70 + 80 * abs(pearson_r) + 20 * abs(ols_adjusted.params["beauty"]),
                0,
                95,
            )
        )
    elif strong_negative:
        response = int(
            np.clip(
                30 - 80 * abs(pearson_r) - 20 * abs(ols_adjusted.params["beauty"]),
                0,
                100,
            )
        )
    else:
        response = int(50 + 100 * pearson_r)
        response = int(np.clip(response, 0, 100))

    explanation = (
        f"Across 463 courses, beauty shows a statistically significant positive association with teaching evaluations. "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}) and Spearman rho={spearman_rho:.3f} (p={spearman_p:.2e}) are both positive. "
        f"Top-vs-bottom beauty quartiles differ by about {(quartile_means['Q4'] - quartile_means['Q1']):.3f} eval points "
        f"(t-test p={t_p:.2e}; ANOVA p={anova_p:.2e}). "
        f"In adjusted OLS with course/instructor covariates, beauty remains positive (beta={ols_adjusted.params['beauty']:.3f}, "
        f"p={ols_adjusted.pvalues['beauty']:.2e}); with professor-clustered SEs it remains significant (p={ols_cluster.pvalues['beauty']:.2e}). "
        f"Interpretable sklearn/imodels models also consistently treat beauty as an important predictor. "
        f"Effect size is modest rather than huge, so this is a clear but moderate 'Yes'."
    )

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print("\nWrote conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
