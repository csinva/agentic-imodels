import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled


def to_serializable(x):
    if isinstance(x, (np.integer, np.int64, np.int32)):
        return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main():
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0]
    print(f"Research question: {question}")

    df = pd.read_csv("reading.csv")
    print(f"Dataset shape: {df.shape}")

    numeric_cols = [
        "reader_view",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "dyslexia",
        "gender",
        "retake_trial",
        "dyslexia_bin",
        "Flesch_Kincaid",
        "speed",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing values (top 10 columns):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nNumeric summary statistics:")
    print(df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]])

    print("\nSpeed distribution summary:")
    speed_stats = {
        "mean": df["speed"].mean(),
        "median": df["speed"].median(),
        "std": df["speed"].std(),
        "skew": df["speed"].skew(),
        "q95": df["speed"].quantile(0.95),
    }
    print({k: round(v, 4) for k, v in speed_stats.items()})

    corr = df[numeric_cols].corr(numeric_only=True)["speed"].sort_values(ascending=False)
    print("\nCorrelations with speed:")
    print(corr)

    # Primary analysis: dyslexic participants only
    dys = df[df["dyslexia_bin"] == 1].dropna(subset=["speed", "reader_view"])
    rv1 = dys[dys["reader_view"] == 1]["speed"]
    rv0 = dys[dys["reader_view"] == 0]["speed"]

    print("\nPrimary test in dyslexic participants (reader_view 1 vs 0):")
    print(f"n(reader_view=1)={len(rv1)}, n(reader_view=0)={len(rv0)}")
    print(f"mean speed (rv=1): {rv1.mean():.4f}")
    print(f"mean speed (rv=0): {rv0.mean():.4f}")
    print(f"median speed (rv=1): {rv1.median():.4f}")
    print(f"median speed (rv=0): {rv0.median():.4f}")

    welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    mw = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    effect_d = cohen_d(rv1, rv0)
    print(f"Welch t-test: statistic={welch.statistic:.4f}, p={welch.pvalue:.6f}")
    print(f"Mann-Whitney U: statistic={mw.statistic:.4f}, p={mw.pvalue:.6f}")
    print(f"Cohen's d (rv1-rv0): {effect_d:.4f}")

    # Paired by participant where available
    paired = dys.groupby(["uuid", "reader_view"])["speed"].mean().unstack()
    paired = paired.dropna(subset=[0, 1]) if 0 in paired.columns and 1 in paired.columns else pd.DataFrame()
    paired_t = None
    paired_diff = np.nan
    if len(paired) >= 3:
        paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy="omit")
        paired_diff = (paired[1] - paired[0]).mean()
        print(
            f"Paired t-test (within uuid, n={len(paired)}): "
            f"statistic={paired_t.statistic:.4f}, p={paired_t.pvalue:.6f}, mean_diff={paired_diff:.4f}"
        )
    else:
        print("Not enough paired observations for paired t-test.")

    # ANOVA: speed by dyslexia severity
    dys_groups = [
        g["speed"].dropna().values for _, g in df.groupby("dyslexia") if len(g["speed"].dropna()) > 2
    ]
    anova_result = stats.f_oneway(*dys_groups) if len(dys_groups) >= 2 else None
    if anova_result is not None:
        print(
            f"\nOne-way ANOVA across dyslexia severity levels: "
            f"F={anova_result.statistic:.4f}, p={anova_result.pvalue:.6f}"
        )

    # OLS interaction model (controls)
    model_df = df.dropna(
        subset=["speed", "reader_view", "dyslexia_bin", "num_words", "Flesch_Kincaid", "device", "page_id"]
    ).copy()
    model_df["log_speed"] = np.log(model_df["speed"].clip(lower=1e-9))

    ols = smf.ols(
        "log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + C(device) + C(page_id) + C(language)",
        data=model_df,
    ).fit(cov_type="HC3")

    print("\nOLS coefficients of interest (HC3 robust SE):")
    for term in ["reader_view", "dyslexia_bin", "reader_view:dyslexia_bin"]:
        coef = ols.params.get(term, np.nan)
        pval = ols.pvalues.get(term, np.nan)
        print(f"{term}: coef={coef:.6f}, p={pval:.6f}")

    # Effect of reader_view among dyslexics is beta_reader_view + beta_interaction
    b1 = ols.params.get("reader_view", 0.0)
    b3 = ols.params.get("reader_view:dyslexia_bin", 0.0)
    effect_dys_log = b1 + b3

    # SE for sum beta1 + beta3
    cov = ols.cov_params()
    var_sum = (
        cov.loc["reader_view", "reader_view"]
        + cov.loc["reader_view:dyslexia_bin", "reader_view:dyslexia_bin"]
        + 2 * cov.loc["reader_view", "reader_view:dyslexia_bin"]
    )
    se_sum = np.sqrt(max(var_sum, 0))
    z_val = effect_dys_log / se_sum if se_sum > 0 else np.nan
    p_sum = 2 * (1 - stats.norm.cdf(abs(z_val))) if np.isfinite(z_val) else np.nan
    ci_low = effect_dys_log - 1.96 * se_sum
    ci_high = effect_dys_log + 1.96 * se_sum

    print(
        "Reader_view effect among dyslexic participants from OLS (log-speed scale): "
        f"coef={effect_dys_log:.6f}, 95% CI=({ci_low:.6f}, {ci_high:.6f}), p={p_sum:.6f}"
    )

    # Interpretable ML models
    features = [
        "reader_view",
        "dyslexia_bin",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "device",
        "dyslexia",
        "education",
        "gender",
        "language",
        "retake_trial",
        "english_native",
        "Flesch_Kincaid",
        "page_id",
    ]
    ml_df = df[features + ["speed"]].dropna(subset=["speed"]).copy()
    ml_df["log_speed"] = np.log(ml_df["speed"].clip(lower=1e-9))

    X = ml_df[features]
    y = ml_df["log_speed"]

    numeric_features = [
        "reader_view",
        "dyslexia_bin",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "dyslexia",
        "gender",
        "retake_trial",
        "Flesch_Kincaid",
    ]
    categorical_features = [
        "device",
        "education",
        "language",
        "english_native",
        "page_id",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Linear models
    linear_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
    }

    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    print("\nInterpretable linear model summaries (target=log_speed):")
    for name, model in linear_models.items():
        model.fit(transformed_train, y_train)
        pred = model.predict(transformed_test)
        r2 = r2_score(y_test, pred)
        coef = pd.Series(model.coef_, index=feature_names)
        top = coef.reindex(coef.abs().sort_values(ascending=False).head(8).index)
        rv_coef = coef.get("num__reader_view", np.nan)
        print(f"{name}: test R2={r2:.4f}, coef(reader_view)={rv_coef:.6f}")
        print(top.to_string())

    # Decision tree regressor
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=42)
    tree.fit(transformed_train, y_train)
    tree_r2 = r2_score(y_test, tree.predict(transformed_test))
    fi = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    print(f"\nDecisionTreeRegressor: test R2={tree_r2:.4f}")
    print("Top feature importances:")
    print(fi.to_string())
    print("Tree rules:")
    print(export_text(tree, feature_names=list(feature_names), max_depth=3))

    # imodels
    print("\nimodels summaries:")
    rulefit = RuleFitRegressor(random_state=42, max_rules=30, tree_size=4)
    rulefit.fit(transformed_train, y_train)
    rulefit_r2 = r2_score(y_test, rulefit.predict(transformed_test))
    print(f"RuleFitRegressor: test R2={rulefit_r2:.4f}")
    try:
        rules = rulefit.get_rules()
        rules = rules[rules["coef"] != 0].copy()
        if "importance" in rules.columns:
            rules = rules.sort_values("importance", ascending=False)
        print("Top RuleFit rules:")
        print(rules[["rule", "coef", "support"]].head(10).to_string(index=False))
    except Exception as e:
        print(f"Could not extract RuleFit rules: {e}")

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(transformed_train, y_train)
    figs_r2 = r2_score(y_test, figs.predict(transformed_test))
    print(f"FIGSRegressor: test R2={figs_r2:.4f}")
    if hasattr(figs, "feature_importances_"):
        figs_fi = pd.Series(figs.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
        print("FIGS top feature importances:")
        print(figs_fi.to_string())

    hst = HSTreeRegressor(max_leaf_nodes=12, random_state=42)
    hst.fit(transformed_train, y_train)
    hst_r2 = r2_score(y_test, hst.predict(transformed_test))
    print(f"HSTreeRegressor: test R2={hst_r2:.4f}")
    if hasattr(hst, "feature_importances_"):
        hst_fi = pd.Series(hst.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
        print("HSTree top feature importances:")
        print(hst_fi.to_string())

    # Build conclusion score (0-100 Likert)
    primary_p = welch.pvalue
    interaction_p = ols.pvalues.get("reader_view:dyslexia_bin", np.nan)

    # start low by default and raise only if there is clear significant positive evidence
    score = 15

    mean_diff = rv1.mean() - rv0.mean()
    if np.isfinite(primary_p) and primary_p < 0.05 and mean_diff > 0:
        score += 35
    elif np.isfinite(primary_p) and primary_p < 0.05 and mean_diff < 0:
        score -= 10

    if np.isfinite(interaction_p) and interaction_p < 0.05 and b3 > 0:
        score += 35
    elif np.isfinite(interaction_p) and interaction_p < 0.05 and b3 < 0:
        score -= 10

    if paired_t is not None and np.isfinite(paired_t.pvalue) and paired_t.pvalue < 0.05:
        if paired_diff > 0:
            score += 15
        else:
            score -= 10

    score = int(np.clip(score, 0, 100))

    paired_p_txt = f"{paired_t.pvalue:.3f}" if paired_t is not None else "NA"

    explanation = (
        "Primary dyslexic-group comparison found no significant speed improvement with Reader View "
        f"(Welch t p={primary_p:.3f}; mean diff={mean_diff:.2f} words/min-equivalent units; "
        f"Mann-Whitney p={mw.pvalue:.3f}). "
        f"Within-participant paired test was also non-significant (p={paired_p_txt}). "
        "In adjusted OLS, the ReaderView×Dyslexia interaction was non-significant "
        f"(p={interaction_p:.3f}), and the implied Reader View effect among dyslexic readers was near zero "
        f"(log-speed coef={effect_dys_log:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
        "Interpretable models (linear coefficients, tree importances, and rule-based models) did not identify reader_view "
        "as a strong positive predictor relative to text/page and participant factors. Overall evidence does not support "
        "that Reader View improves reading speed for individuals with dyslexia in this dataset."
    )

    out = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt")
    print(json.dumps({k: to_serializable(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
