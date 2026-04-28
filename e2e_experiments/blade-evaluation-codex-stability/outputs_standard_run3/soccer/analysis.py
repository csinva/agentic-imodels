import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def extract_skin_coef_from_pipeline(model, feature_name="skin_tone"):
    """Get coefficient for skin_tone from linear pipeline with preprocessing."""
    try:
        prep = model.named_steps["prep"]
        reg = model.named_steps["model"]
        feat_names = prep.get_feature_names_out()
        idx = np.where(feat_names == f"num__{feature_name}")[0]
        if len(idx) == 0:
            return np.nan
        return float(reg.coef_[idx[0]])
    except Exception:
        return np.nan


def extract_skin_importance_from_tree_pipeline(model, feature_name="skin_tone"):
    try:
        prep = model.named_steps["prep"]
        tree = model.named_steps["model"]
        feat_names = prep.get_feature_names_out()
        idx = np.where(feat_names == f"num__{feature_name}")[0]
        if len(idx) == 0:
            return np.nan
        return float(tree.feature_importances_[idx[0]])
    except Exception:
        return np.nan


def main():
    with open("info.json", "r") as f:
        info = json.load(f)
    question = info["research_questions"][0]

    df = pd.read_csv("soccer.csv")

    # Core engineered variables for analysis
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df["red_card_rate"] = df["redCards"] / df["games"].replace(0, np.nan)
    df["red_card_any"] = (df["redCards"] > 0).astype(int)
    df["position"] = df["position"].fillna("Unknown")

    analysis_cols = [
        "skin_tone",
        "redCards",
        "red_card_rate",
        "red_card_any",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "seIAT",
        "seExp",
        "leagueCountry",
        "position",
    ]

    dfa = df[analysis_cols].dropna(subset=["skin_tone", "red_card_rate"]).copy()

    print("Research question:", question)
    print("\nData shape (full):", df.shape)
    print("Data shape (analysis subset):", dfa.shape)

    # -------------------------
    # 1) EDA
    # -------------------------
    numeric_cols = dfa.select_dtypes(include=[np.number]).columns.tolist()
    print("\nSummary statistics (numeric):")
    print(dfa[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nDistribution of skin_tone:")
    print(dfa["skin_tone"].value_counts(normalize=True).sort_index())

    print("\nDistribution of redCards:")
    print(dfa["redCards"].value_counts(normalize=True).sort_index())

    corr_with_red_rate = (
        dfa[numeric_cols]
        .corr(numeric_only=True)["red_card_rate"]
        .sort_values(ascending=False)
    )
    print("\nCorrelations with red_card_rate:")
    print(corr_with_red_rate)

    # -------------------------
    # 2) Statistical tests
    # -------------------------
    dfa["skin_group"] = np.where(dfa["skin_tone"] >= 0.5, "dark", "light")
    dark = dfa.loc[dfa["skin_group"] == "dark", "red_card_rate"]
    light = dfa.loc[dfa["skin_group"] == "light", "red_card_rate"]

    t_stat, t_p = stats.ttest_ind(dark, light, equal_var=False, nan_policy="omit")

    # ANOVA across 5 ordinal buckets
    bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["very_light", "light", "medium", "dark", "very_dark"]
    dfa["skin_5cat"] = pd.cut(dfa["skin_tone"], bins=bins, labels=labels, include_lowest=True)
    groups = [
        dfa.loc[dfa["skin_5cat"] == lab, "red_card_rate"].dropna().values
        for lab in labels
    ]
    groups = [g for g in groups if len(g) > 1]
    f_stat, anova_p = stats.f_oneway(*groups)

    pearson_r, pearson_p = stats.pearsonr(dfa["skin_tone"], dfa["red_card_rate"])

    print("\nT-test (dark>=0.5 vs light<0.5) on red_card_rate:")
    print(f"dark_mean={dark.mean():.6f}, light_mean={light.mean():.6f}, t={t_stat:.4f}, p={t_p:.4g}")

    print("\nANOVA on red_card_rate across skin_tone categories:")
    print(f"F={f_stat:.4f}, p={anova_p:.4g}")

    print("\nPearson correlation (skin_tone vs red_card_rate):")
    print(f"r={pearson_r:.4f}, p={pearson_p:.4g}")

    # OLS with controls (interpretable coefficient + p-value)
    ols_cols = [
        "red_card_rate",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "leagueCountry",
        "position",
    ]
    ols_df = dfa[ols_cols].dropna().copy()

    X_ols = pd.get_dummies(
        ols_df.drop(columns=["red_card_rate"]),
        columns=["leagueCountry", "position"],
        drop_first=True,
    )
    X_ols = sm.add_constant(X_ols)
    X_ols = X_ols.astype(float)
    y_ols = ols_df["red_card_rate"]

    ols_model = sm.OLS(y_ols, X_ols).fit()
    ols_coef = safe_float(ols_model.params.get("skin_tone", np.nan))
    ols_p = safe_float(ols_model.pvalues.get("skin_tone", np.nan))

    print("\nOLS result for skin_tone (controlling for covariates):")
    print(f"coef={ols_coef:.6f}, p={ols_p:.4g}")

    # -------------------------
    # 3) Interpretable models
    # -------------------------
    model_df = dfa[
        [
            "red_card_rate",
            "red_card_any",
            "skin_tone",
            "games",
            "yellowCards",
            "yellowReds",
            "goals",
            "victories",
            "ties",
            "defeats",
            "height",
            "weight",
            "meanIAT",
            "meanExp",
            "seIAT",
            "seExp",
            "leagueCountry",
            "position",
        ]
    ].copy()

    y_reg = model_df["red_card_rate"]
    y_cls = model_df["red_card_any"]
    X = model_df.drop(columns=["red_card_rate", "red_card_any"])

    num_features = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "seIAT",
        "seExp",
    ]
    cat_features = ["leagueCountry", "position"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_features,
            ),
        ]
    )

    lin = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    ridge = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=0))])
    lasso = Pipeline([("prep", preprocessor), ("model", Lasso(alpha=1e-4, random_state=0, max_iter=10000))])
    dt_reg = Pipeline([("prep", preprocessor), ("model", DecisionTreeRegressor(max_depth=4, random_state=0))])
    dt_cls = Pipeline([("prep", preprocessor), ("model", DecisionTreeClassifier(max_depth=4, random_state=0))])

    lin.fit(X, y_reg)
    ridge.fit(X, y_reg)
    lasso.fit(X, y_reg)
    dt_reg.fit(X, y_reg)
    dt_cls.fit(X, y_cls)

    lin_skin = extract_skin_coef_from_pipeline(lin)
    ridge_skin = extract_skin_coef_from_pipeline(ridge)
    lasso_skin = extract_skin_coef_from_pipeline(lasso)
    dt_reg_skin_imp = extract_skin_importance_from_tree_pipeline(dt_reg)
    dt_cls_skin_imp = extract_skin_importance_from_tree_pipeline(dt_cls)

    print("\nSklearn interpretable model signals for skin_tone:")
    print(f"LinearRegression coef: {lin_skin:.6f}")
    print(f"Ridge coef: {ridge_skin:.6f}")
    print(f"Lasso coef: {lasso_skin:.6f}")
    print(f"DecisionTreeRegressor importance: {dt_reg_skin_imp:.6f}")
    print(f"DecisionTreeClassifier importance: {dt_cls_skin_imp:.6f}")

    # imodels on numeric-only subset for speed and interpretability
    imodel_cols = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "seIAT",
        "seExp",
    ]
    im_df = model_df[imodel_cols + ["red_card_rate"]].dropna().copy()

    # cap size for faster rule/tree fitting while preserving statistical power
    if len(im_df) > 25000:
        im_df = im_df.sample(25000, random_state=0)

    X_im = im_df[imodel_cols]
    y_im = im_df["red_card_rate"]

    rulefit = RuleFitRegressor(random_state=0, max_rules=40, tree_size=4)
    figs = FIGSRegressor(random_state=0, max_rules=20)
    hstree = HSTreeRegressor(random_state=0, max_leaf_nodes=20)

    rulefit.fit(X_im, y_im, feature_names=imodel_cols)
    figs.fit(X_im, y_im, feature_names=imodel_cols)
    hstree.fit(X_im, y_im)

    # RuleFit: sum importance of linear terms/rules involving skin_tone
    skin_rulefit_importance = 0.0
    try:
        rules_df = rulefit.get_rules()
        use_rules = rules_df[(rules_df["coef"] != 0)]
        mask = use_rules["rule"].astype(str).str.contains("skin_tone", case=False, regex=False)
        skin_rulefit_importance = float(use_rules.loc[mask, "importance"].sum())
    except Exception:
        skin_rulefit_importance = np.nan

    # FIGS/HSTree feature importances if available
    skin_figs_importance = np.nan
    if hasattr(figs, "feature_importances_"):
        try:
            figs_imp = figs.feature_importances_
            if len(figs_imp) == len(imodel_cols):
                skin_figs_importance = float(figs_imp[imodel_cols.index("skin_tone")])
        except Exception:
            pass

    skin_hs_importance = np.nan
    if hasattr(hstree, "feature_importances_"):
        try:
            hs_imp = hstree.feature_importances_
            if len(hs_imp) == len(imodel_cols):
                skin_hs_importance = float(hs_imp[imodel_cols.index("skin_tone")])
        except Exception:
            pass

    print("\nimodels signals for skin_tone:")
    print(f"RuleFit importance (rules + terms containing skin_tone): {skin_rulefit_importance:.6f}")
    print(f"FIGS importance: {skin_figs_importance:.6f}")
    print(f"HSTree importance: {skin_hs_importance:.6f}")

    # -------------------------
    # 4) Translate evidence to Likert 0-100
    # -------------------------
    score = 50

    # Primary inferential tests
    if np.isfinite(t_p) and dark.mean() > light.mean():
        score += 15 if t_p < 0.05 else 3
    elif np.isfinite(t_p):
        score -= 15 if t_p < 0.05 else 3

    if np.isfinite(anova_p):
        score += 10 if anova_p < 0.05 else 0

    if np.isfinite(pearson_p) and np.isfinite(pearson_r):
        if pearson_p < 0.05:
            score += 10 if pearson_r > 0 else -10
        else:
            score += 1 if pearson_r > 0 else -1

    # Controlled model is highest-weight evidence
    if np.isfinite(ols_p) and np.isfinite(ols_coef):
        if ols_p < 0.05:
            score += 25 if ols_coef > 0 else -25
        else:
            score += 2 if ols_coef > 0 else -2

    # Secondary consistency checks from interpretable ML models
    linear_signs = [x for x in [lin_skin, ridge_skin, lasso_skin] if np.isfinite(x)]
    if len(linear_signs) > 0:
        frac_pos = np.mean(np.array(linear_signs) > 0)
        if frac_pos >= 2 / 3:
            score += 7
        elif frac_pos <= 1 / 3:
            score -= 7

    importance_evidence = [
        x
        for x in [dt_reg_skin_imp, dt_cls_skin_imp, skin_rulefit_importance, skin_figs_importance, skin_hs_importance]
        if np.isfinite(x)
    ]
    if len(importance_evidence) > 0:
        # if models repeatedly use skin_tone (non-trivial importance), nudge toward Yes
        nontrivial = np.mean(np.array(importance_evidence) > 0.005)
        if nontrivial >= 0.4:
            score += 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Using {len(dfa):,} player-referee observations with rated skin tone, dark-skin players "
        f"(>=0.5) had a red-card rate of {dark.mean():.4f} vs {light.mean():.4f} for light-skin players "
        f"(t-test p={t_p:.3g}). Skin tone was {'positively' if pearson_r >= 0 else 'negatively'} correlated "
        f"with red-card rate (r={pearson_r:.3f}, p={pearson_p:.3g}), and in OLS with controls, the skin-tone "
        f"coefficient was {ols_coef:.5f} (p={ols_p:.3g}). Interpretable sklearn/imodels models generally "
        f"showed a {'positive' if np.mean(np.array(linear_signs) > 0) >= 0.5 else 'non-positive'} skin-tone effect, "
        f"supporting a {'yes' if score >= 60 else 'weak/uncertain'} answer to the research question."
    )

    out = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        f.write(json.dumps(out))

    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
