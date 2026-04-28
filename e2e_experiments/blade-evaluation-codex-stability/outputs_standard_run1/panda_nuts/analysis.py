import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def _find_param_key(param_index, token):
    for key in param_index:
        if token in key:
            return key
    return None


def evidence_from_p(p_value):
    if pd.isna(p_value):
        return 0.2
    if p_value < 0.001:
        return 0.95
    if p_value < 0.01:
        return 0.88
    if p_value < 0.05:
        return 0.72
    if p_value < 0.10:
        return 0.45
    return 0.2


def main():
    info_path = Path("info.json")
    data_path = Path("panda_nuts.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    # Basic cleanup for categorical consistency
    for col in ["sex", "help", "hammer"]:
        df[col] = df[col].astype(str).str.strip()

    df["sex"] = df["sex"].str.lower().replace({"female": "f", "male": "m"})
    df["help"] = df["help"].replace({"Y": "y", "yes": "y", "Yes": "y", "n": "N", "no": "N", "No": "N"})

    # Response variable: nut-cracking efficiency (nuts opened per second)
    df["efficiency"] = np.where(df["seconds"] > 0, df["nuts_opened"] / df["seconds"], np.nan)
    df = df.dropna(subset=["efficiency", "age", "sex", "help", "hammer", "chimpanzee"]).copy()

    # Set category order so statsmodels parameter names are stable
    if set(df["sex"].unique()) >= {"f", "m"}:
        df["sex"] = pd.Categorical(df["sex"], categories=["f", "m"])
    if set(df["help"].unique()) >= {"N", "y"}:
        df["help"] = pd.Categorical(df["help"], categories=["N", "y"])

    print("Research question:", research_question)
    print("\nData shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)

    numeric_cols = ["age", "nuts_opened", "seconds", "efficiency"]
    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\nCategory counts:")
    for col in ["sex", "help", "hammer"]:
        print(f"{col}:\n{df[col].value_counts(dropna=False)}\n")

    print("\nCorrelation matrix (numeric):")
    print(df[numeric_cols].corr())

    print("\nMean efficiency by group:")
    print("sex:\n", df.groupby("sex")["efficiency"].agg(["mean", "std", "count"]))
    print("help:\n", df.groupby("help")["efficiency"].agg(["mean", "std", "count"]))

    # Statistical tests
    pearson_r, pearson_p = stats.pearsonr(df["age"], df["efficiency"])
    spearman_r, spearman_p = stats.spearmanr(df["age"], df["efficiency"])

    eff_f = df.loc[df["sex"] == "f", "efficiency"]
    eff_m = df.loc[df["sex"] == "m", "efficiency"]
    sex_t = stats.ttest_ind(eff_f, eff_m, equal_var=False)
    sex_u = stats.mannwhitneyu(eff_f, eff_m, alternative="two-sided")

    eff_help_no = df.loc[df["help"] == "N", "efficiency"]
    eff_help_yes = df.loc[df["help"] == "y", "efficiency"]
    help_t = stats.ttest_ind(eff_help_no, eff_help_yes, equal_var=False)
    help_u = stats.mannwhitneyu(eff_help_no, eff_help_yes, alternative="two-sided")

    sex_anova = stats.f_oneway(eff_f, eff_m)
    help_anova = stats.f_oneway(eff_help_no, eff_help_yes)

    print("\nStatistical tests:")
    print(f"Age vs efficiency Pearson r={pearson_r:.3f}, p={pearson_p:.3g}")
    print(f"Age vs efficiency Spearman rho={spearman_r:.3f}, p={spearman_p:.3g}")
    print(f"Sex Welch t-test statistic={sex_t.statistic:.3f}, p={sex_t.pvalue:.3g}")
    print(f"Sex Mann-Whitney U statistic={sex_u.statistic:.3f}, p={sex_u.pvalue:.3g}")
    print(f"Help Welch t-test statistic={help_t.statistic:.3f}, p={help_t.pvalue:.3g}")
    print(f"Help Mann-Whitney U statistic={help_u.statistic:.3f}, p={help_u.pvalue:.3g}")
    print(f"Sex ANOVA F={sex_anova.statistic:.3f}, p={sex_anova.pvalue:.3g}")
    print(f"Help ANOVA F={help_anova.statistic:.3f}, p={help_anova.pvalue:.3g}")

    # OLS models (with and without clustered SE by chimpanzee)
    formula = "efficiency ~ age + C(sex) + C(help) + C(hammer)"
    ols_model = smf.ols(formula, data=df).fit()
    ols_cluster = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

    print("\nOLS summary (standard SE):")
    print(ols_model.summary())
    print("\nOLS summary (clustered SE by chimpanzee):")
    print(ols_cluster.summary())

    print("\nANOVA table (Type II):")
    print(sm.stats.anova_lm(ols_model, typ=2))

    # Interpretable sklearn models
    predictors = ["age", "sex", "help", "hammer"]
    X = df[predictors]
    y = df["efficiency"]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", ["age"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["sex", "help", "hammer"]),
        ]
    )

    sklearn_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
    }

    print("\nInterpretable sklearn regression models:")
    for model_name, model in sklearn_models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        r2 = r2_score(y, preds)
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        coefficients = pipe.named_steps["model"].coef_
        top_effects = sorted(zip(feature_names, coefficients), key=lambda z: abs(z[1]), reverse=True)[:6]
        print(f"{model_name} R^2={r2:.3f}, top coefficients={top_effects}")

    tree = Pipeline([("prep", preprocessor), ("model", DecisionTreeRegressor(max_depth=3, random_state=0))])
    tree.fit(X, y)
    tree_r2 = r2_score(y, tree.predict(X))
    tree_features = tree.named_steps["prep"].get_feature_names_out()
    tree_imps = tree.named_steps["model"].feature_importances_
    top_tree = sorted(zip(tree_features, tree_imps), key=lambda z: z[1], reverse=True)
    print(f"DecisionTreeRegressor R^2={tree_r2:.3f}, feature importances={top_tree}")

    # iModels interpretable models
    print("\niModels interpretable models:")
    try:
        from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

        X_enc = preprocessor.fit_transform(X)
        enc_feature_names = list(preprocessor.get_feature_names_out())

        rulefit = RuleFitRegressor(random_state=0, max_rules=40)
        rulefit.fit(X_enc, y, feature_names=enc_feature_names)
        rulefit_r2 = rulefit.score(X_enc, y)

        rules_df = None
        if hasattr(rulefit, "get_rules"):
            rules_df = rulefit.get_rules()
        elif hasattr(rulefit, "_get_rules"):
            rules_df = rulefit._get_rules()
        elif hasattr(rulefit, "rules_"):
            rules_df = rulefit.rules_

        print(f"RuleFitRegressor R^2={rulefit_r2:.3f}")
        if isinstance(rules_df, pd.DataFrame) and "importance" in rules_df.columns:
            show_cols = [c for c in ["rule", "type", "coef", "support", "importance"] if c in rules_df.columns]
            top_rules = rules_df.sort_values("importance", ascending=False).head(8)
            print("Top RuleFit terms:")
            print(top_rules[show_cols].to_string(index=False))

        figs = FIGSRegressor(random_state=0, max_rules=8)
        figs.fit(X_enc, y, feature_names=enc_feature_names)
        figs_r2 = figs.score(X_enc, y)
        print(f"FIGSRegressor R^2={figs_r2:.3f}")
        if hasattr(figs, "feature_importances_"):
            figs_imp = sorted(zip(enc_feature_names, figs.feature_importances_), key=lambda z: z[1], reverse=True)
            print("FIGS feature importances:", figs_imp)

        hstree = HSTreeRegressor(max_leaf_nodes=8)
        hstree.fit(X_enc, y, feature_names=enc_feature_names)
        hstree_r2 = hstree.score(X_enc, y)
        print(f"HSTreeRegressor R^2={hstree_r2:.3f}")

        hs_imp = None
        if hasattr(hstree, "feature_importances_"):
            hs_imp = hstree.feature_importances_
        elif hasattr(hstree, "estimator_") and hasattr(hstree.estimator_, "feature_importances_"):
            hs_imp = hstree.estimator_.feature_importances_
        if hs_imp is not None:
            hstree_imp = sorted(zip(enc_feature_names, hs_imp), key=lambda z: z[1], reverse=True)
            print("HSTree feature importances:", hstree_imp)

    except Exception as exc:
        print("iModels section skipped due to error:", repr(exc))

    # Build final Likert response using clustered OLS p-values for core variables
    age_key = _find_param_key(ols_cluster.params.index, "age")
    sex_key = _find_param_key(ols_cluster.params.index, "C(sex)")
    help_key = _find_param_key(ols_cluster.params.index, "C(help)")

    age_p = ols_cluster.pvalues.get(age_key, np.nan)
    sex_p = ols_cluster.pvalues.get(sex_key, np.nan)
    help_p = ols_cluster.pvalues.get(help_key, np.nan)

    age_coef = ols_cluster.params.get(age_key, np.nan)
    sex_coef = ols_cluster.params.get(sex_key, np.nan)
    help_coef = ols_cluster.params.get(help_key, np.nan)

    age_evidence = evidence_from_p(age_p)
    sex_evidence = evidence_from_p(sex_p)
    help_evidence = evidence_from_p(help_p)

    response = int(np.clip(round(100 * (age_evidence + sex_evidence + help_evidence) / 3.0), 0, 100))

    explanation = (
        f"Using efficiency = nuts_opened/seconds, clustered OLS (controlling for hammer type) found age positively "
        f"associated with efficiency (coef={age_coef:.3f}, p={age_p:.3g}) and males higher than females "
        f"(coef={sex_coef:.3f}, p={sex_p:.3g}). Help showed a negative but weaker/non-robust association "
        f"(coef={help_coef:.3f}, p={help_p:.3g}). Correlation and group tests agree for age and sex, while help is "
        f"mixed; interpretable sklearn/imodels models also prioritize age/sex over help."
    )

    result = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))

    print("\nFinal result written to conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
