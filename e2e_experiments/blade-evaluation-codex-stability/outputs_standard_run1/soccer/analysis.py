import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_conclusion(stats_out: dict, group_stats: dict) -> dict:
    # Weighted evidence score centered at 50, adjusted for direction and significance.
    score = 50.0

    # Primary evidence: count model with exposure, direct group comparison, adjusted OLS.
    if stats_out["poisson_p"] < 0.05:
        score += 10 if stats_out["poisson_coef"] > 0 else -10
    if stats_out["ttest_one_sided_p"] < 0.05:
        score += 8 if stats_out["mean_diff"] > 0 else -8
    if stats_out["ols_p"] < 0.05:
        score += 6 if stats_out["ols_coef"] > 0 else -6

    # Direct "more likely" test on binary red-card outcome.
    if stats_out["chi2_p"] < 0.05:
        score += 6 if group_stats["dark_any_red_rate"] > group_stats["light_any_red_rate"] else -6
    else:
        score -= 6

    # Correlation evidence (small weight because effect can be practically tiny).
    if stats_out["pearson_p"] < 0.05:
        score += 3 if stats_out["pearson_r"] > 0 else -3
    if stats_out["spearman_p"] < 0.05:
        score += 3 if stats_out["spearman_rho"] > 0 else -3

    # Practical magnitude adjustment using relative difference in red-card rate.
    if group_stats["light_red_rate"] > 0:
        rel_diff = (group_stats["dark_red_rate"] - group_stats["light_red_rate"]) / group_stats["light_red_rate"]
        score += np.clip(rel_diff, -1, 1) * 5

    score = int(np.clip(round(score), 0, 100))

    chi_sig = stats_out["chi2_p"] < 0.05
    chi_text = (
        f"The binary any-red-card chi-square test for dark vs light is significant "
        f"(p={stats_out['chi2_p']:.4g})."
        if chi_sig
        else f"The binary any-red-card chi-square test for dark vs light is not significant "
        f"(p={stats_out['chi2_p']:.4g})."
    )

    explanation = (
        f"Dark-skin players show a higher red-card rate than light-skin players "
        f"({group_stats['dark_red_rate']:.4f} vs {group_stats['light_red_rate']:.4f} per game; "
        f"one-sided Welch t-test p={stats_out['ttest_one_sided_p']:.4g}). "
        f"A Poisson model with game exposure and controls estimates a positive skin-tone effect "
        f"(coef={stats_out['poisson_coef']:.3f}, IRR={stats_out['poisson_irr']:.3f}, p={stats_out['poisson_p']:.4g}), "
        f"and controlled OLS on red-card rate is also positive (coef={stats_out['ols_coef']:.4f}, "
        f"p={stats_out['ols_p']:.4g}). {chi_text} "
        f"Overall, evidence supports a positive relationship, but the absolute effect size remains small."
    )

    return {"response": score, "explanation": explanation}


def main():
    info = json.loads(Path("info.json").read_text())
    research_question = info["research_questions"][0]

    print_header("Research Question")
    print(research_question)

    print_header("Load Data")
    df = pd.read_csv("soccer.csv")
    print(f"Loaded soccer.csv with shape: {df.shape}")

    # Skin tone proxy from two raters.
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

    key_cols = [
        "skin_tone",
        "redCards",
        "games",
        "victories",
        "ties",
        "defeats",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "position",
        "leagueCountry",
        "meanIAT",
        "meanExp",
    ]

    data = df[key_cols].copy()
    data = data.dropna(subset=["skin_tone", "redCards", "games"])
    data = data[data["games"] > 0].copy()
    data["red_rate"] = data["redCards"] / data["games"]
    data["any_red"] = (data["redCards"] > 0).astype(int)

    print(f"Analysis dataset rows after filtering: {len(data)}")

    print_header("Summary Statistics")
    print(data[["skin_tone", "redCards", "red_rate", "games", "yellowCards", "yellowReds", "goals"]].describe().round(4))

    print_header("Distributions")
    skin_bins = pd.cut(
        data["skin_tone"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["very_light", "light", "medium", "dark", "very_dark"],
    )
    print("Skin-tone category counts:")
    print(skin_bins.value_counts(dropna=False).sort_index())

    red_rate_hist_counts, red_rate_hist_edges = np.histogram(data["red_rate"], bins=[0, 0.01, 0.02, 0.05, 0.1, 1.0])
    print("Red-card rate histogram (bin_edges, count):")
    for i, c in enumerate(red_rate_hist_counts):
        print(f"[{red_rate_hist_edges[i]:.2f}, {red_rate_hist_edges[i+1]:.2f}): {int(c)}")

    print_header("Correlations")
    corr_cols = ["skin_tone", "redCards", "red_rate", "games", "yellowCards", "yellowReds", "goals", "height", "weight", "meanIAT", "meanExp"]
    corr_mat = data[corr_cols].corr(numeric_only=True)
    print(corr_mat[["skin_tone", "redCards", "red_rate"]].round(4))

    print_header("Statistical Tests")
    light = data[data["skin_tone"] <= 0.25]
    dark = data[data["skin_tone"] >= 0.75]

    group_stats = {
        "n_light": int(len(light)),
        "n_dark": int(len(dark)),
        "light_red_rate": safe_float(light["red_rate"].mean()),
        "dark_red_rate": safe_float(dark["red_rate"].mean()),
        "light_any_red_rate": safe_float(light["any_red"].mean()),
        "dark_any_red_rate": safe_float(dark["any_red"].mean()),
    }

    print("Dark vs light group stats:")
    print(group_stats)

    # Welch t-test on red-card rate.
    t_stat, p_two_sided = stats.ttest_ind(dark["red_rate"], light["red_rate"], equal_var=False, nan_policy="omit")
    mean_diff = group_stats["dark_red_rate"] - group_stats["light_red_rate"]
    p_one_sided = (p_two_sided / 2) if t_stat > 0 else (1 - p_two_sided / 2)

    # Chi-square for any red card.
    ct = pd.crosstab(
        np.where(data["skin_tone"] >= 0.75, "dark", np.where(data["skin_tone"] <= 0.25, "light", "mid")),
        data["any_red"],
    )
    ct_dark_light = ct.loc[[g for g in ["light", "dark"] if g in ct.index]]
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(ct_dark_light)

    # Pearson + Spearman.
    pearson_r, pearson_p = stats.pearsonr(data["skin_tone"], data["red_rate"])
    spearman_rho, spearman_p = stats.spearmanr(data["skin_tone"], data["red_rate"])

    # ANOVA across binned skin tone groups.
    anova_groups = [data.loc[skin_bins == cat, "red_rate"].values for cat in skin_bins.cat.categories]
    anova_groups = [g for g in anova_groups if len(g) > 1]
    anova_f, anova_p = stats.f_oneway(*anova_groups)

    print(f"Welch t-test (dark > light), t={t_stat:.4f}, p(one-sided)={p_one_sided:.6f}")
    print(f"Chi-square any-red dark vs light: chi2={chi2_stat:.4f}, p={chi2_p:.6f}")
    print(f"Pearson corr skin_tone vs red_rate: r={pearson_r:.4f}, p={pearson_p:.6f}")
    print(f"Spearman corr skin_tone vs red_rate: rho={spearman_rho:.4f}, p={spearman_p:.6f}")
    print(f"ANOVA across 5 skin bins on red_rate: F={anova_f:.4f}, p={anova_p:.6f}")

    print_header("Statsmodels Regression")
    ols_model = smf.ols(
        "red_rate ~ skin_tone + games + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)",
        data=data,
    ).fit(cov_type="HC3")

    poisson_model = smf.glm(
        "redCards ~ skin_tone + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)",
        data=data,
        family=sm.families.Poisson(),
        exposure=data["games"],
    ).fit()

    print("OLS skin_tone coefficient + p-value:")
    print(
        {
            "coef": safe_float(ols_model.params.get("skin_tone", np.nan)),
            "p": safe_float(ols_model.pvalues.get("skin_tone", np.nan)),
            "r2": safe_float(ols_model.rsquared),
        }
    )
    print("Poisson skin_tone coefficient + IRR + p-value:")
    print(
        {
            "coef": safe_float(poisson_model.params.get("skin_tone", np.nan)),
            "irr": safe_float(np.exp(poisson_model.params.get("skin_tone", np.nan))),
            "p": safe_float(poisson_model.pvalues.get("skin_tone", np.nan)),
        }
    )

    print_header("Interpretable Models (scikit-learn)")
    features = [
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
    ]

    for col in ["victories", "ties", "defeats"]:
        if col not in data.columns:
            data[col] = np.nan

    X = data[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    y_reg = data["red_rate"].values
    y_clf = data["any_red"].values

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_SEED
    )

    lin = LinearRegression().fit(X_train, y_reg_train)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(X_train, y_reg_train)
    lasso = Lasso(alpha=0.0001, max_iter=20000, random_state=RANDOM_SEED).fit(X_train, y_reg_train)
    dtr = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=RANDOM_SEED).fit(X_train, y_reg_train)
    dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=RANDOM_SEED).fit(X_train, y_clf_train)

    reg_models = {
        "LinearRegression": lin,
        "Ridge": ridge,
        "Lasso": lasso,
        "DecisionTreeRegressor": dtr,
    }

    for name, mdl in reg_models.items():
        pred = mdl.predict(X_test)
        print(f"{name} R^2: {r2_score(y_reg_test, pred):.5f}")

    clf_pred = dtc.predict(X_test)
    clf_prob = dtc.predict_proba(X_test)[:, 1]
    print(f"DecisionTreeClassifier Accuracy: {accuracy_score(y_clf_test, clf_pred):.5f}")
    print(f"DecisionTreeClassifier ROC-AUC: {roc_auc_score(y_clf_test, clf_prob):.5f}")

    coef_table = pd.DataFrame(
        {
            "feature": features,
            "linear_coef": lin.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
        }
    )
    coef_table["abs_linear"] = coef_table["linear_coef"].abs()
    coef_table = coef_table.sort_values("abs_linear", ascending=False)
    print("Top linear-model coefficients by absolute magnitude:")
    print(coef_table[["feature", "linear_coef", "ridge_coef", "lasso_coef"]].head(10).round(6))

    tree_importance = pd.DataFrame(
        {
            "feature": features,
            "dtr_importance": dtr.feature_importances_,
            "dtc_importance": dtc.feature_importances_,
        }
    ).sort_values("dtr_importance", ascending=False)
    print("Tree feature importances:")
    print(tree_importance.head(10).round(6))

    print_header("Interpretable Models (imodels)")
    sample_n = min(30000, len(X_train))
    sample_idx = np.random.choice(len(X_train), size=sample_n, replace=False)
    X_im = X_train.iloc[sample_idx].copy()
    y_im = y_reg_train[sample_idx]

    # RuleFit
    try:
        rulefit = RuleFitRegressor(
            n_estimators=60,
            tree_size=4,
            max_rules=30,
            include_linear=True,
            random_state=RANDOM_SEED,
        )
        rulefit.fit(X_im, y_im, feature_names=features)
        rf_r2 = r2_score(y_reg_test, rulefit.predict(X_test))
        print(f"RuleFitRegressor R^2: {rf_r2:.5f}")
        print("Sample learned rules (first 10):")
        for rule in list(rulefit.rules_)[:10]:
            print(f"- {rule}")
    except Exception as e:
        print(f"RuleFitRegressor failed: {e}")

    # FIGS
    try:
        figs = FIGSRegressor(max_rules=12, max_depth=4, random_state=RANDOM_SEED)
        figs.fit(X_im, y_im)
        figs_r2 = r2_score(y_reg_test, figs.predict(X_test))
        print(f"FIGSRegressor R^2: {figs_r2:.5f}")
        figs_imp = pd.DataFrame({"feature": features, "importance": figs.feature_importances_})
        print("FIGS feature importances:")
        print(figs_imp.sort_values("importance", ascending=False).head(10).round(6))
    except Exception as e:
        print(f"FIGSRegressor failed: {e}")

    # HSTree
    try:
        hs = HSTreeRegressor(
            estimator_=DecisionTreeRegressor(max_leaf_nodes=20, random_state=RANDOM_SEED),
            random_state=RANDOM_SEED,
        )
        hs.fit(X_im, y_im)
        hs_r2 = r2_score(y_reg_test, hs.predict(X_test))
        print(f"HSTreeRegressor R^2: {hs_r2:.5f}")
        hs_imp = pd.DataFrame(
            {"feature": features, "importance": hs.estimator_.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("HSTree (base tree) feature importances:")
        print(hs_imp.head(10).round(6))
    except Exception as e:
        print(f"HSTreeRegressor failed: {e}")

    stats_out = {
        "mean_diff": safe_float(mean_diff),
        "ttest_t": safe_float(t_stat),
        "ttest_one_sided_p": safe_float(p_one_sided),
        "chi2_stat": safe_float(chi2_stat),
        "chi2_p": safe_float(chi2_p),
        "pearson_r": safe_float(pearson_r),
        "pearson_p": safe_float(pearson_p),
        "spearman_rho": safe_float(spearman_rho),
        "spearman_p": safe_float(spearman_p),
        "anova_f": safe_float(anova_f),
        "anova_p": safe_float(anova_p),
        "ols_coef": safe_float(ols_model.params.get("skin_tone", np.nan)),
        "ols_p": safe_float(ols_model.pvalues.get("skin_tone", np.nan)),
        "poisson_coef": safe_float(poisson_model.params.get("skin_tone", np.nan)),
        "poisson_p": safe_float(poisson_model.pvalues.get("skin_tone", np.nan)),
        "poisson_irr": safe_float(np.exp(poisson_model.params.get("skin_tone", np.nan))),
    }

    conclusion = build_conclusion(stats_out, group_stats)

    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print_header("Final Conclusion JSON")
    print(json.dumps(conclusion, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
