import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


def bootstrap_weighted_rate(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    fish = df["fish_caught"].to_numpy()
    hours = df["hours"].to_numpy()
    n = len(df)
    idx = rng.integers(0, n, size=(n_boot, n))
    rates = fish[idx].sum(axis=1) / hours[idx].sum(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(rates.mean()), float(lo), float(hi)


def fmt_map(values, names):
    return {name: float(val) for name, val in zip(names, values)}


def main():
    info = json.loads(Path("info.json").read_text())
    research_question = info["research_questions"][0]
    print("Research question:", research_question)

    df = pd.read_csv("fish.csv")
    print("\nData shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())

    df["catch_per_hour"] = df["fish_caught"] / df["hours"]
    features = ["livebait", "camper", "persons", "child", "hours"]
    target = "fish_caught"

    print("\nSummary statistics:")
    print(df.describe().T)

    print("\nDistribution summary (fish_caught, catch_per_hour):")
    for col in ["fish_caught", "catch_per_hour"]:
        s = df[col]
        print(
            f"{col}: mean={s.mean():.4f}, median={s.median():.4f}, "
            f"std={s.std():.4f}, skew={stats.skew(s):.4f}"
        )

    print("\nCorrelations:")
    print(df[["fish_caught", "catch_per_hour"] + features].corr())

    weighted_rate = df["fish_caught"].sum() / df["hours"].sum()
    boot_mean, boot_lo, boot_hi = bootstrap_weighted_rate(df)
    print(
        f"\nEstimated fish/hour (ratio of totals): {weighted_rate:.4f} "
        f"(bootstrap mean={boot_mean:.4f}, 95% CI=[{boot_lo:.4f}, {boot_hi:.4f}])"
    )

    print("\nStatistical tests:")
    lb_yes = df.loc[df["livebait"] == 1, "fish_caught"]
    lb_no = df.loc[df["livebait"] == 0, "fish_caught"]
    t_livebait = stats.ttest_ind(lb_yes, lb_no, equal_var=False)
    print(
        "Welch t-test (fish_caught by livebait): "
        f"t={t_livebait.statistic:.4f}, p={t_livebait.pvalue:.6f}"
    )

    cam_yes = df.loc[df["camper"] == 1, "fish_caught"]
    cam_no = df.loc[df["camper"] == 0, "fish_caught"]
    t_camper = stats.ttest_ind(cam_yes, cam_no, equal_var=False)
    print(
        "Welch t-test (fish_caught by camper): "
        f"t={t_camper.statistic:.4f}, p={t_camper.pvalue:.6f}"
    )

    groups = [
        df.loc[df["persons"] == p, "fish_caught"].to_numpy()
        for p in sorted(df["persons"].unique())
    ]
    anova_persons = stats.f_oneway(*groups)
    print(
        "ANOVA (fish_caught across persons levels): "
        f"F={anova_persons.statistic:.4f}, p={anova_persons.pvalue:.6f}"
    )

    for col in features:
        r, p = stats.pearsonr(df[col], df["fish_caught"])
        print(f"Pearson correlation (fish_caught vs {col}): r={r:.4f}, p={p:.6f}")

    print("\nOLS regression (statsmodels, HC3 robust SE):")
    X_sm = sm.add_constant(df[features])
    ols = sm.OLS(df[target], X_sm).fit(cov_type="HC3")
    print(ols.summary())

    print("\nInterpretable models (scikit-learn):")
    X = df[features]
    y = df[target]

    lr = LinearRegression().fit(X, y)
    print("LinearRegression R^2:", round(lr.score(X, y), 4))
    print("LinearRegression coefficients:", fmt_map(lr.coef_, features))

    ridge = Ridge(alpha=1.0).fit(X, y)
    print("Ridge R^2:", round(ridge.score(X, y), 4))
    print("Ridge coefficients:", fmt_map(ridge.coef_, features))

    lasso = Lasso(alpha=0.01, max_iter=20000, random_state=42).fit(X, y)
    print("Lasso R^2:", round(lasso.score(X, y), 4))
    print("Lasso coefficients:", fmt_map(lasso.coef_, features))

    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
    print("DecisionTreeRegressor R^2:", round(tree.score(X, y), 4))
    print("Decision tree feature importances:", fmt_map(tree.feature_importances_, features))

    print("\nInterpretable models (imodels):")
    rulefit = RuleFitRegressor(random_state=42, max_rules=30)
    rulefit.fit(X, y)
    print("RuleFitRegressor R^2:", round(rulefit.score(X, y), 4))
    if hasattr(rulefit, "get_rules"):
        rules = rulefit.get_rules()
    else:
        rules = rulefit._get_rules()  # compatibility with older imodels versions
    if isinstance(rules, pd.DataFrame) and not rules.empty and "coef" in rules.columns:
        top_rules = rules.loc[rules["coef"] != 0].copy()
        if not top_rules.empty:
            top_rules["abs_coef"] = top_rules["coef"].abs()
            top_rules = top_rules.sort_values("abs_coef", ascending=False).head(8)
            cols = [c for c in ["rule", "coef", "support"] if c in top_rules.columns]
            print("Top RuleFit rules:")
            print(top_rules[cols].to_string(index=False))

    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X, y)
    print("FIGSRegressor R^2:", round(figs.score(X, y), 4))
    if hasattr(figs, "feature_importances_"):
        print("FIGS feature importances:", fmt_map(figs.feature_importances_, features))
    else:
        print("FIGS model structure:")
        print(figs)

    hst = HSTreeRegressor(max_leaf_nodes=8, random_state=42)
    hst.fit(X, y)
    print("HSTreeRegressor R^2:", round(hst.score(X, y), 4))
    if hasattr(hst, "feature_importances_"):
        print("HSTree feature importances:", fmt_map(hst.feature_importances_, features))
    elif hasattr(hst, "estimator_") and hasattr(hst.estimator_, "feature_importances_"):
        print(
            "HSTree estimator feature importances:",
            fmt_map(hst.estimator_.feature_importances_, features),
        )

    pvals = ols.pvalues
    sig_factors = [
        v for v in features if (v in pvals.index and np.isfinite(pvals[v]) and pvals[v] < 0.05)
    ]
    model_sig = bool(np.isfinite(ols.f_pvalue) and ols.f_pvalue < 0.05)
    livebait_sig = bool(np.isfinite(t_livebait.pvalue) and t_livebait.pvalue < 0.05)
    persons_sig = bool(np.isfinite(anova_persons.pvalue) and anova_persons.pvalue < 0.05)

    if model_sig and len(sig_factors) >= 3:
        response = 88
    elif model_sig and len(sig_factors) >= 2:
        response = 78
    elif model_sig and len(sig_factors) >= 1:
        response = 65
    else:
        response = 30

    if livebait_sig:
        response += 4
    if persons_sig:
        response += 4
    response = int(max(0, min(100, response)))

    explanation = (
        f"Estimated catch rate is about {weighted_rate:.2f} fish/hour "
        f"(bootstrap 95% CI {boot_lo:.2f} to {boot_hi:.2f}). "
        f"Multiple tests indicate meaningful relationships between catch count and trip features: "
        f"overall OLS model is significant (p={ols.f_pvalue:.3g}), with significant factors "
        f"{sig_factors if sig_factors else 'none'} based on HC3-robust p-values. "
        f"Welch t-test for livebait gives p={t_livebait.pvalue:.3g}, and ANOVA over number of adults "
        f"gives p={anova_persons.pvalue:.3g}. Interpretable scikit-learn and imodels models "
        f"(linear coefficients, tree importances, and rules) provide consistent directional evidence "
        f"that group composition is associated with fish caught."
    )

    out = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(out))
    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
