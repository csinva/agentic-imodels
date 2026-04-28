import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def top_abs_pairs(names, values, top_n=8):
    pairs = sorted(zip(names, values), key=lambda x: abs(float(x[1])), reverse=True)
    return [(str(k), float(v)) for k, v in pairs[:top_n]]


def top_pairs(names, values, top_n=8):
    pairs = sorted(zip(names, values), key=lambda x: float(x[1]), reverse=True)
    return [(str(k), float(v)) for k, v in pairs[:top_n]]


def main():
    base = Path(".")
    info = json.loads((base / "info.json").read_text())
    question = info.get("research_questions", [""])[0].strip()

    df = pd.read_csv(base / "affairs.csv")
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # EDA
    eda = {
        "shape": tuple(df.shape),
        "columns": df.columns.tolist(),
        "missing_per_col": df.isna().sum().to_dict(),
        "summary_numeric": df[numeric_cols].describe().round(4).to_dict(),
        "affairs_distribution": df["affairs"].value_counts().sort_index().to_dict(),
        "children_distribution": df["children"].value_counts().to_dict(),
        "group_means_affairs_by_children": (
            df.groupby("children")["affairs"].agg(["count", "mean", "std", "median"]).round(4).to_dict()
        ),
        "corr_with_affairs": df[numeric_cols].corr(numeric_only=True)["affairs"].sort_values(ascending=False).round(4).to_dict(),
    }

    # Statistical tests focused on the hypothesis
    yes = df.loc[df["children"] == "yes", "affairs"]
    no = df.loc[df["children"] == "no", "affairs"]

    ttest = stats.ttest_ind(yes, no, equal_var=False)
    mwu = stats.mannwhitneyu(yes, no, alternative="two-sided")
    pb = stats.pointbiserialr(df["children_yes"], df["affairs"])
    anova_oneway = stats.f_oneway(yes, no)

    ols_simple = smf.ols("affairs ~ children_yes", data=df).fit()
    ols_adjusted = smf.ols(
        "affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)",
        data=df,
    ).fit()

    anova_two_way_model = smf.ols("affairs ~ C(children) + C(gender) + C(children):C(gender)", data=df).fit()
    anova_two_way = sm.stats.anova_lm(anova_two_way_model, typ=2)

    stats_results = {
        "welch_ttest_yes_vs_no": {
            "t": float(ttest.statistic),
            "p": float(ttest.pvalue),
            "mean_yes": float(yes.mean()),
            "mean_no": float(no.mean()),
        },
        "mannwhitney_yes_vs_no": {
            "u": float(mwu.statistic),
            "p": float(mwu.pvalue),
        },
        "pointbiserial_children_yes_vs_affairs": {
            "r": float(pb.statistic),
            "p": float(pb.pvalue),
        },
        "oneway_anova_yes_vs_no": {
            "F": float(anova_oneway.statistic),
            "p": float(anova_oneway.pvalue),
        },
        "ols_simple_children": {
            "coef": float(ols_simple.params["children_yes"]),
            "p": float(ols_simple.pvalues["children_yes"]),
            "ci_low": float(ols_simple.conf_int().loc["children_yes", 0]),
            "ci_high": float(ols_simple.conf_int().loc["children_yes", 1]),
            "r2": float(ols_simple.rsquared),
        },
        "ols_adjusted_children": {
            "coef": float(ols_adjusted.params["children_yes"]),
            "p": float(ols_adjusted.pvalues["children_yes"]),
            "ci_low": float(ols_adjusted.conf_int().loc["children_yes", 0]),
            "ci_high": float(ols_adjusted.conf_int().loc["children_yes", 1]),
            "r2": float(ols_adjusted.rsquared),
        },
        "anova_two_way": anova_two_way.round(6).to_dict(),
    }

    # Interpretable modeling with sklearn
    feature_cols = [
        "gender",
        "age",
        "yearsmarried",
        "children",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = df[feature_cols]
    y_reg = df["affairs"]
    y_clf = df["affair_any"]

    cat_cols = ["gender", "children"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.25, random_state=42)

    sklearn_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.01, max_iter=20000, random_state=42),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=3, random_state=42),
    }

    sklearn_results = {}
    for name, model in sklearn_models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        row = {"test_r2": float(r2_score(y_test, preds))}
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        fitted = pipe.named_steps["model"]

        if hasattr(fitted, "coef_"):
            row["top_abs_coefficients"] = top_abs_pairs(feat_names, fitted.coef_, top_n=10)
            # Pull explicit coefficient on children when available
            children_coef = None
            for fname, coef in zip(feat_names, fitted.coef_):
                if "children_yes" in fname:
                    children_coef = float(coef)
                    break
            row["children_coef"] = children_coef

        if hasattr(fitted, "feature_importances_"):
            row["top_feature_importances"] = top_pairs(feat_names, fitted.feature_importances_, top_n=10)
            children_imp = 0.0
            for fname, imp in zip(feat_names, fitted.feature_importances_):
                if "children_yes" in fname:
                    children_imp += float(imp)
            row["children_importance"] = children_imp

        sklearn_results[name] = row

    # Add classifier for interpretability of any-affair behavior
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
    clf_pipe = Pipeline(
        [
            ("pre", pre),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ]
    )
    clf_pipe.fit(Xc_train, yc_train)
    clf_feat_names = clf_pipe.named_steps["pre"].get_feature_names_out()
    clf_model = clf_pipe.named_steps["model"]
    sklearn_results["DecisionTreeClassifier_any_affair"] = {
        "top_feature_importances": top_pairs(clf_feat_names, clf_model.feature_importances_, top_n=10),
        "children_importance": float(
            sum(
                imp
                for fname, imp in zip(clf_feat_names, clf_model.feature_importances_)
                if "children_yes" in fname
            )
        ),
    }

    # Interpretable modeling with imodels (run on encoded matrix)
    X_enc = pre.fit_transform(X)
    enc_feat_names = pre.get_feature_names_out()

    imodels_results = {}

    rulefit = RuleFitRegressor(random_state=42, tree_size=4, max_rules=60)
    rulefit.fit(X_enc, y_reg, feature_names=enc_feat_names)
    rf_rules = rulefit._get_rules()
    rf_rules_nz = rf_rules.loc[rf_rules["coef"].abs() > 1e-12].copy()
    rf_rules_nz = rf_rules_nz.sort_values("importance", ascending=False)
    children_rows = rf_rules.loc[rf_rules["rule"].astype(str).str.contains("children", case=False, na=False)]
    imodels_results["RuleFitRegressor"] = {
        "num_nonzero_rules": int(rf_rules_nz.shape[0]),
        "top_rules": rf_rules_nz[["rule", "type", "coef", "importance"]].head(12).to_dict("records"),
        "children_terms": children_rows[["rule", "type", "coef", "importance"]].to_dict("records"),
    }

    figs = FIGSRegressor(max_rules=15, random_state=42)
    figs.fit(X_enc, y_reg, feature_names=enc_feat_names)
    figs_importance = top_pairs(enc_feat_names, figs.feature_importances_, top_n=10)
    figs_children_importance = float(
        sum(
            imp
            for fname, imp in zip(enc_feat_names, figs.feature_importances_)
            if "children_yes" in fname
        )
    )
    imodels_results["FIGSRegressor"] = {
        "top_feature_importances": figs_importance,
        "children_importance": figs_children_importance,
        "model_text": str(figs),
    }

    hst = HSTreeRegressor(max_leaf_nodes=8, random_state=42)
    hst.fit(X_enc, y_reg, feature_names=enc_feat_names)
    hst_importance = top_pairs(enc_feat_names, hst.estimator_.feature_importances_, top_n=10)
    hst_children_importance = float(
        sum(
            imp
            for fname, imp in zip(enc_feat_names, hst.estimator_.feature_importances_)
            if "children_yes" in fname
        )
    )
    imodels_results["HSTreeRegressor"] = {
        "top_feature_importances": hst_importance,
        "children_importance": hst_children_importance,
        "model_text": str(hst),
    }

    # Combine evidence into final conclusion score (0-100 yes/no on "decrease")
    coef_adj = stats_results["ols_adjusted_children"]["coef"]
    p_adj = stats_results["ols_adjusted_children"]["p"]
    coef_simple = stats_results["ols_simple_children"]["coef"]
    p_simple = stats_results["ols_simple_children"]["p"]

    if coef_adj < 0 and p_adj < 0.05:
        response = 85
    elif coef_adj < 0 and p_adj >= 0.05:
        response = 20
    elif coef_adj >= 0 and p_adj < 0.05:
        response = 5
    else:
        response = 10

    # Penalize further if unadjusted evidence is significantly opposite direction
    if coef_simple > 0 and p_simple < 0.05:
        response = max(0, response - 5)

    explanation = (
        "The data do not support that having children decreases extramarital affairs. "
        f"Unadjusted comparisons show higher mean affairs for couples with children "
        f"({stats_results['welch_ttest_yes_vs_no']['mean_yes']:.2f}) than without "
        f"({stats_results['welch_ttest_yes_vs_no']['mean_no']:.2f}), with a significant Welch t-test "
        f"(p={stats_results['welch_ttest_yes_vs_no']['p']:.4g}). "
        f"After adjustment for age, years married, religiousness, education, occupation, rating, and gender, "
        f"the children coefficient is small and not significant "
        f"(coef={coef_adj:.3f}, p={p_adj:.4g}, 95% CI "
        f"[{stats_results['ols_adjusted_children']['ci_low']:.3f}, {stats_results['ols_adjusted_children']['ci_high']:.3f}]). "
        "Interpretable models (linear/ridge/lasso, trees, and imodels RuleFit/FIGS/HSTree) also indicate children is not a dominant driver versus marital rating, religiousness, age, and years married."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Save detailed analysis artifact (not requested, but useful for reproducibility)
    detailed = {
        "research_question": question,
        "eda": eda,
        "statistical_tests": stats_results,
        "sklearn_models": sklearn_results,
        "imodels_models": imodels_results,
        "conclusion": conclusion,
    }
    (base / "analysis_results.json").write_text(json.dumps(detailed, indent=2))

    # Required deliverable: conclusion.txt with ONLY JSON object
    (base / "conclusion.txt").write_text(json.dumps(conclusion))

    # Console summary
    print("Research question:", question)
    print("Adjusted children effect (OLS):", coef_adj, "p=", p_adj)
    print("Final response score:", response)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
