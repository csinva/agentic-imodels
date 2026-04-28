import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


RANDOM_STATE = 0


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def format_p(p: Optional[float]) -> str:
    if p is None:
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def get_reader_coef(model, feature_names: List[str]) -> Optional[float]:
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_
    if isinstance(coef, (list, tuple)):
        coef = np.array(coef)
    if getattr(coef, "ndim", 1) > 1:
        coef = coef.ravel()
    pairs = dict(zip(feature_names, coef))

    if "reader_view" in pairs:
        return safe_float(pairs["reader_view"])

    # Fallback: first transformed feature that starts with reader_view
    for k, v in pairs.items():
        if "reader_view" in k:
            return safe_float(v)
    return None


def summarize_eda(df: pd.DataFrame) -> None:
    print("=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print("\nNumeric summary stats:")
    print(df[numeric_cols].describe().T.to_string())

    print("\nDistribution snapshots:")
    for col in ["speed", "running_time", "adjusted_running_time", "correct_rate"]:
        if col in df.columns:
            series = df[col].dropna()
            hist_counts, hist_bins = np.histogram(series, bins=8)
            print(f"{col}: mean={series.mean():.3f}, median={series.median():.3f}, std={series.std():.3f}")
            print(f"  bins={np.round(hist_bins, 2)}")
            print(f"  counts={hist_counts}")

    if "speed" in df.columns and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True)["speed"].drop("speed", errors="ignore")
        corr = corr.dropna().sort_values(key=np.abs, ascending=False)
        print("\nTop absolute correlations with speed:")
        print(corr.head(10).to_string())


def choose_dyslexia_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "dyslexia_bin" in df.columns:
        subset = df[df["dyslexia_bin"] == 1].copy()
        if len(subset) > 0:
            return subset
    if "dyslexia" in df.columns:
        subset = df[df["dyslexia"] > 0].copy()
        if len(subset) > 0:
            return subset
    return df.copy()


def run_stat_tests(df: pd.DataFrame, dys_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}

    # Welch's t-test in dyslexia subgroup
    if "reader_view" in dys_df.columns and "speed" in dys_df.columns:
        g1 = dys_df.loc[dys_df["reader_view"] == 1, "speed"].dropna()
        g0 = dys_df.loc[dys_df["reader_view"] == 0, "speed"].dropna()
        if len(g1) >= 3 and len(g0) >= 3:
            t_res = stats.ttest_ind(g1, g0, equal_var=False)
            out["welch_ttest"] = {
                "effect": safe_float(g1.mean() - g0.mean()),
                "stat": safe_float(t_res.statistic),
                "p": safe_float(t_res.pvalue),
                "n_group1": float(len(g1)),
                "n_group0": float(len(g0)),
            }

    # Paired t-test by participant (uuid) if available
    if {"uuid", "reader_view", "speed"}.issubset(dys_df.columns):
        paired = dys_df.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
        if {0, 1}.issubset(paired.columns):
            paired = paired[[0, 1]].dropna()
            if len(paired) >= 3:
                pt = stats.ttest_rel(paired[1], paired[0])
                out["paired_ttest"] = {
                    "effect": safe_float((paired[1] - paired[0]).mean()),
                    "stat": safe_float(pt.statistic),
                    "p": safe_float(pt.pvalue),
                    "n_pairs": float(len(paired)),
                }

    # OLS with controls on dyslexia subgroup
    candidate_covars = [
        "num_words",
        "Flesch_Kincaid",
        "correct_rate",
        "age",
        "gender",
        "retake_trial",
    ]
    covars = [c for c in candidate_covars if c in dys_df.columns]
    cat_terms = []
    for c in ["device", "education", "page_id", "language", "english_native"]:
        if c in dys_df.columns:
            cat_terms.append(f"C({c})")

    if "reader_view" in dys_df.columns:
        formula = "speed ~ reader_view"
        if covars:
            formula += " + " + " + ".join(covars)
        if cat_terms:
            formula += " + " + " + ".join(cat_terms)

        ols_model = smf.ols(formula, data=dys_df).fit(cov_type="HC3")
        out["ols_controls"] = {
            "effect": safe_float(ols_model.params.get("reader_view")),
            "p": safe_float(ols_model.pvalues.get("reader_view")),
        }

    # Participant/page fixed effects in dyslexia subgroup
    if {"uuid", "page_id", "reader_view", "speed"}.issubset(dys_df.columns):
        fe_formula = "speed ~ reader_view + C(uuid) + C(page_id)"
        fe_model = smf.ols(fe_formula, data=dys_df).fit()
        out["ols_fixed_effects"] = {
            "effect": safe_float(fe_model.params.get("reader_view")),
            "p": safe_float(fe_model.pvalues.get("reader_view")),
        }

    # Interaction in full sample: does effect differ by dyslexia status?
    if {"reader_view", "dyslexia_bin", "speed"}.issubset(df.columns):
        int_formula = "speed ~ C(reader_view) * C(dyslexia_bin)"
        for c in ["num_words", "Flesch_Kincaid", "correct_rate", "age"]:
            if c in df.columns:
                int_formula += f" + {c}"
        if "page_id" in df.columns:
            int_formula += " + C(page_id)"
        if "device" in df.columns:
            int_formula += " + C(device)"

        int_model = smf.ols(int_formula, data=df).fit(cov_type="HC3")
        main_term = "C(reader_view)[T.1]"
        int_term = "C(reader_view)[T.1]:C(dyslexia_bin)[T.1.0]"
        out["interaction_full_sample"] = {
            "effect_main": safe_float(int_model.params.get(main_term)),
            "p_main": safe_float(int_model.pvalues.get(main_term)),
            "effect_interaction": safe_float(int_model.params.get(int_term)),
            "p_interaction": safe_float(int_model.pvalues.get(int_term)),
        }

    return out


def run_interpretable_models(dys_df: pd.DataFrame) -> Dict[str, object]:
    results: Dict[str, object] = {}

    feature_candidates = [
        "reader_view",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "gender",
        "retake_trial",
        "Flesch_Kincaid",
        "dyslexia",
        "device",
        "education",
        "page_id",
        "language",
        "english_native",
    ]

    if "speed" not in dys_df.columns:
        return results

    feats = [f for f in feature_candidates if f in dys_df.columns]
    if "reader_view" not in feats:
        return results

    model_df = dys_df[feats + ["speed"]].copy()
    model_df = model_df.dropna(subset=["speed"]).copy()
    y = model_df["speed"].astype(float)
    X_raw = model_df[feats]

    numeric_features = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X_raw.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
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

    sk_models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "lasso": Lasso(alpha=0.05, random_state=RANDOM_STATE, max_iter=10000),
        "decision_tree": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
    }

    for name, reg in sk_models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", reg)])
        pipe.fit(X_raw, y)
        model = pipe.named_steps["model"]
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()

        if name in {"linear_regression", "ridge", "lasso"}:
            reader_coef = get_reader_coef(model, feature_names)
            results[name] = {"reader_view_coef": safe_float(reader_coef)}
        else:
            fi = model.feature_importances_
            order = np.argsort(fi)[::-1]
            top = [
                {"feature": feature_names[i], "importance": safe_float(fi[i])}
                for i in order[:8]
                if fi[i] > 0
            ]
            results[name] = {
                "top_features": top,
                "reader_view_importance": safe_float(
                    next((fi[i] for i in order if "reader_view" in feature_names[i]), 0.0)
                ),
            }

    # Imodels expects fully numeric arrays; manually encode/impute
    X_imp = X_raw.copy()
    for c in numeric_features:
        X_imp[c] = X_imp[c].fillna(X_imp[c].median())
    for c in categorical_features:
        mode = X_imp[c].mode()
        fill_val = mode.iloc[0] if not mode.empty else "missing"
        X_imp[c] = X_imp[c].fillna(fill_val)
    X_im = pd.get_dummies(X_imp, drop_first=True)

    imodel_specs = {
        "rulefit": RuleFitRegressor(random_state=RANDOM_STATE, max_rules=40),
        "figs": FIGSRegressor(max_rules=20, random_state=RANDOM_STATE),
        "hstree": HSTreeRegressor(),
    }

    for name, mdl in imodel_specs.items():
        try:
            mdl.fit(X_im, y)
            item: Dict[str, object] = {}

            if hasattr(mdl, "feature_importances_"):
                fi = np.asarray(mdl.feature_importances_)
                order = np.argsort(fi)[::-1]
                top = [
                    {"feature": X_im.columns[i], "importance": safe_float(fi[i])}
                    for i in order[:8]
                    if fi[i] > 0
                ]
                item["top_features"] = top
                item["reader_view_importance"] = safe_float(
                    next((fi[i] for i in order if X_im.columns[i] == "reader_view"), 0.0)
                )

            if name == "rulefit" and hasattr(mdl, "_get_rules"):
                rules_df = mdl._get_rules()
                # Reader_view linear term (if present)
                rv_rows = rules_df[rules_df["rule"] == "reader_view"]
                if len(rv_rows) > 0:
                    item["reader_view_coef"] = safe_float(rv_rows.iloc[0]["coef"])
                # top non-zero rules/terms
                nonzero = rules_df[rules_df["coef"] != 0].sort_values("importance", ascending=False)
                top_rules = nonzero.head(8)[["rule", "type", "coef", "importance"]]
                item["top_rules"] = top_rules.to_dict(orient="records")

            results[name] = item
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return results


def score_conclusion(stat_results: Dict[str, Dict[str, Optional[float]]], model_results: Dict[str, object]) -> Tuple[int, str]:
    def get_test(name: str, key_effect: str = "effect", key_p: str = "p") -> Tuple[Optional[float], Optional[float]]:
        t = stat_results.get(name, {})
        return safe_float(t.get(key_effect)), safe_float(t.get(key_p))

    welch_eff, welch_p = get_test("welch_ttest")
    paired_eff, paired_p = get_test("paired_ttest")
    ols_eff, ols_p = get_test("ols_controls")
    fe_eff, fe_p = get_test("ols_fixed_effects")

    tests = [
        ("Welch t-test", welch_eff, welch_p),
        ("Paired t-test", paired_eff, paired_p),
        ("OLS + controls", ols_eff, ols_p),
        ("OLS fixed effects", fe_eff, fe_p),
    ]

    pos_sig = sum(1 for _, e, p in tests if e is not None and p is not None and e > 0 and p < 0.05)
    neg_sig = sum(1 for _, e, p in tests if e is not None and p is not None and e < 0 and p < 0.05)

    # Interpretable model consistency check (direction of reader_view term)
    linear_dirs = []
    for name in ["linear_regression", "ridge", "lasso", "rulefit"]:
        item = model_results.get(name, {})
        if isinstance(item, dict):
            coef = safe_float(item.get("reader_view_coef"))
            if coef is not None:
                linear_dirs.append(coef)

    model_pos = sum(1 for c in linear_dirs if c > 0)
    model_neg = sum(1 for c in linear_dirs if c < 0)

    # Core decision logic
    if pos_sig >= 2 and neg_sig == 0:
        score = 85
        verdict = "evidence supports improved reading speed"
    elif pos_sig >= 1 and neg_sig == 0:
        score = 70
        verdict = "some evidence suggests improved reading speed"
    elif neg_sig >= 1 and pos_sig == 0:
        score = 12
        verdict = "evidence suggests no improvement (and possible slowdown)"
    else:
        # mostly non-significant evidence
        score = 22
        verdict = "no statistically reliable improvement detected"

    explanation_parts = [
        "Question: whether Reader View improves reading speed for participants with dyslexia.",
        (
            f"Welch t-test (reader_view=1 vs 0 in dyslexia subgroup): effect={welch_eff:.2f} wpm, p={format_p(welch_p)}; "
            if welch_eff is not None
            else "Welch t-test unavailable; "
        )
        + (
            f"paired t-test by participant: effect={paired_eff:.2f} wpm, p={format_p(paired_p)}; "
            if paired_eff is not None
            else "paired test unavailable; "
        )
        + (
            f"OLS with controls: coef={ols_eff:.2f}, p={format_p(ols_p)}; "
            if ols_eff is not None
            else "OLS controls unavailable; "
        )
        + (
            f"participant/page fixed-effects OLS: coef={fe_eff:.2f}, p={format_p(fe_p)}."
            if fe_eff is not None
            else "fixed-effects OLS unavailable."
        ),
        (
            f"Interpretable model direction for reader_view terms: {model_pos} positive vs {model_neg} negative coefficients across "
            "LinearRegression/Ridge/Lasso/RuleFit."
            if linear_dirs
            else "Interpretable model direction for reader_view could not be fully estimated."
        ),
        f"Conclusion: {verdict}.",
    ]

    explanation = " ".join(explanation_parts)
    return int(score), explanation


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_q = info.get("research_questions", ["No research question found"])[0]
    print("Research question:", research_q)

    df = pd.read_csv("reading.csv")

    summarize_eda(df)

    dys_df = choose_dyslexia_subset(df)
    print("\n=== DYSLEXIA SUBGROUP ===")
    print(f"Rows: {len(dys_df)}, Participants: {dys_df['uuid'].nunique() if 'uuid' in dys_df.columns else 'NA'}")
    if {"reader_view", "speed"}.issubset(dys_df.columns):
        print("Speed by reader_view in dyslexia subgroup:")
        print(dys_df.groupby("reader_view")["speed"].describe().to_string())

    stat_results = run_stat_tests(df, dys_df)
    print("\n=== STATISTICAL TESTS ===")
    print(json.dumps(stat_results, indent=2))

    model_results = run_interpretable_models(dys_df)
    print("\n=== INTERPRETABLE MODELS ===")
    print(json.dumps(model_results, indent=2, default=str))

    response, explanation = score_conclusion(stat_results, model_results)
    conclusion = {
        "response": int(np.clip(response, 0, 100)),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
