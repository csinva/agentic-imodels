"""
Research question: Does 'Reader View' improve reading speed for individuals with dyslexia?
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("reading.csv")
print("Shape:", df.shape)
print(df[["reader_view","dyslexia","dyslexia_bin","speed"]].describe())
print("\nDyslexia distribution:\n", df["dyslexia"].value_counts())
print("Reader-view distribution:\n", df["reader_view"].value_counts())

# Log-transform speed (heavy right skew)
df["log_speed"] = np.log1p(df["speed"])

# ── 2. Bivariate: reader_view × speed among dyslexic readers ─────────────────
dys = df[df["dyslexia_bin"] == 1].copy()
nondys = df[df["dyslexia_bin"] == 0].copy()

print("\n=== Dyslexic readers: mean log_speed by reader_view ===")
print(dys.groupby("reader_view")["log_speed"].agg(["mean","std","count"]))

t_stat, p_biv_dys = stats.ttest_ind(
    dys[dys["reader_view"]==1]["log_speed"].dropna(),
    dys[dys["reader_view"]==0]["log_speed"].dropna(),
)
print(f"Bivariate t-test (dyslexic): t={t_stat:.3f}, p={p_biv_dys:.4f}")

print("\n=== Non-dyslexic readers: mean log_speed by reader_view ===")
print(nondys.groupby("reader_view")["log_speed"].agg(["mean","std","count"]))
t_stat2, p_biv_nondys = stats.ttest_ind(
    nondys[nondys["reader_view"]==1]["log_speed"].dropna(),
    nondys[nondys["reader_view"]==0]["log_speed"].dropna(),
)
print(f"Bivariate t-test (non-dyslexic): t={t_stat2:.3f}, p={p_biv_nondys:.4f}")

# ── 3. OLS with controls: full dataset, interaction term ─────────────────────
# Encode categoricals
df["device_enc"] = pd.Categorical(df["device"]).codes
df["education_enc"] = pd.Categorical(df["education"]).codes
df["english_native_enc"] = (df["english_native"] == "Y").astype(int)

ctrl_cols = ["age","device_enc","education_enc","english_native_enc",
             "retake_trial","Flesch_Kincaid","num_words"]

# Interaction: reader_view * dyslexia_bin
df["rv_x_dys"] = df["reader_view"] * df["dyslexia_bin"]
df_clean = df[["log_speed","reader_view","dyslexia_bin","rv_x_dys"] + ctrl_cols].dropna()

X_full = sm.add_constant(df_clean[["reader_view","dyslexia_bin","rv_x_dys"] + ctrl_cols])
ols_full = sm.OLS(df_clean["log_speed"], X_full).fit()
print("\n=== OLS full dataset with interaction ===")
print(ols_full.summary())

# ── 4. OLS within dyslexic subset only ───────────────────────────────────────
dys_clean = df[df["dyslexia_bin"]==1][["log_speed","reader_view"]+ctrl_cols].dropna()
X_dys = sm.add_constant(dys_clean[["reader_view"]+ctrl_cols])
ols_dys = sm.OLS(dys_clean["log_speed"], X_dys).fit()
print("\n=== OLS (dyslexic readers only, controlled) ===")
print(ols_dys.summary())

p_val_controlled_dys = ols_dys.pvalues["reader_view"]
coef_controlled_dys  = ols_dys.params["reader_view"]
print(f"\nreader_view coef={coef_controlled_dys:.4f}, p={p_val_controlled_dys:.4f}")

# ── 5. Interpretable models: SmartAdditiveRegressor and HingeGAMRegressor ────
from agentic_imodels import SmartAdditiveRegressor, HingeGAMRegressor

num_cols = ["reader_view","dyslexia_bin","rv_x_dys","age",
            "device_enc","education_enc","english_native_enc",
            "retake_trial","Flesch_Kincaid","num_words"]

df_model = df[["log_speed"] + num_cols].dropna()
X_m = df_model[num_cols].astype(float)
y_m = df_model["log_speed"].values

print("\n=== SmartAdditiveRegressor (full data) ===")
smart = SmartAdditiveRegressor()
smart.fit(X_m, y_m)
print(smart)

print("\n=== HingeGAMRegressor (full data) ===")
hinge = HingeGAMRegressor()
hinge.fit(X_m, y_m)
print(hinge)

# Also fit on dyslexic subset
dys_model = df[df["dyslexia_bin"]==1][["log_speed"] + num_cols].dropna()
X_dys_m = dys_model[num_cols].astype(float)
y_dys_m = dys_model["log_speed"].values

print("\n=== SmartAdditiveRegressor (dyslexic subset) ===")
smart_dys = SmartAdditiveRegressor()
smart_dys.fit(X_dys_m, y_dys_m)
print(smart_dys)

print("\n=== HingeGAMRegressor (dyslexic subset) ===")
hinge_dys = HingeGAMRegressor()
hinge_dys.fit(X_dys_m, y_dys_m)
print(hinge_dys)

# ── 6. Summary statistics for conclusion ─────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Bivariate p (dyslexic, reader_view effect on log_speed): {p_biv_dys:.4f}")
print(f"Controlled OLS p (dyslexic only): {p_val_controlled_dys:.4f}, coef={coef_controlled_dys:.4f}")
interaction_p = ols_full.pvalues.get("rv_x_dys", float("nan"))
interaction_coef = ols_full.params.get("rv_x_dys", float("nan"))
print(f"Interaction (reader_view × dyslexia) in full model: coef={interaction_coef:.4f}, p={interaction_p:.4f}")

rv_coef_full = ols_full.params["reader_view"]
rv_p_full = ols_full.pvalues["reader_view"]
print(f"Main reader_view effect (full model): coef={rv_coef_full:.4f}, p={rv_p_full:.4f}")

# ── 7. Write conclusion ───────────────────────────────────────────────────────
import json

# Interpretation logic:
# - Does reader_view improve speed for DYSLEXIC individuals?
# - Check: bivariate effect in dyslexic subset, controlled OLS in dyslexic subset,
#   interaction term in full model
# - Corroborate with interpretable model feature importance

sig_biv = p_biv_dys < 0.05
sig_controlled = p_val_controlled_dys < 0.05
sig_interaction = interaction_p < 0.05
positive_direction = coef_controlled_dys > 0

# Calibrate score
if sig_controlled and positive_direction and sig_biv:
    base_score = 70
elif sig_controlled and not sig_biv:
    base_score = 55
elif sig_biv and not sig_controlled:
    base_score = 45
elif not sig_controlled and not sig_biv:
    base_score = 20
else:
    base_score = 30

# Adjust for interaction significance
if sig_interaction:
    base_score = min(base_score + 10, 100)

# Adjust for direction
if not positive_direction:
    base_score = max(5, base_score - 30)

score = int(base_score)

explanation = (
    f"The research question asks whether Reader View improves reading speed for individuals with dyslexia. "
    f"Bivariate t-test on dyslexic readers: t={t_stat:.3f}, p={p_biv_dys:.4f}. "
    f"Controlled OLS (dyslexic subset, controlling for age, device, education, English nativity, retake, Flesch-Kincaid, num_words): "
    f"reader_view coef={coef_controlled_dys:.4f}, p={p_val_controlled_dys:.4f}. "
    f"Interaction term (reader_view × dyslexia_bin) in full model: coef={interaction_coef:.4f}, p={interaction_p:.4f}. "
    f"The SmartAdditiveRegressor and HingeGAMRegressor were fitted on both the full dataset and dyslexic subset — "
    f"their printed forms show whether reader_view receives nonzero importance/coefficient. "
    f"A positive, statistically significant controlled effect in dyslexic readers {'was' if sig_controlled else 'was NOT'} found "
    f"(p={'<0.05' if sig_controlled else '>0.05'}). "
    f"The interaction term {'was' if sig_interaction else 'was NOT'} significant. "
    f"Direction is {'positive (Reader View increases speed)' if positive_direction else 'negative or null (Reader View does not increase speed)'}. "
    f"Based on these results, the Likert score is calibrated to {score}/100."
)

result = {"response": score, "explanation": explanation}
print("\nConclusion:", json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclustion.txt written successfully.")
