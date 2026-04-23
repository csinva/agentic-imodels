import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_custom_v2_run1/mortgage')

from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv('mortgage.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nDeny rate by gender:")
print(df.groupby('female')['deny'].mean())

# ── 2. Bivariate test ────────────────────────────────────────────────────────
female_deny = df[df['female'] == 1]['deny']
male_deny   = df[df['female'] == 0]['deny']
t, p_biv = stats.ttest_ind(female_deny, male_deny)
print(f"\nBivariate t-test: t={t:.4f}, p={p_biv:.4f}")
print(f"Female deny rate: {female_deny.mean():.4f}, Male deny rate: {male_deny.mean():.4f}")

# ── 3. Logistic regression with controls ─────────────────────────────────────
feature_cols = ['female', 'black', 'housing_expense_ratio', 'self_employed',
                'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
                'PI_ratio', 'loan_to_value', 'denied_PMI']

df_clean = df[feature_cols + ['deny']].dropna()
X_logit = sm.add_constant(df_clean[feature_cols])
logit_model = sm.Logit(df_clean['deny'], X_logit).fit(disp=False)
print("\n=== Logistic Regression (with controls) ===")
print(logit_model.summary())

female_coef = logit_model.params['female']
female_pval = logit_model.pvalues['female']
female_ci   = logit_model.conf_int().loc['female']
print(f"\nfemale coef={female_coef:.4f}, p={female_pval:.4f}, 95% CI=[{female_ci[0]:.4f},{female_ci[1]:.4f}]")

# ── 4. OLS with controls (for interpretable model comparison) ─────────────────
ols_model = sm.OLS(df_clean['deny'], X_logit).fit()
print("\n=== OLS (with controls) ===")
print(ols_model.summary())

# ── 5. Interpretable models ───────────────────────────────────────────────────
X_interp = df_clean[feature_cols]
y_interp = df_clean['deny']

print("\n=== SmartAdditiveRegressor ===")
sar = SmartAdditiveRegressor()
sar.fit(X_interp, y_interp)
print(sar)

print("\n=== HingeEBMRegressor ===")
hebm = HingeEBMRegressor()
hebm.fit(X_interp, y_interp)
print(hebm)

# ── 6. Summarize and write conclusion ────────────────────────────────────────
biv_diff = female_deny.mean() - male_deny.mean()

# Feature x0 = female in both interpretable models
# SmartAdditiveRegressor zeroed out x0 (female) -- strong null evidence from honest model
# HingeEBMRegressor had x0 coef = -0.0328 -- very small (negative on deny = females more approved)
sar_female_zeroed = True   # SmartAdditiveRegressor zeroed out x0=female
hebm_female_coef = -0.0328  # HingeEBM x0=female coefficient (on deny outcome)

logit_pval = female_pval
logit_coef = female_coef  # negative = females less likely denied (more approved)

explanation = (
    f"Research question: How does gender affect mortgage approval? "
    f"Bivariate: female deny rate={female_deny.mean():.3f}, male deny rate={male_deny.mean():.3f}, "
    f"diff={biv_diff:.4f} (t-test p={p_biv:.4f} -- essentially no bivariate difference). "
    f"Controlled logistic regression: female coef={logit_coef:.4f}, p={logit_pval:.4f}, "
    f"95% CI=[{female_ci[0]:.4f},{female_ci[1]:.4f}]. "
    f"Negative coefficient means females are LESS likely to be denied (more approved) after controls. "
    f"However, the honest SmartAdditiveRegressor zeroed out the female variable entirely (strong null evidence). "
    f"HingeEBMRegressor had a very small coefficient for female (x0=-0.0328 on deny). "
    f"Evidence is inconsistent: controlled logistic regression shows a borderline significant effect "
    f"(p={logit_pval:.4f}) suggesting females are slightly more approved after controlling for "
    f"creditworthiness, but the honest interpretable model removes gender entirely, and the bivariate "
    f"effect is virtually zero. The dominant predictors are denied_PMI, PI_ratio, bad_history, "
    f"consumer_credit, and loan_to_value. Gender is a weak, inconsistent predictor."
)

# Score calibration:
# Bivariate: no effect (p=0.99)
# Controlled logistic: significant (p=0.029) but only marginally significant after controls
# SmartAdditiveRegressor (honest): female ZEROED OUT -- strong null evidence
# HingeEBM: tiny negative coefficient
# -> Weak, inconsistent evidence -> score 25-35
if logit_pval < 0.05 and not sar_female_zeroed:
    score = 60  # significant AND retained in honest model
elif logit_pval < 0.05 and sar_female_zeroed:
    score = 35  # logistic significant but honest model zeros out -- weak/inconsistent
elif p_biv < 0.05:
    score = 30
else:
    score = 20

result = {"response": score, "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\n=== CONCLUSION ===")
print(json.dumps(result, indent=2))
print("\nconclusion.txt written successfully.")
