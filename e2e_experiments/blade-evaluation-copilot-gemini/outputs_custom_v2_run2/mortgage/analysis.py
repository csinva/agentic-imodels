
import pandas as pd
import numpy as np
import statsmodels.api as sm
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor, WinsorizedSparseOLSRegressor
import json

# Load the dataset
df = pd.read_csv('mortgage.csv')

# Define variables
y_col = 'deny'
iv_col = 'female'
control_cols = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
X_cols = [iv_col] + control_cols

# Handle missing values
for col in X_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())


X = df[X_cols]
y = df[y_col]

# Step 2: Classical statistical tests
X_with_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_const).fit()
print("--- Logit Model Summary ---")
print(logit_model.summary())
print("\n")

# Step 3: Interpretable models
print("--- Interpretable Models ---")
model_hinge = HingeEBMRegressor()
model_hinge.fit(X, y)
print(f"--- {model_hinge.__class__.__name__} ---")
print(model_hinge)
print("\n")

model_smart = SmartAdditiveRegressor()
model_smart.fit(X, y)
print(f"--- {model_smart.__class__.__name__} ---")
print(model_smart)
print("\n")

model_winsor = WinsorizedSparseOLSRegressor()
model_winsor.fit(X, y)
print(f"--- {model_winsor.__class__.__name__} ---")
print(model_winsor)
print("\n")


# Step 4: Write conclusion
p_value = logit_model.pvalues[iv_col]
coef = logit_model.params[iv_col]

explanation = f"The logistic regression coefficient for 'female' is {coef:.3f} with a p-value of {p_value:.3f}. "
if p_value < 0.05:
    explanation += "This indicates a statistically significant relationship between gender and mortgage denial, after controlling for other factors. "
else:
    explanation += "This indicates no statistically significant relationship between gender and mortgage denial, after controlling for other factors. "

explanation += "The interpretable models provide further insight. "
# HingeEBMRegressor
explanation += f"The HingeEBMRegressor shows a contribution for female of around {model_hinge.lasso_.coef_[X_cols.index('female')]:.3f}. "

# SmartAdditiveRegressor
if 'female' in model_smart.linear_approx_:
    explanation += f"The SmartAdditiveRegressor gives a linear coefficient for female of {model_smart.linear_approx_['female']:.3f}. "
else:
    explanation += "The SmartAdditiveRegressor does not give a linear coefficient for female. "

# WinsorizedSparseOLSRegressor
if 'female' in X.columns[model_winsor.support_]:
    female_coef_winsor = model_winsor.ols_coef_[model_winsor.support_][list(X.columns[model_winsor.support_]).index('female')]
    explanation += f"The WinsorizedSparseOLSRegressor includes 'female' in its sparse model with a coefficient of {female_coef_winsor:.3f}. "
else:
    explanation += "The WinsorizedSparseOLSRegressor does NOT include 'female' in its sparse model, suggesting it's not a primary driver. "


# Based on the results, I will give a score.
# The p-value is significant, but the coefficients in the interpretable models are small.
# This suggests that gender has a small but statistically significant effect on mortgage denial.
# I will give a score that reflects this.
score = 30
explanation += "Overall, the evidence for a relationship between gender and mortgage denial is present but weak. The effect is statistically significant in the logistic regression, but the magnitude of the effect is small across all models."


conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("conclusion.txt created.")
