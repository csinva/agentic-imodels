
import json
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# 1. Load and preprocess data
df = pd.read_csv('teachingratings.csv')
df = pd.get_dummies(df, drop_first=True, dtype=float) # Ensure dummy columns are float

# Define variables
TARGET = 'eval'
FEATURE_OF_INTEREST = 'beauty'
# From info.json, 'prof' is an identifier, and 'rownames' is like an index.
# 'allstudents' is highly correlated with 'students'. Let's use 'students'.
CONTROLS = [col for col in df.columns if col not in [TARGET, FEATURE_OF_INTEREST, 'prof', 'rownames', 'allstudents']]
X = df[[FEATURE_OF_INTEREST] + CONTROLS]
y = df[TARGET]

# Convert all columns to numeric, coercing errors
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop rows with NaN values that might have been introduced
X = X.dropna()
y = y[X.index] # Align y with X

# 2. Classical statistical test (OLS)
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()
ols_summary = ols_model.summary()
print("--- OLS Results ---")
print(ols_summary)
print("\n")

# 3. Interpretable models
print("--- Interpretable Models ---")
# SmartAdditiveRegressor
sa_model = SmartAdditiveRegressor().fit(X, y)
print("--- SmartAdditiveRegressor ---")
print(sa_model)
print("\n")

# HingeEBMRegressor
hebm_model = HingeEBMRegressor().fit(X, y)
print("--- HingeEBMRegressor ---")
print(hebm_model)
print("\n")

# 4. Synthesize results and conclude
# OLS results: beauty coefficient is 0.1330, p-value is 0.000. Significant positive effect.
# SmartAdditiveRegressor: beauty is the 2nd most important feature (19.8%). The effect is linear.
# HingeEBMRegressor: beauty is the most important feature. The effect is also linear.

explanation = (
    "The analysis consistently shows a statistically significant and positive relationship between a teacher's beauty rating and their teaching evaluation score. "
    "An OLS regression, controlling for factors like age, gender, and course characteristics, finds that a one-unit increase in the beauty score is associated with a 0.133 point increase in the evaluation score (p < 0.001). "
    "This finding is corroborated by two interpretable models. The SmartAdditiveRegressor identifies 'beauty' as the second most important feature, contributing 19.8% of the model's output, with a linear positive effect. "
    "Similarly, the HingeEBMRegressor, a more complex model, also ranks 'beauty' as the most important predictor with a linear positive relationship. "
    "The consistency and strength of this effect across different models provide strong evidence for the impact of beauty on teaching evaluations."
)

# Based on SKILL.md scoring: Strong significant effect that persists across models and is top-ranked in importance -> 75-100
response_score = 85

conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. Conclusion written to conclusion.txt")
