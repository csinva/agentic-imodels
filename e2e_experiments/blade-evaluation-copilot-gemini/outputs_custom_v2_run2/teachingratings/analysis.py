
import json
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
)
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('teachingratings.csv')

# Drop unnecessary columns
df = df.drop(columns=['rownames', 'prof'])

# One-hot encode categorical features
categorical_features = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Convert boolean columns to integers
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Define variables
outcome_variable = 'eval'
independent_variable = 'beauty'
control_variables = [col for col in df.columns if col not in [outcome_variable, independent_variable]]

X = df[control_variables + [independent_variable]]
y = df[outcome_variable]

# Add a constant for the intercept
X_with_const = sm.add_constant(X)

# OLS regression
ols_model = sm.OLS(y, X_with_const).fit()
ols_summary = ols_model.summary()
print("--- OLS Summary ---")
print(ols_summary)


# Interpretable models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- SmartAdditiveRegressor ---")
smart_additive_model = SmartAdditiveRegressor()
smart_additive_model.fit(X_train, y_train)
print(smart_additive_model)


print("\n--- HingeEBMRegressor ---")
hinge_ebm_model = HingeEBMRegressor()
hinge_ebm_model.fit(X_train, y_train)
print(hinge_ebm_model)

# Conclusion
# The OLS regression shows a statistically significant positive relationship
# between beauty and teaching evaluations (p < 0.001), even after controlling for other factors.
# The coefficient for beauty is 0.1330.
# The interpretable models also show that beauty is an important feature.
# In the SmartAdditiveRegressor, beauty is the second most important feature.
# In the HingeEBMRegressor, beauty is also a significant factor.
# The effect appears to be roughly linear.

explanation = (
    "The analysis indicates a clear and statistically significant positive relationship between a teacher's perceived beauty and their teaching evaluations. "
    "An Ordinary Least Squares (OLS) regression, controlling for factors like age, gender, and tenure, shows that for every one-unit increase in the beauty score, the teaching evaluation increases by approximately 0.13 points (p < 0.001). "
    "This finding is supported by interpretable machine learning models. Both the SmartAdditiveRegressor and HingeEBMRegressor identify 'beauty' as one of the most important predictors of teaching evaluations. "
    "The relationship appears to be generally linear, meaning the effect is consistent across the range of beauty scores. "
    "Given the consistent, statistically significant, and robust positive effect across multiple models, there is strong evidence to support the claim that beauty has a positive impact on teaching evaluations."
)

# Based on the SKILL.md scoring guidelines, a strong significant effect that persists across models and is top-ranked in importance should get a score of 75-100.
conclusion = {
    "response": 85,
    "explanation": explanation,
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("\nConclusion written to conclusion.txt")
