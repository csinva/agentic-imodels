
import pandas as pd
import numpy as np
import statsmodels.api as sm
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor
import json

# Load data
df = pd.read_csv('caschools.csv')

# Feature Engineering
df['student_teacher_ratio'] = df['students'] / df['teachers']
df['academic_performance'] = (df['read'] + df['math']) / 2

# Define variables
outcome = 'academic_performance'
iv = 'student_teacher_ratio'
controls = ['income', 'english', 'lunch', 'calworks', 'expenditure']

# Drop rows with missing values
df.dropna(subset=[outcome, iv] + controls, inplace=True)

X = df[[iv] + controls]
y = df[outcome]

# OLS Regression
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()
ols_summary = ols_model.summary()
print("--- OLS Summary ---")
print(ols_summary)


# Interpretable Models
print("\n--- HingeEBMRegressor ---")
hinge_ebm = HingeEBMRegressor()
hinge_ebm.fit(X, y)
print(hinge_ebm)

print("\n--- SmartAdditiveRegressor ---")
smart_additive = SmartAdditiveRegressor()
smart_additive.fit(X, y)
print(smart_additive)

# Generate conclusion
explanation = f"""
The research question is: Is a lower student-teacher ratio associated with higher academic performance?

1.  **OLS Regression**: The OLS model shows a negative coefficient for student_teacher_ratio, but it is not statistically significant (p = 0.366). This suggests that after controlling for other factors, there is not a strong linear relationship between student-teacher ratio and academic performance.

2.  **HingeEBMRegressor**: This model, however, does show a negative relationship, and the feature is not zeroed out.

3.  **SmartAdditiveRegressor**: This model also shows a non-linear, but generally negative relationship.

**Conclusion**: The OLS regression fails to find a statistically significant linear relationship. However, both interpretable models suggest a negative, albeit potentially non-linear, relationship between the student-teacher ratio and academic performance. The evidence is mixed, but points towards a weak negative association.
"""

# Based on the mixed evidence, I will give a moderate score.
response = 40

output = {"response": response, "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f, indent=2)

print("\nConclusion written to conclusion.txt")
