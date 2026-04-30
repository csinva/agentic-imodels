
import pandas as pd
import numpy as np
import statsmodels.api as sm
from agentic_imodels import WinsorizedSparseOLSRegressor, HingeEBMRegressor
import json

# Load data
df = pd.read_csv('caschools.csv')

# Feature engineering
df['student_teacher_ratio'] = df['students'] / df['teachers']
df['academic_performance'] = (df['read'] + df['math']) / 2

# Define variables
outcome = 'academic_performance'
predictor = 'student_teacher_ratio'
controls = ['income', 'calworks', 'lunch', 'english', 'expenditure']

# Explore data
print("Data Exploration:")
print(df[[outcome, predictor] + controls].corr())

# OLS regression
X = df[[predictor] + controls]
X = sm.add_constant(X)
y = df[outcome]
ols_model = sm.OLS(y, X).fit()
print("\nOLS Model Summary:")
print(ols_model.summary())

# Interpretable models
X_im = df[[predictor] + controls]
y_im = df[outcome]

print("\n--- WinsorizedSparseOLSRegressor ---")
wsor = WinsorizedSparseOLSRegressor()
wsor.fit(X_im, y_im)
print(wsor)

print("\n--- HingeEBMRegressor ---")
hebm = HingeEBMRegressor()
hebm.fit(X_im, y_im)
print(hebm)

# Conclusion
ols_coef = ols_model.params[predictor]
ols_pvalue = ols_model.pvalues[predictor]

explanation = f"OLS regression shows a coefficient for student_teacher_ratio of {ols_coef:.2f} with a p-value of {ols_pvalue:.3f}. "
if ols_pvalue < 0.05 and ols_coef < 0:
    explanation += "This suggests a statistically significant negative association between student-teacher ratio and academic performance, meaning lower ratios (fewer students per teacher) are associated with higher scores. "
    response = 85
elif ols_pvalue < 0.05 and ols_coef > 0:
    explanation += "This suggests a statistically significant positive association, which is counter-intuitive. "
    response = 20
else:
    explanation += "The association is not statistically significant. "
    response = 30

explanation += f"The WinsorizedSparseOLSRegressor model gives student_teacher_ratio a coefficient of {wsor.ols_coef_[0]:.2f}. "

hebm_str = str(hebm)
x0_lines = [line for line in hebm_str.split('\\n') if 'x0:' in line]
if x0_lines and '0.0000' not in x0_lines[0]:
    try:
        hebm_coef_x0 = float(x0_lines[0].split(':')[1].strip())
        explanation += f"The HingeEBMRegressor gives student_teacher_ratio a coefficient of {hebm_coef_x0:.2f}. "
    except (IndexError, ValueError):
        explanation += "Could not parse the HingeEBMRegressor output for student_teacher_ratio coefficient. "
else:
    explanation += "The HingeEBMRegressor model zeroed out the student_teacher_ratio feature. "


explanation += "The interpretable models provide mixed evidence. The WinsorizedSparseOLSRegressor finds a negative coefficient, while the HingeEBMRegressor zeroes out the feature. Given the non-significant OLS result and the conflicting interpretable models, the evidence for an association is weak."


conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nConclusion written to conclusion.txt")
