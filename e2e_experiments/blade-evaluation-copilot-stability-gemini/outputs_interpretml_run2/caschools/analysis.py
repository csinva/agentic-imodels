
import pandas as pd
import json
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv('caschools.csv')

# Feature Engineering
df['student_teacher_ratio'] = df['students'] / df['teachers']
df['academic_performance'] = (df['read'] + df['math']) / 2

# Clean data - drop rows with missing values that could affect the model
df = df.dropna(subset=['student_teacher_ratio', 'academic_performance', 'income', 'english', 'calworks'])

# Define features and target
features = ['student_teacher_ratio', 'income', 'english', 'calworks']
target = 'academic_performance'

X = df[features]
y = df[target]

# Add a constant for the intercept term for statsmodels
X_sm = sm.add_constant(X)

# OLS Regression
model = sm.OLS(y, X_sm).fit()
p_value = model.pvalues['student_teacher_ratio']
coefficient = model.params['student_teacher_ratio']

# Interpret results
explanation = f"The OLS regression coefficient for student_teacher_ratio is {coefficient:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05 and coefficient < 0:
    explanation += "This indicates a statistically significant negative relationship between student-teacher ratio and academic performance. Lower ratios are associated with higher scores."
    response = 90  # Strong "Yes"
elif p_value < 0.05 and coefficient > 0:
    explanation += "This indicates a statistically significant positive relationship, which is counterintuitive. Higher ratios are associated with higher scores."
    response = 10 # Strong "No"
else:
    explanation += "The relationship is not statistically significant, so we cannot conclude that a lower student-teacher ratio is associated with higher performance."
    response = 20 # Leaning "No"

# EBM for further insight
ebm = ExplainableBoostingRegressor(interactions=0)
ebm.fit(X, y)
ebm_local = ebm.explain_local(X)

# Check the mean effect of student_teacher_ratio from EBM
mean_effect = pd.Series(ebm_local.data(0)['scores']).mean()
explanation += f" The Explainable Boosting Regressor shows an average effect of {mean_effect:.4f} for the student-teacher ratio."


# Write conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
