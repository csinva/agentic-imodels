
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('caschools.csv')

# Create student-teacher ratio
df['student_teacher_ratio'] = df['students'] / df['teachers']

# Create composite academic performance score
df['academic_performance'] = (df['read'] + df['math']) / 2

# Fit a linear regression model
X = df[['student_teacher_ratio']]
y = df['academic_performance']
X = sm.add_constant(X)  # Add a constant to the model
model = sm.OLS(y, X).fit()

# Get the p-value for the student-teacher ratio coefficient
p_value = model.pvalues['student_teacher_ratio']

# Determine the response based on the p-value
# We can be confident in the relationship if p < 0.05
if p_value < 0.05 and model.params['student_teacher_ratio'] < 0:
    # A significant negative coefficient means lower ratio is associated with higher performance
    response = 90  # Strong "Yes"
    explanation = f"A statistically significant negative relationship (p={p_value:.3f}) was found between student-teacher ratio and academic performance. Lower student-teacher ratios are associated with higher scores."
elif p_value < 0.05 and model.params['student_teacher_ratio'] >= 0:
    response = 10 # Strong "No"
    explanation = f"A statistically significant positive relationship (p={p_value:.3f}) was found. This suggests lower student-teacher ratios are associated with lower scores, which is counter-intuitive."
else:
    response = 20  # Leaning "No"
    explanation = f"No statistically significant relationship (p={p_value:.3f}) was found between student-teacher ratio and academic performance. The observed relationship could be due to chance."


# Create the conclusion file
conclusion = {'response': response, 'explanation': explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
