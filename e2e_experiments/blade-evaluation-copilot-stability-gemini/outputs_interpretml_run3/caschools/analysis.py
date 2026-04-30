
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load data
df = pd.read_csv('caschools.csv')

# Create student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Create composite academic performance score
df['performance'] = (df['read'] + df['math']) / 2

# Research question: Is a lower student-teacher ratio associated with higher academic performance?

# Using statsmodels for regression analysis to get p-values
X = df['str']
y = df['performance']
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(y, X).fit()
p_value = model.pvalues['str']
coefficient = model.params['str']

# Interpretation
# A negative coefficient means that as the student-teacher ratio decreases, performance increases.
# A small p-value (typically < 0.05) suggests that this relationship is statistically significant.

is_significant = p_value < 0.05
is_negative_correlation = coefficient < 0

if is_significant and is_negative_correlation:
    # Strong evidence for the hypothesis
    response = 90
    explanation = f"Yes, there is a statistically significant negative correlation (p-value: {p_value:.4f}, coefficient: {coefficient:.4f}) between student-teacher ratio and academic performance. Lower student-teacher ratios are associated with higher scores."
elif is_significant and not is_negative_correlation:
    # Significant, but in the opposite direction
    response = 10
    explanation = f"The relationship is statistically significant (p-value: {p_value:.4f}), but the correlation is positive (coefficient: {coefficient:.4f}). This suggests lower student-teacher ratios are associated with lower scores, which is counter-intuitive and may indicate confounding factors."
else:
    # Not statistically significant
    response = 20
    explanation = f"The relationship is not statistically significant (p-value: {p_value:.4f}). There is not enough evidence to conclude that student-teacher ratio has a meaningful impact on academic performance in this dataset."


# Save conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
