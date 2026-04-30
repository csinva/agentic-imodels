
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from imodels import RuleFitRegressor
import json
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv('caschools.csv')

# Create student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Academic performance is the average of read and math scores
df['academic_performance'] = (df['read'] + df['math']) / 2

# Check for correlation between student-teacher ratio and academic performance
corr, p_value = pearsonr(df['str'], df['academic_performance'])

# Fit a linear regression model
X = df[['str']]
y = df['academic_performance']
model = LinearRegression()
model.fit(X, y)
coef = model.coef_[0]

# Fit a RuleFit model for interpretability
rulefit = RuleFitRegressor()
# rulefit.fit(X, y)
# rules = rulefit.get_rules()

# Interpretation
explanation = f"The correlation between student-teacher ratio and academic performance is {corr:.3f} with a p-value of {p_value:.3f}. "
explanation += f"The linear regression coefficient is {coef:.3f}, suggesting that for each unit increase in student-teacher ratio, academic performance changes by {coef:.3f}. "

if p_value < 0.05 and coef < 0:
    response = 80  # Strong "Yes"
    explanation += "There is a statistically significant negative relationship, supporting the hypothesis that a lower student-teacher ratio is associated with higher academic performance."
elif p_value < 0.05 and coef > 0:
    response = 20  # Strong "No"
    explanation += "There is a statistically significant positive relationship, which contradicts the hypothesis."
else:
    response = 50  # Neutral
    explanation += "The relationship is not statistically significant, so we cannot conclude that there is a strong association."

# Save the conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
