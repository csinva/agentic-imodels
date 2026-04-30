
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('caschools.csv')

# Create the student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Create the academic performance score
df['score'] = (df['read'] + df['math']) / 2

# Prepare the data for regression
X = df[['str', 'english', 'lunch', 'income']]
X = sm.add_constant(X)
y = df['score']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the coefficient for the student-teacher ratio
str_coef = model.params['str']
p_value = model.pvalues['str']

# Determine the response
explanation = f"The coefficient for the student-teacher ratio is {str_coef:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05 and str_coef < 0:
    response = 90
    explanation += "This indicates a statistically significant negative relationship between student-teacher ratio and academic performance. Lower student-teacher ratios are associated with higher scores."
elif p_value < 0.05 and str_coef > 0:
    response = 10
    explanation += "This indicates a statistically significant positive relationship between student-teacher ratio and academic performance. Higher student-teacher ratios are associated with higher scores, which is counter-intuitive."
else:
    response = 50
    explanation += "The relationship is not statistically significant, so we cannot conclude that there is a strong association between student-teacher ratio and academic performance."

# Write the conclusion to a file
conclusion = {'response': response, 'explanation': explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
