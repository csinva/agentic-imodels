
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
try:
    df = pd.read_csv('caschools.csv')
except FileNotFoundError:
    print("Error: caschools.csv not found. Make sure the file is in the correct directory.")
    exit()

# Create the student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Create an academic performance score as the average of read and math scores
df['academic_performance'] = (df['read'] + df['math']) / 2

# Define the independent variable (X) and the dependent variable (y)
X = df[['str']]
y = df['academic_performance']

# Add a constant to the independent variable
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get the p-value for the student-teacher ratio coefficient
p_value = model.pvalues['str']

# Determine the response based on the p-value
# A low p-value (e.g., < 0.05) indicates a statistically significant relationship.
# The coefficient will tell us the direction.
if p_value < 0.05 and model.params['str'] < 0:
    # Significant negative correlation, so lower STR is associated with higher performance
    response = 90
    explanation = "A statistically significant negative relationship was found between student-teacher ratio and academic performance (p-value < 0.05). This indicates that a lower student-teacher ratio is associated with higher academic performance."
elif p_value < 0.05 and model.params['str'] > 0:
    # Significant positive correlation
    response = 10
    explanation = "A statistically significant positive relationship was found between student-teacher ratio and academic performance (p-value < 0.05). This indicates that a higher student-teacher ratio is associated with higher academic performance, which is counter-intuitive."
else:
    # Not a significant relationship
    response = 10
    explanation = "No statistically significant relationship was found between student-teacher ratio and academic performance (p-value >= 0.05)."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
