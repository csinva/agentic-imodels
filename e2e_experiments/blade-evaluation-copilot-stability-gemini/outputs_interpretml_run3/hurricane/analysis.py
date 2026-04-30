
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('hurricane.csv')

# Prepare the data
X = df[['masfem']]
y = df['alldeaths']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Get the p-value for the 'masfem' coefficient
p_value = model.pvalues['masfem']

# Define a significance level
alpha = 0.05

# Determine the response based on the p-value
if p_value < alpha:
    # Significant relationship
    response = 90
    explanation = f"There is a statistically significant relationship (p={p_value:.3f}) between the femininity of a hurricane's name and the number of deaths. More feminine names are associated with more deaths."
else:
    # No significant relationship
    response = 10
    explanation = f"There is no statistically significant relationship (p={p_value:.3f}) between the femininity of a hurricane's name and the number of deaths."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
