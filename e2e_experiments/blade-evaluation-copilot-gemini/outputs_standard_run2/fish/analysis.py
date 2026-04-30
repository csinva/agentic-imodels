
import pandas as pd
import json
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('fish.csv')

# Create the target variable: fish caught per hour
# Add a small epsilon to hours to avoid division by zero
df['fish_per_hour'] = df['fish_caught'] / (df['hours'] + 1e-6)

# Define features (X) and target (y)
features = ['livebait', 'camper', 'persons', 'child']
X = df[features]
y = df['fish_per_hour']

# Add a constant to the features
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Get the average fish per hour
average_fish_per_hour = df['fish_per_hour'].mean()

# Interpret the results
explanation = f"The average number of fish caught per hour is {average_fish_per_hour:.2f}. "
explanation += "To understand the factors influencing this, a linear regression was performed. "
explanation += "The model summary is as follows: \n" + str(model.summary())

# Based on the p-values from the regression, we can determine which factors are significant.
# A p-value less than 0.05 is typically considered significant.
significant_factors = model.pvalues[model.pvalues < 0.05].index.tolist()

if 'const' in significant_factors:
    significant_factors.remove('const')

if significant_factors:
    explanation += f"\nThe significant factors influencing the number of fish caught per hour are: {', '.join(significant_factors)}. "
    response = 80 # Strong "Yes" that there are influencing factors
else:
    explanation += "\nNo factors were found to be statistically significant in influencing the number of fish caught per hour. "
    response = 20 # Weak "No"

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)
