
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('fish.csv')

# Calculate fish caught per hour, handling cases where hours is zero to avoid division by zero
df['fish_caught_per_hour'] = df['fish_caught'] / df['hours'].replace(0, 0.001)

# Define the independent variables (features) and the dependent variable (target)
features = ['livebait', 'camper', 'persons', 'child']
X = df[features]
y = df['fish_caught_per_hour']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Get the model summary
summary = model.summary()

# Extract p-values to determine significance
p_values = model.pvalues

# Determine which factors are significant (p-value < 0.05)
significant_factors = p_values[p_values < 0.05].index.tolist()

# Create the explanation
explanation = f"The analysis aimed to identify factors influencing the rate of fish caught per hour. An Ordinary Least Squares (OLS) regression was performed with 'fish_caught_per_hour' as the dependent variable and '{', '.join(features)}' as independent variables. "

if 'const' in significant_factors:
    significant_factors.remove('const')

if significant_factors:
    explanation += f"The statistically significant factors (p < 0.05) are: {', '.join(significant_factors)}. "
    explanation += "These factors have a demonstrable effect on the number of fish caught per hour. "
    response = 80 # Strong "Yes"
else:
    explanation += "No factors were found to be statistically significant. "
    explanation += "This suggests that, based on the provided data, there is no strong evidence that these factors influence the fishing rate."
    response = 20 # Weak "No"


# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
