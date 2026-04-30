
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('hurricane.csv')

# Drop rows with missing values in the columns we are using
df.dropna(subset=['alldeaths', 'masfem', 'category', 'ndam', 'wind'], inplace=True)

# The research question is whether hurricanes with more feminine names cause more deaths.
# We will test this by regressing 'alldeaths' on 'masfem' and other relevant covariates.
# 'masfem' is a scale from 1 (very masculine) to 11 (very feminine).
# A positive coefficient for 'masfem' would support the hypothesis.

# Define the dependent and independent variables
y = df['alldeaths']
X = df[['masfem', 'category', 'ndam', 'wind']]

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Get the p-value for the 'masfem' coefficient
p_value_masfem = model.pvalues['masfem']

# The coefficient for 'masfem'
coef_masfem = model.params['masfem']

# We will consider a p-value less than 0.05 to be statistically significant.
significant = p_value_masfem < 0.05

# Formulate the explanation
explanation = f"The research question is whether more feminine hurricane names are associated with more deaths. "
explanation += f"A regression of 'alldeaths' on 'masfem' (femininity of name), controlling for 'category', 'ndam' (damage), and 'wind' was performed. "
explanation += f"The coefficient for 'masfem' was {coef_masfem:.4f} with a p-value of {p_value_masfem:.4f}. "

if significant and coef_masfem > 0:
    response = 80  # Strong "Yes"
    explanation += "This indicates a statistically significant positive relationship, suggesting that more feminine-named hurricanes are associated with more deaths."
elif significant and coef_masfem < 0:
    response = 20  # Strong "No"
    explanation += "This indicates a statistically significant negative relationship, suggesting that more masculine-named hurricanes are associated with more deaths."
else:
    response = 10  # Strong "No"
    explanation += "The relationship is not statistically significant, so we cannot conclude that there is a relationship between the femininity of a hurricane's name and the number of deaths."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
