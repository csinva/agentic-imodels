
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
try:
    df = pd.read_csv('hurricane.csv')
except FileNotFoundError:
    print("Error: hurricane.csv not found. Make sure the data file is in the same directory.")
    exit()

# The research question is whether hurricanes with more feminine names cause more deaths.
# The 'masfem' variable measures the masculinity-femininity of a name, where a higher value is more feminine.
# The 'alldeaths' variable is the number of deaths.
# We need to control for other factors that affect the number of deaths, such as the intensity of the hurricane.
# We will use 'category', 'wind', 'min_pressure', and 'ndam' as control variables.

# Prepare the data for regression
# We need to handle missing values if any. For simplicity, we'll drop rows with missing values in the relevant columns.
df_clean = df[['alldeaths', 'masfem', 'category', 'wind', 'min', 'ndam']].dropna()

# Define the dependent and independent variables
y = df_clean['alldeaths']
X = df_clean[['masfem', 'category', 'wind', 'min', 'ndam']]

# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Get the p-value for the 'masfem' coefficient
masfem_p_value = model.pvalues['masfem']

# Interpret the results
# A low p-value (typically < 0.05) suggests that the femininity of a hurricane's name is a statistically significant predictor of the number of deaths.
is_significant = masfem_p_value < 0.05

if is_significant:
    response = 90  # Strong "Yes"
    explanation = f"The analysis shows a statistically significant relationship (p-value = {masfem_p_value:.4f}) between the femininity of a hurricane's name ('masfem') and the number of deaths ('alldeaths'), even after controlling for the hurricane's intensity (category, wind speed, pressure, and damages). The positive coefficient for 'masfem' suggests that more feminine-named hurricanes are associated with more deaths, supporting the hypothesis."
else:
    response = 10  # Strong "No"
    explanation = f"The analysis did not find a statistically significant relationship (p-value = {masfem_p_value:.4f}) between the femininity of a hurricane's name ('masfem') and the number of deaths ('alldeaths') when controlling for the hurricane's intensity. This suggests that the perceived femininity of a name does not have a statistically significant impact on the death toll."

# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. conclusion.txt has been created.")
print(model.summary())

