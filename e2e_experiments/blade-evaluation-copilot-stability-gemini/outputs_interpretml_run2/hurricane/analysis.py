
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('hurricane.csv')

# The research question is whether hurricanes with more feminine names cause more deaths.
# We will use a regression model to test this.
# The dependent variable is 'alldeaths'.
# The independent variable of interest is 'masfem'.
# We should also control for other factors that might affect the number of deaths,
# such as the severity of the hurricane.
# We will use 'min' (minimum pressure) and 'wind' and 'category' as control variables.

# Define the variables for the regression
y = df['alldeaths']
X = df[['masfem', 'min', 'wind', 'category']]

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the summary of the model
summary = model.summary()

# The coefficient for 'masfem' tells us the relationship between the femininity of a hurricane's name and the number of deaths.
# A positive coefficient would support the hypothesis.
# The p-value for this coefficient tells us if the relationship is statistically significant.

# Extract the coefficient and p-value for 'masfem'
masfem_coef = model.params['masfem']
masfem_pvalue = model.pvalues['masfem']

# We will base our conclusion on the statistical significance of the 'masfem' coefficient.
# If the p-value is less than 0.05, we will consider the result significant.
if masfem_pvalue < 0.05 and masfem_coef > 0:
    response = 90
    explanation = f"The regression model shows a statistically significant positive relationship between the femininity of a hurricane's name and the number of deaths (p-value: {masfem_pvalue:.4f}). This supports the hypothesis that hurricanes with more feminine names are perceived as less threatening, leading to more fatalities."
else:
    response = 10
    explanation = f"The regression model does not show a statistically significant positive relationship between the femininity of a hurricane's name and the number of deaths (p-value: {masfem_pvalue:.4f}). This does not support the hypothesis."

# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
print(summary)
