
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('hurricane.csv')

# The research question is whether hurricanes with more feminine names (higher masfem) 
# are perceived as less threatening, leading to more deaths.

# We will test the relationship between 'masfem' and 'alldeaths'.
# We should control for other factors that could influence the number of deaths,
# such as the intensity of the hurricane (e.g., 'min_pressure', 'wind', 'category')
# and the damage caused ('ndam').

# For simplicity and to directly address the research question, let's start with a
# simple linear regression model.
# y = alldeaths
# x = masfem

# However, a multiple regression model will provide a more robust answer by
# controlling for confounding variables.
# Let's include 'category', 'ndam', and 'min' (min pressure) as control variables.
# Note that 'min' is a measure of pressure, so lower is more intense.
# We will use 'category' as a proxy for intensity.

# Drop rows with missing values in the relevant columns
df.dropna(subset=['masfem', 'category', 'ndam', 'alldeaths'], inplace=True)

X = df[['masfem', 'category', 'ndam']]
y = df['alldeaths']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Get the p-value for the 'masfem' coefficient
p_value_masfem = model.pvalues['masfem']

# The coefficient for 'masfem'
coef_masfem = model.params['masfem']

# We will consider a p-value less than 0.05 to be statistically significant.
is_significant = p_value_masfem < 0.05

# The research question suggests a positive correlation between masfem and alldeaths
# (more feminine name -> more deaths).
# So we check if the coefficient is positive.
is_positive_correlation = coef_masfem > 0

if is_significant and is_positive_correlation:
    # Strong "Yes"
    response = 90
    explanation = f"The regression model shows a statistically significant (p={p_value_masfem:.3f}) and positive (coefficient={coef_masfem:.3f}) relationship between the femininity of a hurricane's name and the number of deaths, even after controlling for the hurricane's category and the damage caused. This supports the hypothesis."
else:
    # Strong "No"
    response = 10
    explanation = f"The regression model does not show a statistically significant (p={p_value_masfem:.3f}) or the expected positive relationship between the femininity of a hurricane's name and the number of deaths. The coefficient for 'masfem' was {coef_masfem:.3f}. This does not support the hypothesis."


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
print(model.summary())
