
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
try:
    df = pd.read_csv('hurricane.csv')
except FileNotFoundError:
    print("Error: hurricane.csv not found. Make sure the file is in the correct directory.")
    exit()

# The research question is whether more feminine names lead to more deaths.
# This implies we are looking for a positive correlation between 'masfem' and 'alldeaths'.

# Select variables for the regression model.
# Dependent variable: 'alldeaths'
# Independent variable of interest: 'masfem'
# Control variables: 'category', 'min_pressure', 'wind' to account for storm intensity.
# 'ndam' (damage) is also a potential outcome variable, not a cause, so it's excluded.
# We will use 'min' for minimum pressure.
features = ['masfem', 'category', 'min', 'wind']
target = 'alldeaths'

# Create the final dataframe for modeling
df_model = df[features + [target]].copy()

# Handle missing values - 'alldeaths' for Katrina is missing.
# Given the context of the original study, this is a significant outlier.
# The original paper imputed this value, but a simple approach is to drop it
# to avoid making assumptions. The row for Katrina has a missing 'ndam' as well.
df_model.dropna(inplace=True)


# Define the independent variables (X) and the dependent variable (y)
X = df_model[features]
y = df_model[target]

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()

# Get the coefficient for 'masfem'
masfem_coef = model.params['masfem']
p_value = model.pvalues['masfem']

# Interpret the results
# A positive coefficient for 'masfem' would suggest that as the name becomes more feminine (higher masfem score),
# the number of deaths increases.
# The p-value tells us if this finding is statistically significant.
is_significant = p_value < 0.05

explanation = (
    f"The analysis aimed to determine if hurricanes with more feminine names cause more deaths. "
    f"A multiple linear regression model was built to predict 'alldeaths' using 'masfem' (femininity score) "
    f"while controlling for storm intensity (category, min pressure, wind speed). "
    f"The coefficient for 'masfem' was {masfem_coef:.4f} with a p-value of {p_value:.4f}. "
)

if is_significant and masfem_coef > 0:
    response = 85
    explanation += (
        "The result is statistically significant (p < 0.05) and the coefficient is positive. "
        "This indicates a positive correlation: hurricanes with more feminine-rated names are associated with more deaths, "
        "supporting the hypothesis. The effect size should be considered, but the relationship is statistically present."
    )
elif is_significant and masfem_coef <= 0:
    response = 10
    explanation += (
        "The result is statistically significant, but the coefficient is not positive. "
        "This suggests a relationship exists, but it does not support the original hypothesis that more feminine names lead to more deaths. "
        "In fact, it might suggest the opposite or no meaningful relationship in the hypothesized direction."
    )
else: # Not significant
    response = 5
    explanation += (
        "The result is not statistically significant (p >= 0.05). "
        "Therefore, we cannot conclude that there is a reliable relationship between the femininity of a hurricane's name and the number of deaths it causes based on this data. "
        "The null hypothesis (no effect) cannot be rejected."
    )


# Create the conclusion.txt file
output = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(output, f, indent=2)

print("Analysis complete. conclusion.txt has been created.")
print(model.summary())

