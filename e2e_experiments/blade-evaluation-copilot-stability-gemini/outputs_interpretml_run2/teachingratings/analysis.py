
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the data
df = pd.read_csv('teachingratings.csv')

# The research question is about the impact of 'beauty' on 'eval'
# I will fit a linear model to see the relationship.
# I will include other variables as controls.

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['minority', 'gender', 'credits', 'division', 'native', 'tenure'], drop_first=True)

# Define the variables for the model
X = df[['beauty', 'age', 'students', 'allstudents', 'minority_yes', 'gender_male', 'credits_single', 'division_upper', 'native_yes', 'tenure_yes']]
y = df['eval']

# Convert boolean columns to integers
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Get the coefficient for 'beauty' and its p-value
beauty_coef = model.params['beauty']
beauty_pvalue = model.pvalues['beauty']

# Interpret the results
explanation = f"The coefficient for beauty is {beauty_coef:.4f} with a p-value of {beauty_pvalue:.4f}. "
if beauty_pvalue < 0.05:
    explanation += "This indicates a statistically significant positive relationship between beauty and teaching evaluations. As the beauty score increases, the teaching evaluation tends to increase as well."
    response = 80  # Strong "Yes"
else:
    explanation += "This indicates that there is no statistically significant relationship between beauty and teaching evaluations."
    response = 10  # Strong "No"

# Save the conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
