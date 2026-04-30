
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('teachingratings.csv')

# The research question is: "What is the impact of beauty on teaching evaluations received by teachers?"
# We will use a linear regression model to answer this question.
# The dependent variable is 'eval' (teaching evaluation).
# The independent variable of interest is 'beauty'.
# We will also include other variables as controls.

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['minority', 'gender', 'credits', 'division', 'native', 'tenure'], drop_first=True)

# Define the dependent and independent variables
y = df['eval']
X = df.drop(columns=['eval', 'rownames', 'prof'])

# Convert boolean columns to integers
for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get the coefficient and p-value for the 'beauty' variable
beauty_coef = model.params['beauty']
beauty_pvalue = model.pvalues['beauty']

# Determine the response based on the p-value
if beauty_pvalue < 0.05:
    # If the p-value is significant, there is a relationship.
    # The score will be based on the magnitude of the coefficient.
    # A positive coefficient means a positive impact, so a higher score.
    if beauty_coef > 0:
        response = 75 + min(25, int(beauty_coef * 100))
    else:
        response = 25 - min(25, int(abs(beauty_coef) * 100))
    explanation = f"There is a statistically significant relationship between beauty and teaching evaluations (p-value: {beauty_pvalue:.4f}). The coefficient for beauty is {beauty_coef:.4f}, suggesting that for each unit increase in beauty, the teaching evaluation is expected to {'increase' if beauty_coef > 0 else 'decrease'} by {abs(beauty_coef):.4f}, holding all other factors constant."
else:
    # If the p-value is not significant, there is no relationship.
    response = 0
    explanation = f"There is no statistically significant relationship between beauty and teaching evaluations (p-value: {beauty_pvalue:.4f})."


# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
