
import pandas as pd
import json
import statsmodels.api as sm

# Load data
df = pd.read_csv('teachingratings.csv')

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define independent and dependent variables
X = df.drop(['eval', 'rownames', 'prof'], axis=1)

# Convert boolean columns to integers
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)

X = sm.add_constant(X)
y = df['eval']

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Get the coefficient for beauty
beauty_coef = model.params['beauty']
p_value = model.pvalues['beauty']

# Determine the response
if p_value < 0.05 and beauty_coef > 0:
    response = 85
    explanation = f"A statistically significant positive relationship was found between beauty and teaching evaluations (coefficient: {beauty_coef:.3f}, p-value: {p_value:.3f}). This suggests that instructors with higher beauty ratings tend to receive higher teaching evaluations."
else:
    response = 15
    explanation = f"No statistically significant positive relationship was found between beauty and teaching evaluations (coefficient: {beauty_coef:.3f}, p-value: {p_value:.3f})."

# Create conclusion file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
