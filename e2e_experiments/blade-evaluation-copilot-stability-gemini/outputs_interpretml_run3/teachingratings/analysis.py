
import json
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('teachingratings.csv')

# Prepare data for modeling
# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['minority', 'gender', 'credits', 'division', 'native', 'tenure'], drop_first=True)

# Define independent and dependent variables
X = df.drop(['eval', 'rownames', 'prof'], axis=1)
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)
X = sm.add_constant(X)  # Add a constant for the intercept
y = df['eval']

# Build and fit the model
model = sm.OLS(y, X).fit()

# Get the coefficient for 'beauty'
beauty_coef = model.params['beauty']
p_value = model.pvalues['beauty']

# Determine the response
# A low p-value (typically < 0.05) indicates a statistically significant relationship.
if p_value < 0.05:
    # A positive coefficient means higher beauty is associated with higher evaluations.
    if beauty_coef > 0:
        response = 95  # Strong "Yes"
        explanation = f"A statistically significant positive relationship was found between beauty and teaching evaluations (p-value: {p_value:.4f}). The coefficient for beauty was {beauty_coef:.4f}, suggesting that a one-unit increase in the beauty score is associated with a {beauty_coef:.4f}-unit increase in the teaching evaluation score, holding other factors constant."
    else:
        response = 5  # Strong "No" but for a negative relationship
        explanation = f"A statistically significant negative relationship was found between beauty and teaching evaluations (p-value: {p_value:.4f}). The coefficient for beauty was {beauty_coef:.4f}, suggesting that higher beauty scores are associated with lower teaching evaluations."
else:
    response = 10  # Strong "No"
    explanation = f"No statistically significant relationship was found between beauty and teaching evaluations (p-value: {p_value:.4f}). The coefficient for beauty was {beauty_coef:.4f}, which is not statistically different from zero."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. Conclusion written to conclusion.txt")
