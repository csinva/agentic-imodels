
import json
import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('teachingratings.csv')

# Prepare the data
y = df['eval']
X = df[['beauty']]
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Get the p-value for the 'beauty' coefficient
p_value = model.pvalues['beauty']

# Determine the response based on the p-value
if p_value < 0.05:
    # Strong "Yes" if statistically significant
    response = 90
    explanation = f"A statistically significant relationship (p={p_value:.3f}) was found between beauty and teaching evaluations. The model suggests that higher beauty ratings are associated with higher teaching evaluations."
else:
    # Leaning "No" if not statistically significant
    response = 20
    explanation = f"No statistically significant relationship (p={p_value:.3f}) was found between beauty and teaching evaluations. The model does not provide strong evidence that beauty impacts teaching evaluations."

# Create the conclusion file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
