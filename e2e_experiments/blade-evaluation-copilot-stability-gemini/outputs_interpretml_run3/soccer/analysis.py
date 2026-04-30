
import pandas as pd
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('soccer.csv')

# Preprocessing
# Create an average skin tone rating
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Drop rows with missing skin tone ratings
df.dropna(subset=['skin_tone'], inplace=True)

# Define the dependent and independent variables
y = df['redCards']
X = df[['skin_tone', 'games', 'victories', 'defeats', 'goals', 'yellowCards', 'yellowReds']]

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Fit a Poisson regression model
# Poisson is suitable for count data like number of red cards
poisson_model = sm.Poisson(y, X).fit()

# Get the summary of the model
summary = poisson_model.summary()

# Extract the p-value for the skin_tone coefficient
p_value_skin_tone = poisson_model.pvalues['skin_tone']

# Interpret the results
# A low p-value (typically < 0.05) indicates a statistically significant relationship.
if p_value_skin_tone < 0.05:
    response = 90  # Strong "Yes"
    explanation = f"There is a statistically significant relationship (p-value: {p_value_skin_tone:.4f}) between skin tone and the number of red cards received. The model suggests that players with darker skin tones are more likely to receive red cards."
else:
    response = 10  # Strong "No"
    explanation = f"There is no statistically significant relationship (p-value: {p_value_skin_tone:.4f}) between skin tone and the number of red cards received. The model does not provide evidence to support the claim that players with darker skin tones are more likely to receive red cards."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
