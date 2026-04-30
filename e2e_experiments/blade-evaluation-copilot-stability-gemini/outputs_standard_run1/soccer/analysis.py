
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('soccer.csv')

# Preprocessing
# Create a binary skin tone variable
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df['dark_skin'] = (df['skin_tone'] > df['skin_tone'].median()).astype(int)

# Aggregate data to player level
player_level_data = df.groupby('playerShort').agg({
    'dark_skin': 'mean',
    'redCards': 'sum',
    'games': 'sum',
    'victories': 'sum',
    'defeats': 'sum',
    'goals': 'sum',
    'yellowCards': 'sum',
    'yellowReds': 'sum',
    'height': 'mean',
    'weight': 'mean',
}).reset_index()

# Dependent variable: whether a player has received at least one red card
player_level_data['has_red_card'] = (player_level_data['redCards'] > 0).astype(int)

# Independent variables
features = ['dark_skin', 'games', 'victories', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'height', 'weight']
X = player_level_data[features]
y = player_level_data['has_red_card']

# Drop rows with missing values
X = X.dropna()
y = y[X.index]


# Add a constant for the intercept
X = sm.add_constant(X)

# Logistic Regression with statsmodels
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Get the p-value for the 'dark_skin' coefficient
p_value_dark_skin = result.pvalues['dark_skin']

# Determine the response based on the p-value
if p_value_dark_skin < 0.05:
    # Significant relationship
    response = 80
    explanation = f"There is a statistically significant relationship (p={p_value_dark_skin:.3f}) between skin tone and the likelihood of receiving a red card. The logistic regression model shows that players with darker skin tones are more likely to receive red cards."
else:
    # No significant relationship
    response = 20
    explanation = f"There is no statistically significant relationship (p={p_value_dark_skin:.3f}) between skin tone and the likelihood of receiving a red card. The logistic regression model does not provide enough evidence to support the claim that players with darker skin tones are more likely to receive red cards."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
