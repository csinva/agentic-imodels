
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

# Aggregate red cards per player
player_red_cards = df.groupby('playerShort')['redCards'].sum().reset_index()
player_dark_skin = df.groupby('playerShort')['dark_skin'].first().reset_index()

# Merge aggregated data
player_data = pd.merge(player_red_cards, player_dark_skin, on='playerShort')

# Add other potentially relevant player-level features
player_data = pd.merge(player_data, df[['playerShort', 'games', 'victories', 'defeats', 'goals', 'yellowCards', 'position']].groupby('playerShort').mean(numeric_only=True).reset_index(), on='playerShort')

# Drop rows with missing values
player_data.dropna(inplace=True)

# Define features (X) and target (y)
X = player_data[['dark_skin', 'games', 'victories', 'defeats', 'goals', 'yellowCards']]
y = (player_data['redCards'] > 0).astype(int) # Binary target: 1 if player has ever received a red card, 0 otherwise

# Add a constant for the intercept term
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# Get the p-value for the 'dark_skin' coefficient
p_value = result.pvalues['dark_skin']

# Determine the response based on the p-value
if p_value < 0.05:
    response = 90  # Strong "Yes"
    explanation = f"There is a statistically significant relationship (p={p_value:.4f}) between a player having a darker skin tone and receiving more red cards. The logistic regression model shows that darker skin tone is a significant predictor of receiving a red card."
else:
    response = 10  # Strong "No"
    explanation = f"There is no statistically significant relationship (p={p_value:.4f}) between a player having a darker skin tone and receiving more red cards. The logistic regression model does not show that darker skin tone is a significant predictor of receiving a red card."

# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
