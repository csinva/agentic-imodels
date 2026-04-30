import numpy as np
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
player_skin_tone = df.groupby('playerShort')['dark_skin'].first().reset_index()

player_data = pd.merge(player_red_cards, player_skin_tone, on='playerShort')

# Logistic Regression
X = player_data[['dark_skin']]
y = (player_data['redCards'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Statistical Analysis
logit_model = sm.Logit(y, sm.add_constant(X))
result = logit_model.fit()

# Interpretation
p_value = result.pvalues['dark_skin']
odds_ratio = np.exp(result.params['dark_skin'])

if p_value < 0.05:
    if odds_ratio > 1:
        response = 80
        explanation = f"There is a statistically significant relationship (p={p_value:.3f}) between skin tone and red cards. Players with darker skin tones have {odds_ratio:.2f} times higher odds of receiving a red card."
    else:
        response = 20
        explanation = f"There is a statistically significant relationship (p={p_value:.3f}), but the effect is in the opposite direction. Players with darker skin tones have {odds_ratio:.2f} times lower odds of receiving a red card."
else:
    response = 10
    explanation = f"There is no statistically significant relationship (p={p_value:.3f}) between skin tone and red cards."

# Save conclusion
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
