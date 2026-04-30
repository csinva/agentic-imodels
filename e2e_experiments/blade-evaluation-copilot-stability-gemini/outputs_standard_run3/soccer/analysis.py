
import pandas as pd
import statsmodels.api as sm
import json

# Load data
df = pd.read_csv('soccer.csv')

# Preprocess data
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df.dropna(subset=['skin_tone'], inplace=True)

# Define player groups
dark_skin_threshold = df['skin_tone'].median()
df['skin_tone_group'] = (df['skin_tone'] > dark_skin_threshold).astype(int)

# Aggregate red cards per player
player_red_cards = df.groupby('playerShort').agg({
    'redCards': 'sum',
    'skin_tone_group': 'first'
}).reset_index()

# Logistic regression model
X = player_red_cards['skin_tone_group']
y = (player_red_cards['redCards'] > 0).astype(int)
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()
p_value = model.pvalues['skin_tone_group']
significant = p_value < 0.05

# Determine response
if significant:
    response = 85
    explanation = "A statistically significant relationship (p < 0.05) was found between darker skin tone and the likelihood of receiving red cards, suggesting potential bias."
else:
    response = 15
    explanation = "No statistically significant relationship was found between skin tone and the likelihood of receiving red cards."

# Create conclusion file
with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)
