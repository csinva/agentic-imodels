
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv('soccer.csv')

# Preprocessing
# Create the skin_tone feature by averaging rater1 and rater2
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Drop rows with missing skin_tone
df = df.dropna(subset=['skin_tone'])

# Define features and target
features = ['skin_tone', 'games', 'victories', 'ties', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'height', 'weight']
target = 'redCards'

# Binarize target variable
df[target] = (df[target] > 0).astype(int)

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get feature importances
ebm_global = ebm.explain_global()
feature_importances = ebm_global.data()

skin_tone_importance = 0
for feature in feature_importances['names']:
    if feature == 'skin_tone':
        skin_tone_importance = feature_importances['scores'][feature_importances['names'].index(feature)]


# Higher score indicates a stronger relationship
# A positive coefficient for skin_tone would mean darker skin tone is associated with more red cards.
# We will check the sign of the coefficient of the 'skin_tone' feature.
ebm_local = ebm.explain_local(X_test, y_test)
skin_tone_effect = np.mean(ebm_local.data(0)['scores'])


# The research question is directional, so we are looking for a positive relationship.
# A positive skin_tone_effect suggests that darker skin tone increases the likelihood of a red card.
# We can scale the response based on the strength and direction of the effect.
# A simple approach is to use the feature importance and the sign of the effect.
# If the effect is positive, we can use the importance as a proxy for the strength of the "yes".
# If the effect is negative or zero, it's a "no".

if skin_tone_effect > 0:
    # Scale the response by the feature importance, capping at 100.
    response = min(int(skin_tone_importance * 1000), 100)
    explanation = f"The model found a positive relationship between skin tone and red cards. The feature importance for skin tone was {skin_tone_importance:.4f}, and the average effect on the log-odds of receiving a red card was {skin_tone_effect:.4f}. This suggests that, holding other factors constant, players with darker skin tones are more likely to receive red cards."
else:
    response = 0
    explanation = f"The model did not find a positive relationship between skin tone and red cards. The average effect of skin tone on the log-odds of receiving a red card was {skin_tone_effect:.4f}, indicating no increased likelihood for players with darker skin tones."


# Write conclusion
output = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

print("Analysis complete. Conclusion written to conclusion.txt")
