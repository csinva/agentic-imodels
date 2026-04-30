
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load data
df = pd.read_csv('boxes.csv')

# Preprocess data
# Create the target variable: 1 if the child chose the majority option, 0 otherwise
df['chose_majority'] = (df['y'] == 2).astype(int)

# One-hot encode the 'culture' feature
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
culture_encoded = encoder.fit_transform(df[['culture']])
culture_df = pd.DataFrame(culture_encoded, columns=encoder.get_feature_names_out(['culture']))
df = pd.concat([df.drop('culture', axis=1), culture_df], axis=1)

# Define features (X) and target (y)
features = ['age', 'gender', 'majority_first'] + list(culture_df.columns)
X = df[features]
y = df['chose_majority']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42, interactions=10)
ebm.fit(X_train, y_train)

# Get global feature importances
global_explanation = ebm.explain_global()
feature_importances = global_explanation.data()

# Find the importance of the 'age' feature and any interactions involving 'age' and 'culture'
age_importance = 0
age_culture_interaction_importance = 0
explanation_text = "The analysis focused on how age and culture influence a child's tendency to choose the majority option. "

for name, score in zip(feature_importances['names'], feature_importances['scores']):
    if name == 'age':
        age_importance = score
        explanation_text += f"Age is a significant predictor, with an importance score of {age_importance:.4f}. "
    # Interactions are strings with ' & '
    elif isinstance(name, str) and 'age' in name and 'culture' in name:
        age_culture_interaction_importance += score

explanation_text += f"The interaction between age and culture has a combined importance score of {age_culture_interaction_importance:.4f}. "

# Determine the response score based on the interaction importance
# A higher interaction score suggests that the development of majority preference with age differs across cultures.
# We can scale the interaction importance to a 0-100 score.
# The total importance is 1. Let's consider an interaction importance > 0.1 as strong.
response_score = min(100, int((age_culture_interaction_importance / 0.1) * 100)) if age_culture_interaction_importance > 0 else 0


if response_score > 50:
    explanation_text += "This indicates a notable interaction effect, suggesting that the way children's reliance on majority preference changes with age is indeed different across various cultural contexts. "
else:
    explanation_text += "This indicates a weak interaction effect, suggesting that the developmental trajectory of majority preference reliance is relatively consistent across the studied cultures. "

# Create the conclusion file
conclusion = {
    "response": response_score,
    "explanation": explanation_text
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
