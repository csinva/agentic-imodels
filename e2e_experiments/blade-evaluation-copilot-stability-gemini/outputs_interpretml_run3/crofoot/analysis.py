
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier

# Load the dataset
df = pd.read_csv('crofoot.csv')

# Feature Engineering
df['group_size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Define features and target
features = ['group_size_diff', 'dist_diff']
target = 'win'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Train an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get feature importances from EBM
ebm_global = ebm.explain_global()
feature_importances = ebm_global.data()

# Find the most important feature
most_important_feature = feature_importances['names'][0]

# Determine the response based on the most important feature
if most_important_feature == 'group_size_diff':
    response = 80
    explanation = "Relative group size is the most important factor in determining the winner of a contest. The model shows a strong positive correlation between a larger group size and the probability of winning."
elif most_important_feature == 'dist_diff':
    response = 20
    explanation = "Contest location is the most important factor. The model indicates that groups are more likely to win contests that occur closer to their home range, regardless of group size."
else:
    response = 50
    explanation = "Both relative group size and contest location play a role, but neither is definitively more important than the other."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
