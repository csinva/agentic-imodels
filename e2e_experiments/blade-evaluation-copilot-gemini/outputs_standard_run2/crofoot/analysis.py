
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from imodels import RuleFitClassifier

# Load the dataset
data = pd.read_csv('crofoot.csv')

# Create interaction terms and other features
data['group_size_diff'] = data['n_focal'] - data['n_other']
data['dist_diff'] = data['dist_focal'] - data['dist_other']

# Define features (X) and target (y)
features = ['group_size_diff', 'dist_diff']
X = data[features]
y = data['win']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Logistic Regression with statsmodels for p-values ---
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
# print(result.summary())

# --- Interpretable Model with RuleFit ---
# Train a RuleFit model
rulefit = RuleFitClassifier()
rulefit.fit(X_train.values, y_train, feature_names=features)

# --- Interpretation and Conclusion ---
# Based on the statsmodels summary, we can check the p-values
p_values = result.pvalues
group_size_p_value = p_values['group_size_diff']
dist_diff_p_value = p_values['dist_diff']

# A common threshold for significance is 0.05
is_group_size_significant = group_size_p_value < 0.05
is_dist_diff_significant = dist_diff_p_value < 0.05

# Formulate a response
explanation = f"The p-value for relative group size is {group_size_p_value:.3f} and for contest location (distance difference) is {dist_diff_p_value:.3f}. "
response = 0

if is_group_size_significant and is_dist_diff_significant:
    explanation += "Both relative group size and contest location are statistically significant predictors of winning an intergroup contest."
    response = 95
elif is_group_size_significant:
    explanation += "Only relative group size is a statistically significant predictor."
    response = 70
elif is_dist_diff_significant:
    explanation += "Only contest location is a statistically significant predictor."
    response = 70
else:
    explanation += "Neither relative group size nor contest location are statistically significant predictors."
    response = 10


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
