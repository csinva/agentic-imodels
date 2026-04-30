
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imodels import RuleFitClassifier
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('crofoot.csv')

# Feature Engineering
df['relative_group_size'] = df['n_focal'] - df['n_other']
df['relative_dist_from_home'] = df['dist_focal'] - df['dist_other']

# Define features and target
features = ['relative_group_size', 'relative_dist_from_home']
target = 'win'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Interpretable Models ---

# 1. Logistic Regression (from scikit-learn)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_preds)

# 2. RuleFit Classifier (from imodels)
rulefit = RuleFitClassifier()
rulefit.fit(X_train, y_train)
rulefit_preds = rulefit.predict(X_test)
rulefit_accuracy = accuracy_score(y_test, rulefit_preds)

# --- Statistical Tests ---

# Logistic Regression with statsmodels for p-values
X_sm = sm.add_constant(X)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit()
p_values = result.pvalues

# --- Interpretation ---

# Analyze the coefficients from the logistic regression model
log_reg_coefs = pd.DataFrame(log_reg.coef_, columns=features, index=['Coefficient']).T

# Analyze the rules from the RuleFit model
# rules = rulefit.get_rules()
# # Filter for rules with non-zero importance
# important_rules = rules[rules.importance > 0].sort_values("importance", ascending=False)


# --- Conclusion ---
# Based on the p-values from the statsmodels logistic regression,
# we can determine the significance of the features.
# A p-value less than 0.05 is typically considered statistically significant.

explanation = "Based on the analysis: "
response = 50  # Default to a neutral score

# Check significance of relative_group_size
if p_values['relative_group_size'] < 0.05:
    explanation += "Relative group size is a significant predictor of winning a contest. "
    if log_reg_coefs.loc['relative_group_size', 'Coefficient'] > 0:
        explanation += "Larger relative group size increases the probability of winning. "
        response += 25
    else:
        explanation += "Smaller relative group size increases the probability of winning. "
        response -= 25
else:
    explanation += "Relative group size is not a statistically significant predictor. "


# Check significance of relative_dist_from_home
if p_values['relative_dist_from_home'] < 0.05:
    explanation += "Contest location (relative distance from home range) is a significant predictor. "
    if log_reg_coefs.loc['relative_dist_from_home', 'Coefficient'] < 0:
        explanation += "Contests closer to the focal group's home range increase their probability of winning."
        response += 25
    else:
        explanation += "Contests further from the focal group's home range increase their probability of winning."
        response -= 25
else:
    explanation += "Contest location is not a statistically significant predictor. "

# Clamp response to be between 0 and 100
response = max(0, min(100, response))


conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
