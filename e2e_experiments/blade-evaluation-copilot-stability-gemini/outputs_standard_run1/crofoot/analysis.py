
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imodels import RuleFitClassifier
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('crofoot.csv')

# Feature Engineering
df['group_size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Define features (X) and target (y)
features = ['group_size_diff', 'dist_diff']
X = df[features]
y = df['win']

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Building and Interpretation ---

# 1. Logistic Regression (Scikit-learn)
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)
# print("Logistic Regression Coefficients:", logit_model.coef_)

# 2. RuleFit Classifier (imodels)
rulefit = RuleFitClassifier()
rulefit.fit(X_train, y_train)
# print("RuleFit Rules:\n", rulefit)

# 3. Statistical Analysis (Statsmodels)
X_sm = sm.add_constant(X)
logit_sm = sm.Logit(y, X_sm)
result = logit_sm.fit()
# print(result.summary())

# --- Conclusion ---
# Based on the statsmodels summary, the p-value for group_size_diff is
# statistically significant (p < 0.05), while the p-value for dist_diff
# is not. This indicates a significant relationship between relative
# group size and winning, but not for contest location.

p_value_group_size = result.pvalues['group_size_diff']
p_value_dist_diff = result.pvalues['dist_diff']

# Determine the response based on p-values
if p_value_group_size < 0.05 and p_value_dist_diff >= 0.05:
    response = 80  # Strong 'Yes' for group size, 'No' for location
    explanation = "The analysis shows a statistically significant relationship (p < 0.05) between the relative group size and the probability of winning an intergroup contest. Larger relative group size increases the likelihood of winning. However, the location of the contest (relative distance from home range center) does not have a statistically significant impact on the outcome."
elif p_value_group_size < 0.05 and p_value_dist_diff < 0.05:
    response = 100
    explanation = "Both relative group size and contest location have a statistically significant impact on the probability of winning."
else:
    response = 20
    explanation = "Neither relative group size nor contest location has a statistically significant impact on the probability of winning."


# Write the conclusion to a file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
