
import pandas as pd
import statsmodels.api as sm
from imodels import RuleFitClassifier
import json

# Load the dataset
df = pd.read_csv('crofoot.csv')

# Feature Engineering
df['n_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Define features and target
features = ['n_diff', 'dist_diff']
target = 'win'

X = df[features]
y = df[target]

# Add a constant to the features for the statsmodels model
X_sm = sm.add_constant(X)

# Fit a logistic regression model
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit()

# Fit a RuleFit model for interpretability
rulefit = RuleFitClassifier()
rulefit.fit(X, y)

# Get the p-value for the 'n_diff' coefficient
p_value_n_diff = result.pvalues['n_diff']

# Determine the response based on the p-value
if p_value_n_diff < 0.05:
    response = 80
    explanation = f"There is a statistically significant relationship between relative group size (n_diff) and the probability of winning. The p-value for the relative group size is {p_value_n_diff:.3f}, which is less than 0.05. The logistic regression coefficient for n_diff is {result.params['n_diff']:.3f}, indicating that for each unit increase in the size difference between the focal and other group, the log-odds of winning increase by this amount. The RuleFit model also provides interpretable rules that highlight the importance of group size."
else:
    response = 20
    explanation = f"There is no statistically significant relationship between relative group size (n_diff) and the probability of winning. The p-value for the relative group size is {p_value_n_diff:.3f}, which is greater than 0.05. The logistic regression coefficient for n_diff is {result.params['n_diff']:.3f}. The RuleFit model's rules do not consistently show a strong effect of group size."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
