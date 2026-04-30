
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('mortgage.csv')

# Drop rows with any missing values
df = df.dropna()

# --- 1. Data Exploration ---
# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# --- 2. Statistical Tests ---
# Separate approved and denied applications by gender
male_approved = df[(df['female'] == 0) & (df['deny'] == 0)]
female_approved = df[(df['female'] == 1) & (df['deny'] == 0)]
male_denied = df[(df['female'] == 0) & (df['deny'] == 1)]
female_denied = df[(df['female'] == 1) & (df['deny'] == 1)]

# Calculate approval rates
male_approval_rate = len(male_approved) / (len(male_approved) + len(male_denied))
female_approval_rate = len(female_approved) / (len(female_approved) + len(female_denied))

# Perform t-test on approval rates
deny_by_gender = df.groupby('female')['deny'].value_counts(normalize=True).unstack()
ttest_result = ttest_ind(df[df['female']==0]['deny'], df[df['female']==1]['deny'])


# --- 3. Interpretable Models ---
# Logistic Regression with statsmodels for p-values
X = df.drop(['deny', 'accept'], axis=1)
# one-hot encode categorical features
X = pd.get_dummies(X, columns=['mortgage_credit', 'consumer_credit'], drop_first=True)
y = df['deny']
X = sm.add_constant(X)
logit_model = sm.Logit(y, X.astype(float))
result = logit_model.fit()
p_value_gender = result.pvalues['female']
coefficient_gender = result.params['female']


# --- 4. Interpretation and Conclusion ---
# The p-value from the logistic regression for the 'female' coefficient is the most direct test.
# A small p-value (e.g., < 0.05) would indicate a statistically significant relationship.
# The t-test on denial rates is also informative.

# Based on the p-value, decide the response.
# If p > 0.05, there is no significant effect.
if p_value_gender > 0.05:
    response = 10 # Very weak relationship
    explanation = f"The p-value for gender in the logistic regression model is {p_value_gender:.3f}, which is not statistically significant at the 0.05 level. This suggests that gender does not have a significant impact on mortgage denial, after controlling for other factors. The coefficient for female was {coefficient_gender:.3f}, indicating a very small effect. The t-test p-value of {ttest_result.pvalue:.3f} also supports this conclusion."
else:
    response = 90 # Strong relationship
    explanation = f"The p-value for gender in the logistic regression model is {p_value_gender:.3f}, which is statistically significant at the 0.05 level. This suggests that gender has a significant impact on mortgage denial, after controlling for other factors. The coefficient for female was {coefficient_gender:.3f}. The t-test p-value of {ttest_result.pvalue:.3f} also supports this conclusion."


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
print(f"P-value for gender: {p_value_gender}")
print(f"T-test result: {ttest_result}")
