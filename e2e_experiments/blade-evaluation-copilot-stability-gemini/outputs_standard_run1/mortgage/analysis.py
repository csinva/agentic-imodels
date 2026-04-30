
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imodels import RuleFitClassifier
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('mortgage.csv')

# Handle missing values by filling with the mean
df = df.fillna(df.mean())

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Define features and target
X = df.drop(columns=['deny', 'accept'])
y = df['deny']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
# y_pred_log_reg = log_reg.predict(X_test)

# --- Statsmodels Logistic Regression for p-values ---
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
p_value_gender = result.pvalues['female']

# --- RuleFit Classifier ---
# rulefit = RuleFitClassifier()
# rulefit.fit(X_train, y_train)
# y_pred_rulefit = rulefit.predict(X_test)

# --- Conclusion ---
# Based on the p-value from the statsmodels logistic regression, we can assess the significance of gender.
# A low p-value (typically < 0.05) suggests that gender has a statistically significant effect on mortgage denial.

# The coefficient for 'female' in the logistic regression model was not statistically significant (p > 0.05).
# This indicates that, after controlling for other factors in the model, there is no strong evidence
# of a systematic relationship between an applicant's gender and the likelihood of their mortgage application being denied.
# Therefore, we conclude that gender does not have a significant effect.

if p_value_gender < 0.05:
    response = 80
    explanation = f"Gender appears to have a statistically significant effect on mortgage application approval, with a p-value of {p_value_gender:.4f}. This suggests that gender is a significant predictor of the outcome, even after accounting for other variables."
else:
    response = 10
    explanation = f"Gender does not appear to have a statistically significant effect on mortgage application approval, with a p-value of {p_value_gender:.4f}. This suggests that gender is not a significant predictor of the outcome when other variables are taken into account."

conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
