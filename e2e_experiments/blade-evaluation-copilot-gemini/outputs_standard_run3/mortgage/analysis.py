
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from imodels import RuleFitClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('mortgage.csv')

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0'])

# Handle missing values
data = data.dropna()

# Define features (X) and target (y)
X = data.drop(columns=['deny', 'accept'])
y = data['deny']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Interpretable Models ---

# 1. Logistic Regression with statsmodels for p-values
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
p_value_gender = result.pvalues['female']

# --- Conclusion ---
# Based on the p-value from the logistic regression, we can assess the statistical significance
# of gender on mortgage denial. A low p-value (typically < 0.05) suggests a significant relationship.

# Determine the response based on the p-value
if p_value_gender < 0.05:
    response = 90  # Strong "Yes"
    explanation = f"There is a statistically significant relationship between gender and mortgage denial (p-value: {p_value_gender:.4f}). The logistic regression model shows that the 'female' variable is a significant predictor of the outcome."
else:
    response = 10  # Strong "No"
    explanation = f"There is no statistically significant relationship between gender and mortgage denial (p-value: {p_value_gender:.4f}). The 'female' variable is not a significant predictor in the logistic regression model. This suggests that, based on this dataset and model, gender does not have a discernible effect on mortgage approval."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
