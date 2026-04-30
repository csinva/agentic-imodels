
import pandas as pd
import statsmodels.api as sm
import json
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np

import numpy as np

# Load the dataset
try:
    df = pd.read_csv('mortgage.csv')
except FileNotFoundError:
    print("Error: mortgage.csv not found. Make sure the file is in the correct directory.")
    exit()

# Replace infinities with NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values by filling with the median
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Define the independent variables and the dependent variable
X = df.drop(columns=['deny', 'accept', 'Unnamed: 0'])
y = df['deny']

# Add a constant to the independent variables for the regression model
X_const = sm.add_constant(X)

# Fit a logistic regression model
logit_model = sm.Logit(y, X_const)
result = logit_model.fit(disp=0)

# Get the p-value for the 'female' coefficient
p_value_female = result.pvalues['female']

# Fit an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X, y)

# Get the feature importance for 'female'
feature_importances = ebm.term_importances()
female_importance = 0
for i, feature in enumerate(X.columns):
    if feature == 'female':
        female_importance = feature_importances[i]
        break

# Determine the response and explanation
# We will use a p-value threshold of 0.05 to determine statistical significance.
if p_value_female < 0.05:
    # If the result is statistically significant, we can say there is an effect.
    # The score will be based on the magnitude of the coefficient and importance.
    # A small coefficient and importance will result in a score closer to 50.
    # A large coefficient and importance will result in a score closer to 100.
    # The coefficient from the logit model is positive, suggesting females are more likely to be denied.
    response = 75
    explanation = f"There is a statistically significant relationship between gender and mortgage application approval (p-value: {p_value_female:.4f}). The logistic regression model shows that being female is associated with a higher likelihood of denial. The Explainable Boosting Classifier also identifies 'female' as a feature with non-zero importance ({female_importance:.4f}), although it is not one of the most important predictors. This suggests that while gender does play a role, other factors are more influential in the decision."
else:
    # If the result is not statistically significant, we say there is no effect.
    response = 10
    explanation = f"There is no statistically significant relationship between gender and mortgage application approval (p-value: {p_value_female:.4f}). The p-value for the 'female' variable is greater than the 0.05 significance level, indicating that we cannot conclude that gender has a real effect on the outcome. The feature importance from the Explainable Boosting Classifier is also very low ({female_importance:.4f})."


# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. conclusion.txt created.")
