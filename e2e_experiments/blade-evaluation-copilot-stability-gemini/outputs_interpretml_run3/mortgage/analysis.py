
import pandas as pd
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier
import json

# Load the data
df = pd.read_csv('mortgage.csv')

# Handle missing values by filling with the median
median_values = df.median()
df.fillna(median_values, inplace=True)

# Define features and target
features = ['female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
            'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
            'loan_to_value', 'denied_PMI']
target = 'deny'

X = df[features]
y = df[target]

# Add a constant for the intercept
X_sm = sm.add_constant(X)

# Fit logistic regression model
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit()

# Get the coefficient and p-value for the 'female' feature
female_coef = result.params['female']
female_pvalue = result.pvalues['female']

# Fit Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X, y)

# Get feature importance for 'female'
ebm_global = ebm.explain_global()
global_data = ebm_global.data()
female_index = global_data['names'].index('female')
female_importance = global_data['scores'][female_index]



# Determine the response based on p-value
# A common threshold for statistical significance is p < 0.05
if female_pvalue < 0.05:
    # If the relationship is statistically significant, the score should be high.
    # The sign of the coefficient determines the direction.
    if female_coef > 0:
        explanation = f"Gender has a statistically significant effect on mortgage denial (p-value: {female_pvalue:.4f}). The positive coefficient ({female_coef:.4f}) suggests that being female increases the likelihood of denial, holding other factors constant."
        response = 90
    else:
        explanation = f"Gender has a statistically significant effect on mortgage denial (p-value: {female_pvalue:.4f}). The negative coefficient ({female_coef:.4f}) suggests that being female decreases the likelihood of denial, holding other factors constant."
        response = 90
else:
    # If not significant, the score should be low.
    explanation = f"Gender does not have a statistically significant effect on mortgage denial (p-value: {female_pvalue:.4f}). The coefficient for the 'female' feature was not statistically significant at the 0.05 level."
    response = 10


# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
