
import pandas as pd
import json
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

# Load the dataset
df = pd.read_csv('mortgage.csv')

# Clean the data
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Research Question: How does gender affect whether banks approve an individual’s mortgage application?

# 1. Statistical Test: Two-sample t-test
# Compare the mean denial rates for female and male applicants.
denial_female = df[df['female'] == 1]['deny']
denial_male = df[df['female'] == 0]['deny']

ttest_result = ttest_ind(denial_female, denial_male, equal_var=False)

# 2. Modeling: Logistic Regression
# Build a logistic regression model to predict mortgage denial based on gender and other relevant factors.
# This helps to see if gender is a significant predictor when controlling for other variables.
features = ['female', 'black', 'housing_expense_ratio', 'self_employed', 'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value']
X = df[features]
y = df['deny']

# Using statsmodels for more detailed output (p-values, coefficients)
X_sm = sm.add_constant(X)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit()
p_value_gender = result.pvalues['female']
coefficient_gender = result.params['female']

# Interpretation
explanation = f"To answer the research question, I performed a two-sample t-test and a logistic regression analysis. The t-test compares the mortgage denial rates between female and male applicants. The result was a t-statistic of {ttest_result.statistic:.3f} and a p-value of {ttest_result.pvalue:.3f}. "

if ttest_result.pvalue < 0.05:
    explanation += "The difference in denial rates between genders is statistically significant. "
else:
    explanation += "The difference in denial rates between genders is not statistically significant. "

explanation += f"Furthermore, I built a logistic regression model to predict mortgage denial. The model included gender along with other financial and demographic variables. The p-value for the 'female' coefficient was {p_value_gender:.3f}. "

if p_value_gender < 0.05:
    explanation += "This indicates that gender is a statistically significant predictor of mortgage denial, even after controlling for other factors. "
    if coefficient_gender > 0:
        explanation += "The positive coefficient suggests that being female is associated with a higher likelihood of denial. Based on this, I would conclude that gender does affect mortgage application approval."
        response = 80
    else:
        explanation += "The negative coefficient suggests that being female is associated with a lower likelihood of denial. Based on this, I would conclude that gender does affect mortgage application approval."
        response = 80
else:
    explanation += "This indicates that gender is not a statistically significant predictor of mortgage denial when controlling for other factors. Based on this, I would conclude that there is not strong evidence that gender affects mortgage application approval."
    response = 20

# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. conclusion.txt created.")
print(result.summary())
