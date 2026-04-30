
import pandas as pd
import json
from scipy import stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('affairs.csv')

# Binarize the 'affairs' column to represent having an affair or not
df['had_affair'] = (df['affairs'] > 0).astype(int)

# Convert 'children' to a binary variable
df['children'] = (df['children'] == 'yes').astype(int)

# Perform a t-test to see if there's a significant difference in affairs
# between those with and without children.
with_children = df[df['children'] == 1]['had_affair']
without_children = df[df['children'] == 0]['had_affair']

ttest_result = stats.ttest_ind(with_children, without_children)

# To get a more nuanced view, we can use a logistic regression model.
# This will allow us to see the effect of having children while controlling for other variables.
X = df[['children', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']]
X = sm.add_constant(X)
y = df['had_affair']

logit_model = sm.Logit(y, X)
result = logit_model.fit()

# The coefficient for 'children' will tell us the direction and strength of the relationship.
# A negative coefficient suggests that having children is associated with a lower likelihood of having an affair.
# The p-value will tell us if this is statistically significant.
p_value = result.pvalues['children']
coefficient = result.params['children']

# Based on the results, we can form a conclusion.
# A low p-value (typically < 0.05) would suggest a significant relationship.
if p_value < 0.05 and coefficient < 0:
    response = 80  # Strong "Yes", children decrease affairs
    explanation = f"There is a statistically significant negative relationship between having children and engaging in extramarital affairs (p-value: {p_value:.4f}, coefficient: {coefficient:.4f}). This suggests that individuals with children are less likely to have affairs."
else:
    response = 20  # Leaning "No"
    explanation = f"The relationship between having children and engaging in extramarital affairs is not statistically significant (p-value: {p_value:.4f}, coefficient: {coefficient:.4f}). Therefore, we cannot conclude that having children decreases engagement in affairs."


# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
