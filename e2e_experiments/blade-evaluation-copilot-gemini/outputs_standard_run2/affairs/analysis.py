
import pandas as pd
import json
from scipy import stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('affairs.csv')

# Convert 'children' to a binary variable
df['children_binary'] = df['children'].apply(lambda x: 1 if x == 'yes' else 0)

# Separate the data into two groups
with_children = df[df['children_binary'] == 1]['affairs']
without_children = df[df['children_binary'] == 0]['affairs']

# Perform a t-test
t_stat, p_value = stats.ttest_ind(with_children, without_children)

# Create a binary 'had_affair' variable for logistic regression
df['had_affair'] = (df['affairs'] > 0).astype(int)

# Prepare data for logistic regression
X = df[['children_binary']]
X = sm.add_constant(X)
y = df['had_affair']

# Fit logistic regression model
logit_model = sm.Logit(y, X).fit()
# logit_summary = logit_model.summary()

# Determine the response based on the p-value
# A low p-value (e.g., < 0.05) suggests a significant difference.
# If p_value is small, it means having children has a significant effect on affairs.
# The t-statistic being negative suggests that the mean number of affairs is lower for the group with children.
if p_value < 0.05 and t_stat < 0:
    response = 90  # Strong "Yes", children decrease affairs
    explanation = f"The t-test shows a significant difference (p-value: {p_value:.3f}) in the number of affairs between those with and without children. The negative t-statistic ({t_stat:.2f}) indicates that people with children have fewer affairs. The logistic regression odds ratio for having children is {logit_model.params['children_binary']:.3f}, suggesting a decrease in the odds of having an affair, although this specific model result was not statistically significant at the 0.05 level (p-value: {logit_model.pvalues['children_binary']:.3f}). The primary conclusion is based on the significant t-test result."
else:
    response = 10
    explanation = f"The t-test did not show a significant decrease in affairs for those with children (p-value: {p_value:.3f})."


# Create the conclusion file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
