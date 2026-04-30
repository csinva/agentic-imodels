
import json
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind

# Load the dataset
data = pd.read_csv("affairs.csv")

# Convert 'children' to a binary variable
data['children_binary'] = data['children'].apply(lambda x: 1 if x == 'yes' else 0)

# Separate the 'affairs' column for the two groups
affairs_with_children = data[data['children'] == 'yes']['affairs']
affairs_without_children = data[data['children'] == 'no']['affairs']

# Perform an independent t-test
ttest_result = ttest_ind(affairs_with_children, affairs_without_children)

# The research question is: "Does having children decrease (if at all) the engagement in extramarital affairs?"
# We are looking for a statistically significant negative relationship.
# A two-tailed t-test was performed. If the t-statistic is negative and the p-value is low,
# it suggests that the mean of the 'with children' group is lower than the 'without children' group.

# My interpretation is that a small p-value (e.g., < 0.05) would indicate a significant difference.
# The sign of the t-statistic will tell us the direction.
# A negative t-statistic means the first group (with children) has a lower mean.

# Based on the t-test, I will determine the response.
# If p-value is high (>0.05), there is no significant effect, so the answer is "No" (0).
# If p-value is low (<0.05) and the t-statistic is negative, it means having children is associated with fewer affairs, so "Yes" (100).
# If p-value is low (<0.05) and the t-statistic is positive, it means having children is associated with more affairs, which is counter to the hypothesis, but still a significant relationship. I will still score it high, as there is a relationship.

explanation = f"The t-test comparing the number of affairs between those with and without children resulted in a t-statistic of {ttest_result.statistic:.3f} and a p-value of {ttest_result.pvalue:.3f}. "

if ttest_result.pvalue < 0.05:
    if ttest_result.statistic < 0:
        explanation += "This indicates that having children is associated with a statistically significant decrease in the number of extramarital affairs."
        response = 90  # Strong "Yes"
    else:
        explanation += "This indicates that having children is associated with a statistically significant increase in the number of extramarital affairs, which is contrary to the research question but still a significant finding."
        response = 10 # Weak "No" because the relationship is in the opposite direction of the hypothesis
else:
    explanation += "This indicates that there is no statistically significant difference in the number of extramarital affairs between those with and without children."
    response = 10 # Strong "No"

# Create the conclusion dictionary
conclusion = {"response": response, "explanation": explanation}

# Write the conclusion to a file
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)
