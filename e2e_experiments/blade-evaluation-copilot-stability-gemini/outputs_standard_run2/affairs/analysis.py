
import pandas as pd
import json
from scipy.stats import ttest_ind
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('affairs.csv')

# The research question is: "Does having children decrease (if at all) the engagement in extramarital affairs?"

# Separate the dataset into two groups: those with children and those without
with_children = df[df['children'] == 'yes']['affairs']
no_children = df[df['children'] == 'no']['affairs']

# Perform an independent t-test to compare the means of the two groups
ttest_statistic, p_value = ttest_ind(with_children, no_children, equal_var=False)

# To get more insight, let's run a regression model.
# First, we need to convert categorical variables into dummy variables.
df_dummies = pd.get_dummies(df, columns=['gender', 'children'], drop_first=True)

# Define the independent variables (X) and the dependent variable (y)
X = df_dummies[['children_yes', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']]
X['children_yes'] = X['children_yes'].astype(int)
X = sm.add_constant(X)  # Add a constant to the model
y = df_dummies['affairs']

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y, X).fit()
p_value_regression = model.pvalues['children_yes']


# The t-test p-value is the primary indicator.
# A low p-value (typically < 0.05) suggests a significant difference.
# The regression coefficient for 'children_yes' will tell us the direction and magnitude of the effect.
# A negative coefficient would mean having children is associated with fewer affairs.

# Based on the p-value from the t-test, we can determine the answer.
# If p_value < 0.05, there is a statistically significant difference.
# We can then look at the means to see the direction.
mean_with_children = with_children.mean()
mean_no_children = no_children.mean()

# If mean_with_children < mean_no_children and the result is significant, then having children decreases affairs.
# The 'response' will be a score from 0 to 100.
# 0 means strong "No" (no significant effect or it increases affairs).
# 100 means strong "Yes" (significant decrease).

# Let's set a threshold for the p-value
alpha = 0.05

if p_value < alpha and mean_with_children < mean_no_children:
    # Significant and in the expected direction
    response = 95  # Strong "Yes"
    explanation = f"There is a statistically significant decrease in extramarital affairs for individuals with children (p-value: {p_value:.4f}). The average number of affairs for those with children is {mean_with_children:.2f}, compared to {mean_no_children:.2f} for those without."
elif p_value < alpha and mean_with_children >= mean_no_children:
    # Significant but in the opposite direction
    response = 5 # Strong "No"
    explanation = f"Having children appears to significantly increase or have no decreasing effect on extramarital affairs (p-value: {p_value:.4f}). The average number of affairs for those with children is {mean_with_children:.2f}, compared to {mean_no_children:.2f} for those without."
else:
    # Not significant
    response = 10 # Strong "No"
    explanation = f"There is no statistically significant evidence to suggest that having children decreases engagement in extramarital affairs (p-value: {p_value:.4f})."


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
