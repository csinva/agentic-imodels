
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind
import json

# Load the dataset
try:
    data = pd.read_csv('affairs.csv')
except FileNotFoundError:
    print("Error: 'affairs.csv' not found. Make sure the dataset is in the same directory.")
    exit()

# --- Data Preparation ---
# Binarize the 'affairs' column to represent having an affair (1) or not (0)
data['had_affair'] = (data['affairs'] > 0).astype(int)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['gender', 'children'], drop_first=True)

# --- Statistical Analysis ---
# 1. T-test: Compare the mean of 'had_affair' for those with and without children
children_yes = data[data['children_yes'] == 1]['had_affair']
children_no = data[data['children_yes'] == 0]['had_affair']

# Check if there are samples in both groups
if not children_yes.empty and not children_no.empty:
    ttest_result = ttest_ind(children_yes, children_no)
    p_value_ttest = ttest_result.pvalue
else:
    p_value_ttest = 1.0 # Assign a non-significant p-value if one group is empty

# 2. Logistic Regression: Model the likelihood of an affair
# Define dependent and independent variables
y = data['had_affair']
X = data.drop(['affairs', 'had_affair', 'rownames'], axis=1)
X = sm.add_constant(X) # Add a constant for the intercept

# Fit the logistic regression model
try:
    model = sm.Logit(y, X).fit(disp=0)
    p_value_regression = model.pvalues['children_yes']
    coef_children = model.params['children_yes']
except Exception as e:
    # Handle cases where the model fails to converge or other errors
    p_value_regression = 1.0
    coef_children = 0

# --- Interpretation and Conclusion ---
# Base the conclusion on the statistical significance from the regression model
# A positive coefficient for 'children_yes' would mean having children increases the odds of an affair.
# A negative coefficient would mean having children decreases the odds.

significant_regression = p_value_regression < 0.05
significant_ttest = p_value_ttest < 0.05

if significant_regression and coef_children < 0:
    # Significant negative relationship
    response = 90
    explanation = (
        f"Yes, there is a statistically significant negative relationship (p={p_value_regression:.3f}). "
        f"Having children is associated with a decrease in the likelihood of engaging in extramarital affairs. "
        f"The logistic regression coefficient for having children is {coef_children:.3f}, indicating lower odds of an affair."
    )
elif significant_regression and coef_children > 0:
    # Significant positive relationship
    response = 10
    explanation = (
        f"No, the relationship is statistically significant but in the opposite direction (p={p_value_regression:.3f}). "
        f"Having children is associated with an increase in the likelihood of engaging in extramarital affairs. "
        f"The logistic regression coefficient is {coef_children:.3f}."
    )
else:
    # Not a significant relationship
    response = 20
    explanation = (
        f"No, there is no statistically significant relationship (p={p_value_regression:.3f}). "
        f"The data does not provide strong evidence that having children decreases engagement in extramarital affairs. "
        f"The coefficient for having children was {coef_children:.3f}, which is not statistically significant."
    )

# Also consider the t-test result for a more complete picture
explanation += (
    f" A t-test comparing the mean affair rates for those with and without children also showed a p-value of {p_value_ttest:.3f}, "
    f"which points to a {'significant' if significant_ttest else 'non-significant'} difference."
)


# --- Output ---
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
