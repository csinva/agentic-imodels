
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('mortgage.csv')

# The research question is about the effect of gender on mortgage approval.
# The target variable is 'deny'. The feature of interest is 'female'.

# 1. Statistical Test
# We can perform a t-test to see if there is a significant difference in the mean
# denial rate between female and male applicants.

female_denial_rate = df[df['female'] == 1]['deny']
male_denial_rate = df[df['female'] == 0]['deny']

ttest_result = ttest_ind(female_denial_rate, male_denial_rate)

# 2. Interpretable Model
# We can use an Explainable Boosting Classifier to model the relationship
# between all features and the denial outcome. This will help us understand
# the importance of 'female' in the context of other variables.

X = df.drop(['deny', 'accept', 'Unnamed: 0'], axis=1)
y = df['deny']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get the feature importance for 'female'
feature_importances = ebm.explain_global()
female_importance = None
for feature, importance in zip(feature_importances.data()['names'], feature_importances.data()['scores']):
    if feature == 'female':
        female_importance = importance
        break

# 3. Conclusion
# Based on the t-test and the feature importance from the EBM, we can draw a conclusion.
# A significant p-value from the t-test (e.g., < 0.05) would suggest a relationship.
# A high feature importance for 'female' would also suggest a relationship.

p_value = ttest_result.pvalue

# Let's define a threshold for significance
alpha = 0.05

explanation = f"The p-value from the t-test is {p_value:.4f}. "
if p_value < alpha:
    explanation += "This indicates a statistically significant difference in mortgage denial rates between genders. "
else:
    explanation += "This does not indicate a statistically significant difference in mortgage denial rates between genders. "

if female_importance is not None:
    explanation += f"The Explainable Boosting Model assigned a feature importance of {female_importance:.4f} to the 'female' feature. "
else:
    explanation += "The 'female' feature was not assigned an importance score by the model. "


# Based on the p-value, we can determine the response.
# If the p-value is significant, we can say there is a relationship.
if p_value < alpha:
    response = 80  # Strong "Yes"
else:
    response = 20  # Strong "No"


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
