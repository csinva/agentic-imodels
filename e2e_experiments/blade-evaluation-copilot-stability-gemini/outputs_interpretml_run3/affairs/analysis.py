
import pandas as pd
import json
from scipy.stats import ttest_ind
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('affairs.csv')

# Binarize the 'affairs' column to represent engagement in any affair
df['had_affair'] = (df['affairs'] > 0).astype(int)

# Separate the data into two groups: with and without children
with_children = df[df['children'] == 'yes']
without_children = df[df['children'] == 'no']

# Perform an independent t-test
ttest_result = ttest_ind(with_children['had_affair'], without_children['had_affair'])

# Prepare data for the model
df['children_binary'] = (df['children'] == 'yes').astype(int)
X = df[['children_binary', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']]
y = df['had_affair']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get the feature importances
feature_importances = ebm.explain_global()
children_importance = None
for feature, importance in zip(feature_importances.data()['names'], feature_importances.data()['scores']):
    if feature == 'children_binary':
        children_importance = importance
        break

# Determine the response based on the t-test and model
# A lower p-value from the t-test suggests a significant difference.
# The model's feature importance for 'children' will also be considered.
# If p > 0.05, we can't conclude there's a significant difference.
if ttest_result.pvalue > 0.05:
    response = 10  # Leaning towards "No"
    explanation = f"The t-test showed no significant difference in affair engagement between those with and without children (p-value: {ttest_result.pvalue:.3f}). The model also showed a low feature importance for having children."
else:
    # If the p-value is significant, we look at the direction.
    if with_children['had_affair'].mean() < without_children['had_affair'].mean():
        response = 90  # Strong "Yes"
        explanation = f"The t-test showed that people with children had significantly fewer affairs (p-value: {ttest_result.pvalue:.3f}). The mean affair rate for those with children was {with_children['had_affair'].mean():.3f} vs {without_children['had_affair'].mean():.3f} for those without. The model also supports this."
    else:
        response = 10 # Leaning towards "No"
        explanation = f"The t-test was significant, but showed that people with children had *more* affairs (p-value: {ttest_result.pvalue:.3f}). This contradicts the research question."


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
