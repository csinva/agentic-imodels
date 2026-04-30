
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('affairs.csv')

# Convert categorical variables to numerical
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
df['children'] = df['children'].apply(lambda x: 1 if x == 'yes' else 0)

# Create a binary target variable for affairs
df['had_affair'] = df['affairs'].apply(lambda x: 1 if x > 0 else 0)

# Separate the data into two groups: with and without children
with_children = df[df['children'] == 1]
without_children = df[df['children'] == 0]

# Perform a t-test to compare the mean number of affairs
ttest_result = ttest_ind(with_children['had_affair'], without_children['had_affair'])

# Define features and target
features = ['gender', 'age', 'yearsmarried', 'children', 'religiousness', 'education', 'occupation', 'rating']
target = 'had_affair'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Train an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Get the feature importances from the EBM
ebm_importances = ebm.term_importances()

# Find the importance of the 'children' feature
children_importance = ebm_importances[features.index('children')]

# Determine the response based on the t-test and feature importance
if ttest_result.pvalue < 0.05 and children_importance > 0:
    response = 20  # A small but significant effect
    explanation = "The t-test indicates a significant difference in affair rates between those with and without children (p-value: {:.3f}). The Explainable Boosting Classifier also shows that having children has a small but noticeable impact on the likelihood of having an affair.".format(ttest_result.pvalue)
else:
    response = 80  # A strong indication of no effect
    explanation = "The t-test did not find a significant difference in affair rates between those with and without children (p-value: {:.3f}). The Explainable Boosting Classifier also indicates that having children is not a strong predictor of extramarital affairs.".format(ttest_result.pvalue)

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
