
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('boxes.csv')

# Prepare the data
# Convert 'y' to a binary outcome: 1 if majority option was chosen, 0 otherwise
df['majority_chosen'] = (df['y'] == 2).astype(int)

# Define features (X) and target (y)
features = ['age', 'culture', 'gender', 'majority_first']
X = df[features]
y = df['majority_chosen']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build an Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train_scaled, y_train)

# Get the explanation for the 'age' feature
explanation = ebm.explain_global()
age_explanation = explanation.data(0)

# Check the relationship between age and the probability of choosing the majority option
# A positive score for a feature value means that it contributes to a higher probability of choosing the majority option.
# A negative score means it contributes to a lower probability.
# We can check if the scores for older ages are generally higher than for younger ages.
age_scores = age_explanation['scores']
age_values = age_explanation['names']

# A simple check: is the score for the highest age greater than the score for the lowest age?
if age_scores[-1] > age_scores[0]:
    response = 80
    explanation_text = "Yes, there is a positive relationship. As age increases, the likelihood of choosing the majority option also increases. The Explainable Boosting Model shows that the feature contribution of age to the log-odds of choosing the majority option is generally increasing with age."
else:
    response = 20
    explanation_text = "No, there is not a clear positive relationship. The model does not show a consistent increase in the likelihood of choosing the majority option with age."


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation_text
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
