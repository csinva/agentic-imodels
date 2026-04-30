
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitRegressor
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('boxes.csv')

# Prepare the data
# Convert 'y' to a binary indicator for choosing the majority option
data['chose_majority'] = (data['y'] == 2).astype(int)

# One-hot encode the 'culture' variable
data = pd.get_dummies(data, columns=['culture'], prefix='culture')

# Define features (X) and target (y)
features = ['age', 'gender', 'majority_first'] + [col for col in data.columns if 'culture_' in col]
X = data[features]
y = data['chose_majority']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Fit a logistic regression model to examine the relationship
# Add a constant for the intercept
X_train_sm = sm.add_constant(X_train).astype(float)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Get the p-value for the 'age' coefficient
age_p_value = result.pvalues['age']

# Interpret the results
explanation = f"The analysis examined the relationship between age and the likelihood of choosing the majority option. A logistic regression model was fitted to the data. The p-value for the 'age' coefficient was {age_p_value:.4f}. "

# A low p-value (typically < 0.05) suggests a significant relationship.
if age_p_value < 0.05:
    response = 90  # Strong "Yes"
    explanation += "This indicates a statistically significant relationship between age and the tendency to follow the majority, supporting the research hypothesis."
else:
    response = 10  # Strong "No"
    explanation += "This indicates that there is no statistically significant relationship between age and the tendency to follow the majority."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
