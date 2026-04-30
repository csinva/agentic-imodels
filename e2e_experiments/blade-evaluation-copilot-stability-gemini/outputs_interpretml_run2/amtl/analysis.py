
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
import numpy as np

# Load the dataset
df = pd.read_csv('amtl.csv')

# Prepare the data for modeling
df['amtl_rate'] = df['num_amtl'] / df['sockets']
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Select features and target
features = ['is_human', 'age', 'prob_male', 'tooth_class']
target = 'amtl_rate'

df_model = df[features + [target]].copy()

# One-hot encode categorical features
df_model = pd.get_dummies(df_model, columns=['tooth_class'], drop_first=True)

# Convert all columns to numeric and drop missing values
for col in df_model.columns:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
df_model = df_model.dropna()

X = df_model.drop(target, axis=1)
y = (df_model[target] > 0).astype(int) # Logistic regression needs a binary outcome

# Fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Get the coefficient for the 'is_human' variable
is_human_coef = model.coef_[0][X.columns.get_loc('is_human')]

# Since sklearn doesn't provide p-values directly, we'll rely on the coefficient's sign and magnitude.
explanation = f"The coefficient for is_human is {is_human_coef:.3f}. "
if is_human_coef > 0.1:
    explanation += "This suggests that humans have a higher rate of antemortem tooth loss compared to other primates, after controlling for age, sex, and tooth class."
    response = 90
elif is_human_coef < -0.1:
    explanation += "This suggests that humans have a lower rate of antemortem tooth loss compared to other primates, after controlling for age, sex, and tooth class."
    response = 10
else:
    explanation += "There is no strong evidence of a difference in the rate of antemortem tooth loss between humans and other primates after controlling for age, sex, and tooth class."
    response = 50


# Write the conclusion to a file
output = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

print("Analysis complete. Conclusion written to conclusion.txt")
