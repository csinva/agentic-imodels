
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv('amtl.csv')

# Create the target variable: AMTL rate
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Create dummy variables for categorical features
df = pd.get_dummies(df, columns=['genus', 'tooth_class'])

# Define the model
# Target: amtl_rate
# Predictors: age, prob_male, genus, tooth_class
predictors = ['age', 'prob_male'] + [col for col in df.columns if 'genus_' in col or 'tooth_class_' in col]
X = df[predictors].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

y = df['amtl_rate'].copy()
y.fillna(y.median(), inplace=True)


# Fit the model
model = LinearRegression()
model.fit(X, y)

# Check the coefficient for Homo sapiens
homo_sapiens_coef_index = X.columns.get_loc('genus_Homo sapiens')
homo_sapiens_coef = model.coef_[homo_sapiens_coef_index]


# Interpretation
explanation = f"The coefficient for Homo sapiens is {homo_sapiens_coef:.4f}. "
if homo_sapiens_coef > 0:
    response = 80
    explanation += "The positive coefficient suggests that Homo sapiens have a higher rate of antemortem tooth loss compared to the reference primate genus (Pan), after controlling for age, sex, and tooth class. Note: p-value not calculated with sklearn."
else:
    response = 20
    explanation += "The non-positive coefficient suggests that Homo sapiens do not have a higher rate of antemortem tooth loss compared to the reference primate genus (Pan). Note: p-value not calculated with sklearn."

# Create conclusion file
output = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

print("Analysis complete. Conclusion written to conclusion.txt")
