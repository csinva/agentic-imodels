
import pandas as pd
import json
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imodels import RuleFitClassifier
import numpy as np

# Load the dataset
df = pd.read_csv('amtl.csv')

# Create the AMTL rate
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Create a binary variable for human vs. non-human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Prepare the data for modeling
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)

# Define features (X) and target (y)
features = ['is_human', 'age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar']
X = df[features].astype(float)
y = df['amtl_rate']

# Binomial regression using statsmodels for interpretability
X_sm = sm.add_constant(X)
glm_binom = sm.GLM(y, X_sm, family=sm.families.Binomial())
res = glm_binom.fit()

# Get the coefficient for the 'is_human' variable
is_human_coef = res.params['is_human']
p_value = res.pvalues['is_human']

# Interpret the results
explanation = f"The coefficient for 'is_human' in the binomial regression model is {is_human_coef:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05:
    if is_human_coef > 0:
        explanation += "This indicates that being human is significantly associated with a higher rate of antemortem tooth loss, after controlling for age, sex, and tooth class."
        response = 95
    else:
        explanation += "This indicates that being human is significantly associated with a lower rate of antemortem tooth loss, after controlling for age, sex, and tooth class."
        response = 5
else:
    explanation += "There is no statistically significant evidence to suggest a difference in antemortem tooth loss rates between humans and non-human primates after controlling for age, sex, and tooth class."
    response = 10

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. Conclusion written to conclusion.txt")
