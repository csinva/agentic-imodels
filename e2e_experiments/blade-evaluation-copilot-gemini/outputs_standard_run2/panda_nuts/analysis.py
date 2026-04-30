
import pandas as pd
import json
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Preprocess the data
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['help'] = LabelEncoder().fit_transform(df['help'])

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Fit a regression model
X = df[['age', 'sex', 'help']]
y = df['efficiency']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Get p-values
p_values = model.pvalues

# Formulate the explanation
explanation = f"To answer the question, I performed a multiple regression analysis with nut-cracking efficiency as the dependent variable and age, sex, and help as independent variables. The p-values for age, sex, and help are {p_values['age']:.3f}, {p_values['sex']:.3f}, and {p_values['help']:.3f} respectively. "

# Determine the response
# Base the response on the significance of the p-values.
# A lower p-value for 'help' would indicate a stronger relationship.
# Let's consider a p-value of 0.05 as the threshold for significance.
# We can scale the response based on how far the p-value is from this threshold.
# For simplicity, if any of the variables are significant, we can give a high score.
# If not, a low score.

significant = any(p < 0.05 for p in p_values)

if significant:
    response = 80
    explanation += "At least one of the variables (age, sex, or help) has a statistically significant influence on nut-cracking efficiency (p < 0.05), suggesting a relationship exists."
else:
    response = 20
    explanation += "None of the variables (age, sex, or help) have a statistically significant influence on nut-cracking efficiency (p > 0.05), suggesting no strong relationship."


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
