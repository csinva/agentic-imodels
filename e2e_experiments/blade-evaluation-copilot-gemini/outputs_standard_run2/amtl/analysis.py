
import pandas as pd
import json
import statsmodels.api as sm
from imodels import FIGSRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('amtl.csv')

# Create the target variable: proportion of teeth lost
df['amtl_prop'] = df['num_amtl'] / df['sockets']

# Create a binary variable for human vs. non-human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)

# Convert boolean columns to integers
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

# Define features and target
features = ['is_human', 'age', 'prob_male'] + [col for col in df.columns if 'tooth_class' in col]
X = df[features]
y = df['amtl_prop']

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit a GLM (Generalized Linear Model) with a binomial family
# to model the proportion of tooth loss
glm_binomial = sm.GLM(y, X, family=sm.families.Binomial())
results = glm_binomial.fit()

# Get the coefficient for the 'is_human' variable
is_human_coef = results.params['is_human']
p_value = results.pvalues['is_human']

# Interpret the results
explanation = f"The coefficient for 'is_human' is {is_human_coef:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05:
    if is_human_coef > 0:
        explanation += "This indicates that humans have a statistically significant higher frequency of antemortem tooth loss compared to non-human primates, after controlling for age, sex, and tooth class."
        response = 95
    else:
        explanation += "This indicates that humans have a statistically significant lower frequency of antemortem tooth loss compared to non-human primates, after controlling for age, sex, and tooth class."
        response = 5
else:
    explanation += "There is no statistically significant difference in antemortem tooth loss between humans and non-human primates after controlling for age, sex, and tooth class."
    response = 10

# Save the conclusion
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
