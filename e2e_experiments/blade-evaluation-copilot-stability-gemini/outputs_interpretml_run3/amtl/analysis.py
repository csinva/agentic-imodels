
import pandas as pd
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('amtl.csv')

# Create the response variable: AMTL frequency
df['amtl_freq'] = df['num_amtl'] / df['sockets']

# Create a binary variable for Homo sapiens
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Prepare the data for the model
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)
df['tooth_class_Posterior'] = df['tooth_class_Posterior'].astype(int)
df['tooth_class_Premolar'] = df['tooth_class_Premolar'].astype(int)


# Define the independent variables (predictors)
X = df[['is_human', 'age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar']]
X = sm.add_constant(X)

# Define the dependent variable (outcome)
y = df['amtl_freq']

print("Data types of X:")
print(X.dtypes)
print("Data types of y:")
print(y.dtypes)

# Build the logistic regression model
# We use a Binomial family with a logit link function, which is standard for proportions.
# We also need to provide the number of trials (sockets) for each observation.
logit_model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=df['sockets'])
result = logit_model.fit()

# Interpret the results
p_value_human = result.pvalues['is_human']
coefficient_human = result.params['is_human']

# Determine the conclusion
if p_value_human < 0.05 and coefficient_human > 0:
    response = 95
    explanation = "The model shows a statistically significant positive relationship (p < 0.05) between being a modern human and the frequency of antemortem tooth loss, after controlling for age, sex, and tooth class. The coefficient for 'is_human' is positive, indicating that humans have a higher likelihood of AMTL."
elif p_value_human < 0.05 and coefficient_human <= 0:
    response = 5
    explanation = "The model shows a statistically significant negative relationship (p < 0.05) between being a modern human and the frequency of antemortem tooth loss, after controlling for age, sex, and tooth class. This suggests humans have a lower likelihood of AMTL."
else:
    response = 50
    explanation = "The model does not show a statistically significant relationship (p >= 0.05) between being a modern human and the frequency of antemortem tooth loss, after controlling for age, sex, and tooth class. There is not enough evidence to conclude that humans have a different rate of AMTL than other primates in this dataset."

# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
print(result.summary())
