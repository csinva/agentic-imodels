
import pandas as pd
import statsmodels.api as sm
import json
import numpy as np

# Load the data
df = pd.read_csv('boxes.csv')

# The research question is about reliance on majority preference.
# The outcome variable 'y' is 1=unchosen, 2=majority, 3=minority.
# We want to model the probability of choosing the majority option.
# So, we create a binary variable 'chose_majority' which is 1 if y==2, and 0 otherwise.
df['chose_majority'] = (df['y'] == 2).astype(int)

# The independent variables are age and culture.
# We also want to see if the effect of age differs across cultures, so we include an interaction term.
# We will treat 'culture' as a categorical variable.
X = df[['age', 'culture']]
X = pd.get_dummies(X, columns=['culture'], drop_first=True)
X = sm.add_constant(X)

# Create interaction terms between age and each culture dummy variable
for col in X.columns:
    if 'culture' in col:
        X['age_x_' + col] = df['age'] * X[col]


y = df['chose_majority']

# Fit a logistic regression model
try:
    model = sm.Logit(y.astype(float), X.astype(float))
    result = model.fit()
    # Get the p-value for the interaction terms
    p_values = result.pvalues
    interaction_p_values = p_values[p_values.index.str.startswith('age_x_culture')]

    # Check if any of the interaction terms are significant
    significant_interaction = any(interaction_p_values < 0.05)

    # The main effect of age
    age_p_value = p_values['age']
    significant_age = age_p_value < 0.05

    explanation = "To answer the research question, a logistic regression was performed to model the choice of the majority option as a function of age, culture, and their interaction. "
    if significant_interaction:
        response = 90
        explanation += "The analysis found a significant interaction effect between age and culture (p < 0.05 for at least one interaction term), indicating that the development of reliance on majority preference with age differs across cultural contexts. "
    elif significant_age:
        response = 70
        explanation += "The analysis found a significant main effect of age (p < 0.05), suggesting that reliance on majority preference changes with age. However, no significant interaction with culture was found, so this developmental trend does not significantly differ across the cultural contexts in this dataset. "
    else:
        response = 10
        explanation += "The analysis found no significant effect of age or its interaction with culture on the choice of the majority option. This suggests that, in this dataset, there is no strong evidence that children's reliance on majority preference develops over age, nor that this development differs across cultures."

except Exception as e:
    response = 0
    explanation = f"An error occurred during the analysis: {e}. Therefore, no conclusion could be drawn."


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
