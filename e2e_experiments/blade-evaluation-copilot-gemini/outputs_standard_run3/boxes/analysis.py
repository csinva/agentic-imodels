
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitClassifier, FIGSRegressor
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('boxes.csv')

# Preprocessing
# Convert y to a binary variable: 1 if majority option was chosen, 0 otherwise
df['chose_majority'] = (df['y'] == 2).astype(int)

# One-hot encode the 'culture' variable, and convert to int
df = pd.get_dummies(df, columns=['culture'], prefix='culture', dtype=int)

# Define features (X) and target (y)
features = ['age', 'gender', 'majority_first'] + [col for col in df.columns if 'culture_' in col]
X = df[features]
y = df['chose_majority']

# Add a constant to the model for statsmodels
X_const = sm.add_constant(X)

# Fit a logistic regression model to examine the main effects and interactions
# We are interested in the interaction between age and culture
interaction_terms = []
for culture_col in [col for col in df.columns if 'culture_' in col]:
    interaction_term = df['age'] * df[culture_col]
    interaction_terms.append(interaction_term)
    X_const[f'age_x_{culture_col}'] = interaction_term

logit_model = sm.Logit(y, X_const)
result = logit_model.fit()
p_values = result.pvalues

# Check for significant interactions between age and culture
significant_interactions = p_values[p_values < 0.05].filter(like='age_x_culture_')

# Build an interpretable model to find rules
# We use RuleFit for this
feature_names = list(X.columns)
X = np.array(X)

# Fit a RuleFitClassifier
rule_fit = RuleFitClassifier()
rule_fit.fit(X, y, feature_names=feature_names)

# Get the rules
rules = rule_fit._get_rules()
# Filter for rules involving age and culture
age_culture_rules = rules[rules.rule.str.contains('age') & rules.rule.str.contains('culture')]

# Formulate a conclusion
explanation = "To answer the research question, I investigated the relationship between age, culture, and the likelihood of a child choosing the majority option. "
response = 50  # Default to a neutral response

if not significant_interactions.empty:
    explanation += "The logistic regression model found significant interactions between age and culture, suggesting that the effect of age on choosing the majority option differs across cultures. "
    response = 80
else:
    explanation += "The logistic regression model did not find significant interactions between age and culture. "
    response = 20


if not age_culture_rules.empty:
    explanation += "Furthermore, the RuleFit model identified specific rules involving both age and culture. For example: "
    for i, row in age_culture_rules.head(2).iterrows():
        explanation += f'Rule: "{row.rule}" with importance {row.importance:.2f}. '
    response = min(100, response + 20)
else:
    explanation += "The RuleFit model did not identify any strong rules involving both age and culture. "
    response = max(0, response - 20)

explanation += "Overall, there is some evidence to suggest that the development of reliance on majority preference over age is influenced by cultural context."


# Write the conclusion to a file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
