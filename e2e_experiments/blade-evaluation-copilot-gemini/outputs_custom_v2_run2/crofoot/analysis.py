
import json
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv('crofoot.csv')

# Feature engineering
df['rel_grp_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_focal'] - df['dist_other']

# Define variables
outcome = 'win'
ivs = ['rel_grp_size', 'rel_dist']
X = df[ivs]
y = df[outcome]

# --- Classical statistical test ---
X_sm = sm.add_constant(X)
logit_model = sm.Logit(y, X_sm).fit()
summary = logit_model.summary()
p_values = logit_model.pvalues

# --- Interpretable models ---
models = {}
for model_class in [SmartAdditiveRegressor, HingeEBMRegressor]:
    model = model_class()
    model.fit(X, y)
    models[model_class.__name__] = str(model)

# --- Interpretation and Conclusion ---
explanation = "Analysis of capuchin contest outcomes:\n\n"
explanation += f"Statsmodels Logit Results:\n{summary}\n\n"

rel_grp_size_p = p_values['rel_grp_size']
rel_dist_p = p_values['rel_dist']

explanation += f"The p-value for relative group size is {rel_grp_size_p:.3f}. "
explanation += f"The p-value for relative distance from home range center is {rel_dist_p:.3f}.\n"

for name, model_str in models.items():
    explanation += f"\n--- {name} ---\n{model_str}\n"

# Corrected interpretation based on the statistical output
explanation += '''
Conclusion: The logistic regression analysis did not find statistically significant effects of relative group size (p=0.147) or relative distance (p=0.739) on the probability of winning a contest.

The interpretable models provide further insight:
- The SmartAdditiveRegressor shows a small positive linear effect for relative group size, but the effect for relative distance is non-linear and inconsistent.
- The HingeEBMRegressor assigned zero coefficients to both features, suggesting they have little to no predictive power in this model.

Contrary to the initial hypothesis, the statistical evidence from this dataset is weak. There is no significant linear relationship between relative group size or contest location and the probability of winning. The interpretable models either show a very weak effect or no effect at all. Therefore, we cannot conclude that these factors are strong predictors of contest outcomes in this context.
'''

# Update the score to reflect the weak evidence
score = 20

# Write the corrected conclusion to the file
output = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(output, f, indent=2)

print("Corrected conclusion written to conclusion.txt")
