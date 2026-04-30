
import json
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor
import numpy as np

# 1. Load Data
df = pd.read_csv('hurricane.csv')

# 2. Prepare data for modeling
# The dependent variable (DV) is 'alldeaths'.
# The independent variable (IV) of interest is 'masfem'.
# Control variables: 'category', 'ndam', 'min' (minimum pressure)
# The research question suggests a causal link where feminine names lead to higher deaths.
# The outcome 'alldeaths' is a count, and its distribution is highly skewed.
# A log transformation is appropriate to handle the skewness and the multiplicative nature of the effects.
# We add a small constant to avoid log(0).
df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

# Define variables for the model
dv = 'log_alldeaths'
iv = 'masfem'
control_cols = ['category', 'ndam', 'min']
feature_cols = [iv] + control_cols

X = df[feature_cols]
y = df[dv]

# Handle potential missing values if any, using a simple mean imputation
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())


# 3. Classical Statistical Test (GLM with controls)
# Using a Gaussian family for the log-transformed outcome is equivalent to log-linear regression.
X_sm = sm.add_constant(X)
glm_gaussian = sm.GLM(y, X_sm, family=sm.families.Gaussian())
results = glm_gaussian.fit()
statsmodels_summary = results.summary().as_text()

# 4. Interpretable Models for Shape, Direction, Importance
# Fit two models from agentic_imodels as per SKILL.md
# Using HingeEBMRegressor (high-rank) and SmartAdditiveRegressor (honest)
models = {
    "HingeEBMRegressor": HingeEBMRegressor(),
    "SmartAdditiveRegressor": SmartAdditiveRegressor()
}

model_outputs = {}
for name, model in models.items():
    model.fit(X, y)
    model_outputs[name] = str(model)

# 5. Synthesize Results and Write Conclusion
explanation = "Analysis of the relationship between hurricane name femininity (`masfem`) and deaths (`alldeaths`).\\n\\n"

# Interpret statsmodels results
explanation += "--- Classical Analysis (GLM) ---\\n"
explanation += statsmodels_summary + "\\n\\n"
p_value = results.pvalues[iv]
coef = results.params[iv]

if p_value < 0.05:
    explanation += f"The GLM model shows a statistically significant relationship (p={p_value:.4f}) between '{iv}' and '{dv}'. "
    if coef > 0:
        explanation += "The coefficient is positive, suggesting that more feminine-named hurricanes are associated with more deaths, even after controlling for intensity.\\n"
    else:
        explanation += "The coefficient is negative, suggesting a negative association.\\n"
else:
    explanation += f"The GLM model does not show a statistically significant relationship (p={p_value:.4f}) between '{iv}' and '{dv}' after controlling for hurricane intensity.\\n"


# Interpret agentic_imodels results
explanation += "\\n--- Interpretable Models ---\\n"
for name, output in model_outputs.items():
    explanation += f"--- {name} ---\\n{output}\\n\\n"

# Check if 'masfem' is considered important by the interpretable models
hinge_ebm_importance = "masfem" in model_outputs["HingeEBMRegressor"]
smart_additive_importance = "masfem" in model_outputs["SmartAdditiveRegressor"]

explanation += "\\n--- Synthesis ---\\n"
explanation += f"HingeEBMRegressor importance for 'masfem': {hinge_ebm_importance}.\\n"
explanation += f"SmartAdditiveRegressor importance for 'masfem': {smart_additive_importance}.\\n"

# Calibrate Likert score based on SKILL.md guidelines
score = 0
if p_value < 0.05 and coef > 0:
    # Significant positive effect in GLM
    score += 50
    if hinge_ebm_importance or smart_additive_importance:
        # Corroborated by at least one interpretable model
        score += 25
    if hinge_ebm_importance and smart_additive_importance:
        # Corroborated by both
        score += 15
elif p_value < 0.05 and coef < 0:
    # Significant but in the opposite direction of the hypothesis
    score = 10 # Low score as it contradicts the research question's premise
    explanation += "The statistical model found a significant effect, but in the opposite direction of the research hypothesis.\\n"
else:
    # Not significant
    score = 20
    explanation += "The primary statistical model did not find a significant effect. "
    if not hinge_ebm_importance and not smart_additive_importance:
        score -= 15 # Penalize if interpretable models also ignore it
        explanation += "Furthermore, the feature was zeroed out or ranked as unimportant by the interpretable models, providing strong null evidence.\\n"

# Final score clamping
score = max(0, min(100, score))


final_conclusion = {
    "response": int(score),
    "explanation": explanation
}

# 6. Write conclusion.txt
with open('conclusion.txt', 'w') as f:
    json.dump(final_conclusion, f, indent=2)

print("Analysis complete. Conclusion written to conclusion.txt")
print("\\nFull GLM results:\\n", statsmodels_summary)
print("\\nInterpretable model outputs:\\n")
for name, output in model_outputs.items():
    print(f"--- {name} ---\\n{output}\\n")

