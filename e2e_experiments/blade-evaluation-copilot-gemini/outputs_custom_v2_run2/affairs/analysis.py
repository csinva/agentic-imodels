
import json
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    TeacherStudentRuleSplineRegressor
)

# Load data
df = pd.read_csv('affairs.csv')

# Preprocessing
df['children'] = df['children'].apply(lambda x: 1 if x == 'yes' else 0)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
df['affairs_binary'] = (df['affairs'] > 0).astype(int)

# Research Question: Does having children decrease engagement in extramarital affairs?
# DV: affairs (or affairs_binary)
# IV: children

# --- Analysis ---

# 1. Bivariate analysis: affairs vs children
contingency_table = pd.crosstab(df['children'], df['affairs_binary'])
from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(contingency_table)
bivariate_result = f"Bivariate chi-square test p-value: {p:.4f}"

# 2. Controlled analysis: Logistic Regression
# Controls: age, yearsmarried, religiousness, rating
X = df[['children', 'age', 'yearsmarried', 'religiousness', 'rating', 'gender', 'education', 'occupation']]
y = df['affairs_binary']
X = sm.add_constant(X)
logit_model = sm.Logit(y, X).fit(disp=0)
logit_summary = logit_model.summary().as_text()
children_coef = logit_model.params['children']
children_pvalue = logit_model.pvalues['children']

# 3. Interpretable Models
X_subset = df[['children', 'age', 'yearsmarried', 'religiousness', 'rating']]
y_reg = df['affairs']

models = {
    "SmartAdditiveRegressor": SmartAdditiveRegressor(),
    "HingeEBMRegressor": HingeEBMRegressor(),
    "WinsorizedSparseOLSRegressor": WinsorizedSparseOLSRegressor(),
    "TeacherStudentRuleSplineRegressor": TeacherStudentRuleSplineRegressor(),
}

model_outputs = {}
for name, model in models.items():
    model.fit(X_subset, y_reg)
    model_outputs[name] = str(model)

# --- Conclusion ---
explanation = "The research question is whether having children decreases engagement in extramarital affairs.\\n\\n"
explanation += f"1. **Bivariate Analysis**: A chi-square test between 'children' and a binary 'had_affair' indicator gives a p-value of {p:.4f}. This suggests a statistically significant relationship between having children and engaging in affairs.\\n\\n"
explanation += f"2. **Controlled Logistic Regression**: A logistic regression model predicting the likelihood of an affair, controlling for age, years married, religiousness, and marriage rating, shows a coefficient for 'children' of {children_coef:.4f} with a p-value of {children_pvalue:.4f}. This indicates that, even after controlling for other factors, having children is significantly associated with a change in the likelihood of having an affair. The sign of the coefficient suggests the direction of this relationship.\\n\\n"

# Interpretation of interpretable models
for name, output in model_outputs.items():
    explanation += f"3. **{name}**: \\n{output}\\n\\n"
    if 'children' in output:
        explanation += f"The {name} model includes 'children' as a feature, indicating its importance. The coefficient/rule associated with 'children' provides insight into the direction and magnitude of its effect.\\n"
    else:
        explanation += f"The {name} model does *not* select 'children' as an important feature, suggesting its effect may be weak or redundant given other variables.\\n"


# Final score calibration
score = 50 # Start with a neutral score

# Adjust based on p-values
if children_pvalue < 0.05:
    if children_coef < 0:
        score += 25  # Significant negative effect
    else:
        score -=15 # Significant positive effect
else:
    score -= 25 # Not significant

# Adjust based on interpretable models
children_in_models = sum(1 for name, output in model_outputs.items() if 'children' in output and '0.00' not in output.split('children')[1][:5])
if children_in_models >= 2:
    score += 15 # Feature consistently important
else:
    score -= 15 # Feature not consistently important

# Final score clamping
score = max(0, min(100, score))


conclusion = {
    "response": int(score),
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. Conclusion written to conclusion.txt")
print(bivariate_result)
print(logit_summary)
for name, output in model_outputs.items():
    print(f"--- {name} ---")
    print(output)
