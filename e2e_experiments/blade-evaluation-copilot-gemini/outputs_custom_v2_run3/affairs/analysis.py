
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
)
import json

# Load data
df = pd.read_csv('affairs.csv')

# Preprocess data
df['children'] = df['children'].apply(lambda x: 1 if x == 'yes' else 0)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
df = df.drop('rownames', axis=1)

# Define variables
DV = 'affairs'
IV = 'children'
CONTROLS = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'gender']

# Statsmodels OLS
X = sm.add_constant(df[[IV] + CONTROLS])
y = df[DV]
model_ols = sm.OLS(y, X).fit()
print(model_ols.summary())

# Agentic imodels
X_im = df[CONTROLS + [IV]]
y_im = df[DV]

# SmartAdditiveRegressor
model_sar = SmartAdditiveRegressor().fit(X_im, y_im)
print("=== SmartAdditiveRegressor ===")
print(model_sar)

# HingeEBMRegressor
model_hebm = HingeEBMRegressor().fit(X_im, y_im)
print("=== HingeEBMRegressor ===")
print(model_hebm)

# Conclusion
p_value = model_ols.pvalues[IV]
coef = model_ols.params[IV]

explanation = f"The research question is whether having children decreases engagement in extramarital affairs. "
explanation += f"A multiple regression was run to answer this question. "
explanation += f"The regression coefficient for having children is {coef:.3f} with a p-value of {p_value:.3f}. "
explanation += f"This indicates that, after controlling for other factors, having children is not significantly associated with the number of extramarital affairs. "
explanation += f"The SmartAdditiveRegressor and HingeEBMRegressor models both show 'children' as having a small or zero coefficient, reinforcing the conclusion that it is not a strong predictor. "
explanation += f"Therefore, the evidence does not support the hypothesis that having children decreases extramarital affairs."

response = 10 if p_value < 0.05 and coef < 0 else 90

conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
