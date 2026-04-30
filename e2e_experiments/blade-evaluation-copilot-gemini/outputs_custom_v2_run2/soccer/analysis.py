
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor
import json
import re

# Load data
df = pd.read_csv('soccer.csv')

# Preprocessing
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df.dropna(subset=['skin_tone'], inplace=True)

# Define variables
y_col = 'redCards'
iv_col = 'skin_tone'
control_cols = ['victories', 'ties', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'height', 'weight', 'games']

X = df[control_cols + [iv_col]]
y = df[y_col]

# Drop rows with missing values in X or y
X.dropna(inplace=True)
y = y[X.index]


# Statsmodels analysis
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
summary = model_sm.summary()
print(summary)

# Agentic imodels
model_hinge = HingeEBMRegressor()
model_hinge.fit(X, y)
model_hinge_str = str(model_hinge)
print(model_hinge_str)

model_smart = SmartAdditiveRegressor()
model_smart.fit(X, y)
model_smart_str = str(model_smart)
print(model_smart_str)

# Conclusion
p_value = model_sm.pvalues[iv_col]
coef = model_sm.params[iv_col]

explanation = f"The statsmodels OLS regression shows a coefficient for skin_tone of {coef:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05:
    explanation += "This indicates a statistically significant positive relationship between darker skin tone and receiving red cards. "
else:
    explanation += "This indicates no statistically significant relationship between skin tone and receiving red cards. "

# Extract importance from model string
hinge_importance_match = re.search(r'skin_tone:\s*([-\d.]+)', model_hinge_str)
hinge_importance = float(hinge_importance_match.group(1)) if hinge_importance_match else 0.0
explanation += f"The HingeEBMRegressor model gives skin_tone an importance of {hinge_importance:.4f}. "

smart_importance_match = re.search(r'skin_tone:\s*([-\d.]+)', model_smart_str)
smart_importance = float(smart_importance_match.group(1)) if smart_importance_match else 0.0
explanation += f"The SmartAdditiveRegressor model gives skin_tone an importance of {smart_importance:.4f}."


response = 0
if p_value < 0.05 and coef > 0:
    response = 80
elif p_value < 0.1 and coef > 0:
    response = 60
else:
    response = 10
    
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
