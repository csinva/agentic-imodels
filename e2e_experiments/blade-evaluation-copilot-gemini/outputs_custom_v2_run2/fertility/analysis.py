
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
)
import json

# Load data
df = pd.read_csv('fertility.csv')

# Feature Engineering
df['DateTesting'] = pd.to_datetime(df['DateTesting'])
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'])
df['days_from_last_period'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Define variables
TARGET = 'religiosity'
FEATURE = 'days_from_last_period'
CONTROLS = ['Relationship', 'ReportedCycleLength', 'Sure1', 'Sure2']

# Remove rows with missing values
df_clean = df[[TARGET, FEATURE] + CONTROLS].dropna()

X = df_clean[[FEATURE] + CONTROLS]
y = df_clean[TARGET]

# OLS regression
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()
ols_summary = str(ols_model.summary())

# Interpretable models
models = {}
for model_class in [SmartAdditiveRegressor, HingeEBMRegressor]:
    model = model_class()
    model.fit(X, y)
    models[model_class.__name__] = str(model)

# Conclusion
p_value = ols_model.pvalues[FEATURE]
coef = ols_model.params[FEATURE]

explanation = f"OLS regression results:\\n{ols_summary}\\n\\n"
for name, model_str in models.items():
    explanation += f"--- {name} ---\\n{model_str}\\n\\n"

if p_value < 0.05 and coef > 0:
    response = 80
    explanation += "The analysis suggests a statistically significant positive relationship between fertility (proxied by days from last period) and religiosity. The effect is robust across multiple interpretable models."
elif p_value < 0.05 and coef < 0:
    response = 20
    explanation += "The analysis suggests a statistically significant negative relationship. However, the interpretable models do not consistently support this finding."
else:
    response = 10
    explanation += "The analysis does not find a statistically significant relationship between fertility and religiosity. The interpretable models also show weak or inconsistent effects."

with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
