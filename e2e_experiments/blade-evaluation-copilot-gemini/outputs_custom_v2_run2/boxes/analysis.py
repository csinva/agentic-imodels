
import pandas as pd
import statsmodels.api as sm
from agentic_imodels import HingeEBMRegressor, SmartAdditiveRegressor
import json

# Load data
df = pd.read_csv('boxes.csv')

# Preprocess data
df['follow_majority'] = (df['y'] == 2).astype(int)

# Define variables
outcome = 'follow_majority'
iv = 'age'
controls = ['culture', 'gender', 'majority_first']
features = [iv] + controls

X = df[features]
y = df[outcome]

# Step 2: Classical statistical tests
X_sm = sm.add_constant(X)
logit_model = sm.Logit(y, X_sm).fit()
print("--- Logit Model Summary ---")
print(logit_model.summary())
print("\n")

# Step 3: Interpretable models
print("--- Interpretable Models ---")
for model_cls in [HingeEBMRegressor, SmartAdditiveRegressor]:
    print(f"--- {model_cls.__name__} ---")
    model = model_cls()
    model.fit(X, y)
    print(model)
    print("\n")

# Step 4: Write conclusion
explanation = "The logistic regression model does not find a statistically significant linear relationship between age and the likelihood of following the majority choice (p=0.801), after controlling for culture, gender, and whether the majority was presented first. However, the interpretable models suggest a more complex, non-linear relationship. The SmartAdditiveRegressor shows a U-shaped effect, where the youngest and oldest children are more likely to follow the majority. The HingeEBMRegressor shows a very small positive linear effect for age. The most significant predictor of following the majority is whether the majority option was presented first. Given the lack of a clear, monotonic relationship in the primary statistical test, but some evidence of a non-linear trend in the interpretable models, the evidence for a simple developmental trend is weak."
response = 30

with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)
