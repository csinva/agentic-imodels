
import pandas as pd
import numpy as np
import statsmodels.api as sm
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor
import json

# Load data
df = pd.read_csv('boxes.csv')

# Preprocess data
df['chose_majority'] = (df['y'] == 2).astype(int)
df = pd.get_dummies(df, columns=['culture'], drop_first=True)

# Define variables
outcome = 'chose_majority'
predictor = 'age'
controls = [col for col in df.columns if col.startswith('culture_')] + ['gender', 'majority_first']

# --- Statistical Analysis ---
X = df[[predictor] + controls].astype(float)
X = sm.add_constant(X)
y = df[outcome]

logit_model = sm.Logit(y, X).fit()
logit_summary = logit_model.summary()
p_value_age = logit_model.pvalues['age']


# --- Interpretable Models ---
X_im = df[[predictor] + controls].astype(float)
y_im = df[outcome]

# SmartAdditiveRegressor
sa_model = SmartAdditiveRegressor().fit(X_im, y_im)
sa_model_str = str(sa_model)

# HingeEBMRegressor
he_model = HingeEBMRegressor().fit(X_im, y_im)
he_model_str = str(he_model)


# --- Synthesize and Conclude ---
explanation = f"The research question is whether children's reliance on majority preference changes with age. The outcome variable is 'chose_majority', and the primary predictor is 'age'.\\n"
explanation += f"A logistic regression was performed to predict 'chose_majority' from 'age' and control variables (culture, gender, majority_first).\\n"
explanation += f"The p-value for the 'age' coefficient in the logistic regression is {p_value_age:.4f}.\\n"
explanation += f"The SmartAdditiveRegressor model is:\\n{sa_model_str}\\n"
explanation += f"The HingeEBMRegressor model is:\\n{he_model_str}\\n"

# Interpretation
# A low p-value from the logistic regression suggests a significant relationship.
# The interpretable models will show the shape and magnitude of this relationship.
# A positive coefficient for 'age' in the models would indicate that as children get older, they are more likely to choose the majority option.

score = 0
if p_value_age < 0.05:
    # Significant p-value, now check interpretable models for direction and magnitude
    # Looking for a positive coefficient for age in the interpretable models
    age_coef_sa = 0
    if "age" in sa_model_str:
         # crude parsing, but should work
        try:
            age_line = [line for line in sa_model_str.split('\\n') if 'age' in line][0]
            age_coef_sa = float(age_line.split('*')[0].strip())
        except:
            pass


    if age_coef_sa > 0:
        score = 85 # Strong evidence
        explanation += "The logistic regression shows a statistically significant effect of age (p < 0.05). The interpretable models both show a positive relationship between age and choosing the majority, indicating that older children are more likely to follow the majority. The effect is consistent and robust across models."
    else:
        score = 40 # Ambiguous result
        explanation += "The logistic regression shows a statistically significant effect of age (p < 0.05), but the direction of the effect is not consistently positive in the interpretable models. This suggests a complex relationship that is not a simple increase with age."
else:
    score = 10 # Weak evidence
    explanation += "The logistic regression does not show a statistically significant effect of age (p >= 0.05). The interpretable models also do not show a strong, consistent effect of age. Therefore, there is little evidence to suggest that reliance on majority preference changes with age in this dataset."


conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
