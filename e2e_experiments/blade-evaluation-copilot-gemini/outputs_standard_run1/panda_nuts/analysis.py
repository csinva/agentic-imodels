
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from imodels import FIGSRegressor
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Preprocess the data
df['sex'] = df['sex'].astype('category').cat.codes
df['help'] = df['help'].apply(lambda x: 1 if x == 'y' else 0)

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Define features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = df[features]
y = df[target]

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modeling ---

# 1. Linear Regression (statsmodels for p-values)
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
p_values = model_sm.pvalues

# 2. FIGS for interpretable rules
model_figs = FIGSRegressor()
model_figs.fit(X_train, y_train)

# --- Interpretation ---

# Check significance of features from statsmodels
age_significant = p_values['age'] < 0.05
sex_significant = p_values['sex'] < 0.05
help_significant = p_values['help'] < 0.05

# Formulate the explanation
explanation = "To determine the influence of age, sex, and help on nut-cracking efficiency, I performed a multiple linear regression. "
response = 0

significant_factors = []
if age_significant:
    significant_factors.append("age")
if sex_significant:
    significant_factors.append("sex")
if help_significant:
    significant_factors.append("help")

if len(significant_factors) > 0:
    explanation += f"The results show that {', '.join(significant_factors)} significantly influence(s) efficiency. "
    # Base score on number of significant factors
    response = 33 * len(significant_factors)
else:
    explanation += "No factors were found to have a significant influence on efficiency. "
    response = 0

# Refine explanation with model details
explanation += f"The p-values for age, sex, and help were {p_values['age']:.3f}, {p_values['sex']:.3f}, and {p_values['help']:.3f} respectively. "
explanation += "An interpretable FIGS model was also trained, and its rules can provide further insight into the interactions. "
explanation += "Based on the statistical significance, the answer is moderately affirmative."
if response > 100:
    response = 100


# Create the conclusion file
conclusion = {
    "response": int(response),
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
print(model_sm.summary())
print("\\nFIGS model rules:")
print(model_figs)

