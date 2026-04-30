
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from imodels import RuleFitRegressor

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Preprocess the data
df['sex'] = df['sex'].astype('category').cat.codes
df['help'] = df['help'].astype('category').cat.codes
df['hammer'] = df['hammer'].astype('category').cat.codes

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Define features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = df[features]
y = df[target]

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Statsmodels OLS for p-values
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
p_values = model_sm.pvalues

# Model 2: RuleFitRegressor for interpretable rules
model_rf = RuleFitRegressor()
model_rf.fit(X_train, y_train)
rules = model_rf._get_rules()

# Interpretation
explanation = "Based on the analysis:\\n"
explanation += f"Statsmodels OLS p-values for age, sex, and help are {p_values['age']:.3f}, {p_values['sex']:.3f}, and {p_values['help']:.3f} respectively.\\n"

significant_features = []
if p_values['age'] < 0.05:
    significant_features.append('age')
if p_values['sex'] < 0.05:
    significant_features.append('sex')
if p_values['help'] < 0.05:
    significant_features.append('help')

if significant_features:
    explanation += f"The following features have a statistically significant impact on nut-cracking efficiency: {', '.join(significant_features)}.\\n"
    response = 80
else:
    explanation += "None of the features (age, sex, help) have a statistically significant impact on nut-cracking efficiency.\\n"
    response = 20

# Add RuleFit insights
explanation += "RuleFitRegressor provides additional insights into feature interactions."

# Create conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
