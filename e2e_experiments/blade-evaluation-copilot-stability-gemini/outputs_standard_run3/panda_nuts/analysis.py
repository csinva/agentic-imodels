
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitRegressor
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Preprocess the data
df['sex'] = df['sex'].astype('category').cat.codes
df['help'] = df['help'].apply(lambda x: 1 if x == 'y' else 0)
df['hammer'] = df['hammer'].astype('category').cat.codes

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Handle potential division by zero
df.loc[df.seconds == 0, 'efficiency'] = 0

# Define features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = df[features]
y = df[target]

# Split data for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Building and Interpretation ---

# 1. Multiple Linear Regression (statsmodels for p-values)
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
p_values = model_sm.pvalues

# 2. RuleFit Regressor (imodels for readable rules)
model_rf = RuleFitRegressor()
model_rf.fit(X_train_scaled, y_train)
rules = model_rf._get_rules()

# --- Statistical Analysis ---
# Check p-values from the statsmodels regression
age_p_value = p_values.get('age', 1)
sex_p_value = p_values.get('sex', 1)
help_p_value = p_values.get('help', 1)

# --- Conclusion ---
# Base the response on the statistical significance of the features.
# A lower p-value indicates a stronger relationship.
# We can average the significance, but let's simplify and check if any are significant.
# A common threshold for significance is p < 0.05.

significant_features = []
if age_p_value < 0.05:
    significant_features.append("age")
if sex_p_value < 0.05:
    significant_features.append("sex")
if help_p_value < 0.05:
    significant_features.append("help")

if len(significant_features) > 0:
    response = 80 # Strong "Yes"
    explanation = f"The analysis indicates a statistically significant relationship between nut-cracking efficiency and the following factors: {', '.join(significant_features)}. The p-values for these variables were less than 0.05, suggesting they are important predictors of efficiency."
else:
    response = 20 # Strong "No"
    explanation = "The analysis did not find a statistically significant relationship between age, sex, or receiving help and nut-cracking efficiency. The p-values for all tested variables were greater than 0.05."


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
print(model_sm.summary())
