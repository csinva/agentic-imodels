
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('panda_nuts.csv')

# Preprocessing
df['sex'] = df['sex'].astype('category').cat.codes
df['help'] = df['help'].apply(lambda x: 1 if x.lower() == 'y' else 0)
df['hammer'] = df['hammer'].astype('category').cat.codes

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Handle potential division by zero
df.loc[df.seconds == 0, 'efficiency'] = 0


# Features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression (Scikit-learn)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)

# Model 2: Explainable Boosting Regressor (InterpretML)
ebm = ExplainableBoostingRegressor(random_state=42)
ebm.fit(X_train, y_train)
ebm_preds = ebm.predict(X_test)
ebm_mse = mean_squared_error(y_test, ebm_preds)

# Model 3: OLS for statistical significance
X_train_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_ols).fit()
p_values = ols_model.pvalues

# Interpretation
explanation = f"Linear Regression MSE: {lr_mse:.4f}, EBM MSE: {ebm_mse:.4f}. "
explanation += "OLS p-values for age, sex, and help are {:.4f}, {:.4f}, and {:.4f} respectively. ".format(p_values['age'], p_values['sex'], p_values['help'])

# Based on the p-values, we can determine the significance of each feature.
# A common threshold for significance is p < 0.05.
significant_features = []
if p_values['age'] < 0.05:
    significant_features.append('age')
if p_values['sex'] < 0.05:
    significant_features.append('sex')
if p_values['help'] < 0.05:
    significant_features.append('help')

if len(significant_features) > 0:
    response = 80  # Strong "Yes"
    explanation += f"The features {', '.join(significant_features)} are statistically significant predictors of nut-cracking efficiency."
else:
    response = 20  # Strong "No"
    explanation += "None of the features (age, sex, help) are statistically significant predictors of nut-cracking efficiency."


# Create conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
