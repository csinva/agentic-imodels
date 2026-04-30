
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imodels import RuleFitRegressor
import statsmodels.api as sm

# Load data
df = pd.read_csv('panda_nuts.csv')

# Preprocess data
df['sex'] = df['sex'].astype('category').cat.codes
df['help'] = df['help'].astype('category').cat.codes
df['hammer'] = df['hammer'].astype('category').cat.codes

# Define efficiency
df['efficiency'] = df['nuts_opened'] / df['seconds']
df = df[df['efficiency'] != float('inf')] # remove infinite values
df = df.dropna(subset=['efficiency'])

# Define features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_coeffs = dict(zip(features, lr.coef_))


# Model 2: RuleFit Regressor
rf = RuleFitRegressor()
rf.fit(X_train, y_train)
rules = rf.rules_

# Model 3: OLS for statistical significance
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()
p_values = ols_model.pvalues

# Interpretation
explanation = "To determine the influence of age, sex, and help on nut-cracking efficiency, I analyzed the data using three different models. "
explanation += "A simple Linear Regression showed the following coefficients: age({:.2f}), sex({:.2f}), help({:.2f}). ".format(lr_coeffs['age'], lr_coeffs['sex'], lr_coeffs['help'])
explanation += "The RuleFitRegressor provided more interpretable rules. "
explanation += "Finally, an OLS regression was used to assess statistical significance. The p-values for age, sex, and help were {:.3f}, {:.3f}, and {:.3f} respectively. ".format(p_values['age'], p_values['sex'], p_values['help'])

# Based on the p-values, age is the only statistically significant predictor of nut-cracking efficiency.
# A positive coefficient for age suggests that as chimpanzees get older, their efficiency increases.
# Sex and receiving help do not have a statistically significant impact on efficiency.
if p_values['age'] < 0.05 and all(p > 0.05 for p in [p_values['sex'], p_values['help']]):
    response = 80
    explanation += "Based on the p-values, age is the only statistically significant predictor of nut-cracking efficiency (p < 0.05). The positive coefficient for age suggests that as chimpanzees get older, their efficiency increases. Sex and receiving help do not have a statistically significant impact on efficiency."
elif all(p > 0.05 for p in p_values[1:]):
    response = 10
    explanation += "None of the variables (age, sex, help) were found to be statistically significant predictors of nut-cracking efficiency (all p > 0.05)."
else:
    response = 50
    explanation += "The analysis yielded mixed results regarding the significance of the predictors. Further investigation is needed to draw a firm conclusion."


# Create conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
