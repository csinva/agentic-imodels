
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from imodels import RuleFitRegressor

# Load data
df = pd.read_csv('fish.csv')

# Create the target variable: fish caught per hour
# Add a small epsilon to hours to avoid division by zero
df['fish_per_hour'] = df['fish_caught'] / (df['hours'] + 1e-6)

# Define features (X) and target (y)
features = ['livebait', 'camper', 'persons', 'child']
X = df[features]
y = df['fish_per_hour']

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_})

# Fit a RuleFit model for more interpretable rules
rulefit = RuleFitRegressor()
rulefit.fit(X, y)
rules = rulefit._get_rules()

# The research question is about the rate of fish caught per hour.
# A positive coefficient for a feature means it increases the rate.
# We can check the coefficients of the linear model.
# The RuleFit model will give us more specific conditions.

# From the linear model, let's see which features have a positive impact.
positive_impact_features = coef_df[coef_df['coefficient'] > 0]['feature'].tolist()

# Let's formulate an explanation based on the models.
explanation = "Based on the analysis, several factors influence the number of fish caught per hour. "
explanation += "Using a linear regression model, the following factors were found to have a positive correlation with the number of fish caught per hour: "
explanation += ", ".join(positive_impact_features) + ". "
explanation += "This suggests that visitors with these characteristics tend to catch fish at a higher rate. "
explanation += "The RuleFit model provides more specific insights. For example, some of the top rules are: "
explanation += ", ".join(rules.head(3)['rule'].tolist()) + ". "
explanation += "These rules highlight specific combinations of factors that lead to a higher catch rate."


# The question is "How many fish on average do visitors takes per hour, when fishing?".
# This is asking for an estimation of the rate. The models provide this.
# The question can be interpreted as "Can we estimate the rate of fish caught per hour?".
# Since we can build models that explain the variance in the catch rate, the answer is yes.
# The R-squared of the model would be a good indicator of how well we can estimate it.
r_squared = model.score(X, y)

# A higher R-squared means a better estimation.
# Let's set a threshold for the response.
# If R-squared > 0.1, we can say we have a reasonable estimation.
response = int(r_squared * 100)


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
