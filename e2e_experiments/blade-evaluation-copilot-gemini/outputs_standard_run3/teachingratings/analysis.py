
import pandas as pd
import json
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from imodels import RuleFitRegressor

def analyze_data():
    # Load data
    data = pd.read_csv('teachingratings.csv')

    # One-hot encode categorical variables
    categorical_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

    # Define features (X) and target (y)
    # Exclude non-feature columns
    X = data.drop(columns=['rownames', 'eval', 'prof'])
    y = data['eval']

    # Add a constant for the intercept term
    X_const = sm.add_constant(X)

    # Fit OLS model for statistical significance
    ols_model = sm.OLS(y, X_const).fit()
    p_value_beauty = ols_model.pvalues['beauty']
    coef_beauty = ols_model.params['beauty']

    # Fit interpretable models
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_coef_beauty = lr.coef_[X.columns.get_loc('beauty')]

    # RuleFit Regressor for interaction effects and rules
    rf = RuleFitRegressor()
    rf.fit(X, y)
    rules = rf.rules_
    rules_df = pd.DataFrame({'rule': [str(r) for r in rules]})
    beauty_rules = rules_df[rules_df.rule.str.contains('beauty')]

    # Interpretation
    explanation = f"The research question is: What is the impact of beauty on teaching evaluations? To answer this, a multiple linear regression was performed. The coefficient for 'beauty' is {coef_beauty:.4f}, with a p-value of {p_value_beauty:.4f}. "

    if p_value_beauty < 0.05:
        explanation += "This indicates a statistically significant positive relationship between beauty and teaching evaluations. As the beauty score increases, the teaching evaluation tends to increase as well. "
        # Assign a score based on significance and effect size
        response = 85  # Strong "Yes"
    else:
        explanation += "This indicates that there is no statistically significant relationship between beauty and teaching evaluations. "
        response = 15  # Strong "No"

    explanation += f"A simple linear regression model also found a coefficient of {lr_coef_beauty:.4f} for beauty. "
    if not beauty_rules.empty:
        explanation += f"Furthermore, a RuleFit model identified the following rules involving beauty: {beauty_rules['rule'].tolist()}. This suggests that the effect of beauty might be nuanced and interact with other factors."
    else:
        explanation += "A RuleFit model did not find any significant rules involving beauty."


    # Create conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f, indent=4)

if __name__ == '__main__':
    analyze_data()
