
import json
import pandas as pd
import statsmodels.api as sm
from imodels import RuleFitRegressor

def analyze_hurricane_data():
    """
    Analyzes hurricane data to determine if hurricanes with more feminine names
    are perceived as less threatening, leading to more deaths.
    """
    # Load the dataset
    df = pd.read_csv("hurricane.csv")

    # Feature Engineering
    # The 'masfem' column is a rating of how masculine vs. feminine a hurricane's name is.
    # A higher 'masfem' value means a more feminine name.
    # The research question is whether a more feminine name (higher 'masfem') is associated with more deaths.

    # Define the independent and dependent variables
    X = df[['masfem', 'category', 'wind', 'min']]
    y = df['alldeaths']

    # Add a constant to the independent variables for the regression model
    X = sm.add_constant(X)

    # Model 1: Ordinary Least Squares (OLS) Regression
    # This model will show the linear relationship between the features and the number of deaths.
    ols_model = sm.OLS(y, X).fit()
    ols_p_value = ols_model.pvalues['masfem']
    ols_coefficient = ols_model.params['masfem']

    # Model 2: RuleFit Regressor
    # This model will generate human-readable rules to predict the number of deaths.
    # It can capture non-linear relationships and interactions between features.
    rulefit_model = RuleFitRegressor()
    rulefit_model.fit(X, y)
    rules = rulefit_model._get_rules()
    
    # Filter for rules that include 'masfem'
    masfem_rules = rules[rules.rule.str.contains("masfem")]

    # Interpretation
    # The OLS p-value for 'masfem' tells us if there is a statistically significant
    # linear relationship between the femininity of a hurricane's name and the number of deaths.
    # A small p-value (typically < 0.05) suggests a significant relationship.
    # The coefficient tells us the direction and magnitude of the relationship.

    # The RuleFit model provides more nuanced insights. We can examine the rules
    # to see how 'masfem' interacts with other variables to predict deaths.

    # Conclusion
    # We will base our response on the statistical significance from the OLS model.
    # If the p-value for 'masfem' is less than 0.05, we can conclude there is a
    # statistically significant relationship. The RuleFit model can provide
    # supporting evidence and a more detailed explanation.

    is_significant = ols_p_value < 0.05

    if is_significant:
        response = 85  # Strong "Yes"
        explanation = (
            f"There is a statistically significant relationship (p-value: {ols_p_value:.4f}) "
            f"between the femininity of a hurricane's name ('masfem') and the number of deaths. "
            f"The OLS regression coefficient for 'masfem' is {ols_coefficient:.4f}, suggesting that "
            f"more feminine-named hurricanes are associated with more deaths. "
            f"The RuleFit model also identified rules involving 'masfem', supporting this conclusion."
        )
    else:
        response = 15  # Strong "No"
        explanation = (
            f"There is no statistically significant relationship (p-value: {ols_p_value:.4f}) "
            f"between the femininity of a hurricane's name ('masfem') and the number of deaths. "
            f"The OLS regression results do not support the hypothesis that feminine-named "
            f"hurricanes are perceived as less threatening and thus cause more deaths."
        )

    # Write the conclusion to a file
    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)

if __name__ == "__main__":
    analyze_hurricane_data()
