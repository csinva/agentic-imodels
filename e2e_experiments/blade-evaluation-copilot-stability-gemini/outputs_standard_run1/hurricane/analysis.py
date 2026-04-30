
import json
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from imodels import RuleFitRegressor
from sklearn.ensemble import RandomForestRegressor

def analyze_hurricane_data():
    """
    Analyzes hurricane data to determine if hurricanes with more feminine names
    are perceived as less threatening, leading to more deaths.
    """
    # Load the dataset
    df = pd.read_csv("hurricane.csv")

    # Feature Engineering and Selection
    features = ['masfem', 'category', 'wind', 'min']
    target = 'alldeaths'
    X = df[features]
    y = df[target]

    # Split data for modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Statsmodels OLS for statistical significance
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()
    p_value_masfem = ols_model.pvalues['masfem']

    # Model 2: RuleFitRegressor for interpretable rules
    rulefit_model = RuleFitRegressor()
    rulefit_model.fit(X_train, y_train)
    rules = rulefit_model._get_rules()
    
    # Model 3: Random Forest for feature importance
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    feature_importances = pd.Series(rf_model.feature_importances_, index=features)

    # Interpretation and Conclusion
    explanation = f"OLS p-value for 'masfem': {p_value_masfem:.4f}. "
    
    if p_value_masfem < 0.05:
        response = 80  # Strong "Yes"
        explanation += "The low p-value suggests a statistically significant relationship between the femininity of a hurricane's name and the number of deaths. "
    else:
        response = 20  # Strong "No"
        explanation += "The p-value is not statistically significant, indicating no clear relationship between the name's femininity and the death toll. "

    explanation += f"Random Forest feature importances: {feature_importances.to_dict()}. "
    explanation += "RuleFit analysis provides further insights into interactions."

    # Create conclusion.txt
    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)

if __name__ == "__main__":
    analyze_hurricane_data()
