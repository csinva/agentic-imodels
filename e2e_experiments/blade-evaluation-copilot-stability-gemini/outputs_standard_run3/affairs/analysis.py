
import json
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm

def analysis():
    # Load data
    df = pd.read_csv('affairs.csv')

    # Convert categorical variables to numeric
    df['children'] = df['children'].apply(lambda x: 1 if x == 'yes' else 0)

    # Separate groups
    children_yes = df[df['children'] == 1]['affairs']
    children_no = df[df['children'] == 0]['affairs']

    # Perform t-test
    ttest_result = ttest_ind(children_yes, children_no, equal_var=False)

    # Perform regression analysis
    X = df[['children', 'age', 'yearsmarried', 'religiousness', 'education', 'rating']]
    X = sm.add_constant(X)
    y = df['affairs']
    model = sm.OLS(y, X).fit()
    
    # Get the coefficient for 'children'
    children_coefficient = model.params['children']
    p_value = model.pvalues['children']

    # Interpretation
    # We will use the p-value from the regression as it controls for other factors.
    # A low p-value (e.g., < 0.05) suggests a significant relationship.
    # A negative coefficient for 'children' would suggest that having children is associated with fewer affairs.
    
    if p_value < 0.05 and children_coefficient < 0:
        # Significant negative relationship
        response = 80
        explanation = f"Yes, having children is associated with a decrease in extramarital affairs. The p-value is {p_value:.3f} and the coefficient for children is {children_coefficient:.3f}, suggesting a statistically significant negative relationship when controlling for other factors."
    elif p_value < 0.05 and children_coefficient >= 0:
        # Significant positive relationship
        response = 20
        explanation = f"No, having children is not associated with a decrease in extramarital affairs. In fact, the relationship is positive and statistically significant (p-value: {p_value:.3f}, coefficient: {children_coefficient:.3f})."
    else:
        # Not a significant relationship
        response = 20
        explanation = f"No, there is not a statistically significant relationship between having children and the number of extramarital affairs. The p-value is {p_value:.3f}, which is greater than 0.05."

    # Create conclusion file
    conclusion = {"response": response, "explanation": explanation}
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analysis()
