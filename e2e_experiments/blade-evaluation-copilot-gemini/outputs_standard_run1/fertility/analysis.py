
import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from imodels import RuleFitRegressor

def analyze_data():
    # Load data
    data = pd.read_csv('fertility.csv')

    # Preprocessing
    # Create a composite religiosity score
    data['religiosity'] = data[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Convert date columns to datetime
    for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Calculate days from last period
    data['days_from_last_period'] = (data['DateTesting'] - data['StartDateofLastPeriod']).dt.days

    # Calculate cycle day
    data['cycle_day'] = data['days_from_last_period'] % data['ReportedCycleLength']

    # Define fertile window (e.g., days 10-17 for a 28-day cycle)
    # This is a simplification; a more accurate approach would use ovulation prediction
    data['fertile'] = ((data['cycle_day'] >= 10) & (data['cycle_day'] <= 17)).astype(int)

    # Drop rows with missing values
    data.dropna(subset=['fertile', 'religiosity'], inplace=True)

    # Statistical Analysis
    # T-test to compare religiosity between fertile and non-fertile groups
    fertile_religiosity = data[data['fertile'] == 1]['religiosity']
    non_fertile_religiosity = data[data['fertile'] == 0]['religiosity']
    
    if len(fertile_religiosity) > 1 and len(non_fertile_religiosity) > 1:
        ttest_result = stats.ttest_ind(fertile_religiosity, non_fertile_religiosity)
        p_value = ttest_result.pvalue
    else:
        p_value = 1.0

    # Modeling
    X = data[['fertile']]
    y = data['religiosity']

    # Use statsmodels for regression to get p-value
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    p_value_regression = model.pvalues['fertile'] if 'fertile' in model.pvalues else 1.0

    # Rule-based model for interpretability
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        rulefit = RuleFitRegressor()
        rulefit.fit(X_train, y_train)
        rules = rulefit.get_rules()
        # Filter for rules related to the 'fertile' feature
        fertile_rules = rules[rules.rule.str.contains('fertile')]
    except Exception as e:
        fertile_rules = pd.DataFrame() # empty dataframe if model fails

    # Conclusion
    # Base the response on the p-value from the t-test and regression
    # A lower p-value indicates a stronger relationship
    if p_value < 0.05 and p_value_regression < 0.05:
        response = 80  # Strong "Yes"
        explanation = f"There is a statistically significant relationship between fertility and religiosity (t-test p-value: {p_value:.3f}, regression p-value: {p_value_regression:.3f}). "
        if not fertile_rules.empty:
            explanation += f"The RuleFit model found the following rules related to fertility: {fertile_rules['rule'].tolist()}"
        else:
            explanation += "The RuleFit model did not yield specific rules for fertility's impact."
    else:
        response = 20  # Strong "No"
        explanation = f"There is no statistically significant relationship between fertility and religiosity (t-test p-value: {p_value:.3f}, regression p-value: {p_value_regression:.3f})."

    # Write conclusion to file
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze_data()
