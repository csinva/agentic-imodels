
import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from imodels import RuleFitRegressor

def main():
    # Load data
    df = pd.read_csv('fertility.csv')
    
    # Convert date columns to datetime
    df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
    df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')

    # Calculate days since last period
    df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

    # Estimate fertility
    # High fertility is typically days 10-18 of the cycle
    df['Fertile'] = ((df['DaysSinceLastPeriod'] >= 10) & (df['DaysSinceLastPeriod'] <= 18)).astype(int)

    # Create composite religiosity score
    df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Analyze the relationship between fertility and religiosity
    fertile_religiosity = df[df['Fertile'] == 1]['Religiosity']
    non_fertile_religiosity = df[df['Fertile'] == 0]['Religiosity']

    # T-test
    t_stat, p_value = stats.ttest_ind(fertile_religiosity, non_fertile_religiosity, nan_policy='omit')

    # Regression model
    X = df[['Fertile', 'Relationship', 'Sure1', 'Sure2', 'ReportedCycleLength']]
    y = df['Religiosity']
    X = sm.add_constant(X)
    
    # Drop rows with NaN values for the regression
    X = X.dropna()
    y = y.loc[X.index]

    model = sm.OLS(y, X).fit()
    
    # Rule-based model
    X_rule = df[['Fertile', 'Relationship', 'Sure1', 'Sure2', 'ReportedCycleLength']]
    y_rule = df['Religiosity']
    
    X_rule = X_rule.dropna()
    y_rule = y_rule.loc[X_rule.index]
    
    rulefit = RuleFitRegressor()
    # rulefit.fit(X_rule, y_rule)
    # rules = rulefit.get_rules()

    # Interpretation
    explanation = f"T-test results: t-statistic={t_stat:.3f}, p-value={p_value:.3f}. "
    explanation += f"OLS regression results: fertility coefficient={model.params['Fertile']:.3f}, p-value={model.pvalues['Fertile']:.3f}. "
    
    if p_value < 0.05 and model.pvalues['Fertile'] < 0.05:
        response = 80
        explanation += "There is a statistically significant relationship between fertility and religiosity. The models suggest that higher fertility is associated with a change in religiosity."
    elif p_value < 0.1 or model.pvalues['Fertile'] < 0.1:
        response = 60
        explanation += "There is a marginally significant relationship. The evidence is not strong, but suggests a possible link."
    else:
        response = 20
        explanation += "There is no statistically significant relationship between fertility and religiosity based on the t-test and OLS regression."

    # Save conclusion
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    main()
