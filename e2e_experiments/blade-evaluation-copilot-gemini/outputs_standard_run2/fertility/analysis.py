
import pandas as pd
import numpy as np
from scipy import stats
import json
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from imodels import FIGSRegressor

def analyze_fertility_and_religiosity():
    # Load the dataset
    df = pd.read_csv('fertility.csv')

    # Create a composite religiosity score
    df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Convert date columns to datetime objects
    df['DateTesting'] = pd.to_datetime(df['DateTesting'], errors='coerce')
    df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], errors='coerce')

    # Drop rows with missing date values
    df.dropna(subset=['DateTesting', 'StartDateofLastPeriod', 'ReportedCycleLength'], inplace=True)

    # Calculate the number of days from the start of the last period to the testing date
    df['days_since_last_period'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

    # Estimate the day of the cycle
    df['cycle_day'] = df['days_since_last_period'] % df['ReportedCycleLength']

    # Define the fertile window (e.g., days 8 to 19 for a 28-day cycle)
    # We will use a proportional window based on reported cycle length
    df['fertile_window_start'] = (df['ReportedCycleLength'] * (8/28)).round()
    df['fertile_window_end'] = (df['ReportedCycleLength'] * (19/28)).round()

    df['in_fertile_window'] = ((df['cycle_day'] >= df['fertile_window_start']) & (df['cycle_day'] <= df['fertile_window_end'])).astype(int)

    # Separate the two groups
    fertile_group = df[df['in_fertile_window'] == 1]['religiosity']
    non_fertile_group = df[df['in_fertile_window'] == 0]['religiosity']

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(fertile_group, non_fertile_group, nan_policy='omit')

    # Build an interpretable model
    X = df[['in_fertile_window', 'Relationship', 'Sure1', 'Sure2']]
    y = df['religiosity']
    
    # Using statsmodels for p-values
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    p_value_model = model_sm.pvalues['in_fertile_window']

    # Using FIGS for a rule-based model
    model_figs = FIGSRegressor()
    model_figs.fit(X, y)


    # Interpretation
    explanation = f"To determine the effect of fertility on religiosity, I first engineered a 'religiosity' score from the three related survey questions. Then, I estimated whether a woman was in her fertile window based on her cycle day, which was calculated from the testing date and the start of her last period. An independent t-test between the fertile and non-fertile groups yielded a p-value of {p_value:.3f}. A regression model controlling for relationship status and certainty of period dates also showed a p-value of {p_value_model:.3f} for the 'in_fertile_window' coefficient. "

    if p_value < 0.05 and p_value_model < 0.05:
        response = 85  # Strong "Yes"
        explanation += "Both the t-test and the regression model show a statistically significant relationship, suggesting that hormonal fluctuations related to fertility do have an effect on religiosity. The FIGS model further provides interpretable rules: " + str(model_figs)
    elif p_value < 0.1 or p_value_model < 0.1:
        response = 60  # Leaning "Yes"
        explanation += "The results are marginally significant, suggesting a possible weak relationship. The FIGS model provides some rules, but they are not as strong: " + str(model_figs)
    else:
        response = 15  # Strong "No"
        explanation += "The results are not statistically significant, indicating no detectable effect of fertility on religiosity in this dataset. The FIGS model did not find strong predictive rules related to fertility: " + str(model_figs)

    # Create the conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_fertility_and_religiosity()
