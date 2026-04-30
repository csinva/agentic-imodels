
import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def estimate_cycle_day(df):
    df['DateTesting'] = pd.to_datetime(df['DateTesting'], errors='coerce')
    df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], errors='coerce')
    
    # Calculate days since last period
    df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
    
    # Filter out invalid data
    df = df[df['DaysSinceLastPeriod'] >= 0]
    
    # Estimate fertile window (days 7-14 of a 28-day cycle is a common approximation)
    # High fertility is roughly in the week leading up to ovulation.
    # Ovulation is ~14 days before the *next* period.
    # With a 28-day cycle, that's day 14. Fertile window is ~days 8-15.
    # We will use a simplified assumption that the fertile window is between day 7 and day 19
    df['FertileWindow'] = ((df['DaysSinceLastPeriod'] >= 7) & (df['DaysSinceLastPeriod'] <= 19)).astype(int)
    return df

def analyze():
    # Load data
    df = pd.read_csv('fertility.csv')

    # Create composite religiosity score
    df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Estimate cycle day and fertile window
    df = estimate_cycle_day(df)
    
    # Drop rows with missing values that are critical for the analysis
    df.dropna(subset=['Religiosity', 'FertileWindow'], inplace=True)

    # Perform a t-test to compare religiosity between fertile and non-fertile windows
    fertile_group = df[df['FertileWindow'] == 1]['Religiosity']
    non_fertile_group = df[df['FertileWindow'] == 0]['Religiosity']
    
    ttest_result = stats.ttest_ind(fertile_group, non_fertile_group, nan_policy='omit')

    # Build a regression model to control for other factors
    X = df[['FertileWindow', 'Relationship', 'ReportedCycleLength']]
    y = df['Religiosity']
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X, missing='drop').fit()
    
    p_value_fertile_window = model.pvalues['FertileWindow']

    # Based on the p-value, decide on the response
    # If p < 0.05, there is a significant effect.
    if p_value_fertile_window < 0.05:
        response = 80 # Strong "Yes"
        explanation = f"There is a statistically significant relationship between being in the fertile window and religiosity (p-value = {p_value_fertile_window:.4f}). The regression model shows that fertility has a significant effect on religiosity, even after controlling for relationship status and cycle length."
    else:
        response = 20 # Strong "No"
        explanation = f"There is no statistically significant relationship between being in the fertile window and religiosity (p-value = {p_value_fertile_window:.4f}). The regression model shows that fertility does not have a significant effect on religiosity when controlling for other factors."

    # Create conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze()
