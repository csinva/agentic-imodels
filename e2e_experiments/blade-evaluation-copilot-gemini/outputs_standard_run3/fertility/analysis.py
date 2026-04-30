
import json
import pandas as pd
import numpy as np
from scipy import stats

def analyze():
    # Load data
    data = pd.read_csv('fertility.csv')

    # Preprocessing
    # Convert date columns to datetime
    for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Create religiosity score
    data['religiosity'] = data[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Estimate cycle day and fertility
    data['cycle_day'] = (data['DateTesting'] - data['StartDateofLastPeriod']).dt.days
    
    # Exclude rows with invalid cycle day
    data = data[data['cycle_day'] >= 0]
    
    # Normalize cycle day by cycle length
    data['normalized_cycle_day'] = data['cycle_day'] / data['ReportedCycleLength']
    
    # Define fertile window (e.g., days 10-17 for a 28-day cycle)
    # This corresponds to a normalized cycle day of roughly 0.35 to 0.6
    data['fertile'] = ((data['normalized_cycle_day'] >= 0.35) & (data['normalized_cycle_day'] <= 0.6)).astype(int)

    # Analysis
    # Compare religiosity between fertile and non-fertile groups
    fertile_group = data[data['fertile'] == 1]['religiosity']
    non_fertile_group = data[data['fertile'] == 0]['religiosity']

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(fertile_group, non_fertile_group, nan_policy='omit')

    # Interpretation
    explanation = f"A t-test was performed to compare religiosity between fertile and non-fertile groups. The p-value is {p_value:.3f}. "
    if p_value < 0.05:
        explanation += "There is a statistically significant difference in religiosity between the two groups. "
        if t_stat > 0:
            explanation += "The fertile group has higher religiosity."
            response = 80
        else:
            explanation += "The fertile group has lower religiosity."
            response = 20
    else:
        explanation += "There is no statistically significant difference in religiosity between the two groups."
        response = 50

    # Save conclusion
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze()
