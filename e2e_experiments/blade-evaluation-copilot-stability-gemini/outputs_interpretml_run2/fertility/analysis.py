
import pandas as pd
import numpy as np
import json
from scipy.stats import ttest_ind

def estimate_fertility(day_of_cycle, cycle_length):
    """
    Crude estimation of fertility based on cycle day.
    This is a simplified model. A more accurate model would use probabilities from medical data.
    High fertility is generally considered to be in the days leading up to and including ovulation.
    Ovulation is roughly cycle_length - 14 days.
    Let's define a high-fertility window.
    """
    ovulation_day = cycle_length - 14
    fertility_window_start = ovulation_day - 5
    fertility_window_end = ovulation_day + 1
    if fertility_window_start <= day_of_cycle <= fertility_window_end:
        return 'high'
    else:
        return 'low'

def analyze():
    df = pd.read_csv('fertility.csv')

    # Combine religiosity scores
    df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Convert date columns to datetime
    df['DateTesting'] = pd.to_datetime(df['DateTesting'], errors='coerce')
    df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], errors='coerce')

    # Drop rows with missing date information
    df.dropna(subset=['DateTesting', 'StartDateofLastPeriod', 'ReportedCycleLength'], inplace=True)

    # Calculate day of cycle
    df['day_of_cycle'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
    
    # Filter out unrealistic cycle days
    df = df[df['day_of_cycle'] >= 0]
    df = df[df['day_of_cycle'] <= df['ReportedCycleLength']]


    # Estimate fertility
    df['fertility_phase'] = df.apply(lambda row: estimate_fertility(row['day_of_cycle'], row['ReportedCycleLength']), axis=1)

    # Separate into high and low fertility groups
    high_fertility_group = df[df['fertility_phase'] == 'high']['religiosity']
    low_fertility_group = df[df['fertility_phase'] == 'low']['religiosity']

    # Perform t-test
    if len(high_fertility_group) > 1 and len(low_fertility_group) > 1:
        ttest_result = ttest_ind(high_fertility_group, low_fertility_group, nan_policy='omit')
        p_value = ttest_result.pvalue
        statistic = ttest_result.statistic
    else:
        p_value = 1.0
        statistic = 0.0


    # Interpret results
    # A low p-value (e.g., < 0.05) would suggest a significant difference.
    # The sign of the statistic tells us the direction.
    
    explanation = f"A t-test was performed to compare the mean religiosity scores between women in high and low fertility phases of their menstrual cycle. The high fertility group had a mean religiosity of {high_fertility_group.mean():.2f}, while the low fertility group had a mean of {low_fertility_group.mean():.2f}. The t-statistic is {statistic:.2f} and the p-value is {p_value:.3f}."

    if p_value < 0.05 and statistic > 0:
        # Higher religiosity in high fertility group
        response = 80
        explanation += " There is a statistically significant increase in religiosity during the high-fertility phase."
    elif p_value < 0.05 and statistic < 0:
        # Lower religiosity in high fertility group
        response = 20
        explanation += " There is a statistically significant decrease in religiosity during the high-fertility phase."
    else:
        # No significant difference
        response = 10
        explanation += " There is no statistically significant difference in religiosity between the high and low fertility phases."


    # Create conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze()
