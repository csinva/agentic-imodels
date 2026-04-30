
import pandas as pd
import json
from scipy import stats
import statsmodels.api as sm

def analyze_data():
    # Load the dataset
    data = pd.read_csv('fertility.csv')

    # Convert date columns to datetime objects
    data['DateTesting'] = pd.to_datetime(data['DateTesting'], errors='coerce')
    data['StartDateofLastPeriod'] = pd.to_datetime(data['StartDateofLastPeriod'], errors='coerce')

    # Drop rows with missing date values
    data.dropna(subset=['DateTesting', 'StartDateofLastPeriod'], inplace=True)

    # Calculate days from last period
    data['DaysFromLastPeriod'] = (data['DateTesting'] - data['StartDateofLastPeriod']).dt.days

    # Define high-fertility window (days 6-14 of the cycle)
    data['HighFertility'] = data['DaysFromLastPeriod'].apply(lambda x: 1 if 6 <= x <= 14 else 0)

    # Create a composite religiosity score
    data['Religiosity'] = data[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Separate data into high-fertility and low-fertility groups
    high_fertility_group = data[data['HighFertility'] == 1]['Religiosity']
    low_fertility_group = data[data['HighFertility'] == 0]['Religiosity']

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(high_fertility_group, low_fertility_group, nan_policy='omit')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        explanation = f"There is a statistically significant difference in religiosity between high-fertility and low-fertility groups (p-value: {p_value:.3f}). This suggests that hormonal fluctuations associated with fertility may have an effect on women's religiosity."
        response = 80  # Strong "Yes"
    else:
        explanation = f"There is no statistically significant difference in religiosity between high-fertility and low-fertility groups (p-value: {p_value:.3f}). This suggests that hormonal fluctuations associated with fertility may not have a significant effect on women's religiosity."
        response = 20  # Strong "No"

    # Save the conclusion to a JSON file
    conclusion = {"response": response, "explanation": explanation}
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_data()
