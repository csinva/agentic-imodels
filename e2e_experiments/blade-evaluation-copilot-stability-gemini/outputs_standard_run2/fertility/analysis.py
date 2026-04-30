
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import json
from sklearn.model_selection import train_test_split
from imodels import RuleFitRegressor
from sklearn.preprocessing import OneHotEncoder

def run_analysis():
    # Load the dataset
    data = pd.read_csv('fertility.csv')

    # Create a composite religiosity score
    data['religiosity'] = data[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Convert date columns to datetime objects
    data['DateTesting'] = pd.to_datetime(data['DateTesting'], errors='coerce')
    data['StartDateofLastPeriod'] = pd.to_datetime(data['StartDateofLastPeriod'], errors='coerce')

    # Drop rows with missing date values
    data.dropna(subset=['DateTesting', 'StartDateofLastPeriod'], inplace=True)

    # Calculate the day of the menstrual cycle
    data['cycle_day'] = (data['DateTesting'] - data['StartDateofLastPeriod']).dt.days

    # Define the high-fertility window (days 6 to 14)
    # This is a simplification. A more accurate approach would use reverse-ovulation calculation.
    data['high_fertility'] = data['cycle_day'].apply(lambda x: 1 if 6 <= x <= 14 else 0)

    # Separate the two groups
    high_fertility_group = data[data['high_fertility'] == 1]['religiosity']
    low_fertility_group = data[data['high_fertility'] == 0]['religiosity']

    # Perform an independent t-test
    t_stat, p_value = ttest_ind(high_fertility_group, low_fertility_group, nan_policy='omit')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        explanation = f"There is a statistically significant difference in religiosity between high and low fertility groups (p-value: {p_value:.3f})."
        response = 80 # Strong "Yes"
    else:
        explanation = f"There is no statistically significant difference in religiosity between high and low fertility groups (p-value: {p_value:.3f})."
        response = 20 # Strong "No"

    # Build an interpretable model
    features = ['Relationship', 'ReportedCycleLength', 'high_fertility']
    X = data[features]
    y = data['religiosity']

    # Handle missing values
    X.loc[:, 'ReportedCycleLength'] = X['ReportedCycleLength'].fillna(X['ReportedCycleLength'].mean())

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['Relationship'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a RuleFit model
    model = RuleFitRegressor()
    model.fit(X_train, y_train)

    # Get the rules
    rules = model._get_rules()
    explanation += " " + str(rules.head())


    # Write the conclusion to a file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    run_analysis()
