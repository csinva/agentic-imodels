
import json
import pandas as pd
from scipy.stats import ttest_ind

def analyze_data():
    # Load the dataset
    try:
        data = pd.read_csv('reading.csv')
    except FileNotFoundError:
        conclusion = {
            "response": 0,
            "explanation": "Error: reading.csv not found. Cannot perform analysis."
        }
        with open('conclusion.txt', 'w') as f:
            json.dump(conclusion, f)
        return

    # Filter for individuals with dyslexia
    dyslexic_data = data[data['dyslexia_bin'] == 1].copy()

    # Separate into two groups: with and without reader view
    reader_view_on = dyslexic_data[dyslexic_data['reader_view'] == 1]
    reader_view_off = dyslexic_data[dyslexic_data['reader_view'] == 0]

    # Ensure there is data in both groups to compare
    if reader_view_on.empty or reader_view_off.empty:
        conclusion = {
            "response": 0,
            "explanation": "Insufficient data for dyslexic individuals with and without reader view to perform a comparison."
        }
        with open('conclusion.txt', 'w') as f:
            json.dump(conclusion, f)
        return

    # Perform an independent t-test on the 'speed' column
    # The 'speed' column is likely words per minute or a similar metric.
    # A higher speed is better.
    ttest_result = ttest_ind(reader_view_on['speed'], reader_view_off['speed'], nan_policy='omit')
    p_value = ttest_result.pvalue
    mean_speed_on = reader_view_on['speed'].mean()
    mean_speed_off = reader_view_off['speed'].mean()

    # Interpret the results
    significant = p_value < 0.05
    improves_speed = mean_speed_on > mean_speed_off

    if significant and improves_speed:
        response = 95
        explanation = (
            f"Yes, 'Reader View' significantly improves reading speed for individuals with dyslexia. "
            f"The average reading speed with Reader View was {mean_speed_on:.2f}, compared to {mean_speed_off:.2f} without it. "
            f"This difference is statistically significant (p-value: {p_value:.4f})."
        )
    elif significant and not improves_speed:
        response = 5
        explanation = (
            f"No, 'Reader View' significantly worsens reading speed for individuals with dyslexia. "
            f"The average reading speed with Reader View was {mean_speed_on:.2f}, compared to {mean_speed_off:.2f} without it. "
            f"This difference is statistically significant (p-value: {p_value:.4f})."
        )
    else: # Not significant
        response = 10
        explanation = (
            f"No, there is no statistically significant evidence that 'Reader View' improves reading speed for individuals with dyslexia. "
            f"Although the average reading speed with Reader View was {mean_speed_on:.2f} compared to {mean_speed_off:.2f} without it, "
            f"this difference is not statistically significant (p-value: {p_value:.4f})."
        )

    # Create the conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f, indent=4)

if __name__ == '__main__':
    analyze_data()
