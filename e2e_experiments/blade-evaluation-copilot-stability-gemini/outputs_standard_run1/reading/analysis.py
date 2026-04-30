
import pandas as pd
from scipy.stats import ttest_ind
import json

# Load the dataset
try:
    df = pd.read_csv('reading.csv')
except FileNotFoundError:
    print("Error: reading.csv not found. Make sure the file is in the same directory.")
    exit()

# Filter for individuals with dyslexia
dyslexia_df = df[df['dyslexia_bin'] == 1]

# Separate into two groups: with and without reader view
reader_view_on = dyslexia_df[dyslexia_df['reader_view'] == 1]['speed']
reader_view_off = dyslexia_df[dyslexia_df['reader_view'] == 0]['speed']

# Perform independent t-test
# Drop NaN values to avoid errors
reader_view_on = reader_view_on.dropna()
reader_view_off = reader_view_off.dropna()

if len(reader_view_on) > 1 and len(reader_view_off) > 1:
    t_stat, p_value = ttest_ind(reader_view_on, reader_view_off, equal_var=False) # Welch's t-test

    # Determine the response based on the p-value
    if p_value < 0.05:
        # Significant difference
        if t_stat > 0:
            # Reading speed is higher with reader view
            response = 90
            explanation = "There is a statistically significant improvement in reading speed for individuals with dyslexia when using Reader View (p-value < 0.05)."
        else:
            # Reading speed is lower with reader view
            response = 10
            explanation = "There is a statistically significant decrease in reading speed for individuals with dyslexia when using Reader View (p-value < 0.05)."
    else:
        # No significant difference
        response = 50
        explanation = "There is no statistically significant difference in reading speed for individuals with dyslexia when using Reader View (p-value >= 0.05)."
else:
    response = 0
    explanation = "Not enough data to perform the analysis. There are not enough samples in one or both groups (Reader View on/off) for individuals with dyslexia."


# Create the conclusion.txt file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
