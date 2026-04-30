
import pandas as pd
from scipy.stats import ttest_ind
import json

# Load the dataset
df = pd.read_csv('reading.csv')

# The research question is: "Does 'Reader View' --- a modified web page layout --- improve reading speed for individuals with dyslexia?"
# Filter the data for individuals with dyslexia
dyslexia_df = df[df['dyslexia_bin'] == 1]

# Separate the reading speeds for those who used Reader View and those who did not
reader_view_speed = dyslexia_df[dyslexia_df['reader_view'] == 1]['speed']
no_reader_view_speed = dyslexia_df[dyslexia_df['reader_view'] == 0]['speed']

# Perform an independent t-test
ttest_result = ttest_ind(reader_view_speed, no_reader_view_speed, nan_policy='omit')

# Check for statistical significance
alpha = 0.05
if ttest_result.pvalue < alpha:
    # The difference is statistically significant
    if reader_view_speed.mean() > no_reader_view_speed.mean():
        response = 90
        explanation = "Yes, for individuals with dyslexia, 'Reader View' is associated with a statistically significant increase in reading speed. The average reading speed with 'Reader View' was higher than without it."
    else:
        response = 10
        explanation = "No, for individuals with dyslexia, 'Reader View' is associated with a statistically significant decrease in reading speed. The average reading speed with 'Reader View' was lower than without it."
else:
    # The difference is not statistically significant
    response = 50
    explanation = "There is no statistically significant evidence to suggest that 'Reader View' improves reading speed for individuals with dyslexia. The difference in mean reading speed between the two groups was not significant."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
