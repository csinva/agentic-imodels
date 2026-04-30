
import pandas as pd
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('reading.csv')

# Separate data for individuals with and without dyslexia
dyslexic = df[df['dyslexia_bin'] == 1]
non_dyslexic = df[df['dyslexia_bin'] == 0]

# Separate data based on 'Reader View' for dyslexic group
dyslexic_reader_view = dyslexic[dyslexic['reader_view'] == 1]
dyslexic_no_reader_view = dyslexic[dyslexic['reader_view'] == 0]

# Perform a t-test to compare reading speeds
# The research question is about the effect of 'Reader View' on reading speed for individuals with dyslexia
t_stat, p_value = stats.ttest_ind(
    dyslexic_reader_view['speed'],
    dyslexic_no_reader_view['speed'],
    equal_var=False  # Assume unequal variances
)

# Determine the response based on the p-value
# A small p-value (typically < 0.05) indicates a significant difference
if p_value < 0.05:
    # If there's a significant difference, we need to check the direction
    if dyslexic_reader_view['speed'].mean() > dyslexic_no_reader_view['speed'].mean():
        # Reader view improves reading speed
        response = 90
        explanation = "The analysis shows a statistically significant improvement in reading speed for individuals with dyslexia when using 'Reader View'. The mean reading speed was higher with 'Reader View' enabled."
    else:
        # Reader view worsens reading speed
        response = 10
        explanation = "The analysis shows a statistically significant decrease in reading speed for individuals with dyslexia when using 'Reader View'. The mean reading speed was lower with 'Reader View' enabled."
else:
    # No significant difference
    response = 50
    explanation = "The analysis shows no statistically significant difference in reading speed for individuals with dyslexia when using 'Reader View'. The p-value was greater than 0.05."

# Create the conclusion dictionary
conclusion = {
    "response": response,
    "explanation": explanation
}

# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
