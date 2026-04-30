
import pandas as pd
from scipy.stats import ttest_ind
import json

# Load the dataset
df = pd.read_csv('reading.csv')

# Separate the data for dyslexic individuals
dyslexic_df = df[df['dyslexia_bin'] == 1]

# Separate dyslexic readers by whether they used reader_view
reader_view_on = dyslexic_df[dyslexic_df['reader_view'] == 1]['speed']
reader_view_off = dyslexic_df[dyslexic_df['reader_view'] == 0]['speed']

# Perform an independent t-test
t_stat, p_value = ttest_ind(reader_view_on, reader_view_off, nan_policy='omit')

# Interpret the results
# A low p-value (typically < 0.05) suggests a significant difference.
# We are looking for a positive t-statistic, which would indicate that the mean speed for the 'on' group is higher.
# However, 'speed' is likely words per minute, so a higher value is better.
# Let's check the means.
mean_speed_on = reader_view_on.mean()
mean_speed_off = reader_view_off.mean()

explanation = f"The mean reading speed for dyslexic individuals with reader view was {mean_speed_on:.2f} and without it was {mean_speed_off:.2f}. "
explanation += f"The t-statistic is {t_stat:.2f} and the p-value is {p_value:.3f}. "

if p_value < 0.05 and mean_speed_on > mean_speed_off:
    response = 90  # Strong "Yes"
    explanation += "There is a statistically significant improvement in reading speed for dyslexic individuals when using Reader View."
elif p_value < 0.05 and mean_speed_on <= mean_speed_off:
    response = 10 # Strong "No"
    explanation += "There is a statistically significant difference, but it does not improve reading speed."
else:
    response = 50  # Neutral
    explanation += "There is no statistically significant difference in reading speed for dyslexic individuals when using Reader View."

# Create the conclusion file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. conclusion.txt created.")
